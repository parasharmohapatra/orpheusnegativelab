import type { LinearImage, Mat3 } from '../types';
import { IDENTITY } from '../color/matrices';
import type { OutputSpaceId } from '../color/spaces';
import { OUTPUT_SPACES } from '../color/spaces';
import { chromaRadius, decodeFull } from '../decode/index';
import { denoiseChroma } from '../decode/denoise';
import { type FullImage, regionToFloat } from '../decode/fullImage';
import type { TiffTransfer } from '../decode/tiff';
import { makeIccProfile } from '../encode/iccProfile';
import { addBorder, encodeJpeg, encodePng, encodeTiff, quantize } from '../encode/writers';
import { applyAffine, compose, outputSize, outputToSource, withOrientation, type Affine, type Geometry } from '../geometry/transform';
import { Renderer } from '../gpu/renderer';
import type { RenderParams } from '../pipeline/params';
import { renderCpu } from '../pipeline/reference';

export type ExportFormat = 'jpeg' | 'png' | 'tiff';

export interface ExportSettings {
  format: ExportFormat;
  bits: 8 | 16;
  quality: number;
  space: OutputSpaceId;
  /** Long edge in pixels; null = full resolution. */
  longEdge: number | null;
  /** White border as a fraction of the short edge; 0 = none. */
  border: number;
}

export interface ExportJob {
  file: File;
  params: RenderParams;
  geometry: Geometry;
  settings: ExportSettings;
  tiffTransfer?: TiffTransfer;
}

export interface ExportResult {
  bytes: Uint8Array;
  width: number;
  height: number;
  mime: string;
  ext: string;
}

const TILE = 2048;

/**
 * Render a frame at full resolution in tiles (bounded GPU memory, no texture-size
 * limit), quantize, and encode. Runs in the export worker.
 */
export async function exportFrame(job: ExportJob, onProgress?: (f: number) => void): Promise<ExportResult> {
  const { image, meta } = await decodeFull(job.file, { full: true, tiffTransfer: job.tiffTransfer });
  onProgress?.(0.3);
  const geo = withOrientation(job.geometry, meta.orientation);
  let [outW, outH] = outputSize(image.width, image.height, geo);
  if (job.settings.longEdge && Math.max(outW, outH) > job.settings.longEdge) {
    const s = job.settings.longEdge / Math.max(outW, outH);
    outW = Math.max(1, Math.round(outW * s));
    outH = Math.max(1, Math.round(outH * s));
  }
  const toSource = outputToSource(image.width, image.height, geo);
  const rgb = renderTiled(image, job.params, toSource, outW, outH, chromaRadius(meta.kind, true), (f) => onProgress?.(0.3 + f * 0.55));

  let data = rgb;
  let w = outW;
  let h = outH;
  if (job.settings.border > 0) {
    const b = addBorder(rgb, outW, outH, Math.round(Math.min(outW, outH) * job.settings.border));
    data = b.data;
    w = b.w;
    h = b.h;
  }
  const s = job.settings;
  const icc = makeIccProfile(s.space);
  const bits = s.format === 'jpeg' ? 8 : s.bits;
  const samples = quantize(data, bits);
  onProgress?.(0.9);
  let bytes: Uint8Array;
  if (s.format === 'jpeg') bytes = await encodeJpeg(samples as Uint8Array, w, h, s.quality, icc);
  else if (s.format === 'png') bytes = encodePng(samples, w, h, icc, OUTPUT_SPACES[s.space].label);
  else bytes = encodeTiff(samples, w, h, icc);
  onProgress?.(1);
  const mime = s.format === 'jpeg' ? 'image/jpeg' : s.format === 'png' ? 'image/png' : 'image/tiff';
  return { bytes, width: w, height: h, mime, ext: s.format === 'jpeg' ? 'jpg' : s.format === 'png' ? 'png' : 'tif' };
}

let renderer: Renderer | null | undefined;
function getRenderer(): Renderer | null {
  if (renderer === undefined) {
    try {
      const r = new Renderer(new OffscreenCanvas(1, 1));
      renderer = r.canFloatTarget ? r : null;
    } catch {
      renderer = null;
    }
  }
  return renderer;
}

function renderTiled(
  img: FullImage,
  params: RenderParams,
  toSource: Affine,
  outW: number,
  outH: number,
  nrRadius: number,
  onProgress: (f: number) => void,
): Float32Array {
  // Extra source margin so chroma NR near tile edges sees real neighbours.
  const nrMargin = Math.ceil(nrRadius * 2) + 2;
  const out = new Float32Array(outW * outH * 3);
  const r = getRenderer();
  const tiles: { x: number; y: number; w: number; h: number }[] = [];
  for (let y = 0; y < outH; y += TILE) for (let x = 0; x < outW; x += TILE) tiles.push({ x, y, w: Math.min(TILE, outW - x), h: Math.min(TILE, outH - y) });
  if (r) r.setParams(params);
  tiles.forEach((t, k) => {
    // Output uv of this tile → which part of the source it needs (+2 px for bilinear).
    const corners = [
      [t.x / outW, t.y / outH],
      [(t.x + t.w) / outW, t.y / outH],
      [t.x / outW, (t.y + t.h) / outH],
      [(t.x + t.w) / outW, (t.y + t.h) / outH],
    ].map(([u, v]) => applyAffine(toSource, u, v));
    const m = 2 + nrMargin;
    const sx0 = Math.max(0, Math.floor(Math.min(...corners.map((c) => c[0])) * img.width) - m);
    const sy0 = Math.max(0, Math.floor(Math.min(...corners.map((c) => c[1])) * img.height) - m);
    const sx1 = Math.min(img.width, Math.ceil(Math.max(...corners.map((c) => c[0])) * img.width) + m);
    const sy1 = Math.min(img.height, Math.ceil(Math.max(...corners.map((c) => c[1])) * img.height) + m);
    const rw = Math.max(1, sx1 - sx0);
    const rh = Math.max(1, sy1 - sy0);
    const region: LinearImage = { width: rw, height: rh, data: regionToFloat(img, sx0, sy0, rw, rh) };
    denoiseChroma(region.data, rw, rh, nrRadius);
    // Full-source uv → region uv.
    const sub: Affine = [img.width / rw, 0, -sx0 / rw, 0, img.height / rh, -sy0 / rh];
    const view: Affine = [t.w / outW, 0, t.x / outW, 0, t.h / outH, t.y / outH];
    const geo = compose(sub, toSource);
    let tile: Float32Array;
    if (r) {
      r.setSource(region);
      const target = r.createTarget(t.w, t.h, true);
      r.draw({ view, geo, surround: [0, 0, 0], mode: 0, negToOut: IDENTITY as Mat3 }, t.w, t.h, target);
      const rgba = r.readFloat(target);
      r.deleteTarget(target);
      tile = new Float32Array(t.w * t.h * 3);
      for (let i = 0, j = 0; j < rgba.length; i += 3, j += 4) {
        tile[i] = rgba[j];
        tile[i + 1] = rgba[j + 1];
        tile[i + 2] = rgba[j + 2];
      }
    } else {
      tile = renderCpu(region, params, compose(geo, view), t.w, t.h);
    }
    for (let y = 0; y < t.h; y++) out.set(tile.subarray(y * t.w * 3, (y + 1) * t.w * 3), ((t.y + y) * outW + t.x) * 3);
    onProgress((k + 1) / tiles.length);
  });
  return out;
}
