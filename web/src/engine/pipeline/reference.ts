/**
 * CPU reference implementation of the per-pixel conversion.
 *
 * This is the oracle for tests and the source of truth for the GLSL shader in
 * `gpu/shaders/convert.frag.glsl`: change both together, and the GPU parity check
 * will tell you if they drift. It also renders thumbnails and is the export fallback
 * on devices without float render targets.
 */
import type { LinearImage, Vec3 } from '../types';
import { LMS_TO_WORKING, OKLAB_M2, OKLAB_M2_INV, WORKING_TO_LMS } from '../color/oklab';
import { encode, srgbDecode } from '../color/transfer';
import { applyAffine, type Affine } from '../geometry/transform';
import { LUT_SIZE } from './curves';
import { HSL_CENTRES, PAPER_X_MAX, PAPER_X_MIN, type RenderParams } from './params';

const LOG2_018 = Math.log2(0.18);

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

function bend(n: number, m: number): number {
  if (n < 0) return n * (1 + m);
  if (n > 1) return 1 + (n - 1) * (1 - m);
  return n + m * n * (1 - n);
}

/** Linear working value → print value (linear light), through tone sliders and the paper curve. */
export function paper(p: RenderParams, v: number): number {
  let x = Math.log2(Math.max(v, 1e-10)) - LOG2_018;
  x += p.shadows * 1.5 * Math.exp(-((x + 3) ** 2) / 4.5) + p.highlights * 1.5 * Math.exp(-((x - 1.5) ** 2) / 4.5);
  if (p.printMode === 0) return 0.18 * Math.pow(2, x);
  const f = Math.min(1, Math.max(0, (x - PAPER_X_MIN) / (PAPER_X_MAX - PAPER_X_MIN))) * (LUT_SIZE - 1);
  const i = Math.min(LUT_SIZE - 2, Math.floor(f));
  const t = f - i;
  return srgbDecode(p.paperLut[i] * (1 - t) + p.paperLut[i + 1] * t);
}

/** Normalized, balanced density for one camera-RGB pixel (steps 1–5); exposed for the neutral picker. */
export function balancedDensity(p: RenderParams, rgb: Vec3, withOffset = true): Vec3 {
  const D: Vec3 = [0, 0, 0];
  for (let c = 0; c < 3; c++) D[c] = -Math.log10(Math.max(rgb[c], 1e-6)) + p.logBase[c];
  if (p.filmType === 1) {
    const s = p.bwMix[0] + p.bwMix[1] + p.bwMix[2] || 1;
    const d = (p.bwMix[0] * D[0] + p.bwMix[1] * D[1] + p.bwMix[2] * D[2]) / s;
    D[0] = D[1] = D[2] = d;
  }
  const n: Vec3 = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    n[c] = bend((D[c] - p.dLo[c]) * p.invRange[c], p.mid[c]) + (withOffset ? p.offset[c] : 0);
  }
  return n;
}

function hslAdjust(p: RenderParams, hueDeg: number): [number, number, number] {
  const cs = HSL_CENTRES;
  let i = cs.length - 1;
  for (let k = 0; k < cs.length; k++) if (hueDeg >= cs[k]) i = k;
  const j = (i + 1) % cs.length;
  let span = cs[j] - cs[i];
  if (span <= 0) span += 360;
  let d = hueDeg - cs[i];
  if (d < 0) d += 360;
  const t = smoothstep(0, 1, d / span);
  return [
    p.hslHue[i] * (1 - t) + p.hslHue[j] * t,
    p.hslSat[i] * (1 - t) + p.hslSat[j] * t,
    p.hslLum[i] * (1 - t) + p.hslLum[j] * t,
  ];
}

/** Full conversion of one camera-RGB pixel to an encoded output pixel in [0, 1]. */
export function convertPixel(p: RenderParams, rgb: Vec3): Vec3 {
  const n = balancedDensity(p, rgb);
  const E: Vec3 = [0, 0, 0];
  for (let c = 0; c < 3; c++) E[c] = 0.18 * Math.pow(2, p.hiStops + p.sceneStops * (n[c] - 1));
  const m = p.camToWork;
  const W: Vec3 = [
    m[0] * E[0] + m[1] * E[1] + m[2] * E[2],
    m[3] * E[0] + m[4] * E[1] + m[5] * E[2],
    m[6] * E[0] + m[7] * E[1] + m[8] * E[2],
  ];

  let y: Vec3;
  if (p.preserveHue) {
    const mx = Math.max(W[0], W[1], W[2], 1e-10);
    const k = paper(p, mx) / mx;
    y = [W[0] * k, W[1] * k, W[2] * k];
  } else {
    y = [paper(p, W[0]), paper(p, W[1]), paper(p, W[2])];
  }

  if (p.saturation !== 0 || p.vibrance !== 0 || p.hslActive) {
    const a = WORKING_TO_LMS;
    const lms = [
      Math.cbrt(a[0] * y[0] + a[1] * y[1] + a[2] * y[2]),
      Math.cbrt(a[3] * y[0] + a[4] * y[1] + a[5] * y[2]),
      Math.cbrt(a[6] * y[0] + a[7] * y[1] + a[8] * y[2]),
    ];
    const M = OKLAB_M2;
    let L = M[0] * lms[0] + M[1] * lms[1] + M[2] * lms[2];
    const A = M[3] * lms[0] + M[4] * lms[1] + M[5] * lms[2];
    const B = M[6] * lms[0] + M[7] * lms[1] + M[8] * lms[2];
    let C = Math.hypot(A, B);
    let h = (Math.atan2(B, A) * 180) / Math.PI;
    if (h < 0) h += 360;
    if (p.hslActive) {
      const [dh, ds, dl] = hslAdjust(p, h);
      h += dh * 30;
      L += dl * 0.15 * smoothstep(0.01, 0.08, C);
      C *= 1 + ds;
    }
    C *= 1 + p.saturation;
    C *= 1 + p.vibrance * (1 - smoothstep(0, 0.25, C));
    C = Math.max(0, C);
    const hr = (h * Math.PI) / 180;
    const lab = [L, C * Math.cos(hr), C * Math.sin(hr)];
    const Mi = OKLAB_M2_INV;
    const l3 = [
      (Mi[0] * lab[0] + Mi[1] * lab[1] + Mi[2] * lab[2]) ** 3,
      (Mi[3] * lab[0] + Mi[4] * lab[1] + Mi[5] * lab[2]) ** 3,
      (Mi[6] * lab[0] + Mi[7] * lab[1] + Mi[8] * lab[2]) ** 3,
    ];
    const b = LMS_TO_WORKING;
    y = [
      b[0] * l3[0] + b[1] * l3[1] + b[2] * l3[2],
      b[3] * l3[0] + b[4] * l3[1] + b[5] * l3[2],
      b[6] * l3[0] + b[7] * l3[1] + b[8] * l3[2],
    ];
  }

  const o = p.workToOut;
  const out: Vec3 = [
    o[0] * y[0] + o[1] * y[1] + o[2] * y[2],
    o[3] * y[0] + o[4] * y[1] + o[5] * y[2],
    o[6] * y[0] + o[7] * y[1] + o[8] * y[2],
  ];
  for (let c = 0; c < 3; c++) {
    let v = encode(Math.min(1, Math.max(0, out[c])), p.outTransfer);
    if (p.curves) {
      const f = v * (LUT_SIZE - 1);
      const i = Math.min(LUT_SIZE - 2, Math.floor(f));
      const t = f - i;
      v = p.curves[c * LUT_SIZE + i] * (1 - t) + p.curves[c * LUT_SIZE + i + 1] * t;
    }
    out[c] = v;
  }
  return out;
}

/** Bilinear sample at normalized source coords (texel centres at (i + 0.5) / size, clamp to edge). */
export function sampleBilinear(img: LinearImage, u: number, v: number, out: Vec3): Vec3 {
  const { width: w, height: h, data } = img;
  const x = Math.min(w - 1, Math.max(0, u * w - 0.5));
  const y = Math.min(h - 1, Math.max(0, v * h - 0.5));
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(w - 1, x0 + 1);
  const y1 = Math.min(h - 1, y0 + 1);
  const tx = x - x0;
  const ty = y - y0;
  for (let c = 0; c < 3; c++) {
    const a = data[(y0 * w + x0) * 3 + c];
    const b = data[(y0 * w + x1) * 3 + c];
    const d = data[(y1 * w + x0) * 3 + c];
    const e = data[(y1 * w + x1) * 3 + c];
    out[c] = (a * (1 - tx) + b * tx) * (1 - ty) + (d * (1 - tx) + e * tx) * ty;
  }
  return out;
}

/**
 * Render a whole frame on the CPU. `toSource` maps output uv → source uv.
 * Writes encoded values to `out` (Float32, RGB interleaved) and returns it.
 */
export function renderCpu(
  src: LinearImage,
  p: RenderParams,
  toSource: Affine,
  outW: number,
  outH: number,
  out = new Float32Array(outW * outH * 3),
): Float32Array {
  const px: Vec3 = [0, 0, 0];
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const [u, v] = applyAffine(toSource, (x + 0.5) / outW, (y + 0.5) / outH);
      const i = (y * outW + x) * 3;
      if (u < 0 || u > 1 || v < 0 || v > 1) {
        out[i] = out[i + 1] = out[i + 2] = 0;
        continue;
      }
      sampleBilinear(src, u, v, px);
      const o = convertPixel(p, px);
      out[i] = o[0];
      out[i + 1] = o[1];
      out[i + 2] = o[2];
    }
  }
  return out;
}

/** Encoded float RGB → 8-bit RGBA for canvases. */
export function toRGBA8(rgb: Float32Array, w: number, h: number): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0, j = 0; i < w * h * 3; i += 3, j += 4) {
    out[j] = rgb[i] * 255 + 0.5;
    out[j + 1] = rgb[i + 1] * 255 + 0.5;
    out[j + 2] = rgb[i + 2] * 255 + 0.5;
    out[j + 3] = 255;
  }
  return out;
}
