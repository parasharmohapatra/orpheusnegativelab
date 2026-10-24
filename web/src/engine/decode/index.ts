import type { DecodedFrame, InputKind, Mat3, SourceMeta } from '../types';
import { ANALYSIS_MAX, PROXY_MAX } from '../types';
import { IDENTITY, normalizeRows } from '../color/matrices';
import { srgbDecode } from '../color/transfer';
import { downsample, downsampleLinear, type FullImage, linearLut } from './fullImage';
import { decodeTiff, type TiffTransfer } from './tiff';

export const RAW_EXTENSIONS = [
  'cr2', 'cr3', 'crw', 'nef', 'nrw', 'arw', 'srf', 'sr2', 'dng', 'raf', 'orf', 'rw2', 'pef', 'srw', 'x3f', 'erf', 'mos', 'mrw', '3fr', 'fff', 'iiq', 'rwl',
];
export const TIFF_EXTENSIONS = ['tif', 'tiff'];
export const BITMAP_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp', 'avif'];
export const ACCEPTED_EXTENSIONS = [...RAW_EXTENSIONS, ...TIFF_EXTENSIONS, ...BITMAP_EXTENSIONS];

export function extensionOf(name: string): string {
  const i = name.lastIndexOf('.');
  return i < 0 ? '' : name.slice(i + 1).toLowerCase();
}

export function kindOf(name: string): InputKind | null {
  const ext = extensionOf(name);
  if (RAW_EXTENSIONS.includes(ext)) return 'raw';
  if (TIFF_EXTENSIONS.includes(ext)) return 'tiff';
  if (BITMAP_EXTENSIONS.includes(ext)) return 'bitmap';
  return null;
}

export interface DecodeOptions {
  /** Full resolution (export) instead of a fast half-size proxy decode. */
  full: boolean;
  tiffTransfer?: TiffTransfer;
}

export interface FullDecode {
  image: FullImage;
  meta: SourceMeta;
}

const FLIP_TO_ORIENTATION: Record<number, SourceMeta['orientation']> = {
  0: { rot90: 0, flipH: false },
  3: { rot90: 2, flipH: false },
  5: { rot90: 3, flipH: false },
  6: { rot90: 1, flipH: false },
};

async function decodeRawFile(file: File, full: boolean): Promise<FullDecode> {
  const { decodeRaw } = await import('./librawModule');
  const r = await decodeRaw(await file.arrayBuffer(), { halfSize: !full, quality: 3 });
  const m = r.rgbCam;
  const toRec709 = m && m.length === 9 && m.some((v) => v !== 0) ? normalizeRows(m as Mat3) : IDENTITY;
  return {
    image: { width: r.width, height: r.height, data: r.data, lut: linearLut(16) },
    meta: {
      kind: 'raw',
      fullWidth: full ? r.width : r.width * 2,
      fullHeight: full ? r.height : r.height * 2,
      toRec709,
      orientation: FLIP_TO_ORIENTATION[r.flip] ?? FLIP_TO_ORIENTATION[0],
      clipFraction: r.clipFraction,
      lowPrecision: false,
      exif: r.exif,
    },
  };
}

async function decodeBitmapFile(file: File): Promise<FullDecode> {
  // createImageBitmap applies EXIF orientation, so the pixels are already upright.
  const bmp = await createImageBitmap(file, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const canvas = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = canvas.getContext('2d', { colorSpace: 'srgb' })!;
  ctx.drawImage(bmp, 0, 0);
  const rgba = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
  const data = new Uint8Array(bmp.width * bmp.height * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    data[j] = rgba[i];
    data[j + 1] = rgba[i + 1];
    data[j + 2] = rgba[i + 2];
  }
  const w = bmp.width;
  const h = bmp.height;
  bmp.close();
  return {
    image: { width: w, height: h, data, lut: linearLut(8, srgbDecode) },
    meta: {
      kind: 'bitmap',
      fullWidth: w,
      fullHeight: h,
      toRec709: IDENTITY,
      orientation: { rot90: 0, flipH: false },
      lowPrecision: true,
      exif: {},
    },
  };
}

async function decodeTiffFile(file: File, transfer: TiffTransfer): Promise<FullDecode> {
  const t = await decodeTiff(await file.arrayBuffer(), transfer);
  return {
    image: t.image,
    meta: {
      kind: 'tiff',
      fullWidth: t.image.width,
      fullHeight: t.image.height,
      toRec709: t.toRec709,
      orientation: { rot90: 0, flipH: false },
      lowPrecision: t.lowPrecision,
      exif: {},
    },
  };
}

export async function decodeFull(file: File, opts: DecodeOptions): Promise<FullDecode> {
  const kind = kindOf(file.name);
  if (kind === 'raw') return decodeRawFile(file, opts.full);
  if (kind === 'tiff') return decodeTiffFile(file, opts.tiffTransfer ?? 'auto');
  if (kind === 'bitmap') return decodeBitmapFile(file);
  throw new Error(`Unsupported file type: ${file.name}`);
}

/** Decode for editing: a proxy for the viewer and a small image for statistics. */
export async function decodeForEditing(file: File, opts: Omit<DecodeOptions, 'full'> = {}): Promise<DecodedFrame> {
  const { image, meta } = await decodeFull(file, { ...opts, full: false });
  const proxy = downsample(image, PROXY_MAX);
  const analysis = downsampleLinear(proxy, ANALYSIS_MAX);
  return { meta, proxy, analysis };
}
