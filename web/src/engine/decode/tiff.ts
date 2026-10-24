import { fromArrayBuffer } from 'geotiff';
import type { Mat3 } from '../types';
import { IDENTITY } from '../color/matrices';
import { srgbDecode } from '../color/transfer';
import { type FullImage, linearLut } from './fullImage';
import { parseIcc } from './icc';

export type TiffTransfer = 'auto' | 'linear' | 'gamma22' | 'srgb';

export interface TiffResult {
  image: FullImage;
  toRec709: Mat3;
  lowPrecision: boolean;
}

/**
 * Decode a TIFF scan. Linearization comes from the embedded ICC profile when there is
 * one; otherwise 16-bit files are assumed linear (scanner "raw" output such as VueScan
 * or SilverFast HDR) and 8-bit files sRGB. `transfer` overrides the guess.
 */
export async function decodeTiff(buf: ArrayBuffer, transfer: TiffTransfer = 'auto'): Promise<TiffResult> {
  const tiff = await fromArrayBuffer(buf);
  const img = await tiff.getImage();
  const w = img.getWidth();
  const h = img.getHeight();
  const spp = img.getSamplesPerPixel();
  const bps = img.getBitsPerSample() as number;
  const ifd = img.fileDirectory;
  const fmt = ifd.hasTag('SampleFormat') ? ((await ifd.loadValue('SampleFormat')) as unknown) : undefined;
  const isFloat = (ArrayBuffer.isView(fmt) || Array.isArray(fmt) ? (fmt as ArrayLike<number>)[0] : fmt) === 3;
  const icc = ifd.hasTag('ICCProfile') ? ((await ifd.loadValue('ICCProfile')) as unknown as ArrayLike<number>) : undefined;
  const info = icc ? parseIcc(Uint8Array.from(icc as ArrayLike<number>)) : null;

  const rasters = (await img.readRasters({ interleave: true, samples: spp >= 3 ? [0, 1, 2] : [0] })) as unknown as
    | Uint8Array
    | Uint16Array
    | Float32Array;

  let data: Uint8Array | Uint16Array | Float32Array;
  if (spp >= 3) {
    data = rasters;
  } else {
    // Greyscale (B&W scans): replicate to RGB.
    const Ctor = rasters.constructor as new (n: number) => typeof rasters;
    data = new Ctor(w * h * 3);
    for (let i = 0; i < w * h; i++) data[i * 3] = data[i * 3 + 1] = data[i * 3 + 2] = rasters[i];
  }

  let decode: ((v: number) => number) | null;
  if (transfer === 'linear') decode = null;
  else if (transfer === 'gamma22') decode = (v) => Math.pow(v, 2.2);
  else if (transfer === 'srgb') decode = srgbDecode;
  else if (info) decode = info.decode;
  else decode = bps === 8 ? srgbDecode : null;

  let image: FullImage;
  if (isFloat) {
    const f = data as Float32Array;
    if (decode) for (let i = 0; i < f.length; i++) f[i] = decode(Math.max(0, f[i]));
    image = { width: w, height: h, data: f };
  } else if (bps === 16) {
    image = { width: w, height: h, data: data as Uint16Array, lut: linearLut(16, decode ?? undefined) };
  } else if (bps === 8) {
    image = { width: w, height: h, data: data as Uint8Array, lut: linearLut(8, decode ?? undefined) };
  } else {
    throw new Error(`Unsupported TIFF: ${bps}-bit samples.`);
  }
  return { image, toRec709: info?.toRec709 ?? IDENTITY, lowPrecision: bps === 8 };
}
