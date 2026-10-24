import type { LinearImage } from '../types';

/**
 * A decoded image at full resolution, kept in its native sample type to save memory
 * (a 60 MP RAW is 360 MB as Uint16 but 720 MB as Float32). `lut` linearizes integer
 * samples; float samples are already linear.
 */
export interface FullImage {
  width: number;
  height: number;
  data: Uint16Array | Uint8Array | Float32Array;
  /** Integer sample → linear value. Length 65536 or 256. Absent for float data. */
  lut?: Float32Array;
}

export function linearLut(bits: 8 | 16, decode: (v: number) => number = (v) => v): Float32Array {
  const n = bits === 8 ? 256 : 65536;
  const lut = new Float32Array(n);
  for (let i = 0; i < n; i++) lut[i] = decode(i / (n - 1));
  return lut;
}

/** Extract a region as linear Float32 RGB (for GPU tile upload). */
export function regionToFloat(img: FullImage, x: number, y: number, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h * 3);
  const { data, lut } = img;
  for (let row = 0; row < h; row++) {
    const src = ((y + row) * img.width + x) * 3;
    const dst = row * w * 3;
    if (lut) {
      for (let k = 0; k < w * 3; k++) out[dst + k] = lut[data[src + k]];
    } else {
      out.set((data as Float32Array).subarray(src, src + w * 3), dst);
    }
  }
  return out;
}

/** Area-average downsample to fit within `max` on the long edge, in linear light. */
export function downsample(img: FullImage, max: number): LinearImage {
  const s = Math.max(1, Math.max(img.width, img.height) / max);
  const w = Math.max(1, Math.round(img.width / s));
  const h = Math.max(1, Math.round(img.height / s));
  const out = new Float32Array(w * h * 3);
  const { data, lut } = img;
  const val = lut ? (i: number) => lut[data[i]] : (i: number) => data[i];
  const sx = img.width / w;
  const sy = img.height / h;
  for (let oy = 0; oy < h; oy++) {
    const y0 = Math.floor(oy * sy);
    const y1 = Math.max(y0 + 1, Math.floor((oy + 1) * sy));
    for (let ox = 0; ox < w; ox++) {
      const x0 = Math.floor(ox * sx);
      const x1 = Math.max(x0 + 1, Math.floor((ox + 1) * sx));
      let r = 0, g = 0, b = 0;
      for (let y = y0; y < y1; y++) {
        let i = (y * img.width + x0) * 3;
        for (let x = x0; x < x1; x++, i += 3) {
          r += val(i);
          g += val(i + 1);
          b += val(i + 2);
        }
      }
      const n = (y1 - y0) * (x1 - x0);
      const o = (oy * w + ox) * 3;
      out[o] = r / n;
      out[o + 1] = g / n;
      out[o + 2] = b / n;
    }
  }
  return { width: w, height: h, data: out };
}

/** Downsample an already-linear image further (proxy → analysis). */
export function downsampleLinear(img: LinearImage, max: number): LinearImage {
  return downsample({ width: img.width, height: img.height, data: img.data }, max);
}
