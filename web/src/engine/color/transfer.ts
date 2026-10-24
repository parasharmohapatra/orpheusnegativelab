/** Transfer function ids shared with the shader (keep in sync with convert.frag.glsl). */
export const TRANSFER = { srgb: 0, gamma22: 1, linear: 2, gamma18: 3 } as const;
export type TransferId = (typeof TRANSFER)[keyof typeof TRANSFER];

export function srgbEncode(x: number): number {
  return x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
}

export function srgbDecode(x: number): number {
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}

export function encode(x: number, t: TransferId): number {
  const v = Math.max(0, x);
  switch (t) {
    case TRANSFER.srgb:
      return srgbEncode(v);
    case TRANSFER.gamma22:
      return Math.pow(v, 1 / 2.19921875);
    case TRANSFER.gamma18:
      return Math.pow(v, 1 / 1.8);
    default:
      return v;
  }
}

export function decode(x: number, t: TransferId): number {
  const v = Math.max(0, x);
  switch (t) {
    case TRANSFER.srgb:
      return srgbDecode(v);
    case TRANSFER.gamma22:
      return Math.pow(v, 2.19921875);
    case TRANSFER.gamma18:
      return Math.pow(v, 1.8);
    default:
      return v;
  }
}

/** 256-entry sRGB → linear table for decoding 8-bit images quickly. */
export const SRGB_TO_LINEAR_8 = (() => {
  const t = new Float32Array(256);
  for (let i = 0; i < 256; i++) t[i] = srgbDecode(i / 255);
  return t;
})();
