/** Latest preview histogram, published by the viewer and drawn by the panel. */
export interface HistogramData {
  r: Uint32Array;
  g: Uint32Array;
  b: Uint32Array;
  clipLo: number;
  clipHi: number;
}

let latest: HistogramData | null = null;
const subs = new Set<(h: HistogramData | null) => void>();

export function publishHistogram(rgba: Uint8Array): void {
  const r = new Uint32Array(256);
  const g = new Uint32Array(256);
  const b = new Uint32Array(256);
  let lo = 0;
  let hi = 0;
  const n = rgba.length / 4;
  for (let i = 0; i < rgba.length; i += 4) {
    r[rgba[i]]++;
    g[rgba[i + 1]]++;
    b[rgba[i + 2]]++;
    if (rgba[i] === 0 && rgba[i + 1] === 0 && rgba[i + 2] === 0) lo++;
    if (rgba[i] === 255 || rgba[i + 1] === 255 || rgba[i + 2] === 255) hi++;
  }
  latest = { r, g, b, clipLo: lo / n, clipHi: hi / n };
  subs.forEach((s) => s(latest));
}

export function subscribeHistogram(fn: (h: HistogramData | null) => void): () => void {
  subs.add(fn);
  fn(latest);
  return () => subs.delete(fn);
}
