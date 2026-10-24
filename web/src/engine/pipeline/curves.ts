import type { CurvePoint } from './recipe';

export const LUT_SIZE = 1024;

/**
 * Monotone cubic (Fritsch–Carlson) interpolation through the points, sampled to a LUT.
 * Monotone means no overshoot: dragging a point never makes the curve fold back.
 */
export function curveToLut(points: CurvePoint[], size = LUT_SIZE): Float32Array {
  const pts = [...points].sort((a, b) => a.x - b.x);
  const out = new Float32Array(size);
  if (pts.length < 2) {
    for (let i = 0; i < size; i++) out[i] = i / (size - 1);
    return out;
  }
  const n = pts.length;
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const d: number[] = [];
  for (let i = 0; i < n - 1; i++) d.push((ys[i + 1] - ys[i]) / Math.max(1e-9, xs[i + 1] - xs[i]));
  const m = new Array<number>(n);
  m[0] = d[0];
  m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (d[i] === 0) {
      m[i] = 0;
      m[i + 1] = 0;
      continue;
    }
    const a = m[i] / d[i];
    const b = m[i + 1] / d[i];
    const s = a * a + b * b;
    if (s > 9) {
      const t = 3 / Math.sqrt(s);
      m[i] = t * a * d[i];
      m[i + 1] = t * b * d[i];
    }
  }
  let seg = 0;
  for (let k = 0; k < size; k++) {
    const x = k / (size - 1);
    if (x <= xs[0]) {
      out[k] = ys[0];
      continue;
    }
    if (x >= xs[n - 1]) {
      out[k] = ys[n - 1];
      continue;
    }
    while (seg < n - 2 && x > xs[seg + 1]) seg++;
    const h = xs[seg + 1] - xs[seg];
    const t = (x - xs[seg]) / h;
    const t2 = t * t;
    const t3 = t2 * t;
    out[k] =
      (2 * t3 - 3 * t2 + 1) * ys[seg] +
      (t3 - 2 * t2 + t) * h * m[seg] +
      (-2 * t3 + 3 * t2) * ys[seg + 1] +
      (t3 - t2) * h * m[seg + 1];
    out[k] = Math.min(1, Math.max(0, out[k]));
  }
  return out;
}

export function sampleLut(lut: Float32Array, x: number): number {
  const f = Math.min(1, Math.max(0, x)) * (lut.length - 1);
  const i = Math.floor(f);
  const j = Math.min(lut.length - 1, i + 1);
  const t = f - i;
  return lut[i] * (1 - t) + lut[j] * t;
}

export function isIdentityCurve(points: CurvePoint[]): boolean {
  return points.every((p) => Math.abs(p.x - p.y) < 1e-4);
}

/** Bake per-channel curve then master curve into three LUTs, packed R,G,B consecutively. */
export function bakeCurves(c: { rgb: CurvePoint[]; r: CurvePoint[]; g: CurvePoint[]; b: CurvePoint[] }): Float32Array {
  const master = curveToLut(c.rgb);
  const out = new Float32Array(LUT_SIZE * 3);
  [c.r, c.g, c.b].forEach((pts, ch) => {
    const lut = curveToLut(pts);
    for (let i = 0; i < LUT_SIZE; i++) out[ch * LUT_SIZE + i] = sampleLut(master, lut[i]);
  });
  return out;
}
