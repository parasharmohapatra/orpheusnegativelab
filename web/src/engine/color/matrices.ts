import type { Mat3, Vec3 } from '../types';

export const IDENTITY: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

export function mul(a: Mat3, b: Mat3): Mat3 {
  const r = new Array(9) as Mat3;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
    }
  }
  return r;
}

export function apply(m: Mat3, v: Vec3): Vec3 {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}

export function invert(m: Mat3): Mat3 {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-12) throw new Error('Singular matrix');
  const s = 1 / det;
  return [
    A * s, -(b * i - c * h) * s, (b * f - c * e) * s,
    B * s, (a * i - c * g) * s, -(a * f - c * d) * s,
    C * s, -(a * h - b * g) * s, (a * e - b * d) * s,
  ];
}

export function diag(v: Vec3): Mat3 {
  return [v[0], 0, 0, 0, v[1], 0, 0, 0, v[2]];
}

export function scale(m: Mat3, s: number): Mat3 {
  return m.map((x) => x * s) as Mat3;
}

/** Normalize rows so each sums to 1 (white maps to white). */
export function normalizeRows(m: Mat3): Mat3 {
  const r = [...m] as Mat3;
  for (let i = 0; i < 3; i++) {
    const s = r[i * 3] + r[i * 3 + 1] + r[i * 3 + 2];
    if (Math.abs(s) > 1e-9) for (let j = 0; j < 3; j++) r[i * 3 + j] /= s;
  }
  return r;
}

export function xyToXYZ(x: number, y: number): Vec3 {
  return [x / y, 1, (1 - x - y) / y];
}

export interface Primaries {
  r: [number, number];
  g: [number, number];
  b: [number, number];
  w: [number, number];
}

export const D65: [number, number] = [0.3127, 0.329];
export const D50: [number, number] = [0.3457, 0.3585];

export const PRIMARIES = {
  rec709: { r: [0.64, 0.33], g: [0.3, 0.6], b: [0.15, 0.06], w: D65 },
  rec2020: { r: [0.708, 0.292], g: [0.17, 0.797], b: [0.131, 0.046], w: D65 },
  displayP3: { r: [0.68, 0.32], g: [0.265, 0.69], b: [0.15, 0.06], w: D65 },
  adobeRGB: { r: [0.64, 0.33], g: [0.21, 0.71], b: [0.15, 0.06], w: D65 },
  proPhoto: { r: [0.7347, 0.2653], g: [0.1596, 0.8404], b: [0.0366, 0.0001], w: D50 },
} satisfies Record<string, Primaries>;

/** RGB → XYZ for the given primaries, white normalized to Y = 1. */
export function rgbToXYZ(p: Primaries): Mat3 {
  const R = xyToXYZ(...p.r);
  const G = xyToXYZ(...p.g);
  const B = xyToXYZ(...p.b);
  const W = xyToXYZ(...p.w);
  const M: Mat3 = [R[0], G[0], B[0], R[1], G[1], B[1], R[2], G[2], B[2]];
  const S = apply(invert(M), W);
  return mul(M, diag(S));
}

const BRADFORD: Mat3 = [
  0.8951, 0.2664, -0.1614,
  -0.7502, 1.7135, 0.0367,
  0.0389, -0.0685, 1.0296,
];

/** Bradford chromatic adaptation from white `src` to white `dst` (XYZ). */
export function bradford(src: Vec3, dst: Vec3): Mat3 {
  const s = apply(BRADFORD, src);
  const d = apply(BRADFORD, dst);
  return mul(invert(BRADFORD), mul(diag([d[0] / s[0], d[1] / s[1], d[2] / s[2]]), BRADFORD));
}

/** Linear RGB in primaries `a` → linear RGB in primaries `b`, adapting white points. */
export function rgbToRgb(a: Primaries, b: Primaries): Mat3 {
  let m = rgbToXYZ(a);
  if (a.w[0] !== b.w[0] || a.w[1] !== b.w[1]) {
    m = mul(bradford(xyToXYZ(...a.w), xyToXYZ(...b.w)), m);
  }
  return mul(invert(rgbToXYZ(b)), m);
}

/**
 * CIE daylight / Planckian-ish chromaticity for a correlated colour temperature.
 * Uses the Kim et al. cubic spline approximation of the Planckian locus (1667–25000 K).
 */
export function cctToXY(T: number): [number, number] {
  const t = Math.min(25000, Math.max(1667, T));
  const t2 = t * t;
  const t3 = t2 * t;
  const x =
    t <= 4000
      ? -0.2661239e9 / t3 - 0.234358e6 / t2 + 0.8776956e3 / t + 0.17991
      : -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.24039;
  const x2 = x * x;
  const x3 = x2 * x;
  let y: number;
  if (t <= 2222) y = -1.1063814 * x3 - 1.3481102 * x2 + 2.18555832 * x - 0.20219683;
  else if (t <= 4000) y = -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867;
  else y = 3.081758 * x3 - 5.8733867 * x2 + 3.75112997 * x - 0.37001483;
  return [x, y];
}

/**
 * White-balance matrix in linear Rec.709 for the temperature/tint sliders.
 * `temp` and `tint` are in [-1, 1]; 0/0 is identity. Positive temp warms, positive tint adds magenta.
 */
export function whiteBalance709(temp: number, tint: number): Mat3 {
  if (temp === 0 && tint === 0) return IDENTITY;
  // Pretend the scene was lit by a source of this CCT and adapt it back to D65.
  // Warming = treating the image as if shot under a cooler (bluer) source.
  const cct = 6504 * Math.pow(2, temp * 1.2);
  let [x, y] = cctToXY(cct);
  // Tint moves perpendicular to the locus, approximately along the green–magenta axis.
  y -= tint * 0.03;
  x += tint * 0.008;
  const src = xyToXYZ(x, y);
  const dst = xyToXYZ(...D65);
  const toXYZ = rgbToXYZ(PRIMARIES.rec709);
  return mul(invert(toXYZ), mul(bradford(src, dst), toXYZ));
}
