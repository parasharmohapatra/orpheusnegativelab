import { describe, expect, it } from 'vitest';
import { denoiseChroma } from '../../src/engine/decode/denoise';

function rng(seed: number) {
  return () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
}

describe('chroma noise reduction', () => {
  const w = 64;
  const h = 48;
  const make = () => {
    const r = rng(7);
    const d = new Float32Array(w * h * 3);
    for (let i = 0; i < w * h; i++) {
      // Left half darker than right (a luminance edge); blue very noisy, like a negative scan.
      const base = (i % w) < w / 2 ? 0.01 : 0.03;
      d[i * 3] = base * (1 + (r() - 0.5) * 0.1);
      d[i * 3 + 1] = base * 0.7 * (1 + (r() - 0.5) * 0.1);
      d[i * 3 + 2] = base * 0.15 * (1 + (r() - 0.5) * 1.2);
    }
    d[(20 * w + 10) * 3 + 1] = 0.5; // hot green pixel
    return d;
  };
  const lum = (d: Float32Array, i: number) => (d[i * 3] + 2 * d[i * 3 + 1] + d[i * 3 + 2]) / 4;
  const blueSpread = (d: Float32Array) => {
    let s = 0;
    let s2 = 0;
    let n = 0;
    for (let y = 8; y < h - 8; y++) for (let x = 4; x < w / 2 - 6; x++) {
      const i = y * w + x;
      const v = d[i * 3 + 2] / d[i * 3 + 1];
      s += v;
      s2 += v * v;
      n++;
    }
    return Math.sqrt(s2 / n - (s / n) ** 2);
  };

  it('cuts blue/green ratio noise by more than 3×', () => {
    const noisy = make();
    const clean = make();
    denoiseChroma(clean, w, h, 2);
    expect(blueSpread(clean)).toBeLessThan(blueSpread(noisy) / 3);
  });

  it('leaves luminance (and the edge) untouched away from the hot pixel', () => {
    const noisy = make();
    const clean = make();
    denoiseChroma(clean, w, h, 2);
    for (const i of [5 * w + 5, 30 * w + w / 2 - 1, 30 * w + w / 2, 40 * w + 60]) expect(lum(clean, i)).toBeCloseTo(lum(noisy, i), 6);
  });

  it('removes the colour of an isolated hot pixel', () => {
    const clean = make();
    denoiseChroma(clean, w, h, 2);
    const i = 20 * w + 10;
    const g = clean[i * 3 + 1] / ((clean[i * 3] + clean[i * 3 + 2]) / 2);
    expect(g).toBeLessThan(3);
  });
});

describe('median3 sorting network', () => {
  it('matches a sort-based 3×3 median', async () => {
    const { median3 } = await import('../../src/engine/decode/denoise');
    const r = rng(42);
    const w = 23;
    const h = 17;
    const a = new Float32Array(w * h).map(() => r());
    const ref = a.slice();
    median3(a, new Float32Array(w * h), w, h);
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const v: number[] = [];
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) v.push(ref[(y + dy) * w + x + dx]);
      v.sort((p, q) => p - q);
      expect(a[y * w + x]).toBe(v[4]);
    }
  });
  it('half-resolution path still cleans chroma', () => {
    const w = 96;
    const h = 64;
    const r = rng(3);
    const d = new Float32Array(w * h * 3);
    for (let i = 0; i < w * h; i++) {
      d[i * 3] = 0.02;
      d[i * 3 + 1] = 0.014;
      d[i * 3 + 2] = 0.003 * (1 + (r() - 0.5) * 1.2);
    }
    const before = d.slice();
    denoiseChroma(d, w, h, 6);
    let s0 = 0;
    let s1 = 0;
    for (let i = 0; i < w * h; i++) {
      s0 += Math.abs(before[i * 3 + 2] / before[i * 3 + 1] - 0.214);
      s1 += Math.abs(d[i * 3 + 2] / d[i * 3 + 1] - 0.214);
    }
    expect(s1).toBeLessThan(s0 / 4);
  });
});
