import { describe, expect, it } from 'vitest';
import { buildPhotoParams } from '../../src/engine/pipeline/params';
import { convertPixel } from '../../src/engine/pipeline/reference';
import type { Vec3 } from '../../src/engine/types';

const samples: Vec3[] = [
  [0.05, 0.05, 0.05], [0.2, 0.2, 0.2], [0.46, 0.46, 0.46], [0.7, 0.7, 0.7],
  [0.8, 0.3, 0.2], [0.2, 0.5, 0.8], [0.3, 0.6, 0.25],
];

describe('camera (positive photo) mode', () => {
  it('with no look, leaves shadows and midtones essentially unchanged', () => {
    const p = buildPhotoParams({ lookId: null, strength: 1, grain: 0 });
    for (const s of samples) {
      const o = convertPixel(p, s);
      // Below the soft highlight shoulder the photo passes through.
      if (Math.max(...s) <= 0.7) for (let c = 0; c < 3; c++) expect(o[c]).toBeCloseTo(s[c], 1);
    }
    expect(convertPixel(p, [0.46, 0.46, 0.46])[1]).toBeCloseTo(0.46, 2);
  });

  it('rolls bright highlights off instead of clipping', () => {
    const p = buildPhotoParams({ lookId: null, strength: 1, grain: 0 });
    const a = convertPixel(p, [0.9, 0.9, 0.9])[1];
    const b = convertPixel(p, [1, 1, 1])[1];
    expect(b).toBeGreaterThan(a);
    expect(b).toBeLessThanOrEqual(1);
  });

  it('looks change the photo in the expected direction', () => {
    const grey: Vec3 = [0.46, 0.46, 0.46];
    const gold = convertPixel(buildPhotoParams({ lookId: 'gold200', strength: 1, grain: 0 }), grey);
    const t800 = convertPixel(buildPhotoParams({ lookId: 'cinestill800t', strength: 1, grain: 0 }), grey);
    expect(gold[0] - gold[2]).toBeGreaterThan(0.02);
    expect(t800[2] - t800[0]).toBeGreaterThan(0.02);
    const half = convertPixel(buildPhotoParams({ lookId: 'gold200', strength: 0.5, grain: 0 }), grey);
    expect(half[0] - half[2]).toBeLessThan(gold[0] - gold[2]);
  });
});
