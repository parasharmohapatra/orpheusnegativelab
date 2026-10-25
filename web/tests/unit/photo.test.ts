import { describe, expect, it } from 'vitest';
import { IDENTITY } from '../../src/engine/color/matrices';
import { srgbDecode } from '../../src/engine/color/transfer';
import { analyzeFrame, FULL_RECT } from '../../src/engine/pipeline/analysis';
import { buildParams } from '../../src/engine/pipeline/params';
import { convertPixel } from '../../src/engine/pipeline/reference';
import { defaultPhotoRecipe, defaultRecipe, setFilmType, type Recipe } from '../../src/engine/pipeline/recipe';
import type { LinearImage, Vec3 } from '../../src/engine/types';

// A small "photo": a few colour patches in linear sRGB.
const enc: Vec3[] = [[0.05, 0.05, 0.05], [0.2, 0.2, 0.2], [0.46, 0.46, 0.46], [0.7, 0.7, 0.7], [0.8, 0.3, 0.2], [0.2, 0.5, 0.8], [0.3, 0.6, 0.25]];
const lin = enc.map((p) => p.map(srgbDecode) as Vec3);
const img: LinearImage = (() => {
  const w = 70;
  const h = 40;
  const data = new Float32Array(w * h * 3);
  for (let i = 0; i < w * h; i++) data.set(lin[Math.floor((i % w) / 10)], i * 3);
  return { width: w, height: h, data };
})();
const stats = analyzeFrame(img, { cropArea: FULL_RECT, autoFrame: false, baseOverride: null, blackClip: 0.3, whiteClip: 0.3, bwMix: [0.25, 0.5, 0.25] });
const render = (r: Recipe, rgb: Vec3, wb?: Vec3) => convertPixel(buildParams(r, stats, null, { toRec709: IDENTITY, daylightWB: wb }, 'srgb'), rgb);

describe('photo mode', () => {
  it('passes an ordinary photo through unchanged with no look (below the soft shoulder)', () => {
    const r = defaultPhotoRecipe();
    lin.forEach((p, i) => {
      if (Math.max(...enc[i]) > 0.7) return;
      const o = render(r, p);
      for (let c = 0; c < 3; c++) expect(o[c]).toBeCloseTo(enc[i][c], 1);
    });
    expect(render(r, lin[2])[1]).toBeCloseTo(0.46, 2);
  });

  it('applies film looks in the expected direction', () => {
    const grey = lin[2];
    const gold = defaultPhotoRecipe();
    gold.look = { id: 'gold200', strength: 1 };
    const t800 = defaultPhotoRecipe();
    t800.look = { id: 'cinestill800t', strength: 1 };
    const g = render(gold, grey);
    const c = render(t800, grey);
    expect(g[0] - g[2]).toBeGreaterThan(0.02);
    expect(c[2] - c[0]).toBeGreaterThan(0.02);
  });

  it('B&W looks turn a colour photo black and white', () => {
    const r = defaultPhotoRecipe();
    r.look = { id: 'trix', strength: 1 };
    for (const p of lin) {
      const o = render(r, p);
      expect(Math.abs(o[0] - o[2])).toBeLessThan(0.02);
    }
  });

  it("applies the camera's daylight white balance for RAW photos", () => {
    const r = defaultPhotoRecipe();
    // Raw camera data is green-heavy; the daylight multipliers bring grey back to neutral.
    const o = render(r, [0.09, 0.2, 0.12], [2.2, 1, 1.6667]);
    expect(Math.abs(o[0] - o[1])).toBeLessThan(0.02);
    expect(Math.abs(o[2] - o[1])).toBeLessThan(0.02);
  });

  it('switching film type carries the matching defaults both ways', () => {
    const r = defaultRecipe();
    setFilmType(r, 'photo');
    expect([r.tone.printCurve, r.convert.autoExposure, r.geometry.autoFrame]).toEqual(['natural', false, false]);
    setFilmType(r, 'c41');
    expect([r.tone.printCurve, r.convert.autoExposure, r.geometry.autoFrame]).toEqual(['standard', true, true]);
  });
});
