import { describe, expect, it } from 'vitest';
import type { Vec3 } from '../../src/engine/types';
import { IDENTITY } from '../../src/engine/color/matrices';
import { linear709ToLab, deltaE2000 } from '../../src/engine/color/deltaE';
import { srgbDecode } from '../../src/engine/color/transfer';
import { analyzeFrame, FULL_RECT } from '../../src/engine/pipeline/analysis';
import { buildParams } from '../../src/engine/pipeline/params';
import { convertPixel } from '../../src/engine/pipeline/reference';
import { defaultRecipe, type Recipe } from '../../src/engine/pipeline/recipe';
import { computeRollStats } from '../../src/engine/pipeline/roll';
import { basePixel, makeNegative, negativePixel, NEUTRAL_18, PORTRA_ISH, SCENE } from './filmSim';

const META = { toRec709: IDENTITY };

function analyze(img: ReturnType<typeof makeNegative>['img'], r: Recipe) {
  return analyzeFrame(img, {
    cropArea: FULL_RECT,
    autoFrame: true,
    baseOverride: null,
    blackClip: r.convert.blackClip,
    whiteClip: r.convert.whiteClip,
    bwMix: r.film.bwMix,
  });
}

/** Convert every scene patch and return linear Rec.709 results. */
function convertScene(r: Recipe, exposure = 1, rollFrames?: number[]) {
  const { img } = makeNegative(PORTRA_ISH, exposure);
  const stats = analyze(img, r);
  const roll = rollFrames ? computeRollStats(rollFrames.map((e) => analyze(makeNegative(PORTRA_ISH, e).img, r))) : null;
  const p = buildParams(r, stats, roll, META, 'srgb');
  return { stats, out: SCENE.map((E) => convertPixel(p, negativePixel(E, PORTRA_ISH, exposure)).map(srgbDecode) as Vec3) };
}

/** Fit one global gain (in log space, on unclipped patches) then report ΔE2000. */
function scoreAgainstScene(out: Vec3[]) {
  const logs: number[] = [];
  out.forEach((o, i) => {
    if (o[1] > 0.002 && o[1] < 0.95 && SCENE[i][1] > 0.002) logs.push(Math.log(o[1] / SCENE[i][1]));
  });
  const gain = Math.exp(logs.reduce((a, b) => a + b, 0) / logs.length);
  const des = out.map((o, i) => {
    const ref = SCENE[i].map((v) => Math.min(1, v * gain)) as Vec3;
    return deltaE2000(linear709ToLab(ref), linear709ToLab(o));
  });
  const checker = des.slice(0, 24).sort((a, b) => a - b);
  return {
    mean: checker.reduce((a, b) => a + b, 0) / checker.length,
    p95: checker[Math.floor(checker.length * 0.95)],
  };
}

describe('film base detection', () => {
  it('finds the orange mask on the rebate and the frame inside it', () => {
    const { img } = makeNegative(PORTRA_ISH);
    const s = analyze(img, defaultRecipe());
    const truth = basePixel(PORTRA_ISH);
    expect(s.baseFromBorder).toBe(true);
    for (let c = 0; c < 3; c++) expect(s.base[c]).toBeCloseTo(truth[c], 5);
    // Border is 12 px of 120 × 76; the frame starts just inside it.
    expect(s.frame.x0).toBeGreaterThan(0.09);
    expect(s.frame.x0).toBeLessThan(0.13);
  });
});

describe('film holder', () => {
  it('excludes a flare-lifted holder border from the statistics', () => {
    const plain = analyze(makeNegative(PORTRA_ISH).img, defaultRecipe());
    const held = analyze(makeNegative(PORTRA_ISH, 1, { holder: 4, border: 8 }).img, defaultRecipe());
    expect(held.frame.x0).toBeGreaterThan(0.1);
    expect(held.baseFromBorder).toBe(true);
    for (let c = 0; c < 3; c++) expect(held.lHi[c]).toBeCloseTo(plain.lHi[c], 2);
  });
});

describe('dense outliers', () => {
  it('ignores a small dense cluster (holder leaking into a tilted frame) when measuring highlights', () => {
    const r = defaultRecipe();
    const clean = makeNegative(PORTRA_ISH, 1, { patch: 40 });
    const leaked = makeNegative(PORTRA_ISH, 1, { patch: 40 });
    const { width: w, height: h, data } = leaked.img;
    // ~1% of the frame: a wedge of near-opaque holder in one corner, inside the rebate.
    const opaque = basePixel(PORTRA_ISH).map((b) => b * 0.0015);
    let n = 0;
    for (let y = 12; y < h - 12 && n < w * h * 0.01; y++) for (let x = 12; x < 12 + Math.max(0, 40 - (y - 12)); x++) { data.set(opaque, (y * w + x) * 3); n++; }
    const a = analyze(clean.img, r);
    const b = analyze(leaked.img, r);
    for (let c = 0; c < 3; c++) expect(b.lHi[c]).toBeCloseTo(a.lHi[c], 1);
  });
});

describe('negative → positive round trip', () => {
  it('recovers scene colour from a masked, crossed-over negative (linear print, endpoints only)', () => {
    const r = defaultRecipe();
    r.tone.printCurve = 'linear';
    r.convert.balance = 'endpoints';
    r.convert.rangeAuto = 'physical';
    const { out } = convertScene(r);
    const s = scoreAgainstScene(out);
    expect(s.mean).toBeLessThan(2);
    expect(s.p95).toBeLessThan(5);
  });

  it('stays close with midtone balancing on', () => {
    const r = defaultRecipe();
    r.tone.printCurve = 'linear';
    r.convert.rangeAuto = 'physical';
    const { out } = convertScene(r);
    const s = scoreAgainstScene(out);
    expect(s.mean).toBeLessThan(3);
  });

  it('keeps the 18% grey neutral with the default print curve', () => {
    const { out } = convertScene(defaultRecipe());
    const lab = linear709ToLab(out[NEUTRAL_18]);
    expect(Math.hypot(lab[1], lab[2])).toBeLessThan(1.5);
  });

  it('is monotonic in exposure along the neutral wedge', () => {
    const { out } = convertScene(defaultRecipe());
    const wedge = out.slice(24).map((o) => o[1]);
    for (let i = 1; i < wedge.length; i++) expect(wedge[i]).toBeGreaterThan(wedge[i - 1]);
  });
});

describe('roll consistency', () => {
  const evs = Array.from({ length: 36 }, (_, i) => Math.pow(2, -2 + (4 * i) / 35));

  it('keeps grey neutral across ±2 EV in roll mode', () => {
    const r = defaultRecipe();
    r.convert.rangeMode = 'roll';
    for (const e of evs.filter((_, i) => i % 5 === 0)) {
      const { out } = convertScene(r, e, evs);
      const lab = linear709ToLab(out[NEUTRAL_18]);
      expect(Math.hypot(lab[1], lab[2])).toBeLessThan(1.5);
    }
  });

  it('frame mode normalizes exposure; roll mode keeps part of the difference', () => {
    const frame = defaultRecipe();
    const roll = defaultRecipe();
    roll.convert.rangeMode = 'roll';
    const g = (r: Recipe, e: number) => convertScene(r, e, evs).out[NEUTRAL_18][1];
    const frameRatio = g(frame, 2) / g(frame, 0.5);
    const rollRatio = g(roll, 2) / g(roll, 0.5);
    expect(Math.abs(Math.log2(frameRatio))).toBeLessThan(0.15);
    // λ_exp = 0.8 corrects most of a 2-stop misexposure (endpoints and metering); the rest shows through.
    expect(rollRatio).toBeGreaterThan(1.15);
    expect(rollRatio).toBeLessThan(2);
  });
});

describe('film models', () => {
  it('B&W produces neutral output', () => {
    const r = defaultRecipe();
    r.film.type = 'bw';
    const { out } = convertScene(r);
    for (const o of out) {
      expect(Math.abs(o[0] - o[1])).toBeLessThan(1e-3);
      expect(Math.abs(o[2] - o[1])).toBeLessThan(1e-3);
    }
  });
});

describe('film looks', () => {
  const run = (mut: (r: Recipe) => void) => {
    const r = defaultRecipe();
    mut(r);
    return convertScene(r).out;
  };
  it('strength 0 is identical to no look', () => {
    const none = run(() => undefined);
    const zero = run((r) => (r.look = { id: 'cinestill800t', strength: 0 }));
    none.forEach((o, i) => o.forEach((v, c) => expect(zero[i][c]).toBeCloseTo(v, 6)));
  });
  it('every colour look changes the image and stays within range', async () => {
    const { FILM_LOOKS } = await import('../../src/engine/pipeline/looks');
    const none = run(() => undefined);
    for (const look of FILM_LOOKS.filter((l) => !l.bw)) {
      const out = run((r) => (r.look = { id: look.id, strength: 1 }));
      const diff = out.reduce((a, o, i) => a + Math.abs(o[0] - none[i][0]) + Math.abs(o[2] - none[i][2]), 0);
      expect(diff, look.id).toBeGreaterThan(0.01);
      out.flat().forEach((v) => {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      });
    }
  });
  it('800T is cooler than Gold 200 on the grey patch', () => {
    const cool = run((r) => (r.look = { id: 'cinestill800t', strength: 1 }))[NEUTRAL_18];
    const warm = run((r) => (r.look = { id: 'gold200', strength: 1 }))[NEUTRAL_18];
    expect(cool[2] - cool[0]).toBeGreaterThan(warm[2] - warm[0]);
  });
  it('B&W looks keep B&W film neutral', () => {
    const out = run((r) => {
      r.film.type = 'bw';
      r.look = { id: 'trix', strength: 1 };
    });
    for (const o of out) expect(Math.abs(o[0] - o[2])).toBeLessThan(1e-3);
  });
});

describe('print curve and metering', () => {
  it('standard paper has a gentle toe and keeps grey at 0.46', async () => {
    const { bakePaper, PAPER_X_MIN, PAPER_X_MAX } = await import('../../src/engine/pipeline/params');
    const lut = bakePaper('standard', 0, 0, 0);
    const at = (x: number) => lut[Math.round(((x - PAPER_X_MIN) / (PAPER_X_MAX - PAPER_X_MIN)) * (lut.length - 1))];
    expect(at(0)).toBeCloseTo(0.46, 2);
    expect(at(-2)).toBeGreaterThan(0.18);
    expect(at(-3)).toBeGreaterThan(0.11);
    expect(at(3)).toBeLessThan(0.98);
    for (const curve of ['soft', 'standard', 'dense'] as const) {
      for (const [c, bl, wh] of [[0, 0, 0], [1, 1, 1], [-1, -1, -1]]) {
        const l = bakePaper(curve, c, bl, wh);
        for (let i = 1; i < l.length; i++) expect(l[i]).toBeGreaterThanOrEqual(l[i - 1] - 1e-6);
      }
    }
  });

  it('meters dark-median frames up, bounded, and can be turned off', () => {
    const { img } = makeNegative(PORTRA_ISH);
    const r = defaultRecipe();
    const stats = analyze(img, r);
    // Pretend the scene is mostly shadows with a small bright area.
    const dark = { ...stats, bw: { ...stats.bw, med: stats.bw.lo + 0.2 * (stats.bw.hi - stats.bw.lo) } };
    const on = buildParams(r, dark, null, META, 'srgb');
    expect(on.autoEv).toBeGreaterThan(0.5);
    expect(on.autoEv).toBeLessThanOrEqual(2);
    r.convert.autoExposure = false;
    expect(buildParams(r, dark, null, META, 'srgb').autoEv).toBe(0);
  });
});

describe('metering never darkens', () => {
  it('leaves bright-median (high-key) frames alone', () => {
    const { img } = makeNegative(PORTRA_ISH);
    const r = defaultRecipe();
    const stats = analyze(img, r);
    const bright = { ...stats, bw: { ...stats.bw, med: stats.bw.lo + 0.95 * (stats.bw.hi - stats.bw.lo) } };
    expect(buildParams(r, bright, null, META, 'srgb').autoEv).toBe(0);
  });
});
