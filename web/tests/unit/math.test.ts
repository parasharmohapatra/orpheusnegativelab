import { describe, expect, it } from 'vitest';
import { apply, invert, mul, IDENTITY, PRIMARIES, rgbToRgb, rgbToXYZ, whiteBalance709 } from '../../src/engine/color/matrices';
import { srgbDecode, srgbEncode } from '../../src/engine/color/transfer';
import { oklabToWorking, workingToOklab } from '../../src/engine/color/oklab';
import { curveToLut, bakeCurves } from '../../src/engine/pipeline/curves';
import { applyAffine, invertAffine, outputToSource, withOrientation } from '../../src/engine/geometry/transform';
import { defaultRecipe, migrateRecipe, applyPartial } from '../../src/engine/pipeline/recipe';
import type { Mat3 } from '../../src/engine/types';

const close = (a: number[], b: number[], eps = 1e-4) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i], -Math.log10(eps)));

describe('colour matrices', () => {
  it('sRGB → XYZ matches the standard matrix', () => {
    close(rgbToXYZ(PRIMARIES.rec709), [0.4124, 0.3576, 0.1805, 0.2126, 0.7152, 0.0722, 0.0193, 0.1192, 0.9505], 1e-3);
  });
  it('round-trips and maps white to white', () => {
    const m = rgbToRgb(PRIMARIES.rec709, PRIMARIES.proPhoto);
    close(apply(m, [1, 1, 1]), [1, 1, 1], 1e-3);
    close(mul(m, invert(m)), IDENTITY as unknown as number[]);
  });
  it('white balance is identity at zero and keeps white neutral-ish otherwise', () => {
    expect(whiteBalance709(0, 0)).toEqual(IDENTITY);
    const warm = apply(whiteBalance709(0.5, 0), [0.5, 0.5, 0.5]);
    expect(warm[0]).toBeGreaterThan(warm[2]);
  });
  it('sRGB transfer round-trips', () => {
    for (const v of [0, 0.001, 0.2, 0.5, 1]) expect(srgbDecode(srgbEncode(v))).toBeCloseTo(v, 6);
  });
  it('OkLab round-trips in the working space', () => {
    close(oklabToWorking(workingToOklab([0.3, 0.2, 0.1])), [0.3, 0.2, 0.1], 1e-5);
  });
});

describe('curves', () => {
  it('identity points give an identity LUT', () => {
    const lut = curveToLut([{ x: 0, y: 0 }, { x: 1, y: 1 }]);
    expect(lut[512]).toBeCloseTo(512 / 1023, 5);
  });
  it('stays monotone through steep points', () => {
    const lut = curveToLut([{ x: 0, y: 0 }, { x: 0.3, y: 0.05 }, { x: 0.35, y: 0.8 }, { x: 1, y: 1 }]);
    for (let i = 1; i < lut.length; i++) expect(lut[i]).toBeGreaterThanOrEqual(lut[i - 1] - 1e-6);
  });
  it('bakes channel then master', () => {
    const r = defaultRecipe().curves;
    r.rgb = [{ x: 0, y: 0.1 }, { x: 1, y: 1 }];
    const baked = bakeCurves(r);
    expect(baked[0]).toBeCloseTo(0.1, 5);
  });
});

describe('geometry', () => {
  const g = (o: Partial<ReturnType<typeof defaultRecipe>['geometry']>) => ({ ...defaultRecipe().geometry, ...o });
  it('identity maps uv to uv', () => {
    close(applyAffine(outputToSource(300, 200, g({})), 0.25, 0.75), [0.25, 0.75]);
  });
  it('90° clockwise: output top-left comes from source bottom-left', () => {
    close(applyAffine(outputToSource(300, 200, g({ rot90: 1 })), 0, 0), [0, 1]);
    close(applyAffine(outputToSource(300, 200, g({ rot90: 1 })), 1, 0), [0, 0]);
  });
  it('flip mirrors horizontally', () => {
    close(applyAffine(outputToSource(300, 200, g({ flipH: true })), 0.2, 0.5), [0.8, 0.5]);
  });
  it('crop selects a sub-rectangle and inverts cleanly', () => {
    const m = outputToSource(300, 200, g({ crop: { x: 0.1, y: 0.2, w: 0.5, h: 0.5 } }));
    close(applyAffine(m, 0, 0), [0.1, 0.2]);
    close(applyAffine(invertAffine(m), 0.35, 0.45), [0.5, 0.5]);
  });
  it('straighten keeps the centre fixed', () => {
    close(applyAffine(outputToSource(300, 200, g({ straighten: 7 })), 0.5, 0.5), [0.5, 0.5]);
  });
  it('composes file orientation with user rotation', () => {
    expect(withOrientation(g({ rot90: 1 }), { rot90: 3, flipH: false }).rot90).toBe(0);
  });
});

describe('recipes', () => {
  it('migrates partial and junk input to a full recipe', () => {
    expect(migrateRecipe(null)).toEqual(defaultRecipe());
    const r = migrateRecipe({ tone: { exposure: 1 } });
    expect(r.tone.exposure).toBe(1);
    expect(r.tone.contrast).toBe(0);
  });
  it('applies partial sections without sharing references', () => {
    const part = { color: { saturation: 0.3 } };
    const r = applyPartial(defaultRecipe(), part);
    expect(r.color.saturation).toBe(0.3);
    expect(r.color.hslHue).toHaveLength(8);
  });
});

// Keep an explicit reference so the Mat3 import is used for typing above.
export type _M = Mat3;
