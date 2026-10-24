import type { Mat3, SourceMeta, Vec3 } from '../types';
import { diag, mul, whiteBalance709 } from '../color/matrices';
import { REC709_TO_WORKING, workingToOutput, OUTPUT_SPACES, type OutputSpaceId } from '../color/spaces';
import type { TransferId } from '../color/transfer';
import type { FrameStats } from './analysis';
import { bakeCurves, curveToLut, isIdentityCurve, LUT_SIZE, sampleLut } from './curves';
import { findLook } from './looks';
import { rangeLambdas, type Recipe } from './recipe';
import { blendEndpoints, type RollStats } from './roll';

export const FILM_TYPE_ID = { c41: 0, bw: 1, e6: 2 } as const;

/** Typical negative gamma; scene stops = density range / (γ · log10 2). */
const NEG_GAMMA = 0.6;
const SLIDE_GAMMA = 1.5;
/** Where the measured highlight percentile lands, in stops above 18% grey. */
export const HI_STOPS = 2.5;
/** 'fill' range: measured shadows land at HI_STOPS − FILL_STOPS, just above paper black. */
export const FILL_STOPS = 7.5;

/** Hue centres of the 8 HSL bands, OkLCh degrees. */
export const HSL_CENTRES = [25, 55, 100, 140, 195, 250, 290, 330];

/**
 * Every number the per-pixel conversion needs. Computed once per edit on the CPU,
 * then either uploaded as uniforms or fed to the reference implementation.
 */
export interface RenderParams {
  /** 0 C-41, 1 B&W, 2 E-6, 3 positive photo (no inversion: E is the input itself). */
  filmType: 0 | 1 | 2 | 3;
  logBase: Vec3;
  bwMix: Vec3;
  /** Normalized density: n = (D − dLo) · invRange (sign flipped for slides). */
  dLo: Vec3;
  invRange: Vec3;
  /** Midtone bend per channel, n' = n + m·n(1 − n). */
  mid: Vec3;
  /** Neutral-picker offsets added to n'. */
  offset: Vec3;
  sceneStops: number;
  hiStops: number;
  /** Camera RGB → working (linear Rec.2020), including exposure and white balance. */
  camToWork: Mat3;
  /** 0 = linear, 1 = paper curve (`paperLut`). */
  printMode: 0 | 1;
  /** Encoded print value over x = stops from 18% grey in [PAPER_X_MIN, PAPER_X_MAX]. */
  paperLut: Float32Array;
  /** Exposure added by midtone metering (stops), for display. */
  autoEv: number;
  /** Source texture holds sRGB-encoded 8-bit values (camera/photos) rather than linear. */
  srcSrgb: boolean;
  /** Film grain amount, 0..1 (GPU only; applied per output pixel). */
  grain: number;
  grainSeed: number;
  shadows: number;
  highlights: number;
  preserveHue: boolean;
  saturation: number;
  vibrance: number;
  hslHue: number[];
  hslSat: number[];
  hslLum: number[];
  hslActive: boolean;
  workToOut: Mat3;
  outTransfer: TransferId;
  /** Three 1024-entry LUTs (R, G, B), null when the curves are identity. */
  curves: Float32Array | null;
}

export interface FrameEndpoints {
  dLo: Vec3;
  dHi: Vec3;
  dMed: Vec3;
  sceneStops: number;
}

/** Resolve density endpoints for a frame, honouring base mode and the roll blend. */
export function resolveEndpoints(
  recipe: Recipe,
  stats: FrameStats,
  roll: RollStats | null,
): { base: Vec3; ends: FrameEndpoints } {
  const manual = recipe.film.baseMode === 'manual' && recipe.film.baseRgb;
  const base: Vec3 = manual ? recipe.film.baseRgb! : roll ? roll.base : stats.base;
  const logB = base.map((b) => Math.log10(Math.max(b, 1e-6))) as Vec3;
  const { exp, col } = rangeLambdas(recipe.convert);
  const isBW = recipe.film.type === 'bw';
  let dLo: Vec3;
  let dHi: Vec3;
  let dMed: Vec3;
  if (isBW) {
    const mixSum = recipe.film.bwMix.reduce((a, b) => a + b, 0) || 1;
    const lb = recipe.film.bwMix.reduce((a, m, c) => a + (m / mixSum) * logB[c], 0);
    const lo = (roll ? roll.bwLo + exp * (stats.bw.lo - roll.bwLo) : stats.bw.lo) + lb;
    const hi = (roll ? roll.bwHi + exp * (stats.bw.hi - roll.bwHi) : stats.bw.hi) + lb;
    dLo = [lo, lo, lo];
    dHi = [hi, hi, hi];
    dMed = [stats.bw.med + lb, stats.bw.med + lb, stats.bw.med + lb];
  } else {
    const lLo = roll ? blendEndpoints(stats.lLo, roll.lLo, exp, col) : stats.lLo;
    const lHi = roll ? blendEndpoints(stats.lHi, roll.lHi, exp, col) : stats.lHi;
    dLo = lLo.map((v, c) => v + logB[c]) as Vec3;
    dHi = lHi.map((v, c) => v + logB[c]) as Vec3;
    dMed = stats.lMed.map((v, c) => v + logB[c]) as Vec3;
  }
  let auto = FILL_STOPS;
  if (recipe.convert.rangeAuto === 'physical') {
    const range = Math.max(0.05, dHi[1] - dLo[1]);
    const gamma = recipe.film.type === 'e6' ? SLIDE_GAMMA : NEG_GAMMA;
    auto = Math.min(11, Math.max(4.5, range / (gamma * Math.LOG10E * Math.LN2)));
  }
  return { base, ends: { dLo, dHi, dMed, sceneStops: recipe.convert.sceneStops ?? auto } };
}

export const PAPER_X_MIN = -8;
export const PAPER_X_MAX = 6;

/**
 * Print curves as encoded output vs stops from 18% grey. Shaped like a darkroom print:
 * a gentle toe (shadows keep separation instead of sinking), grey at 0.46 encoded,
 * and a long shoulder so highlights roll off rather than clip.
 */
type PaperId = Exclude<Recipe['tone']['printCurve'], 'linear'> | 'photo';

/** For ordinary photos: sRGB's own response (so no look = no change) with a soft shoulder. */
const PHOTO_POINTS: [number, number][] = (() => {
  const enc = (v: number) => (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055);
  const pts: [number, number][] = [];
  for (let x = -8; x <= 1.5; x += 0.5) pts.push([x, enc(0.18 * 2 ** x)]);
  pts.push([2, 0.9], [2.5, 0.965], [3.2, 1], [6, 1]);
  return pts;
})();

const PAPER_POINTS: Record<PaperId, [number, number][]> = {
  photo: PHOTO_POINTS,
  soft: [[-8, 0], [-6, 0.035], [-5, 0.07], [-4, 0.115], [-3, 0.17], [-2, 0.245], [-1, 0.345], [0, 0.46], [1, 0.63], [2, 0.8], [2.5, 0.87], [3, 0.92], [4, 0.975], [6, 1]],
  standard: [[-8, 0], [-6, 0.02], [-5, 0.045], [-4, 0.085], [-3, 0.135], [-2, 0.205], [-1, 0.315], [0, 0.46], [1, 0.67], [2, 0.86], [2.5, 0.92], [3, 0.96], [4, 0.99], [6, 1]],
  dense: [[-8, 0], [-6, 0.01], [-5, 0.025], [-4, 0.05], [-3, 0.09], [-2, 0.16], [-1, 0.285], [0, 0.46], [1, 0.69], [2, 0.89], [2.5, 0.94], [3, 0.975], [4, 0.995], [6, 1]],
};

/** Bake the paper curve, with contrast (stretch around grey) and blacks/whites (lift or crush the ends). */
export function bakePaper(curve: Recipe['tone']['printCurve'] | 'photo', contrast: number, blacks: number, whites: number): Float32Array {
  const pts = PAPER_POINTS[curve === 'linear' ? 'standard' : curve];
  const span = PAPER_X_MAX - PAPER_X_MIN;
  const base = curveToLut(pts.map(([x, y]) => ({ x: (x - PAPER_X_MIN) / span, y })), LUT_SIZE);
  const k = Math.pow(2, contrast * 0.8);
  const a = Math.max(-0.3, Math.min(0.3, blacks * 0.12));
  const b = Math.max(-0.3, Math.min(0.3, whites * 0.12));
  const out = new Float32Array(LUT_SIZE);
  for (let i = 0; i < LUT_SIZE; i++) {
    const x = PAPER_X_MIN + (i / (LUT_SIZE - 1)) * span;
    let y = sampleLut(base, (x * k - PAPER_X_MIN) / span);
    // Monotone for |a|, |b| < 1/3.
    y = y + a * (1 - y) ** 3 + b * y ** 3;
    out[i] = Math.min(1, Math.max(0, y));
  }
  return out;
}

/** Where auto exposure puts the frame's median brightness, in stops from grey. */
const METER_TARGET = 0.8;
const METER_STRENGTH = 0.7;

export function buildParams(
  recipe: Recipe,
  stats: FrameStats,
  roll: RollStats | null,
  meta: Pick<SourceMeta, 'toRec709'>,
  output: OutputSpaceId,
): RenderParams {
  const { base, ends } = resolveEndpoints(recipe, stats, roll);
  const filmType = FILM_TYPE_ID[recipe.film.type];
  const bal = recipe.convert.balance;

  let dLo: Vec3;
  let invRange: Vec3;
  const g = 1;
  if (bal === 'none' && filmType !== 1) {
    // Only divide out the base: use green's endpoints for every channel.
    dLo = [ends.dLo[g], ends.dLo[g], ends.dLo[g]];
    const inv = 1 / Math.max(0.05, ends.dHi[g] - ends.dLo[g]);
    invRange = [inv, inv, inv];
  } else {
    dLo = [...ends.dLo];
    invRange = ends.dLo.map((lo, c) => 1 / Math.max(0.05, ends.dHi[c] - lo)) as Vec3;
  }
  if (filmType === 2) {
    // Slides: highlights are the thinnest density.
    dLo = ends.dHi.map((hi) => (bal === 'none' ? ends.dHi[g] : hi)) as Vec3;
    invRange = invRange.map((v) => -v) as Vec3;
  }

  const mid: Vec3 = [0, 0, 0];
  if (bal === 'endpoints+mid' && filmType !== 1) {
    const med = ends.dMed.map((d, c) => Math.min(0.95, Math.max(0.05, (d - dLo[c]) * invRange[c])));
    for (let c = 0; c < 3; c++) {
      if (c === g) continue;
      const m = (recipe.convert.midStrength * (med[g] - med[c])) / (med[c] * (1 - med[c]));
      mid[c] = Math.min(0.6, Math.max(-0.6, m));
    }
  }

  // The film look adds offsets on top of the user's settings, scaled by its strength.
  const look = findLook(recipe.look.id);
  const ls = look ? recipe.look.strength : 0;
  const clamp1 = (v: number) => Math.max(-1.5, Math.min(1.5, v));
  const t = { ...recipe.tone };
  if (look?.tone) for (const [k, v] of Object.entries(look.tone)) t[k as keyof typeof look.tone] = clamp1(t[k as keyof typeof look.tone] + (v ?? 0) * ls);
  const c = { ...recipe.color, hslHue: [...recipe.color.hslHue], hslSat: [...recipe.color.hslSat], hslLum: [...recipe.color.hslLum] };
  if (look?.color) {
    for (const k of ['temp', 'tint', 'saturation', 'vibrance'] as const) c[k] = clamp1(c[k] + (look.color[k] ?? 0) * ls);
    for (const k of ['hslHue', 'hslSat', 'hslLum'] as const) look.color[k]?.forEach((v, i) => (c[k][i] = clamp1(c[k][i] + v * ls)));
  }
  // Midtone metering: highlights set the scale, but frames whose midtones fall low
  // (backlit, dusk, small bright areas) are lifted part of the way. It never darkens:
  // bright, high-key frames are already placed well by the highlight anchor.
  let autoEv = 0;
  if (recipe.convert.autoExposure && stats.bw.hi > stats.bw.lo) {
    const nMed = (stats.bw.med - stats.bw.lo) / (stats.bw.hi - stats.bw.lo);
    const nOf = filmType === 2 ? 1 - nMed : nMed;
    const medStops = HI_STOPS + ends.sceneStops * (nOf - 1);
    const { exp } = rangeLambdas(recipe.convert);
    autoEv = Math.max(0, Math.min(2, METER_STRENGTH * exp * (METER_TARGET - medStops)));
  }
  const exposureGain = Math.pow(2, t.exposure + autoEv);
  const camToWork = mul(
    REC709_TO_WORKING,
    mul(whiteBalance709(c.temp, c.tint), mul(meta.toRec709, diag([exposureGain, exposureGain, exposureGain]))),
  );

  const printMode = t.printCurve === 'linear' ? 0 : 1;
  const paperLut = bakePaper(t.printCurve, t.contrast, t.blacks, t.whites);

  const hslActive = [...c.hslHue, ...c.hslSat, ...c.hslLum].some((v) => v !== 0);
  const cv = recipe.curves;
  const lookCurves = look?.curves && ls > 0 ? look.curves : null;
  const curvesIdentity = !lookCurves && [cv.rgb, cv.r, cv.g, cv.b].every(isIdentityCurve);

  return {
    filmType,
    logBase: base.map((b) => Math.log10(Math.max(b, 1e-6))) as Vec3,
    bwMix: recipe.film.bwMix,
    dLo,
    invRange,
    mid,
    offset: [...recipe.convert.neutralOffset],
    sceneStops: ends.sceneStops,
    hiStops: HI_STOPS,
    camToWork,
    printMode,
    paperLut,
    autoEv,
    srcSrgb: false,
    grain: 0,
    grainSeed: 0,
    shadows: t.shadows,
    highlights: t.highlights,
    preserveHue: t.preserveHue,
    saturation: c.saturation,
    vibrance: c.vibrance,
    hslHue: [...c.hslHue],
    hslSat: [...c.hslSat],
    hslLum: [...c.hslLum],
    hslActive,
    workToOut: workingToOutput(output),
    outTransfer: OUTPUT_SPACES[output].transfer,
    curves: curvesIdentity ? null : bakeCurves(cv, lookCurves, ls),
  };
}

/**
 * Parameters for an ordinary (positive) photo with a film look: the camera mode.
 * No inversion; the look's tone, colour and curves are applied to the photo's own tones.
 */
export function buildPhotoParams(opts: { lookId: string | null; strength: number; grain: number; exposure?: number; grainSeed?: number }): RenderParams {
  const look = findLook(opts.lookId);
  const ls = look ? opts.strength : 0;
  const clamp1 = (v: number) => Math.max(-1.5, Math.min(1.5, v));
  const tone = { contrast: 0, highlights: 0, shadows: 0, whites: 0, blacks: 0 };
  if (look?.tone) for (const [k, v] of Object.entries(look.tone)) tone[k as keyof typeof tone] = clamp1((v ?? 0) * ls);
  const c = { temp: 0, tint: 0, saturation: 0, vibrance: 0, hslHue: new Array(8).fill(0), hslSat: new Array(8).fill(0), hslLum: new Array(8).fill(0) };
  if (look?.color) {
    for (const k of ['temp', 'tint', 'saturation', 'vibrance'] as const) c[k] = clamp1((look.color[k] ?? 0) * ls);
    for (const k of ['hslHue', 'hslSat', 'hslLum'] as const) look.color[k]?.forEach((v, i) => (c[k][i] = clamp1(v * ls)));
  }
  // B&W looks only shape tone, so a colour photo also needs desaturating.
  if (look?.bw && ls > 0) c.saturation = -Math.min(1, ls);
  const gain = Math.pow(2, opts.exposure ?? 0);
  const ident: [number, number][] = [[0, 0], [1, 1]];
  const line = ident.map(([x, y]) => ({ x, y }));
  const lookCurves = look?.curves && ls > 0 ? look.curves : null;
  return {
    filmType: 3,
    logBase: [0, 0, 0],
    bwMix: [0.25, 0.5, 0.25],
    dLo: [0, 0, 0],
    invRange: [1, 1, 1],
    mid: [0, 0, 0],
    offset: [0, 0, 0],
    sceneStops: 1,
    hiStops: 0,
    camToWork: mul(REC709_TO_WORKING, mul(whiteBalance709(c.temp, c.tint), diag([gain, gain, gain]))),
    printMode: 1,
    paperLut: bakePaper('photo', tone.contrast, tone.blacks, tone.whites),
    autoEv: 0,
    shadows: tone.shadows,
    highlights: tone.highlights,
    preserveHue: false,
    saturation: c.saturation,
    vibrance: c.vibrance,
    hslHue: c.hslHue,
    hslSat: c.hslSat,
    hslLum: c.hslLum,
    hslActive: [...c.hslHue, ...c.hslSat, ...c.hslLum].some((v) => v !== 0),
    workToOut: workingToOutput('srgb'),
    outTransfer: OUTPUT_SPACES.srgb.transfer,
    curves: lookCurves ? bakeCurves({ rgb: line, r: line, g: line, b: line }, lookCurves, ls) : null,
    srcSrgb: true,
    grain: opts.grain,
    grainSeed: opts.grainSeed ?? 0,
  };
}
