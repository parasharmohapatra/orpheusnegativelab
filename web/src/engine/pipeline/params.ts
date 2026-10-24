import type { Mat3, SourceMeta, Vec3 } from '../types';
import { diag, mul, whiteBalance709 } from '../color/matrices';
import { REC709_TO_WORKING, workingToOutput, OUTPUT_SPACES, type OutputSpaceId } from '../color/spaces';
import type { TransferId } from '../color/transfer';
import type { FrameStats } from './analysis';
import { bakeCurves, isIdentityCurve } from './curves';
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
  filmType: 0 | 1 | 2;
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
  /** 0 = linear, 1 = paper sigmoid. */
  printMode: 0 | 1;
  paperC: number;
  paperX0: number;
  paperSLo: number;
  paperInvRange: number;
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

function solvePaperX0(c: number, xb: number, xw: number, greyEncoded: number): number {
  const s = (x: number, x0: number) => 1 / (1 + Math.exp(-c * (x - x0)));
  const f = (x0: number) => (s(0, x0) - s(xb, x0)) / (s(xw, x0) - s(xb, x0));
  let lo = -8;
  let hi = 8;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (f(mid) > greyEncoded) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

const PAPER_C: Record<Recipe['tone']['printCurve'], number> = { linear: 0, soft: 0.62, standard: 0.78, dense: 0.95 };

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

  const t = recipe.tone;
  const exposureGain = Math.pow(2, t.exposure);
  const camToWork = mul(
    REC709_TO_WORKING,
    mul(whiteBalance709(recipe.color.temp, recipe.color.tint), mul(meta.toRec709, diag([exposureGain, exposureGain, exposureGain]))),
  );

  const printMode = t.printCurve === 'linear' ? 0 : 1;
  const paperC = PAPER_C[t.printCurve] * Math.pow(2, t.contrast * 0.8);
  const xb = -7 - t.blacks * 1.5;
  const xw = 3 - t.whites * 1.2;
  const paperX0 = printMode ? solvePaperX0(paperC, xb, xw, 0.46) : 0;
  const sig = (x: number) => 1 / (1 + Math.exp(-paperC * (x - paperX0)));
  const paperSLo = printMode ? sig(xb) : 0;
  const paperInvRange = printMode ? 1 / (sig(xw) - paperSLo) : 1;

  const c = recipe.color;
  const hslActive = [...c.hslHue, ...c.hslSat, ...c.hslLum].some((v) => v !== 0);
  const cv = recipe.curves;
  const curvesIdentity = [cv.rgb, cv.r, cv.g, cv.b].every(isIdentityCurve);

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
    paperC,
    paperX0,
    paperSLo,
    paperInvRange,
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
    curves: curvesIdentity ? null : bakeCurves(cv),
  };
}
