import type { Vec3 } from '../types';

export type FilmType = 'c41' | 'bw' | 'e6';
export type RangeMode = 'frame' | 'roll' | 'blend';
export type BalanceMode = 'endpoints' | 'endpoints+mid' | 'neutral' | 'none';
export type PrintCurve = 'linear' | 'soft' | 'standard' | 'dense';

export interface CurvePoint {
  x: number;
  y: number;
}

export const HSL_BANDS = ['Red', 'Orange', 'Yellow', 'Green', 'Aqua', 'Blue', 'Purple', 'Magenta'] as const;

export interface CropRect {
  /** Normalized to the oriented, straightened image. */
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Every edit for one frame. Plain JSON: persisted, synced between frames and diffed for undo.
 * Bump RECIPE_VERSION and add a step to `migrateRecipe` whenever the shape changes.
 */
export interface Recipe {
  v: 1;
  film: {
    type: FilmType;
    /** 'auto' detects the base per frame and uses the roll median; 'manual' uses `rgb`. */
    baseMode: 'auto' | 'manual';
    baseRgb: Vec3 | null;
    /** Channel weights for B&W density. */
    bwMix: Vec3;
  };
  convert: {
    rangeMode: RangeMode;
    /** How much each frame's own exposure / colour deviates from the roll (Blend mode). */
    lambdaExp: number;
    lambdaCol: number;
    balance: BalanceMode;
    /** Normalized-density offsets set by the neutral eyedropper (balance = 'neutral'). */
    neutralOffset: Vec3;
    /** Midtone balance strength for 'endpoints+mid'. */
    midStrength: number;
    /** Scene range in stops; null = automatic (see `rangeAuto`). */
    sceneStops: number | null;
    /**
     * Automatic scene range: 'fill' stretches the measured density range to fill the
     * print (like a lab scanner or NLP); 'physical' keeps the scene's own contrast
     * using a typical negative gamma.
     */
    rangeAuto: 'fill' | 'physical';
    /** Meter midtones and nudge exposure toward a natural brightness. */
    autoExposure: boolean;
    /** Percent of pixels allowed to clip at each end when measuring the density range. */
    blackClip: number;
    whiteClip: number;
  };
  tone: {
    /** Stops. */
    exposure: number;
    /** All below in [-1, 1]. */
    contrast: number;
    highlights: number;
    shadows: number;
    whites: number;
    blacks: number;
    printCurve: PrintCurve;
    preserveHue: boolean;
  };
  color: {
    temp: number;
    tint: number;
    saturation: number;
    vibrance: number;
    hslHue: number[];
    hslSat: number[];
    hslLum: number[];
  };
  curves: {
    rgb: CurvePoint[];
    r: CurvePoint[];
    g: CurvePoint[];
    b: CurvePoint[];
  };
  /** Film look layered on top of the adjustments above (see looks.ts). */
  look: {
    id: string | null;
    /** 0..1.5; 1 = as designed. */
    strength: number;
  };
  geometry: {
    rot90: 0 | 1 | 2 | 3;
    flipH: boolean;
    /** Degrees, positive = clockwise. */
    straighten: number;
    crop: CropRect;
    /** Inset the statistics area automatically to skip the film rebate and holder. */
    autoFrame: boolean;
  };
}

export const RECIPE_VERSION = 1;

const LINE: CurvePoint[] = [
  { x: 0, y: 0 },
  { x: 1, y: 1 },
];

export function defaultRecipe(): Recipe {
  return {
    v: 1,
    film: { type: 'c41', baseMode: 'auto', baseRgb: null, bwMix: [0.25, 0.5, 0.25] },
    convert: {
      rangeMode: 'frame',
      lambdaExp: 1,
      lambdaCol: 1,
      balance: 'endpoints+mid',
      neutralOffset: [0, 0, 0],
      midStrength: 0.75,
      sceneStops: null,
      rangeAuto: 'fill',
      autoExposure: true,
      blackClip: 0.3,
      whiteClip: 0.3,
    },
    tone: {
      exposure: 0,
      contrast: 0,
      highlights: 0,
      shadows: 0,
      whites: 0,
      blacks: 0,
      printCurve: 'standard',
      preserveHue: false,
    },
    color: {
      temp: 0,
      tint: 0,
      saturation: 0,
      vibrance: 0,
      hslHue: new Array(8).fill(0),
      hslSat: new Array(8).fill(0),
      hslLum: new Array(8).fill(0),
    },
    curves: { rgb: [...LINE], r: [...LINE], g: [...LINE], b: [...LINE] },
    look: { id: null, strength: 1 },
    geometry: {
      rot90: 0,
      flipH: false,
      straighten: 0,
      crop: { x: 0, y: 0, w: 1, h: 1 },
      autoFrame: true,
    },
  };
}

/** λ values each range mode implies. Blend uses whatever the recipe holds. */
export function rangeLambdas(c: Recipe['convert']): { exp: number; col: number } {
  if (c.rangeMode === 'frame') return { exp: 1, col: 1 };
  if (c.rangeMode === 'roll') return { exp: 0.8, col: 0.2 };
  return { exp: c.lambdaExp, col: c.lambdaCol };
}

export type RecipeSection = Exclude<keyof Recipe, 'v'>;
export const RECIPE_SECTIONS: { key: RecipeSection; label: string }[] = [
  { key: 'film', label: 'Film & base' },
  { key: 'convert', label: 'Conversion' },
  { key: 'tone', label: 'Tone' },
  { key: 'color', label: 'Colour' },
  { key: 'curves', label: 'Curves' },
  { key: 'look', label: 'Film look' },
  { key: 'geometry', label: 'Crop & rotate' },
];

export type PartialRecipe = { [K in RecipeSection]?: Partial<Recipe[K]> };

/** Overlay a partial recipe (preset or synced sections) onto a full one. */
export function applyPartial(base: Recipe, part: PartialRecipe): Recipe {
  const out = structuredClone(base);
  for (const key of Object.keys(part) as RecipeSection[]) {
    const p = part[key];
    if (p) Object.assign(out[key], structuredClone(p));
  }
  return out;
}

export function pickSections(r: Recipe, sections: RecipeSection[]): PartialRecipe {
  const out: PartialRecipe = {};
  for (const s of sections) (out as Record<string, unknown>)[s] = structuredClone(r[s]);
  return out;
}

/** Accepts anything that looks like a stored recipe and returns a valid current-version one. */
export function migrateRecipe(raw: unknown): Recipe {
  const d = defaultRecipe();
  if (!raw || typeof raw !== 'object') return d;
  const r = raw as Partial<Recipe> & { v?: number };
  // v1 is the first version; unknown or missing fields fall back to defaults.
  const merged = applyPartial(d, r as PartialRecipe);
  merged.v = 1;
  return merged;
}
