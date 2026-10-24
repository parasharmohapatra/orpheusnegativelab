import type { LinearImage, Vec3 } from '../../src/engine/types';
import { srgbDecode } from '../../src/engine/color/transfer';

/** ColorChecker Classic, approximate sRGB 8-bit values (D65). */
const CHECKER_8BIT: Vec3[] = [
  [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67], [133, 128, 177], [103, 189, 170],
  [214, 126, 44], [80, 91, 166], [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
  [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31], [187, 86, 149], [8, 133, 161],
  [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
];

/** Scene patches in linear Rec.709: the checker plus a neutral wedge that sets the extremes. */
export const SCENE: Vec3[] = [
  ...CHECKER_8BIT.map((p) => p.map((v) => srgbDecode(v / 255)) as Vec3),
  ...[0.0015, 0.01, 0.05, 0.18, 0.45, 1.0].map((v) => [v, v, v] as Vec3),
];
export const NEUTRAL_18 = 24 + 3;

export interface FilmModel {
  /** Per-layer contrast (different values = crossover). */
  gamma: Vec3;
  fog: number;
  /** Orange mask density per channel. */
  mask: Vec3;
  /** Light source as seen by the camera. */
  light: Vec3;
}

export const PORTRA_ISH: FilmModel = {
  gamma: [0.64, 0.6, 0.56],
  fog: 0.15,
  mask: [0.25, 0.6, 0.9],
  light: [0.85, 0.92, 0.97],
};

const E_TOE = 0.0005;

/** Transmitted, camera-captured light for a scene value through the simulated negative. */
export function negativePixel(E: Vec3, film: FilmModel, exposure = 1): Vec3 {
  return [0, 1, 2].map((c) => {
    const D = film.fog + film.mask[c] + film.gamma[c] * Math.log10(Math.max(E[c] * exposure, E_TOE) / E_TOE);
    return film.light[c] * Math.pow(10, -D);
  }) as Vec3;
}

export function basePixel(film: FilmModel): Vec3 {
  return [0, 1, 2].map((c) => film.light[c] * Math.pow(10, -(film.fog + film.mask[c]))) as Vec3;
}

/**
 * A scanned frame: a grid of patches surrounded by unexposed rebate.
 * Returns the image and the pixel centre of each patch.
 */
export function makeNegative(
  film: FilmModel,
  exposure = 1,
  opts: { patch?: number; border?: number; cols?: number; holder?: number } = {},
): { img: LinearImage; centres: [number, number][] } {
  const patch = opts.patch ?? 16;
  const border = opts.border ?? 12;
  const cols = opts.cols ?? 6;
  const rows = Math.ceil(SCENE.length / cols);
  const holder = opts.holder ?? 0;
  const w = cols * patch + (border + holder) * 2;
  const h = rows * patch + (border + holder) * 2;
  const data = new Float32Array(w * h * 3);
  const base = basePixel(film);
  // A camera scan's holder is lifted by flare to a few percent of the base, not black.
  const holderPx = base.map((b) => b * 0.06) as Vec3;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const inHolder = x < holder || y < holder || x >= w - holder || y >= h - holder;
    data.set(inHolder ? holderPx : base, (y * w + x) * 3);
  }
  const centres: [number, number][] = [];
  SCENE.forEach((E, k) => {
    const px = negativePixel(E, film, exposure);
    const x0 = holder + border + (k % cols) * patch;
    const y0 = holder + border + Math.floor(k / cols) * patch;
    for (let y = y0; y < y0 + patch; y++) for (let x = x0; x < x0 + patch; x++) data.set(px, (y * w + x) * 3);
    centres.push([x0 + patch / 2, y0 + patch / 2]);
  });
  return { img: { width: w, height: h, data }, centres };
}
