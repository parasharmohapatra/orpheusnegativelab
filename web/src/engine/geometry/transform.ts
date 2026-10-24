import type { CropRect, Recipe } from '../pipeline/recipe';

/** 2D affine map as a row-major 3×3 with an implicit last row [0 0 1]. */
export type Affine = [number, number, number, number, number, number];

export const AFFINE_IDENTITY: Affine = [1, 0, 0, 0, 1, 0];

/** a ∘ b : apply b first, then a. */
export function compose(a: Affine, b: Affine): Affine {
  return [
    a[0] * b[0] + a[1] * b[3],
    a[0] * b[1] + a[1] * b[4],
    a[0] * b[2] + a[1] * b[5] + a[2],
    a[3] * b[0] + a[4] * b[3],
    a[3] * b[1] + a[4] * b[4],
    a[3] * b[2] + a[4] * b[5] + a[5],
  ];
}

export function invertAffine(m: Affine): Affine {
  const det = m[0] * m[4] - m[1] * m[3];
  const s = 1 / det;
  const a = m[4] * s;
  const b = -m[1] * s;
  const d = -m[3] * s;
  const e = m[0] * s;
  return [a, b, -(a * m[2] + b * m[5]), d, e, -(d * m[2] + e * m[5])];
}

export function applyAffine(m: Affine, x: number, y: number): [number, number] {
  return [m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]];
}

const translate = (x: number, y: number): Affine => [1, 0, x, 0, 1, y];
const scale = (sx: number, sy: number): Affine => [sx, 0, 0, 0, sy, 0];
const rotate = (rad: number): Affine => {
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [c, -s, 0, s, c, 0];
};

export type Geometry = Pick<Recipe['geometry'], 'rot90' | 'flipH' | 'straighten'> & { crop: CropRect };

/** Size of the image after 90° rotation (straightening keeps the canvas size). */
export function orientedSize(w: number, h: number, rot90: number): [number, number] {
  return rot90 % 2 === 1 ? [h, w] : [w, h];
}

/** Output pixel size of the cropped frame for a source of w×h. */
export function outputSize(w: number, h: number, g: Geometry): [number, number] {
  const [ow, oh] = orientedSize(w, h, g.rot90);
  return [Math.max(1, Math.round(ow * g.crop.w)), Math.max(1, Math.round(oh * g.crop.h))];
}

/**
 * Maps normalized output coordinates (0..1 across the cropped frame, y down)
 * to normalized source coordinates (0..1 across the decoded image, y down).
 * The source aspect matters for straightening, so pass its pixel size.
 */
export function outputToSource(srcW: number, srcH: number, g: Geometry): Affine {
  const [ow, oh] = orientedSize(srcW, srcH, g.rot90);
  // 1. crop: output uv → oriented normalized
  let m: Affine = [g.crop.w, 0, g.crop.x, 0, g.crop.h, g.crop.y];
  // 2. oriented normalized → oriented pixels
  m = compose(scale(ow, oh), m);
  // 3. undo straighten about the centre
  const theta = (g.straighten * Math.PI) / 180;
  m = compose(translate(ow / 2, oh / 2), compose(rotate(-theta), compose(translate(-ow / 2, -oh / 2), m)));
  // 4. undo the mirror
  if (g.flipH) m = compose([-1, 0, ow, 0, 1, 0], m);
  // 5. undo the 90° rotation: oriented pixels → source pixels
  let r: Affine;
  switch (g.rot90) {
    case 1: r = [0, 1, 0, -1, 0, srcH]; break; // sx = oy, sy = H - ox
    case 2: r = [-1, 0, srcW, 0, -1, srcH]; break;
    case 3: r = [0, -1, srcW, 1, 0, 0]; break; // sx = W - oy, sy = ox
    default: r = AFFINE_IDENTITY;
  }
  m = compose(r, m);
  // 6. source pixels → source normalized
  return compose(scale(1 / srcW, 1 / srcH), m);
}

/** Largest centred crop of the same aspect that stays inside an image straightened by `deg`. */
export function inscribedScale(w: number, h: number, deg: number): number {
  const t = Math.abs((deg * Math.PI) / 180);
  const c = Math.cos(t);
  const s = Math.sin(t);
  return Math.min(w / (w * c + h * s), h / (w * s + h * c));
}

export function clampCrop(c: CropRect): CropRect {
  const w = Math.min(1, Math.max(0.02, c.w));
  const h = Math.min(1, Math.max(0.02, c.h));
  return { w, h, x: Math.min(1 - w, Math.max(0, c.x)), y: Math.min(1 - h, Math.max(0, c.y)) };
}

/** Compose the file's own orientation with the user's edit (both clockwise steps, mirror after). */
export function withOrientation(
  g: Geometry,
  o: { rot90: 0 | 1 | 2 | 3; flipH: boolean },
): Geometry {
  // user ∘ file = flipU · rotU · flipF · rotF, and rot · flip = flip · rot⁻¹.
  const r = (o.rot90 + (o.flipH ? 4 - g.rot90 : g.rot90)) % 4;
  return { ...g, rot90: r as 0 | 1 | 2 | 3, flipH: o.flipH !== g.flipH };
}
