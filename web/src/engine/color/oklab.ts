import type { Mat3, Vec3 } from '../types';
import { apply, invert, mul } from './matrices';
import { WORKING_TO_REC709 } from './spaces';

// Björn Ottosson's OkLab, defined on linear sRGB (Rec.709) values.
const M1_709: Mat3 = [
  0.4122214708, 0.5363325363, 0.0514459929,
  0.2119034982, 0.6806995451, 0.1073969566,
  0.0883024619, 0.2817188376, 0.6299787005,
];
export const OKLAB_M2: Mat3 = [
  0.2104542553, 0.793617785, -0.0040720468,
  1.9779984951, -2.428592205, 0.4505937099,
  0.0259040371, 0.7827717662, -0.808675766,
];

/** Working (linear Rec.2020) → OkLab LMS, so the shader needs one matrix. */
export const WORKING_TO_LMS: Mat3 = mul(M1_709, WORKING_TO_REC709);
export const LMS_TO_WORKING: Mat3 = invert(WORKING_TO_LMS);
export const OKLAB_M2_INV: Mat3 = invert(OKLAB_M2);

export function workingToOklab(rgb: Vec3): Vec3 {
  const lms = apply(WORKING_TO_LMS, rgb);
  return apply(OKLAB_M2, [Math.cbrt(lms[0]), Math.cbrt(lms[1]), Math.cbrt(lms[2])]);
}

export function oklabToWorking(lab: Vec3): Vec3 {
  const l = apply(OKLAB_M2_INV, lab);
  return apply(LMS_TO_WORKING, [l[0] ** 3, l[1] ** 3, l[2] ** 3]);
}
