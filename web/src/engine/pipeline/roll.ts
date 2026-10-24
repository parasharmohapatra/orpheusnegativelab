import type { Vec3 } from '../types';
import type { FrameStats } from './analysis';

/** Robust roll-wide values derived from every analysed frame. */
export interface RollStats {
  base: Vec3;
  lLo: Vec3;
  lHi: Vec3;
  bwLo: number;
  bwHi: number;
  frames: number;
}

function median(xs: number[]): number {
  const s = xs.filter(Number.isFinite).sort((a, b) => a - b);
  if (!s.length) return NaN;
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function computeRollStats(frames: FrameStats[]): RollStats | null {
  if (!frames.length) return null;
  // Prefer bases measured on the rebate; fall back to all frames.
  const fromBorder = frames.filter((f) => f.baseFromBorder);
  const baseSrc = fromBorder.length >= Math.max(1, frames.length * 0.25) ? fromBorder : frames;
  const ch = (pick: (f: FrameStats) => Vec3, src = frames) =>
    [0, 1, 2].map((c) => median(src.map((f) => pick(f)[c]))) as Vec3;
  return {
    base: ch((f) => f.base, baseSrc),
    lLo: ch((f) => f.lLo),
    lHi: ch((f) => f.lHi),
    bwLo: median(frames.map((f) => f.bw.lo)),
    bwHi: median(frames.map((f) => f.bw.hi)),
    frames: frames.length,
  };
}

/**
 * Blend a frame's endpoint with the roll's. The deviation is split into a luminance
 * part (shared by all channels: exposure varies frame to frame) and a chroma part
 * (per-channel: the film's colour response shouldn't).
 */
export function blendEndpoints(frame: Vec3, roll: Vec3, lambdaExp: number, lambdaCol: number): Vec3 {
  const dev = [0, 1, 2].map((c) => frame[c] - roll[c]);
  const lum = (dev[0] + dev[1] + dev[2]) / 3;
  return [0, 1, 2].map((c) => roll[c] + lambdaExp * lum + lambdaCol * (dev[c] - lum)) as Vec3;
}
