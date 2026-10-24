import type { LinearImage, Vec3 } from '../types';

/** Values at or above this (linear, 1 = sensor/file white) count as clipped. */
export const CLIP_LEVEL = 0.98;
const EPS = 1e-6;
const HIST_BINS = 4096;
/** Histogram range for L = −log10(I). Sensor white is 0; ~6 decades below covers any negative. */
const L_MIN = -0.1;
const L_MAX = 6;

/** Normalized source-space rectangle [x0, x1) × [y0, y1). */
export interface Rect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export const FULL_RECT: Rect = { x0: 0, y0: 0, x1: 1, y1: 1 };

/**
 * Per-frame statistics, stored in base-independent L = −log10(I) space so the roll
 * base can change without re-reading pixels: density D = L + log10(base).
 */
export interface FrameStats {
  /** This frame's own film-base estimate (linear). */
  base: Vec3;
  /** True when the base came from the film rebate rather than the brightest-pixel fallback. */
  baseFromBorder: boolean;
  /** The detected image area inside rebate / holder, normalized source coords. */
  frame: Rect;
  /** Percentile L values per channel inside the statistics area. */
  lLo: Vec3;
  lHi: Vec3;
  lMed: Vec3;
  /** The same for the B&W channel mix. */
  bw: { lo: number; hi: number; med: number; mix: Vec3 };
  /** Fraction of statistics pixels that were clipped in any channel. */
  clipped: number;
  /** Settings the percentiles were measured with. */
  blackClip: number;
  whiteClip: number;
}

function percentileChannel(img: LinearImage, c: number, pct: number, mask?: (i: number) => boolean): number {
  const n = img.width * img.height;
  const vals: number[] = [];
  for (let i = 0; i < n; i++) {
    if (mask && !mask(i)) continue;
    vals.push(img.data[i * 3 + c]);
  }
  if (vals.length === 0) return NaN;
  vals.sort((a, b) => a - b);
  return vals[Math.min(vals.length - 1, Math.max(0, Math.floor((pct / 100) * (vals.length - 1))))];
}

function median(xs: number[]): number {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/**
 * Find the film base and the image area.
 *
 * The unexposed film (rebate, gaps between frames, sprocket strip) is the brightest
 * part of the negative that isn't bare light. Rows and columns at the edges that are
 * mostly near-base, clipped (sprocket holes / light gaps) or opaque (the holder) are border.
 */
export interface FrameDetection {
  base: Vec3;
  fromBorder: boolean;
  frame: Rect;
  /** Pixels connected to the image edge through holder or bare light (1 = exclude). */
  borderMask: Uint8Array | null;
}

export function detectBaseAndFrame(img: LinearImage): FrameDetection {
  const { width: w, height: h, data } = img;
  const unclipped = (i: number) =>
    data[i * 3] < CLIP_LEVEL && data[i * 3 + 1] < CLIP_LEVEL && data[i * 3 + 2] < CLIP_LEVEL;
  const b0: Vec3 = [0, 1, 2].map((c) => percentileChannel(img, c, 99.5, unclipped)) as Vec3;
  if (b0.some((v) => !isFinite(v) || v <= 0)) {
    return { base: [1, 1, 1], fromBorder: false, frame: { x0: 0.01, y0: 0.01, x1: 0.99, y1: 0.99 }, borderMask: null };
  }

  // The holder in a camera scan is never truly black (flare lifts it), so "opaque"
  // means denser than almost everything in the central part of the frame.
  const central: number[] = [];
  for (let y = Math.floor(h * 0.2); y < h * 0.8; y += 2) {
    for (let x = Math.floor(w * 0.2); x < w * 0.8; x += 2) central.push(data[(y * w + x) * 3 + 1]);
  }
  central.sort((a, b) => a - b);
  const denseCentral = central.length ? central[Math.floor(central.length * 0.005)] : 0;
  const opaqueThr = Math.max(0.004 * b0[1], 0.7 * denseCentral);

  // Per-pixel class bits: 1 = film rebate or bare light, 2 = opaque (holder).
  // Opaque is kept separate because very dense highlights can look similar; it may
  // only extend the border a short way in from the edge.
  const cls = new Uint8Array(w * h);
  const near = new Uint8Array(w * h);
  const clippedPx = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = data[i * 3];
    const g = data[i * 3 + 1];
    const b = data[i * 3 + 2];
    const clipped = r >= CLIP_LEVEL || g >= CLIP_LEVEL || b >= CLIP_LEVEL;
    const isNear = g >= 0.8 * b0[1] && r >= 0.75 * b0[0] && b >= 0.75 * b0[2] && !clipped;
    const opaque = g < opaqueThr;
    near[i] = isNear ? 1 : 0;
    clippedPx[i] = clipped ? 1 : 0;
    cls[i] = (clipped || isNear ? 1 : 0) | (opaque ? 2 : 0);
  }

  // Flood from the edges through holder (opaque) and bare light (clipped). This follows
  // tilted or irregular holders exactly, which a rectangular inset can't.
  const borderMask = new Uint8Array(w * h);
  const passable = (i: number) => (cls[i] & 2) !== 0 || clippedPx[i] === 1;
  const stack: number[] = [];
  const seed = (i: number) => {
    if (!borderMask[i] && passable(i)) {
      borderMask[i] = 1;
      stack.push(i);
    }
  };
  for (let x = 0; x < w; x++) {
    seed(x);
    seed((h - 1) * w + x);
  }
  for (let y = 0; y < h; y++) {
    seed(y * w);
    seed(y * w + w - 1);
  }
  while (stack.length) {
    const i = stack.pop()!;
    const x = i % w;
    if (x > 0) seed(i - 1);
    if (x < w - 1) seed(i + 1);
    if (i >= w) seed(i - w);
    if (i < w * (h - 1)) seed(i + w);
  }
  // Grow by one pixel to swallow the soft transition at the holder's edge.
  const grown = borderMask.slice();
  for (let i = 0; i < w * h; i++) {
    if (!borderMask[i]) continue;
    const x = i % w;
    if (x > 0) grown[i - 1] = 1;
    if (x < w - 1) grown[i + 1] = 1;
    if (i >= w) grown[i - w] = 1;
    if (i < w * (h - 1)) grown[i + w] = 1;
  }

  const lineFrac = (count: number, at: (k: number) => number) => (bits: number) => {
    let s = 0;
    for (let k = 0; k < count; k++) if (cls[at(k)] & bits) s++;
    return s / count;
  };
  const scan = (len: number, frac: (idx: number, bits: number) => number, fromEnd: boolean) => {
    const max = Math.floor(len * 0.25);
    const maxOpaque = Math.floor(len * 0.1);
    let gap = 0;
    let last = 0;
    for (let k = 0; k < max; k++) {
      const idx = fromEnd ? len - 1 - k : k;
      const bits = k < maxOpaque ? 3 : 1;
      if (frac(idx, bits) > 0.7) {
        last = k + 1;
        gap = 0;
      } else if (++gap > 2) break;
    }
    // In camera scans a strip of rebate usually sits between the holder and the
    // image; whatever lies outside the innermost such strip near the edge is border,
    // even when the holder itself didn't classify (flare-lifted, not opaque).
    const maxRebate = Math.floor(len * 0.12);
    for (let k = last; k < maxRebate; k++) {
      if (frac(fromEnd ? len - 1 - k : k, 1) > 0.6) last = k + 1;
    }
    return last;
  };
  const rowFrac = (y: number, bits: number) => lineFrac(w, (x) => y * w + x)(bits);
  const colFrac = (x: number, bits: number) => lineFrac(h, (y) => y * w + x)(bits);

  const top = scan(h, rowFrac, false);
  const bottom = scan(h, rowFrac, true);
  const left = scan(w, colFrac, false);
  const right = scan(w, colFrac, true);
  const margin = 0.01;
  const frame: Rect = {
    x0: Math.min(0.4, left / w + margin),
    y0: Math.min(0.4, top / h + margin),
    x1: Math.max(0.6, 1 - right / w - margin),
    y1: Math.max(0.6, 1 - bottom / h - margin),
  };

  // Refine the base from near-base pixels inside the detected border strips.
  const samples: [number[], number[], number[]] = [[], [], []];
  const inStrip = (x: number, y: number) => y < top || y >= h - bottom || x < left || x >= w - right;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!near[i] || !inStrip(x, y)) continue;
      for (let c = 0; c < 3; c++) samples[c].push(data[i * 3 + c]);
    }
  }
  if (samples[1].length >= 50) {
    return { base: samples.map(median) as Vec3, fromBorder: true, frame, borderMask: grown };
  }
  return { base: b0, fromBorder: false, frame, borderMask: grown };
}

/**
 * L above which pixels belong to a small, separate dense cluster (film holder, frame
 * edge) rather than the image: the highest cluster that sits above an empty gap of at
 * least 0.15 D and holds under 4% of pixels. Returns +∞ when there is none.
 */
export function denseOutlierCut(hist: Uint32Array, total: number): number {
  if (total < 1000) return Infinity;
  const binW = (L_MAX - L_MIN) / hist.length;
  const gapBins = Math.ceil(0.15 / binW);
  const empty = Math.max(1, total * 2e-5);
  let above = 0;
  let run = 0;
  for (let i = hist.length - 1; i >= 0; i--) {
    if (hist[i] <= empty) {
      run++;
      if (run >= gapBins && above > 0) {
        if (above > total * 0.04) return Infinity;
        // Everything above the gap is the outlier cluster.
        return L_MIN + (i + run) * binW;
      }
    } else {
      // Count the pixels of this bin (and any thin ones bridged by a short run) as "above".
      above += hist[i];
      run = 0;
    }
    if (above > total * 0.04) return Infinity;
  }
  return Infinity;
}

function percentileFromHist(hist: Uint32Array, total: number, pct: number): number {
  const target = (pct / 100) * total;
  let acc = 0;
  for (let i = 0; i < hist.length; i++) {
    acc += hist[i];
    if (acc >= target) {
      // Interpolate within the bin for stable results across tiny edits.
      const before = acc - hist[i];
      const t = hist[i] ? (target - before) / hist[i] : 0.5;
      return L_MIN + ((i + t) / hist.length) * (L_MAX - L_MIN);
    }
  }
  return L_MAX;
}

/**
 * Measure density statistics inside `area` (normalized source rect), skipping clipped
 * pixels and anything brighter than the film base (sprocket holes, light leaks).
 */
export function measureStats(
  img: LinearImage,
  base: Vec3,
  area: Rect,
  opts: { blackClip: number; whiteClip: number; bwMix: Vec3 },
  exclude?: Uint8Array | null,
): Omit<FrameStats, 'base' | 'baseFromBorder' | 'frame'> {
  const { width: w, height: h, data } = img;
  const hists = [0, 1, 2, 3].map(() => new Uint32Array(HIST_BINS));
  const x0 = Math.max(0, Math.floor(area.x0 * w));
  const x1 = Math.min(w, Math.ceil(area.x1 * w));
  const y0 = Math.max(0, Math.floor(area.y0 * h));
  const y1 = Math.min(h, Math.ceil(area.y1 * h));
  const lBase = base.map((b) => -Math.log10(Math.max(b, EPS)));
  const binScale = HIST_BINS / (L_MAX - L_MIN);
  const mixSum = opts.bwMix[0] + opts.bwMix[1] + opts.bwMix[2] || 1;
  const mix = opts.bwMix.map((m) => m / mixSum);
  let total = 0;
  let clipped = 0;
  let considered = 0;
  // Pass 1: L values of usable pixels, and a green histogram to find stray dense clusters.
  const kept = new Float32Array((x1 - x0) * (y1 - y0) * 3);
  const gHist = new Uint32Array(HIST_BINS);
  const binOf = (l: number) => Math.min(HIST_BINS - 1, Math.max(0, Math.floor((l - L_MIN) * binScale)));
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      if (exclude && exclude[y * w + x]) continue;
      const i = (y * w + x) * 3;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      considered++;
      if (r >= CLIP_LEVEL || g >= CLIP_LEVEL || b >= CLIP_LEVEL) {
        clipped++;
        continue;
      }
      const lg = -Math.log10(Math.max(g, EPS));
      // Brighter than base by more than 0.02 D in green: not film image.
      if (lg < lBase[1] - 0.02) continue;
      kept[total * 3] = -Math.log10(Math.max(r, EPS));
      kept[total * 3 + 1] = lg;
      kept[total * 3 + 2] = -Math.log10(Math.max(b, EPS));
      gHist[binOf(lg)]++;
      total++;
    }
  }

  // Pass 2: histograms without the dense outlier cluster (holder edges that leak into a
  // tilted frame would otherwise become the "highlight" and darken the whole print).
  const cut = denseOutlierCut(gHist, total);
  let used = 0;
  for (let k = 0; k < total; k++) {
    const lr = kept[k * 3];
    const lg = kept[k * 3 + 1];
    const lb = kept[k * 3 + 2];
    if (lg > cut) continue;
    const lw = mix[0] * lr + mix[1] * lg + mix[2] * lb;
    hists[0][binOf(lr)]++;
    hists[1][binOf(lg)]++;
    hists[2][binOf(lb)]++;
    hists[3][binOf(lw)]++;
    // Compact in place so the midtone pass sees the same pixels.
    kept[used * 3] = lr;
    kept[used * 3 + 1] = lg;
    kept[used * 3 + 2] = lb;
    used++;
  }
  total = used;
  const lo = (c: number) => (total ? percentileFromHist(hists[c], total, opts.blackClip) : lBase[Math.min(c, 2)]);
  const hi = (c: number) => (total ? percentileFromHist(hists[c], total, 100 - opts.whiteClip) : lBase[Math.min(c, 2)] + 1.5);
  const md = (c: number) => (total ? percentileFromHist(hists[c], total, 50) : lBase[Math.min(c, 2)] + 0.75);
  const lLo: Vec3 = [lo(0), lo(1), lo(2)];
  const lHi: Vec3 = [hi(0), hi(1), hi(2)];
  return {
    lLo,
    lHi,
    lMed: neutralMedians(kept, total, lLo, lHi) ?? [md(0), md(1), md(2)],
    bw: { lo: lo(3), hi: hi(3), med: md(3), mix: opts.bwMix },
    clipped: considered ? clipped / considered : 0,
    blackClip: opts.blackClip,
    whiteClip: opts.whiteClip,
  };
}

/**
 * Per-channel median L over near-neutral midtone pixels: after endpoint
 * normalization, what's left between channels in the greys is crossover the
 * endpoints didn't fix. Using only near-greys keeps colourful subjects from
 * being neutralized (plain gray-world). Two passes re-centre the window on the
 * cast found in the first. Returns null when too few greys exist.
 */
function neutralMedians(kept: Float32Array, total: number, lLo: Vec3, lHi: Vec3): Vec3 | null {
  if (total < 100) return null;
  const inv = lLo.map((l, c) => 1 / Math.max(0.05, lHi[c] - l));
  const nOf = (i: number, c: number) => (kept[i * 3 + c] - lLo[c]) * inv[c];
  let cast = [0, 0, 0];
  let result: Vec3 | null = null;
  for (const win of [0.12, 0.07]) {
    const sel: [number[], number[], number[]] = [[], [], []];
    for (let i = 0; i < total; i++) {
      const ng = nOf(i, 1);
      if (ng < 0.15 || ng > 0.85) continue;
      const dr = nOf(i, 0) - ng - cast[0];
      const db = nOf(i, 2) - ng - cast[2];
      if (Math.abs(dr) > win || Math.abs(db) > win) continue;
      for (let c = 0; c < 3; c++) sel[c].push(kept[i * 3 + c]);
    }
    if (sel[1].length < Math.max(50, total * 0.01)) return result;
    const med = sel.map(median) as Vec3;
    const mn = med.map((m, c) => (m - lLo[c]) * inv[c]);
    cast = [mn[0] - mn[1], 0, mn[2] - mn[1]];
    result = med;
  }
  return result;
}

export function intersectRect(a: Rect, b: Rect): Rect {
  const r = { x0: Math.max(a.x0, b.x0), y0: Math.max(a.y0, b.y0), x1: Math.min(a.x1, b.x1), y1: Math.min(a.y1, b.y1) };
  if (r.x1 - r.x0 < 0.05 || r.y1 - r.y0 < 0.05) return a;
  return r;
}

export interface AnalyzeOptions {
  /** Statistics area from the user's crop (normalized source coords), if any. */
  cropArea: Rect;
  autoFrame: boolean;
  /** Use this base instead of detecting one (manual base). */
  baseOverride: Vec3 | null;
  blackClip: number;
  whiteClip: number;
  bwMix: Vec3;
}

/** Full per-frame analysis on the small analysis image. */
export function analyzeFrame(img: LinearImage, o: AnalyzeOptions): FrameStats {
  const det = detectBaseAndFrame(img);
  const base = o.baseOverride ?? det.base;
  const area = o.autoFrame ? intersectRect(o.cropArea, det.frame) : o.cropArea;
  const s = measureStats(img, base, area, o, o.autoFrame ? det.borderMask : null);
  return { base: det.base, baseFromBorder: det.fromBorder, frame: det.frame, ...s };
}
