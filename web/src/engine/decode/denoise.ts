/**
 * Chroma noise reduction for linear RGB, in place.
 *
 * Camera scans of colour negatives have very little blue (the orange mask absorbs it),
 * so the blue channel is mostly sensor noise. Inverting stretches that into coloured
 * speckle. Like a raw converter's default colour NR, this keeps luminance untouched and
 * smooths only the colour ratios: a 3×3 median first (kills hot/dead pixels), then a
 * separable box blur run twice (≈ Gaussian). Colour is low-frequency, so large radii
 * are processed at half resolution and upsampled.
 */
export function denoiseChroma(data: Float32Array, w: number, h: number, radius: number): void {
  if (radius < 0.5 || w < 3 || h < 3) return;
  const n = w * h;
  const EPS = 1e-7;
  // Colour as ratios to Y = (R + 2G + B) / 4; green follows from the other two.
  const Y = new Float32Array(n);
  const cr = new Float32Array(n);
  const cb = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const y = (data[i * 3] + 2 * data[i * 3 + 1] + data[i * 3 + 2]) * 0.25 + EPS;
    Y[i] = y;
    cr[i] = data[i * 3] / y;
    cb[i] = data[i * 3 + 2] / y;
  }

  const half = radius >= 4 && w >= 16 && h >= 16;
  if (half) {
    // Median at full resolution (hot pixels are single pixels), blur at half.
    const tmp = new Float32Array(n);
    median3(cr, tmp, w, h);
    median3(cb, tmp, w, h);
    const hw = w >> 1;
    const hh = h >> 1;
    const smallR = down2(cr, w, hw, hh);
    const smallB = down2(cb, w, hw, hh);
    const t2 = new Float32Array(hw * hh);
    const r = Math.max(1, Math.round(radius / 2));
    for (const ch of [smallR, smallB]) {
      boxBlur(ch, t2, hw, hh, r);
      boxBlur(ch, t2, hw, hh, r);
    }
    up2(smallR, hw, hh, cr, w, h);
    up2(smallB, hw, hh, cb, w, h);
  } else {
    const tmp = new Float32Array(n);
    const r = Math.max(1, Math.round(radius));
    for (const ch of [cr, cb]) {
      median3(ch, tmp, w, h);
      boxBlur(ch, tmp, w, h, r);
      boxBlur(ch, tmp, w, h, r);
    }
  }

  for (let i = 0; i < n; i++) {
    const y = Y[i];
    const rr = Math.max(0, cr[i]);
    const bb = Math.max(0, cb[i]);
    const gg = Math.max(0, (4 - rr - bb) * 0.5);
    data[i * 3] = rr * y;
    data[i * 3 + 1] = gg * y;
    data[i * 3 + 2] = bb * y;
  }
}

function down2(a: Float32Array, w: number, hw: number, hh: number): Float32Array {
  const out = new Float32Array(hw * hh);
  for (let y = 0; y < hh; y++) {
    const r0 = 2 * y * w;
    const r1 = r0 + w;
    for (let x = 0; x < hw; x++) {
      const c = 2 * x;
      out[y * hw + x] = (a[r0 + c] + a[r0 + c + 1] + a[r1 + c] + a[r1 + c + 1]) * 0.25;
    }
  }
  return out;
}

/** Bilinear upsample (pixel centres aligned) into `out`. */
function up2(a: Float32Array, hw: number, hh: number, out: Float32Array, w: number, h: number): void {
  for (let y = 0; y < h; y++) {
    const fy = Math.min(hh - 1, Math.max(0, (y + 0.5) / 2 - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(hh - 1, y0 + 1);
    const ty = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(hw - 1, Math.max(0, (x + 0.5) / 2 - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(hw - 1, x0 + 1);
      const tx = fx - x0;
      const top = a[y0 * hw + x0] * (1 - tx) + a[y0 * hw + x1] * tx;
      const bot = a[y1 * hw + x0] * (1 - tx) + a[y1 * hw + x1] * tx;
      out[y * w + x] = top * (1 - ty) + bot * ty;
    }
  }
}

/** 3×3 median via a 19-exchange sorting network, in place (edges copied). */
export function median3(a: Float32Array, tmp: Float32Array, w: number, h: number): void {
  tmp.set(a);
  for (let y = 1; y < h - 1; y++) {
    const o0 = (y - 1) * w;
    const o1 = y * w;
    const o2 = (y + 1) * w;
    for (let x = 1; x < w - 1; x++) {
      let p0 = tmp[o0 + x - 1], p1 = tmp[o0 + x], p2 = tmp[o0 + x + 1];
      let p3 = tmp[o1 + x - 1], p4 = tmp[o1 + x], p5 = tmp[o1 + x + 1];
      let p6 = tmp[o2 + x - 1], p7 = tmp[o2 + x], p8 = tmp[o2 + x + 1];
      let t: number;
      // Paeth's median-of-9 network.
      if (p1 > p2) { t = p1; p1 = p2; p2 = t; }
      if (p4 > p5) { t = p4; p4 = p5; p5 = t; }
      if (p7 > p8) { t = p7; p7 = p8; p8 = t; }
      if (p0 > p1) { t = p0; p0 = p1; p1 = t; }
      if (p3 > p4) { t = p3; p3 = p4; p4 = t; }
      if (p6 > p7) { t = p6; p6 = p7; p7 = t; }
      if (p1 > p2) { t = p1; p1 = p2; p2 = t; }
      if (p4 > p5) { t = p4; p4 = p5; p5 = t; }
      if (p7 > p8) { t = p7; p7 = p8; p8 = t; }
      if (p0 > p3) { p3 = p0; }
      if (p5 > p8) { p5 = p8; }
      if (p4 > p7) { t = p4; p4 = p7; p7 = t; }
      if (p3 > p6) { p6 = p3; }
      if (p1 > p4) { p4 = p1; }
      if (p2 > p5) { p2 = p5; }
      if (p4 > p7) { p4 = p7; }
      if (p4 > p2) { t = p4; p4 = p2; p2 = t; }
      if (p6 > p4) { p4 = p6; }
      if (p4 > p2) { p4 = p2; }
      a[o1 + x] = p4;
    }
  }
}

/** Separable box blur of radius r with clamped edges, in place. */
function boxBlur(a: Float32Array, tmp: Float32Array, w: number, h: number, r: number): void {
  const win = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    const o = y * w;
    let acc = 0;
    for (let k = -r; k <= r; k++) acc += a[o + Math.min(w - 1, Math.max(0, k))];
    for (let x = 0; x < w; x++) {
      tmp[o + x] = acc / win;
      acc += a[o + Math.min(w - 1, x + r + 1)] - a[o + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let k = -r; k <= r; k++) acc += tmp[Math.min(h - 1, Math.max(0, k)) * w + x];
    for (let y = 0; y < h; y++) {
      a[y * w + x] = acc / win;
      acc += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x];
    }
  }
}
