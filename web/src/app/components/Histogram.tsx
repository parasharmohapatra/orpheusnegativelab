import { useEffect, useRef, useState } from 'react';
import { subscribeHistogram, type HistogramData } from './histogramBus';

export function Histogram() {
  const ref = useRef<HTMLCanvasElement>(null);
  const [clip, setClip] = useState<[number, number]>([0, 0]);

  useEffect(
    () =>
      subscribeHistogram((h) => {
        const c = ref.current;
        if (!c) return;
        const dpr = Math.min(2, window.devicePixelRatio || 1);
        const w = Math.round(c.clientWidth * dpr);
        const hgt = Math.round(c.clientHeight * dpr);
        c.width = w;
        c.height = hgt;
        const ctx = c.getContext('2d')!;
        ctx.clearRect(0, 0, w, hgt);
        if (!h) return;
        draw(ctx, h, w, hgt);
        setClip([h.clipLo, h.clipHi]);
      }),
    [],
  );

  const pct = (v: number) => (v < 0.001 ? '0%' : `${(v * 100).toFixed(v < 0.01 ? 1 : 0)}%`);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <canvas ref={ref} className="histogram" aria-label="RGB histogram of the converted image" />
      <div className="hist-meta">
        <span title="Pixels at pure black">Shadows clip {pct(clip[0])}</span>
        <span title="Pixels with any channel at full">Highlights clip {pct(clip[1])}</span>
      </div>
    </div>
  );
}

function draw(ctx: CanvasRenderingContext2D, h: HistogramData, w: number, hgt: number) {
  // Ignore the end bins when scaling so clipping spikes don't flatten the rest.
  let max = 1;
  for (const ch of [h.r, h.g, h.b]) for (let i = 2; i < 254; i++) max = Math.max(max, ch[i]);
  const smooth = (ch: Uint32Array, i: number) => (ch[Math.max(0, i - 1)] + 2 * ch[i] + ch[Math.min(255, i + 1)]) / 4;
  ctx.globalCompositeOperation = 'lighter';
  const colors = ['rgba(225, 92, 92, 0.55)', 'rgba(110, 190, 120, 0.5)', 'rgba(95, 140, 230, 0.55)'];
  [h.r, h.g, h.b].forEach((ch, k) => {
    ctx.beginPath();
    ctx.moveTo(0, hgt);
    for (let i = 0; i < 256; i++) {
      const x = (i / 255) * w;
      const y = hgt - Math.min(1, Math.sqrt(smooth(ch, i) / max)) * (hgt - 4);
      ctx.lineTo(x, y);
    }
    ctx.lineTo(w, hgt);
    ctx.closePath();
    ctx.fillStyle = colors[k];
    ctx.fill();
  });
  ctx.globalCompositeOperation = 'source-over';
  ctx.strokeStyle = 'rgba(216, 184, 120, 0.25)';
  ctx.lineWidth = 1;
  for (const f of [0.25, 0.5, 0.75]) {
    ctx.beginPath();
    ctx.moveTo(Math.round(f * w) + 0.5, 0);
    ctx.lineTo(Math.round(f * w) + 0.5, hgt);
    ctx.stroke();
  }
}
