import { useEffect, useRef, useState } from 'react';
import { curveToLut } from '../../../engine/pipeline/curves';
import type { CurvePoint } from '../../../engine/pipeline/recipe';
import { useCurrentFrame, useStore } from '../../store/rollStore';
import { Section, Segmented } from '../controls';

type Channel = 'rgb' | 'r' | 'g' | 'b';
const STROKE: Record<Channel, string> = { rgb: '#f3ece4', r: '#e36b6b', g: '#7cc483', b: '#6f9be6' };
const HIT = 0.04;

export function CurvesPanel() {
  const frame = useCurrentFrame();
  const edit = useStore((s) => s.edit);
  const commit = useStore((s) => s.commit);
  const [ch, setCh] = useState<Channel>('rgb');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragIdx = useRef<number | null>(null);
  const points = frame?.recipe.curves[ch] ?? [];

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !frame) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = (c.width = Math.round(c.clientWidth * dpr));
    const h = (c.height = Math.round(c.clientHeight * dpr));
    const ctx = c.getContext('2d')!;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(205, 189, 184, 0.12)';
    ctx.lineWidth = 1;
    for (const f of [0.25, 0.5, 0.75]) {
      ctx.beginPath();
      ctx.moveTo(f * w, 0);
      ctx.lineTo(f * w, h);
      ctx.moveTo(0, f * h);
      ctx.lineTo(w, f * h);
      ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(205, 189, 184, 0.2)';
    ctx.beginPath();
    ctx.moveTo(0, h);
    ctx.lineTo(w, 0);
    ctx.stroke();
    // Faint other channels for context.
    (['rgb', 'r', 'g', 'b'] as Channel[]).forEach((k) => {
      if (k === ch) return;
      const pts = frame.recipe.curves[k];
      if (pts.every((p) => p.x === p.y)) return;
      drawCurve(ctx, pts, w, h, STROKE[k], 0.3, dpr);
    });
    drawCurve(ctx, points, w, h, STROKE[ch], 1, dpr);
    for (const p of points) {
      ctx.fillStyle = '#170a0e';
      ctx.strokeStyle = '#d8b878';
      ctx.lineWidth = 1.5 * dpr;
      ctx.beginPath();
      ctx.arc(p.x * w, (1 - p.y) * h, 4.5 * dpr, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }, [frame, ch, points]);

  if (!frame) return null;

  const pos = (e: React.PointerEvent) => {
    const b = canvasRef.current!.getBoundingClientRect();
    return { x: Math.min(1, Math.max(0, (e.clientX - b.left) / b.width)), y: Math.min(1, Math.max(0, 1 - (e.clientY - b.top) / b.height)) };
  };
  const setPoints = (pts: CurvePoint[], doCommit: boolean) => edit((d) => void (d.curves[ch] = pts), doCommit);

  const onDown = (e: React.PointerEvent) => {
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    const p = pos(e);
    let i = points.findIndex((q) => Math.hypot(q.x - p.x, q.y - p.y) < HIT);
    if (i < 0) {
      const next = [...points, p].sort((a, b) => a.x - b.x);
      i = next.indexOf(p);
      setPoints(next, false);
    }
    dragIdx.current = i;
  };
  const onMove = (e: React.PointerEvent) => {
    const i = dragIdx.current;
    if (i === null) return;
    const p = pos(e);
    const next = points.map((q) => ({ ...q }));
    const lo = i > 0 ? next[i - 1].x + 0.01 : 0;
    const hi = i < next.length - 1 ? next[i + 1].x - 0.01 : 1;
    next[i] = { x: i === 0 ? Math.min(p.x, hi) : i === next.length - 1 ? Math.max(p.x, lo) : Math.min(hi, Math.max(lo, p.x)), y: p.y };
    setPoints(next, false);
  };
  const onUp = () => {
    dragIdx.current = null;
    commit();
  };
  const onDouble = (e: React.MouseEvent) => {
    const b = canvasRef.current!.getBoundingClientRect();
    const p = { x: (e.clientX - b.left) / b.width, y: 1 - (e.clientY - b.top) / b.height };
    const i = points.findIndex((q) => Math.hypot(q.x - p.x, q.y - p.y) < HIT);
    if (i > 0 && i < points.length - 1) setPoints(points.filter((_, k) => k !== i), true);
  };

  return (
    <Section
      title="Tone curve"
      action={
        <button
          type="button"
          className="btn small ghost"
          onClick={() =>
            setPoints(
              [
                { x: 0, y: 0 },
                { x: 1, y: 1 },
              ],
              true,
            )
          }
        >
          Reset
        </button>
      }
    >
      <Segmented<Channel>
        options={[
          { value: 'rgb', label: 'RGB' },
          { value: 'r', label: 'Red' },
          { value: 'g', label: 'Green' },
          { value: 'b', label: 'Blue' },
        ]}
        value={ch}
        onChange={setCh}
      />
      <div className="curve-wrap">
        <canvas ref={canvasRef} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onDoubleClick={onDouble} aria-label={`${ch.toUpperCase()} tone curve editor`} />
      </div>
      <p className="hint">Click to add a point, drag to shape, double-click a point to remove it. Curves act on the final print.</p>
    </Section>
  );
}

function drawCurve(ctx: CanvasRenderingContext2D, pts: CurvePoint[], w: number, h: number, color: string, alpha: number, dpr: number) {
  const lut = curveToLut(pts, 256);
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5 * dpr;
  ctx.beginPath();
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * w;
    const y = (1 - lut[i]) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
}
