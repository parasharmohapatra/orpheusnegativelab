import { useRef } from 'react';
import { clampCrop } from '../../engine/geometry/transform';
import type { CropRect } from '../../engine/pipeline/recipe';
import { useCurrentFrame, useStore } from '../store/rollStore';
import { cropAspect } from './cropAspect';
import type { Rect } from './Viewer';

type Handle = 'move' | 'nw' | 'ne' | 'sw' | 'se';

/** Crop box over the full (uncropped) image; `rect` is where that image sits on screen. */
export function CropOverlay({ rect }: { rect: Rect }) {
  const frame = useCurrentFrame()!;
  const edit = useStore((s) => s.edit);
  const commit = useStore((s) => s.commit);
  const crop = frame.recipe.geometry.crop;
  const drag = useRef<{ h: Handle; x: number; y: number; start: CropRect } | null>(null);

  const begin = (h: Handle) => (e: React.PointerEvent) => {
    e.stopPropagation();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { h, x: e.clientX, y: e.clientY, start: { ...crop } };
  };

  const move = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const dx = (e.clientX - d.x) / rect.w;
    const dy = (e.clientY - d.y) / rect.h;
    const s = d.start;
    let next: CropRect;
    if (d.h === 'move') {
      next = { ...s, x: s.x + dx, y: s.y + dy };
    } else {
      let x0 = s.x, y0 = s.y, x1 = s.x + s.w, y1 = s.y + s.h;
      if (d.h.includes('w')) x0 += dx;
      if (d.h.includes('e')) x1 += dx;
      if (d.h.includes('n')) y0 += dy;
      if (d.h.includes('s')) y1 += dy;
      x0 = Math.max(0, Math.min(x0, x1 - 0.02));
      y0 = Math.max(0, Math.min(y0, y1 - 0.02));
      x1 = Math.min(1, Math.max(x1, x0 + 0.02));
      y1 = Math.min(1, Math.max(y1, y0 + 0.02));
      next = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
      const ratio = cropAspect.get();
      if (ratio) {
        // Lock aspect in pixels: adjust height to match width, anchored at the dragged corner's opposite.
        const pxRatio = rect.w / rect.h;
        const h = Math.min(1, (next.w * pxRatio) / ratio);
        if (d.h.includes('n')) next.y = next.y + next.h - h;
        next.h = h;
      }
    }
    edit((r) => {
      r.geometry.crop = clampCrop(next);
    }, false, true);
  };

  const end = () => {
    drag.current = null;
    commit();
  };

  const style = { left: rect.x + crop.x * rect.w, top: rect.y + crop.y * rect.h, width: crop.w * rect.w, height: crop.h * rect.h };
  return (
    <div className="crop-box" style={style} onPointerDown={begin('move')} onPointerMove={move} onPointerUp={end}>
      <div className="grid" />
      {(['nw', 'ne', 'sw', 'se'] as const).map((h) => (
        <div key={h} className={`crop-handle ${h}`} onPointerDown={begin(h)} onPointerMove={move} onPointerUp={end} />
      ))}
    </div>
  );
}
