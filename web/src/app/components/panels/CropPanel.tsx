import { useState } from 'react';
import { inscribedScale, orientedSize, withOrientation } from '../../../engine/geometry/transform';
import { autoFrameCrop } from '../../store/frames';
import { useCurrentFrame, useStore } from '../../store/rollStore';
import { Check, Section, Slider } from '../controls';
import { IconFlip, IconRotate } from '../icons';
import { cropAspect } from '../cropAspect';

const ASPECTS: { label: string; value: number | null }[] = [
  { label: 'Free', value: null },
  { label: '3:2', value: 3 / 2 },
  { label: '4:3', value: 4 / 3 },
  { label: '5:4', value: 5 / 4 },
  { label: '1:1', value: 1 },
  { label: '6:7', value: 7 / 6 },
  { label: '16:9', value: 16 / 9 },
];

export function CropPanel() {
  const frame = useCurrentFrame();
  const edit = useStore((s) => s.edit);
  const [aspect, setAspectState] = useState<number | null>(cropAspect.get());
  if (!frame?.meta || !frame.analysis) return null;
  const g = frame.recipe.geometry;
  const [ow, oh] = orientedSize(frame.analysis.width, frame.analysis.height, withOrientation(g, frame.meta.orientation).rot90);

  const applyAspect = (ratio: number | null, portrait = oh > ow) => {
    cropAspect.set(ratio);
    setAspectState(ratio);
    if (!ratio) return;
    const r = portrait && ratio !== 1 ? 1 / ratio : ratio;
    edit((d) => {
      const c = d.geometry.crop;
      const cx = c.x + c.w / 2;
      const cy = c.y + c.h / 2;
      // Largest rect of this pixel ratio inside the current crop.
      let w = c.w;
      let h = (w * ow) / (r * oh);
      if (h > c.h) {
        h = c.h;
        w = (h * r * oh) / ow;
      }
      d.geometry.crop = { x: Math.max(0, cx - w / 2), y: Math.max(0, cy - h / 2), w, h };
    }, true, true);
  };

  const straighten = (deg: number, commit: boolean) =>
    edit((d) => {
      const prev = inscribedScale(ow, oh, d.geometry.straighten);
      d.geometry.straighten = deg;
      const s = inscribedScale(ow, oh, deg);
      const c = d.geometry.crop;
      // Keep the crop inside the rotated image, scaling it about its centre.
      const k = s / prev;
      const w = Math.min(s, c.w * k);
      const h = Math.min(s, c.h * k);
      const cx = c.x + c.w / 2;
      const cy = c.y + c.h / 2;
      const lo = (1 - s) / 2;
      d.geometry.crop = {
        w,
        h,
        x: Math.min(lo + s - w, Math.max(lo, cx - w / 2)),
        y: Math.min(lo + s - h, Math.max(lo, cy - h / 2)),
      };
    }, commit, true);

  return (
    <>
      <Section title="Orientation">
        <div className="row">
          <button type="button" className="btn small" onClick={() => edit((d) => void (d.geometry.rot90 = ((d.geometry.rot90 + 3) % 4) as 0 | 1 | 2 | 3))}>
            <span style={{ transform: 'scaleX(-1)', display: 'inline-flex' }}>
              <IconRotate />
            </span>{' '}
            Left
          </button>
          <button type="button" className="btn small" onClick={() => edit((d) => void (d.geometry.rot90 = ((d.geometry.rot90 + 1) % 4) as 0 | 1 | 2 | 3))}>
            <IconRotate /> Right
          </button>
          <button type="button" className="btn small" onClick={() => edit((d) => void (d.geometry.flipH = !d.geometry.flipH))}>
            <IconFlip /> Flip
          </button>
        </div>
        <Slider label="Straighten" value={g.straighten} min={-10} max={10} step={0.05} format={(v) => `${v > 0 ? '+' : ''}${v.toFixed(2)}°`} onChange={(v) => straighten(v, false)} />
      </Section>
      <Section title="Crop">
        <div className="chips">
          {ASPECTS.map((a) => (
            <button key={a.label} type="button" className={`btn small${aspect === a.value ? ' active' : ''}`} onClick={() => applyAspect(a.value)}>
              {a.label}
            </button>
          ))}
        </div>
        <div className="row">
          <button
            type="button"
            className="btn small"
            onClick={() => {
              const crop = autoFrameCrop(frame);
              if (!crop) return;
              edit((d) => {
                d.geometry.straighten = 0;
                d.geometry.crop = crop;
              }, true, true);
            }}
          >
            Auto frame
          </button>
          <button
            type="button"
            className="btn small"
            onClick={() =>
              edit((d) => {
                d.geometry.crop = { x: 0, y: 0, w: 1, h: 1 };
                d.geometry.straighten = 0;
              }, true, true)
            }
          >
            Reset crop
          </button>
        </div>
        <Check label="Ignore film border when measuring" checked={g.autoFrame} onChange={(v) => edit((d) => void (d.geometry.autoFrame = v))} />
        <p className="hint">Drag the corners to crop. Cropping only frames the picture; colour is always measured on the whole frame, with the film holder and rebate left out automatically.</p>
      </Section>
    </>
  );
}
