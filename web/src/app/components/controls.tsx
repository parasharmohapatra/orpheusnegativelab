import { useRef, useState, type ReactNode } from 'react';
import { useStore } from '../store/rollStore';

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  /** Value restored on double-click. */
  reset?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  /** CSS gradient for hue/temperature-style tracks. */
  track?: string;
}

const signed = (v: number, scale = 100) => {
  const n = Math.round(v * scale);
  return n > 0 ? `+${n}` : `${n}`;
};
export const fmtPercent = (v: number) => signed(v);
export const fmtStops = (v: number) => `${v > 0 ? '+' : ''}${v.toFixed(2)} EV`;

/**
 * Range slider wired to the undo stack: dragging edits live without history,
 * releasing commits one undo step. Double-click resets; Alt drags finely;
 * dragging the value label scrubs.
 */
export function Slider({ label, value, min, max, step = 0.01, reset = 0, onChange, format = fmtPercent, track }: SliderProps) {
  const commit = useStore((s) => s.commit);
  const scrub = useRef<{ x: number; v: number } | null>(null);
  const [fine, setFine] = useState(false);
  const span = max - min;
  const pct = ((value - min) / span) * 100;
  const zero = ((Math.min(max, Math.max(min, reset)) - min) / span) * 100;
  const clamp = (v: number) => Math.min(max, Math.max(min, v));

  return (
    <div className={`slider${track ? ' hue' : ''}`}>
      <div className="label-row">
        <span>{label}</span>
        <button
          type="button"
          className="value"
          title="Drag to adjust, double-click to reset"
          onDoubleClick={() => {
            onChange(reset);
            commit();
          }}
          onPointerDown={(e) => {
            (e.target as HTMLElement).setPointerCapture(e.pointerId);
            scrub.current = { x: e.clientX, v: value };
          }}
          onPointerMove={(e) => {
            if (!scrub.current) return;
            const dx = e.clientX - scrub.current.x;
            onChange(clamp(scrub.current.v + (dx / 200) * span * (e.altKey ? 0.1 : 1)));
          }}
          onPointerUp={() => {
            scrub.current = null;
            commit();
          }}
        >
          {format(value)}
        </button>
      </div>
      <input
        type="range"
        aria-label={label}
        className={value < reset ? 'neg' : ''}
        min={min}
        max={max}
        step={fine ? step / 10 : step}
        value={value}
        style={{ ['--fill' as string]: `${pct}%`, ['--zero' as string]: `${zero}%`, ['--hue-track' as string]: track }}
        onChange={(e) => onChange(Number(e.target.value))}
        onPointerDown={(e) => setFine(e.altKey)}
        onPointerUp={() => commit()}
        onKeyUp={() => commit()}
        onDoubleClick={() => {
          onChange(reset);
          commit();
        }}
      />
    </div>
  );
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  label,
}: {
  options: { value: T; label: string; title?: string }[];
  value: T;
  onChange: (v: T) => void;
  label?: string;
}) {
  return (
    <div className="field">
      {label && <span>{label}</span>}
      <div className="seg" role="radiogroup" aria-label={label}>
        {options.map((o) => (
          <button key={o.value} type="button" role="radio" aria-checked={value === o.value} title={o.title} className={value === o.value ? 'on' : ''} onClick={() => onChange(o.value)}>
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Section({ title, children, defaultOpen = true, action }: { title: string; children: ReactNode; defaultOpen?: boolean; action?: ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className={`section${open ? '' : ' closed'}`}>
      <div className="label-row">
        <button type="button" className="section-head" onClick={() => setOpen(!open)} aria-expanded={open}>
          <span>{title}</span>
        </button>
        <span style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          {action}
          <svg className="chev" width="12" height="12" viewBox="0 0 12 12" onClick={() => setOpen(!open)} aria-hidden="true">
            <path d="M3 4.5l3 3 3-3" fill="none" stroke="currentColor" strokeWidth="1.3" />
          </svg>
        </span>
      </div>
      {open && <div className="section-body">{children}</div>}
    </section>
  );
}

export function Check({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="check">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  );
}
