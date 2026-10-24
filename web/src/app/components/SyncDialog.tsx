import { useState } from 'react';
import { RECIPE_SECTIONS, type RecipeSection } from '../../engine/pipeline/recipe';
import { useStore } from '../store/rollStore';
import { Check } from './controls';

export function SyncDialog() {
  const open = useStore((s) => s.syncOpen);
  const selected = useStore((s) => s.selected);
  const currentId = useStore((s) => s.currentId);
  const set = useStore((s) => s.set);
  const sync = useStore((s) => s.syncFromCurrent);
  const notify = useStore((s) => s.notify);
  // Geometry is per-frame by nature, so it's off by default.
  const [sections, setSections] = useState<RecipeSection[]>(['film', 'convert', 'tone', 'color', 'curves']);
  if (!open) return null;
  const targets = selected.filter((id) => id !== currentId);
  return (
    <div className="scrim" onPointerDown={(e) => e.target === e.currentTarget && set('syncOpen', false)}>
      <div className="dialog" role="dialog" aria-modal="true" aria-label="Sync settings">
        <h3>Sync settings</h3>
        <p className="hint">
          Copy this frame's settings to the other {targets.length} selected frame{targets.length === 1 ? '' : 's'}. Density is still measured per frame unless
          the conversion is set to <em>Roll</em>.
        </p>
        {RECIPE_SECTIONS.map((s) => (
          <Check key={s.key} label={s.label} checked={sections.includes(s.key)} onChange={(v) => setSections(v ? [...sections, s.key] : sections.filter((x) => x !== s.key))} />
        ))}
        <div className="dialog-actions">
          <button type="button" className="btn" onClick={() => set('syncOpen', false)}>
            Cancel
          </button>
          <button
            type="button"
            className="btn primary"
            disabled={!sections.length || !targets.length}
            onClick={() => {
              sync(targets, sections);
              set('syncOpen', false);
              notify(`Synced to ${targets.length} frame${targets.length === 1 ? '' : 's'}.`);
            }}
          >
            Sync
          </button>
        </div>
      </div>
    </div>
  );
}
