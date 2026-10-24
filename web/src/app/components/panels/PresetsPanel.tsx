import { useEffect, useState } from 'react';
import { pickSections, RECIPE_SECTIONS, type RecipeSection } from '../../../engine/pipeline/recipe';
import { deletePreset, listPresets, savePreset, type StoredPreset } from '../../persistence/db';
import { BUILTIN_PRESETS } from '../../store/presets';
import { useCurrentFrame, useStore } from '../../store/rollStore';
import { Check, Section } from '../controls';

const DEFAULT_SECTIONS: RecipeSection[] = ['tone', 'color', 'curves'];

export function PresetsPanel() {
  const frame = useCurrentFrame();
  const selected = useStore((s) => s.selected);
  const applyToFrames = useStore((s) => s.applyToFrames);
  const notify = useStore((s) => s.notify);
  const [mine, setMine] = useState<StoredPreset[]>([]);
  const [name, setName] = useState('');
  const [sections, setSections] = useState<RecipeSection[]>(DEFAULT_SECTIONS);

  useEffect(() => {
    void listPresets().then(setMine);
  }, []);
  if (!frame) return null;

  const targets = selected.length > 1 ? selected : [frame.id];
  const apply = (recipe: StoredPreset['recipe'], label: string) => {
    applyToFrames(targets, recipe);
    notify(`${label} applied to ${targets.length} frame${targets.length > 1 ? 's' : ''}.`);
  };

  return (
    <>
      <Section title={selected.length > 1 ? `Looks · ${selected.length} selected` : 'Looks'}>
        {BUILTIN_PRESETS.map((p) => (
          <button key={p.id} type="button" className="preset" onClick={() => apply(p.recipe, p.name)}>
            <b>{p.name}</b>
            <span>{p.description}</span>
          </button>
        ))}
      </Section>
      <Section title="Your presets">
        {mine.length === 0 && <p className="hint">Save the current frame's settings to reuse them on other rolls.</p>}
        {mine.map((p) => (
          <div key={p.id} className="preset-row">
            <button type="button" className="preset" onClick={() => apply(p.recipe, p.name)}>
              <b>{p.name}</b>
              <span>{Object.keys(p.recipe).map((k) => RECIPE_SECTIONS.find((s) => s.key === k)?.label).join(' · ')}</span>
            </button>
            <button
              type="button"
              className="btn small"
              aria-label={`Delete ${p.name}`}
              onClick={async () => {
                await deletePreset(p.id);
                setMine(await listPresets());
              }}
            >
              ✕
            </button>
          </div>
        ))}
        <input className="text-input" placeholder="Preset name" value={name} onChange={(e) => setName(e.target.value)} />
        {RECIPE_SECTIONS.map((s) => (
          <Check
            key={s.key}
            label={s.label}
            checked={sections.includes(s.key)}
            onChange={(v) => setSections(v ? [...sections, s.key] : sections.filter((x) => x !== s.key))}
          />
        ))}
        <button
          type="button"
          className="btn small"
          disabled={!name.trim() || !sections.length}
          onClick={async () => {
            await savePreset({ id: crypto.randomUUID(), name: name.trim(), recipe: pickSections(frame.recipe, sections), created: Date.now() });
            setMine(await listPresets());
            setName('');
            notify('Preset saved.');
          }}
        >
          Save preset
        </button>
      </Section>
    </>
  );
}
