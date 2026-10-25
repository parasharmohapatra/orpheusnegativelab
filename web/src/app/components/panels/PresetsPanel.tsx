import { useEffect, useMemo, useState } from 'react';
import { FILM_LOOKS, findLook, type FilmLook } from '../../../engine/pipeline/looks';
import { pickSections, RECIPE_SECTIONS, type RecipeSection } from '../../../engine/pipeline/recipe';
import { deletePreset, listPresets, savePreset, type StoredPreset } from '../../persistence/db';
import { renderPreview } from '../../store/frames';
import { BUILTIN_PRESETS } from '../../store/presets';
import { useCurrentFrame, useStore } from '../../store/rollStore';
import { Check, Section, Slider } from '../controls';

const DEFAULT_SECTIONS: RecipeSection[] = ['tone', 'color', 'curves', 'look'];
const FAMILIES: FilmLook['family'][] = ['Kodak', 'Fujifilm', 'CineStill', 'Cinema', 'Other', 'Black & white'];
const PREVIEW_PX = 150;

/** Live previews of every film look on the current frame, rendered in small idle chunks. */
function useLookPreviews(looks: FilmLook[]): Record<string, string> {
  const frame = useCurrentFrame();
  const roll = useStore((s) => s.roll);
  const [previews, setPreviews] = useState<Record<string, string>>({});
  // Re-render when anything but the look itself changes.
  const key = frame ? JSON.stringify([frame.id, frame.stats?.lHi, { ...frame.recipe, look: null }, roll?.base]) : '';

  useEffect(() => {
    if (!frame?.analysis || !frame.stats) return;
    let cancelled = false;
    const queue = [null as FilmLook | null, ...looks];
    const next = () => {
      if (cancelled || !queue.length) return;
      const look = queue.shift()!;
      const url = renderPreview(frame, roll, PREVIEW_PX, { ...frame.recipe, look: { id: look?.id ?? null, strength: 1 } });
      if (url) setPreviews((p) => ({ ...p, [look?.id ?? 'none']: url }));
      setTimeout(next, 0);
    };
    next();
    return () => {
      cancelled = true;
    };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  return previews;
}

export function PresetsPanel() {
  const frame = useCurrentFrame();
  const selected = useStore((s) => s.selected);
  const autoSync = useStore((s) => s.autoSync);
  const edit = useStore((s) => s.edit);
  const applyToFrames = useStore((s) => s.applyToFrames);
  const notify = useStore((s) => s.notify);
  const [mine, setMine] = useState<StoredPreset[]>([]);
  const [name, setName] = useState('');
  const [sections, setSections] = useState<RecipeSection[]>(DEFAULT_SECTIONS);
  const filmType = frame?.recipe.film.type;
  // Photos can take any look; negatives get the looks matching their film.
  const looks = useMemo(() => (filmType === 'photo' ? FILM_LOOKS : FILM_LOOKS.filter((l) => !!l.bw === (filmType === 'bw'))), [filmType]);
  const previews = useLookPreviews(looks);

  useEffect(() => {
    void listPresets().then(setMine);
  }, []);
  if (!frame) return null;

  const bulk = autoSync && selected.length > 1;
  const targets = bulk ? selected : [frame.id];
  const current = findLook(frame.recipe.look.id);
  const apply = (recipe: StoredPreset['recipe'], label: string) => {
    applyToFrames(targets, recipe);
    notify(`${label} applied to ${targets.length} frame${targets.length > 1 ? 's' : ''}.`);
  };
  const pickLook = (id: string | null) => edit((d) => void (d.look = { id, strength: d.look.id === id ? d.look.strength : 1 }));

  return (
    <>
      <Section title={bulk ? `Film look · ${selected.length} frames` : 'Film look'}>
        <div className="look-current">
          <span className="look-name">{current ? current.name : 'No look'}</span>
          <span className="hint">{current ? current.description : 'Pick a film to tint the conversion. Your own adjustments stay on top.'}</span>
        </div>
        {current && (
          <Slider
            label="Strength"
            value={frame.recipe.look.strength}
            min={0}
            max={1.5}
            reset={1}
            format={(v) => `${Math.round(v * 100)}%`}
            onChange={(v) => edit((d) => void (d.look.strength = v), false)}
          />
        )}
        <button type="button" className={`look-card none${!current ? ' on' : ''}`} onClick={() => pickLook(null)}>
          {previews.none ? <img src={previews.none} alt="" /> : <span className="spinner" />}
          <span>None</span>
        </button>
        {FAMILIES.map((fam) => {
          const list = looks.filter((l) => l.family === fam);
          if (!list.length) return null;
          return (
            <div key={fam} className="look-family">
              <div className="label-row">
                <span>{fam}</span>
              </div>
              <div className="look-grid">
                {list.map((l) => (
                  <button key={l.id} type="button" className={`look-card${current?.id === l.id ? ' on' : ''}`} title={l.description} onClick={() => pickLook(l.id)}>
                    {previews[l.id] ? <img src={previews[l.id]} alt="" /> : <span className="spinner" />}
                    <span>{l.name}</span>
                  </button>
                ))}
              </div>
            </div>
          );
        })}
        <p className="hint">Looks are interpretations inspired by each stock, layered on the conversion. They can't recreate a film your negative wasn't shot on.</p>
      </Section>

      <Section title="Print presets" defaultOpen={false}>
        {BUILTIN_PRESETS.map((p) => (
          <button key={p.id} type="button" className="preset" onClick={() => apply(p.recipe, p.name)}>
            <b>{p.name}</b>
            <span>{p.description}</span>
          </button>
        ))}
      </Section>

      <Section title="Your presets" defaultOpen={mine.length > 0}>
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
          <Check key={s.key} label={s.label} checked={sections.includes(s.key)} onChange={(v) => setSections(v ? [...sections, s.key] : sections.filter((x) => x !== s.key))} />
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
