import { useEffect, useRef, useState } from 'react';
import { defaultRecipe, setFilmType, type FilmType, type RangeMode } from '../../engine/pipeline/recipe';
import { BUILTIN_PRESETS } from '../store/presets';
import { FILM_LOOKS } from '../../engine/pipeline/looks';
import { useStore, type FrameEntry } from '../store/rollStore';
import { IconFlip, IconRotate } from './icons';

/** Selection actions shared by the library grid and the editor's filmstrip. */
export function BulkBar({ compact = false }: { compact?: boolean }) {
  const frames = useStore((s) => s.frames);
  const selected = useStore((s) => s.selected);
  const autoSync = useStore((s) => s.autoSync);
  const view = useStore((s) => s.view);
  const set = useStore((s) => s.set);
  const editFrames = useStore((s) => s.editFrames);
  const applyToFrames = useStore((s) => s.applyToFrames);
  const selectAll = useStore((s) => s.selectAll);
  const currentId = useStore((s) => s.currentId);
  const notify = useStore((s) => s.notify);
  const n = selected.length;
  const multi = n > 1;
  const done = (what: string) => notify(`${what} · ${n} frame${n === 1 ? '' : 's'}`);

  const film = (t: FilmType) => {
    editFrames(selected, (d) => setFilmType(d, t));
    done(t === 'bw' ? 'Black & white' : t === 'e6' ? 'Slide' : t === 'photo' ? 'Photo' : 'Colour negative');
  };
  const range = (m: RangeMode) => {
    editFrames(selected, (d) => void (d.convert.rangeMode = m));
    done(m === 'roll' ? 'Balanced per roll' : 'Balanced per frame');
  };

  return (
    <div className={`bulkbar${compact ? ' compact' : ''}`} role="toolbar" aria-label="Selection actions">
      <div className="bulk-count">
        <b>{n}</b>
        <span>of {frames.length} selected</span>
      </div>
      <div className="bulk-group">
        <button type="button" className="btn small ghost" onClick={selectAll} disabled={n === frames.length} title="Select all (⌘A)">
          All
        </button>
        <button type="button" className="btn small ghost" onClick={() => currentId && set('selected', [currentId])} disabled={!multi} title="Clear selection (Esc)">
          None
        </button>
      </div>
      {view === 'edit' && multi && (
        <label className={`bulk-sync${autoSync ? ' on' : ''}`} title="When on, every adjustment you make applies to all selected frames">
          <input type="checkbox" checked={autoSync} onChange={(e) => set('autoSync', e.target.checked)} />
          Edit all {n}
        </label>
      )}
      <span className="bulk-sep" />
      <div className="bulk-group">
        <select className="text-input small" value="" aria-label="Film type" onChange={(e) => e.target.value && film(e.target.value as FilmType)}>
          <option value="">Film…</option>
          <option value="c41">Colour negative</option>
          <option value="photo">Photo (no inversion)</option>
          <option value="bw">B&amp;W negative</option>
          <option value="e6">Slide</option>
        </select>
        <select className="text-input small" value="" aria-label="Colour balance" onChange={(e) => e.target.value && range(e.target.value as RangeMode)}>
          <option value="">Balance…</option>
          <option value="frame">Per frame</option>
          <option value="roll">Per roll</option>
        </select>
        <select
          className="text-input small"
          value=""
          aria-label="Apply a look"
          onChange={(e) => {
            const v = e.target.value;
            if (v === 'film:none') {
              editFrames(selected, (d) => void (d.look.id = null));
              return done('Film look removed');
            }
            const look = v.startsWith('film:') ? FILM_LOOKS.find((l) => l.id === v.slice(5)) : null;
            if (look) {
              editFrames(selected, (d) => void (d.look = { id: look.id, strength: d.look.id === look.id ? d.look.strength : 1 }));
              return done(look.name);
            }
            const p = BUILTIN_PRESETS.find((x) => x.id === v);
            if (!p) return;
            applyToFrames(selected, p.recipe);
            done(p.name);
          }}
        >
          <option value="">Look…</option>
          <optgroup label="Film looks">
            <option value="film:none">No film look</option>
            {FILM_LOOKS.map((l) => (
              <option key={l.id} value={`film:${l.id}`}>
                {l.name}
              </option>
            ))}
          </optgroup>
          <optgroup label="Print presets">
            {BUILTIN_PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </optgroup>
        </select>
      </div>
      <div className="bulk-group">
        <button type="button" className="btn small icon" title="Rotate left" aria-label="Rotate left" onClick={() => editFrames(selected, (d) => void (d.geometry.rot90 = ((d.geometry.rot90 + 3) % 4) as 0 | 1 | 2 | 3))}>
          <span style={{ transform: 'scaleX(-1)', display: 'inline-flex' }}>
            <IconRotate />
          </span>
        </button>
        <button type="button" className="btn small icon" title="Rotate right" aria-label="Rotate right" onClick={() => editFrames(selected, (d) => void (d.geometry.rot90 = ((d.geometry.rot90 + 1) % 4) as 0 | 1 | 2 | 3))}>
          <IconRotate />
        </button>
        <button type="button" className="btn small icon" title="Flip" aria-label="Flip" onClick={() => editFrames(selected, (d) => void (d.geometry.flipH = !d.geometry.flipH))}>
          <IconFlip />
        </button>
      </div>
      <span className="bulk-sep" />
      <div className="bulk-group">
        <button type="button" className="btn small" disabled={!multi} onClick={() => set('syncOpen', true)} title="Copy the current frame's settings to the selection">
          Sync from current
        </button>
        <button
          type="button"
          className="btn small"
          onClick={() => {
            editFrames(selected, (d) => {
              const keep = d.geometry;
              Object.assign(d, defaultRecipe(), { geometry: keep });
            });
            done('Reset');
          }}
          title="Reset all adjustments (keeps crop and rotation)"
        >
          Reset
        </button>
        <button type="button" className="btn small primary" onClick={() => set('exportOpen', true)}>
          Export {n}
        </button>
      </div>
    </div>
  );
}

const SIZES = { s: 150, m: 220, l: 320 } as const;

export function Library() {
  const frames = useStore((s) => s.frames);
  const selected = useStore((s) => s.selected);
  const currentId = useStore((s) => s.currentId);
  const toggle = useStore((s) => s.toggleSelect);
  const open = useStore((s) => s.open);
  const [size, setSize] = useState<keyof typeof SIZES>('m');
  const gridRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    gridRef.current?.querySelector('.card.current')?.scrollIntoView({ block: 'nearest' });
  }, [currentId]);

  const ready = frames.filter((f) => f.status === 'ready').length;
  return (
    <div className="library">
      <div className="library-head">
        <div>
          <h2 className="library-title">Roll</h2>
          <span className="hint">
            {frames.length} frames{ready < frames.length ? ` · ${ready} developed` : ''} · click to select, ⇧ for a range, ⌘ to add, double-click to edit
          </span>
        </div>
        <div className="seg" role="radiogroup" aria-label="Thumbnail size" style={{ width: 150 }}>
          {(Object.keys(SIZES) as (keyof typeof SIZES)[]).map((k) => (
            <button key={k} type="button" role="radio" aria-checked={size === k} className={size === k ? 'on' : ''} onClick={() => setSize(k)}>
              {k.toUpperCase()}
            </button>
          ))}
        </div>
      </div>
      <BulkBar />
      <div className="grid" ref={gridRef} style={{ ['--card' as string]: `${SIZES[size]}px` }}>
        {frames.map((f, i) => (
          <Card
            key={f.id}
            f={f}
            index={i}
            selected={selected.includes(f.id)}
            current={f.id === currentId}
            onClick={(e) => toggle(f.id, e.shiftKey ? 'range' : e.metaKey || e.ctrlKey ? 'toggle' : 'single')}
            onCheck={() => toggle(f.id, 'toggle')}
            onOpen={() => open(f.id)}
          />
        ))}
      </div>
    </div>
  );
}

function Card({
  f,
  index,
  selected,
  current,
  onClick,
  onCheck,
  onOpen,
}: {
  f: FrameEntry;
  index: number;
  selected: boolean;
  current: boolean;
  onClick: (e: React.MouseEvent) => void;
  onCheck: () => void;
  onOpen: () => void;
}) {
  const r = f.recipe;
  const tags: string[] = [];
  if (r.film.type === 'bw') tags.push('B&W');
  if (r.film.type === 'e6') tags.push('Slide');
  if (r.film.type === 'photo') tags.push('Photo');
  if (r.convert.rangeMode !== 'frame') tags.push(r.convert.rangeMode === 'roll' ? 'Roll' : 'Blend');
  const look = FILM_LOOKS.find((l) => l.id === r.look.id);
  if (look) tags.push(look.name);
  return (
    <div className={['card', selected ? 'selected' : '', current ? 'current' : '', f.status === 'error' ? 'error' : ''].join(' ')}>
      <button type="button" className="card-img" onClick={onClick} onDoubleClick={onOpen} title={f.error ? `${f.name}: ${f.error}` : `${f.name} — double-click to edit`}>
        {f.thumb ? <img src={f.thumb} alt={f.name} draggable={false} /> : f.status === 'error' ? <span className="status">Can't read this file</span> : <span className="spinner" />}
      </button>
      <label className="card-check" title={selected ? 'Deselect' : 'Add to selection'}>
        <input type="checkbox" checked={selected} onChange={onCheck} />
      </label>
      <div className="card-meta">
        <span className="card-num">{index + 1}</span>
        <span className="card-name">{f.name}</span>
        {f.history.length > 0 && <span className="card-dot" title="Edited" />}
        {tags.map((t) => (
          <span key={t} className="card-tag">
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}
