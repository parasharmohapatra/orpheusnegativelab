import { useEffect, useRef, useState } from 'react';
import { ACCEPTED_EXTENSIONS } from '../../engine/decode/index';
import { openFiles } from '../store/frames';
import { useCurrentFrame, useStore, type Surround, type Tool } from '../store/rollStore';
import { BulkBar } from './Library';
import { IconAdjust, IconColor, IconConvert, IconCrop, IconCurves, IconHelp, IconPresets, IconSettings } from './icons';

const ACCEPT = ACCEPTED_EXTENSIONS.map((e) => `.${e}`).join(',');

/** Hidden inputs for "Open files" and "Open folder"; returns click handlers. */
export function useFilePickers() {
  const files = useRef<HTMLInputElement | null>(null);
  const folder = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    const make = (dir: boolean) => {
      const el = document.createElement('input');
      el.type = 'file';
      el.multiple = true;
      if (dir) el.setAttribute('webkitdirectory', '');
      else el.accept = ACCEPT;
      el.onchange = () => {
        if (el.files?.length) void openFiles([...el.files]);
        el.value = '';
      };
      return el;
    };
    files.current = make(false);
    folder.current = make(true);
  }, []);
  return { openFiles: () => files.current?.click(), openFolder: () => folder.current?.click() };
}

export function Header() {
  const frame = useCurrentFrame();
  const frames = useStore((s) => s.frames);
  const compare = useStore((s) => s.compare);
  const set = useStore((s) => s.set);
  const undo = useStore((s) => s.undo);
  const redo = useStore((s) => s.redo);
  const pickers = useFilePickers();
  const view = useStore((s) => s.view);
  return (
    <header className="header">
      <div className="brand">
        <h1 className="wordmark">
          Orpheus<small>Negative Lab</small>
        </h1>
        {frame && view === 'edit' && <span className="filename">{frame.name}</span>}
      </div>
      {frames.length > 0 && (
        <div className="seg view-toggle" role="radiogroup" aria-label="View">
          <button type="button" role="radio" aria-checked={view === 'library'} className={view === 'library' ? 'on' : ''} onClick={() => set('view', 'library')} title="Grid (G)">
            Library
          </button>
          <button type="button" role="radio" aria-checked={view === 'edit'} className={view === 'edit' ? 'on' : ''} onClick={() => set('view', 'edit')} title="Edit (D)">
            Develop
          </button>
        </div>
      )}
      <div className="header-actions">
        <button type="button" className="btn" onClick={pickers.openFolder} title="Open a folder of scans">
          Open roll
        </button>
        {frames.length > 0 && (
          <>
            <button type="button" className="btn" onClick={undo} disabled={!frame?.history.length} title="Undo (⌘Z)">
              Undo
            </button>
            <button type="button" className="btn" onClick={redo} disabled={!frame?.future.length} title="Redo (⇧⌘Z)">
              Redo
            </button>
            <button
              type="button"
              className={`btn${compare ? ' active' : ''}`}
              onPointerDown={() => set('compare', true)}
              onPointerUp={() => set('compare', false)}
              onPointerLeave={() => compare && set('compare', false)}
              title="Hold to see the negative (or hold \)"
            >
              Compare
            </button>
            <button type="button" className="btn primary" onClick={() => set('exportOpen', true)} disabled={!frame} title="Export (E)">
              Export
            </button>
          </>
        )}
      </div>
    </header>
  );
}

const TOOLS: { id: Tool; label: string; key: string; icon: () => JSX.Element }[] = [
  { id: 'convert', label: 'Convert', key: '1', icon: IconConvert },
  { id: 'adjust', label: 'Adjust', key: '2', icon: IconAdjust },
  { id: 'color', label: 'Colour', key: '3', icon: IconColor },
  { id: 'curves', label: 'Curves', key: '4', icon: IconCurves },
  { id: 'crop', label: 'Crop', key: 'C', icon: IconCrop },
  { id: 'presets', label: 'Looks', key: 'P', icon: IconPresets },
];

export function ToolRail() {
  const tool = useStore((s) => s.tool);
  const set = useStore((s) => s.set);
  const [menu, setMenu] = useState<null | 'settings' | 'help'>(null);
  return (
    <nav className="rail" aria-label="Tools">
      {TOOLS.map((t) => (
        <button key={t.id} type="button" aria-label={t.label} aria-pressed={tool === t.id} className={`rail-btn${tool === t.id ? ' active' : ''}`} onClick={() => set('tool', t.id)}>
          <t.icon />
          <span className="tip">
            {t.label} <span className="kbd">{t.key}</span>
          </span>
        </button>
      ))}
      <div className="rail-spacer" />
      <button type="button" aria-label="View settings" className={`rail-btn${menu === 'settings' ? ' active' : ''}`} onClick={() => setMenu(menu === 'settings' ? null : 'settings')}>
        <IconSettings />
        <span className="tip">View</span>
      </button>
      <button type="button" aria-label="Shortcuts" className={`rail-btn${menu === 'help' ? ' active' : ''}`} onClick={() => setMenu(menu === 'help' ? null : 'help')}>
        <IconHelp />
        <span className="tip">Shortcuts</span>
      </button>
      {menu === 'settings' && <ViewSettings onClose={() => setMenu(null)} />}
      {menu === 'help' && <ShortcutHelp onClose={() => setMenu(null)} />}
    </nav>
  );
}

function Popover({ children, onClose }: { children: React.ReactNode; onClose: () => void }) {
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [onClose]);
  return (
    <div className="scrim" style={{ background: 'transparent', placeItems: 'end start', padding: '0 0 130px 86px' }} onPointerDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog" style={{ width: 300, gap: 16, padding: 22 }}>
        {children}
      </div>
    </div>
  );
}

function ViewSettings({ onClose }: { onClose: () => void }) {
  const surround = useStore((s) => s.surround);
  const frameRule = useStore((s) => s.frameRule);
  const set = useStore((s) => s.set);
  const opts: { value: Surround; label: string; swatch: string }[] = [
    { value: 'neutral', label: 'Neutral', swatch: '#121212' },
    { value: 'grey18', label: '18% grey', swatch: '#767676' },
    { value: 'black', label: 'Black', swatch: '#000' },
    { value: 'white', label: 'White', swatch: '#f0f0f0' },
  ];
  return (
    <Popover onClose={onClose}>
      <div className="label-row">
        <span>Surround</span>
      </div>
      <p className="hint" style={{ marginTop: -8 }}>
        A neutral backdrop keeps your colour judgement honest.
      </p>
      <div className="row">
        {opts.map((o) => (
          <button key={o.value} type="button" className={`btn small${surround === o.value ? ' active' : ''}`} onClick={() => set('surround', o.value)} title={o.label}>
            <span className="swatch" style={{ width: 14, height: 14, background: o.swatch }} />
          </button>
        ))}
      </div>
      <label className="check">
        <input type="checkbox" checked={frameRule} onChange={(e) => set('frameRule', e.target.checked)} />
        Gold frame around the print
      </label>
    </Popover>
  );
}

function ShortcutHelp({ onClose }: { onClose: () => void }) {
  const rows: [string, string][] = [
    ['← / →', 'Previous / next frame'],
    ['1 2 3 4', 'Convert, Adjust, Colour, Curves'],
    ['C / P', 'Crop / Film looks'],
    ['\\ (hold)', 'Show the negative'],
    ['⌘Z / ⇧⌘Z', 'Undo / redo'],
    ['G / D', 'Library grid / develop'],
    ['⌘A / Esc', 'Select all / clear selection'],
    ['Space', 'Add or remove frame from selection'],
    ['B / N', 'Sample base / pick neutral'],
    ['E', 'Export'],
    ['Esc', 'Cancel picker or dialog'],
  ];
  return (
    <Popover onClose={onClose}>
      {rows.map(([k, v]) => (
        <div key={k} className="label-row" style={{ textTransform: 'none', letterSpacing: 0, fontSize: 12.5 }}>
          <span className="kbd">{k}</span>
          <span style={{ color: 'var(--text-muted)' }}>{v}</span>
        </div>
      ))}
    </Popover>
  );
}

export function Filmstrip() {
  const frames = useStore((s) => s.frames);
  const currentId = useStore((s) => s.currentId);
  const selected = useStore((s) => s.selected);
  const toggle = useStore((s) => s.toggleSelect);
  const open = useStore((s) => s.stripOpen);
  const set = useStore((s) => s.set);
  const stripRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = stripRef.current?.querySelector('.thumb.current') as HTMLElement | null;
    el?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }, [currentId, open]);

  const idx = frames.findIndex((f) => f.id === currentId) + 1;
  const ready = frames.filter((f) => f.status === 'ready').length;
  return (
    <footer className={`filmstrip${open ? '' : ' closed'}`}>
      <div className="strip-bar">
        <button type="button" className="strip-toggle" onClick={() => set('stripOpen', !open)} aria-expanded={open} title={open ? 'Hide filmstrip' : 'Show filmstrip'}>
          <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true" style={{ transform: open ? 'none' : 'rotate(180deg)' }}>
            <path d="M3 4.5l3 3 3-3" fill="none" stroke="currentColor" strokeWidth="1.3" />
          </svg>
          <span>
            Frame {idx} of {frames.length}
            {ready < frames.length ? ` · ${ready} developed` : ''}
          </span>
        </button>
        {selected.length > 1 ? <BulkBar compact /> : <span className="hint strip-hint">⇧ or ⌘ click to select several frames, then edit them together</span>}
        <button type="button" className="btn small" onClick={() => set('view', 'library')} title="Grid view (G)">
          Grid
        </button>
      </div>
      {open && (
        <div className="strip" ref={stripRef}>
          {frames.map((f, i) => {
            const sel = selected.includes(f.id);
            return (
              <button
                key={f.id}
                type="button"
                className={['thumb', f.id === currentId ? 'current' : '', sel ? 'selected' : '', f.status === 'error' ? 'error' : ''].join(' ')}
                title={f.error ? `${f.name}: ${f.error}` : f.name}
                onClick={(e) => toggle(f.id, e.shiftKey ? 'range' : e.metaKey || e.ctrlKey ? 'toggle' : 'single')}
              >
                {f.thumb ? <img src={f.thumb} alt={f.name} draggable={false} /> : f.status === 'error' ? <span className="status">Can't read</span> : <span className="spinner" />}
                <span className="num">{i + 1}</span>
                {selected.length > 1 && <span className={`tick${sel ? ' on' : ''}`} aria-hidden="true" />}
              </button>
            );
          })}
        </div>
      )}
    </footer>
  );
}

export function EmptyState() {
  const pickers = useFilePickers();
  return (
    <div className="empty">
      <div className="empty-card">
        <h2>
          Develop your <i>negatives</i>
        </h2>
        <div className="empty-rule" />
        <p>
          Drop in a roll of camera scans or film scanner TIFFs. Orpheus measures the orange mask, inverts in density like a darkroom print, and balances every
          frame. Everything runs in your browser; your files never leave this device.
        </p>
        <div className="row" style={{ width: 'min(380px, 100%)' }}>
          <button type="button" className="btn primary" onClick={pickers.openFolder}>
            Open folder
          </button>
          <button type="button" className="btn" onClick={pickers.openFiles}>
            Choose files
          </button>
        </div>
        <div className="empty-formats">
          RAW: CR2 · CR3 · NEF · ARW · DNG · RAF · ORF · RW2 · PEF
          <br />
          16-bit TIFF · JPEG · PNG
        </div>
      </div>
    </div>
  );
}
