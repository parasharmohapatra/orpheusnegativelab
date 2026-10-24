import { useState } from 'react';
import { OUTPUT_SPACES, type OutputSpaceId } from '../../engine/color/spaces';
import type { ExportFormat, ExportSettings } from '../../engine/export/exportFrame';
import { saveSetting } from '../persistence/db';
import { canWriteFolders, runExport } from '../io/exporter';
import { useStore } from '../store/rollStore';
import { Segmented } from './controls';

type Scope = 'current' | 'selected' | 'all';

export function ExportDialog() {
  const open = useStore((s) => s.exportOpen);
  const frames = useStore((s) => s.frames);
  const selected = useStore((s) => s.selected);
  const currentId = useStore((s) => s.currentId);
  const batch = useStore((s) => s.batch);
  const stored = useStore((s) => s.exportSettings);
  const set = useStore((s) => s.set);
  const [s, setS] = useState<ExportSettings>(stored);
  const [scope, setScope] = useState<Scope>(selected.length > 1 ? 'selected' : 'current');
  if (!open) return null;

  const ids = scope === 'all' ? frames.map((f) => f.id) : scope === 'selected' ? selected : currentId ? [currentId] : [];
  const running = batch?.running;
  const patch = (p: Partial<ExportSettings>) => setS((prev) => ({ ...prev, ...p }));
  const close = () => !running && set('exportOpen', false);

  return (
    <div className="scrim" onPointerDown={(e) => e.target === e.currentTarget && close()}>
      <div className="dialog" role="dialog" aria-modal="true" aria-label="Export">
        <h3>Export</h3>
        <Segmented<Scope>
          label="Frames"
          options={[
            { value: 'current', label: 'This frame' },
            { value: 'selected', label: `Selected (${selected.length})` },
            { value: 'all', label: `Roll (${frames.length})` },
          ]}
          value={scope}
          onChange={setScope}
        />
        <Segmented<ExportFormat>
          label="Format"
          options={[
            { value: 'jpeg', label: 'JPEG' },
            { value: 'tiff', label: 'TIFF' },
            { value: 'png', label: 'PNG' },
          ]}
          value={s.format}
          onChange={(format) => patch({ format, bits: format === 'jpeg' ? 8 : format === 'tiff' ? 16 : s.bits })}
        />
        <div className="grid2">
          {s.format === 'jpeg' ? (
            <label className="field">
              <span>Quality · {Math.round(s.quality * 100)}</span>
              <input type="range" min={0.6} max={1} step={0.01} value={s.quality} onChange={(e) => patch({ quality: Number(e.target.value) })} style={{ accentColor: 'var(--accent)' }} />
            </label>
          ) : (
            <Segmented<'8' | '16'>
              label="Bit depth"
              options={[
                { value: '8', label: '8-bit' },
                { value: '16', label: '16-bit' },
              ]}
              value={String(s.bits) as '8' | '16'}
              onChange={(v) => patch({ bits: Number(v) as 8 | 16 })}
            />
          )}
          <label className="field">
            <span>Colour space</span>
            <select className="text-input" value={s.space} onChange={(e) => patch({ space: e.target.value as OutputSpaceId })}>
              {Object.values(OUTPUT_SPACES).map((o) => (
                <option key={o.id} value={o.id}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Size</span>
            <select className="text-input" value={s.longEdge ?? 0} onChange={(e) => patch({ longEdge: Number(e.target.value) || null })}>
              <option value={0}>Full resolution</option>
              <option value={4096}>4096 px long edge</option>
              <option value={3000}>3000 px</option>
              <option value={2048}>2048 px</option>
              <option value={1080}>1080 px</option>
            </select>
          </label>
          <label className="field">
            <span>Border</span>
            <select className="text-input" value={s.border} onChange={(e) => patch({ border: Number(e.target.value) })}>
              <option value={0}>None</option>
              <option value={0.02}>Thin white</option>
              <option value={0.05}>White mat</option>
              <option value={0.1}>Wide white mat</option>
            </select>
          </label>
        </div>
        <p className="hint">
          {canWriteFolders()
            ? 'You will pick a folder; files are saved into an onl_positives subfolder next to your scans.'
            : ids.length > 1
              ? 'Your browser cannot write to folders, so multiple frames download as one ZIP.'
              : 'The file downloads to your Downloads folder.'}{' '}
          Rendering happens on this device; nothing is uploaded.
        </p>
        {batch && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div className="hist-meta">
              <span>{running ? `Developing ${batch.current ?? ''}` : 'Finished'}</span>
              <span>
                {batch.done} / {batch.total}
              </span>
            </div>
            <div className="progress">
              <div style={{ width: `${((batch.done + (running ? batch.progress : 0)) / Math.max(1, batch.total)) * 100}%` }} />
            </div>
            {batch.errors.map((e) => (
              <p key={e.name} className="hint warn">
                {e.name}: {e.error}
              </p>
            ))}
          </div>
        )}
        <div className="dialog-actions">
          {running ? (
            <button type="button" className="btn" onClick={() => batch && set('batch', { ...batch, cancel: true })}>
              Cancel after this frame
            </button>
          ) : (
            <button type="button" className="btn" onClick={close}>
              Close
            </button>
          )}
          <button
            type="button"
            className="btn primary"
            disabled={running || !ids.length}
            onClick={() => {
              set('exportSettings', s);
              void saveSetting('export', s);
              void runExport(ids, s);
            }}
          >
            Export {ids.length > 1 ? `${ids.length} frames` : ''}
          </button>
        </div>
      </div>
    </div>
  );
}
