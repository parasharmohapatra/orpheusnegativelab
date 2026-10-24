import { useEffect, useState } from 'react';
import { openFiles } from './store/frames';
import { useStore } from './store/rollStore';
import { loadSetting } from './persistence/db';
import type { ExportSettings } from '../engine/export/exportFrame';
import { ExportDialog } from './components/ExportDialog';
import { Panel } from './components/Panel';
import { EmptyState, Filmstrip, Header, ToolRail } from './components/Shell';
import { SyncDialog } from './components/SyncDialog';
import { Viewer } from './components/Viewer';

/** Collect files from a drop, descending into folders. */
async function filesFromDrop(dt: DataTransfer): Promise<File[]> {
  const entries = [...dt.items].map((i) => i.webkitGetAsEntry?.()).filter(Boolean) as FileSystemEntry[];
  if (!entries.length) return [...dt.files];
  const out: File[] = [];
  const walk = async (e: FileSystemEntry): Promise<void> => {
    if (e.isFile) {
      out.push(await new Promise<File>((res, rej) => (e as FileSystemFileEntry).file(res, rej)));
    } else if (e.isDirectory) {
      const reader = (e as FileSystemDirectoryEntry).createReader();
      let batch: FileSystemEntry[];
      do {
        batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        for (const c of batch) await walk(c);
      } while (batch.length);
    }
  };
  for (const e of entries) await walk(e);
  return out;
}

function useShortcuts() {
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest('input[type="text"], textarea, select, [contenteditable]')) return;
      const s = useStore.getState();
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) s.redo();
        else s.undo();
        return;
      }
      if (mod && e.key.toLowerCase() === 'a') {
        e.preventDefault();
        s.selectAll();
        return;
      }
      if (mod) return;
      switch (e.key) {
        case 'ArrowLeft':
          s.step(-1);
          break;
        case 'ArrowRight':
          s.step(1);
          break;
        case '\\':
          s.set('compare', true);
          break;
        case '1':
          s.set('tool', 'convert');
          break;
        case '2':
          s.set('tool', 'adjust');
          break;
        case '3':
          s.set('tool', 'color');
          break;
        case '4':
          s.set('tool', 'curves');
          break;
        case 'c':
          s.set('tool', 'crop');
          break;
        case 'p':
          s.set('tool', 'presets');
          break;
        case 'b':
          s.set('tool', 'convert');
          s.set('picker', s.picker === 'base' ? null : 'base');
          break;
        case 'n':
          s.set('tool', 'convert');
          s.set('picker', s.picker === 'neutral' ? null : 'neutral');
          break;
        case 'e':
          if (s.currentId) s.set('exportOpen', true);
          break;
        case 'Escape':
          s.set('picker', null);
          if (!s.batch?.running) s.set('exportOpen', false);
          s.set('syncOpen', false);
          break;
        default:
          return;
      }
      e.preventDefault();
    };
    const up = (e: KeyboardEvent) => {
      if (e.key === '\\') useStore.getState().set('compare', false);
    };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
    };
  }, []);
}

export function App() {
  const hasFrames = useStore((s) => s.frames.length > 0);
  const toast = useStore((s) => s.toast);
  const [dragging, setDragging] = useState(false);
  useShortcuts();

  useEffect(() => {
    void loadSetting<ExportSettings | null>('export', null).then((s) => s && useStore.getState().set('exportSettings', s));
  }, []);

  return (
    <div
      className="app"
      onDragOver={(e) => {
        if (![...e.dataTransfer.types].includes('Files')) return;
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => {
        if (e.relatedTarget === null) setDragging(false);
      }}
      onDrop={async (e) => {
        e.preventDefault();
        setDragging(false);
        void openFiles(await filesFromDrop(e.dataTransfer));
      }}
    >
      <Header />
      {hasFrames ? (
        <div className="workspace">
          <ToolRail />
          <Viewer />
          <Panel />
        </div>
      ) : (
        <EmptyState />
      )}
      {hasFrames ? <Filmstrip /> : <div />}
      <ExportDialog />
      <SyncDialog />
      {dragging && (
        <div className="drop-overlay">
          <div>Drop to develop</div>
        </div>
      )}
      {toast && <div className={`toast ${toast.kind === 'error' ? 'error' : ''}`}>{toast.text}</div>}
    </div>
  );
}
