import { zipSync } from 'fflate';
import type { ExportSettings } from '../../engine/export/exportFrame';
import { exportInWorker } from '../../engine/workers/client';
import { paramsFor } from '../store/frames';
import { useStore } from '../store/rollStore';

type DirHandle = FileSystemDirectoryHandle & {
  getDirectoryHandle(name: string, o?: { create?: boolean }): Promise<FileSystemDirectoryHandle>;
};
declare global {
  interface Window {
    showDirectoryPicker?: (o?: { mode?: 'read' | 'readwrite'; id?: string }) => Promise<DirHandle>;
  }
}

export const canWriteFolders = () => typeof window.showDirectoryPicker === 'function';

function baseName(name: string): string {
  const i = name.lastIndexOf('.');
  return i > 0 ? name.slice(0, i) : name;
}

function download(bytes: Uint8Array, name: string, mime: string): void {
  const url = URL.createObjectURL(new Blob([bytes as BlobPart], { type: mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/**
 * Export frames one at a time in the export worker. On Chromium the user picks a
 * folder and files land in an `onl_positives` subfolder; elsewhere a single file
 * downloads directly and several are bundled into a ZIP.
 */
export async function runExport(ids: string[], settings: ExportSettings): Promise<void> {
  const st = useStore.getState();
  let dir: FileSystemDirectoryHandle | null = null;
  if (canWriteFolders()) {
    try {
      const root = await window.showDirectoryPicker!({ mode: 'readwrite', id: 'orpheus-export' });
      dir = await root.getDirectoryHandle('onl_positives', { create: true });
    } catch (e) {
      if ((e as DOMException).name === 'AbortError') return;
      dir = null;
    }
  }

  const zipped: Record<string, Uint8Array> = {};
  st.set('batch', { running: true, total: ids.length, done: 0, progress: 0, errors: [], cancel: false });
  const update = (patch: Partial<NonNullable<ReturnType<typeof useStore.getState>['batch']>>) => {
    const b = useStore.getState().batch;
    if (b) useStore.getState().set('batch', { ...b, ...patch });
  };

  for (let i = 0; i < ids.length; i++) {
    if (useStore.getState().batch?.cancel) break;
    const s = useStore.getState();
    const f = s.frames.find((x) => x.id === ids[i]);
    if (!f) continue;
    update({ current: f.name, progress: 0 });
    const params = paramsFor(f, s.roll, settings.space);
    if (!params) {
      update({ done: i + 1, errors: [...(useStore.getState().batch?.errors ?? []), { name: f.name, error: f.error ?? 'Not analysed yet' }] });
      continue;
    }
    try {
      const result = await exportInWorker({ file: f.file, params, geometry: f.recipe.geometry, settings }, (p) => update({ progress: p }));
      const name = `${baseName(f.name)}.${result.ext}`;
      if (dir) {
        const fh = await dir.getFileHandle(name, { create: true });
        const w = await fh.createWritable();
        await w.write(result.bytes as BlobPart);
        await w.close();
      } else if (ids.length === 1) {
        download(result.bytes, name, result.mime);
      } else {
        zipped[name] = result.bytes;
      }
    } catch (e) {
      update({ errors: [...(useStore.getState().batch?.errors ?? []), { name: f.name, error: (e as Error).message }] });
    }
    update({ done: i + 1 });
  }

  if (!dir && Object.keys(zipped).length) {
    download(zipSync(zipped, { level: 0 }), 'onl_positives.zip', 'application/zip');
  }
  const b = useStore.getState().batch;
  update({ running: false, current: undefined });
  const ok = (b?.done ?? 0) - (b?.errors.length ?? 0);
  useStore.getState().notify(
    b?.cancel ? `Export cancelled after ${ok} frame${ok === 1 ? '' : 's'}.` : `Exported ${ok} frame${ok === 1 ? '' : 's'}${dir ? ' to onl_positives/' : ''}.`,
    b?.errors.length ? 'error' : 'info',
  );
}
