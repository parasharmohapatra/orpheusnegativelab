import { openDB, type IDBPDatabase } from 'idb';
import { migrateRecipe, type PartialRecipe, type Recipe } from '../../engine/pipeline/recipe';

export interface StoredPreset {
  id: string;
  name: string;
  recipe: PartialRecipe;
  created: number;
}

let dbp: Promise<IDBPDatabase> | null = null;

function db(): Promise<IDBPDatabase> {
  dbp ??= openDB('orpheus-negative-lab', 1, {
    upgrade(d) {
      d.createObjectStore('recipes');
      d.createObjectStore('presets', { keyPath: 'id' });
      d.createObjectStore('settings');
    },
  }).catch((e) => {
    dbp = null;
    throw e;
  });
  return dbp;
}

/** All persistence is best-effort: private windows and blocked storage just mean nothing is remembered. */
async function safe<T>(fn: (d: IDBPDatabase) => Promise<T>, fallback: T): Promise<T> {
  try {
    return await fn(await db());
  } catch {
    return fallback;
  }
}

export const loadRecipe = (fingerprint: string) =>
  safe(async (d) => {
    const raw = await d.get('recipes', fingerprint);
    return raw ? migrateRecipe(raw) : null;
  }, null as Recipe | null);

export const saveRecipe = (fingerprint: string, recipe: Recipe) => safe((d) => d.put('recipes', recipe, fingerprint).then(() => undefined), undefined);

export const listPresets = () => safe((d) => d.getAll('presets') as Promise<StoredPreset[]>, [] as StoredPreset[]);
export const savePreset = (p: StoredPreset) => safe((d) => d.put('presets', p).then(() => undefined), undefined);
export const deletePreset = (id: string) => safe((d) => d.delete('presets', id), undefined);

export const loadSetting = <T>(key: string, fallback: T) => safe(async (d) => ((await d.get('settings', key)) as T) ?? fallback, fallback);
export const saveSetting = (key: string, value: unknown) => safe((d) => d.put('settings', value, key).then(() => undefined), undefined);

export async function requestPersistence(): Promise<void> {
  try {
    await navigator.storage?.persist?.();
  } catch {
    /* optional */
  }
}

/** Stable id for a file across sessions: name, size and a hash of its first 64 KB. */
export async function fingerprint(file: File): Promise<string> {
  try {
    const head = await file.slice(0, 65536).arrayBuffer();
    const digest = await crypto.subtle.digest('SHA-1', head);
    const hex = [...new Uint8Array(digest)].slice(0, 8).map((b) => b.toString(16).padStart(2, '0')).join('');
    return `${file.name}|${file.size}|${hex}`;
  } catch {
    return `${file.name}|${file.size}|${file.lastModified}`;
  }
}
