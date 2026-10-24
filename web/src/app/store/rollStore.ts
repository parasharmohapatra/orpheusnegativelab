import { create } from 'zustand';
import { produce } from 'immer';
import type { InputKind, LinearImage, SourceMeta } from '../../engine/types';
import type { OutputSpaceId } from '../../engine/color/spaces';
import type { FrameStats } from '../../engine/pipeline/analysis';
import { applyPartial, defaultRecipe, pickSections, type PartialRecipe, type Recipe, type RecipeSection } from '../../engine/pipeline/recipe';
import { computeRollStats, type RollStats } from '../../engine/pipeline/roll';
import type { ExportSettings } from '../../engine/export/exportFrame';

export type Tool = 'convert' | 'adjust' | 'color' | 'curves' | 'crop' | 'presets';
export type Picker = null | 'base' | 'neutral';
export type Surround = 'neutral' | 'grey18' | 'black' | 'white';

export interface FrameEntry {
  id: string;
  name: string;
  file: File;
  kind: InputKind;
  fingerprint: string;
  status: 'pending' | 'decoding' | 'ready' | 'error';
  error?: string;
  meta?: SourceMeta;
  analysis?: LinearImage;
  stats?: FrameStats;
  /** Key of the recipe fields the stats were measured with. */
  statsKey?: string;
  recipe: Recipe;
  history: Recipe[];
  future: Recipe[];
  thumb?: string;
}

export interface BatchState {
  running: boolean;
  total: number;
  done: number;
  current?: string;
  progress: number;
  errors: { name: string; error: string }[];
  cancel: boolean;
}

interface State {
  frames: FrameEntry[];
  currentId: string | null;
  selected: string[];
  roll: RollStats | null;
  tool: Tool;
  compare: boolean;
  picker: Picker;
  zoom: 'fit' | number;
  pan: [number, number];
  surround: Surround;
  frameRule: boolean;
  previewSpace: OutputSpaceId;
  exportOpen: boolean;
  syncOpen: boolean;
  exportSettings: ExportSettings;
  batch: BatchState | null;
  toast: { id: number; text: string; kind: 'info' | 'error' } | null;
  /** Snapshot taken when an uncommitted edit (slider drag) starts. */
  editSnapshot: Recipe | null;
}

interface Actions {
  addFrames(entries: FrameEntry[]): void;
  setCurrent(id: string): void;
  step(delta: number): void;
  toggleSelect(id: string, mode: 'single' | 'toggle' | 'range'): void;
  selectAll(): void;
  patchFrame(id: string, patch: Partial<FrameEntry>): void;
  removeFrame(id: string): void;
  /** Edit the current frame's recipe. `commit` pushes an undo step (use false while dragging). */
  edit(fn: (r: Recipe) => void, commit?: boolean): void;
  commit(): void;
  undo(): void;
  redo(): void;
  applyToFrames(ids: string[], part: PartialRecipe): void;
  syncFromCurrent(ids: string[], sections: RecipeSection[]): void;
  recomputeRoll(): void;
  set<K extends keyof State>(key: K, value: State[K]): void;
  notify(text: string, kind?: 'info' | 'error'): void;
}

const MAX_HISTORY = 100;

export const useStore = create<State & Actions>((set, get) => ({
  frames: [],
  currentId: null,
  selected: [],
  roll: null,
  tool: 'convert',
  compare: false,
  picker: null,
  zoom: 'fit',
  pan: [0, 0],
  surround: 'neutral',
  frameRule: true,
  previewSpace: 'srgb',
  exportOpen: false,
  syncOpen: false,
  exportSettings: { format: 'jpeg', bits: 8, quality: 0.92, space: 'srgb', longEdge: null, border: 0 },
  batch: null,
  toast: null,
  editSnapshot: null,

  addFrames(entries) {
    set((s) => {
      const known = new Set(s.frames.map((f) => f.fingerprint));
      const fresh = entries.filter((e) => !known.has(e.fingerprint));
      const frames = [...s.frames, ...fresh].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
      return { frames, currentId: s.currentId ?? fresh[0]?.id ?? null, selected: s.currentId ? s.selected : fresh[0] ? [fresh[0].id] : [] };
    });
  },

  setCurrent(id) {
    get().commit();
    set({ currentId: id, selected: [id], zoom: 'fit', pan: [0, 0], picker: null });
  },

  step(delta) {
    const { frames, currentId } = get();
    const i = frames.findIndex((f) => f.id === currentId);
    const next = frames[Math.min(frames.length - 1, Math.max(0, i + delta))];
    if (next && next.id !== currentId) get().setCurrent(next.id);
  },

  toggleSelect(id, mode) {
    const { frames, selected, currentId } = get();
    if (mode === 'single') return get().setCurrent(id);
    if (mode === 'toggle') {
      const has = selected.includes(id);
      const next = has ? selected.filter((x) => x !== id) : [...selected, id];
      return set({ selected: next.length ? next : [id] });
    }
    const a = frames.findIndex((f) => f.id === currentId);
    const b = frames.findIndex((f) => f.id === id);
    const [lo, hi] = a < b ? [a, b] : [b, a];
    set({ selected: frames.slice(lo, hi + 1).map((f) => f.id) });
  },

  selectAll() {
    set((s) => ({ selected: s.frames.map((f) => f.id) }));
  },

  patchFrame(id, patch) {
    set((s) => ({ frames: s.frames.map((f) => (f.id === id ? { ...f, ...patch } : f)) }));
  },

  removeFrame(id) {
    set((s) => {
      const frames = s.frames.filter((f) => f.id !== id);
      const currentId = s.currentId === id ? (frames[0]?.id ?? null) : s.currentId;
      return { frames, currentId, selected: s.selected.filter((x) => x !== id) };
    });
  },

  edit(fn, commit = true) {
    const { currentId, frames, editSnapshot } = get();
    const f = frames.find((x) => x.id === currentId);
    if (!f) return;
    const next = produce(f.recipe, fn);
    if (next === f.recipe) return;
    set((s) => ({
      editSnapshot: editSnapshot ?? f.recipe,
      frames: s.frames.map((x) => (x.id === f.id ? { ...x, recipe: next } : x)),
    }));
    if (commit) get().commit();
  },

  commit() {
    const { editSnapshot, currentId } = get();
    if (!editSnapshot) return;
    set((s) => ({
      editSnapshot: null,
      frames: s.frames.map((x) =>
        x.id === currentId && x.recipe !== editSnapshot
          ? { ...x, history: [...x.history.slice(-MAX_HISTORY + 1), editSnapshot], future: [] }
          : x,
      ),
    }));
  },

  undo() {
    get().commit();
    set((s) => ({
      frames: s.frames.map((x) => {
        if (x.id !== s.currentId || !x.history.length) return x;
        const prev = x.history[x.history.length - 1];
        return { ...x, recipe: prev, history: x.history.slice(0, -1), future: [x.recipe, ...x.future] };
      }),
    }));
  },

  redo() {
    set((s) => ({
      frames: s.frames.map((x) => {
        if (x.id !== s.currentId || !x.future.length) return x;
        const [next, ...rest] = x.future;
        return { ...x, recipe: next, history: [...x.history, x.recipe], future: rest };
      }),
    }));
  },

  applyToFrames(ids, part) {
    get().commit();
    set((s) => ({
      frames: s.frames.map((x) =>
        ids.includes(x.id) ? { ...x, recipe: applyPartial(x.recipe, part), history: [...x.history.slice(-MAX_HISTORY + 1), x.recipe], future: [] } : x,
      ),
    }));
  },

  syncFromCurrent(ids, sections) {
    const cur = get().frames.find((f) => f.id === get().currentId);
    if (!cur) return;
    get().applyToFrames(
      ids.filter((id) => id !== cur.id),
      pickSections(cur.recipe, sections),
    );
  },

  recomputeRoll() {
    const stats = get().frames.flatMap((f) => (f.stats ? [f.stats] : []));
    set({ roll: computeRollStats(stats) });
  },

  set(key, value) {
    set({ [key]: value } as Partial<State>);
  },

  notify(text, kind = 'info') {
    const id = Date.now();
    set({ toast: { id, text, kind } });
    setTimeout(() => {
      if (get().toast?.id === id) set({ toast: null });
    }, kind === 'error' ? 6000 : 3000);
  },
}));

export function newFrameEntry(file: File, kind: InputKind, fp: string, recipe?: Recipe | null): FrameEntry {
  return {
    id: crypto.randomUUID(),
    name: file.name,
    file,
    kind,
    fingerprint: fp,
    status: 'pending',
    recipe: recipe ?? defaultRecipe(),
    history: [],
    future: [],
  };
}

export const useCurrentFrame = () => useStore((s) => s.frames.find((f) => f.id === s.currentId) ?? null);
