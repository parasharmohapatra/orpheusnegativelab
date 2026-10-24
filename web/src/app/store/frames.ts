import type { LinearImage, Vec3 } from '../../engine/types';
import type { OutputSpaceId } from '../../engine/color/spaces';
import { kindOf, ACCEPTED_EXTENSIONS } from '../../engine/decode/index';
import { applyAffine, outputSize, outputToSource, withOrientation } from '../../engine/geometry/transform';
import { analyzeFrame, FULL_RECT, type FrameStats, type Rect } from '../../engine/pipeline/analysis';
import { buildParams, type RenderParams } from '../../engine/pipeline/params';
import { balancedDensity, renderCpu, sampleBilinear, toRGBA8 } from '../../engine/pipeline/reference';
import type { Recipe } from '../../engine/pipeline/recipe';
import type { RollStats } from '../../engine/pipeline/roll';
import { decodeFile } from '../../engine/workers/client';
import { fingerprint, loadRecipe, requestPersistence, saveRecipe } from '../persistence/db';
import { newFrameEntry, useStore, type FrameEntry } from './rollStore';

// ---------- proxies (kept out of the store: tens of MB each) ----------

const PROXY_CACHE = 3;
const proxies = new Map<string, LinearImage>();
const inflight = new Map<string, Promise<LinearImage>>();
const listeners = new Set<() => void>();

export function onProxyChange(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function remember(id: string, img: LinearImage): void {
  proxies.delete(id);
  proxies.set(id, img);
  const current = useStore.getState().currentId;
  while (proxies.size > PROXY_CACHE) {
    const oldest = [...proxies.keys()].find((k) => k !== current);
    if (!oldest) break;
    proxies.delete(oldest);
  }
  listeners.forEach((l) => l());
}

export function getProxy(id: string | null): LinearImage | undefined {
  if (!id) return undefined;
  const p = proxies.get(id);
  if (p) {
    proxies.delete(id);
    proxies.set(id, p);
  }
  return p;
}

/** Decode (or re-decode after eviction) the editing proxy for a frame. */
export function ensureProxy(id: string): Promise<LinearImage> {
  const have = proxies.get(id);
  if (have) return Promise.resolve(have);
  const running = inflight.get(id);
  if (running) return running;
  const f = useStore.getState().frames.find((x) => x.id === id);
  if (!f) return Promise.reject(new Error('Unknown frame'));
  const p = decodeFile(f.file, undefined, true)
    .then((d) => {
      remember(id, d.proxy);
      if (!f.meta) ingestDecoded(f.id, d.meta, d.analysis);
      return d.proxy;
    })
    .finally(() => inflight.delete(id));
  inflight.set(id, p);
  return p;
}

// ---------- opening files ----------

export async function openFiles(files: File[]): Promise<void> {
  const accepted = files.filter((f) => kindOf(f.name));
  const skipped = files.length - accepted.length;
  if (!accepted.length) {
    useStore.getState().notify(`No supported files. Accepted: ${ACCEPTED_EXTENSIONS.join(', ')}`, 'error');
    return;
  }
  void requestPersistence();
  const entries = await Promise.all(
    accepted.map(async (file) => {
      const fp = await fingerprint(file);
      return newFrameEntry(file, kindOf(file.name)!, fp, await loadRecipe(fp));
    }),
  );
  useStore.getState().addFrames(entries);
  if (skipped) useStore.getState().notify(`Skipped ${skipped} unsupported file${skipped > 1 ? 's' : ''}.`);
  // Current frame first so the viewer fills quickly, then the rest of the roll.
  const { currentId } = useStore.getState();
  const order = [...entries].sort((a, b) => (a.id === currentId ? -1 : b.id === currentId ? 1 : 0));
  for (const e of order) void ingest(e.id);
}

async function ingest(id: string): Promise<void> {
  const st = useStore.getState();
  const f = st.frames.find((x) => x.id === id);
  if (!f || f.status !== 'pending') return;
  st.patchFrame(id, { status: 'decoding' });
  try {
    const d = await decodeFile(f.file, undefined, id === st.currentId);
    ingestDecoded(id, d.meta, d.analysis);
    if (id === useStore.getState().currentId || proxies.size < PROXY_CACHE) remember(id, d.proxy);
  } catch (err) {
    useStore.getState().patchFrame(id, { status: 'error', error: err instanceof Error ? err.message : String(err) });
  }
}

function ingestDecoded(id: string, meta: FrameEntry['meta'], analysis: LinearImage): void {
  const st = useStore.getState();
  st.patchFrame(id, { status: 'ready', meta, analysis });
  refreshStats(id);
}

// ---------- statistics ----------

/** Recipe fields that change what the statistics measure. */
function statsKeyOf(r: Recipe): string {
  const g = r.geometry;
  return JSON.stringify([r.film.baseMode, r.film.baseRgb, r.film.bwMix, r.convert.blackClip, r.convert.whiteClip, g.crop, g.rot90, g.flipH, g.straighten, g.autoFrame]);
}

/** The user's crop as a rectangle in source coordinates (bounding box). */
export function cropAreaOf(f: FrameEntry): Rect {
  const g = f.recipe.geometry;
  if (g.crop.w >= 0.999 && g.crop.h >= 0.999 && g.straighten === 0) return FULL_RECT;
  const img = f.analysis!;
  const m = outputToSource(img.width, img.height, withOrientation(g, f.meta!.orientation));
  const pts = [[0, 0], [1, 0], [0, 1], [1, 1]].map(([u, v]) => applyAffine(m, u, v));
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  return { x0: Math.max(0, Math.min(...xs)), y0: Math.max(0, Math.min(...ys)), x1: Math.min(1, Math.max(...xs)), y1: Math.min(1, Math.max(...ys)) };
}

function computeStats(f: FrameEntry): FrameStats {
  const r = f.recipe;
  return analyzeFrame(f.analysis!, {
    cropArea: cropAreaOf(f),
    autoFrame: r.geometry.autoFrame,
    baseOverride: r.film.baseMode === 'manual' ? r.film.baseRgb : null,
    blackClip: r.convert.blackClip,
    whiteClip: r.convert.whiteClip,
    bwMix: r.film.bwMix,
  });
}

function refreshStats(id: string): void {
  const st = useStore.getState();
  const f = st.frames.find((x) => x.id === id);
  if (!f?.analysis || !f.meta) return;
  const key = statsKeyOf(f.recipe);
  if (f.statsKey === key && f.stats) return;
  st.patchFrame(id, { stats: computeStats(f), statsKey: key });
  st.recomputeRoll();
}

export function paramsFor(f: FrameEntry | null, roll: RollStats | null, space: OutputSpaceId): RenderParams | null {
  if (!f?.stats || !f.meta) return null;
  return buildParams(f.recipe, f.stats, roll, f.meta, space);
}

// ---------- thumbnails ----------

const THUMB = 240;
const thumbQueue = new Set<string>();
let thumbTimer: ReturnType<typeof setTimeout> | null = null;

function scheduleThumbs(ids: string[], delay = 250): void {
  ids.forEach((id) => thumbQueue.add(id));
  if (thumbTimer) clearTimeout(thumbTimer);
  thumbTimer = setTimeout(drainThumbs, delay);
}

function drainThumbs(): void {
  const id = thumbQueue.values().next().value as string | undefined;
  if (!id) return;
  thumbQueue.delete(id);
  const st = useStore.getState();
  const f = st.frames.find((x) => x.id === id);
  const p = paramsFor(f ?? null, st.roll, 'srgb');
  if (f?.analysis && p) {
    const g = withOrientation(f.recipe.geometry, f.meta!.orientation);
    const [ow, oh] = outputSize(f.analysis.width, f.analysis.height, g);
    const s = Math.min(1, THUMB / Math.max(ow, oh));
    const w = Math.max(1, Math.round(ow * s));
    const h = Math.max(1, Math.round(oh * s));
    const rgb = renderCpu(f.analysis, p, outputToSource(f.analysis.width, f.analysis.height, g), w, h);
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    canvas.getContext('2d')!.putImageData(new ImageData(toRGBA8(rgb, w, h), w, h), 0, 0);
    st.patchFrame(id, { thumb: canvas.toDataURL('image/jpeg', 0.85) });
  }
  if (thumbQueue.size) thumbTimer = setTimeout(drainThumbs, 0);
}

// ---------- pickers ----------

/** Average a small neighbourhood of the proxy at normalized source coords. */
export function sampleProxy(img: LinearImage, u: number, v: number, radius = 3): Vec3 {
  const acc: Vec3 = [0, 0, 0];
  const px: Vec3 = [0, 0, 0];
  let n = 0;
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      sampleBilinear(img, u + dx / img.width, v + dy / img.height, px);
      acc[0] += px[0];
      acc[1] += px[1];
      acc[2] += px[2];
      n++;
    }
  }
  return [acc[0] / n, acc[1] / n, acc[2] / n];
}

/** Offsets that make this pixel neutral after balancing (neutral eyedropper). */
export function neutralOffsetFor(p: RenderParams, rgb: Vec3): Vec3 {
  const n = balancedDensity(p, rgb, false);
  return [n[1] - n[0], 0, n[1] - n[2]];
}

// ---------- reacting to edits ----------

const saveTimers = new Map<string, ReturnType<typeof setTimeout>>();
let lastRoll: RollStats | null = null;

useStore.subscribe((s, prev) => {
  if (s.frames !== prev.frames) {
    const before = new Map(prev.frames.map((f) => [f.id, f]));
    for (const f of s.frames) {
      const old = before.get(f.id);
      if (old && old.recipe === f.recipe && old.stats === f.stats) continue;
      if (old && old.recipe !== f.recipe) {
        refreshStats(f.id);
        clearTimeout(saveTimers.get(f.id));
        saveTimers.set(f.id, setTimeout(() => void saveRecipe(f.fingerprint, useStore.getState().frames.find((x) => x.id === f.id)!.recipe), 600));
      }
      if (f.stats) scheduleThumbs([f.id], f.id === s.currentId ? 300 : 50);
    }
  }
  if (s.roll !== lastRoll) {
    lastRoll = s.roll;
    scheduleThumbs(s.frames.filter((f) => f.stats).map((f) => f.id), 400);
  }
  if (s.currentId !== prev.currentId && s.currentId) void ensureProxy(s.currentId).catch(() => undefined);
});
