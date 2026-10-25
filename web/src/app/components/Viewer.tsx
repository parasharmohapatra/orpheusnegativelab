import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { LinearImage, Mat3, Vec3 } from '../../engine/types';
import { mul } from '../../engine/color/matrices';
import { REC709_TO_WORKING, workingToOutput } from '../../engine/color/spaces';
import { outputSize, outputToSource, withOrientation, applyAffine, type Affine, type Geometry } from '../../engine/geometry/transform';
import { Renderer, WebGLUnsupportedError } from '../../engine/gpu/renderer';
import type { RenderParams } from '../../engine/pipeline/params';
import { ensureProxy, getProxy, neutralOffsetFor, onProxyChange, paramsFor, sampleProxy } from '../store/frames';
import { useCurrentFrame, useStore, type Surround } from '../store/rollStore';
import { publishHistogram } from './histogramBus';
import { CropOverlay } from './CropOverlay';

const SURROUND: Record<Surround, Vec3> = {
  neutral: [0.071, 0.071, 0.071],
  grey18: [0.46, 0.46, 0.46],
  black: [0, 0, 0],
  white: [0.94, 0.94, 0.94],
};
const PAD = 44;
const CAPTION = 36;

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Where the output image sits in the viewer, in CSS pixels. */
function layout(cw: number, ch: number, outW: number, outH: number, zoom: 'fit' | number, pan: [number, number]): Rect {
  const availW = Math.max(40, cw - PAD * 2);
  const availH = Math.max(40, ch - PAD * 2 - CAPTION);
  const fit = Math.min(availW / outW, availH / outH);
  const z = zoom === 'fit' ? 1 : zoom;
  const w = outW * fit * z;
  const h = outH * fit * z;
  return { x: (cw - w) / 2 + pan[0], y: PAD + (availH - h) / 2 + pan[1], w, h };
}

export function Viewer() {
  const frame = useCurrentFrame();
  const roll = useStore((s) => s.roll);
  const tool = useStore((s) => s.tool);
  const compare = useStore((s) => s.compare);
  const picker = useStore((s) => s.picker);
  const zoom = useStore((s) => s.zoom);
  const pan = useStore((s) => s.pan);
  const surround = useStore((s) => s.surround);
  const frameRule = useStore((s) => s.frameRule);
  const previewSpace = useStore((s) => s.previewSpace);
  const set = useStore((s) => s.set);
  const edit = useStore((s) => s.edit);

  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<Renderer | null>(null);
  const uploaded = useRef<LinearImage | null>(null);
  const [size, setSize] = useState<[number, number]>([0, 0]);
  const [proxy, setProxy] = useState<LinearImage | undefined>(undefined);
  const [glError, setGlError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [contextVersion, setContextVersion] = useState(0);
  const dragRef = useRef<{ x: number; y: number; pan: [number, number] } | null>(null);

  // ----- renderer lifecycle -----
  useEffect(() => {
    const canvas = canvasRef.current!;
    try {
      const r = new Renderer(canvas);
      const gl = r.gl as WebGL2RenderingContext & { drawingBufferColorSpace?: string };
      if ('drawingBufferColorSpace' in gl && matchMedia('(color-gamut: p3)').matches) {
        gl.drawingBufferColorSpace = 'display-p3';
        set('previewSpace', 'displayP3');
      }
      rendererRef.current = r;
    } catch (e) {
      setGlError(e instanceof WebGLUnsupportedError ? e.message : `Could not start the GPU renderer: ${(e as Error).message}`);
    }
    const lost = (e: Event) => {
      e.preventDefault();
      if (rendererRef.current) rendererRef.current.lost = true;
    };
    const restored = () => {
      try {
        rendererRef.current = new Renderer(canvas);
        uploaded.current = null;
        setContextVersion((v) => v + 1);
      } catch (e) {
        setGlError((e as Error).message);
      }
    };
    canvas.addEventListener('webglcontextlost', lost);
    canvas.addEventListener('webglcontextrestored', restored);
    return () => {
      canvas.removeEventListener('webglcontextlost', lost);
      canvas.removeEventListener('webglcontextrestored', restored);
      rendererRef.current?.dispose();
      rendererRef.current = null;
    };
  }, [set]);

  useLayoutEffect(() => {
    const el = wrapRef.current!;
    const ro = new ResizeObserver(([e]) => setSize([e.contentRect.width, e.contentRect.height]));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ----- proxy for the current frame -----
  const frameId = frame?.id ?? null;
  useEffect(() => {
    const sync = () => setProxy(getProxy(frameId));
    sync();
    if (frameId && !getProxy(frameId)) {
      setLoading(true);
      ensureProxy(frameId)
        .catch(() => undefined)
        .finally(() => setLoading(false));
    }
    return onProxyChange(sync);
  }, [frameId]);

  // ----- derived render state -----
  const params: RenderParams | null = useMemo(() => paramsFor(frame, roll, previewSpace), [frame, roll, previewSpace]);
  const cropping = tool === 'crop';
  const geometry: Geometry | null = useMemo(() => {
    if (!frame?.meta) return null;
    const g = frame.recipe.geometry;
    const shown = cropping ? { ...g, crop: { x: 0, y: 0, w: 1, h: 1 } } : g;
    return withOrientation(shown, frame.meta.orientation);
  }, [frame?.meta, frame?.recipe.geometry, cropping]);

  const out = proxy && geometry ? outputSize(proxy.width, proxy.height, geometry) : null;
  const rect = out ? layout(size[0], size[1], out[0], out[1], zoom, pan) : null;
  const geo: Affine | null = useMemo(() => (proxy && geometry ? outputToSource(proxy.width, proxy.height, geometry) : null), [proxy, geometry]);
  const meta = frame?.meta;
  const negToOut: Mat3 | null = useMemo(
    () => (meta ? mul(workingToOutput(previewSpace), mul(REC709_TO_WORKING, meta.toRec709)) : null),
    [meta, previewSpace],
  );

  // ----- draw -----
  const histTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    const r = rendererRef.current;
    const canvas = canvasRef.current;
    if (!r || !canvas || r.lost || !size[0]) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = Math.round(size[0] * dpr);
    const H = Math.round(size[1] * dpr);
    if (canvas.width !== W || canvas.height !== H) {
      canvas.width = W;
      canvas.height = H;
    }
    const sur = SURROUND[surround];
    if (!proxy || !params || !rect || !geo || !negToOut) {
      const gl = r.gl;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, W, H);
      gl.clearColor(sur[0], sur[1], sur[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      return;
    }
    if (uploaded.current !== proxy) {
      r.setSource(proxy);
      uploaded.current = proxy;
    }
    const raf = requestAnimationFrame(() => {
      r.setParams(params);
      const view: Affine = [size[0] / rect.w, 0, -rect.x / rect.w, 0, size[1] / rect.h, -rect.y / rect.h];
      r.draw({ view, geo, surround: sur, mode: compare ? 1 : 0, negToOut }, W, H);
      // Histogram of the converted output (throttled, small render).
      if (histTimer.current) clearTimeout(histTimer.current);
      histTimer.current = setTimeout(() => {
        if (r.lost || !out) return;
        const hw = 256;
        const hh = Math.max(1, Math.round((hw * out[1]) / out[0]));
        const t = r.createTarget(hw, hh, false);
        r.draw({ view: [1, 0, 0, 0, 1, 0], geo, surround: [0, 0, 0], mode: 0, negToOut }, hw, hh, t);
        const px = r.readBytes(t);
        r.deleteTarget(t);
        publishHistogram(px);
        // Restore the on-screen image (drawing to the target doesn't touch it, but keep state tidy).
      }, 120);
    });
    return () => cancelAnimationFrame(raf);
  }, [proxy, params, rect?.x, rect?.y, rect?.w, rect?.h, geo, size, surround, compare, negToOut, contextVersion]); // eslint-disable-line react-hooks/exhaustive-deps

  // ----- interaction -----
  const toCanvas = (e: React.PointerEvent | React.WheelEvent) => {
    const b = wrapRef.current!.getBoundingClientRect();
    return [e.clientX - b.left, e.clientY - b.top] as const;
  };

  const pick = useCallback(
    (x: number, y: number) => {
      if (!rect || !geo || !proxy || !params || !picker) return;
      const ou = (x - rect.x) / rect.w;
      const ov = (y - rect.y) / rect.h;
      if (ou < 0 || ou > 1 || ov < 0 || ov > 1) return;
      const [su, sv] = applyAffine(geo, ou, ov);
      const rgb = sampleProxy(proxy, su, sv, picker === 'base' ? 4 : 2);
      if (picker === 'base') {
        edit((r) => {
          r.film.baseMode = 'manual';
          r.film.baseRgb = rgb;
        });
        useStore.getState().notify('Film base set from the sampled area.');
      } else {
        const off = neutralOffsetFor(params, rgb);
        edit((r) => {
          r.convert.neutralOffset = off;
        }, true, true);
        useStore.getState().notify('Neutral point set.');
      }
      set('picker', null);
    },
    [rect, geo, proxy, params, picker, edit, set],
  );

  const onPointerDown = (e: React.PointerEvent) => {
    const [x, y] = toCanvas(e);
    if (picker) return pick(x, y);
    if (cropping) return;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    dragRef.current = { x, y, pan };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d || zoom === 'fit') return;
    const [x, y] = toCanvas(e);
    set('pan', [d.pan[0] + x - d.x, d.pan[1] + y - d.y]);
  };
  const onPointerUp = () => {
    dragRef.current = null;
  };
  const onWheel = (e: React.WheelEvent) => {
    if (!rect || cropping) return;
    const [x, y] = toCanvas(e);
    const z0 = zoom === 'fit' ? 1 : zoom;
    const z1 = Math.min(8, Math.max(1, z0 * Math.exp(-e.deltaY * 0.0015)));
    if (z1 === 1) {
      set('zoom', 'fit');
      set('pan', [0, 0]);
      return;
    }
    // Keep the point under the cursor fixed.
    const k = z1 / z0;
    const cx = size[0] / 2;
    const cy = PAD + (size[1] - PAD * 2 - CAPTION) / 2;
    set('zoom', z1);
    set('pan', [(pan[0] - (x - cx)) * k + (x - cx), (pan[1] - (y - cy)) * k + (y - cy)]);
  };
  const toggleZoom = () => {
    if (zoom === 'fit') set('zoom', 2.5);
    else {
      set('zoom', 'fit');
      set('pan', [0, 0]);
    }
  };

  const caption = (() => {
    if (!frame) return '';
    const parts: string[] = [];
    const idx = useStore.getState().frames.findIndex((f) => f.id === frame.id) + 1;
    parts.push(`Frame ${idx}`);
    const t = frame.recipe.film.type;
    const film = t === 'bw' ? 'Black & white' : t === 'e6' ? 'Slide' : t === 'photo' ? 'Photo' : 'Colour negative';
    parts.push(film);
    if (frame.meta) parts.push(`${frame.meta.fullWidth} × ${frame.meta.fullHeight}`);
    if (frame.meta?.exif.model) parts.push(frame.meta.exif.model);
    return parts.join('  ·  ');
  })();

  const classes = ['viewer', picker ? 'picking' : '', dragRef.current ? 'panning' : ''].join(' ');
  return (
    <div
      ref={wrapRef}
      className={classes}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onWheel={onWheel}
      onDoubleClick={() => !cropping && !picker && toggleZoom()}
    >
      <canvas ref={canvasRef} />
      {rect && frameRule && !cropping && zoom === 'fit' && (
        <div className="frame-rule" style={{ left: rect.x - 14, top: rect.y - 14, width: rect.w + 28, height: rect.h + 28 }} />
      )}
      {rect && zoom === 'fit' && (
        <div className="caption" style={{ top: rect.y + rect.h + (frameRule ? 24 : 12) }}>
          {caption}
        </div>
      )}
      {rect && cropping && frame && <CropOverlay rect={rect} />}
      {compare && <div className="viewer-badge">Negative — release \ to compare</div>}
      {picker && <div className="viewer-badge">{picker === 'base' ? 'Click unexposed film (rebate between frames)' : 'Click something that should be neutral grey'}</div>}
      {frame?.status === 'error' && <div className="viewer-msg error">{frame.error}</div>}
      {!glError && frame && !proxy && frame.status !== 'error' && <div className="viewer-msg">{loading || frame.status !== 'ready' ? 'Developing…' : ''}</div>}
      {glError && <div className="viewer-msg error">{glError}</div>}
      {rect && !cropping && (
        <div className="viewer-zoom" onPointerDown={(e) => e.stopPropagation()}>
          <button type="button" className="btn small" onClick={toggleZoom} title="Double-click the image also toggles zoom">
            {zoom === 'fit' ? 'Fit' : `${Math.round((zoom as number) * 100)}%`}
          </button>
        </div>
      )}
    </div>
  );
}
