import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FILM_LOOKS, findLook } from '../../engine/pipeline/looks';
import { buildPhotoParams } from '../../engine/pipeline/params';
import { Renderer } from '../../engine/gpu/renderer';
import { IDENTITY } from '../../engine/color/matrices';
import type { Affine } from '../../engine/geometry/transform';
import { useStore } from '../store/rollStore';

type Facing = 'environment' | 'user';
interface Shot {
  file: File;
  url: string;
  width: number;
  height: number;
}

const MAX_EDGE = 4096;
const PREFS_KEY = 'orpheus-camera';

function loadPrefs(): { lookId: string | null; strength: number; grain: boolean } {
  try {
    const raw = JSON.parse(localStorage.getItem(PREFS_KEY) ?? 'null');
    if (raw && typeof raw === 'object') return { lookId: raw.lookId ?? 'portra400', strength: raw.strength ?? 1, grain: raw.grain ?? true };
  } catch {
    /* private mode etc. */
  }
  return { lookId: 'portra400', strength: 1, grain: true };
}

/** Canvas uv → image uv so the image covers the canvas (like object-fit: cover). */
function coverView(cw: number, ch: number, iw: number, ih: number): Affine {
  const s = Math.max(cw / iw, ch / ih);
  const w = iw * s;
  const h = ih * s;
  return [cw / w, 0, -(cw - w) / 2 / w, 0, ch / h, -(ch - h) / 2 / h];
}

const MIRROR: Affine = [-1, 0, 1, 0, 1, 0];
const STRAIGHT: Affine = [1, 0, 0, 0, 1, 0];

/**
 * A simple film camera: live preview with a film look, shutter, and Save to Photos
 * through the share sheet (the only way a web page can reach the camera roll).
 */
export function CameraView() {
  const set = useStore((s) => s.set);
  const hasFrames = useStore((s) => s.frames.length > 0);
  const [prefs, setPrefs] = useState(loadPrefs);
  const [facing, setFacing] = useState<Facing>('environment');
  const [error, setError] = useState<string | null>(null);
  const [shot, setShot] = useState<Shot | null>(null);
  const [busy, setBusy] = useState(false);
  const [flash, setFlash] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rendererRef = useRef<Renderer | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    try {
      localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
    } catch {
      /* ignore */
    }
  }, [prefs]);

  const params = useMemo(
    () => buildPhotoParams({ lookId: prefs.lookId, strength: prefs.strength, grain: prefs.grain ? 0.6 : 0 }),
    [prefs],
  );
  const paramsRef = useRef(params);
  paramsRef.current = params;

  // ----- camera stream -----
  useEffect(() => {
    let cancelled = false;
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('This browser cannot open the camera. You can still choose a photo from your library.');
      return;
    }
    navigator.mediaDevices
      .getUserMedia({ audio: false, video: { facingMode: facing, width: { ideal: MAX_EDGE }, height: { ideal: MAX_EDGE * 0.75 } } })
      .then(async (stream) => {
        if (cancelled) return stream.getTracks().forEach((t) => t.stop());
        streamRef.current = stream;
        const v = videoRef.current!;
        v.srcObject = stream;
        await v.play().catch(() => undefined);
      })
      .catch((e: DOMException) => {
        setError(
          e.name === 'NotAllowedError'
            ? 'Camera access was blocked. Allow it in your browser settings, or choose a photo from your library.'
            : `Could not open the camera (${e.message || e.name}). You can still choose a photo from your library.`,
        );
      });
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, [facing]);

  // ----- live preview -----
  useEffect(() => {
    const canvas = canvasRef.current!;
    let r: Renderer;
    try {
      r = new Renderer(canvas);
      rendererRef.current = r;
    } catch (e) {
      setError(`This device can't run the live preview: ${(e as Error).message}`);
      return;
    }
    let raf = 0;
    let frame = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const v = videoRef.current;
      if (!v || v.readyState < 2 || !v.videoWidth || shot) return;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const W = Math.round(canvas.clientWidth * dpr);
      const H = Math.round(canvas.clientHeight * dpr);
      if (canvas.width !== W || canvas.height !== H) {
        canvas.width = W;
        canvas.height = H;
      }
      r.setSourceImage(v, v.videoWidth, v.videoHeight);
      // Re-seed grain every few frames so it shimmers like a viewfinder, not a static mask.
      r.setParams({ ...paramsRef.current, grainSeed: (frame++ >> 2) % 997 });
      r.draw(
        { view: coverView(W, H, v.videoWidth, v.videoHeight), geo: facing === 'user' ? MIRROR : STRAIGHT, surround: [0, 0, 0], mode: 0, negToOut: IDENTITY },
        W,
        H,
      );
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      r.dispose();
      rendererRef.current = null;
    };
  }, [facing, shot]);

  // ----- full-resolution render of a captured or chosen image -----
  const develop = useCallback(
    async (source: ImageBitmap, mirror: boolean) => {
      const s = Math.min(1, MAX_EDGE / Math.max(source.width, source.height));
      const w = Math.round(source.width * s);
      const h = Math.round(source.height * s);
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const r = new Renderer(canvas, { preserveDrawingBuffer: true });
      try {
        r.setSourceImage(source, source.width, source.height);
        r.setParams({ ...paramsRef.current, grainSeed: (Math.random() * 1e9) >>> 0 });
        r.draw({ view: STRAIGHT, geo: mirror ? MIRROR : STRAIGHT, surround: [0, 0, 0], mode: 0, negToOut: IDENTITY }, w, h);
        const blob = await new Promise<Blob | null>((res) => canvas.toBlob(res, 'image/jpeg', 0.92));
        if (!blob) throw new Error('Could not encode the photo.');
        const look = findLook(prefs.lookId);
        const stamp = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15);
        const file = new File([blob], `orpheus-${look ? look.id : 'photo'}-${stamp}.jpg`, { type: 'image/jpeg' });
        setShot({ file, url: URL.createObjectURL(blob), width: w, height: h });
      } finally {
        r.dispose();
        source.close();
      }
    },
    [prefs.lookId],
  );

  const capture = async () => {
    const v = videoRef.current;
    const track = streamRef.current?.getVideoTracks()[0];
    if (!v || !track || busy) return;
    setBusy(true);
    setFlash(true);
    setTimeout(() => setFlash(false), 140);
    try {
      let bmp: ImageBitmap | null = null;
      // Full-resolution still where supported (Chrome on Android); otherwise the video frame.
      const IC = (window as unknown as { ImageCapture?: new (t: MediaStreamTrack) => { takePhoto(): Promise<Blob> } }).ImageCapture;
      if (IC) {
        try {
          bmp = await createImageBitmap(await new IC(track).takePhoto());
        } catch {
          bmp = null;
        }
      }
      bmp ??= await createImageBitmap(v);
      await develop(bmp, facing === 'user');
    } catch (e) {
      useStore.getState().notify(`Capture failed: ${(e as Error).message}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  const fromLibrary = async (file: File) => {
    setBusy(true);
    try {
      await develop(await createImageBitmap(file, { imageOrientation: 'from-image' }), false);
    } catch (e) {
      useStore.getState().notify(`Could not open that photo: ${(e as Error).message}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    if (!shot) return;
    const nav = navigator as Navigator & { canShare?: (d: ShareData) => boolean };
    if (nav.canShare?.({ files: [shot.file] })) {
      try {
        await nav.share({ files: [shot.file], title: 'Orpheus photo' });
        return;
      } catch (e) {
        if ((e as DOMException).name === 'AbortError') return;
      }
    }
    const a = document.createElement('a');
    a.href = shot.url;
    a.download = shot.file.name;
    a.click();
  };

  const retake = () => {
    if (shot) URL.revokeObjectURL(shot.url);
    setShot(null);
  };

  const close = () => {
    retake();
    set('view', hasFrames ? 'library' : 'edit');
  };

  const looks = [null, ...FILM_LOOKS];
  const current = findLook(prefs.lookId);

  return (
    <div className="camera">
      <video ref={videoRef} playsInline muted autoPlay className="camera-video" />
      <canvas ref={canvasRef} className="camera-preview" style={{ visibility: shot ? 'hidden' : 'visible' }} />
      {flash && <div className="camera-flash" />}

      <div className="camera-top">
        <button type="button" className="cam-icon" onClick={close} aria-label="Close camera">
          ✕
        </button>
        <span className="camera-title">{current ? current.name : 'No film look'}</span>
        <button type="button" className={`cam-pill${prefs.grain ? ' on' : ''}`} onClick={() => setPrefs((p) => ({ ...p, grain: !p.grain }))}>
          Grain
        </button>
      </div>

      {error && !shot && (
        <div className="camera-error">
          <p>{error}</p>
          <button type="button" className="btn primary" onClick={() => fileRef.current?.click()}>
            Choose a photo
          </button>
        </div>
      )}

      {shot ? (
        <div className="camera-review">
          <img src={shot.url} alt="Your photo with the film look applied" />
          <div className="camera-review-bar">
            <button type="button" className="btn" onClick={retake}>
              Retake
            </button>
            <button type="button" className="btn primary" onClick={save}>
              Save to Photos
            </button>
          </div>
          <p className="camera-hint">On iPhone choose “Save Image” in the share sheet.</p>
        </div>
      ) : (
        <div className="camera-bottom">
          {current && (
            <label className="camera-strength">
              <span>Strength</span>
              <input
                type="range"
                min={0}
                max={1.5}
                step={0.05}
                value={prefs.strength}
                onChange={(e) => setPrefs((p) => ({ ...p, strength: Number(e.target.value) }))}
                aria-label="Look strength"
              />
              <span className="camera-strength-val">{Math.round(prefs.strength * 100)}%</span>
            </label>
          )}
          <div className="camera-looks" role="radiogroup" aria-label="Film look">
            {looks.map((l) => (
              <button
                key={l?.id ?? 'none'}
                type="button"
                role="radio"
                aria-checked={prefs.lookId === (l?.id ?? null)}
                className={`cam-chip${prefs.lookId === (l?.id ?? null) ? ' on' : ''}`}
                onClick={() => setPrefs((p) => ({ ...p, lookId: l?.id ?? null }))}
              >
                {l ? l.name : 'None'}
              </button>
            ))}
          </div>
          <div className="camera-controls">
            <button type="button" className="cam-icon" onClick={() => fileRef.current?.click()} aria-label="Choose a photo from your library" title="Library">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                <rect x="3" y="5" width="18" height="14" rx="1" />
                <path d="M3 16l5-5 4 4 3-3 6 6" />
              </svg>
            </button>
            <button type="button" className="shutter" onClick={capture} disabled={busy || !!error} aria-label="Take photo">
              <span />
            </button>
            <button type="button" className="cam-icon" onClick={() => setFacing((f) => (f === 'environment' ? 'user' : 'environment'))} aria-label="Switch camera" title="Flip">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                <path d="M4 8h3l2-3h6l2 3h3v11H4z" />
                <path d="M9 13a3 3 0 015-2.2M15 13a3 3 0 01-5 2.2M14 9.5v1.5h-1.5M10 16.5V15h1.5" />
              </svg>
            </button>
          </div>
        </div>
      )}

      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = '';
          if (f) void fromLibrary(f);
        }}
      />
      {busy && <div className="camera-busy">Developing…</div>}
    </div>
  );
}
