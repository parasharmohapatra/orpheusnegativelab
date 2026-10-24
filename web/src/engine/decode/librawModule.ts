/**
 * LibRaw (WebAssembly) RAW decoder.
 *
 * Decodes to LINEAR camera RGB: no white balance, no gamma, no auto-brightness,
 * no colour-space conversion and no rotation. Built from web/wasm/libraw (see
 * README there); the prebuilt module lives in public/wasm/ and is loaded lazily,
 * once per worker. Single-threaded; meant to run inside a Web Worker.
 */

export interface RawDecodeResult {
  width: number;
  height: number;
  /** Interleaved RGB, length w*h*3, linear, black-subtracted, scaled so 65535 = sensor white. */
  data: Uint16Array;
  /**
   * 9 values row-major, camera RGB -> linear sRGB (LibRaw rgb_cam). Like dcraw's,
   * it expects white-balanced camera RGB (rows sum to 1), e.g. after dividing out
   * the film base or multiplying by preMul.
   */
  rgbCam: number[];
  /** 9 values row-major (LibRaw cam_xyz, first 3 rows). */
  camXyz: number[];
  /** 4 daylight multipliers (LibRaw pre_mul; [3] mirrors [1] for 3-colour sensors). */
  preMul: number[];
  /** LibRaw sizes.flip (0, 3, 5, 6). The image is NOT rotated. */
  flip: number;
  /** Fraction of output pixels at the clip level, per channel. */
  clipFraction: [number, number, number];
  exif: {
    make?: string;
    model?: string;
    iso?: number;
    /** Seconds. */
    shutter?: number;
    /** f-number. */
    aperture?: number;
    /** Millimetres. */
    focalLength?: number;
    /** Milliseconds since the Unix epoch (camera clock, read as UTC). */
    timestamp?: number;
  };
}

export interface RawDecodeOptions {
  halfSize: boolean;
  /** LibRaw user_qual: 0 linear, 1 VNG, 2 PPG, 3 AHD (default), 4 DCB, 11 DHT, 12 AAHD. */
  quality?: number;
}

/** A LibRaw failure; `code` is LibRaw's error code (LIBRAW_* in libraw_const.h). */
export class LibRawError extends Error {
  readonly code: number;
  constructor(message: string, code: number) {
    super(message);
    this.name = 'LibRawError';
    this.code = code;
  }
}

interface LibRawWasm {
  HEAPU8: Uint8Array;
  HEAPU16: Uint16Array;
  HEAP32: Int32Array;
  HEAPU32: Uint32Array;
  HEAPF32: Float32Array;
  HEAPF64: Float64Array;
  UTF8ToString(ptr: number, maxBytes?: number): string;
  _malloc(size: number): number;
  _free(ptr: number): void;
  _orp_decode(buf: number, len: number, halfSize: number, quality: number): number;
  _orp_free(result: number): void;
  _orp_version(): number;
}

// Byte offsets into orp_result (web/wasm/libraw/shim.c).
const R = {
  timestamp: 0,
  status: 8,
  width: 12,
  height: 16,
  dataPtr: 20,
  flip: 24,
  rgbCam: 36,
  camXyz: 72,
  preMul: 108,
  clip: 124,
  iso: 136,
  shutter: 140,
  aperture: 144,
  focalLength: 148,
  make: 152,
  model: 216,
  error: 280,
} as const;

let modulePromise: Promise<LibRawWasm> | null = null;

function loadModule(): Promise<LibRawWasm> {
  if (!modulePromise) {
    // Absolute URL: Vite's dev server rewrites root-relative dynamic imports
    // (adds ?import) and refuses to serve public/ files that way.
    const url = new URL(`${import.meta.env.BASE_URL}wasm/libraw.js`, self.location.href).href;
    modulePromise = import(/* @vite-ignore */ url)
      .then((m: { default: () => Promise<LibRawWasm> }) => m.default())
      .catch((err: unknown) => {
        modulePromise = null; // allow a retry after a transient fetch failure
        throw err;
      });
  }
  return modulePromise;
}

export async function isRawSupported(): Promise<boolean> {
  try {
    await loadModule();
    return true;
  } catch {
    return false;
  }
}

export async function decodeRaw(buf: ArrayBuffer, opts: RawDecodeOptions): Promise<RawDecodeResult> {
  const M = await loadModule();
  const len = buf.byteLength;
  const inPtr = M._malloc(len);
  if (!inPtr) throw new LibRawError('Out of memory copying RAW file into WebAssembly heap', -100007);

  let res = 0;
  try {
    M.HEAPU8.set(new Uint8Array(buf), inPtr);
    res = M._orp_decode(inPtr, len, opts.halfSize ? 1 : 0, opts.quality ?? 3);
  } finally {
    M._free(inPtr);
  }
  if (!res) throw new LibRawError('Out of memory allocating decode result', -100007);

  try {
    // Heap views can be replaced when memory grows, so always read through M.
    const i32 = (off: number) => M.HEAP32[(res + off) >> 2];
    const f32 = (off: number, n: number) => {
      const i = (res + off) >> 2;
      return Array.from(M.HEAPF32.subarray(i, i + n));
    };
    const str = (off: number, max: number) => M.UTF8ToString(res + off, max);

    const status = i32(R.status);
    if (status !== 0) throw new LibRawError(str(R.error, 256) || `LibRaw error ${status}`, status);

    const width = i32(R.width);
    const height = i32(R.height);
    const px = M.HEAPU32[(res + R.dataPtr) >> 2] >> 1;
    const data = M.HEAPU16.slice(px, px + width * height * 3); // copy out of the heap

    const clip = f32(R.clip, 3);
    const opt = (v: number) => (v > 0 && Number.isFinite(v) ? v : undefined);
    const make = str(R.make, 64);
    const model = str(R.model, 64);
    const ts = M.HEAPF64[(res + R.timestamp) >> 3];

    return {
      width,
      height,
      data,
      rgbCam: f32(R.rgbCam, 9),
      camXyz: f32(R.camXyz, 9),
      preMul: f32(R.preMul, 4),
      flip: i32(R.flip),
      clipFraction: [clip[0], clip[1], clip[2]],
      exif: {
        make: make || undefined,
        model: model || undefined,
        iso: opt(f32(R.iso, 1)[0]),
        shutter: opt(f32(R.shutter, 1)[0]),
        aperture: opt(f32(R.aperture, 1)[0]),
        focalLength: opt(f32(R.focalLength, 1)[0]),
        timestamp: ts > 0 ? ts * 1000 : undefined,
      },
    };
  } finally {
    M._orp_free(res);
  }
}
