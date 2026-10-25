export type Vec3 = [number, number, number];
/** Row-major 3×3 matrix. */
export type Mat3 = [number, number, number, number, number, number, number, number, number];

/** Interleaved linear-light RGB, three floats per pixel. */
export interface LinearImage {
  width: number;
  height: number;
  data: Float32Array;
}

export type InputKind = 'raw' | 'tiff' | 'bitmap';

/** Everything the decoder learns about a file besides its pixels. */
export interface SourceMeta {
  kind: InputKind;
  /** Full-resolution size before any downsampling. */
  fullWidth: number;
  fullHeight: number;
  /** Camera (or file) RGB → linear Rec.709 primaries. Rows sum to 1 for white-balanced data. */
  toRec709: Mat3;
  /** Default 90° rotation steps (clockwise) and mirror from the file's orientation. */
  orientation: { rot90: 0 | 1 | 2 | 3; flipH: boolean };
  /** Fraction of pixels at the clip level per channel, if known. */
  clipFraction?: Vec3;
  /** True for 8-bit sources, which can posterize when inverted. */
  lowPrecision: boolean;
  /** Camera daylight white balance (per-channel gain, green = 1) for RAW photos; [1,1,1] otherwise. */
  daylightWB?: Vec3;
  exif: {
    make?: string;
    model?: string;
    iso?: number;
    shutter?: number;
    aperture?: number;
    focalLength?: number;
    timestamp?: number;
  };
}

export interface DecodedFrame {
  meta: SourceMeta;
  /** Editing proxy, long edge ≤ PROXY_MAX. */
  proxy: LinearImage;
  /** Small image for statistics and thumbnails, long edge ≤ ANALYSIS_MAX. */
  analysis: LinearImage;
}

export const PROXY_MAX = 3000;
export const ANALYSIS_MAX = 512;
