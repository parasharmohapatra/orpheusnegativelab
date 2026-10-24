import type { Mat3, Vec3 } from '../types';
import { bradford, invert, mul, rgbToXYZ, PRIMARIES, xyToXYZ, D50, D65 } from '../color/matrices';

/** What we need from an embedded matrix/TRC ICC profile. */
export interface IccInfo {
  /** Linear file RGB → linear Rec.709 (D65). */
  toRec709: Mat3;
  /** Encoded sample (0..1) → linear, or null if the TRC is linear. */
  decode: ((v: number) => number) | null;
}

function s15(view: DataView, off: number): number {
  return view.getInt32(off) / 65536;
}

/**
 * Parse an RGB matrix/TRC ICC profile (what scanners and photo apps embed).
 * Returns null for anything we can't interpret (LUT-based profiles, CMYK, …).
 */
export function parseIcc(bytes: Uint8Array): IccInfo | null {
  try {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const tag = (sig: string) => {
      const count = view.getUint32(128);
      for (let i = 0; i < count; i++) {
        const o = 132 + i * 12;
        const s = String.fromCharCode(bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]);
        if (s === sig) return { off: view.getUint32(o + 4), len: view.getUint32(o + 8) };
      }
      return null;
    };
    const xyz = (sig: string): Vec3 | null => {
      const t = tag(sig);
      return t ? [s15(view, t.off + 8), s15(view, t.off + 12), s15(view, t.off + 16)] : null;
    };
    const r = xyz('rXYZ');
    const g = xyz('gXYZ');
    const b = xyz('bXYZ');
    if (!r || !g || !b) return null;
    // Colorants are in the D50 PCS; adapt to D65 then go to Rec.709.
    const toXYZ50: Mat3 = [r[0], g[0], b[0], r[1], g[1], b[1], r[2], g[2], b[2]];
    const toXYZ65 = mul(bradford(xyToXYZ(...D50), xyToXYZ(...D65)), toXYZ50);
    const toRec709 = mul(invert(rgbToXYZ(PRIMARIES.rec709)), toXYZ65);

    let decode: IccInfo['decode'] = null;
    const trc = tag('gTRC') ?? tag('rTRC');
    if (trc) {
      const type = String.fromCharCode(...bytes.subarray(trc.off, trc.off + 4));
      if (type === 'curv') {
        const n = view.getUint32(trc.off + 8);
        if (n === 1) {
          const gamma = view.getUint16(trc.off + 12) / 256;
          decode = (v) => Math.pow(v, gamma);
        } else if (n > 1) {
          const table = new Float32Array(n);
          for (let i = 0; i < n; i++) table[i] = view.getUint16(trc.off + 12 + i * 2) / 65535;
          decode = (v) => {
            const f = v * (n - 1);
            const i = Math.min(n - 2, Math.floor(f));
            return table[i] + (table[i + 1] - table[i]) * (f - i);
          };
        }
      } else if (type === 'para') {
        const fn = view.getUint16(trc.off + 8);
        const p = Array.from({ length: 7 }, (_, i) => s15(view, trc.off + 12 + i * 4));
        const [G, a, bb, c, d, e, f] = p;
        decode = (x) => {
          switch (fn) {
            case 0: return Math.pow(x, G);
            case 1: return x >= -bb / a ? Math.pow(a * x + bb, G) : 0;
            case 2: return x >= -bb / a ? Math.pow(a * x + bb, G) + c : c;
            case 3: return x >= d ? Math.pow(a * x + bb, G) : c * x;
            default: return x >= d ? Math.pow(a * x + bb, G) + e : c * x + f;
          }
        };
      }
    }
    return { toRec709, decode };
  } catch {
    return null;
  }
}
