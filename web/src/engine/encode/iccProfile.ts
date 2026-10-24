import type { Vec3 } from '../types';
import { apply, bradford, rgbToXYZ, xyToXYZ, D50 } from '../color/matrices';
import { OUTPUT_SPACES, type OutputSpaceId } from '../color/spaces';
import { srgbDecode, TRANSFER } from '../color/transfer';

/**
 * Generate a minimal ICC v2 matrix/TRC display profile for an output space, so we
 * don't have to ship (or license) third-party .icc files.
 */
export function makeIccProfile(id: OutputSpaceId): Uint8Array {
  const space = OUTPUT_SPACES[id];
  const toXYZ = rgbToXYZ(space.primaries);
  const adapt = bradford(xyToXYZ(...space.primaries.w), xyToXYZ(...D50));
  const col = (c: number): Vec3 => apply(adapt, [toXYZ[c], toXYZ[3 + c], toXYZ[6 + c]]);

  const tags: { sig: string; data: Uint8Array }[] = [];
  const desc = `Orpheus ${space.label}`;
  tags.push({ sig: 'desc', data: descTag(desc) });
  tags.push({ sig: 'cprt', data: textTag('No copyright, use freely') });
  tags.push({ sig: 'wtpt', data: xyzTag([0.9642, 1, 0.8249]) });
  tags.push({ sig: 'rXYZ', data: xyzTag(col(0)) });
  tags.push({ sig: 'gXYZ', data: xyzTag(col(1)) });
  tags.push({ sig: 'bXYZ', data: xyzTag(col(2)) });
  const trc = curveTag(space.transfer);
  tags.push({ sig: 'rTRC', data: trc }, { sig: 'gTRC', data: trc }, { sig: 'bTRC', data: trc });

  const tableSize = 4 + tags.length * 12;
  let offset = 128 + tableSize;
  const placed: { sig: string; off: number; data: Uint8Array }[] = [];
  const seen = new Map<Uint8Array, number>();
  for (const t of tags) {
    const prior = seen.get(t.data);
    if (prior !== undefined) {
      placed.push({ sig: t.sig, off: prior, data: t.data });
      continue;
    }
    placed.push({ sig: t.sig, off: offset, data: t.data });
    seen.set(t.data, offset);
    offset += pad4(t.data.length);
  }
  const size = offset;
  const out = new Uint8Array(size);
  const v = new DataView(out.buffer);
  v.setUint32(0, size);
  ascii(out, 4, 'none');
  v.setUint32(8, 0x02100000);
  ascii(out, 12, 'mntr');
  ascii(out, 16, 'RGB ');
  ascii(out, 20, 'XYZ ');
  const d = new Date();
  [d.getUTCFullYear(), d.getUTCMonth() + 1, d.getUTCDate(), d.getUTCHours(), d.getUTCMinutes(), d.getUTCSeconds()].forEach(
    (n, i) => v.setUint16(24 + i * 2, n),
  );
  ascii(out, 36, 'acsp');
  v.setInt32(68, s15(0.9642));
  v.setInt32(72, s15(1));
  v.setInt32(76, s15(0.8249));
  ascii(out, 80, 'ORPH');
  v.setUint32(128, tags.length);
  placed.forEach((t, i) => {
    ascii(out, 132 + i * 12, t.sig);
    v.setUint32(136 + i * 12, t.off);
    v.setUint32(140 + i * 12, t.data.length);
    out.set(t.data, t.off);
  });
  return out;
}

const pad4 = (n: number) => (n + 3) & ~3;
const s15 = (x: number) => Math.round(x * 65536);

function ascii(buf: Uint8Array, off: number, s: string): void {
  for (let i = 0; i < s.length; i++) buf[off + i] = s.charCodeAt(i);
}

function xyzTag(xyz: Vec3): Uint8Array {
  const b = new Uint8Array(20);
  const v = new DataView(b.buffer);
  ascii(b, 0, 'XYZ ');
  xyz.forEach((x, i) => v.setInt32(8 + i * 4, s15(x)));
  return b;
}

function textTag(s: string): Uint8Array {
  const b = new Uint8Array(8 + s.length + 1);
  ascii(b, 0, 'text');
  ascii(b, 8, s);
  return b;
}

function descTag(s: string): Uint8Array {
  const n = s.length + 1;
  const b = new Uint8Array(12 + n + 8 + 3 + 67);
  const v = new DataView(b.buffer);
  ascii(b, 0, 'desc');
  v.setUint32(8, n);
  ascii(b, 12, s);
  return b;
}

function curveTag(t: number): Uint8Array {
  if (t === TRANSFER.gamma22 || t === TRANSFER.gamma18) {
    const b = new Uint8Array(14);
    const v = new DataView(b.buffer);
    ascii(b, 0, 'curv');
    v.setUint32(8, 1);
    v.setUint16(12, t === TRANSFER.gamma22 ? 563 : 461);
    return b;
  }
  if (t === TRANSFER.linear) {
    const b = new Uint8Array(12);
    ascii(b, 0, 'curv');
    return b;
  }
  const n = 1024;
  const b = new Uint8Array(12 + n * 2);
  const v = new DataView(b.buffer);
  ascii(b, 0, 'curv');
  v.setUint32(8, n);
  for (let i = 0; i < n; i++) v.setUint16(12 + i * 2, Math.round(srgbDecode(i / (n - 1)) * 65535));
  return b;
}
