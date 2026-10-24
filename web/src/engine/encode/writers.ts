import { zlibSync } from 'fflate';

/** Encoded float RGB (0..1) → integer samples, with triangular dither for 8-bit. */
export function quantize(rgb: Float32Array, bits: 8 | 16): Uint8Array | Uint16Array {
  if (bits === 16) {
    const out = new Uint16Array(rgb.length);
    for (let i = 0; i < rgb.length; i++) out[i] = Math.round(Math.min(1, Math.max(0, rgb[i])) * 65535);
    return out;
  }
  const out = new Uint8Array(rgb.length);
  // Cheap deterministic hash noise so exports are reproducible.
  let s = 0x9e3779b9;
  const rnd = () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
  for (let i = 0; i < rgb.length; i++) {
    const v = Math.min(1, Math.max(0, rgb[i])) * 255 + (rnd() - rnd()) * 0.5;
    out[i] = Math.min(255, Math.max(0, Math.round(v)));
  }
  return out;
}

/** Add a white (or any colour) border around encoded float RGB. */
export function addBorder(rgb: Float32Array, w: number, h: number, px: number, value = 1): { data: Float32Array; w: number; h: number } {
  const W = w + px * 2;
  const H = h + px * 2;
  const out = new Float32Array(W * H * 3).fill(value);
  for (let y = 0; y < h; y++) out.set(rgb.subarray(y * w * 3, (y + 1) * w * 3), ((y + px) * W + px) * 3);
  return { data: out, w: W, h: H };
}

// ---------- PNG ----------

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bufs: Uint8Array[]): number {
  let c = 0xffffffff;
  for (const b of bufs) for (let i = 0; i < b.length; i++) c = CRC_TABLE[(c ^ b[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const v = new DataView(out.buffer);
  v.setUint32(0, data.length);
  const t = new Uint8Array([...type].map((c) => c.charCodeAt(0)));
  out.set(t, 4);
  out.set(data, 8);
  v.setUint32(8 + data.length, crc32([t, data]));
  return out;
}

/** RGB PNG, 8 or 16 bits per sample, with an embedded ICC profile. */
export function encodePng(samples: Uint8Array | Uint16Array, w: number, h: number, icc: Uint8Array, iccName: string): Uint8Array {
  const bits = samples instanceof Uint16Array ? 16 : 8;
  const bpp = bits === 16 ? 6 : 3;
  const stride = w * bpp;
  // Big-endian bytes per row, then Paeth filter.
  const raw = new Uint8Array(stride * h);
  if (bits === 16) {
    for (let i = 0; i < samples.length; i++) {
      raw[i * 2] = samples[i] >> 8;
      raw[i * 2 + 1] = samples[i] & 0xff;
    }
  } else {
    raw.set(samples as Uint8Array);
  }
  const filtered = new Uint8Array((stride + 1) * h);
  for (let y = 0; y < h; y++) {
    const o = y * (stride + 1);
    filtered[o] = 4;
    for (let x = 0; x < stride; x++) {
      const i = y * stride + x;
      const a = x >= bpp ? raw[i - bpp] : 0;
      const b = y > 0 ? raw[i - stride] : 0;
      const c = x >= bpp && y > 0 ? raw[i - stride - bpp] : 0;
      const p = a + b - c;
      const pa = Math.abs(p - a);
      const pb = Math.abs(p - b);
      const pc = Math.abs(p - c);
      const pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      filtered[o + 1 + x] = (raw[i] - pred) & 0xff;
    }
  }
  const ihdr = new Uint8Array(13);
  const v = new DataView(ihdr.buffer);
  v.setUint32(0, w);
  v.setUint32(4, h);
  ihdr[8] = bits;
  ihdr[9] = 2; // truecolour
  const name = new TextEncoder().encode(iccName.slice(0, 79));
  const iccp = new Uint8Array(name.length + 2 + 0);
  iccp.set(name);
  const iccData = concat([iccp, zlibSync(icc)]);
  const parts = [
    new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', ihdr),
    chunk('iCCP', iccData),
    chunk('IDAT', zlibSync(filtered, { level: 6 })),
    chunk('IEND', new Uint8Array(0)),
  ];
  return concat(parts);
}

export function concat(parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

// ---------- TIFF ----------

/** Baseline RGB TIFF, 8 or 16 bit, Deflate with horizontal predictor, ICC embedded. */
export function encodeTiff(samples: Uint8Array | Uint16Array, w: number, h: number, icc: Uint8Array): Uint8Array {
  const bits = samples instanceof Uint16Array ? 16 : 8;
  const rowsPerStrip = Math.max(1, Math.floor((256 * 1024) / (w * 3 * (bits / 8))));
  const strips: Uint8Array[] = [];
  for (let y0 = 0; y0 < h; y0 += rowsPerStrip) {
    const rows = Math.min(rowsPerStrip, h - y0);
    const n = rows * w * 3;
    if (bits === 16) {
      const s = (samples as Uint16Array).slice(y0 * w * 3, y0 * w * 3 + n);
      for (let r = 0; r < rows; r++) for (let x = w - 1; x > 0; x--) for (let c = 0; c < 3; c++) {
        const i = (r * w + x) * 3 + c;
        s[i] = (s[i] - s[i - 3]) & 0xffff;
      }
      strips.push(zlibSync(new Uint8Array(s.buffer), { level: 6 }));
    } else {
      const s = (samples as Uint8Array).slice(y0 * w * 3, y0 * w * 3 + n);
      for (let r = 0; r < rows; r++) for (let x = w - 1; x > 0; x--) for (let c = 0; c < 3; c++) {
        const i = (r * w + x) * 3 + c;
        s[i] = (s[i] - s[i - 3]) & 0xff;
      }
      strips.push(zlibSync(s, { level: 6 }));
    }
  }

  type Entry = { tag: number; type: number; values: number[] | Uint8Array };
  const SHORT = 3, LONG = 4, RATIONAL = 5, UNDEFINED = 7, ASCII = 2;
  const software = new TextEncoder().encode('Orpheus Negative Lab\0');
  const entries: Entry[] = [
    { tag: 256, type: LONG, values: [w] },
    { tag: 257, type: LONG, values: [h] },
    { tag: 258, type: SHORT, values: [bits, bits, bits] },
    { tag: 259, type: SHORT, values: [8] },
    { tag: 262, type: SHORT, values: [2] },
    { tag: 273, type: LONG, values: strips.map(() => 0) },
    { tag: 277, type: SHORT, values: [3] },
    { tag: 278, type: LONG, values: [rowsPerStrip] },
    { tag: 279, type: LONG, values: strips.map((s) => s.length) },
    { tag: 282, type: RATIONAL, values: [300, 1] },
    { tag: 283, type: RATIONAL, values: [300, 1] },
    { tag: 284, type: SHORT, values: [1] },
    { tag: 296, type: SHORT, values: [2] },
    { tag: 305, type: ASCII, values: software },
    { tag: 317, type: SHORT, values: [2] },
    { tag: 34675, type: UNDEFINED, values: icc },
  ];
  const typeSize = (t: number) => (t === SHORT ? 2 : t === LONG ? 4 : t === RATIONAL ? 8 : 1);
  const count = (e: Entry) => (e.type === RATIONAL ? e.values.length / 2 : e.values.length);
  const byteLen = (e: Entry) => count(e) * typeSize(e.type);

  const ifdOffset = 8;
  const ifdSize = 2 + entries.length * 12 + 4;
  let extra = ifdOffset + ifdSize;
  const extraOffsets = new Map<number, number>();
  for (const e of entries) {
    if (byteLen(e) > 4) {
      extraOffsets.set(e.tag, extra);
      extra += byteLen(e) + (byteLen(e) & 1);
    }
  }
  let dataOffset = extra;
  const stripOffsets = strips.map((s) => {
    const o = dataOffset;
    dataOffset += s.length;
    return o;
  });
  entries.find((e) => e.tag === 273)!.values = stripOffsets;

  const out = new Uint8Array(dataOffset);
  const v = new DataView(out.buffer);
  out.set([0x49, 0x49, 42, 0]);
  v.setUint32(4, ifdOffset, true);
  v.setUint16(ifdOffset, entries.length, true);
  const writeValues = (e: Entry, at: number) => {
    if (e.values instanceof Uint8Array) {
      out.set(e.values, at);
      return;
    }
    e.values.forEach((val, i) => {
      if (e.type === SHORT) v.setUint16(at + i * 2, val, true);
      else v.setUint32(at + i * 4, val, true);
    });
  };
  entries.forEach((e, i) => {
    const o = ifdOffset + 2 + i * 12;
    v.setUint16(o, e.tag, true);
    v.setUint16(o + 2, e.type, true);
    v.setUint32(o + 4, count(e), true);
    const ext = extraOffsets.get(e.tag);
    if (ext !== undefined) {
      v.setUint32(o + 8, ext, true);
      writeValues(e, ext);
    } else {
      writeValues(e, o + 8);
    }
  });
  v.setUint32(ifdOffset + 2 + entries.length * 12, 0, true);
  strips.forEach((s, i) => out.set(s, stripOffsets[i]));
  return out;
}

// ---------- JPEG ----------

/** Insert an ICC profile (APP2 ICC_PROFILE segments) right after SOI / APP0. */
export function spliceJpegIcc(jpeg: Uint8Array, icc: Uint8Array): Uint8Array {
  let pos = 2;
  // Skip an existing APP0 (JFIF) so the ICC follows it, as most writers do.
  if (jpeg[pos] === 0xff && jpeg[pos + 1] === 0xe0) pos += 2 + ((jpeg[pos + 2] << 8) | jpeg[pos + 3]);
  const sig = new TextEncoder().encode('ICC_PROFILE\0');
  const maxChunk = 65519 - sig.length - 2;
  const n = Math.ceil(icc.length / maxChunk);
  const segs: Uint8Array[] = [];
  for (let i = 0; i < n; i++) {
    const part = icc.subarray(i * maxChunk, (i + 1) * maxChunk);
    const len = 2 + sig.length + 2 + part.length;
    const seg = new Uint8Array(2 + len);
    seg[0] = 0xff;
    seg[1] = 0xe2;
    seg[2] = len >> 8;
    seg[3] = len & 0xff;
    seg.set(sig, 4);
    seg[4 + sig.length] = i + 1;
    seg[5 + sig.length] = n;
    seg.set(part, 6 + sig.length);
    segs.push(seg);
  }
  return concat([jpeg.subarray(0, pos), ...segs, jpeg.subarray(pos)]);
}

/** JPEG via the browser encoder (8-bit), then our ICC profile spliced in. */
export async function encodeJpeg(samples: Uint8Array, w: number, h: number, quality: number, icc: Uint8Array): Promise<Uint8Array> {
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d', { colorSpace: 'srgb' })!;
  const img = ctx.createImageData(w, h);
  for (let i = 0, j = 0; i < samples.length; i += 3, j += 4) {
    img.data[j] = samples[i];
    img.data[j + 1] = samples[i + 1];
    img.data[j + 2] = samples[i + 2];
    img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality });
  return spliceJpegIcc(new Uint8Array(await blob.arrayBuffer()), icc);
}
