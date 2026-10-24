import { describe, expect, it } from 'vitest';
import { fromArrayBuffer } from 'geotiff';
import { unzlibSync } from 'fflate';
import { makeIccProfile } from '../../src/engine/encode/iccProfile';
import { encodePng, encodeTiff, quantize, spliceJpegIcc } from '../../src/engine/encode/writers';
import { parseIcc } from '../../src/engine/decode/icc';
import { apply } from '../../src/engine/color/matrices';

function gradient(w: number, h: number): Float32Array {
  const rgb = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = (y * w + x) * 3;
    rgb[i] = x / (w - 1);
    rgb[i + 1] = y / (h - 1);
    rgb[i + 2] = 0.5;
  }
  return rgb;
}

describe('ICC profiles', () => {
  it('generates profiles our own parser reads back as the right primaries', () => {
    const srgb = parseIcc(makeIccProfile('srgb'))!;
    // sRGB → Rec.709 should be identity (within the s15Fixed16 precision).
    const white = apply(srgb.toRec709, [1, 1, 1]);
    white.forEach((v) => expect(v).toBeCloseTo(1, 3));
    expect(srgb.decode!(0.5)).toBeCloseTo(0.214, 3);
    const adobe = parseIcc(makeIccProfile('adobeRGB'))!;
    expect(adobe.decode!(0.5)).toBeCloseTo(Math.pow(0.5, 563 / 256), 4);
    // Adobe green is outside sRGB: it maps to a negative red in Rec.709.
    expect(apply(adobe.toRec709, [0, 1, 0])[0]).toBeLessThan(0);
  });
});

describe('TIFF writer', () => {
  it('round-trips 16-bit samples through geotiff with predictor + deflate', async () => {
    const w = 37;
    const h = 23;
    const samples = quantize(gradient(w, h), 16) as Uint16Array;
    const bytes = encodeTiff(samples, w, h, makeIccProfile('srgb'));
    const tiff = await fromArrayBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
    const img = await tiff.getImage();
    expect(img.getWidth()).toBe(w);
    expect(img.getHeight()).toBe(h);
    const back = (await img.readRasters({ interleave: true })) as unknown as Uint16Array;
    expect(Array.from(back)).toEqual(Array.from(samples));
  });
  it('round-trips 8-bit samples', async () => {
    const w = 64;
    const h = 5000; // several strips
    const samples = quantize(gradient(w, h), 8) as Uint8Array;
    const bytes = encodeTiff(samples, w, h, makeIccProfile('displayP3'));
    const img = await (await fromArrayBuffer(bytes.buffer as ArrayBuffer)).getImage();
    const back = (await img.readRasters({ interleave: true })) as unknown as Uint8Array;
    expect(back.length).toBe(samples.length);
    expect(Array.from(back.subarray(0, 3000))).toEqual(Array.from(samples.subarray(0, 3000)));
    expect(Array.from(back.subarray(-3000))).toEqual(Array.from(samples.subarray(-3000)));
  });
});

describe('PNG writer', () => {
  it('writes valid chunks whose IDAT un-filters to the samples', () => {
    const w = 9;
    const h = 4;
    const samples = quantize(gradient(w, h), 16) as Uint16Array;
    const png = encodePng(samples, w, h, makeIccProfile('srgb'), 'sRGB');
    expect(Array.from(png.subarray(0, 8))).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
    // Walk chunks.
    const v = new DataView(png.buffer);
    let o = 8;
    const types: string[] = [];
    let idat: Uint8Array | null = null;
    while (o < png.length) {
      const len = v.getUint32(o);
      const type = String.fromCharCode(...png.subarray(o + 4, o + 8));
      types.push(type);
      if (type === 'IDAT') idat = png.subarray(o + 8, o + 8 + len);
      o += 12 + len;
    }
    expect(types).toEqual(['IHDR', 'iCCP', 'IDAT', 'IEND']);
    const raw = unzlibSync(idat!);
    expect(raw.length).toBe((w * 6 + 1) * h);
    // Undo Paeth on the first row (no row above): plain Sub-like prediction from the left.
    const row0 = raw.subarray(1, 1 + w * 6);
    const out = new Uint8Array(w * 6);
    for (let x = 0; x < w * 6; x++) out[x] = (row0[x] + (x >= 6 ? out[x - 6] : 0)) & 0xff;
    const firstSample = (out[0] << 8) | out[1];
    expect(firstSample).toBe(samples[0]);
  });
});

describe('JPEG ICC splice', () => {
  it('inserts APP2 ICC_PROFILE after APP0', () => {
    const jpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x04, 0x4a, 0x46, 0xff, 0xd9]);
    const icc = makeIccProfile('srgb');
    const out = spliceJpegIcc(jpeg, icc);
    // SOI (2) + APP0 marker (2) + APP0 length 4 (incl. itself) = ICC starts at byte 8.
    expect(out[8]).toBe(0xff);
    expect(out[9]).toBe(0xe2);
    expect(String.fromCharCode(...out.subarray(12, 23))).toBe('ICC_PROFILE');
    expect(out.length).toBe(jpeg.length + icc.length + 18);
  });
});
