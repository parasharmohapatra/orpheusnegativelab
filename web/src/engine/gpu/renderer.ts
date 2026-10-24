import type { LinearImage, Mat3, Vec3 } from '../types';
import type { Affine } from '../geometry/transform';
import { LMS_TO_WORKING, OKLAB_M2, OKLAB_M2_INV, WORKING_TO_LMS } from '../color/oklab';
import { LUT_SIZE } from '../pipeline/curves';
import type { RenderParams } from '../pipeline/params';
import fragSrc from './shaders/convert.frag.glsl?raw';
import vertSrc from './shaders/fullscreen.vert.glsl?raw';

type GLCanvas = HTMLCanvasElement | OffscreenCanvas;

export interface DrawOptions {
  /** Canvas/target uv → output uv. */
  view: Affine;
  /** Output uv → source uv (relative to the uploaded texture). */
  geo: Affine;
  surround: Vec3;
  /** 0 = converted positive, 1 = the negative as scanned. */
  mode: 0 | 1;
  /** Camera RGB → output primaries, used for mode 1. */
  negToOut: Mat3;
}

export interface FloatTarget {
  fb: WebGLFramebuffer;
  tex: WebGLTexture;
  width: number;
  height: number;
}

export class WebGLUnsupportedError extends Error {}

/**
 * Thin WebGL2 wrapper around the single conversion shader. Used on the main thread
 * for the live preview and inside the export worker (OffscreenCanvas) for full-res tiles.
 */
export class Renderer {
  readonly gl: WebGL2RenderingContext;
  readonly canFloatTarget: boolean;
  readonly maxTextureSize: number;
  private prog: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private src: WebGLTexture;
  private curves: WebGLTexture;
  private loc = new Map<string, WebGLUniformLocation | null>();
  private srcSize: [number, number] = [0, 0];
  private hasCurves = false;
  lost = false;

  constructor(canvas: GLCanvas, opts: { preserveDrawingBuffer?: boolean } = {}) {
    const gl = canvas.getContext('webgl2', {
      antialias: false,
      alpha: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: opts.preserveDrawingBuffer ?? false,
      powerPreference: 'high-performance',
    }) as WebGL2RenderingContext | null;
    if (!gl) throw new WebGLUnsupportedError('WebGL2 is not available in this browser.');
    this.gl = gl;
    this.canFloatTarget = !!gl.getExtension('EXT_color_buffer_float');
    this.maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.prog = this.link();
    this.vao = gl.createVertexArray()!;
    this.src = this.makeTexture(gl.LINEAR);
    this.curves = this.makeTexture(gl.NEAREST);
    gl.bindTexture(gl.TEXTURE_2D, this.curves);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, LUT_SIZE, 4, 0, gl.RED, gl.FLOAT, new Float32Array(LUT_SIZE * 4));
  }

  private link(): WebGLProgram {
    const gl = this.gl;
    const compile = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`Shader compile failed: ${gl.getShaderInfoLog(s)}`);
      return s;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vertSrc));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fragSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`Program link failed: ${gl.getProgramInfoLog(p)}`);
    return p;
  }

  private makeTexture(filter: number): WebGLTexture {
    const gl = this.gl;
    const t = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  private u(name: string): WebGLUniformLocation | null {
    if (!this.loc.has(name)) this.loc.set(name, this.gl.getUniformLocation(this.prog, name));
    return this.loc.get(name)!;
  }

  /** Upload a whole image, or the sub-rectangle [x, x+w) × [y, y+h) of it. */
  setSource(img: LinearImage, region?: { x: number; y: number; w: number; h: number }): void {
    const gl = this.gl;
    const r = region ?? { x: 0, y: 0, w: img.width, h: img.height };
    if (r.w > this.maxTextureSize || r.h > this.maxTextureSize) {
      throw new Error(`Image region ${r.w}×${r.h} exceeds the GPU texture limit (${this.maxTextureSize}).`);
    }
    gl.bindTexture(gl.TEXTURE_2D, this.src);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.UNPACK_ROW_LENGTH, img.width);
    gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, r.x);
    gl.pixelStorei(gl.UNPACK_SKIP_ROWS, r.y);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB16F, r.w, r.h, 0, gl.RGB, gl.FLOAT, img.data);
    gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0);
    gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, 0);
    gl.pixelStorei(gl.UNPACK_SKIP_ROWS, 0);
    this.srcSize = [r.w, r.h];
  }

  get sourceSize(): [number, number] {
    return this.srcSize;
  }

  setParams(p: RenderParams): void {
    const gl = this.gl;
    gl.useProgram(this.prog);
    const v3 = (n: string, v: ArrayLike<number>) => gl.uniform3f(this.u(n), v[0], v[1], v[2]);
    const m3 = (n: string, m: Mat3) => gl.uniformMatrix3fv(this.u(n), true, m);
    gl.uniform1i(this.u('u_filmType'), p.filmType);
    v3('u_logBase', p.logBase);
    v3('u_bwMix', p.bwMix);
    v3('u_dLo', p.dLo);
    v3('u_invRange', p.invRange);
    v3('u_mid', p.mid);
    v3('u_offset', p.offset);
    gl.uniform1f(this.u('u_sceneStops'), p.sceneStops);
    gl.uniform1f(this.u('u_hiStops'), p.hiStops);
    m3('u_camToWork', p.camToWork);
    gl.uniform1i(this.u('u_printMode'), p.printMode);
    gl.uniform2f(this.u('u_tone'), p.shadows, p.highlights);
    gl.uniform1i(this.u('u_preserveHue'), p.preserveHue ? 1 : 0);
    gl.uniform1f(this.u('u_sat'), p.saturation);
    gl.uniform1f(this.u('u_vib'), p.vibrance);
    gl.uniform1i(this.u('u_hslActive'), p.hslActive ? 1 : 0);
    gl.uniform1fv(this.u('u_hslHue'), p.hslHue);
    gl.uniform1fv(this.u('u_hslSat'), p.hslSat);
    gl.uniform1fv(this.u('u_hslLum'), p.hslLum);
    m3('u_workToLms', WORKING_TO_LMS);
    m3('u_m2', OKLAB_M2);
    m3('u_m2inv', OKLAB_M2_INV);
    m3('u_lmsToWork', LMS_TO_WORKING);
    m3('u_workToOut', p.workToOut);
    gl.uniform1i(this.u('u_outTransfer'), p.outTransfer);
    this.hasCurves = !!p.curves;
    gl.uniform1i(this.u('u_hasCurves'), this.hasCurves ? 1 : 0);
    gl.bindTexture(gl.TEXTURE_2D, this.curves);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    if (p.curves) gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, LUT_SIZE, 3, gl.RED, gl.FLOAT, p.curves);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 3, LUT_SIZE, 1, gl.RED, gl.FLOAT, p.paperLut);
  }

  /** Draw into the canvas (target = null) or a float target, at width × height. */
  draw(o: DrawOptions, width: number, height: number, target: FloatTarget | WebGLFramebuffer | null = null): void {
    const gl = this.gl;
    gl.useProgram(this.prog);
    const fb = target && 'fb' in target ? target.fb : target;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.viewport(0, 0, width, height);
    gl.uniform3f(this.u('u_view0'), o.view[0], o.view[1], o.view[2]);
    gl.uniform3f(this.u('u_view1'), o.view[3], o.view[4], o.view[5]);
    gl.uniform3f(this.u('u_geo0'), o.geo[0], o.geo[1], o.geo[2]);
    gl.uniform3f(this.u('u_geo1'), o.geo[3], o.geo[4], o.geo[5]);
    gl.uniform3f(this.u('u_surround'), o.surround[0], o.surround[1], o.surround[2]);
    gl.uniform1i(this.u('u_mode'), o.mode);
    gl.uniformMatrix3fv(this.u('u_negToOut'), true, o.negToOut);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.src);
    gl.uniform1i(this.u('u_src'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.curves);
    gl.uniform1i(this.u('u_curves'), 1);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  createTarget(width: number, height: number, float: boolean): FloatTarget {
    const gl = this.gl;
    const tex = this.makeTexture(gl.NEAREST);
    if (float) {
      if (!this.canFloatTarget) throw new Error('Float render targets are not supported on this device.');
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
    const fb = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (status !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`Render target incomplete (0x${status.toString(16)}).`);
    return { fb, tex, width, height };
  }

  deleteTarget(t: FloatTarget): void {
    this.gl.deleteFramebuffer(t.fb);
    this.gl.deleteTexture(t.tex);
  }

  /** Read a float target top row first, as RGBA floats. */
  readFloat(t: FloatTarget): Float32Array {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb);
    const buf = new Float32Array(t.width * t.height * 4);
    gl.readPixels(0, 0, t.width, t.height, gl.RGBA, gl.FLOAT, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return flipRows(buf, t.width, t.height, 4);
  }

  /** Read an 8-bit target top row first, as RGBA bytes. */
  readBytes(t: FloatTarget): Uint8Array {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb);
    const buf = new Uint8Array(t.width * t.height * 4);
    gl.readPixels(0, 0, t.width, t.height, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return flipRows(buf, t.width, t.height, 4);
  }

  dispose(): void {
    const gl = this.gl;
    gl.deleteTexture(this.src);
    gl.deleteTexture(this.curves);
    gl.deleteProgram(this.prog);
    gl.deleteVertexArray(this.vao);
  }
}

function flipRows<T extends Float32Array | Uint8Array>(buf: T, w: number, h: number, ch: number): T {
  const row = w * ch;
  const tmp = buf.slice(0, row) as T;
  for (let y = 0; y < h >> 1; y++) {
    const a = y * row;
    const b = (h - 1 - y) * row;
    tmp.set(buf.subarray(a, a + row));
    buf.copyWithin(a, b, b + row);
    buf.set(tmp, b);
  }
  return buf;
}
