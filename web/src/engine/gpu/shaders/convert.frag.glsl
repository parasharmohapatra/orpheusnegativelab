#version 300 es
// Per-pixel negative → positive conversion. Mirrors src/engine/pipeline/reference.ts
// line for line; keep them in sync (the GPU parity self-test compares the two).
precision highp float;
precision highp int;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_src;
uniform sampler2D u_curves;   // R32F, 1024 × 4: rows 0–2 RGB curves, row 3 paper curve

// Canvas uv → output uv, then output uv → source uv (affine rows).
uniform vec3 u_view0, u_view1;
uniform vec3 u_geo0, u_geo1;
uniform vec3 u_surround;      // encoded colour outside the frame
uniform int u_mode;           // 0 = positive, 1 = raw negative

uniform int u_filmType;       // 0 C-41, 1 B&W, 2 E-6
uniform vec3 u_logBase, u_bwMix, u_dLo, u_invRange, u_mid, u_offset;
uniform float u_sceneStops, u_hiStops;
uniform mat3 u_camToWork;
uniform mat3 u_negToOut;      // for u_mode 1: camera → output primaries
uniform int u_printMode;
uniform vec2 u_tone;          // shadows, highlights
uniform bool u_preserveHue;
uniform float u_sat, u_vib;
uniform bool u_hslActive;
uniform float u_hslHue[8], u_hslSat[8], u_hslLum[8];
uniform mat3 u_workToLms, u_m2, u_m2inv, u_lmsToWork, u_workToOut;
uniform int u_outTransfer;    // 0 sRGB, 1 γ2.2, 2 linear, 3 γ1.8
uniform bool u_hasCurves;

const float LOG2_018 = -2.4739311883324122;
const float PAPER_X_MIN = -8.0;
const float PAPER_X_MAX = 6.0;
const float HSL_CENTRES[8] = float[8](25.0, 55.0, 100.0, 140.0, 195.0, 250.0, 290.0, 330.0);

float srgbDecode(float x) {
  return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}

float encodeT(float v, int t) {
  v = max(v, 0.0);
  if (t == 0) return v <= 0.0031308 ? 12.92 * v : 1.055 * pow(v, 1.0 / 2.4) - 0.055;
  if (t == 1) return pow(v, 1.0 / 2.19921875);
  if (t == 3) return pow(v, 1.0 / 1.8);
  return v;
}

float bend(float n, float m) {
  if (n < 0.0) return n * (1.0 + m);
  if (n > 1.0) return 1.0 + (n - 1.0) * (1.0 - m);
  return n + m * n * (1.0 - n);
}

float paper(float v) {
  float x = log2(max(v, 1e-10)) - LOG2_018;
  x += u_tone.x * 1.5 * exp(-(x + 3.0) * (x + 3.0) / 4.5) + u_tone.y * 1.5 * exp(-(x - 1.5) * (x - 1.5) / 4.5);
  if (u_printMode == 0) return 0.18 * exp2(x);
  float f = clamp((x - PAPER_X_MIN) / (PAPER_X_MAX - PAPER_X_MIN), 0.0, 1.0) * 1023.0;
  int i = min(1022, int(floor(f)));
  float t = f - float(i);
  float enc = mix(texelFetch(u_curves, ivec2(i, 3), 0).r, texelFetch(u_curves, ivec2(i + 1, 3), 0).r, t);
  return srgbDecode(enc);
}

float cbrtS(float x) { return sign(x) * pow(abs(x), 1.0 / 3.0); }

vec3 hslAdjust(float hueDeg) {
  int i = 7;
  for (int k = 0; k < 8; k++) if (hueDeg >= HSL_CENTRES[k]) i = k;
  int j = (i + 1) % 8;
  float span = HSL_CENTRES[j] - HSL_CENTRES[i];
  if (span <= 0.0) span += 360.0;
  float d = hueDeg - HSL_CENTRES[i];
  if (d < 0.0) d += 360.0;
  float t = smoothstep(0.0, 1.0, d / span);
  return vec3(mix(u_hslHue[i], u_hslHue[j], t), mix(u_hslSat[i], u_hslSat[j], t), mix(u_hslLum[i], u_hslLum[j], t));
}

float curve(float v, int ch) {
  float f = v * 1023.0;
  int i = min(1022, int(floor(f)));
  float t = f - float(i);
  float a = texelFetch(u_curves, ivec2(i, ch), 0).r;
  float b = texelFetch(u_curves, ivec2(i + 1, ch), 0).r;
  return mix(a, b, t);
}

vec3 convertPixel(vec3 rgb) {
  vec3 D = -log(max(rgb, vec3(1e-6))) / log(10.0) + u_logBase;
  if (u_filmType == 1) {
    float s = dot(u_bwMix, vec3(1.0));
    float d = dot(u_bwMix, D) / (s == 0.0 ? 1.0 : s);
    D = vec3(d);
  }
  vec3 n;
  for (int c = 0; c < 3; c++) n[c] = bend((D[c] - u_dLo[c]) * u_invRange[c], u_mid[c]) + u_offset[c];
  vec3 E = 0.18 * exp2(u_hiStops + u_sceneStops * (n - 1.0));
  vec3 W = u_camToWork * E;

  vec3 y;
  if (u_preserveHue) {
    float mx = max(max(W.r, W.g), max(W.b, 1e-10));
    y = W * (paper(mx) / mx);
  } else {
    y = vec3(paper(W.r), paper(W.g), paper(W.b));
  }

  if (u_sat != 0.0 || u_vib != 0.0 || u_hslActive) {
    vec3 lms = u_workToLms * y;
    lms = vec3(cbrtS(lms.x), cbrtS(lms.y), cbrtS(lms.z));
    vec3 lab = u_m2 * lms;
    float L = lab.x;
    float C = length(lab.yz);
    float h = degrees(atan(lab.z, lab.y));
    if (h < 0.0) h += 360.0;
    if (u_hslActive) {
      vec3 adj = hslAdjust(h);
      h += adj.x * 30.0;
      L += adj.z * 0.15 * smoothstep(0.01, 0.08, C);
      C *= 1.0 + adj.y;
    }
    C *= 1.0 + u_sat;
    C *= 1.0 + u_vib * (1.0 - smoothstep(0.0, 0.25, C));
    C = max(0.0, C);
    float hr = radians(h);
    vec3 l = u_m2inv * vec3(L, C * cos(hr), C * sin(hr));
    y = u_lmsToWork * (l * l * l);
  }

  vec3 o = clamp(u_workToOut * y, 0.0, 1.0);
  for (int c = 0; c < 3; c++) {
    float v = encodeT(o[c], u_outTransfer);
    if (u_hasCurves) v = curve(v, c);
    o[c] = v;
  }
  return o;
}

void main() {
  vec3 p = vec3(v_uv, 1.0);
  vec2 ouv = vec2(dot(u_view0, p), dot(u_view1, p));
  if (ouv.x < 0.0 || ouv.x > 1.0 || ouv.y < 0.0 || ouv.y > 1.0) {
    outColor = vec4(u_surround, 1.0);
    return;
  }
  vec3 q = vec3(ouv, 1.0);
  vec2 suv = vec2(dot(u_geo0, q), dot(u_geo1, q));
  if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) {
    outColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }
  vec3 rgb = texture(u_src, suv).rgb;
  if (u_mode == 1) {
    vec3 o = clamp(u_negToOut * rgb, 0.0, 1.0);
    outColor = vec4(encodeT(o.r, u_outTransfer), encodeT(o.g, u_outTransfer), encodeT(o.b, u_outTransfer), 1.0);
    return;
  }
  outColor = vec4(convertPixel(rgb), 1.0);
}
