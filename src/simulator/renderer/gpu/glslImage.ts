import {
  CLINICAL_GREY_CURVE,
  HIGH_CONTRAST_GAMMA,
  LOG_FLOOR,
  LOG_KNEE_SPAN,
  LOG_KNEE_START,
  NOISE_MAGNITUDE_BINS,
  NOISE_PHASE_BINS,
  NOISE_RMS,
  NOISE_TAIL,
  REF_DB,
  S_CURVE_MIX,
} from '../postprocess/consolePipeline';
import { colorMapDefinesGlsl, colorMapGlsl } from '../postprocess/colorMap';
import { sriDefinesGlsl } from '../postprocess/sriFilter';
import { tintMapGlsl } from '../postprocess/tintMap';
import { psfDefinesGlsl } from '../acoustic/psf';
import { TRANS_K } from '../transmissionCode';
import { ST_POW_ALPHA_MIN } from '@/simulator/doppler/color/spatiotemporalFilter';

/**
 * GPU image chain after envelope detection (decision 54), mirroring the CPU code step by step:
 *  - receiver noise passes (consolePipeline.ts `receiverNoise`, decision 91): white complex noise from the same
 *    integer hash per line, sample, frame and seed, filtered along the beam and then across lines by the same
 *    kernel table the CPU builds (`buildNoiseKernels`), in two passes into a float texture.
 *  - log pass (`GLSL_LOG_FRAG`): evaluates |envelope + receiver noise| × compensation → log compression ONCE
 *    per sample into R16F (avoiding redundant sqrt/log calls during spatial filtering).
 *  - console pass (`GLSL_CONSOLE_FRAG`): structure-oriented anisotropic speckle reduction (SRI) → axial edge
 *    enhancement → persistence against history → grey map → packed RGBA8 (grey, structure, tissue, trans code).
 *  - present pass (`GLSL_PRESENT_FRAG`): integer bilinear gather of the scan-conversion LUT, plus complex-plane
 *    bilinear Color Doppler interpolation (aliasing-safe e^{i·π·v/scale}) with smooth jet edges & tissue priority.
 */
const glslFloat = (v: number): string => (Number.isInteger(v) ? `${v}.0` : `${v}`);

const GLSL_HASH = /* glsl */ `
// core/random.ts hash3: identical 32-bit integer mixing
uint hash3u(uint x, uint y, uint z, uint seed) {
  uint h = (x * 0x8da6b343u) ^ (y * 0xd8163841u) ^ (z * 0xcb1ab31fu) ^ seed;
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  return h ^ (h >> 16u);
}
`;

/** Receiver noise, first pass: white complex samples filtered along the beam (row 0 of the kernel table, centre 8). */
export const GLSL_NOISE_AXIAL_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uKernels;  // buildNoiseKernels: row 0 axial taps, row 1 + sample lateral taps, radius in .g of the centre
uniform int uSamples;
uniform uint uSeed;
uniform uint uFrameIndex;
layout(location = 0) out vec4 outNoise;  // re, 0, im, 1
#define NOISE_RMS ${glslFloat(NOISE_RMS)}
#define NOISE_TAIL ${glslFloat(NOISE_TAIL)}
#define NOISE_MAGNITUDE_BINS ${NOISE_MAGNITUDE_BINS}
#define NOISE_PHASE_BINS ${NOISE_PHASE_BINS}
${psfDefinesGlsl()}
${GLSL_HASH}
// consolePipeline.ts receiverNoise: magnitude from the high 16 bits of one hash, phase from its low 10 bits
vec2 white(int li, int si) {
  uint h = hash3u(uint(li), uint(si), uFrameIndex, uSeed);
  float r = NOISE_RMS * sqrt(-log(1.0 - NOISE_TAIL * (float(h >> 16u) / float(NOISE_MAGNITUDE_BINS))));
  float phi = 6.283185307179586 * (float(h & uint(NOISE_PHASE_BINS - 1)) / float(NOISE_PHASE_BINS));
  return vec2(r * cos(phi), r * sin(phi));
}

void main() {
  int si = int(gl_FragCoord.x);
  int li = int(gl_FragCoord.y);
  int R = int(texelFetch(uKernels, ivec2(PSF_LATERAL_RADIUS, 0), 0).g + 0.5);
  int last = uSamples - 1;
  vec2 acc = vec2(0.0);
  for (int j = -PSF_AXIAL_RADIUS; j <= PSF_AXIAL_RADIUS; j++) {
    if (j < -R || j > R) continue;
    acc += texelFetch(uKernels, ivec2(PSF_LATERAL_RADIUS + j, 0), 0).r * white(li, clamp(si + j, 0, last));
  }
  outNoise = vec4(acc.x, 0.0, acc.y, 1.0);
}
`;

/** Receiver noise, second pass: the axial result filtered across lines (row 1 + sample of the kernel table). */
export const GLSL_NOISE_LATERAL_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uNoiseAx;
uniform sampler2D uKernels;
uniform int uLines;
layout(location = 0) out vec4 outNoise;  // re, 0, im, 1
${psfDefinesGlsl()}

void main() {
  int si = int(gl_FragCoord.x);
  int li = int(gl_FragCoord.y);
  int R = int(texelFetch(uKernels, ivec2(PSF_LATERAL_RADIUS, si + 1), 0).g + 0.5);
  int last = uLines - 1;
  vec4 acc = vec4(0.0);
  for (int j = -PSF_LATERAL_RADIUS; j <= PSF_LATERAL_RADIUS; j++) {
    if (j < -R || j > R) continue;
    acc += texelFetch(uKernels, ivec2(PSF_LATERAL_RADIUS + j, si + 1), 0).r * texelFetch(uNoiseAx, ivec2(si, clamp(li + j, 0, last)), 0);
  }
  outNoise = vec4(acc.x, 0.0, acc.z, 1.0);
}
`;

/** Console pass 1: detection with receiver noise + depth/TGC amplification + log compression → 0..1 (evaluated once). */
export const GLSL_LOG_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uEnv;    // pass D: envelope amplitude (r), transmission (g)
uniform sampler2D uComp;   // console amplification per sample (r), one row
uniform sampler2D uNoise;  // receiver noise passes: re (r), im (b)
uniform float uDynRange;   // dB
layout(location = 0) out vec4 outLog;
#define REF_DB ${glslFloat(REF_DB)}
#define LOG_FLOOR ${glslFloat(LOG_FLOOR)}
#define LOG_KNEE_START ${glslFloat(LOG_KNEE_START)}
#define LOG_KNEE_SPAN ${glslFloat(LOG_KNEE_SPAN)}

// src/simulator/renderer/postprocess/consolePipeline.ts: softKneeCompress
float softKneeCompress(float y) {
  if (y <= 0.0) return 0.0;
  if (y <= LOG_KNEE_START) return y;
  float w = LOG_KNEE_SPAN;
  if (y >= LOG_KNEE_START + w) return 1.0;
  float h = 1.0 - LOG_KNEE_START;
  float m = w / h;
  float t = (y - LOG_KNEE_START) / w;
  return LOG_KNEE_START + h * t * (m + (3.0 - 2.0 * m) * t + (m - 2.0) * t * t);
}

void main() {
  int si = int(gl_FragCoord.x);
  int li = int(gl_FragCoord.y);
  float amp = texelFetch(uEnv, ivec2(si, li), 0).r;
  vec4 nz = texelFetch(uNoise, ivec2(si, li), 0);
  float x = amp + nz.r;
  float a = sqrt(x * x + nz.b * nz.b) * texelFetch(uComp, ivec2(si, 0), 0).r;
  float db = 20.0 * log(a + LOG_FLOOR) * 0.4342944819032518 - REF_DB;
  float y = softKneeCompress((db + uDynRange) / uDynRange);
  outLog = vec4(y, 0.0, 0.0, 1.0);
}
`;

/** Console pass 2: 2-scale structure-oriented anisotropic SRI + axial edge enhancement + persistence + grey map. */
export const GLSL_CONSOLE_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uLog;    // log pass: compressed level 0..1 (r)
uniform sampler2D uEnv;    // pass D: transmission (g)
uniform sampler2D uIds;    // pass B ids: structure / 255 (r), tissue / 255 (g)
uniform sampler2D uHist;   // previous console output before the grey map (r)
uniform int uSamples;
uniform int uLines;
uniform float uSri;        // structure-oriented anisotropic speckle reduction 0..1
uniform float uEdge;       // edge enhancement strength (edgeEnhance · 0.8)
uniform float uPersist;    // weight of the history (0 without history)
uniform int uGrayMap;      // 0 linear, 1 s-curve, 2 high-contrast, 3 clinical
layout(location = 0) out vec4 outHist;
layout(location = 1) out vec4 outPacked;
#define TRANS_K ${glslFloat(TRANS_K)}
#define GREY_C ${glslFloat(CLINICAL_GREY_CURVE)}
#define GREY_LOG ${glslFloat(Math.log1p(CLINICAL_GREY_CURVE))}
#define S_CURVE_MIX ${glslFloat(S_CURVE_MIX)}
#define HIGH_CONTRAST_GAMMA ${glslFloat(HIGH_CONTRAST_GAMMA)}
${sriDefinesGlsl()}

float logAt(int si, int li) {
  return texelFetch(uLog, ivec2(si, li), 0).r;
}

float sriAt(int si, int li, float strength) {
  float cc = logAt(si, li);
  if (!(strength > 0.0) || si <= 0 || si >= uSamples - 1 || li <= 0 || li >= uLines - 1) return cc;
  float nw = logAt(si - 1, li - 1);
  float nn = logAt(si, li - 1);
  float ne = logAt(si + 1, li - 1);
  float ww = logAt(si - 1, li);
  float ee = logAt(si + 1, li);
  float sw = logAt(si - 1, li + 1);
  float ss = logAt(si, li + 1);
  float se = logAt(si + 1, li + 1);

  float gs = (ne + 2.0 * ee + se) - (nw + 2.0 * ww + sw);
  float gl = (sw + 2.0 * ss + se) - (nw + 2.0 * nn + ne);

  bool hasCoarse = si >= 2 && si < uSamples - 2 && li >= 2 && li < uLines - 2;
  float nw2 = 0.0, nn2 = 0.0, ne2 = 0.0, ww2 = 0.0, ee2 = 0.0, sw2 = 0.0, ss2 = 0.0, se2 = 0.0;
  if (hasCoarse) {
    nw2 = logAt(si - 2, li - 2);
    nn2 = logAt(si, li - 2);
    ne2 = logAt(si + 2, li - 2);
    ww2 = logAt(si - 2, li);
    ee2 = logAt(si + 2, li);
    sw2 = logAt(si - 2, li + 2);
    ss2 = logAt(si, li + 2);
    se2 = logAt(si + 2, li + 2);
    float gs2 = (ne2 + 2.0 * ee2 + se2) - (nw2 + 2.0 * ww2 + sw2);
    float gl2 = (sw2 + 2.0 * ss2 + se2) - (nw2 + 2.0 * nn2 + ne2);
    gs = 0.65 * gs + 0.35 * gs2;
    gl = 0.65 * gl + 0.35 * gl2;
  }

  float mag = length(vec2(gs, gl));
  float edgeConf = min(1.0, mag / SRI_EDGE_REF);
  vec2 nrm = mag > 1e-5 ? vec2(gs, gl) / mag : vec2(1.0, 0.0);
  vec2 tng = vec2(-nrm.y, nrm.x);

  float wt0 = tng.x * tng.x; wt0 *= wt0;
  float wt1 = tng.y * tng.y; wt1 *= wt1;
  float dT2 = (tng.x + tng.y) * 0.7071067811865476; float wt2 = dT2 * dT2; wt2 *= wt2;
  float dT3 = (tng.x - tng.y) * 0.7071067811865476; float wt3 = dT3 * dT3; wt3 *= wt3;

  float wn0 = nrm.x * nrm.x; wn0 *= wn0;
  float wn1 = nrm.y * nrm.y; wn1 *= wn1;
  float dN2 = (nrm.x + nrm.y) * 0.7071067811865476; float wn2 = dN2 * dN2; wn2 *= wn2;
  float dN3 = (nrm.x - nrm.y) * 0.7071067811865476; float wn3 = dN3 * dN3; wn3 *= wn3;

  float dirScale = 0.2 + 0.8 * edgeConf;
  float isoFloor = 0.18 * (1.0 - edgeConf);
  float g0 = wt0 * dirScale + isoFloor;
  float g1 = wt1 * dirScale + isoFloor;
  float g2 = wt2 * dirScale + isoFloor;
  float g3 = wt3 * dirScale + isoFloor;

  float we  = g0 * exp(-(ee - cc) * (ee - cc) * SRI_INV_2_SIGMA2);
  float w_w = g0 * exp(-(ww - cc) * (ww - cc) * SRI_INV_2_SIGMA2);
  float ws  = g1 * exp(-(ss - cc) * (ss - cc) * SRI_INV_2_SIGMA2);
  float wn  = g1 * exp(-(nn - cc) * (nn - cc) * SRI_INV_2_SIGMA2);
  float wse = g2 * exp(-(se - cc) * (se - cc) * SRI_INV_2_SIGMA2);
  float wnw = g2 * exp(-(nw - cc) * (nw - cc) * SRI_INV_2_SIGMA2);
  float wne = g3 * exp(-(ne - cc) * (ne - cc) * SRI_INV_2_SIGMA2);
  float wsw = g3 * exp(-(sw - cc) * (sw - cc) * SRI_INV_2_SIGMA2);

  float tangW1 = we + w_w + ws + wn + wse + wnw + wne + wsw + SRI_CENTER_WEIGHT;
  float tangMean = (we * ee + w_w * ww + ws * ss + wn * nn + wse * se + wnw * nw + wne * ne + wsw * sw + SRI_CENTER_WEIGHT * cc) / tangW1;

  float normW = 2.0 * (wn0 + wn1 + wn2 + wn3);
  float normMean = normW > 1e-6 ? (wn0 * (ee + ww) + wn1 * (ss + nn) + wn2 * (se + nw) + wn3 * (ne + sw)) / normW : cc;

  if (hasCoarse) {
    float cwe  = g0 * exp(-(ee2 - cc) * (ee2 - cc) * SRI_INV_2_COARSE_SIGMA2);
    float cw_w = g0 * exp(-(ww2 - cc) * (ww2 - cc) * SRI_INV_2_COARSE_SIGMA2);
    float cws  = g1 * exp(-(ss2 - cc) * (ss2 - cc) * SRI_INV_2_COARSE_SIGMA2);
    float cwn  = g1 * exp(-(nn2 - cc) * (nn2 - cc) * SRI_INV_2_COARSE_SIGMA2);
    float cwse = g2 * exp(-(se2 - cc) * (se2 - cc) * SRI_INV_2_COARSE_SIGMA2);
    float cwnw = g2 * exp(-(nw2 - cc) * (nw2 - cc) * SRI_INV_2_COARSE_SIGMA2);
    float cwne = g3 * exp(-(ne2 - cc) * (ne2 - cc) * SRI_INV_2_COARSE_SIGMA2);
    float cwsw = g3 * exp(-(sw2 - cc) * (sw2 - cc) * SRI_INV_2_COARSE_SIGMA2);

    float tangW2 = cwe + cw_w + cws + cwn + cwse + cwnw + cwne + cwsw + SRI_CENTER_WEIGHT;
    float tangMean2 = (cwe * ee2 + cw_w * ww2 + cws * ss2 + cwn * nn2 + cwse * se2 + cwnw * nw2 + cwne * ne2 + cwsw * sw2 + SRI_CENTER_WEIGHT * cc) / tangW2;
    tangMean = (1.0 - SRI_COARSE_SHARE) * tangMean + SRI_COARSE_SHARE * tangMean2;

    float loC = cc - 0.28, hiC = cc + 0.28;
    float normMean2 = normW > 1e-6
      ? (wn0 * (clamp(ee2, loC, hiC) + clamp(ww2, loC, hiC)) +
         wn1 * (clamp(ss2, loC, hiC) + clamp(nn2, loC, hiC)) +
         wn2 * (clamp(se2, loC, hiC) + clamp(nw2, loC, hiC)) +
         wn3 * (clamp(ne2, loC, hiC) + clamp(sw2, loC, hiC))) / normW
      : cc;
    normMean = 0.75 * normMean + 0.25 * normMean2;
  }

  float smoothMix = strength * (SRI_SMOOTH_BASE + SRI_SMOOTH_COH * edgeConf);
  float smoothed = cc + (tangMean - cc) * smoothMix;
  float sharp = (cc - normMean) * (strength * SRI_SHARP_GAIN * edgeConf);
  return clamp(smoothed + sharp, 0.0, 1.0);
}

void main() {
  int si = int(gl_FragCoord.x);
  int li = int(gl_FragCoord.y);
  float c = sriAt(si, li, uSri);
  float y = c;
  if (uEdge > 0.0 && si > 0 && si < uSamples - 1) {
    float m = 0.5 * (logAt(si - 1, li) + logAt(si + 1, li));
    y = clamp(c + (c - m) * uEdge, 0.0, 1.0);
  }
  if (uPersist > 0.0) y = y * (1.0 - uPersist) + texelFetch(uHist, ivec2(si, li), 0).r * uPersist;
  outHist = vec4(y, 0.0, 0.0, 1.0);
  float g = y;
  if (uGrayMap == 1) g = g * g * (3.0 - 2.0 * g) * S_CURVE_MIX + g * (1.0 - S_CURVE_MIX);
  else if (uGrayMap == 2) g = pow(g, HIGH_CONTRAST_GAMMA);
  else if (uGrayMap == 3) g = (exp(g * GREY_LOG) - 1.0) / GREY_C;
  vec4 ids = texelFetch(uIds, ivec2(si, li), 0);
  float t = clamp(texelFetch(uEnv, ivec2(si, li), 0).g, 1e-4, 1.0);
  float code = clamp(floor(-log(t) / TRANS_K * 255.0 + 0.5), 0.0, 255.0);
  outPacked = vec4(floor(g * 255.0 + 0.5) / 255.0, ids.r, ids.g, code / 255.0);
}
`;

export const GLSL_PRESENT_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
precision highp sampler2D;
uniform sampler2D uPacked;  // console output: grey (r)
uniform usampler2D uLut;    // scanConvert.ts packScanLutTexels, top row first
uniform sampler2D uColor;   // pre-encoded colour phasor per polar sample: (cos(phi), sin(phi), variance, valid)
uniform sampler2D uPolar;   // the LUT's polar coordinates per pixel: r in cm (r), theta in rad (g), top row first
uniform int uHeightPx;
uniform int uLines;
uniform int uSamples;
uniform int uColorOn;
uniform int uTintMap;       // 0 gray, 1 cool-blue, 2 sepia, 3 amber
uniform vec4 uBox;          // rMin, rMax, thetaMin, thetaMax
uniform vec3 uColorMap;     // scale (m/s), show variance (0/1), baselineShift (m/s)
uniform int uColorPalette;  // 0 vivid, 1 classic, 2 variance, 3 ice-fire
out vec4 outColor;
${colorMapDefinesGlsl()}
${tintMapGlsl()}

float greyF(int s, int l) { return floor(texelFetch(uPacked, ivec2(s, l), 0).r * 255.0 + 0.5); }
${colorMapGlsl()}

void main() {
  int x = int(gl_FragCoord.x);
  int y = uHeightPx - 1 - int(gl_FragCoord.y);
  uvec4 L = texelFetch(uLut, ivec2(x, y), 0);
  if ((L.b & 2048u) == 0u) {
    outColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }
  int l0 = int(L.r);
  int s0 = int(L.g);
  uint lt = L.b & 1023u;
  uint st = L.a & 1023u;
  int dl = int((L.b >> 10u) & 1u);
  int ds = int((L.a >> 10u) & 1u);
  float fl = float(lt) * (1.0 / 1024.0);
  float fs = float(st) * (1.0 / 1024.0);

  // Clamped Catmull-Rom 4x4 bicubic gather matching CPU scanConvertLut
  int maxS = uSamples - 1;
  int maxL = uLines - 1;
  int sm1 = max(0, s0 - 1);
  int s1 = min(maxS, s0 + 1);
  int s2 = min(maxS, s0 + 2);
  int lm1 = max(0, l0 - 1);
  int l1 = min(maxL, l0 + 1);
  int l2 = min(maxL, l0 + 2);

  float fs2 = fs * fs;
  float fs3 = fs2 * fs;
  vec4 ws = vec4(-0.5 * fs + fs2 - 0.5 * fs3, 1.0 - 2.5 * fs2 + 1.5 * fs3, 0.5 * fs + 2.0 * fs2 - 1.5 * fs3, -0.5 * fs2 + 0.5 * fs3);

  float c0 = dot(vec4(greyF(sm1, lm1), greyF(s0, lm1), greyF(s1, lm1), greyF(s2, lm1)), ws);
  float v00 = greyF(s0, l0);
  float v01 = greyF(s1, l0);
  float c1 = dot(vec4(greyF(sm1, l0), v00, v01, greyF(s2, l0)), ws);
  float v10 = greyF(s0, l1);
  float v11 = greyF(s1, l1);
  float c2 = dot(vec4(greyF(sm1, l1), v10, v11, greyF(s2, l1)), ws);
  float c3 = dot(vec4(greyF(sm1, l2), greyF(s0, l2), greyF(s1, l2), greyF(s2, l2)), ws);

  float fl2 = fl * fl;
  float fl3 = fl2 * fl;
  vec4 wl = vec4(-0.5 * fl + fl2 - 0.5 * fl3, 1.0 - 2.5 * fl2 + 1.5 * fl3, 0.5 * fl + 2.0 * fl2 - 1.5 * fl3, -0.5 * fl2 + 0.5 * fl3);
  float lo = min(min(v00, v10), min(v01, v11));
  float hi = max(max(v00, v10), max(v01, v11));
  float g = clamp(dot(vec4(c0, c1, c2, c3), wl), lo, hi);
  vec3 rgb = applyTintMap(g, uTintMap);
  if (uColorOn == 1) {
    vec2 rt = texelFetch(uPolar, ivec2(x, y), 0).rg;
    if (rt.x >= uBox.x && rt.x <= uBox.y && rt.y >= uBox.z && rt.y <= uBox.w) {
      int cl1 = min(uLines - 1, l0 + dl);
      int cs1 = min(uSamples - 1, s0 + ds);
      vec4 d00 = texelFetch(uColor, ivec2(s0, l0), 0);
      vec4 d10 = texelFetch(uColor, ivec2(s0, cl1), 0);
      vec4 d01 = texelFetch(uColor, ivec2(cs1, l0), 0);
      vec4 d11 = texelFetch(uColor, ivec2(cs1, cl1), 0);
      float v00 = step(0.5, d00.a);
      float v10 = step(0.5, d10.a);
      float v01 = step(0.5, d01.a);
      float v11 = step(0.5, d11.a);
      float w00 = (1.0 - fl) * (1.0 - fs) * v00;
      float w10 = fl * (1.0 - fs) * v10;
      float w01 = (1.0 - fl) * fs * v01;
      float w11 = fl * fs * v11;
      float wSum = w00 + w10 + w01 + w11;
      if (wSum > 0.18) {
        vec2 ph = d00.rg * w00 + d10.rg * w10 + d01.rg * w01 + d11.rg * w11;
        float vari = (d00.b * w00 + d10.b * w10 + d01.b * w01 + d11.b * w11) / wSum;
        float p00 = mix(${glslFloat(ST_POW_ALPHA_MIN)}, 1.0, clamp((d00.a - 0.5) * 2.0, 0.0, 1.0));
        float p10 = mix(${glslFloat(ST_POW_ALPHA_MIN)}, 1.0, clamp((d10.a - 0.5) * 2.0, 0.0, 1.0));
        float p01 = mix(${glslFloat(ST_POW_ALPHA_MIN)}, 1.0, clamp((d01.a - 0.5) * 2.0, 0.0, 1.0));
        float p11 = mix(${glslFloat(ST_POW_ALPHA_MIN)}, 1.0, clamp((d11.a - 0.5) * 2.0, 0.0, 1.0));
        float powAlpha = (p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11) / wSum;
        float bt00 = clamp(d00.a - 1.0, 0.0, 1.0);
        float bt10 = clamp(d10.a - 1.0, 0.0, 1.0);
        float bt01 = clamp(d01.a - 1.0, 0.0, 1.0);
        float bt11 = clamp(d11.a - 1.0, 0.0, 1.0);
        float breakthrough = (bt00 * w00 + bt10 * w10 + bt01 * w01 + bt11 * w11) / wSum;
        float scale = max(1e-3, uColorMap.x);
        float baseline = uColorMap.z;
        float v = baseline + (scale / 3.141592653589793) * atan(ph.y, ph.x);
        float edgeAlpha = smoothstep(0.18, 0.72, wSum);
        float lum = dot(rgb, vec3(0.299, 0.587, 0.114));
        float tissuePrio = 1.0 - smoothstep(COLOR_TISSUE_PRIO_LO, COLOR_TISSUE_PRIO_HI, lum);
        float effTwp = mix(tissuePrio, 1.0, breakthrough);
        float velAlpha = 0.62 + 0.38 * smoothstep(0.05, 0.22, abs(v) / scale);
        float blend = COLOR_BLEND * edgeAlpha * powAlpha * velAlpha * effTwp;
        rgb = floor(colorMap(v, scale, vari, uColorMap.y > 0.5, baseline, uColorPalette) * blend + rgb * (1.0 - blend) + 0.5);
      }
    }
  }
  outColor = vec4(rgb / 255.0, 1.0);
}
`;
