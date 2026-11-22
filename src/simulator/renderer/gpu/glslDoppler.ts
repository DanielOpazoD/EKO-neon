import { Tissue } from '@/simulator/anatomy/tissue';
import {
  COLOR_THRESHOLD,
  DOPPLER_SHADOW_TRANSMISSION,
} from '@/simulator/doppler/color/colorDoppler';
import {
  COLOR_PACKET,
  KASAI_LIN_BASIS,
  LAMINAR_DECORRELATION,
  NOISE_COHERENCE,
  NOISE_POWER,
  NOISE_POWER_GATE,
  TRANSIT_JITTER_RAD,
} from '@/simulator/doppler/color/kasaiEstimator';
import {
  ATTEN_NP_PER_DB,
  glslFloat,
  SOFT_TISSUE_ATTEN_DB,
} from '../acoustic/acoustics';

/**
 * Shared GLSL physical constants for the deferred Color Doppler pipeline.
 * Every numeric constant is imported directly from the TypeScript acoustic and Doppler
 * modules so CPU (`computeColorField` / `kasaiEstimate`) and GPU deferred passes stay locked in sync.
 */
export function dopplerDefinesGlsl(): string {
  const linBasisLiterals = Array.from(KASAI_LIN_BASIS, (v) => glslFloat(v)).join(', ');
  return [
    `#define T_BLOOD ${Tissue.Blood}`,
    `#define ATTEN_NP_PER_DB ${glslFloat(ATTEN_NP_PER_DB)}`,
    `#define SOFT_TISSUE_ATTEN_DB ${glslFloat(SOFT_TISSUE_ATTEN_DB)}`,
    `#define DOPPLER_SHADOW_TRANSMISSION ${glslFloat(DOPPLER_SHADOW_TRANSMISSION)}`,
    `#define COLOR_THRESHOLD ${glslFloat(COLOR_THRESHOLD)}`,
    `#define COLOR_PACKET ${COLOR_PACKET}`,
    `#define NOISE_POWER ${glslFloat(NOISE_POWER)}`,
    `#define TRANSIT_JITTER_RAD ${glslFloat(TRANSIT_JITTER_RAD)}`,
    `#define NOISE_COHERENCE ${glslFloat(NOISE_COHERENCE)}`,
    `#define NOISE_POWER_GATE ${glslFloat(NOISE_POWER_GATE)}`,
    `#define LAMINAR_DECORRELATION ${glslFloat(LAMINAR_DECORRELATION)}`,
    `const float KASAI_LIN_BASIS[COLOR_PACKET] = float[COLOR_PACKET](${linBasisLiterals});`,
  ].join('\n');
}

const GLSL_DOPPLER_HASH = /* glsl */ `
// Identical 32-bit integer hash to core/random.ts hash3
uint hash3u(uint x, uint y, uint z, uint seed) {
  uint h = (x * 0x8da6b343u) ^ (y * 0xd8163841u) ^ (z * 0xcb1ab31fu) ^ seed;
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  return h ^ (h >> 16u);
}

// Pair of independent unit Gaussians via Box-Muller transform
vec2 hashGaussianPair(int li, int si, uint stream, uint seed) {
  float u1 = float(hash3u(uint(li), uint(si), stream, seed)) * (1.0 / 4294967296.0);
  float u2 = float(hash3u(uint(li), uint(si), stream + 1u, seed)) * (1.0 / 4294967296.0);
  float r = sqrt(-2.0 * log(max(1e-12, u1)));
  float ang = 6.283185307179586 * u2;
  return vec2(r * cos(ang), r * sin(ang));
}

// Resolution-cell spatially correlated unit-variance Gaussian pair (2x2 box: Var = 4 * 0.25 = 1.0)
vec2 correlatedGaussianPair(int li, int si, uint stream, uint seed) {
  int l0 = max(0, li - 1);
  int s0 = max(0, si - 1);
  return 0.5 * (
    hashGaussianPair(li, si, stream, seed) +
    hashGaussianPair(l0, si, stream, seed) +
    hashGaussianPair(li, s0, stream, seed) +
    hashGaussianPair(l0, s0, stream, seed)
  );
}
`;

/**
 * Deferred Doppler Pass D1 — Polar G-Buffer Gate, Acoustic Shadowing, Gain Blooming &
 * Kasai 8-Pulse Complex Echo Packet Autocorrelation.
 *
 * Reads the GPU-resident Polar G-Buffer produced independently by B-mode reconstruction
 * (`uGbufEnv` transmission + `uGbufIds` tissue IDs) plus sparse ROI-only hemodynamic flow (`uFlowRaw`).
 * Executed strictly inside a scissor-clipped Color ROI box so mid-range GPUs shade only ~15-25% of texels.
 *
 * Output (`RGBA32F`):
 *  - Kasai mode (`uUseKasai != 0`): `(r0, r1r, r1i, 1.0)` when valid, `vec4(0.0)` otherwise.
 *  - Expected mode (`uUseKasai == 0`): `(vSigned, dispersion, relTrans, 1.0)` when valid, `vec4(0.0)` otherwise.
 */
export const GLSL_DOPPLER_AUTOCORR_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D uGbufEnv;   // Polar G-Buffer: envelope (r), exact two-way transmission (g)
uniform sampler2D uGbufIds;   // Polar G-Buffer: structure/255 (r), tissue/255 (g)
uniform sampler2D uFlowRaw;   // Sparse ROI flow: axial v m/s (r), dispersion (g), present (b)
uniform int uSamples;
uniform int uLines;
uniform float uDepthCm;
uniform float uAttenFreqMHz;
uniform float uScaleMps;
uniform float uWallFilterMps;
uniform int uBloomSamples;
uniform int uInvert;
uniform int uUseKasai;
uniform uint uRealization;
uniform uint uSeed;

layout(location = 0) out vec4 outCorr;

const float PI = 3.141592653589793;
const float INV_SQRT2 = 0.7071067811865476;
${dopplerDefinesGlsl()}
${GLSL_DOPPLER_HASH}

int tissueAt(int si, int li) {
  return int(floor(texelFetch(uGbufIds, ivec2(si, li), 0).g * 255.0 + 0.5));
}

void main() {
  int si = int(gl_FragCoord.x);
  int li = int(gl_FragCoord.y);
  if (si < 0 || si >= uSamples || li < 0 || li >= uLines) {
    outCorr = vec4(0.0);
    return;
  }

  // 1. Acoustic shadow gate against the Polar G-Buffer's exact two-way transmission
  float trans = texelFetch(uGbufEnv, ivec2(si, li), 0).g;
  float rCm = ((float(si) + 0.5) / float(uSamples)) * uDepthCm;
  float expectedTrans = exp(-ATTEN_NP_PER_DB * SOFT_TISSUE_ATTEN_DB * uAttenFreqMHz * rCm);
  float relTrans = trans / max(1e-12, expectedTrans);
  if (relTrans < DOPPLER_SHADOW_TRANSMISSION) {
    outCorr = vec4(0.0);
    return;
  }

  // 2. Tissue gate & gain-driven blooming dilation on the Polar G-Buffer
  bool isBlood = tissueAt(si, li) == T_BLOOD;
  if (!isBlood && uBloomSamples > 0) {
    int maxBloom = min(uBloomSamples, 12);
    for (int d = 1; d <= 12; d++) {
      if (d > maxBloom || isBlood) break;
      if (si - d >= 0 && tissueAt(si - d, li) == T_BLOOD) isBlood = true;
      if (!isBlood && si + d < uSamples && tissueAt(si + d, li) == T_BLOOD) isBlood = true;
    }
    if (!isBlood && li > 0 && tissueAt(si, li - 1) == T_BLOOD) isBlood = true;
    if (!isBlood && li < uLines - 1 && tissueAt(si, li + 1) == T_BLOOD) isBlood = true;
  }
  if (!isBlood) {
    outCorr = vec4(0.0);
    return;
  }

  // 3. Raw hemodynamic velocity sample inside the Color ROI box
  vec4 raw = texelFetch(uFlowRaw, ivec2(si, li), 0);
  if (raw.b < 0.5) {
    outCorr = vec4(0.0);
    return;
  }

  float v = uInvert != 0 ? -raw.r : raw.r;
  float disp = raw.g;
  float rel = min(1.0, relTrans);

  // Expected (noise-free) mode forwards exact flow parameters directly to Pass D2
  if (uUseKasai == 0) {
    outCorr = vec4(v, disp, rel, 1.0);
    return;
  }

  // 4. Kasai 8-pulse complex echo packet synthesis + continuous regression wall filter
  float scale = max(1e-3, uScaleMps);
  uint baseStream = uRealization * 64u;
  float amp = sqrt(rel);
  vec2 zSpec = correlatedGaussianPair(li, si, baseStream, uSeed) * (amp * INV_SQRT2);
  float ar = zSpec.x;
  float ai = zSpec.y;

  float jit[COLOR_PACKET];
  vec2 jp0 = correlatedGaussianPair(li, si, baseStream + 2u, uSeed);
  vec2 jp1 = correlatedGaussianPair(li, si, baseStream + 4u, uSeed);
  vec2 jp2 = correlatedGaussianPair(li, si, baseStream + 6u, uSeed);
  vec2 jp3 = correlatedGaussianPair(li, si, baseStream + 8u, uSeed);
  jit[0] = jp0.x; jit[1] = jp0.y;
  jit[2] = jp1.x; jit[3] = jp1.y;
  jit[4] = jp2.x; jit[5] = jp2.y;
  jit[6] = jp3.x; jit[7] = jp3.y;

  float phi = (PI * v) / scale;
  float turbulent = (PI * 0.9 * disp * abs(v)) / scale * INV_SQRT2;
  float jitter = length(vec2(TRANSIT_JITTER_RAD, turbulent));
  float noiseAmp = sqrt(NOISE_POWER * 0.5);

  float xr[COLOR_PACKET];
  float xi[COLOR_PACKET];
  for (int p = 0; p < COLOR_PACKET; p++) {
    float ph = float(p) * phi + jitter * jit[p];
    float cs = cos(ph);
    float sn = sin(ph);
    vec2 nz = hashGaussianPair(li, si, baseStream + 24u + 2u * uint(p), uSeed);
    xr[p] = ar * cs - ai * sn + noiseAmp * nz.x;
    xi[p] = ar * sn + ai * cs + noiseAmp * nz.y;
  }

  // Continuous regression wall filter (mean + orthonormal linear trend removal)
  float cut = max(0.0, uWallFilterMps / scale);
  float meanShare = min(1.0, cut / 0.1);
  float trendShare = min(1.0, max(0.0, (cut - 0.1) / 0.25));
  if (meanShare > 0.0) {
    float mr = 0.0;
    float mi = 0.0;
    for (int p = 0; p < COLOR_PACKET; p++) {
      mr += xr[p];
      mi += xi[p];
    }
    mr = (mr / float(COLOR_PACKET)) * meanShare;
    mi = (mi / float(COLOR_PACKET)) * meanShare;
    for (int p = 0; p < COLOR_PACKET; p++) {
      xr[p] -= mr;
      xi[p] -= mi;
    }
    if (trendShare > 0.0) {
      float pr = 0.0;
      float pi = 0.0;
      for (int p = 0; p < COLOR_PACKET; p++) {
        pr += xr[p] * KASAI_LIN_BASIS[p];
        pi += xi[p] * KASAI_LIN_BASIS[p];
      }
      pr *= trendShare;
      pi *= trendShare;
      for (int p = 0; p < COLOR_PACKET; p++) {
        xr[p] -= pr * KASAI_LIN_BASIS[p];
        xi[p] -= pi * KASAI_LIN_BASIS[p];
      }
    }
  }

  // Lag-0 and lag-1 complex autocorrelations R0, R1
  float a0 = 0.0;
  float a1r = 0.0;
  float a1i = 0.0;
  for (int p = 0; p < COLOR_PACKET; p++) {
    a0 += xr[p] * xr[p] + xi[p] * xi[p];
  }
  for (int p = 0; p < COLOR_PACKET - 1; p++) {
    a1r += xr[p + 1] * xr[p] + xi[p + 1] * xi[p];
    a1i += xi[p + 1] * xr[p] - xr[p + 1] * xi[p];
  }

  outCorr = vec4(
    a0 / float(COLOR_PACKET),
    a1r / float(COLOR_PACKET - 1),
    a1i / float(COLOR_PACKET - 1),
    1.0
  );
}
`;

/**
 * Deferred Doppler Pass D2 — Resolution-Cell Autocorrelation Averaging, Coherence & Power Gating,
 * Nyquist Phase Aliasing & GPU-Native Circular Phasor Temporal Persistence.
 *
 * Outputs MRT (`RGBA16F`):
 *  - location 0 (`outDisplay`): `(cos(phi_disp), sin(phi_disp), variance, valid)` ready for direct
 *    binding to `GLSL_PRESENT_FRAG`'s `uColor` sampler with zero CPU readback or repack.
 *  - location 1 (`outHist`): `(cos(phi_base) * power, sin(phi_base) * power, variance, power)`
 *    ping-ponged across Doppler updates for aliasing-safe circular temporal persistence.
 */
export const GLSL_DOPPLER_SHADE_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D uCorr;     // Pass D1 output (RGBA32F)
uniform sampler2D uHist;     // Previous ping-pong circular phasor history (RGBA16F)
uniform ivec4 uBoxBounds;    // (siMin, liMin, siMax, liMax) inclusive
uniform float uScaleMps;
uniform float uBaselineMps;
uniform float uWallFilterMps;
uniform float uGainLin;
uniform float uPersist;
uniform int uUseHistory;
uniform int uUseKasai;

layout(location = 0) out vec4 outDisplay;
layout(location = 1) out vec4 outHist;

const float PI = 3.141592653589793;
${dopplerDefinesGlsl()}

float wallFilterResponse(float vAbs, float cutoffMps) {
  if (cutoffMps <= 0.0) return 1.0;
  float x = (vAbs / cutoffMps);
  x = x * x;
  return x / sqrt(1.0 + x * x);
}

float aliasVelocity(float v, float scale, float baseline) {
  float span = 2.0 * scale;
  float lo = baseline - scale;
  float x = mod(v - lo, span);
  if (x < 0.0) x += span;
  return x + lo;
}

float safeAtan2(float y, float x) {
  return (abs(x) + abs(y) > 1e-12) ? atan(y, x) : 0.0;
}

void main() {
  int si = int(gl_FragCoord.x);
  int li = int(gl_FragCoord.y);

  vec4 center = texelFetch(uCorr, ivec2(si, li), 0);
  if (center.a < 0.5) {
    outDisplay = vec4(0.0);
    outHist = vec4(0.0);
    return;
  }

  float scale = max(1e-3, uScaleMps);
  float power = 0.0;
  float vEstimate = 0.0;
  float variance = 0.0;

  if (uUseKasai != 0) {
    // 3x3 binomial [1 2 1] resolution-cell averaging over valid neighbors strictly inside uBoxBounds
    float sumR0 = 0.0;
    float sumR1r = 0.0;
    float sumR1i = 0.0;
    float wSum = 0.0;
    for (int dl = -1; dl <= 1; dl++) {
      int ll = li + dl;
      if (ll < uBoxBounds.y || ll > uBoxBounds.w) continue;
      float wl = (dl == 0) ? 2.0 : 1.0;
      for (int ds = -1; ds <= 1; ds++) {
        int ss = si + ds;
        if (ss < uBoxBounds.x || ss > uBoxBounds.z) continue;
        vec4 tap = texelFetch(uCorr, ivec2(ss, ll), 0);
        if (tap.a < 0.5) continue;
        float wt = wl * ((ds == 0) ? 2.0 : 1.0);
        sumR0 += wt * tap.r;
        sumR1r += wt * tap.g;
        sumR1i += wt * tap.b;
        wSum += wt;
      }
    }
    if (wSum <= 0.0) {
      outDisplay = vec4(0.0);
      outHist = vec4(0.0);
      return;
    }
    float R0 = sumR0 / wSum;
    if (R0 <= 0.0) {
      outDisplay = vec4(0.0);
      outHist = vec4(0.0);
      return;
    }
    float R1r = sumR1r / wSum;
    float R1i = sumR1i / wSum;
    vEstimate = (scale * safeAtan2(R1i, R1r)) / PI;
    float rawPow = max(0.0, R0 - NOISE_POWER);
    power = min(1.0, uGainLin * rawPow);
    float coherence = length(vec2(R1r, R1i)) / R0;
    if (coherence < NOISE_COHERENCE && power < NOISE_POWER_GATE * COLOR_THRESHOLD) {
      outDisplay = vec4(0.0);
      outHist = vec4(0.0);
      return;
    }
    float rawVar = 1.0 - coherence;
    variance = clamp((rawVar - LAMINAR_DECORRELATION) / (1.0 - LAMINAR_DECORRELATION), 0.0, 1.0);
  } else {
    float vTrue = center.r;
    float disp = center.g;
    float rel = center.b;
    if (abs(vTrue) < uWallFilterMps) {
      outDisplay = vec4(0.0);
      outHist = vec4(0.0);
      return;
    }
    power = min(1.0, uGainLin * rel * wallFilterResponse(abs(vTrue), uWallFilterMps));
    vEstimate = vTrue;
    variance = min(1.0, disp * 1.6);
  }

  if (power < COLOR_THRESHOLD) {
    outDisplay = vec4(0.0);
    outHist = vec4(0.0);
    return;
  }

  float v = aliasVelocity(vEstimate, scale, uBaselineMps);
  float vel = v;
  float powOut = power;
  float a1 = (PI * (v - uBaselineMps)) / scale;

  // Circular power-weighted phasor persistence against previous deferred Doppler frame
  if (uUseHistory != 0 && uPersist > 0.0) {
    vec4 prev = texelFetch(uHist, ivec2(si, li), 0);
    if (prev.a > 0.0) {
      float p = clamp(uPersist, 0.0, 0.99);
      float re = (1.0 - p) * power * cos(a1) + p * prev.r;
      float im = (1.0 - p) * power * sin(a1) + p * prev.g;
      vel = aliasVelocity(uBaselineMps + (scale * safeAtan2(im, re)) / PI, scale, uBaselineMps);
      variance = variance * (1.0 - p) + prev.b * p;
      powOut = power * (1.0 - p) + prev.a * p;
    }
  }

  float phiDisp = (PI * vel) / scale;
  float phiBase = (PI * (vel - uBaselineMps)) / scale;
  outDisplay = vec4(cos(phiDisp), sin(phiDisp), variance, 1.0);
  outHist = vec4(cos(phiBase) * powOut, sin(phiBase) * powOut, variance, powOut);
}
`;
