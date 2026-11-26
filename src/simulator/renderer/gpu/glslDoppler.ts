import { Tissue } from '@/simulator/anatomy/tissue';
import {
  COLOR_THRESHOLD,
  DOPPLER_SHADOW_TRANSMISSION,
} from '@/simulator/doppler/color/colorDoppler';
import {
  CLUTTER_BLOOD_RATIO_POW,
  CLUTTER_JITTER_RAD,
  CLUTTER_SIGMA_V_MPS,
  COLOR_PACKET,
  COLOR_TURBULENT_WIDTH_K,
  KASAI_LIN_BASIS,
  KASAI_QUAD_BASIS,
  LAMINAR_DECORRELATION,
  NOISE_COHERENCE,
  NOISE_POWER,
  NOISE_POWER_GATE,
  TRANSIT_JITTER_RAD,
  WF_MEAN_SPAN,
  WF_QUAD_SPAN,
  WF_QUAD_START,
  WF_TREND_SPAN,
  WF_TREND_START,
} from '@/simulator/doppler/color/kasaiEstimator';
import { spatiotemporalDefinesGlsl } from '@/simulator/doppler/color/spatiotemporalFilter';
import {
  ATTEN_NP_PER_DB,
  glslFloat,
  SOFT_TISSUE_ATTEN_DB,
} from '../acoustic/acoustics';

/**
 * Shared GLSL physical constants for the deferred Color Doppler pipeline.
 * Every numeric constant is imported directly from the TypeScript acoustic and Doppler
 * modules so CPU (`computeColorField` / `kasaiEstimate` / `spatiotemporalFilter`) and GPU deferred passes stay locked in sync.
 */
export function dopplerDefinesGlsl(): string {
  const linBasisLiterals = Array.from(KASAI_LIN_BASIS, (v) => glslFloat(v)).join(', ');
  const quadBasisLiterals = Array.from(KASAI_QUAD_BASIS, (v) => glslFloat(v)).join(', ');
  return [
    `#define T_BLOOD ${Tissue.Blood}`,
    `#define T_MYOCARDIUM ${Tissue.Myocardium}`,
    `#define T_VALVE ${Tissue.Valve}`,
    `#define ATTEN_NP_PER_DB ${glslFloat(ATTEN_NP_PER_DB)}`,
    `#define SOFT_TISSUE_ATTEN_DB ${glslFloat(SOFT_TISSUE_ATTEN_DB)}`,
    `#define DOPPLER_SHADOW_TRANSMISSION ${glslFloat(DOPPLER_SHADOW_TRANSMISSION)}`,
    `#define COLOR_THRESHOLD ${glslFloat(COLOR_THRESHOLD)}`,
    `#define COLOR_PACKET ${COLOR_PACKET}`,
    `#define NOISE_POWER ${glslFloat(NOISE_POWER)}`,
    `#define TRANSIT_JITTER_RAD ${glslFloat(TRANSIT_JITTER_RAD)}`,
    `#define CLUTTER_BLOOD_RATIO_POW ${glslFloat(CLUTTER_BLOOD_RATIO_POW)}`,
    `#define CLUTTER_SIGMA_V_MPS ${glslFloat(CLUTTER_SIGMA_V_MPS)}`,
    `#define CLUTTER_JITTER_RAD ${glslFloat(CLUTTER_JITTER_RAD)}`,
    `#define WF_MEAN_SPAN ${glslFloat(WF_MEAN_SPAN)}`,
    `#define WF_TREND_START ${glslFloat(WF_TREND_START)}`,
    `#define WF_TREND_SPAN ${glslFloat(WF_TREND_SPAN)}`,
    `#define WF_QUAD_START ${glslFloat(WF_QUAD_START)}`,
    `#define WF_QUAD_SPAN ${glslFloat(WF_QUAD_SPAN)}`,
    `#define NOISE_COHERENCE ${glslFloat(NOISE_COHERENCE)}`,
    `#define NOISE_POWER_GATE ${glslFloat(NOISE_POWER_GATE)}`,
    `#define LAMINAR_DECORRELATION ${glslFloat(LAMINAR_DECORRELATION)}`,
    `#define COLOR_TURBULENT_WIDTH_K ${glslFloat(COLOR_TURBULENT_WIDTH_K)}`,
    spatiotemporalDefinesGlsl(),
    `const float KASAI_LIN_BASIS[COLOR_PACKET] = float[COLOR_PACKET](${linBasisLiterals});`,
    `const float KASAI_QUAD_BASIS[COLOR_PACKET] = float[COLOR_PACKET](${quadBasisLiterals});`,
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
 *  - Expected mode (`uUseKasai == 0`): `(vSigned, sigmaVMps, relTrans, 1.0)` when valid, `vec4(0.0)` otherwise.
 */
export const GLSL_DOPPLER_AUTOCORR_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D uGbufEnv;   // Polar G-Buffer: envelope (r), exact two-way transmission (g)
uniform sampler2D uGbufIds;   // Polar G-Buffer: structure/255 (r), tissue/255 (g)
uniform sampler2D uFlowRaw;   // Sparse ROI flow: axial v m/s (r), absolute sigmaVMps (g), present (b)
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
  int tId = tissueAt(si, li);
  bool isBlood = tId == T_BLOOD;
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
  bool isClutterTissue = (tId == T_MYOCARDIUM || tId == T_VALVE);
  if (!isBlood && !isClutterTissue) {
    outCorr = vec4(0.0);
    return;
  }

  // 3. Raw hemodynamic velocity (r,g,b) or myocardial/valve clutter velocity (a) inside the Color ROI box
  vec4 raw = texelFetch(uFlowRaw, ivec2(si, li), 0);
  bool useBlood = isBlood && raw.b >= 0.5;
  bool useClutter = !useBlood && isClutterTissue && abs(raw.a) > 1e-4;
  if (!useBlood && !useClutter) {
    outCorr = vec4(0.0);
    return;
  }

  float classFlag = useBlood ? 1.0 : 0.5;
  float v = useBlood
    ? (uInvert != 0 ? -raw.r : raw.r)
    : (uInvert != 0 ? -raw.a : raw.a);
  float sigmaV = useBlood ? raw.g : CLUTTER_SIGMA_V_MPS;
  float rel = min(1.0, relTrans) * (useBlood ? 1.0 : CLUTTER_BLOOD_RATIO_POW);

  // Expected (noise-free) mode forwards exact flow/clutter parameters directly to Pass D2
  if (uUseKasai == 0) {
    outCorr = vec4(v, sigmaV, rel, classFlag);
    return;
  }

  // 4. Kasai 8-pulse complex echo packet synthesis + 3-basis Legendre regression wall filter
  float scale = max(1e-3, uScaleMps);
  uint baseStream = uRealization * 64u;
  float amp = sqrt(rel);
  vec2 zSpec = correlatedGaussianPair(li, si, baseStream, uSeed) * (amp * INV_SQRT2);
  vec2 zSpec2 = correlatedGaussianPair(li, si, baseStream + 40u, uSeed) * (amp * INV_SQRT2);
  float lookBoost = sqrt(0.5 * (1.0 + dot(zSpec2, zSpec2) / max(1e-8, dot(zSpec, zSpec))));
  float ar = zSpec.x * lookBoost;
  float ai = zSpec.y * lookBoost;

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
  float turbulent = (PI * sigmaV) / scale * INV_SQRT2;
  float baseJitter = useBlood ? TRANSIT_JITTER_RAD : CLUTTER_JITTER_RAD;
  float jitter = length(vec2(baseJitter, turbulent));
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

  // Continuous 3-basis orthonormal Legendre regression wall filter (P0 mean + P1 linear + P2 quadratic)
  float cut = max(0.0, uWallFilterMps / scale);
  float meanShare = min(1.0, cut / WF_MEAN_SPAN);
  float trendShare = min(1.0, max(0.0, (cut - WF_TREND_START) / WF_TREND_SPAN));
  float quadShare = min(1.0, max(0.0, (cut - WF_QUAD_START) / WF_QUAD_SPAN));
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
      if (quadShare > 0.0) {
        float qr = 0.0;
        float qi = 0.0;
        for (int p = 0; p < COLOR_PACKET; p++) {
          qr += xr[p] * KASAI_QUAD_BASIS[p];
          qi += xi[p] * KASAI_QUAD_BASIS[p];
        }
        qr *= quadShare;
        qi *= quadShare;
        for (int p = 0; p < COLOR_PACKET; p++) {
          xr[p] -= qr * KASAI_QUAD_BASIS[p];
          xi[p] -= qi * KASAI_QUAD_BASIS[p];
        }
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
    classFlag
  );
}
`;

/**
 * Deferred Doppler Pass D2 — Resolution-Cell Autocorrelation Averaging, Coherence & Power Gating,
 * Nyquist Phase Aliasing & GPU-Native Flow-Regime-Adaptive Circular Temporal Persistence
 * (asymmetric attack/decay + blood-safe hysteresis decay on momentary stochastic drop-outs).
 *
 * Outputs MRT (`RGBA16F`):
 *  - location 0 (`outShade`): `(cos(phi_base), sin(phi_base), variance, powOut)` fed into Pass D3.
 *  - location 1 (`outHist`): `(cos(phi_base) * powOut, sin(phi_base) * powOut, variance, powOut)`
 *    ping-ponged across Doppler updates (pre-spatial, preventing recursive spatial blur drift).
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
uniform float uSmoothing;
uniform int uUseHistory;
uniform int uUseKasai;

layout(location = 0) out vec4 outShade;
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

  // Shadowed or non-Doppler sample: cut immediately so moving structures never carry stale trails
  vec4 center = texelFetch(uCorr, ivec2(si, li), 0);
  if (center.a < 0.25) {
    outShade = vec4(0.0);
    outHist = vec4(0.0);
    return;
  }
  bool isClutter = center.a < 0.75;

  float scale = max(1e-3, uScaleMps);
  bool hasCurr = false;
  float power = 0.0;
  float vEstimate = 0.0;
  float variance = 0.0;

  if (uUseKasai != 0) {
    // 3x3 binomial [1 2 1] resolution-cell averaging strictly within the same class (blood 1.0 vs clutter 0.5)
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
        if (tap.a < 0.25 || abs(tap.a - center.a) >= 0.25) continue;
        float wt = wl * ((ds == 0) ? 2.0 : 1.0);
        sumR0 += wt * tap.r;
        sumR1r += wt * tap.g;
        sumR1i += wt * tap.b;
        wSum += wt;
      }
    }
    if (wSum > 0.0) {
      float R0 = sumR0 / wSum;
      if (R0 > 0.0) {
        float R1r = sumR1r / wSum;
        float R1i = sumR1i / wSum;
        vEstimate = (scale * safeAtan2(R1i, R1r)) / PI;
        float rawPow = max(0.0, R0 - NOISE_POWER);
        power = min(1.0, uGainLin * rawPow);
        float coherence = length(vec2(R1r, R1i)) / R0;
        bool rejectedNoise = coherence < NOISE_COHERENCE && power < NOISE_POWER_GATE * COLOR_THRESHOLD;
        float rawVar = 1.0 - coherence;
        variance = clamp((rawVar - LAMINAR_DECORRELATION) / (1.0 - LAMINAR_DECORRELATION), 0.0, 1.0);
        hasCurr = !rejectedNoise && power >= COLOR_THRESHOLD;
      }
    }
  } else {
    float vTrue = center.r;
    float sigmaV = center.g;
    float rel = center.b;
    if (abs(vTrue) >= uWallFilterMps) {
      power = min(1.0, uGainLin * rel * wallFilterResponse(abs(vTrue), uWallFilterMps));
      vEstimate = vTrue;
      float effDisp = min(1.0, sigmaV / max(0.08, COLOR_TURBULENT_WIDTH_K * abs(vTrue)));
      variance = min(1.0, effDisp * 1.6);
      hasCurr = power >= COLOR_THRESHOLD;
    }
  }

  float pBase = clamp(uPersist, 0.0, 0.95);
  float sm = clamp(uSmoothing, 0.0, 1.0);
  // Tissue/valve clutter never uses temporal persistence or hysteresis (avoids trails on moving myocardium)
  vec4 prev = (!isClutter && uUseHistory != 0) ? texelFetch(uHist, ivec2(si, li), 0) : vec4(0.0);
  bool hasPrev = (!isClutter && uUseHistory != 0) && prev.a > 0.0;

  if (hasCurr) {
    float v = aliasVelocity(vEstimate, scale, uBaselineMps);
    float a1 = (PI * (v - uBaselineMps)) / scale;
    vec2 ph1 = vec2(cos(a1), sin(a1));
    vec2 phOut = ph1;
    float varOut = variance;
    float powOut = power;

    if (hasPrev && pBase > 0.0) {
      float prevMag = max(1e-6, length(prev.rg));
      vec2 ph0 = prev.rg / prevMag;
      float cosDelta = clamp(dot(ph1, ph0), -1.0, 1.0);
      float phaseDelta = acos(cosDelta);
      float phaseJump = smoothstep(0.45, ST_PHASE_JUMP_REF_RAD, phaseDelta);
      float steadyPhase = 1.0 - phaseJump;
      float maxVar = max(variance, prev.b);
      float laminarCue = clamp(1.0 - maxVar * 2.2, 0.0, 1.0) * steadyPhase;
      float turbCue = clamp(maxVar * 1.6, 0.0, 1.0);

      float rise = clamp((power - prev.a - 0.08) / 0.35, 0.0, 1.0);
      float attackScale = 1.0 - (1.0 - ST_ATTACK_PERSIST_SCALE) * rise * sm;
      float dynamicCue = max(turbCue, phaseJump);
      float pRegime = pBase + sm * (
        ST_LAMINAR_PERSIST_BOOST * laminarCue * (1.0 - pBase) -
        (1.0 - ST_TURB_PERSIST_SCALE) * dynamicCue * pBase
      );
      float pEff = clamp(pRegime * attackScale, 0.0, 0.92);

      vec2 blended = (1.0 - pEff) * power * ph1 + pEff * prev.rg;
      float bLen = length(blended);
      phOut = bLen > 1e-6 ? blended / bLen : ph1;

      float pVar = clamp(pEff + 0.12 * sm * laminarCue, 0.0, 0.92);
      varOut = variance * (1.0 - pVar) + prev.b * pVar;
      powOut = power * (1.0 - pEff) + prev.a * pEff;
    }

    outShade = vec4(phOut, varOut, powOut);
    outHist = isClutter ? vec4(0.0) : vec4(phOut * powOut, varOut, powOut);
    return;
  }

  // Momentary stochastic drop-out over valid blood: smooth hysteresis decay instead of binary blink
  if (hasPrev && pBase > 0.0 && sm > 0.0) {
    float decay = (ST_DROPOUT_DECAY_BASE + ST_DROPOUT_DECAY_PERSIST * pBase) * (0.75 + 0.25 * sm);
    float decayedPow = prev.a * decay;
    if (decayedPow >= COLOR_THRESHOLD * ST_HYSTERESIS_RATIO) {
      float prevMag = max(1e-6, length(prev.rg));
      vec2 ph0 = prev.rg / prevMag;
      outShade = vec4(ph0, prev.b, decayedPow);
      outHist = vec4(ph0 * decayedPow, prev.b, decayedPow);
      return;
    }
  }

  outShade = vec4(0.0);
  outHist = vec4(0.0);
}
`;

/**
 * Deferred Doppler Pass D3 — Coherence- & Variance-Adaptive Circular Spatial Filter
 * (isolated-speckle rejection, blood-safe pinhole closure, edge-preserving circular bilateral
 * smoothing, and laminar-vs-turbulent variance contrast enhancement).
 *
 * Output (`RGBA16F`, location 0 `outDisplay`):
 *  `(cos(phi_base), sin(phi_base), variance, 0.5 + 0.5 * powConf)` ready for direct zero-readback
 *  binding to `GLSL_PRESENT_FRAG`'s `uColor` sampler.
 */
export const GLSL_DOPPLER_SPATIAL_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D uShade;    // Pass D2 output (RGBA16F): cos(phi), sin(phi), var, power
uniform sampler2D uCorr;     // Pass D1 output (RGBA32F): .a = 1.0 blood, 0.5 tissue/valve clutter, 0.0 invalid
uniform ivec4 uBoxBounds;    // (siMin, liMin, siMax, liMax) inclusive
uniform float uSmoothing;    // [0..1]
uniform float uGainDb;       // Color gain (dB) for overgain breakthrough (> 4 dB)

layout(location = 0) out vec4 outDisplay;

${dopplerDefinesGlsl()}

float colorPowerConf(float p) {
  return smoothstep(COLOR_THRESHOLD * ST_HYSTERESIS_RATIO, COLOR_THRESHOLD + 0.30, p);
}

float encodeDisplayAlpha(float powVal, bool isClutter) {
  float bt = isClutter
    ? smoothstep(0.25, 0.90, powVal)
    : smoothstep(4.0, 12.0, uGainDb) * 0.45;
  if (bt > 0.0) return 1.0 + bt;
  return 0.5 + 0.5 * colorPowerConf(powVal);
}

void main() {
  int si = int(gl_FragCoord.x);
  int li = int(gl_FragCoord.y);

  vec4 c = texelFetch(uShade, ivec2(si, li), 0);
  float centerClass = texelFetch(uCorr, ivec2(si, li), 0).a;
  bool isClutter = centerClass > 0.25 && centerClass < 0.75;
  float sm = clamp(uSmoothing, 0.0, 1.0);

  if (sm <= 0.0) {
    if (c.a <= 0.0) {
      outDisplay = vec4(0.0);
    } else {
      outDisplay = vec4(c.rg, c.b, encodeDisplayAlpha(c.a, isClutter));
    }
    return;
  }

  float targetClass = (c.a <= 0.0) ? 1.0 : centerClass;
  int nCount = 0;
  float nGeomW = 0.0;
  float nPowSum = 0.0;
  float nVarPowSum = 0.0;
  vec2 nPhSum = vec2(0.0);

  for (int dl = -1; dl <= 1; dl++) {
    int ll = li + dl;
    if (ll < uBoxBounds.y || ll > uBoxBounds.w) continue;
    float wl = (dl == 0) ? 2.0 : 1.0;
    for (int ds = -1; ds <= 1; ds++) {
      if (dl == 0 && ds == 0) continue;
      int ss = si + ds;
      if (ss < uBoxBounds.x || ss > uBoxBounds.z) continue;
      vec4 tap = texelFetch(uShade, ivec2(ss, ll), 0);
      if (tap.a <= 0.0) continue;
      float tapClass = texelFetch(uCorr, ivec2(ss, ll), 0).a;
      if (abs(tapClass - targetClass) >= 0.25) continue;
      float wGeom = wl * ((ds == 0) ? 2.0 : 1.0);
      float wp = wGeom * tap.a;
      nCount++;
      nGeomW += wGeom;
      nPowSum += wp;
      nVarPowSum += wp * tap.b;
      nPhSum += wp * tap.rg;
    }
  }

  // Case 1: Center uncolored -> safe blood-pool pinhole closure strictly on blood (uCorr.a >= 0.75)
  if (c.a <= 0.0) {
    int minPinhole = (sm >= 0.5) ? ST_PINHOLE_MIN_COUNT : 6;
    if (centerClass >= 0.75 && sm >= 0.25 && nCount >= minPinhole && nGeomW >= 7.0 && nPowSum > 1e-6) {
      float meanPow = nPowSum / nGeomW;
      if (meanPow >= ST_PINHOLE_MIN_POWER) {
        float phLen = length(nPhSum);
        vec2 phFill = phLen > 1e-6 ? (nPhSum / phLen) : vec2(1.0, 0.0);
        float varFill = nVarPowSum / nPowSum;
        float powFill = min(1.0, meanPow * (0.65 + 0.25 * sm));
        outDisplay = vec4(phFill, varFill, encodeDisplayAlpha(powFill, false));
        return;
      }
    }
    outDisplay = vec4(0.0);
    return;
  }

  // Case 2: Center colored -> isolated-speckle flicker rejection
  bool isolatedSingle = sm >= 0.2 && nCount == 0;
  bool isolatedWeakPair =
    sm >= 0.4 &&
    (nCount < ST_MIN_NEIGHBOR_COUNT || nGeomW < ST_MIN_NEIGHBOR_WEIGHT * sm) &&
    c.a < 0.72;
  if (isolatedSingle || isolatedWeakPair) {
    outDisplay = vec4(0.0);
    return;
  }

  // Local circular phase coherence & laminar/turbulent regime classification
  float cWeight = 4.0 * c.a;
  float allPowSum = nPowSum + cWeight;
  vec2 allPh = nPhSum + cWeight * c.rg;
  float spatialCoherence = allPowSum > 1e-6 ? min(1.0, length(allPh) / allPowSum) : 1.0;
  float varLocal = allPowSum > 1e-6 ? (nVarPowSum + cWeight * c.b) / allPowSum : c.b;
  float varLaminar = clamp(1.0 - varLocal / 0.42, 0.0, 1.0);
  float cohLaminar = smoothstep(0.55, 0.92, spatialCoherence);
  float laminarFactor = varLaminar * cohLaminar;

  // Edge-preserving circular phase-similarity bilateral gather (strictly within the same class)
  float bCenterW = 2.0 * c.a;
  vec2 bPh = bCenterW * c.rg;
  float bVar = bCenterW * c.b;
  float bW = bCenterW;
  float bPow = 2.0 * c.a;
  float bGeomW = 2.0;

  for (int dl = -1; dl <= 1; dl++) {
    int ll = li + dl;
    if (ll < uBoxBounds.y || ll > uBoxBounds.w) continue;
    float wl = (dl == 0) ? 2.0 : 1.0;
    for (int ds = -1; ds <= 1; ds++) {
      if (dl == 0 && ds == 0) continue;
      int ss = si + ds;
      if (ss < uBoxBounds.x || ss > uBoxBounds.z) continue;
      vec4 tap = texelFetch(uShade, ivec2(ss, ll), 0);
      if (tap.a <= 0.0) continue;
      float tapClass = texelFetch(uCorr, ivec2(ss, ll), 0).a;
      if (abs(tapClass - centerClass) >= 0.25) continue;
      float wGeom = wl * ((ds == 0) ? 2.0 : 1.0);
      float cosDiff = clamp(dot(tap.rg, c.rg), -1.0, 1.0);
      float wPhase = exp(-(1.0 - cosDiff) * ST_PHASE_BILATERAL_K);
      float wBilat = wGeom * tap.a * wPhase;
      bPh += wBilat * tap.rg;
      bVar += wBilat * tap.b;
      bW += wBilat;
      bPow += wGeom * wPhase * tap.a;
      bGeomW += wGeom * wPhase;
    }
  }

  float bLen = length(bPh);
  vec2 fPh = bLen > 1e-6 ? (bPh / bLen) : c.rg;
  float velMix = sm * mix(ST_SPATIAL_TURB_MIX, ST_SPATIAL_LAMINAR_MIX, laminarFactor);
  vec2 outPhRaw = mix(c.rg, fPh, velMix);
  float outLen = length(outPhRaw);
  vec2 outPh = outLen > 1e-6 ? (outPhRaw / outLen) : c.rg;

  float varSmooth = bW > 1e-6 ? (bVar / bW) : c.b;
  float varLamBase = mix(c.b, varSmooth, min(1.0, sm * 0.85));
  float varLaminarClean =
    varLamBase * (1.0 - sm * ST_LAMINAR_VAR_CLEANUP * laminarFactor * clamp(1.0 - varSmooth / 0.25, 0.0, 1.0));

  float turbConfirm = smoothstep(0.18, 0.45, varLocal);
  float varTurb = clamp(
    max(varSmooth, c.b * 0.88 + varSmooth * 0.12) +
      sm * ST_TURB_VAR_BOOST * turbConfirm * (1.0 - spatialCoherence),
    0.0,
    1.0
  );
  float varOut = mix(varTurb, varLaminarClean, laminarFactor);

  float powSmooth = bGeomW > 1e-6 ? (bPow / bGeomW) : c.a;
  float powMix = sm * (0.25 + 0.45 * laminarFactor);
  float powOut = clamp(mix(c.a, powSmooth, powMix), 0.0, 1.0);

  outDisplay = vec4(outPh, varOut, encodeDisplayAlpha(powOut, isClutter));
}
`;

