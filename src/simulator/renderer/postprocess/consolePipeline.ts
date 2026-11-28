import type { AcquisitionSettings, PolarFrame, PolarFrameSpec } from '../types';
import { hash3 } from '@/core/random';
import { buildNoiseKernels, ENVELOPE_NORM, filterComplex, type PsfKernels } from '../acoustic/psf';
import { COMPOUND_LOOKS, focusingGain } from '../acoustic/acoustics';
import { applySriPolar } from './sriFilter';

/**
 * Console post-processing (spec 7.7): linear envelope amplitude → displayed polar intensity 0..255.
 * Steps: complex receiver noise, band-limited by the receive response and detected together with the echo
 * (decision 91) → baseline depth compensation + user TGC + gain, which amplify noise and echo alike, so the
 * noise rises with depth and gain → log compression / dynamic range → edge enhancement → persistence → gray
 * map. Axial and lateral resolution are not console effects: the renderer forms them with the PSF before
 * envelope detection (decision 52). Pure CPU, deterministic per (frameIndex, seed). The GPU console
 * (gpu/glslImage.ts, decision 54) evaluates the same steps except the mirror artifact, with the same compensation
 * table, noise hashes, kernels and constants.
 */
export interface ConsoleState {
  prev: Float32Array | null; // persistence buffer of the CPU console (0..1)
  /** True while the GPU console's history texture holds the previous output of this state (decision 54). */
  gpuHistory: boolean;
  frameIndex: number;
  seed: number;
}

/**
 * Mean envelope of receiver noise alone, before amplification (amplitude units). 0.0018 from decision 91 until the beam
 * gained its focusing profile (decision 144): the walls before the focus lost 3–4 dB while the noise stayed, the deep
 * chambers and lung read at the noise floor above the walls, and the far background rose from 76 to 93 where CAMUS Good
 * shows 79. The floor follows the signal down: the echo levels are the model's own units, the receiver's noise their
 * calibration against the images. 0.0004 with the compensation at 0.7 dB/cm/MHz (the sweep's choice amplifies the deep
 * noise with the deep echoes): 0.0011 left the far background at 111 and the systolic atrium at 106 where CAMUS Good
 * shows 79 and 60; at 0.0004 they read 85 and 77, with the blood's own backscatter raised (0.012 → 0.018) so the
 * cavity keeps its 50 against 56 instead of falling with the noise.
 */
export const NOISE_FLOOR = 0.00045;
/**
 * RMS of the white complex receiver noise before the receive response (decision 91). The response has unit energy, so it
 * keeps E|n|², and a complex Gaussian detects to a mean envelope of √π/2 times its RMS: noise alone reads NOISE_FLOOR.
 * Compounding averages the looks' noise as it averages their speckle (decision 145): the noise of one frame is that of
 * one look over √COMPOUND_LOOKS (the console adds it once, to the compounded envelope).
 */
export const NOISE_RMS = (NOISE_FLOOR * ENVELOPE_NORM) / Math.sqrt(COMPOUND_LOOKS);
/**
 * White point: envelope amplitude 10^(REF_DB/20) maps to full white at 0 dB gain. 12 dB under the 8 it was since decision
 * 52, set with the clinical grey map and the complex receiver noise against CAMUS Good (decision 91).
 */
export const REF_DB = -4;
/**
 * Convexity C of the clinical grey map, grey = ((1 + C)^y − 1) / C of the compressed level y (decision 91): exponential
 * in dB, it expands the bright end, as the grey std of CAMUS Good images grows with grey level (decision 90).
 */
export const CLINICAL_GREY_CURVE = 3.5;
const CLINICAL_GREY_LOG = Math.log1p(CLINICAL_GREY_CURVE);
/** S-curve grey map: smoothstep weighted against the identity. */
export const S_CURVE_MIX = 0.85;
/** High-contrast grey map exponent. */
export const HIGH_CONTRAST_GAMMA = 1.6;
/** Floor added before the log so a zero envelope compresses to −∞ dB without NaN. */
export const LOG_FLOOR = 1e-6;
/** Soft-knee highlight roll-off start (normalized log 0..1) and input span (+16 dB headroom above REF_DB). */
export const LOG_KNEE_START = 0.8;
export const LOG_KNEE_SPAN = 0.48;

/**
 * C¹-continuous monotonic soft-knee highlight compression on normalized log echo `y = (db + DR) / DR`.
 * Exact identity on [0, LOG_KNEE_START] (cavity blood, clutter, and normal myocardium stay untouched) and
 * a cubic Hermite shoulder over [LOG_KNEE_START, LOG_KNEE_START + LOG_KNEE_SPAN] so hyper-reflective
 * pericardium, fibrous trigones, and calcified valves retain internal tonal gradation instead of flat clipping.
 */
export function softKneeCompress(y: number): number {
  if (y <= 0) return 0;
  if (y <= LOG_KNEE_START) return y;
  const w = LOG_KNEE_SPAN;
  if (y >= LOG_KNEE_START + w) return 1;
  const h = 1 - LOG_KNEE_START;
  const m = w / h;
  const t = (y - LOG_KNEE_START) / w;
  return LOG_KNEE_START + h * t * (m + (3 - 2 * m) * t + (m - 2) * t * t);
}
/** 20 / ln(10): exact multiplier to compute 20*log10(x) via faster Math.log(x). */
const LOG10E_20 = 20 / Math.LN10;
/** 4096-entry LUTs for clinical & high-contrast gray curves (< 0.0001 max error vs exact expm1/pow). */
const GRAY_LUT_SIZE = 4096;
const CLINICAL_GRAY_LUT = Uint8Array.from({ length: GRAY_LUT_SIZE + 1 }, (_, i) => {
  const y = i / GRAY_LUT_SIZE;
  return Math.round((Math.expm1(y * CLINICAL_GREY_LOG) / CLINICAL_GREY_CURVE) * 255);
});
const HIGH_CONTRAST_LUT = Uint8Array.from({ length: GRAY_LUT_SIZE + 1 }, (_, i) => {
  const y = i / GRAY_LUT_SIZE;
  return Math.round(Math.pow(y, HIGH_CONTRAST_GAMMA) * 255);
});
const S_CURVE_LUT = Uint8Array.from({ length: GRAY_LUT_SIZE + 1 }, (_, i) => {
  const y = i / GRAY_LUT_SIZE;
  return Math.round((y * y * (3 - 2 * y) * S_CURVE_MIX + y * (1 - S_CURVE_MIX)) * 255);
});
/**
 * Receiver-noise magnitude table: the Rayleigh tail is truncated at this fraction of the CDF (3.3 RMS), the
 * magnitude comes from the high 16 bits of the hash and the phase from its low 10 bits (decision 91).
 */
export const NOISE_TAIL = 0.999999;
export const NOISE_MAGNITUDE_BINS = 65536;
export const NOISE_PHASE_BINS = 1024;

/**
 * Persistence weight of a frame that arrives `elapsedS` after the previous one, for a scanner whose frame interval is
 * `intervalS` (decision 94, external audit F12). The setting is the weight of the history per acquired frame, so over
 * any stretch of time the history keeps persistence^(elapsed / interval): a frame the renderer could not produce on time
 * still counts as elapsed time, and the image does not smear longer on a slow device. A frame that follows the previous
 * one by less than a quarter of the interval (a late step followed by an early one) weighs as a quarter, so it still
 * contributes.
 */
export function persistenceOverTime(
  persistence: number,
  elapsedS: number,
  intervalS: number,
): number {
  if (!(persistence > 0) || !(intervalS > 0)) return 0;
  return Math.pow(persistence, Math.max(0.25, elapsedS / intervalS));
}

export function createConsoleState(seed: number): ConsoleState {
  return { prev: null, gpuHistory: false, frameIndex: 0, seed };
}

const scratchA = { buf: new Float32Array(0) };
const scratchB = { buf: new Float32Array(0) };
const scratchComp = { buf: new Float64Array(0) };
const scratchNoiseRe = { buf: new Float32Array(0) };
const scratchNoiseIm = { buf: new Float32Array(0) };
const scratchTmpRe = { buf: new Float32Array(0) };
const scratchTmpIm = { buf: new Float32Array(0) };
const noiseKernelCache = new Map<string, PsfKernels>();
/**
 * One 32-bit hash per noise sample: its high 16 bits draw the magnitude (Rayleigh, E|z|² = NOISE_RMS², tail truncated at
 * 3.3 RMS, where |z|² exceeds it with probability 1.5·10⁻⁵) and its low 10 bits the phase; the GPU passes evaluate the
 * same expressions on the same bits.
 */
const NOISE_MAGNITUDE = Float32Array.from(
  { length: NOISE_MAGNITUDE_BINS },
  (_, i) => NOISE_RMS * Math.sqrt(-Math.log(1 - NOISE_TAIL * (i / NOISE_MAGNITUDE_BINS))),
);
const NOISE_COS = Float32Array.from({ length: NOISE_PHASE_BINS }, (_, i) =>
  Math.cos((2 * Math.PI * i) / NOISE_PHASE_BINS),
);
const NOISE_SIN = Float32Array.from({ length: NOISE_PHASE_BINS }, (_, i) =>
  Math.sin((2 * Math.PI * i) / NOISE_PHASE_BINS),
);

/** Receive response of the noise for a frame or M-mode line geometry (a few geometries alternate: frames and lines). */
function noiseKernels(spec: PolarFrameSpec, settings: AcquisitionSettings): PsfKernels {
  const key = `${spec.lines}x${spec.samples}|${spec.depthCm}|${spec.sectorRad}|${spec.focusCm}|${settings.frequencyMHz}|${settings.harmonics ? 1 : 0}`;
  let k = noiseKernelCache.get(key);
  if (!k) {
    if (noiseKernelCache.size >= 8) noiseKernelCache.clear();
    k = buildNoiseKernels(spec, settings.frequencyMHz, settings.harmonics);
    noiseKernelCache.set(key, k);
  }
  return k;
}

/**
 * Receiver noise of one frame or pulse, in place in `re`/`im` (decision 91): white complex Gaussian samples per line,
 * sample, frame index and seed, filtered by the receive response.
 */
export function receiverNoise(
  k: PsfKernels,
  lines: number,
  samples: number,
  frameIndex: number,
  seed: number,
  re: Float32Array,
  im: Float32Array,
  tmpRe: Float32Array,
  tmpIm: Float32Array,
): void {
  for (let li = 0; li < lines; li++)
    for (let si = 0; si < samples; si++) {
      const h = hash3(li, si, frameIndex, seed) * 4294967296;
      const r = NOISE_MAGNITUDE[h >>> 16]!,
        q = h & (NOISE_PHASE_BINS - 1);
      re[li * samples + si] = r * NOISE_COS[q]!;
      im[li * samples + si] = r * NOISE_SIN[q]!;
    }
  filterComplex(re, im, lines, samples, k, tmpRe, tmpIm);
}

/** A view of n samples on a scratch buffer that only grows: frames and M-mode lines alternate without reallocating. */
function ensure(s: { buf: Float32Array }, n: number): Float32Array {
  if (s.buf.length < n) s.buf = new Float32Array(n);
  return s.buf.length === n ? s.buf : s.buf.subarray(0, n);
}

/**
 * Default depth compensation of the (fictional) console, dB per cm and MHz of the two-way path. 0.38 until the heart was
 * moved behind the chest wall (decision 66), then 0.45: the extra 1.2 cm of wall on the path attenuated more than the
 * compensation returned and myocardial grey fell from 100+ to 94 at 0 dB gain. A scanner's own curve assumes the
 * soft-tissue attenuation of ~0.5 dB/cm/MHz each way, 1.0 dB/cm/MHz two-way, whatever lies on the path; the case's
 * `AcquisitionSettings.depthCompensationDbPerCmMHz` overrides it (the clinical console sweep varies it, decision 144).
 * 0.7 since the beam has its focusing profile and the receiver its lower noise floor (decision 144): the sweep against
 * CAMUS Good over cavity, myocardium, atrium, contrast and the surroundings of the ventricle scored 0.64 at 0.7 dB/cm/MHz
 * with the clinical map at 70 dB and 0 dB gain, against 0.71 at 0.45.
 */
export const DEPTH_COMPENSATION_DB_PER_CM_MHZ = 0.7;

/**
 * Per-sample amplification of the console: default depth compensation plus the TGC curve, capped at 60 dB, times the
 * overall gain. The CPU console uses it in float64, the GPU console as a texture.
 */
export function consoleCompensation(
  settings: AcquisitionSettings,
  spec: PolarFrameSpec,
  out: Float32Array | Float64Array,
): void {
  const dr = spec.depthCm / spec.samples;
  const baselineDbPerCm =
    (settings.depthCompensationDbPerCmMHz ?? DEPTH_COMPENSATION_DB_PER_CM_MHZ) *
    settings.frequencyMHz;
  const gainLin = Math.pow(10, settings.gainDb / 20);
  for (let si = 0; si < spec.samples; si++) {
    const r = (si + 0.5) * dr;
    // diffraction correction: the default curve gives back what the beam loses by widening beyond its focus, and
    // nothing of what it loses in front of it — a scanner's near gain leaves the chest wall to fall (decision 144)
    const diffractionDb = -20 * Math.log10(focusingGain(Math.max(r, spec.focusCm), spec.focusCm));
    const compDb = baselineDbPerCm * r + diffractionDb + tgcAtDepth(settings, r);
    out[si] = Math.pow(10, Math.min(compDb, 60) / 20) * gainLin;
  }
}

export function tgcAtDepth(settings: AcquisitionSettings, r: number): number {
  const bands = settings.tgcDb;
  const pos = (r / settings.depthCm) * (bands.length - 1);
  const i0 = Math.max(0, Math.min(bands.length - 1, Math.floor(pos)));
  const i1 = Math.min(bands.length - 1, i0 + 1);
  const t = pos - i0;
  return (bands[i0] ?? 0) * (1 - t) + (bands[i1] ?? 0) * t;
}

/**
 * Case-configurable artifacts (spec 12, case 12): intensities 0..1. The console applies only the mirror: the image beyond
 * a strong specular interface (pericardium/lung) repeats the shallower image. Beam width and side lobes belong to the
 * renderer PSF (the first widens the lateral beam away from the focus, the second raises its side lobes, decision 155)
 * and are carried here only for completeness.
 */
export interface ArtifactSettings {
  sideLobe: number;
  mirror: number;
  beamWidth: number;
}
export const NO_ARTIFACTS: ArtifactSettings = { sideLobe: 0, mirror: 0, beamWidth: 0 };

/**
 * Options of an M-mode column (decision 84): `noisePulses` pulses of one column, each detected with its own receiver noise
 * and averaged per sample (decision 91), `edgeStep` samples between the centre and the neighbours of the edge enhancement
 * (a line finer than a frame keeps the frame's enhancement length), and no persistence history.
 */
export interface ConsoleLineOptions {
  noisePulses: number;
  edgeStep: number;
}

export function applyConsole(
  frame: PolarFrame,
  settings: AcquisitionSettings,
  state: ConsoleState,
  outU8: Uint8ClampedArray,
  artifacts: ArtifactSettings = NO_ARTIFACTS,
  line: ConsoleLineOptions | null = null,
): void {
  const { lines, samples, depthCm } = frame.spec;
  const n = lines * samples;
  const dr = depthCm / samples;
  const dr_ = settings.dynamicRangeDb;
  const a = ensure(scratchA, n);
  const b = ensure(scratchB, n);
  const fi = state.frameIndex;
  if (scratchComp.buf.length < samples) scratchComp.buf = new Float64Array(samples);
  const compensation = scratchComp.buf;
  consoleCompensation(settings, frame.spec, compensation);
  const pulses = line ? Math.max(1, Math.round(line.noisePulses)) : 1;
  // 1) receiver noise joins the echo before detection (decision 91). Adding its envelope to the detected echo, as before,
  // filled every speckle null with a positive offset; white noise per sample drew a grain finer than the resolution.
  // |A + n| with n complex has the distribution of |A·e^{iθ} + n| for any echo phase θ, so the detected echo serves as
  // A. The pulses of an M-mode column average their detected envelopes. Amplification follows, for noise and echo alike.
  const k = noiseKernels(frame.spec, settings);
  const nRe = ensure(scratchNoiseRe, n),
    nIm = ensure(scratchNoiseIm, n),
    tRe = ensure(scratchTmpRe, n),
    tIm = ensure(scratchTmpIm, n);
  a.fill(0);
  for (let p = 0; p < pulses; p++) {
    receiverNoise(k, lines, samples, fi + p, state.seed, nRe, nIm, tRe, tIm);
    for (let i = 0; i < n; i++) {
      const x = (frame.amplitude[i] ?? 0) + nRe[i]!,
        y = nIm[i]!;
      a[i]! += Math.sqrt(x * x + y * y);
    }
  }
  for (let si = 0; si < samples; si++) {
    const g = (compensation[si] ?? 1) / pulses;
    for (let li = 0; li < lines; li++) a[li * samples + si]! *= g;
  }
  // 1b) mirror artifact: beyond the first strong specular interface deeper than 5 cm (pericardium / pleura)
  // the shallower image is duplicated at mirrored depth, attenuated
  if (artifacts.mirror > 0) {
    const minSi = Math.floor(5 / dr);
    for (let li = 0; li < lines; li++) {
      const base = li * samples;
      let s0 = -1;
      for (let si = minSi; si < samples; si++) {
        const ti = frame.tissue[base + si] ?? 0;
        if ((ti === 4 || ti === 9) && (frame.amplitude[base + si] ?? 0) > 0.3) {
          s0 = si;
          break;
        }
      }
      if (s0 < 0) continue;
      const gain = 0.45 * artifacts.mirror;
      for (let si = s0 + 1; si < samples; si++) {
        const src = 2 * s0 - si;
        if (src < 0) break;
        a[base + si] =
          (a[base + si] ?? 0) + (a[base + src] ?? 0) * gain * Math.exp(-(si - s0) * dr * 0.12);
      }
    }
  }
  // 2) axial resolution is formed by the renderer PSF (decision 52)
  // 3) log compression / dynamic range with C¹ soft-knee highlight shoulder → 0..1 (in-place on `a`)
  const invDr = 1 / dr_;
  for (let i = 0; i < n; i++) {
    const db = Math.log(a[i]! + LOG_FLOOR) * LOG10E_20 - REF_DB;
    a[i] = softKneeCompress((db + dr_) * invDr);
  }
  // 4) structure-oriented anisotropic speckle reduction (SRI / XRES-style) on 2D frames
  const sri = !line ? (settings.sri ?? 0) : 0;
  let cur = a;
  let tmp = b;
  if (sri > 0 && lines >= 3 && samples >= 3) {
    applySriPolar(cur, tmp, lines, samples, sri);
    cur = b;
    tmp = a;
  }
  // 5) edge enhancement (unsharp along samples, mild)
  if (settings.edgeEnhance > 0) {
    const e = settings.edgeEnhance * 0.8;
    const k = line ? Math.max(1, Math.round(line.edgeStep)) : 1;
    for (let li = 0; li < lines; li++) {
      const base = li * samples;
      for (let si = k; si < samples - k; si++) {
        const c = cur[base + si]!;
        const m = (cur[base + si - k]! + cur[base + si + k]!) * 0.5;
        const v = c + (c - m) * e;
        tmp[base + si] = v <= 0 ? 0 : v >= 1 ? 1 : v;
      }
      for (let si = 0; si < k; si++) {
        tmp[base + si] = cur[base + si]!;
        tmp[base + samples - 1 - si] = cur[base + samples - 1 - si]!;
      }
    }
    const swap = cur;
    cur = tmp;
    tmp = swap;
  }
  // 6) persistence (reuse `state.prev` in-place to avoid per-frame TypedArray allocation)
  const p = line ? 0 : settings.persistence;
  if (line) {
    // an M-mode column has no history: each column is its own instant
  } else if (state.prev && state.prev.length === n && p > 0) {
    const prev = state.prev;
    const oneMinusP = 1 - p;
    for (let i = 0; i < n; i++) {
      const y = cur[i]! * oneMinusP + prev[i]! * p;
      cur[i] = y;
      prev[i] = y;
    }
  } else if (state.prev && state.prev.length === n) {
    state.prev.set(cur);
  } else {
    state.prev = new Float32Array(cur);
  }
  // 7) gray map (O(1) 4096-entry LUT for non-linear curves)
  const gm = settings.grayMap;
  if (gm === 'clinical') {
    for (let i = 0; i < n; i++) outU8[i] = CLINICAL_GRAY_LUT[(cur[i]! * GRAY_LUT_SIZE + 0.5) | 0]!;
  } else if (gm === 's-curve') {
    for (let i = 0; i < n; i++) outU8[i] = S_CURVE_LUT[(cur[i]! * GRAY_LUT_SIZE + 0.5) | 0]!;
  } else if (gm === 'high-contrast') {
    for (let i = 0; i < n; i++) outU8[i] = HIGH_CONTRAST_LUT[(cur[i]! * GRAY_LUT_SIZE + 0.5) | 0]!;
  } else {
    for (let i = 0; i < n; i++) outU8[i] = (cur[i]! * 255 + 0.5) | 0;
  }
  state.gpuHistory = false;
  state.frameIndex++;
}
