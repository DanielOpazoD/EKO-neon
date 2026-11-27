import { dopplerVelocityFromShiftHz } from '@/clinical/formulas';
import { SPEED_OF_SOUND_MPS } from '@/core/units';
import { hash3 } from '@/core/random';
import { APERTURE_MM } from '@/simulator/probe/transducer';
import {
  clutterFilterPowerResponse,
  depthLimitedPulsedAcquisition,
  scattererTransitTimeS,
} from './scatteringModel';
import {
  SPECTRAL_BINS,
  spectralRange,
  type SpectralSettings,
  type VelocitySample,
} from './spectrum';

/**
 * One contiguous slow-time packet. When `gateDepthCm` is supplied, the intra-packet PRF is
 * physically bounded by the round-trip acoustic travel time to the far edge of the gate
 * (`gateDepthCm + gateLengthCm / 2`), enforcing depth-dependent Nyquist aliasing directly in I/Q.
 */
export function pulsedPacketTiming(
  scaleMps: number,
  frequencyMHz: number,
  count = SPECTRAL_BINS,
  gateDepthCm?: number,
  gateLengthCm = 0,
) {
  const effectiveScaleMps =
    gateDepthCm !== undefined && Number.isFinite(gateDepthCm)
      ? depthLimitedPulsedAcquisition(gateDepthCm, gateLengthCm, frequencyMHz, scaleMps)
          .effectiveScaleMps
      : scaleMps;
  const prfHz = (4 * frequencyMHz * 1e6 * effectiveScaleMps) / SPEED_OF_SOUND_MPS;
  if (!(prfHz > 0) || !Number.isFinite(prfHz)) throw new Error('Invalid pulsed acquisition');
  return { prfHz, effectiveScaleMps, durationS: count / prfHz };
}

/** In-place radix-2 complex FFT, forward convention exp(-i 2πkn/N). */
export function complexFft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  if (n !== im.length || n < 2 || n & (n - 1))
    throw new Error('FFT needs equal power-of-two arrays');
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j]!, re[i]!];
      [im[i], im[j]] = [im[j]!, im[i]!];
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const c = Math.cos((-2 * Math.PI) / size),
      s = Math.sin((-2 * Math.PI) / size);
    for (let start = 0; start < n; start += size) {
      let wr = 1,
        wi = 0;
      for (let j = 0; j < size / 2; j++) {
        const a = start + j,
          b = a + size / 2;
        const tr = wr * re[b]! - wi * im[b]!,
          ti = wr * im[b]! + wi * re[b]!;
        re[b] = re[a]! - tr;
        im[b] = im[a]! - ti;
        re[a] = re[a]! + tr;
        im[a] = im[a]! + ti;
        const next = wr * c - wi * s;
        wi = wr * s + wi * c;
        wr = next;
      }
    }
  }
}

export interface IqTone {
  velocityMps: number;
  power: number;
  phaseRad: number;
  /**
   * Optional Gaussian residence duration (s) of the scatterer crossing the finite sample volume.
   * When finite and comparable to packet duration N/PRF, the slow-time envelope naturally produces
   * physical transit-time spectral broadening in the FFT.
   */
  transitTimeS?: number;
  /** Fractional packet centre (0..1) where the scatterer peaks inside the gate; defaults to 0.5. */
  transitCenterFrac?: number;
  /** Optional axial acceleration (m/s²) across the slow-time packet for physical chirp broadening. */
  accelMps2?: number;
}

let envScratch = new Float64Array(SPECTRAL_BINS);

/**
 * Fill `out[0..n-1]` with `exp(-(i - center)^2 * inv2Sig2)` using an exact float64 outward
 * recurrence from the nearest peak sample `k0`, avoiding underflow and replacing 2*N `Math.exp`
 * calls per tone with 4 `Math.exp` calls. Returns the RMS power-preserving normalization factor.
 */
function fillGaussianEnvelope(
  n: number,
  center: number,
  inv2Sig2: number,
  out: Float64Array,
): number {
  const k0 = Math.max(0, Math.min(n - 1, Math.round(center)));
  const d0 = k0 - center;
  const peak = Math.exp(-d0 * d0 * inv2Sig2);
  out[k0] = peak;
  let energy = peak * peak;
  const stepDecay = Math.exp(-2 * inv2Sig2);

  let val = peak;
  let rRight = Math.exp(-(2 * d0 + 1) * inv2Sig2);
  for (let i = k0 + 1; i < n; i++) {
    val *= rRight;
    if (val < 1e-16) {
      out.fill(0, i, n);
      break;
    }
    out[i] = val;
    energy += val * val;
    rRight *= stepDecay;
  }

  val = peak;
  let rLeft = Math.exp((2 * d0 - 1) * inv2Sig2);
  for (let i = k0 - 1; i >= 0; i--) {
    val *= rLeft;
    if (val < 1e-16) {
      out.fill(0, 0, i + 1);
      break;
    }
    out[i] = val;
    energy += val * val;
    rLeft *= stepDecay;
  }

  return Math.sqrt(n / Math.max(1e-6, energy));
}

function synthesizeIqRange(
  tones: readonly IqTone[],
  count: number,
  frequencyMHz: number,
  prfHz: number,
  timeS: number,
  re: Float64Array,
  im: Float64Array,
): void {
  re.fill(0);
  im.fill(0);
  const n = re.length;
  if (n > envScratch.length) envScratch = new Float64Array(n);
  const env = envScratch;
  const dt = 1 / prfHz;
  const packetDurationS = n * dt;
  const halfIdx = 0.5 * (n - 1);
  const twoPi = 2 * Math.PI;
  const dopplerScale = (2 * frequencyMHz * 1e6) / SPEED_OF_SOUND_MPS;

  for (let t = 0; t < count; t++) {
    const tone = tones[t]!;
    if (!(tone.power > 0)) continue;
    const fd = dopplerScale * tone.velocityMps;
    const angle = (twoPi * fd) / prfHz;
    const baseAmp = Math.sqrt(tone.power);
    const accel = tone.accelMps2 ?? 0;
    const useChirp = Math.abs(accel) > 0.25;
    const tau = tone.transitTimeS;
    const useTransit = tau !== undefined && Number.isFinite(tau) && tau > 0 && tau < 3 * packetDurationS;

    if (useChirp) {
      // Exact quadratic phase centred at packet midpoint: v(t_rel) = v + a * t_rel
      const chirpCoeff = (Math.PI * dopplerScale) * accel;
      const chirpStep = chirpCoeff * dt * dt;
      const centerTimeS = timeS + halfIdx * dt;
      const tRel0 = -halfIdx * dt;
      const phi0 =
        tone.phaseRad +
        twoPi * fd * centerTimeS +
        twoPi * fd * tRel0 +
        chirpCoeff * tRel0 * tRel0;
      let ar = baseAmp * Math.cos(phi0);
      let ai = baseAmp * Math.sin(phi0);
      const dphi0 = angle + chirpStep * (1 - 2 * halfIdx);
      let wr = Math.cos(dphi0);
      let wi = Math.sin(dphi0);
      const dr = Math.cos(2 * chirpStep);
      const di = Math.sin(2 * chirpStep);

      if (useTransit) {
        const center = (tone.transitCenterFrac ?? 0.5) * (n - 1);
        const sigmaIdx = Math.max(2.5, (tau * 0.55) / dt);
        const inv2Sig2 = 1 / (2 * sigmaIdx * sigmaIdx);
        const envNorm = fillGaussianEnvelope(n, center, inv2Sig2, env);
        for (let i = 0; i < n; i++) {
          const w = env[i]! * envNorm;
          re[i] = re[i]! + ar * w;
          im[i] = im[i]! + ai * w;
          const nextA = ar * wr - ai * wi;
          ai = ar * wi + ai * wr;
          ar = nextA;
          const nextW = wr * dr - wi * di;
          wi = wr * di + wi * dr;
          wr = nextW;
        }
      } else {
        for (let i = 0; i < n; i++) {
          re[i] = re[i]! + ar;
          im[i] = im[i]! + ai;
          const nextA = ar * wr - ai * wi;
          ai = ar * wi + ai * wr;
          ar = nextA;
          const nextW = wr * dr - wi * di;
          wi = wr * di + wi * dr;
          wr = nextW;
        }
      }
      continue;
    }

    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const phase = tone.phaseRad + twoPi * fd * timeS;
    let ar = baseAmp * Math.cos(phase);
    let ai = baseAmp * Math.sin(phase);
    if (useTransit) {
      // Normalise Gaussian transit envelope to preserve mean scatterer power while broadening spectrum.
      const center = (tone.transitCenterFrac ?? 0.5) * (n - 1);
      const sigmaIdx = Math.max(2.5, (tau * 0.55) / dt);
      const inv2Sig2 = 1 / (2 * sigmaIdx * sigmaIdx);
      const envNorm = fillGaussianEnvelope(n, center, inv2Sig2, env);
      for (let i = 0; i < n; i++) {
        const w = env[i]! * envNorm;
        re[i] = re[i]! + ar * w;
        im[i] = im[i]! + ai * w;
        const next = ar * c - ai * s;
        ai = ar * s + ai * c;
        ar = next;
      }
    } else {
      for (let i = 0; i < n; i++) {
        re[i] = re[i]! + ar;
        im[i] = im[i]! + ai;
        const next = ar * c - ai * s;
        ai = ar * s + ai * c;
        ar = next;
      }
    }
  }
}

/**
 * Complex baseband from discrete slow-time phase increments 2*pi*(2*f0*v/c)/PRF.
 * Aliasing emerges directly from discrete slow-time sampling whenever |v| > c*PRF/(4*f0),
 * finite scatterer dwell time modulates amplitude across the packet for physical transit broadening,
 * and axial acceleration integrates exact quadratic slow-time phase (intra-packet Doppler chirp).
 */
export function synthesizeIq(
  tones: readonly IqTone[],
  frequencyMHz: number,
  prfHz: number,
  timeS: number,
  re: Float64Array,
  im: Float64Array,
): void {
  synthesizeIqRange(tones, tones.length, frequencyMHz, prfHz, timeS, re, im);
}

const PACKET = SPECTRAL_BINS;
/**
 * Maximum slow-time STFT window duration (s) for pulsed Doppler.
 * At high PW PRFs (>= 4 kHz), all 128 pulses span <= 32 ms; at low TDI PRFs (~1.0–1.5 kHz),
 * clinical scanners shorten the slow-time window to ~32–48 pulses and zero-pad into the 128-bin
 * FFT so rapid e'/a' myocardial deflections stay temporally crisp (~1.3 cm/s bandwidth) instead
 * of smearing over ~100 ms.
 */
const MAX_SLOW_TIME_WINDOW_S = 0.032;
const MIN_ACTIVE_PULSES = 32;
const REALIZATIONS = 4;
// 7-point midpoint quadrature of |N(0,1)|: Phi^-1((1 + (k+0.5)/7)/2).
// Eliminates discrete micro-striations while preserving exact modal peak location.
const HALF_NORMAL_QUANTILES = [
  0.08964234601606024, 0.2718802221644888, 0.4637080482311739, 0.6744897501960817,
  0.9208229761614358, 1.2418670193638584, 1.8027428562655142,
] as const;
const MICROSCATTERERS = HALF_NORMAL_QUANTILES.length;
const re = new Float64Array(PACKET),
  im = new Float64Array(PACKET);
const cwReBySize = new Map<number, Float64Array>();
const cwImBySize = new Map<number, Float64Array>();
function cwBuffersFor(fftSize: number): { re: Float64Array; im: Float64Array } {
  let r = cwReBySize.get(fftSize);
  let i = cwImBySize.get(fftSize);
  if (!r || !i) {
    r = new Float64Array(fftSize);
    i = new Float64Array(fftSize);
    cwReBySize.set(fftSize, r);
    cwImBySize.set(fftSize, i);
  }
  return { re: r, im: i };
}
const CW_STRATA = 128;
// Potential-core (q=0) plus 6-point midpoint quadrature of |N(0,1)| for the turbulent shear layer:
// erythrocytes traversing a stenotic/regurgitant vena contracta potential core remain at peak
// Bernoulli velocity (crisp outer spectral rim) while the surrounding shear layer fills toward zero.
const CW_SHEAR_QUANTILES = [
  0,
  0.10463261332245658,
  0.31863936396437514,
  0.5485222845934818,
  0.8122178056006126,
  1.1503493803760079,
  1.731664400668443,
] as const;
const CW_CORE_POWER_SHARE = 0.32;
const CW_SHEAR_POWER_SHARE = (1 - CW_CORE_POWER_SHARE) / (CW_SHEAR_QUANTILES.length - 1);
const cwStrataW = new Float64Array(CW_STRATA);
const cwStrataV = new Float64Array(CW_STRATA);
const cwStrataDisp = new Float64Array(CW_STRATA);
const cwStrataVPerp = new Float64Array(CW_STRATA);
const cwStrataDepth = new Float64Array(CW_STRATA);
const cwStrataFocus = new Float64Array(CW_STRATA);
const cwMergedPool: VelocitySample[] = Array.from({ length: CW_STRATA }, () => ({
  v: 0,
  weight: 0,
  dispersion: 0,
  vPerp: 0,
  depthCm: 8,
  focusCm: 9,
  gateLengthCm: null,
}));
const tonePool: IqTone[] = [];
const hannCache = new Map<number, { win: Float64Array; power: number }>();
function hannWindowFor(len: number): { win: Float64Array; power: number } {
  let entry = hannCache.get(len);
  if (!entry) {
    const win = Float64Array.from(
      { length: len },
      (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / len),
    );
    let power = 0;
    for (let i = 0; i < len; i++) power += win[i]! * win[i]!;
    entry = { win, power };
    hannCache.set(len, entry);
  }
  return entry;
}

/**
 * IQ → Hann window → FFT power → smooth high-pass sampled-frequency wall filter → displayed bins.
 * Depth-dependent PRF cap (`gateDepthCm`) enforces true physical Nyquist aliasing directly during
 * slow-time sampling, and finite sample-volume transit time broadens fast/short-gate echoes naturally.
 */
export function accumulatePulsedSpectrum(
  samples: readonly VelocitySample[],
  s: SpectralSettings,
  frequencyMHz: number,
  seed: number,
  timeS: number,
  out: Float32Array,
  gateDepthCm?: number,
  focusCm = 9,
): void {
  const { prfHz, effectiveScaleMps } = pulsedPacketTiming(
    s.scaleMps,
    frequencyMHz,
    PACKET,
    gateDepthCm,
    s.gateLengthCm,
  );
  const activePulses = Math.max(
    MIN_ACTIVE_PULSES,
    Math.min(PACKET, Math.round(prfHz * MAX_SLOW_TIME_WINDOW_S)),
  );
  const activeDurationS = activePulses / prfHz;
  const { win, power: activeWinPower } = hannWindowFor(activePulses);
  const activeRe = activePulses === PACKET ? re : re.subarray(0, activePulses);
  const activeIm = activePulses === PACKET ? im : im.subarray(0, activePulses);
  // Display mapping uses the physical Nyquist span when depth-limited so wrapped bins align with the scale.
  const effectiveSettings =
    effectiveScaleMps < s.scaleMps ? { ...s, scaleMps: effectiveScaleMps } : s;
  const { vMin, vMax } = spectralRange(effectiveSettings),
    span = vMax - vMin;
  out.fill(0);
  // Decorrelate scatterer phase constellations across consecutive overlapping packets deterministically.
  const tCell = Math.floor(timeS / Math.max(1e-4, activeDurationS * 0.5));
  const packetSeed = (seed ^ Math.imul(tCell, 0x9e3779b1)) | 0;
  for (let ensemble = 0; ensemble < REALIZATIONS; ensemble++) {
    let toneCount = 0;
    for (let j = 0; j < samples.length; j++) {
      const p = samples[j]!;
      if (!(p.weight > 0)) continue;
      const depth = p.depthCm ?? gateDepthCm ?? 8;
      const aperture =
        p.vPerp && depth > 0 ? (Math.abs(p.vPerp) * (APERTURE_MM / 10)) / depth : 0;
      const gateLen = p.gateLengthCm ?? s.gateLengthCm;
      const tau = scattererTransitTimeS(p.v, p.vPerp ?? 0, depth, p.focusCm ?? focusCm, gateLen);
      const baseAccel = p.accelMps2 ?? 0;
      const subPower = p.weight / MICROSCATTERERS;
      const absV = Math.abs(p.v);
      const signV = Math.sign(p.v);
      const hasV = absV > 0.05;
      for (let k = 0; k < MICROSCATTERERS; k++) {
        // Half-normal deceleration, capped at zero; aperture rays span D/F uniformly.
        const turbulent = Math.min(absV, 0.9 * p.dispersion * absV * HALF_NORMAL_QUANTILES[k]!);
        const geometric = aperture * (hash3(j, k, ensemble + 103, packetSeed) - 0.5);
        const v = p.v - signV * turbulent + geometric;
        const accelScale = hasV ? Math.max(0, Math.min(1.2, v / p.v)) : 1;
        let slot = tonePool[toneCount];
        if (!slot) {
          slot = {
            velocityMps: 0,
            power: 0,
            phaseRad: 0,
            transitTimeS: undefined,
            transitCenterFrac: 0.5,
            accelMps2: 0,
          };
          tonePool[toneCount] = slot;
        }
        slot.velocityMps = v;
        slot.power = subPower;
        slot.phaseRad = 2 * Math.PI * hash3(j, k, ensemble + 211, packetSeed);
        slot.transitTimeS = tau;
        slot.transitCenterFrac = 0.25 + 0.5 * hash3(j, k, ensemble + 307, packetSeed);
        slot.accelMps2 = baseAccel * accelScale;
        toneCount++;
      }
    }
    // Strip timestamps denote the active packet centre, not its first pulse.
    synthesizeIqRange(
      tonePool,
      toneCount,
      frequencyMHz,
      prfHz,
      timeS - activeDurationS / 2,
      activeRe,
      activeIm,
    );
    for (let i = 0; i < activePulses; i++) {
      re[i] = re[i]! * win[i]!;
      im[i] = im[i]! * win[i]!;
    }
    if (activePulses < PACKET) {
      re.fill(0, activePulses);
      im.fill(0, activePulses);
    }
    complexFft(re, im);
    const f0Hz = frequencyMHz * 1e6;
    const normDenom = activePulses * activeWinPower * REALIZATIONS;
    for (let k = 0; k < PACKET; k++) {
      const signed = k < PACKET / 2 ? k : k - PACKET;
      // Exact ASE frequency-to-velocity calibration: fd = (signed / N) * PRF -> v = fd * c / (2 * f0).
      const fdHz = (signed / PACKET) * prfHz;
      const sampledVelocity = dopplerVelocityFromShiftHz(fdHz, f0Hz, 1, SPEED_OF_SOUND_MPS);
      const wfGain = clutterFilterPowerResponse(sampledVelocity, effectiveSettings.wallFilterMps);
      if (wfGain <= 0) continue;
      const v = effectiveSettings.invert ? -sampledVelocity : sampledVelocity;
      const b = ((vMax - v) / span) * PACKET - 0.5;
      const floor = Math.floor(b),
        frac = b - floor;
      const power = ((re[k]! ** 2 + im[k]! ** 2) * wfGain) / normDenom;
      // Baseline is a circular display shift only; it cannot change the wall filter's physical stop band.
      const b0 = ((floor % PACKET) + PACKET) % PACKET,
        b1 = (b0 + 1) % PACKET;
      out[b0] = out[b0]! + power * (1 - frac);
      out[b1] = out[b1]! + power * frac;
    }
  }
}

/**
 * Unaliased continuous-wave baseband ADC timing covering [-vNyqMps, +vNyqMps] without PRF ceiling.
 * Because CW transmits and receives continuously, there is no round-trip pulse-echo Nyquist limit;
 * the baseband digitizer simply runs fast enough that scatterers inside and beyond the displayed
 * scale do not wrap back onto the opposite channel.
 */
export function continuousWavePacketTiming(
  vNyqMps: number,
  frequencyMHz: number,
  fftSize: number,
): { fsHz: number; vNyqMps: number; durationS: number } {
  const safeVNyq = Math.max(0.1, vNyqMps);
  const fsHz = (4 * frequencyMHz * 1e6 * safeVNyq) / SPEED_OF_SOUND_MPS;
  if (!(fsHz > 0) || !Number.isFinite(fsHz)) throw new Error('Invalid continuous-wave acquisition');
  return { fsHz, vNyqMps: safeVNyq, durationS: fftSize / fsHz };
}

/**
 * Continuous-wave baseband I/Q -> Hann window -> complex FFT -> per-bin clutter filter -> display bins.
 * Shares the same coherent I/Q synthesis, transit-time envelope, aperture broadening, and smooth
 * high-pass clutter filter response as `accumulatePulsedSpectrum`, while sampling baseband I/Q fast
 * enough to prevent aliasing and discarding out-of-range velocity bins instead of wrapping them.
 */
export function accumulateContinuousWaveSpectrum(
  samples: readonly VelocitySample[],
  s: SpectralSettings,
  frequencyMHz: number,
  seed: number,
  timeS: number,
  out: Float32Array,
  focusCm = 9,
): void {
  out.fill(0);
  const { vMin, vMax } = spectralRange(s);
  const span = vMax - vMin;
  if (!(span > 0) || samples.length === 0) return;

  let maxSampleAbs = 0;
  let totalWeight = 0;
  for (let j = 0; j < samples.length; j++) {
    const p = samples[j]!;
    if (!(p.weight > 0)) continue;
    totalWeight += p.weight;
    const depth = p.depthCm ?? 8;
    const apHalf = p.vPerp && depth > 0 ? (0.5 * Math.abs(p.vPerp) * (APERTURE_MM / 10)) / depth : 0;
    const reach = Math.abs(p.v) + apHalf;
    if (reach > maxSampleAbs) maxSampleAbs = reach;
  }
  if (!(totalWeight > 0)) return;

  const dispMaxAbs = Math.max(Math.abs(vMin), Math.abs(vMax));
  const vNyq = Math.max(0.1, dispMaxAbs * 1.05, maxSampleAbs * 1.15);
  // Keep FFT velocity bin spacing <= display bin width even when fast jets exceed a zoomed CW scale.
  const targetBins = Math.ceil((2 * vNyq * SPECTRAL_BINS) / Math.max(1e-3, span));
  let fftSize = 256;
  while (fftSize < targetBins && fftSize < 1024) fftSize <<= 1;

  const { fsHz } = continuousWavePacketTiming(vNyq, frequencyMHz, fftSize);
  const activeSamples = Math.max(
    MIN_ACTIVE_PULSES,
    Math.min(fftSize, Math.round(fsHz * MAX_SLOW_TIME_WINDOW_S)),
  );
  const activeDurationS = activeSamples / fsHz;
  const { win, power: activeWinPower } = hannWindowFor(activeSamples);
  const { re: cwRe, im: cwIm } = cwBuffersFor(fftSize);
  const activeRe = activeSamples === fftSize ? cwRe : cwRe.subarray(0, activeSamples);
  const activeIm = activeSamples === fftSize ? cwIm : cwIm.subarray(0, activeSamples);

  // Along a 16 cm CW beam, adjacent spatial stations share near-identical velocities; merge large
  // spatial lists into fine velocity strata (exact incoherent power sum) to keep hot-path < 2 ms.
  let source: readonly VelocitySample[] = samples;
  let sourceCount = samples.length;
  if (samples.length > CW_STRATA) {
    cwStrataW.fill(0);
    cwStrataV.fill(0);
    cwStrataDisp.fill(0);
    cwStrataVPerp.fill(0);
    cwStrataDepth.fill(0);
    cwStrataFocus.fill(0);
    const invTwoNyq = 1 / (2 * vNyq);
    for (let j = 0; j < samples.length; j++) {
      const p = samples[j]!;
      const w = p.weight;
      if (!(w > 0)) continue;
      const idx = Math.max(
        0,
        Math.min(CW_STRATA - 1, Math.floor(((p.v + vNyq) * invTwoNyq) * CW_STRATA)),
      );
      cwStrataW[idx] = cwStrataW[idx]! + w;
      cwStrataV[idx] = cwStrataV[idx]! + w * p.v;
      cwStrataDisp[idx] = cwStrataDisp[idx]! + w * p.dispersion;
      cwStrataVPerp[idx] = cwStrataVPerp[idx]! + w * (p.vPerp ?? 0);
      cwStrataDepth[idx] = cwStrataDepth[idx]! + w * (p.depthCm ?? 8);
      cwStrataFocus[idx] = cwStrataFocus[idx]! + w * (p.focusCm ?? focusCm);
    }
    sourceCount = 0;
    for (let i = 0; i < CW_STRATA; i++) {
      const w = cwStrataW[i]!;
      if (!(w > 0)) continue;
      const invW = 1 / w;
      const dst = cwMergedPool[sourceCount++]!;
      dst.weight = w;
      dst.v = cwStrataV[i]! * invW;
      dst.dispersion = cwStrataDisp[i]! * invW;
      dst.vPerp = cwStrataVPerp[i]! * invW;
      dst.depthCm = cwStrataDepth[i]! * invW;
      dst.focusCm = cwStrataFocus[i]! * invW;
      dst.gateLengthCm = null;
    }
    source = cwMergedPool;
  }

  const tCell = Math.floor(timeS / Math.max(1e-4, activeDurationS * 0.5));
  const packetSeed = (seed ^ Math.imul(tCell, 0x9e3779b1) ^ 0x517cc1b7) | 0;
  const f0Hz = frequencyMHz * 1e6;
  const normDenom = activeSamples * activeWinPower * REALIZATIONS;

  for (let ensemble = 0; ensemble < REALIZATIONS; ensemble++) {
    let toneCount = 0;
    for (let j = 0; j < sourceCount; j++) {
      const p = source[j]!;
      if (!(p.weight > 0)) continue;
      const depth = p.depthCm ?? 8;
      const aperture =
        p.vPerp && depth > 0 ? (Math.abs(p.vPerp) * (APERTURE_MM / 10)) / depth : 0;
      const tau = scattererTransitTimeS(p.v, p.vPerp ?? 0, depth, p.focusCm ?? focusCm, null);
      const absV = Math.abs(p.v);
      const signV = Math.sign(p.v);
      // Laminar blood strata (dispersion < 0.04) stay narrow within one FFT bin; synthesizing a
      // single scatterer per stratum eliminates unphysical multi-tone self-interference across
      // identical frequencies while turbulent strata expand across the potential-core + shear law.
      const nSub = p.dispersion < 0.04 ? 1 : CW_SHEAR_QUANTILES.length;
      for (let k = 0; k < nSub; k++) {
        const q = nSub === 1 ? 0 : CW_SHEAR_QUANTILES[k]!;
        const turbulent = Math.min(absV, 0.9 * p.dispersion * absV * q);
        const geometric = aperture * (hash3(j, k, ensemble + 103, packetSeed) - 0.5);
        const v = p.v - signV * turbulent + geometric;
        const kPower =
          nSub === 1 ? p.weight : p.weight * (k === 0 ? CW_CORE_POWER_SHARE : CW_SHEAR_POWER_SHARE);
        let slot = tonePool[toneCount];
        if (!slot) {
          slot = {
            velocityMps: 0,
            power: 0,
            phaseRad: 0,
            transitTimeS: undefined,
            transitCenterFrac: 0.5,
            accelMps2: 0,
          };
          tonePool[toneCount] = slot;
        }
        slot.velocityMps = v;
        slot.power = kPower;
        slot.phaseRad = 2 * Math.PI * hash3(j, k, ensemble + 211, packetSeed);
        slot.transitTimeS = tau;
        slot.transitCenterFrac = 0.25 + 0.5 * hash3(j, k, ensemble + 307, packetSeed);
        slot.accelMps2 = 0;
        toneCount++;
      }
    }
    synthesizeIqRange(
      tonePool,
      toneCount,
      frequencyMHz,
      fsHz,
      timeS - activeDurationS / 2,
      activeRe,
      activeIm,
    );
    for (let i = 0; i < activeSamples; i++) {
      cwRe[i] = cwRe[i]! * win[i]!;
      cwIm[i] = cwIm[i]! * win[i]!;
    }
    if (activeSamples < fftSize) {
      cwRe.fill(0, activeSamples);
      cwIm.fill(0, activeSamples);
    }
    complexFft(cwRe, cwIm);
    for (let k = 0; k < fftSize; k++) {
      const signed = k < fftSize / 2 ? k : k - fftSize;
      const fdHz = (signed / fftSize) * fsHz;
      const sampledVelocity = dopplerVelocityFromShiftHz(fdHz, f0Hz, 1, SPEED_OF_SOUND_MPS);
      const wfGain = clutterFilterPowerResponse(sampledVelocity, s.wallFilterMps);
      if (wfGain <= 0) continue;
      const v = s.invert ? -sampledVelocity : sampledVelocity;
      const b = ((vMax - v) / span) * SPECTRAL_BINS - 0.5;
      if (b <= -1 || b >= SPECTRAL_BINS) continue;
      const floor = Math.floor(b);
      const frac = b - floor;
      const power = ((cwRe[k]! ** 2 + cwIm[k]! ** 2) * wfGain) / normDenom;
      if (floor >= 0 && floor < SPECTRAL_BINS) {
        out[floor] = out[floor]! + power * (1 - frac);
      }
      const b1 = floor + 1;
      if (b1 >= 0 && b1 < SPECTRAL_BINS) {
        out[b1] = out[b1]! + power * frac;
      }
    }
  }
  // Enforce exact stop-band floor on display bins after linear FFT-to-display resampling so
  // passband edge splatting cannot bleed residual 1e-11 power into |v| < 0.8 * wallFilterMps.
  if (s.wallFilterMps > 0) {
    const stopCut = 0.8 * s.wallFilterMps;
    for (let b = 0; b < SPECTRAL_BINS; b++) {
      const vBin = vMax - ((b + 0.5) / SPECTRAL_BINS) * span;
      if (Math.abs(vBin) < stopCut) out[b] = 0;
    }
  }
}
