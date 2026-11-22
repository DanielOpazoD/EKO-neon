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
  re.fill(0);
  im.fill(0);
  const n = re.length;
  const dt = 1 / prfHz;
  const packetDurationS = n * dt;
  const halfIdx = 0.5 * (n - 1);
  for (const tone of tones) {
    if (!(tone.power > 0)) continue;
    const fd = (2 * frequencyMHz * 1e6 * tone.velocityMps) / SPEED_OF_SOUND_MPS;
    const angle = (2 * Math.PI * fd) / prfHz;
    const baseAmp = Math.sqrt(tone.power);
    const accel = tone.accelMps2 ?? 0;
    const useChirp = Math.abs(accel) > 0.25;
    const tau = tone.transitTimeS;
    const useTransit = tau !== undefined && Number.isFinite(tau) && tau > 0 && tau < 3 * packetDurationS;

    if (useChirp) {
      // Exact quadratic phase centred at packet midpoint: v(t_rel) = v + a * t_rel
      const chirpCoeff = (2 * Math.PI * frequencyMHz * 1e6 * accel) / SPEED_OF_SOUND_MPS;
      const chirpStep = chirpCoeff * dt * dt;
      const centerTimeS = timeS + halfIdx * dt;
      const tRel0 = -halfIdx * dt;
      const phi0 =
        tone.phaseRad +
        2 * Math.PI * fd * centerTimeS +
        2 * Math.PI * fd * tRel0 +
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
        let envEnergy = 0;
        for (let i = 0; i < n; i++) {
          const d = i - center;
          const e = Math.exp(-d * d * inv2Sig2);
          envEnergy += e * e;
        }
        const envNorm = Math.sqrt(n / Math.max(1e-6, envEnergy));
        for (let i = 0; i < n; i++) {
          const d = i - center;
          const w = Math.exp(-d * d * inv2Sig2) * envNorm;
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
    const phase = tone.phaseRad + 2 * Math.PI * fd * timeS;
    let ar = baseAmp * Math.cos(phase);
    let ai = baseAmp * Math.sin(phase);
    if (useTransit) {
      // Normalise Gaussian transit envelope to preserve mean scatterer power while broadening spectrum.
      const center = (tone.transitCenterFrac ?? 0.5) * (n - 1);
      const sigmaIdx = Math.max(2.5, (tau * 0.55) / dt);
      const inv2Sig2 = 1 / (2 * sigmaIdx * sigmaIdx);
      let envEnergy = 0;
      for (let i = 0; i < n; i++) {
        const d = i - center;
        const e = Math.exp(-d * d * inv2Sig2);
        envEnergy += e * e;
      }
      const envNorm = Math.sqrt(n / Math.max(1e-6, envEnergy));
      for (let i = 0; i < n; i++) {
        const d = i - center;
        const w = Math.exp(-d * d * inv2Sig2) * envNorm;
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

const PACKET = SPECTRAL_BINS;
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
const window = Float64Array.from(
  { length: PACKET },
  (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / PACKET),
);
const windowPower = window.reduce((a, b) => a + b * b, 0);

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
  const { prfHz, effectiveScaleMps, durationS } = pulsedPacketTiming(
    s.scaleMps,
    frequencyMHz,
    PACKET,
    gateDepthCm,
    s.gateLengthCm,
  );
  // Display mapping uses the physical Nyquist span when depth-limited so wrapped bins align with the scale.
  const effectiveSettings =
    effectiveScaleMps < s.scaleMps ? { ...s, scaleMps: effectiveScaleMps } : s;
  const { vMin, vMax } = spectralRange(effectiveSettings),
    span = vMax - vMin;
  out.fill(0);
  // Decorrelate erythrocyte phase constellations across consecutive overlapping packets deterministically.
  const tCell = Math.floor(timeS / Math.max(1e-4, durationS * 0.5));
  const packetSeed = (seed ^ Math.imul(tCell, 0x9e3779b1)) | 0;
  for (let ensemble = 0; ensemble < REALIZATIONS; ensemble++) {
    const tones: IqTone[] = [];
    for (let j = 0; j < samples.length; j++) {
      const p = samples[j]!;
      if (!(p.weight > 0)) continue;
      const depth = p.depthCm ?? gateDepthCm ?? 8;
      const aperture =
        p.vPerp && depth > 0 ? (Math.abs(p.vPerp) * (APERTURE_MM / 10)) / depth : 0;
      const gateLen = p.gateLengthCm ?? s.gateLengthCm;
      const tau = scattererTransitTimeS(p.v, p.vPerp ?? 0, depth, p.focusCm ?? focusCm, gateLen);
      const baseAccel = p.accelMps2 ?? 0;
      for (let k = 0; k < MICROSCATTERERS; k++) {
        // Half-normal deceleration, capped at zero; aperture rays span D/F uniformly.
        const turbulent = Math.min(
          Math.abs(p.v),
          0.9 * p.dispersion * Math.abs(p.v) * HALF_NORMAL_QUANTILES[k]!,
        );
        const geometric = aperture * (hash3(j, k, ensemble + 103, packetSeed) - 0.5);
        const v = p.v - Math.sign(p.v) * turbulent + geometric;
        const accelScale = Math.abs(p.v) > 0.05 ? Math.max(0, Math.min(1.2, v / p.v)) : 1;
        tones.push({
          velocityMps: v,
          power: p.weight / MICROSCATTERERS,
          phaseRad: 2 * Math.PI * hash3(j, k, ensemble + 211, packetSeed),
          transitTimeS: tau,
          transitCenterFrac: 0.25 + 0.5 * hash3(j, k, ensemble + 307, packetSeed),
          accelMps2: baseAccel * accelScale,
        });
      }
    }
    // Strip timestamps denote the packet centre, not its first pulse.
    synthesizeIq(tones, frequencyMHz, prfHz, timeS - durationS / 2, re, im);
    for (let i = 0; i < PACKET; i++) {
      re[i] = re[i]! * window[i]!;
      im[i] = im[i]! * window[i]!;
    }
    complexFft(re, im);
    for (let k = 0; k < PACKET; k++) {
      const signed = k < PACKET / 2 ? k : k - PACKET;
      const sampledVelocity = (signed * span) / PACKET;
      const wfGain = clutterFilterPowerResponse(sampledVelocity, effectiveSettings.wallFilterMps);
      if (wfGain <= 0) continue;
      const v = effectiveSettings.invert ? -sampledVelocity : sampledVelocity;
      const b = ((vMax - v) / span) * PACKET - 0.5;
      const floor = Math.floor(b),
        frac = b - floor;
      const power = ((re[k]! ** 2 + im[k]! ** 2) * wfGain) / (PACKET * windowPower * REALIZATIONS);
      // Baseline is a circular display shift only; it cannot change the wall filter's physical stop band.
      const b0 = ((floor % PACKET) + PACKET) % PACKET,
        b1 = (b0 + 1) % PACKET;
      out[b0] = out[b0]! + power * (1 - frac);
      out[b1] = out[b1]! + power * frac;
    }
  }
}
