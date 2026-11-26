import assert from 'node:assert/strict';
import test from 'node:test';
import { SPEED_OF_SOUND_MPS } from '@/core/units';
import { accumulatePulsedSpectrum, synthesizeIq, type IqTone } from './pulsedIq';
import { DEFAULT_SPECTRAL, SPECTRAL_BINS, type VelocitySample } from './spectrum';

/** Direct reference implementation using per-sample Math.exp to verify outward Gaussian recurrence parity. */
function referenceSynthesizeIq(
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
    const baseAmp = Math.sqrt(tone.power);
    const accel = tone.accelMps2 ?? 0;
    const useChirp = Math.abs(accel) > 0.25;
    const tau = tone.transitTimeS;
    const useTransit =
      tau !== undefined && Number.isFinite(tau) && tau > 0 && tau < 3 * packetDurationS;

    let envNorm = 1;
    let center = 0;
    let inv2Sig2 = 0;
    if (useTransit) {
      center = (tone.transitCenterFrac ?? 0.5) * (n - 1);
      const sigmaIdx = Math.max(2.5, (tau * 0.55) / dt);
      inv2Sig2 = 1 / (2 * sigmaIdx * sigmaIdx);
      let energy = 0;
      for (let i = 0; i < n; i++) {
        const d = i - center;
        const e = Math.exp(-d * d * inv2Sig2);
        energy += e * e;
      }
      envNorm = Math.sqrt(n / Math.max(1e-6, energy));
    }

    const chirpCoeff = (2 * Math.PI * frequencyMHz * 1e6 * accel) / SPEED_OF_SOUND_MPS;
    const centerTimeS = timeS + halfIdx * dt;
    for (let i = 0; i < n; i++) {
      const w = useTransit ? Math.exp(-(i - center) * (i - center) * inv2Sig2) * envNorm : 1;
      let phi: number;
      if (useChirp) {
        const tRel = (i - halfIdx) * dt;
        phi =
          tone.phaseRad +
          2 * Math.PI * fd * centerTimeS +
          2 * Math.PI * fd * tRel +
          chirpCoeff * tRel * tRel;
      } else {
        phi = tone.phaseRad + 2 * Math.PI * fd * (timeS + i * dt);
      }
      re[i] = re[i]! + baseAmp * Math.cos(phi) * w;
      im[i] = im[i]! + baseAmp * Math.sin(phi) * w;
    }
  }
}

test('synthesizeIq outward Gaussian recurrence matches exact Math.exp reference within 1e-11', () => {
  const n = 128;
  const f0 = 2.5;
  const prfHz = 5200;
  const timeS = 0.347;
  const tones: IqTone[] = [
    // Narrow transit near packet start (would underflow if marched from right)
    { velocityMps: 0.95, power: 1.4, phaseRad: 0.7, transitTimeS: 0.0005, transitCenterFrac: 0.02 },
    // Narrow transit near packet end (would underflow if marched from left)
    { velocityMps: -1.35, power: 0.8, phaseRad: 2.1, transitTimeS: 0.0006, transitCenterFrac: 0.98 },
    // Combined chirp + transit at middle
    {
      velocityMps: 1.15,
      power: 2.2,
      phaseRad: 4.3,
      transitTimeS: 0.004,
      transitCenterFrac: 0.43,
      accelMps2: 18.5,
    },
    // Negative chirp + transit
    {
      velocityMps: -0.65,
      power: 1.1,
      phaseRad: 1.5,
      transitTimeS: 0.008,
      transitCenterFrac: 0.72,
      accelMps2: -14.2,
    },
    // Pure tone without transit or chirp
    { velocityMps: 0.42, power: 0.5, phaseRad: 3.1 },
  ];

  const gotRe = new Float64Array(n);
  const gotIm = new Float64Array(n);
  const refRe = new Float64Array(n);
  const refIm = new Float64Array(n);

  synthesizeIq(tones, f0, prfHz, timeS, gotRe, gotIm);
  referenceSynthesizeIq(tones, f0, prfHz, timeS, refRe, refIm);

  let maxErr = 0;
  for (let i = 0; i < n; i++) {
    maxErr = Math.max(maxErr, Math.abs(gotRe[i]! - refRe[i]!), Math.abs(gotIm[i]! - refIm[i]!));
  }
  assert.ok(maxErr < 1e-11, `Expected max I/Q error < 1e-11, got ${maxErr}`);
});

test('accumulatePulsedSpectrum tone pool has zero leakage when sample count shrinks between calls', () => {
  const manySamples: VelocitySample[] = Array.from({ length: 24 }, (_, i) => ({
    v: 0.4 + 0.03 * i,
    weight: 1.0,
    dispersion: 0.1,
    vPerp: 0.2,
    depthCm: 8.0,
    focusCm: 8.0,
    gateLengthCm: 0.3,
    frequencyMHz: 2.5,
    accelMps2: 6.0,
  }));
  const oneSample: VelocitySample[] = [
    {
      v: -0.75,
      weight: 1.5,
      dispersion: 0.05,
      vPerp: 0.05,
      depthCm: 8.0,
      focusCm: 8.0,
      gateLengthCm: 0.3,
      frequencyMHz: 2.5,
      accelMps2: 0,
    },
  ];
  const spectral = { ...DEFAULT_SPECTRAL, scaleMps: 1.2 };
  const outClean = new Float32Array(SPECTRAL_BINS);
  const outWarm = new Float32Array(SPECTRAL_BINS);
  const outAfterMany = new Float32Array(SPECTRAL_BINS);

  // First call with 1 sample
  accumulatePulsedSpectrum(oneSample, spectral, 2.5, 1337, 0.25, outClean, 8.0, 8.0);
  // Grow pool with 24 samples
  accumulatePulsedSpectrum(manySamples, spectral, 2.5, 1337, 0.25, outWarm, 8.0, 8.0);
  // Repeat 1-sample call: must be bit-identical to outClean
  accumulatePulsedSpectrum(oneSample, spectral, 2.5, 1337, 0.25, outAfterMany, 8.0, 8.0);

  for (let b = 0; b < SPECTRAL_BINS; b++) {
    assert.equal(outAfterMany[b], outClean[b], `Bin ${b} leaked stale pool tone`);
  }
});
