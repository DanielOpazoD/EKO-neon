import assert from 'node:assert/strict';
import test from 'node:test';
import { SPEED_OF_SOUND_MPS } from '@/core/units';
import {
  accumulateContinuousWaveSpectrum,
  accumulatePulsedSpectrum,
  continuousWavePacketTiming,
  synthesizeIq,
  type IqTone,
} from './pulsedIq';
import {
  DEFAULT_SPECTRAL,
  SPECTRAL_BINS,
  spectralDefaultsFor,
  spectralRange,
  type VelocitySample,
} from './spectrum';

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

test('accumulateContinuousWaveSpectrum runs unaliased baseband I/Q + FFT with per-bin wall filter and zero opposite-channel wrap', () => {
  const sCw = spectralDefaultsFor('cw'); // ±6 m/s, WF 0.2 m/s
  const timing = continuousWavePacketTiming(6.3, 2.5, 256);
  assert.ok(timing.fsHz > 40_000, `CW baseband fsHz should cover ±6.3 m/s unaliased, got ${timing.fsHz}`);

  // 1. Turbulent high-velocity jet (-4.5 m/s) preserves outer vena contracta rim, fills toward baseline,
  //    keeps stopband strictly zero, and never bleeds into the positive velocity channel
  for (const [wf, baselineShiftMps] of [
    [0.2, 0],
    [0.5, 0],
    [0.5, -2.0],
  ] as const) {
    const s = { ...sCw, wallFilterMps: wf, baselineShiftMps };
    const out = new Float32Array(SPECTRAL_BINS);
    accumulateContinuousWaveSpectrum(
      [{ v: -4.5, weight: 1, dispersion: 0.5, vPerp: 0.2, depthCm: 8 }],
      s,
      2.5,
      101,
      0.22,
      out,
      9,
    );
    const { vMin, vMax } = spectralRange(s);
    const span = vMax - vMin;
    let peak = 0;
    let peakV = 0;
    let posEnergy = 0;
    let negEnergy = 0;
    let stopbandMax = 0;
    for (let b = 0; b < SPECTRAL_BINS; b++) {
      const vBin = vMax - ((b + 0.5) / SPECTRAL_BINS) * span;
      const val = out[b]!;
      if (val > peak) {
        peak = val;
        peakV = vBin;
      }
      if (vBin > wf) posEnergy += val;
      if (vBin < -wf) negEnergy += val;
      if (Math.abs(vBin) < 0.8 * wf) stopbandMax = Math.max(stopbandMax, val);
    }
    assert.equal(stopbandMax, 0, `CW I/Q FFT leaked ${stopbandMax} into |v| < 0.8*WF stopband`);
    assert.ok(
      Math.abs(peakV - -4.5) < 0.25 && peak > 0.05,
      `CW I/Q FFT should preserve vena contracta peak near -4.5 m/s, got ${peakV.toFixed(2)} m/s`,
    );
    assert.ok(
      posEnergy / Math.max(1e-12, negEnergy) < 0.01,
      `CW I/Q FFT turbulent tail bled across baseline: ratio=${(posEnergy / negEnergy).toFixed(4)}`,
    );
  }

  // 2. Zoomed CW scale (±2.0 m/s) with fast +4.5 m/s jet must leave the display unaliased (NOT wrap onto negative side)
  const sZoom = { ...sCw, scaleMps: 2.0, baselineShiftMps: 0 };
  const zoomOut = new Float32Array(SPECTRAL_BINS);
  accumulateContinuousWaveSpectrum(
    [{ v: 4.5, weight: 1, dispersion: 0.02, vPerp: 0.05, depthCm: 8 }],
    sZoom,
    2.5,
    101,
    0.22,
    zoomOut,
    9,
  );
  const zoomTotal = zoomOut.reduce((acc, v) => acc + v, 0);
  assert.ok(zoomTotal < 1e-4, `Fast +4.5 m/s tone aliased onto zoomed ±2.0 m/s CW scale (sum=${zoomTotal})`);

  // 3. Invert flips displayed velocity polarity cleanly
  const invOut = new Float32Array(SPECTRAL_BINS);
  accumulateContinuousWaveSpectrum(
    [{ v: -3.2, weight: 1, dispersion: 0.03, vPerp: 0, depthCm: 8 }],
    { ...sCw, invert: true },
    2.5,
    101,
    0.22,
    invOut,
    9,
  );
  let invPeakBin = 0;
  for (let b = 1; b < SPECTRAL_BINS; b++) if (invOut[b]! > invOut[invPeakBin]!) invPeakBin = b;
  assert.ok(invPeakBin < SPECTRAL_BINS / 2, `Inverted -3.2 m/s CW flow should appear in positive upper half, got bin ${invPeakBin}`);

  // 4. Lateral cross-beam velocity widens CW I/Q spectrum via aperture + lateral transit broadening
  const axialOut = new Float32Array(SPECTRAL_BINS);
  const obliqueOut = new Float32Array(SPECTRAL_BINS);
  accumulateContinuousWaveSpectrum(
    [{ v: 2.5, weight: 1, dispersion: 0.01, vPerp: 0, depthCm: 6 }],
    sCw,
    2.5,
    101,
    0.22,
    axialOut,
    9,
  );
  accumulateContinuousWaveSpectrum(
    [{ v: 2.5, weight: 1, dispersion: 0.01, vPerp: 3.5, depthCm: 6 }],
    sCw,
    2.5,
    101,
    0.22,
    obliqueOut,
    9,
  );
  const widthBins = (arr: Float32Array) => {
    let sum = 0;
    let mean = 0;
    for (let b = 0; b < arr.length; b++) {
      sum += arr[b]!;
      mean += b * arr[b]!;
    }
    mean /= Math.max(1e-12, sum);
    let v = 0;
    for (let b = 0; b < arr.length; b++) v += arr[b]! * (b - mean) ** 2;
    return Math.sqrt(v / Math.max(1e-12, sum));
  };
  assert.ok(
    widthBins(obliqueOut) > widthBins(axialOut) * 1.4,
    `Oblique CW scatterer (${widthBins(obliqueOut).toFixed(2)} bins) should broaden over axial (${widthBins(axialOut).toFixed(2)} bins)`,
  );
});

test('PW laminar plug flow preserves a crisp dark hollow spectral window compared to turbulent jet fill', () => {
  const sPw = { ...spectralDefaultsFor('pw'), scaleMps: 1.2, baselineShiftMps: 0, wallFilterMps: 0.08 };
  const lamOut = new Float32Array(SPECTRAL_BINS);
  const turbOut = new Float32Array(SPECTRAL_BINS);
  accumulatePulsedSpectrum(
    [{ v: 0.95, weight: 1, dispersion: 0.04, vPerp: 0.02, depthCm: 9, focusCm: 9, gateLengthCm: 0.35 }],
    sPw,
    2.5,
    42,
    0.2,
    lamOut,
    9,
    9,
  );
  accumulatePulsedSpectrum(
    [{ v: 0.95, weight: 1, dispersion: 0.42, vPerp: 0.02, depthCm: 9, focusCm: 9, gateLengthCm: 0.35 }],
    sPw,
    2.5,
    42,
    0.2,
    turbOut,
    9,
    9,
  );
  const { vMin, vMax } = spectralRange(sPw);
  const span = vMax - vMin;
  const bandEnergy = (arr: Float32Array, loV: number, hiV: number) => {
    let sum = 0;
    for (let b = 0; b < SPECTRAL_BINS; b++) {
      const v = vMax - ((b + 0.5) / SPECTRAL_BINS) * span;
      if (v >= loV && v <= hiV) sum += arr[b]!;
    }
    return sum;
  };
  const lamCore = bandEnergy(lamOut, 0.8, 1.1);
  const lamWindow = bandEnergy(lamOut, 0.25, 0.65);
  const turbCore = bandEnergy(turbOut, 0.8, 1.1);
  const turbWindow = bandEnergy(turbOut, 0.25, 0.65);
  const lamFillRatio = lamWindow / Math.max(1e-9, lamCore);
  const turbFillRatio = turbWindow / Math.max(1e-9, turbCore);
  assert.ok(
    lamFillRatio < 0.05,
    `Laminar PW flow should maintain a dark hollow spectral window (window/core=${lamFillRatio.toFixed(4)})`,
  );
  assert.ok(
    turbFillRatio > lamFillRatio * 8 && turbFillRatio > 0.35,
    `Turbulent PW flow should clearly fill the spectral window (turb=${turbFillRatio.toFixed(3)} vs lam=${lamFillRatio.toFixed(4)})`,
  );
});
