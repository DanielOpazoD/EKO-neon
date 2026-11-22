import assert from 'node:assert/strict';
import test from 'node:test';
import {
  AUDIO_BANDS,
  AUDIO_MAX_HZ,
  computeDopplerAudioFrame,
  mapAudibleDopplerHz,
} from '../audio/dopplerAudioModel';
import { accumulatePulsedSpectrum, pulsedPacketTiming } from './pulsedIq';
import {
  aliasedDopplerShiftHz,
  clutterFilterPowerResponse,
  depthLimitedPulsedAcquisition,
  rayleighBackscatterGain,
  sampleVolumeWeight,
  transitBroadeningMps,
} from './scatteringModel';
import {
  accumulateSpectrum,
  buildSpectralColumn,
  SPECTRAL_BINS,
  spectralDefaultsFor,
  spectralRange,
  spectralSpread,
} from './spectrum';

test('Rayleigh backscatter scales strongly with frequency for blood and more gently for myocardium', () => {
  const bloodLow = rayleighBackscatterGain(1.75, false);
  const bloodRef = rayleighBackscatterGain(2.5, false);
  const bloodHigh = rayleighBackscatterGain(3.5, false);
  assert.ok(Math.abs(bloodRef - 1) < 1e-9);
  assert.ok(bloodHigh > bloodRef * 3, `expected ~f0^4 scaling for blood, got ${bloodHigh}`);
  assert.ok(bloodLow < bloodRef * 0.35, `expected low f0 blood drop, got ${bloodLow}`);

  const tissueHigh = rayleighBackscatterGain(3.5, true);
  assert.ok(
    tissueHigh > 1 && tissueHigh < bloodHigh,
    `myocardium frequency slope (${tissueHigh}) should be gentler than Rayleigh blood (${bloodHigh})`,
  );
});

test('sampleVolumeWeight peaks on-axis at transmit focus and tapers laterally and axially', () => {
  const onFocus = sampleVolumeWeight(9, 9, 0, 0, 0, 0.4);
  const offAxis = sampleVolumeWeight(9, 9, 0, 0.25, 0, 0.4);
  const axialEdge = sampleVolumeWeight(9, 9, 0.25, 0, 0, 0.4);
  const farDefocused = sampleVolumeWeight(15, 9, 0, 0, 0, 0.4);

  assert.ok(onFocus > offAxis, 'off-axis sample should have lower two-way beam sensitivity');
  assert.ok(onFocus > axialEdge, 'gate-edge sample should taper axially');
  assert.ok(onFocus > farDefocused, 'defocused far-field sample should have lower focal gain');
});

test('transit-time broadening increases with shorter PW gate in both analytical and I/Q estimators', () => {
  const shortSig = transitBroadeningMps(1.0, 0.1, 8, 9, 2.5, 0.15);
  const longSig = transitBroadeningMps(1.0, 0.1, 8, 9, 2.5, 0.8);
  assert.ok(
    shortSig > longSig * 3,
    `short gate (${shortSig.toFixed(4)}) should broaden much more than long gate (${longSig.toFixed(4)})`,
  );

  const sBase = { ...spectralDefaultsFor('pw'), scaleMps: 1.5 };
  const spreadShort = spectralSpread(
    { v: 0.9, weight: 1, dispersion: 0.02, depthCm: 8, focusCm: 9, gateLengthCm: 0.12, frequencyMHz: 2.5 },
    sBase,
  );
  const spreadLong = spectralSpread(
    { v: 0.9, weight: 1, dispersion: 0.02, depthCm: 8, focusCm: 9, gateLengthCm: 0.8, frequencyMHz: 2.5 },
    sBase,
  );
  assert.ok(spreadShort.up > spreadLong.up, 'short PW gate should widen spectralSpread.up');

  // Verify slow-time I/Q FFT also exhibits wider spectral bandwidth for a short axial gate
  const outShort = new Float32Array(SPECTRAL_BINS);
  const outLong = new Float32Array(SPECTRAL_BINS);
  const sampleBase = { v: 1.0, weight: 1, dispersion: 0.02, vPerp: 0, depthCm: 6 };
  accumulatePulsedSpectrum(
    [{ ...sampleBase, gateLengthCm: 0.1 }],
    { ...sBase, gateLengthCm: 0.1 },
    2.5,
    42,
    0.2,
    outShort,
    6,
    9,
  );
  accumulatePulsedSpectrum(
    [{ ...sampleBase, gateLengthCm: 1.0 }],
    { ...sBase, gateLengthCm: 1.0 },
    2.5,
    42,
    0.2,
    outLong,
    6,
    9,
  );
  const spreadBins = (arr: Float32Array) => {
    let sum = 0,
      mean = 0;
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
    spreadBins(outShort) > spreadBins(outLong) * 1.15,
    `short-gate I/Q width (${spreadBins(outShort).toFixed(2)} bins) should exceed long-gate (${spreadBins(outLong).toFixed(2)} bins)`,
  );
});

test('depth-dependent PW PRF ceiling forces aliasing at deep gates while shallow gates stay unaliased', () => {
  const shallow = depthLimitedPulsedAcquisition(5, 0.4, 2.5, 1.5);
  const deep = depthLimitedPulsedAcquisition(14, 0.4, 2.5, 1.5);
  assert.ok(shallow.maxPrfHz > deep.maxPrfHz * 1.8, 'shallow gate must support much higher PRF');
  assert.equal(shallow.depthLimited, false);
  assert.equal(deep.depthLimited, true);
  assert.ok(deep.effectiveScaleMps < 0.9, `expected deep Nyquist < 0.9 m/s, got ${deep.effectiveScaleMps}`);

  // Same +1.25 m/s blood flow sampled with identical requested ±1.5 m/s scale:
  // - Shallow gate (5 cm): unaliased peak in positive upper half (b < SPECTRAL_BINS / 2)
  // - Deep gate (14 cm): depth-capped Nyquist (~0.76 m/s) wraps +1.25 m/s into negative lower half (b > SPECTRAL_BINS / 2)
  const settings = { ...spectralDefaultsFor('pw'), scaleMps: 1.5, baselineShiftMps: 0 };
  const shallowOut = new Float32Array(SPECTRAL_BINS);
  const deepOut = new Float32Array(SPECTRAL_BINS);
  const sample = [{ v: 1.25, weight: 1, dispersion: 0.04, vPerp: 0.05 }];

  accumulatePulsedSpectrum(sample, settings, 2.5, 17, 0.3, shallowOut, 5, 9);
  accumulatePulsedSpectrum(sample, settings, 2.5, 17, 0.3, deepOut, 14, 9);

  const peakBin = (arr: Float32Array) => {
    let best = 0;
    for (let b = 1; b < arr.length; b++) if (arr[b]! > arr[best]!) best = b;
    return best;
  };
  const bShallow = peakBin(shallowOut);
  const bDeep = peakBin(deepOut);
  assert.ok(
    bShallow < SPECTRAL_BINS / 2,
    `shallow 5 cm gate should keep +1.25 m/s in positive half, got bin ${bShallow}`,
  );
  assert.ok(
    bDeep > SPECTRAL_BINS / 2,
    `deep 14 cm gate should alias +1.25 m/s into negative half, got bin ${bDeep}`,
  );

  // Confirm packet timing matches depth-dependent PRF
  const deepTiming = pulsedPacketTiming(1.5, 2.5, SPECTRAL_BINS, 14, 0.4);
  assert.ok(Math.abs(deepTiming.prfHz - deep.prfHz) < 1e-6);
  const wrappedHz = aliasedDopplerShiftHz(1.25, 2.5e6, deep.prfHz, true);
  assert.ok(wrappedHz < 0, `expected aliased Doppler shift < 0 Hz at deep gate, got ${wrappedHz}`);
});

test('clutterFilterPowerResponse enforces stopband below 0.8*wf and smooth -3 dB transition at wf', () => {
  assert.equal(clutterFilterPowerResponse(0.05, 0.1), 0);
  assert.equal(clutterFilterPowerResponse(0.079, 0.1), 0);
  const atCutoff = clutterFilterPowerResponse(0.1, 0.1);
  assert.ok(Math.abs(atCutoff - 0.5) < 1e-6, `expected 0.5 (-3 dB) at cutoff, got ${atCutoff}`);
  const passband = clutterFilterPowerResponse(0.25, 0.1);
  assert.ok(passband > 0.95, `expected near-unity passband, got ${passband}`);
});

test('Doppler audio filterbank separates stereo direction, tracks aliased PW Nyquist, and detects turbulence & clicks', () => {
  const sPw = { ...spectralDefaultsFor('pw'), scaleMps: 1.0, baselineShiftMps: 0 };
  const { vMin, vMax } = spectralRange(sPw);

  // 1. Narrow laminar positive flow (+0.72 m/s) -> forward (left) channel active, reverse (right) silent
  const colPos = new Float32Array(SPECTRAL_BINS);
  buildSpectralColumn(
    [{ v: 0.72, weight: 1, dispersion: 0.04 }],
    sPw,
    12,
    7,
    true,
    colPos,
    2.5,
    1.0,
    0,
  );
  const prfPw = depthLimitedPulsedAcquisition(8, 0.4, 2.5, 1.0).prfHz;
  const framePos = computeDopplerAudioFrame(colPos, vMin, vMax, 2.5e6, {
    aliasing: true,
    prfHz: prfPw,
    wallFilterMps: sPw.wallFilterMps,
  });
  const sumGain = (bands: { gain: number }[]) => bands.reduce((acc, b) => acc + b.gain, 0);
  assert.equal(framePos.toward.length, AUDIO_BANDS);
  assert.ok(sumGain(framePos.toward) > 0.05, 'positive flow should drive forward audio channel');
  assert.ok(
    sumGain(framePos.away) < sumGain(framePos.toward) * 0.08,
    'positive laminar flow should not bleed into reverse audio channel',
  );
  // Every PW audio filter centre frequency stays bounded by PRF/2
  for (const b of framePos.toward) {
    assert.ok(b.freqHz <= prfPw / 2 + 1, `PW audio freq ${b.freqHz} exceeded Nyquist ${prfPw / 2}`);
  }

  // 2. Aliased PW flow (+1.18 m/s on ±1.0 m/s scale -> wraps to -0.82 m/s near bottom Nyquist edge)
  const colAlias = new Float32Array(SPECTRAL_BINS);
  accumulateSpectrum([{ v: 1.18, weight: 1, dispersion: 0.05 }], sPw, true, colAlias);
  // Normalize column to 0..1 like strip output
  let maxA = 0;
  for (let i = 0; i < SPECTRAL_BINS; i++) maxA = Math.max(maxA, colAlias[i]!);
  for (let i = 0; i < SPECTRAL_BINS; i++) colAlias[i] = colAlias[i]! / Math.max(1e-6, maxA);
  const frameAlias = computeDopplerAudioFrame(colAlias, vMin, vMax, 2.5e6, {
    aliasing: true,
    prfHz: prfPw,
    wallFilterMps: sPw.wallFilterMps,
  });
  assert.ok(
    sumGain(frameAlias.away) > sumGain(frameAlias.toward),
    'aliased positive flow wrapping to negative bins should flip to reverse stereo channel',
  );
  assert.ok(
    frameAlias.aliasingIntensity > 0.25,
    `expected elevated aliasingIntensity near Nyquist fold, got ${frameAlias.aliasingIntensity}`,
  );

  // 3. Turbulent high-velocity CW stenotic jet (-4.5 m/s) -> broad reverse hiss with lower Q than laminar
  const sCw = spectralDefaultsFor('cw');
  const cwRange = spectralRange(sCw);
  const colCw = new Float32Array(SPECTRAL_BINS);
  buildSpectralColumn(
    [{ v: -4.5, weight: 1, dispersion: 0.55 }],
    sCw,
    14,
    7,
    false,
    colCw,
    2.5,
    1.0,
    0,
  );
  const frameCw = computeDopplerAudioFrame(colCw, cwRange.vMin, cwRange.vMax, 2.5e6, {
    aliasing: false,
    wallFilterMps: sCw.wallFilterMps,
  });
  assert.ok(sumGain(frameCw.away) > 0.05, 'negative CW jet should drive reverse channel');
  assert.ok(
    frameCw.turbulence > framePos.turbulence,
    `CW turbulent jet (${frameCw.turbulence.toFixed(2)}) should have higher turbulence than laminar PW (${framePos.turbulence.toFixed(2)})`,
  );
  assert.ok(
    frameCw.away[12]!.q < framePos.toward[12]!.q,
    'turbulent CW jet should widen bandpass filter (lower Q) compared to laminar flow',
  );
  assert.ok(
    mapAudibleDopplerHz(16000) <= AUDIO_MAX_HZ,
    'extreme CW Doppler shifts must compress safely below AUDIO_MAX_HZ',
  );

  // 4. Valve click column triggers broadband transient cue
  const colClick = new Float32Array(SPECTRAL_BINS);
  buildSpectralColumn([], sPw, 15, 7, true, colClick, 2.5, 1.0, 1.0);
  const frameClick = computeDopplerAudioFrame(colClick, vMin, vMax, 2.5e6, {
    aliasing: true,
    prfHz: prfPw,
    wallFilterMps: sPw.wallFilterMps,
  });
  assert.ok(
    frameClick.clickLevel > 0.4,
    `valve click column should produce clickLevel > 0.4, got ${frameClick.clickLevel}`,
  );
});

test('intra-packet acceleration chirp broadens accelerating PW blood echoes while preserving center velocity', () => {
  const sPw = { ...spectralDefaultsFor('pw'), scaleMps: 1.0, baselineShiftMps: 0 };
  const steadyOut = new Float32Array(SPECTRAL_BINS);
  const chirpOut = new Float32Array(SPECTRAL_BINS);
  const baseSample = { v: 0.45, weight: 1, dispersion: 0.02, vPerp: 0, depthCm: 8 };

  accumulatePulsedSpectrum([baseSample], sPw, 2.5, 31, 0.25, steadyOut, 8, 9);
  accumulatePulsedSpectrum(
    [{ ...baseSample, accelMps2: 18 }],
    sPw,
    2.5,
    31,
    0.25,
    chirpOut,
    8,
    9,
  );

  const moments = (arr: Float32Array) => {
    let sum = 0;
    let mean = 0;
    for (let b = 0; b < arr.length; b++) {
      sum += arr[b]!;
      mean += b * arr[b]!;
    }
    mean /= Math.max(1e-12, sum);
    let variance = 0;
    for (let b = 0; b < arr.length; b++) variance += arr[b]! * (b - mean) ** 2;
    return { mean, std: Math.sqrt(variance / Math.max(1e-12, sum)) };
  };

  const steady = moments(steadyOut);
  const chirp = moments(chirpOut);
  assert.ok(
    Math.abs(chirp.mean - steady.mean) < 1.2,
    `chirp should preserve center bin (~${steady.mean.toFixed(1)}), got ${chirp.mean.toFixed(1)}`,
  );
  assert.ok(
    chirp.std > steady.std * 1.4,
    `accelerating flow (${chirp.std.toFixed(2)} bins) should broaden noticeably over steady flow (${steady.std.toFixed(2)} bins)`,
  );
});

test('displaySpectrum renders directional valve clicks with authentic polarity asymmetry and smooth clutter taper', () => {
  const sPw = spectralDefaultsFor('pw');
  const mvoDisp = new Float32Array(SPECTRAL_BINS);
  const mvcDisp = new Float32Array(SPECTRAL_BINS);
  const dummyOut = new Float32Array(SPECTRAL_BINS);
  buildSpectralColumn([], sPw, 10, 77, true, dummyOut, 2.5, 0.72, 1.0, mvoDisp, 0.45, +0.95);
  buildSpectralColumn([], sPw, 10, 77, true, dummyOut, 2.5, 0.72, 1.0, mvcDisp, 0.45, -0.95);

  const sumHalf = (arr: Float32Array, topHalf: boolean) => {
    const start = topHalf ? 0 : SPECTRAL_BINS / 2;
    const end = topHalf ? SPECTRAL_BINS / 2 : SPECTRAL_BINS;
    let s = 0;
    for (let b = start; b < end; b++) s += arr[b]!;
    return s;
  };

  assert.ok(
    sumHalf(mvoDisp, true) > sumHalf(mvoDisp, false) * 1.5,
    'positive-polarity MVO click should dominate above baseline',
  );
  assert.ok(
    sumHalf(mvcDisp, false) > sumHalf(mvcDisp, true) * 1.5,
    'negative-polarity MVC click should dominate below baseline',
  );
});
