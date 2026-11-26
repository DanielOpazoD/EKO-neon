import assert from 'node:assert/strict';
import test from 'node:test';
import {
  accumulateSpectrum,
  buildSpectralColumn,
  columnsPerSecond,
  displaySpectrum,
  isClickColumn,
  MIRROR_ISOLATION_DB,
  MIRROR_ISOLATION_POWER,
  presentSpectralColumn,
  SPECTRAL_BINS,
  spectralDefaultsFor,
  spectralRange,
  type VelocitySample,
} from './spectrum';
import { autoTraceSpectralEnvelope } from '@/simulator/core/strip/autoTrace';

test('spectralRange and spectralDefaultsFor reflect each modality window', () => {
  const pw = spectralDefaultsFor('pw');
  const cw = spectralDefaultsFor('cw');
  const tdi = spectralDefaultsFor('tdi');

  assert.deepEqual(spectralRange(pw), { vMin: -1.0, vMax: 1.0 });
  assert.deepEqual(spectralRange({ ...pw, baselineShiftMps: 0.4 }), { vMin: -0.6, vMax: 1.4 });
  assert.ok(cw.scaleMps >= 5.0, 'CW should open wide enough for high-velocity jets');
  assert.ok(tdi.scaleMps <= 0.3 && tdi.wallFilterMps <= 0.02, 'TDI should preserve low tissue velocities');
});

test('wall filter suppresses slow velocities below threshold', () => {
  const s = spectralDefaultsFor('pw'); // wallFilterMps = 0.08
  const out = new Float32Array(SPECTRAL_BINS);
  const slowSamples: VelocitySample[] = [
    { v: 0.05, weight: 1, dispersion: 0.1 },
    { v: -0.07, weight: 1, dispersion: 0.1 },
  ];
  accumulateSpectrum(slowSamples, s, true, out);
  const total = out.reduce((acc, v) => acc + v, 0);
  assert.equal(total, 0);
});

test('PW wraps velocities above Nyquist while CW does not alias and suppresses opposite-channel bleed', () => {
  const sPw = { ...spectralDefaultsFor('pw'), scaleMps: 1.0, baselineShiftMps: 0 };
  const pwOut = new Float32Array(SPECTRAL_BINS);
  // +1.5 m/s aliases across ±1.0 m/s to -0.5 m/s (lower half of bins: b > SPECTRAL_BINS/2)
  accumulateSpectrum([{ v: 1.5, weight: 1, dispersion: 0.05 }], sPw, true, pwOut);
  let pwPeakBin = 0;
  for (let b = 1; b < SPECTRAL_BINS; b++) if (pwOut[b]! > pwOut[pwPeakBin]!) pwPeakBin = b;
  assert.ok(
    pwPeakBin > SPECTRAL_BINS / 2,
    `expected aliased +1.5 m/s peak in negative half, got bin ${pwPeakBin}`,
  );

  // CW turbulent negative jet (-4.5 m/s on ±6.0 m/s scale) must keep energy out of positive half (> +0.2 m/s)
  const sCw = spectralDefaultsFor('cw');
  const cwOut = new Float32Array(SPECTRAL_BINS);
  accumulateSpectrum([{ v: -4.5, weight: 1, dispersion: 0.5 }], sCw, false, cwOut);
  let posEnergy = 0;
  let negEnergy = 0;
  const { vMin, vMax } = spectralRange(sCw);
  for (let b = 0; b < SPECTRAL_BINS; b++) {
    const v = vMax - ((b + 0.5) / SPECTRAL_BINS) * (vMax - vMin);
    if (v > 0.2) posEnergy += cwOut[b]!;
    if (v < -0.2) negEnergy += cwOut[b]!;
  }
  assert.ok(negEnergy > 1, 'CW jet should produce substantial negative-channel energy');
  assert.ok(
    posEnergy / negEnergy < 0.01,
    `CW turbulent tail bled across baseline: ratio=${(posEnergy / negEnergy).toFixed(4)}`,
  );

  // CW out-of-range velocity (> scaleMps) leaves display rather than wrapping to opposite side
  const cwOutOfRange = new Float32Array(SPECTRAL_BINS);
  accumulateSpectrum([{ v: 9.0, weight: 1, dispersion: 0 }], sCw, false, cwOutOfRange);
  const sumOutOfRange = cwOutOfRange.reduce((acc, v) => acc + v, 0);
  assert.equal(sumOutOfRange, 0);
});

test('isClickColumn distinguishes valve clicks (including CW & shifted baseline) from laminar and aliased PW flow', () => {
  const s = spectralDefaultsFor('pw');
  const flowCol = new Float32Array(SPECTRAL_BINS);
  buildSpectralColumn(
    [{ v: 0.7, weight: 1, dispersion: 0.05 }],
    s,
    10,
    42,
    true,
    flowCol,
    2.5,
    1.0,
    0,
  );
  assert.equal(isClickColumn(flowCol, s), false);

  const clickCol = new Float32Array(SPECTRAL_BINS);
  buildSpectralColumn([], s, 10, 42, true, clickCol, 2.5, 1.0, 1.0);
  assert.equal(isClickColumn(clickCol, s), true);

  // Moderate CW click tail (click=0.18) must be detected under cw=true even when <60% of full ±6 m/s span
  const sCw = spectralDefaultsFor('cw');
  const cwClickTail = new Float32Array(SPECTRAL_BINS);
  buildSpectralColumn([], sCw, 144, 42, false, cwClickTail, 2.5, 1.0, 0.18);
  assert.equal(isClickColumn(cwClickTail, sCw, true), true);

  // Shifted-baseline PW click must still be rejected by bilateral contiguous check
  const sShift = { ...s, baselineShiftMps: -0.35 };
  const shiftedClick = new Float32Array(SPECTRAL_BINS);
  buildSpectralColumn([], sShift, 12, 42, true, shiftedClick, 2.5, 1.0, 0.35);
  assert.equal(isClickColumn(shiftedClick, sShift, false), true);

  // Aliased PW flow (wrapped cap at Nyquist edge with dark window near baseline) must NOT be flagged as click
  const sAlias = { ...s, scaleMps: 0.6, baselineShiftMps: 0 };
  const aliasCol = new Float32Array(SPECTRAL_BINS);
  buildSpectralColumn(
    [{ v: -0.9, weight: 1, dispersion: 0.06 }],
    sAlias,
    15,
    42,
    true,
    aliasCol,
    2.5,
    1.0,
    0,
  );
  assert.equal(isClickColumn(aliasCol, sAlias, false), false);
});

test('displaySpectrum is deterministic and monotonically brighter with higher gainDb', () => {
  const sLow = { ...spectralDefaultsFor('pw'), gainDb: -6 };
  const sHigh = { ...spectralDefaultsFor('pw'), gainDb: 6 };
  const expected = new Float32Array(SPECTRAL_BINS);
  accumulateSpectrum([{ v: 0.65, weight: 1, dispersion: 0.1 }], sLow, true, expected);

  const d1 = new Float32Array(SPECTRAL_BINS);
  const d2 = new Float32Array(SPECTRAL_BINS);
  const dHigh = new Float32Array(SPECTRAL_BINS);

  displaySpectrum(expected, sLow, 99, 0, 0.25, d1, 2.5, 1.0);
  displaySpectrum(expected, sLow, 99, 0, 0.25, d2, 2.5, 1.0);
  displaySpectrum(expected, sHigh, 99, 0, 0.25, dHigh, 2.5, 1.0);

  assert.deepEqual(d1, d2);
  let strictlyHigher = 0;
  for (let b = 0; b < SPECTRAL_BINS; b++) {
    assert.ok(dHigh[b]! >= d1[b]! - 1e-6, `bin ${b} got dimmer at higher gain`);
    if (dHigh[b]! > d1[b]!) strictlyHigher++;
  }
  assert.ok(strictlyHigher > 10, 'higher gain should visibly brighten unsaturated spectral bins');
  assert.equal(columnsPerSecond(50, 200, 100), 100);
});

test('quadrature mirror (-40 dB) stays invisible at 0 dB gain, surfaces at +12 dB gain, and never alters envelope or auto-trace', () => {
  assert.equal(MIRROR_ISOLATION_DB, -40);
  assert.ok(Math.abs(MIRROR_ISOLATION_POWER - 1e-4) < 1e-9);

  const s0 = { ...spectralDefaultsFor('pw'), gainDb: 0, scaleMps: 1.0, baselineShiftMps: 0 };
  const s12 = { ...s0, gainDb: 12 };
  const expected = new Float32Array(SPECTRAL_BINS);
  accumulateSpectrum([{ v: 0.6, weight: 1, dispersion: 0.04 }], s0, true, expected);

  // Locate primary peak bin (+0.6 m/s) and its exact mirror bin (-0.6 m/s) plus a quiet control bin (-0.85 m/s)
  const { vMin, vMax } = spectralRange(s0);
  const span = vMax - vMin;
  const binOf = (v: number) => Math.round(((vMax - v) / span) * SPECTRAL_BINS - 0.5);
  const primaryBin = binOf(0.6);
  const mirrorBin = binOf(-0.6);
  const quietBin = binOf(-0.85);

  // Average over 40 independent column seeds so random Rayleigh speckle cancels cleanly
  let sumMirror0 = 0;
  let sumQuiet0 = 0;
  let sumMirror12 = 0;
  let sumQuiet12 = 0;
  const col0 = new Float32Array(SPECTRAL_BINS);
  const col12 = new Float32Array(SPECTRAL_BINS);
  for (let c = 0; c < 40; c++) {
    displaySpectrum(expected, s0, c, 0, 0.25, col0, 2.5, 1.0);
    displaySpectrum(expected, s12, c, 0, 0.25, col12, 2.5, 1.0);
    sumMirror0 += col0[mirrorBin]!;
    sumQuiet0 += col0[quietBin]!;
    sumMirror12 += col12[mirrorBin]!;
    sumQuiet12 += col12[quietBin]!;
  }
  const meanMirror0 = sumMirror0 / 40;
  const meanQuiet0 = sumQuiet0 / 40;
  const meanMirror12 = sumMirror12 / 40;
  const meanQuiet12 = sumQuiet12 / 40;

  assert.ok(
    meanMirror0 < 0.06,
    `At 0 dB gain, -40 dB mirror should stay near the black floor (<15/255), got ${(meanMirror0 * 255).toFixed(1)}/255`,
  );
  const mirror12Byte = meanMirror12 * 255;
  assert.ok(
    mirror12Byte >= 45 && mirror12Byte <= 95,
    `At +12 dB gain, -40 dB mirror should emerge clearly in [45, 95]/255, got ${mirror12Byte.toFixed(1)}/255`,
  );
  assert.ok(
    meanMirror12 > meanQuiet12 + 0.04 && meanMirror12 > meanMirror0 * 4,
    `At +12 dB gain, mirror bin (${meanMirror12.toFixed(3)}) must clearly emerge above thermal noise (${meanQuiet12.toFixed(3)}) and 0 dB level (${meanMirror0.toFixed(3)})`,
  );

  // Envelope buffer `out` returned by presentSpectralColumn must remain strictly free of mirror energy
  const envWithDisp = new Float32Array(expected);
  const envNoDisp = new Float32Array(expected);
  const dispOut = new Float32Array(SPECTRAL_BINS);
  presentSpectralColumn(envWithDisp, s12, 7, 42, 2.5, 1.0, 0, dispOut, 0.25);
  presentSpectralColumn(envNoDisp, s12, 7, 42, 2.5, 1.0, 0, undefined, 0.25);
  assert.deepEqual(
    envWithDisp,
    envNoDisp,
    'presentSpectralColumn display mirror pass must never mutate the envelope buffer',
  );
  assert.ok(envWithDisp[primaryBin]! > 0.7, 'primary positive jet must be bright in envelope buffer');
  assert.ok(
    envWithDisp[mirrorBin]! < 0.2,
    `envelope buffer must stay at noise floor at mirror bin, got ${envWithDisp[mirrorBin]}`,
  );

  // Auto-trace reads the envelope buffer, so tracing a strip built from `envWithDisp` tracks +0.6 m/s with zero mirror bias
  const cols = 60;
  const strip = new Float32Array(cols * SPECTRAL_BINS);
  for (let c = 10; c <= 45; c++) {
    for (let b = 0; b < SPECTRAL_BINS; b++) strip[c * SPECTRAL_BINS + b] = envWithDisp[b]!;
  }
  const traced = autoTraceSpectralEnvelope(strip, cols, cols, s12, 10, 45, true);
  assert.equal(traced.kind, 'autoTrace');
  assert.ok(
    traced.velocitiesMps.length >= 20,
    'auto-trace should recover the positive envelope cleanly',
  );
  const meanV =
    traced.velocitiesMps.reduce((acc, v) => acc + v, 0) / traced.velocitiesMps.length;
  assert.ok(
    meanV > 0.5 && meanV < 0.85,
    `auto-trace should follow true +0.6 m/s positive flow, got meanV=${meanV.toFixed(3)} m/s`,
  );
});
