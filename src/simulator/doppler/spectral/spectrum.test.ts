import assert from 'node:assert/strict';
import test from 'node:test';
import {
  accumulateSpectrum,
  buildSpectralColumn,
  columnsPerSecond,
  displaySpectrum,
  isClickColumn,
  SPECTRAL_BINS,
  spectralDefaultsFor,
  spectralRange,
  type VelocitySample,
} from './spectrum';

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

test('isClickColumn distinguishes valve clicks from laminar flow', () => {
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
