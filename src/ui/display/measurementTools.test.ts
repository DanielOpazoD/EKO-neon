import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  aseSweepDurationS,
  dopplerShiftHz,
  dopplerVelocityFromShiftHz,
  vtiFromCmpsAndMs,
  vtiFromEnvelope,
} from '@/clinical/formulas';
import { summarizeEnvelope } from '@/simulator/measurements/vti';
import {
  computeDecelerationTime,
  computeLinearCm,
  computeMModeExcursionCm,
  resampleManualVti,
  stripVelocityAtY,
  summarizeManualVti,
} from './measurementTools';
import { computeSpectralTicks } from './overlayDraw';

describe('Display measurement tools math', () => {
  it('computes 2D caliper linear distance and M-mode TAPSE excursion in cm', () => {
    // 3-4-5 triangle at 25 px/cm => 50 px = 2.0 cm
    const dist = computeLinearCm({ x: 100, y: 120 }, { x: 130, y: 160 }, 25);
    assert.ok(Math.abs(dist - 2.0) < 1e-9);

    // M-mode strip 0..16 cm across 160 px => 22 px vertical excursion = 2.2 cm
    const tapse = computeMModeExcursionCm(
      { x: 50, y: 400 },
      { x: 90, y: 378 },
      { height: 160, topValue: 0, bottomValue: 16 },
    );
    assert.ok(Math.abs(tapse - 2.2) < 1e-9);
  });

  it('extrapolates mitral deceleration time to the zero baseline and rejects invalid slopes', () => {
    const strip = {
      y: 300,
      height: 200,
      topValue: 1.5,
      bottomValue: -0.5,
      secondsPerColumn: 0.005, // 5 ms/px
    };
    // Baseline (0 m/s) sits at y = 300 + (1.5 / 2.0) * 200 = 450
    // Peak 1.0 m/s sits at y = 300 + (0.5 / 2.0) * 200 = 350
    // Mid-slope 0.5 m/s sits at y = 300 + (1.0 / 2.0) * 200 = 400
    assert.ok(Math.abs(stripVelocityAtY(350, strip) - 1.0) < 1e-9);
    assert.ok(Math.abs(stripVelocityAtY(400, strip) - 0.5) < 1e-9);

    // Drop from 1.0 to 0.5 m/s over 20 px (100 ms) => full drop to 0 takes 200 ms at x = 140
    const dt = computeDecelerationTime({ x: 100, y: 350 }, { x: 120, y: 400 }, strip);
    assert.ok(dt !== null);
    assert.ok(Math.abs(dt.dtMs - 200) < 1e-6);
    assert.ok(Math.abs(dt.peakMps - 1.0) < 1e-6);
    assert.ok(Math.abs(dt.intercept.x - 140) < 1e-6);
    assert.ok(Math.abs(dt.intercept.y - 450) < 1e-6);

    // Clicking earlier or higher velocity on second point returns null
    assert.equal(computeDecelerationTime({ x: 120, y: 350 }, { x: 100, y: 400 }, strip), null);
    assert.equal(computeDecelerationTime({ x: 100, y: 400 }, { x: 120, y: 350 }, strip), null);
  });

  it('resamples unsorted manual VTI clicks into a triangular spectral envelope with exact VTI', () => {
    const strip = {
      y: 300,
      height: 200,
      topValue: 2.0,
      bottomValue: -2.0,
    };
    // y=400 => 0 m/s, y=350 => +1.0 m/s; triangle base 40 px (0.20 s at 5 ms/px), height 1.0 m/s
    // Theoretical integral = 0.5 * 0.20 s * 1.0 m/s = 0.10 m = 10.0 cm
    const { sorted, velocitiesMps } = resampleManualVti(
      [
        { x: 120, y: 350 },
        { x: 140, y: 400 },
        { x: 100, y: 400 },
      ],
      strip,
    );
    assert.deepEqual(
      sorted.map((p) => p.x),
      [100, 120, 140],
    );
    assert.equal(velocitiesMps.length, 41);
    const summary = summarizeEnvelope(velocitiesMps, 0.005);
    assert.ok(Math.abs(summary.vtiCm - 10.0) < 0.3, `Expected ~10 cm VTI, got ${summary.vtiCm}`);
    assert.ok(Math.abs(summary.vmaxMps - 1.0) < 1e-6);
  });

  it('integrates sub-pixel manual VTI polygons analytically and satisfies ASE cm/s vs ms scaling', () => {
    // ASE frequency-to-velocity round trip: 100 cm/s at 2.5 MHz, c = 1540 m/s
    const f0Hz = 2.5e6;
    const fdHz = dopplerShiftHz(f0Hz, 1.0, 1, 1540);
    assert.ok(Math.abs(dopplerVelocityFromShiftHz(fdHz, f0Hz, 1, 1540) - 1.0) < 1e-12);
    assert.equal(aseSweepDurationS(25), 4.0);
    assert.equal(aseSweepDurationS(50), 2.0);
    assert.equal(aseSweepDurationS(100), 1.0);

    // Clinical unit identity: ∫ v(cm/s)·dt(ms)/1000 === ∫ v(m/s)·dt(s)·100
    const vMps = [0, 0.5, 1.0, 0.5, 0];
    const vCmps = vMps.map((v) => v * 100);
    assert.ok(Math.abs(vtiFromCmpsAndMs(vCmps, 50) - vtiFromEnvelope(vMps, 0.05)) < 1e-12);

    // Non-integer x coordinates across 25 / 50 / 100 mm/s sweep speeds yield identical exact 15.0 cm VTI
    // Triangle of duration 300 ms and peak 100 cm/s => VTI = 0.5 * 100 * 300 / 1000 = 15.0 cm
    // Mean gradient of linear triangle from 0 to 1.0 m/s = (1/3) * 4 * 1^2 = 1.3333 mmHg
    for (const sweep of [25, 50, 100]) {
      const stripWidthPx = 400;
      const spc = aseSweepDurationS(sweep) / stripWidthPx;
      const dxPx = 0.3 / spc; // exact 300 ms span in fractional pixels
      const x0 = 50.37;
      const strip = {
        y: 200,
        height: 200,
        topValue: 1.0,
        bottomValue: -1.0,
        secondsPerColumn: spc,
      };
      // y=300 => 0 cm/s, y=200 => +100 cm/s
      const { summary } = summarizeManualVti(
        [
          { x: x0, y: 300 },
          { x: x0 + dxPx * 0.4, y: 200 },
          { x: x0 + dxPx, y: 300 },
        ],
        strip,
      );
      assert.ok(
        Math.abs(summary.vtiCm - 15.0) < 1e-9,
        `Expected exact 15.0 cm VTI at ${sweep} mm/s, got ${summary.vtiCm}`,
      );
      assert.ok(Math.abs(summary.vmaxMps - 1.0) < 1e-9);
      assert.ok(Math.abs(summary.meanGradientMmHg - 4 / 3) < 1e-9);
    }
  });

  it('computes ASE baseline-anchored velocity ticks in cm/s (PW/TDI) and m/s (CW)', () => {
    // Symmetric PW ±100 cm/s => +100 cm/s, 50, 0, -50, -100 with 0 dead-on frac 0.5
    const pwSym = computeSpectralTicks(1.0, -1.0, true);
    assert.deepEqual(
      pwSym.map((t) => t.label),
      ['100 cm/s', '50', '0', '-50', '-100'],
    );
    const zeroSym = pwSym.find((t) => Math.abs(t.valueMps) < 1e-6);
    assert.ok(zeroSym && Math.abs(zeroSym.frac - 0.5) < 1e-9);

    // Shifted PW baseline (+140 to -60 cm/s) => 0 stays locked exactly at frac = 1.4 / 2.0 = 0.7
    const pwShifted = computeSpectralTicks(1.4, -0.6, true);
    const zeroShifted = pwShifted.find((t) => Math.abs(t.valueMps) < 1e-6);
    assert.ok(zeroShifted && Math.abs(zeroShifted.frac - 0.7) < 1e-9);
    assert.ok(pwShifted[0]!.label.endsWith(' cm/s'));

    // Depth-limited PW Nyquist (±86 cm/s) => clean 40 cm/s or 50 cm/s ticks anchored at 0
    const pwDepth = computeSpectralTicks(0.86, -0.86, true);
    assert.ok(pwDepth.some((t) => Math.abs(t.valueMps) < 1e-6));
    assert.ok(pwDepth.length >= 3 && pwDepth.length <= 7);

    // CW ±6.0 m/s => clean 2 m/s steps in m/s anchored at 0
    const cw = computeSpectralTicks(6.0, -6.0, false);
    assert.deepEqual(
      cw.map((t) => t.label),
      ['6 m/s', '4', '2', '0', '-2', '-4', '-6'],
    );
  });
});
