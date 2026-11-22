import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { summarizeEnvelope } from '@/simulator/measurements/vti';
import {
  computeDecelerationTime,
  computeLinearCm,
  computeMModeExcursionCm,
  resampleManualVti,
  stripVelocityAtY,
} from './measurementTools';

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
});
