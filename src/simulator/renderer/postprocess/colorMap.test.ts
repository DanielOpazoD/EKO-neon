import assert from 'node:assert/strict';
import test from 'node:test';
import { COLOR_BLEND, COLOR_MAP, colorMap, colorMapDefinesGlsl } from './colorMap';

test('colorMap follows BART polarity (red toward, blue away), brightens with speed, and shifts green on variance', () => {
  const posSlow: [number, number, number] = [0, 0, 0];
  const posFast: [number, number, number] = [0, 0, 0];
  const negFast: [number, number, number] = [0, 0, 0];
  const turb: [number, number, number] = [0, 0, 0];

  colorMap(0.2, 0.62, 0, false, posSlow);
  colorMap(0.55, 0.62, 0, false, posFast);
  colorMap(-0.55, 0.62, 0, false, negFast);
  colorMap(0.3, 0.62, 0.85, true, turb);

  // Positive velocity: red dominant over blue; faster velocity is brighter
  assert.ok(posSlow[0] > posSlow[2], 'toward flow must be red-dominant');
  assert.ok(posFast[0] > posSlow[0], 'faster toward flow must brighten red');
  assert.ok(posFast[1] > posSlow[1], 'high speed above knee adds yellow/green highlight');

  // Negative velocity: blue dominant over red
  assert.ok(negFast[2] > negFast[0], 'away flow must be blue-dominant');

  // High variance shifts green strongly upward when showVariance is enabled
  const laminarSame: [number, number, number] = [0, 0, 0];
  colorMap(0.3, 0.62, 0.0, true, laminarSame);
  assert.ok(turb[1] > laminarSame[1] + 100, 'turbulent variance must shift green channel upward');
  assert.ok(turb[0] < laminarSame[0], 'variance fades dominant channel toward mosaic green');
});

test('colorMapDefinesGlsl exports every COLOR_MAP constant and COLOR_BLEND as GLSL floats', () => {
  const glsl = colorMapDefinesGlsl();
  for (const [k, v] of Object.entries(COLOR_MAP)) {
    const expected = Number.isInteger(v) ? `${v}.0` : `${v}`;
    assert.ok(glsl.includes(`#define CM_${k} ${expected}`), `missing GLSL define for CM_${k}`);
  }
  assert.ok(glsl.includes(`#define COLOR_BLEND ${COLOR_BLEND}`));
});

test('colorMap maps asymmetric baseline-shifted ranges cleanly and keeps laminar flow free of variance tint', () => {
  const scale = 0.6;
  const shift = 0.25; // range [-0.35, +0.85] m/s (clinical PISA baseline shift)
  const slowPos: [number, number, number] = [0, 0, 0];
  const peakPos: [number, number, number] = [0, 0, 0];
  const peakNeg: [number, number, number] = [0, 0, 0];
  const lamNoVar: [number, number, number] = [0, 0, 0];
  const lamLowVar: [number, number, number] = [0, 0, 0];

  colorMap(0.08, scale, 0, false, slowPos, shift);
  colorMap(0.85, scale, 0, false, peakPos, shift);
  colorMap(-0.35, scale, 0, false, peakNeg, shift);
  colorMap(0.4, scale, 0, true, lamNoVar, shift);
  colorMap(0.4, scale, 0.18, true, lamLowVar, shift);

  // Slow flow near baseline starts deep dark maroon rather than harsh neon red
  assert.ok(slowPos[0] < 100 && slowPos[0] > slowPos[2], `expected deep maroon near baseline, got ${slowPos}`);
  // Positive peak (+0.85 m/s) saturates to bright yellow-red; negative peak (-0.35 m/s) saturates to bright cyan-blue
  assert.ok(peakPos[0] >= 250 && peakPos[1] >= 220, `expected bright yellow-red at +0.85, got ${peakPos}`);
  assert.ok(peakNeg[2] >= 250 && peakNeg[1] >= 220, `expected bright cyan-blue at -0.35, got ${peakNeg}`);
  // Laminar low variance (< VARIANCE_MIN) adds zero green shift
  assert.deepEqual(lamLowVar, lamNoVar, 'low laminar variance must remain pure BART without green tint');
});
