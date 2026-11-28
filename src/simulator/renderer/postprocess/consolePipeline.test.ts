import test from 'node:test';
import assert from 'node:assert/strict';
import { allocPolarFrame, DEFAULT_ACQUISITION, type AcquisitionSettings } from '../types';
import {
  applyConsole,
  consoleCompensation,
  createConsoleState,
  LOG_KNEE_SPAN,
  LOG_KNEE_START,
  persistenceOverTime,
  softKneeCompress,
  tgcAtDepth,
} from './consolePipeline';

function makeFrame(lines = 24, samples = 48, fillAmp = 0.02) {
  const frame = allocPolarFrame({
    lines,
    samples,
    sectorRad: Math.PI / 2,
    depthCm: 16,
    focusCm: 8,
    elevationSamples: 1,
  });
  frame.amplitude.fill(fillAmp);
  return frame;
}

function meanU8(buf: Uint8ClampedArray): number {
  let sum = 0;
  for (let i = 0; i < buf.length; i++) sum += buf[i]!;
  return sum / buf.length;
}

test('persistenceOverTime is time-invariant across frame rate drops and clamps short intervals', () => {
  const p = 0.5;
  const dt = 1 / 30;
  // Two frames at nominal interval equal one frame after 2x interval: p^2 === persistenceOverTime(p, 2*dt, dt)
  const w1 = persistenceOverTime(p, dt, dt);
  const w2 = persistenceOverTime(p, 2 * dt, dt);
  assert.ok(Math.abs(w1 - p) < 1e-12);
  assert.ok(Math.abs(w2 - p * p) < 1e-12);

  // Quarter-interval floor prevents near-zero elapsed times from freezing persistence at 1.0
  assert.ok(Math.abs(persistenceOverTime(p, 0, dt) - Math.pow(p, 0.25)) < 1e-12);
  assert.equal(persistenceOverTime(0, dt, dt), 0);
  assert.equal(persistenceOverTime(p, dt, 0), 0);
});

test('tgcAtDepth interpolates linearly across depth bands and consoleCompensation rises with depth & gain', () => {
  const s: AcquisitionSettings = {
    ...DEFAULT_ACQUISITION,
    depthCm: 14,
    tgcDb: [-6, -4, -2, 0, 2, 4, 6, 8],
  };
  assert.ok(Math.abs(tgcAtDepth(s, 0) - -6) < 1e-9);
  assert.ok(Math.abs(tgcAtDepth(s, 14) - 8) < 1e-9);
  assert.ok(Math.abs(tgcAtDepth(s, 7) - 1) < 1e-9);

  const spec = makeFrame().spec;
  const compBase = new Float64Array(spec.samples);
  const compBoost = new Float64Array(spec.samples);
  consoleCompensation({ ...DEFAULT_ACQUISITION, gainDb: 0 }, spec, compBase);
  consoleCompensation({ ...DEFAULT_ACQUISITION, gainDb: 6 }, spec, compBoost);

  // Depth compensation increases monotonically into the far field
  assert.ok(compBase[spec.samples - 1]! > compBase[0]!);
  // +6 dB gain scales linear compensation by 10^(6/20) ≈ 1.99526 at every depth
  const expectedRatio = Math.pow(10, 6 / 20);
  for (let i = 0; i < spec.samples; i++) {
    assert.ok(Math.abs(compBoost[i]! / compBase[i]! - expectedRatio) < 1e-6);
  }
});

test('applyConsole is deterministic per (frameIndex, seed) and monotonic in gain and TGC', () => {
  const frame = makeFrame(24, 48, 0.015);
  const n = frame.spec.lines * frame.spec.samples;
  const outA = new Uint8ClampedArray(n);
  const outB = new Uint8ClampedArray(n);
  const settings: AcquisitionSettings = { ...DEFAULT_ACQUISITION, persistence: 0, sri: 0 };

  const stA = createConsoleState(42);
  const stB = createConsoleState(42);
  applyConsole(frame, settings, stA, outA);
  applyConsole(frame, settings, stB, outB);
  assert.deepEqual(outA, outB, 'identical (frameIndex, seed) must yield bit-identical pixels');
  assert.equal(stA.frameIndex, 1);

  // Higher overall gain must brighten the image
  const outBright = new Uint8ClampedArray(n);
  applyConsole(frame, { ...settings, gainDb: settings.gainDb + 12 }, createConsoleState(42), outBright);
  assert.ok(meanU8(outBright) > meanU8(outA) + 10, 'higher gain must increase mean brightness');

  // Boosting far-field TGC must brighten deep samples without brightening near samples
  const tgcBaseSettings: AcquisitionSettings = { ...settings, gainDb: -18 };
  const outBaseFar = new Uint8ClampedArray(n);
  const outFarTgc = new Uint8ClampedArray(n);
  applyConsole(frame, tgcBaseSettings, createConsoleState(42), outBaseFar);
  applyConsole(
    frame,
    { ...tgcBaseSettings, tgcDb: [0, 0, 0, 0, 0, 0, 12, 12] },
    createConsoleState(42),
    outFarTgc,
  );
  const nearIdx = 2;
  const farIdx = frame.spec.samples - 2;
  assert.equal(outFarTgc[nearIdx], outBaseFar[nearIdx]);
  assert.ok(outFarTgc[farIdx]! > outBaseFar[farIdx]!);
});

test('applyConsole dynamic range and gray maps behave monotonically across amplitude ramp', () => {
  const frame = makeFrame(8, 64, 0);
  for (let li = 0; li < frame.spec.lines; li++) {
    for (let si = 0; si < frame.spec.samples; si++) {
      // Smooth exponential ramp from weak blood to bright pericardium
      frame.amplitude[li * frame.spec.samples + si] = 1e-4 * Math.pow(10, (si / 63) * 3.5);
    }
  }
  const n = frame.spec.lines * frame.spec.samples;
  const base: AcquisitionSettings = {
    ...DEFAULT_ACQUISITION,
    persistence: 0,
    edgeEnhance: 0,
    sri: 0,
    tgcDb: [0, 0, 0, 0, 0, 0, 0, 0],
  };

  // Every gray map produces higher brightness on strong reflectors than on weak echoes
  for (const grayMap of ['linear', 's-curve', 'high-contrast', 'clinical'] as const) {
    const out = new Uint8ClampedArray(n);
    applyConsole(frame, { ...base, grayMap }, createConsoleState(7), out);
    const lowMean =
      (out[2]! + out[64 + 2]! + out[128 + 2]! + out[192 + 2]!) / 4;
    const highMean =
      (out[60]! + out[64 + 60]! + out[128 + 60]! + out[192 + 60]!) / 4;
    assert.ok(highMean > lowMean + 50, `${grayMap} must map strong echoes much brighter than weak echoes`);
  }

  // Narrower dynamic range (higher contrast) suppresses low-amplitude echoes more than wide DR
  const outNarrow = new Uint8ClampedArray(n);
  const outWide = new Uint8ClampedArray(n);
  applyConsole(frame, { ...base, dynamicRangeDb: 40, grayMap: 'linear' }, createConsoleState(7), outNarrow);
  applyConsole(frame, { ...base, dynamicRangeDb: 80, grayMap: 'linear' }, createConsoleState(7), outWide);
  const weakAvgNarrow = (outNarrow[8]! + outNarrow[64 + 8]!) / 2;
  const weakAvgWide = (outWide[8]! + outWide[64 + 8]!) / 2;
  assert.ok(weakAvgWide > weakAvgNarrow, 'wider dynamic range must preserve more low-level echoes');
});

test('softKneeCompress is C1-continuous, exact identity below knee, and preserves highlight gradation above REF_DB', () => {
  assert.equal(softKneeCompress(-0.2), 0);
  assert.equal(softKneeCompress(0), 0);
  for (const y of [0.1, 0.45, 0.72, LOG_KNEE_START]) {
    assert.ok(Math.abs(softKneeCompress(y) - y) < 1e-12, `identity broken at y=${y}`);
  }
  // C1 slope continuity across LOG_KNEE_START
  const eps = 1e-5;
  const slopeLeft = (softKneeCompress(LOG_KNEE_START) - softKneeCompress(LOG_KNEE_START - eps)) / eps;
  const slopeRight = (softKneeCompress(LOG_KNEE_START + eps) - softKneeCompress(LOG_KNEE_START)) / eps;
  assert.ok(Math.abs(slopeLeft - 1) < 1e-4 && Math.abs(slopeRight - 1) < 1e-3);

  // Strictly monotonic across shoulder up to full saturation at LOG_KNEE_START + LOG_KNEE_SPAN
  let prev = softKneeCompress(LOG_KNEE_START);
  for (let i = 1; i <= 24; i++) {
    const y = LOG_KNEE_START + (LOG_KNEE_SPAN * i) / 24;
    const cur = softKneeCompress(y);
    assert.ok(cur > prev, `not strictly monotonic at y=${y}: ${cur} <= ${prev}`);
    prev = cur;
  }
  assert.equal(softKneeCompress(LOG_KNEE_START + LOG_KNEE_SPAN + 0.1), 1);

  // Highlight reflectors above old 1.0 clip retain internal contrast instead of flat-clipping to 255
  assert.ok(softKneeCompress(1.0) > 0.92 && softKneeCompress(1.0) < 0.97);
  assert.ok(softKneeCompress(1.16) > softKneeCompress(1.0) + 0.03);
});

