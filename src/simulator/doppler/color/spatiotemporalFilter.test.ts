import test from 'node:test';
import assert from 'node:assert/strict';
import {
  adaptiveTemporalStep,
  applyColorSpatialFilter,
  colorPowerConfidence,
  spatiotemporalDefinesGlsl,
  ST_HYSTERESIS_RATIO,
  ST_PHASE_BILATERAL_K,
  ST_SPATIAL_LAMINAR_MIX,
  ST_SPATIAL_TURB_MIX,
  type TemporalStepInput,
  type TemporalStepOutput,
} from './spatiotemporalFilter';
import {
  allocColorField,
  COLOR_THRESHOLD,
  computeColorField,
  DEFAULT_COLOR,
  type ColorAcquisition,
  type ColorSettings,
} from './colorDoppler';
import { allocPolarFrame, type PolarFrameSpec } from '@/simulator/renderer/types';
import { Tissue } from '@/simulator/anatomy/tissue';
import { softTissueTransmission } from '@/simulator/renderer/acoustic/acoustics';

const ACQ: ColorAcquisition = { frequencyMHz: 2.5, harmonics: false };

function makeBloodFrame(lines = 20, samples = 36, depthCm = 16) {
  const spec: PolarFrameSpec = {
    lines,
    samples,
    sectorRad: 1.2,
    depthCm,
    focusCm: 8,
    elevationSamples: 1,
  };
  const frame = allocPolarFrame(spec);
  frame.tissue.fill(Tissue.Blood);
  for (let li = 0; li < lines; li++) {
    for (let si = 0; si < samples; si++) {
      const r = ((si + 0.5) / samples) * depthCm;
      frame.transmission[li * samples + si] = softTissueTransmission(
        r,
        ACQ.frequencyMHz,
        ACQ.harmonics,
      );
    }
  }
  return frame;
}

test('adaptiveTemporalStep prevents binary flicker on 1-frame blood drop-outs and cuts immediately on tissue', () => {
  const out: TemporalStepOutput = { valid: false, vel: NaN, variance: 0, power: 0 };
  const base: TemporalStepInput = {
    isBloodValid: true,
    hasCurr: false,
    vCurr: 0,
    varCurr: 0,
    powCurr: 0,
    hasPrev: true,
    vPrev: 0.42,
    varPrev: 0.05,
    powPrev: 0.34,
    scaleMps: 0.62,
    baselineShiftMps: 0,
    persistence: 0.4,
    smoothing: 0.6,
    colorThreshold: COLOR_THRESHOLD,
  };

  // Frame 1 drop-out over blood: stays smoothly visible (decayed power > hysteresis floor) instead of blinking black
  adaptiveTemporalStep(base, out);
  assert.equal(out.valid, true, 'momentary 1-frame drop-out over blood should smoothly decay');
  assert.ok(
    out.power < base.powPrev && out.power >= COLOR_THRESHOLD * ST_HYSTERESIS_RATIO,
    `expected decayed power in [${(COLOR_THRESHOLD * ST_HYSTERESIS_RATIO).toFixed(3)}, ${base.powPrev}), got ${out.power.toFixed(3)}`,
  );
  assert.ok(
    colorPowerConfidence(out.power, COLOR_THRESHOLD) <
      colorPowerConfidence(base.powPrev, COLOR_THRESHOLD),
    'decaying power should fade opacity monotonically',
  );

  // Sustained absence extinguishes cleanly within 2-3 frames
  let p = out.power;
  let survived = 0;
  for (let i = 0; i < 6; i++) {
    adaptiveTemporalStep({ ...base, powPrev: p }, out);
    if (!out.valid) break;
    p = out.power;
    survived++;
  }
  assert.ok(survived <= 2, `sustained drop-out must extinguish promptly, lingered ${survived} extra frames`);

  // Moving valve/myocardium (isBloodValid = false): must cut IMMEDIATELY to zero (zero wall ghosting)
  adaptiveTemporalStep({ ...base, isBloodValid: false, powPrev: 0.9 }, out);
  assert.equal(out.valid, false, 'non-blood/myocardium sample must never retain temporal color trail');
  assert.equal(out.power, 0);

  // smoothing = 0 disables hysteresis hold cleanly
  adaptiveTemporalStep({ ...base, smoothing: 0 }, out);
  assert.equal(out.valid, false, 'smoothing=0 must disable dropout hysteresis');
});

test('adaptiveTemporalStep stabilizes steady laminar flow more strongly than turbulent flow and stays Nyquist-safe', () => {
  const lamOut: TemporalStepOutput = { valid: false, vel: NaN, variance: 0, power: 0 };
  const turbOut: TemporalStepOutput = { valid: false, vel: NaN, variance: 0, power: 0 };
  const shared: Omit<TemporalStepInput, 'varCurr' | 'varPrev'> = {
    isBloodValid: true,
    hasCurr: true,
    vCurr: 0.48,
    powCurr: 0.65,
    hasPrev: true,
    vPrev: 0.36,
    powPrev: 0.65,
    scaleMps: 0.62,
    baselineShiftMps: 0,
    persistence: 0.4,
    smoothing: 0.8,
    colorThreshold: COLOR_THRESHOLD,
  };

  adaptiveTemporalStep({ ...shared, varCurr: 0.04, varPrev: 0.04 }, lamOut);
  adaptiveTemporalStep({ ...shared, varCurr: 0.68, varPrev: 0.65 }, turbOut);

  // Laminar pulls closer to history (0.36) while turbulent responds faster toward current (0.48)
  assert.ok(
    turbOut.vel > lamOut.vel + 0.02,
    `expected turbulent response (${turbOut.vel.toFixed(3)}) closer to current 0.48 than laminar (${lamOut.vel.toFixed(3)})`,
  );

  // Fast attack: sudden flow surge responds faster than steady power
  const steadyOut: TemporalStepOutput = { valid: false, vel: NaN, variance: 0, power: 0 };
  const surgeOut: TemporalStepOutput = { valid: false, vel: NaN, variance: 0, power: 0 };
  adaptiveTemporalStep({ ...shared, varCurr: 0.05, varPrev: 0.05, powPrev: 0.7, powCurr: 0.7 }, steadyOut);
  adaptiveTemporalStep({ ...shared, varCurr: 0.05, varPrev: 0.05, powPrev: 0.28, powCurr: 0.85 }, surgeOut);
  assert.ok(
    surgeOut.vel > steadyOut.vel,
    `expected fast attack on surging flow (${surgeOut.vel.toFixed(3)}) > steady (${steadyOut.vel.toFixed(3)})`,
  );

  // Nyquist-safe circular blend under shifted baseline (+0.25, scale 0.60 -> seam at +0.85 / -0.35)
  const seamOut: TemporalStepOutput = { valid: false, vel: NaN, variance: 0, power: 0 };
  adaptiveTemporalStep(
    {
      ...shared,
      scaleMps: 0.6,
      baselineShiftMps: 0.25,
      vPrev: 0.82,
      vCurr: -0.32,
      varCurr: 0.05,
      varPrev: 0.05,
    },
    seamOut,
  );
  assert.ok(
    seamOut.vel > 0.7 || seamOut.vel < -0.2,
    `expected shifted aliasing seam blend near +0.85/-0.35, got ${seamOut.vel.toFixed(3)}`,
  );
});

test('applyColorSpatialFilter removes isolated speckle, closes blood-only pinholes, and separates laminar vs turbulent flow', () => {
  const nL = 12;
  const nS = 12;
  const n = nL * nS;
  const bloodMask = new Float32Array(n).fill(1);
  const srcVel = new Float32Array(n).fill(NaN);
  const srcVar = new Float32Array(n);
  const srcPow = new Float32Array(n);
  const dstVel = new Float32Array(n).fill(NaN);
  const dstVar = new Float32Array(n);
  const dstPow = new Float32Array(n);

  // 1. Isolated single-pixel speckle at (1, 1)
  srcVel[1 * nS + 1] = 0.4;
  srcVar[1 * nS + 1] = 0.1;
  srcPow[1 * nS + 1] = 0.5;

  // 2. Solid laminar patch at rows 4..8, cols 4..8 with:
  //    - a single blood pinhole at (6, 6)
  //    - a non-blood valve pixel adjacent at (6, 9) (bloodMask = 0)
  //    - mild spatial velocity jitter + 1-pixel spurious variance ripple
  for (let l = 4; l <= 8; l++) {
    for (let s = 4; s <= 8; s++) {
      const i = l * nS + s;
      if (l === 6 && s === 6) continue; // blood pinhole
      const jitter = ((l + s) % 2 === 0 ? 0.06 : -0.06);
      srcVel[i] = 0.42 + jitter;
      srcVar[i] = l === 5 && s === 5 ? 0.18 : 0.04;
      srcPow[i] = 0.72;
    }
  }
  bloodMask[6 * nS + 9] = 0; // non-blood tissue right next to jet

  applyColorSpatialFilter({
    samples: nS,
    liMin: 0,
    liMax: nL - 1,
    siMin: 0,
    siMax: nS - 1,
    scaleMps: 0.62,
    baselineShiftMps: 0,
    smoothing: 0.7,
    bloodMask,
    srcVel,
    srcVar,
    srcPow,
    dstVel,
    dstVar,
    dstPow,
  });

  // Isolated speckle rejected
  assert.ok(Number.isNaN(dstVel[1 * nS + 1]), 'isolated single-pixel speckle must be rejected');

  // Blood pinhole closed cleanly with matching laminar velocity
  assert.ok(Number.isFinite(dstVel[6 * nS + 6]), 'isolated blood-pool pinhole must be closed');
  assert.ok(
    Math.abs(dstVel[6 * nS + 6]! - 0.42) < 0.05,
    `expected closed pinhole velocity ~0.42, got ${dstVel[6 * nS + 6]}`,
  );

  // Adjacent non-blood tissue NEVER filled
  assert.ok(Number.isNaN(dstVel[6 * nS + 9]), 'non-blood pixel must never receive pinhole bleed');

  // Laminar spatial velocity RMS & spurious variance ripple reduced significantly
  const rawErr = Math.abs(srcVel[5 * nS + 5]! - 0.42);
  const filtErr = Math.abs(dstVel[5 * nS + 5]! - 0.42);
  assert.ok(
    filtErr < rawErr * 0.55,
    `expected laminar jitter error (${filtErr.toFixed(4)}) << raw (${rawErr.toFixed(4)})`,
  );
  assert.ok(
    dstVar[5 * nS + 5]! < srcVar[5 * nS + 5]! * 0.55,
    `expected laminar spurious variance ripple (${dstVar[5 * nS + 5]!.toFixed(3)}) damped from ${srcVar[5 * nS + 5]}`,
  );

  // 3. Turbulent mosaic patch retains high variance and distinct phase contrast
  const tSrcVel = new Float32Array(n).fill(NaN);
  const tSrcVar = new Float32Array(n);
  const tSrcPow = new Float32Array(n);
  const tDstVel = new Float32Array(n).fill(NaN);
  const tDstVar = new Float32Array(n);
  const tDstPow = new Float32Array(n);
  const pattern = [0.55, -0.48, 0.22, -0.58, 0.44, -0.18, -0.52, 0.51, 0.12];
  for (let l = 2; l <= 8; l++) {
    for (let s = 2; s <= 8; s++) {
      const i = l * nS + s;
      tSrcVel[i] = pattern[(l * 3 + s) % pattern.length]!;
      tSrcVar[i] = 0.62;
      tSrcPow[i] = 0.78;
    }
  }
  applyColorSpatialFilter({
    samples: nS,
    liMin: 0,
    liMax: nL - 1,
    siMin: 0,
    siMax: nS - 1,
    scaleMps: 0.62,
    baselineShiftMps: 0,
    smoothing: 0.7,
    bloodMask,
    srcVel: tSrcVel,
    srcVar: tSrcVar,
    srcPow: tSrcPow,
    dstVel: tDstVel,
    dstVar: tDstVar,
    dstPow: tDstPow,
  });

  let lamMeanVar = 0;
  let lamCount = 0;
  let turbMeanVar = 0;
  let turbCount = 0;
  let turbVelSpread = 0;
  for (let l = 5; l <= 7; l++) {
    for (let s = 5; s <= 7; s++) {
      const i = l * nS + s;
      lamMeanVar += dstVar[i]!;
      lamCount++;
      turbMeanVar += tDstVar[i]!;
      turbCount++;
      turbVelSpread += Math.abs(tDstVel[i]! - tSrcVel[i]!);
    }
  }
  lamMeanVar /= lamCount;
  turbMeanVar /= turbCount;
  assert.ok(
    turbMeanVar > 0.6 && turbMeanVar - lamMeanVar > 0.52,
    `expected crisp laminar (${lamMeanVar.toFixed(3)}) vs turbulent (${turbMeanVar.toFixed(3)}) variance separation`,
  );
  assert.ok(
    turbVelSpread / turbCount < 0.16,
    `expected turbulent mosaic velocities preserved (mean shift ${(turbVelSpread / turbCount).toFixed(3)} m/s)`,
  );
});

test('computeColorField multi-frame Kasai flicker drops with spatiotemporal filtering while keeping laminar-turbulent separation', () => {
  const frame = makeBloodFrame(20, 36, 16);
  const n = frame.spec.lines * frame.spec.samples;
  const baseBox: ColorSettings = {
    ...DEFAULT_COLOR,
    boxThetaMinRad: -0.55,
    boxThetaMaxRad: 0.55,
    boxRMinCm: 2,
    boxRMaxCm: 14,
    scaleMps: 0.62,
    // the default wall filter (its old 0.05 m/s meant one near it, and asks now for less than the packet's lowest)
    wallFilterMps: DEFAULT_COLOR.wallFilterMps,
    gainDb: -1, // border-line gain that induces stochastic pixel flicker when unfiltered
  };

  const measureFlicker = (settings: ColorSettings) => {
    let prev = allocColorField(n);
    let curr = allocColorField(n);
    let toggles = 0;
    let pairs = 0;
    let velDeltaSum = 0;
    let velPairs = 0;
    for (let step = 0; step < 6; step++) {
      computeColorField(
        frame,
        settings,
        step === 0 ? null : prev,
        (_idx, _li, _si, res) => {
          res.v = 0.42;
          res.disp = 0.04;
          res.present = 1;
        },
        curr,
        ACQ,
        { realization: step + 1, seed: 77 },
      );
      if (step >= 2) {
        for (let li = 4; li < 16; li++) {
          for (let si = 6; si < 30; si++) {
            const idx = li * 36 + si;
            const a = Number.isFinite(prev.vel[idx]);
            const b = Number.isFinite(curr.vel[idx]);
            pairs++;
            if (a !== b) toggles++;
            if (a && b) {
              velDeltaSum += Math.abs(curr.vel[idx]! - prev.vel[idx]!);
              velPairs++;
            }
          }
        }
      }
      const swap = prev;
      prev = curr;
      curr = swap;
    }
    return {
      toggleRate: toggles / Math.max(1, pairs),
      meanVelJitter: velDeltaSum / Math.max(1, velPairs),
    };
  };

  const raw = measureFlicker({ ...baseBox, persistence: 0.35, smoothing: 0 });
  const filtered = measureFlicker({ ...baseBox, persistence: 0.35, smoothing: 0.7 });

  assert.ok(
    filtered.toggleRate < raw.toggleRate * 0.65,
    `expected spatiotemporal filter to cut pixel flicker toggle rate (${(filtered.toggleRate * 100).toFixed(1)}% vs raw ${(raw.toggleRate * 100).toFixed(1)}%)`,
  );
  assert.ok(
    filtered.meanVelJitter < raw.meanVelJitter * 0.8,
    `expected spatiotemporal filter to smooth laminar velocity jitter (${filtered.meanVelJitter.toFixed(4)} vs raw ${raw.meanVelJitter.toFixed(4)} m/s)`,
  );
});

test('spatiotemporalDefinesGlsl exports all shared constants for GPU shader parity', () => {
  const defs = spatiotemporalDefinesGlsl();
  assert.ok(defs.includes(`#define ST_HYSTERESIS_RATIO ${ST_HYSTERESIS_RATIO}`));
  assert.ok(defs.includes(`#define ST_SPATIAL_LAMINAR_MIX ${ST_SPATIAL_LAMINAR_MIX}`));
  assert.ok(defs.includes(`#define ST_SPATIAL_TURB_MIX ${ST_SPATIAL_TURB_MIX}`));
  assert.ok(defs.includes(`#define ST_PHASE_BILATERAL_K ${ST_PHASE_BILATERAL_K}`));
});
