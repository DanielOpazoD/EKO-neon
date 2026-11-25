import test from 'node:test';
import assert from 'node:assert/strict';
import {
  allocColorField,
  computeColorField,
  DEFAULT_COLOR,
  DOPPLER_SHADOW_TRANSMISSION,
  overlayColorField,
  wallFilterResponse,
  type ColorAcquisition,
  type ColorSettings,
} from './colorDoppler';
import type { ScanLut } from '@/simulator/renderer/scanConvert';
import {
  buildFlowParams,
  sampleFlow,
  type FlowSample,
} from '@/simulator/doppler/flow-primitives/flowField';
import { allocPolarFrame, type PolarFrameSpec } from '@/simulator/renderer/types';
import { Tissue } from '@/simulator/anatomy/tissue';
import { softTissueTransmission } from '@/simulator/renderer/acoustic/acoustics';
import { loadCaseById } from '@/cases';
import { buildCaseModels, REST_PATIENT } from '@/simulator/anatomy/caseModels';
import { aorticVelocityProfile } from '@/simulator/doppler/flow-primitives/aorticNetwork';
import { COLOR_MAP } from '@/simulator/renderer/postprocess/colorMap';
import { computeHeartPose } from '@/simulator/anatomy/heartModel';
import { cycleStateAt, sampleTable } from '@/simulator/cardiac-cycle/cycleModel';

const ACQ: ColorAcquisition = { frequencyMHz: 2.5, harmonics: false };

function makeBloodFrame(lines = 16, samples = 32, depthCm = 16): ReturnType<typeof allocPolarFrame> {
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

const FULL_BOX: ColorSettings = {
  ...DEFAULT_COLOR,
  boxThetaMinRad: -0.6,
  boxThetaMaxRad: 0.6,
  boxRMinCm: 1,
  boxRMaxCm: 15,
  scaleMps: 0.6,
  baselineShiftMps: 0,
  wallFilterMps: 0.05,
  gainDb: 0,
  persistence: 0,
  invert: false,
};

test('colorDoppler wraps velocities above Nyquist and respects baseline shift', () => {
  const frame = makeBloodFrame();
  const n = frame.spec.lines * frame.spec.samples;
  const out = allocColorField(n);

  // +0.90 m/s with Nyquist 0.60 m/s wraps around to -0.30 m/s
  computeColorField(
    frame,
    FULL_BOX,
    null,
    (_idx, _li, _si, res) => {
      res.v = 0.9;
      res.disp = 0.05;
      res.present = 1;
    },
    out,
    ACQ,
  );
  const centerIdx = 8 * frame.spec.samples + 16;
  assert.ok(Number.isFinite(out.vel[centerIdx]), 'expected colored sample inside box');
  assert.ok(
    Math.abs(out.vel[centerIdx]! - -0.3) < 1e-4,
    `expected aliased -0.30 m/s, got ${out.vel[centerIdx]}`,
  );

  // Shifting baseline by +0.35 m/s expands positive range up to +0.95 m/s without wrap
  const shifted: ColorSettings = { ...FULL_BOX, baselineShiftMps: 0.35 };
  computeColorField(
    frame,
    shifted,
    null,
    (_idx, _li, _si, res) => {
      res.v = 0.9;
      res.disp = 0.05;
      res.present = 1;
    },
    out,
    ACQ,
  );
  assert.ok(
    Math.abs(out.vel[centerIdx]! - 0.9) < 1e-4,
    `expected unaliased +0.90 m/s after baseline shift, got ${out.vel[centerIdx]}`,
  );
});

test('colorDoppler rejects wall-filtered slow flow and shadowed samples', () => {
  const frame = makeBloodFrame();
  const n = frame.spec.lines * frame.spec.samples;
  const centerIdx = 8 * frame.spec.samples + 16;
  const out = allocColorField(n);

  assert.ok(wallFilterResponse(0.05, 0.05) > 0.7 && wallFilterResponse(0.05, 0.05) < 0.72);

  // Below wall filter -> no color
  computeColorField(
    frame,
    FULL_BOX,
    null,
    (_idx, _li, _si, res) => {
      res.v = 0.02;
      res.disp = 0.02;
      res.present = 1;
    },
    out,
    ACQ,
  );
  assert.ok(Number.isNaN(out.vel[centerIdx]), 'slow flow below wall filter should be suppressed');

  // Deep acoustic shadow -> no color even for fast flow
  frame.transmission[centerIdx] =
    frame.transmission[centerIdx]! * (DOPPLER_SHADOW_TRANSMISSION * 0.25);
  computeColorField(
    frame,
    { ...FULL_BOX, gainDb: 12 },
    null,
    (_idx, _li, _si, res) => {
      res.v = 0.45;
      res.disp = 0.05;
      res.present = 1;
    },
    out,
    ACQ,
  );
  assert.ok(Number.isNaN(out.vel[centerIdx]), 'acoustic shadow must suppress color at any gain');
});

test('colorDoppler encodes laminar vs turbulent variance and uses phasor persistence across Nyquist', () => {
  const frame = makeBloodFrame();
  const n = frame.spec.lines * frame.spec.samples;
  const centerIdx = 8 * frame.spec.samples + 16;

  // Laminar vs turbulent variance (Kasai packet estimate with sufficient gain above noise coherence gate)
  const laminar = allocColorField(n);
  const turbulent = allocColorField(n);
  const kasaiSettings: ColorSettings = { ...FULL_BOX, gainDb: 6 };
  computeColorField(
    frame,
    kasaiSettings,
    null,
    (_idx, _li, _si, res) => {
      res.v = 0.4;
      res.disp = 0.04;
      res.present = 1;
    },
    laminar,
    ACQ,
    { realization: 3, seed: 101 },
  );
  computeColorField(
    frame,
    kasaiSettings,
    null,
    (_idx, _li, _si, res) => {
      res.v = 0.4;
      res.disp = 0.65;
      res.present = 1;
    },
    turbulent,
    ACQ,
    { realization: 3, seed: 101 },
  );
  const meanVar = (f: typeof laminar) => {
    let sum = 0,
      count = 0;
    for (let i = 0; i < f.vel.length; i++) {
      if (Number.isFinite(f.vel[i])) {
        sum += f.variance[i]!;
        count++;
      }
    }
    return count > 0 ? sum / count : 0;
  };
  const meanLam = meanVar(laminar);
  const meanTurb = meanVar(turbulent);
  assert.ok(
    meanTurb > meanLam + 0.15,
    `expected turbulent mean variance (${meanTurb.toFixed(3)}) > laminar (${meanLam.toFixed(3)})`,
  );

  // Phasor persistence across Nyquist wrap: +0.58 and -0.58 at scale 0.60 stay near ±0.60, never 0
  const prev = allocColorField(n);
  const curr = allocColorField(n);
  const persistSettings: ColorSettings = { ...FULL_BOX, persistence: 0.5 };
  computeColorField(
    frame,
    persistSettings,
    null,
    (_idx, _li, _si, res) => {
      res.v = 0.58;
      res.disp = 0.04;
      res.present = 1;
    },
    prev,
    ACQ,
  );
  computeColorField(
    frame,
    persistSettings,
    prev,
    (_idx, _li, _si, res) => {
      res.v = -0.58;
      res.disp = 0.04;
      res.present = 1;
    },
    curr,
    ACQ,
  );
  assert.ok(
    Math.abs(curr.vel[centerIdx]!) > 0.5,
    `expected phasor persistence across Nyquist to stay near ±0.60 m/s, got ${curr.vel[centerIdx]}`,
  );
});

test('flowField reflects aortic stenosis jet acceleration and severe MR systolic regurgitant jet', () => {
  const normal = loadCaseById('normal-excellent');
  const severeAs = loadCaseById('aortic-stenosis-severe');
  const severeMr = loadCaseById('mvp-primary-mr');

  const sampleMidSystole = (
    caseId: ReturnType<typeof loadCaseById>,
    pointFn: (
      flow: ReturnType<typeof buildFlowParams>,
      hp: ReturnType<typeof computeHeartPose>,
    ) => [number, number, number],
  ): FlowSample => {
    const { heart, thorax, tables } = buildCaseModels(caseId, REST_PATIENT);
    const flow = buildFlowParams(caseId, heart, tables, thorax);
    const { ejectionStartS, ejectionEndS } = tables.timings;
    const phase = (ejectionStartS + 0.45 * (ejectionEndS - ejectionStartS)) / tables.rrS;
    const hp = computeHeartPose(heart, cycleStateAt(tables, phase, flow.flowPeaks));
    const [x, y, z] = pointFn(flow, hp);
    const out: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };
    sampleFlow(flow, tables, hp, phase, x, y, z, out);
    return out;
  };

  // Mid-ejection: vena contracta just distal to aortic valve orifice along avAxis
  const avSamplePoint = (
    flow: ReturnType<typeof buildFlowParams>,
    hp: ReturnType<typeof computeHeartPose>,
  ): [number, number, number] => [
    flow.avCenter.x + flow.avAxis.x * 0.25,
    flow.avCenter.y + flow.avAxis.y * 0.25,
    flow.avCenter.z + hp.zAnn * 0.4 + flow.avAxis.z * 0.25,
  ];
  const normalAv = sampleMidSystole(normal, avSamplePoint);
  const asAv = sampleMidSystole(severeAs, avSamplePoint);
  const normalAvSpeed = Math.hypot(normalAv.vx, normalAv.vy, normalAv.vz);
  const asAvSpeed = Math.hypot(asAv.vx, asAv.vy, asAv.vz);
  assert.ok(
    asAvSpeed > normalAvSpeed * 2.5 && asAvSpeed > 3.5,
    `expected severe AS jet (${asAvSpeed.toFixed(2)} m/s) >> normal (${normalAvSpeed.toFixed(2)} m/s)`,
  );
  assert.ok(
    asAv.dispersion > normalAv.dispersion,
    `expected severe AS turbulence dispersion (${asAv.dispersion}) > normal (${normalAv.dispersion})`,
  );

  // Mid-ejection: inside LA 1.0 cm behind mitral orifice along MR jet axis
  const mrSamplePoint = (
    flow: ReturnType<typeof buildFlowParams>,
    hp: ReturnType<typeof computeHeartPose>,
  ): [number, number, number] => {
    const dir = flow.mrJetDirRad;
    return [
      flow.mvCenter.x + Math.sin(dir) * 1.0,
      flow.mvCenter.y,
      hp.zAnn - Math.cos(dir) * 1.0,
    ];
  };
  const normalLa = sampleMidSystole(normal, mrSamplePoint);
  const mrLa = sampleMidSystole(severeMr, mrSamplePoint);
  const normalLaSpeed = Math.hypot(normalLa.vx, normalLa.vy, normalLa.vz);
  const mrLaSpeed = Math.hypot(mrLa.vx, mrLa.vy, mrLa.vz);
  assert.ok(
    mrLaSpeed > 2.5 && mrLaSpeed > normalLaSpeed + 2.0,
    `expected severe MR regurgitant jet (${mrLaSpeed.toFixed(2)} m/s) >> normal LA (${normalLaSpeed.toFixed(2)} m/s)`,
  );

  // Mid-systole: LV-side PISA convergence hemisphere 0.9 cm upstream of mitral orifice stays laminar and accelerates past 0.40 m/s
  const pisaPt = (
    flow: ReturnType<typeof buildFlowParams>,
    hp: ReturnType<typeof computeHeartPose>,
  ): [number, number, number] => [
    flow.mvCenter.x,
    flow.mvCenter.y,
    hp.zAnn + 0.85,
  ];
  const mrPisa = sampleMidSystole(severeMr, pisaPt);
  const mrPisaSpeed = Math.hypot(mrPisa.vx, mrPisa.vy, mrPisa.vz);
  assert.ok(
    mrPisaSpeed >= 0.35 && mrPisaSpeed <= 1.2 && mrPisa.dispersion <= 0.12,
    `expected laminar PISA shell around 0.40 m/s at 0.85 cm upstream, got v=${mrPisaSpeed.toFixed(2)} m/s disp=${mrPisa.dispersion.toFixed(2)}`,
  );
});

test('overlayColorField preserves unaliased baseline-shifted velocity, interpolates shifted aliasing seams, and maintains laminar Kasai fill', () => {
  // 1. Baseline-shifted overlay: scale=0.60, baseline=+0.35 -> range [-0.25, +0.95] m/s
  const lut: ScanLut = {
    key: 'baseline-test',
    lines: 2,
    samples: 2,
    idx: new Int32Array([0, 0]),
    inPix: new Int32Array([0, 1]),
    inIdx: new Int32Array([0, 0]),
    inWl: new Uint16Array([512, 512]),
    inWs: new Uint16Array([0, 0]),
    inDl: new Uint8Array([1, 1]),
    inDs: new Uint8Array([0, 0]),
    wl: new Float32Array([0.5, 0.5]),
    ws: new Float32Array([0.0, 0.0]),
    li: new Int16Array([0, 0]),
    si: new Int16Array([0, 0]),
    rCm: new Float32Array([6, 6]),
    theta: new Float32Array([0, 0]),
  };
  const shiftedSettings: ColorSettings = {
    ...FULL_BOX,
    scaleMps: 0.6,
    baselineShiftMps: 0.35,
    showVariance: false,
  };

  // Unaliased +0.85 m/s must paint bright red/yellow (R >> B), NOT wrap into blue
  const rgbaUnaliased = new Uint8ClampedArray([20, 20, 20, 255, 20, 20, 20, 255]);
  const velPos = new Float32Array([0.85, 0.85, 0.85, 0.85]);
  const zeroVar = new Float32Array(4);
  const fullPow = new Float32Array([1, 1, 1, 1]);
  overlayColorField(rgbaUnaliased, lut, velPos, zeroVar, shiftedSettings, fullPow);
  assert.ok(
    rgbaUnaliased[0]! > 190 && rgbaUnaliased[0]! > rgbaUnaliased[2]! + 120,
    `expected bright red/yellow for unaliased +0.85 m/s under shifted baseline, got rgb=(${rgbaUnaliased[0]},${rgbaUnaliased[1]},${rgbaUnaliased[2]})`,
  );

  // Straddling the shifted aliasing boundary (+0.92 and -0.22 around seam +0.95/-0.25) must stay bright, never dark zero
  const rgbaSeam = new Uint8ClampedArray([20, 20, 20, 255, 20, 20, 20, 255]);
  const velSeam = new Float32Array([0.92, 0.92, -0.22, -0.22]);
  overlayColorField(rgbaSeam, lut, velSeam, zeroVar, shiftedSettings, fullPow);
  const seamPeak = Math.max(rgbaSeam[0]!, rgbaSeam[1]!, rgbaSeam[2]!);
  assert.ok(
    seamPeak > 160,
    `expected bright shifted aliasing seam interpolation > 160, got rgb=(${rgbaSeam[0]},${rgbaSeam[1]},${rgbaSeam[2]})`,
  );

  // 2. Multi-look Kasai laminar fill continuity at 0 dB gain (minimal drop-out holes inside strong laminar flow)
  const frame = makeBloodFrame(20, 40, 16);
  const out = allocColorField(frame.spec.lines * frame.spec.samples);
  computeColorField(
    frame,
    { ...FULL_BOX, gainDb: 0 },
    null,
    (_idx, _li, _si, res) => {
      res.v = 0.45;
      res.disp = 0.04;
      res.present = 1;
    },
    out,
    ACQ,
    { realization: 5, seed: 42 },
  );
  let filled = 0;
  let total = 0;
  for (let li = 4; li < 16; li++) {
    for (let si = 8; si < 32; si++) {
      total++;
      if (Number.isFinite(out.vel[li * 40 + si])) filled++;
    }
  }
  const fillFrac = filled / total;
  assert.ok(
    fillFrac >= 0.92,
    `expected >= 92% laminar fill continuity at 0 dB gain, got ${(fillFrac * 100).toFixed(1)}%`,
  );
});

test('Color Doppler physiological realism: Womersley profile, normal LVOT/MV laminar continuity, Severe AS ascending plume, and tissue write priority', () => {
  // 1. Womersley blunt aortic profile: unit-disk average = 1.0, no-slip at wall, peak/mean ≈ 1.333
  const steps = 1000;
  let diskAvg = 0;
  for (let i = 0; i < steps; i++) {
    const r = (i + 0.5) / steps;
    diskAvg += (2 * r * aorticVelocityProfile(r)) / steps;
  }
  assert.ok(
    Math.abs(diskAvg - 1.0) < 1e-3,
    `expected Womersley disk average = 1.0, got ${diskAvg.toFixed(5)}`,
  );
  assert.equal(aorticVelocityProfile(1.0), 0, 'no-slip boundary condition at aortic wall');
  assert.ok(
    Math.abs(aorticVelocityProfile(0) - 4 / 3) < 1e-6,
    `expected blunt peak-to-mean = 4/3, got ${aorticVelocityProfile(0)}`,
  );

  // 2. Normal LVOT → AV → ascending aorta continuity across STJ (no abrupt velocity cliff or upstream sign flip)
  const normal = loadCaseById('normal-excellent-window');
  const normModels = buildCaseModels(normal, REST_PATIENT);
  const normFlow = buildFlowParams(normal, normModels.heart, normModels.tables, normModels.thorax);
  let sysPeakPhase = 0.24;
  let maxQao = -1;
  for (let p = 0.05; p < 0.45; p += 0.005) {
    const q = sampleTable(normModels.tables.aorticFlowMlps, p);
    if (q > maxQao) {
      maxQao = q;
      sysPeakPhase = p;
    }
  }
  const hpSys = computeHeartPose(
    normModels.heart,
    cycleStateAt(normModels.tables, sysPeakPhase, normFlow.flowPeaks),
  );
  const sTmp: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };
  const sampleAvAxis = (
    flow: ReturnType<typeof buildFlowParams>,
    tables: typeof normModels.tables,
    hp: typeof hpSys,
    phase: number,
    t: number,
    offY = 0,
  ) => {
    sampleFlow(
      flow,
      tables,
      hp,
      phase,
      flow.avCenter.x + flow.avAxis.x * t,
      flow.avCenter.y + flow.avAxis.y * t + offY,
      flow.avCenter.z + hp.zAnn * 0.5 + flow.avAxis.z * t,
      sTmp,
    );
    return {
      speed: Math.hypot(sTmp.vx, sTmp.vy, sTmp.vz),
      along: sTmp.vx * flow.avAxis.x + sTmp.vy * flow.avAxis.y + sTmp.vz * flow.avAxis.z,
      dispersion: sTmp.dispersion,
    };
  };
  const stjBefore = sampleAvAxis(normFlow, normModels.tables, hpSys, sysPeakPhase, 1.7);
  const stjMid = sampleAvAxis(normFlow, normModels.tables, hpSys, sysPeakPhase, 2.1);
  const stjAfter = sampleAvAxis(normFlow, normModels.tables, hpSys, sysPeakPhase, 2.5);
  assert.ok(
    Math.abs(stjMid.speed - stjBefore.speed) < 0.35 && Math.abs(stjAfter.speed - stjMid.speed) < 0.35,
    `expected smooth STJ velocity transition without cliff, got ${stjBefore.speed.toFixed(2)} -> ${stjMid.speed.toFixed(2)} -> ${stjAfter.speed.toFixed(2)} m/s`,
  );
  const upstreamOffAxis = sampleAvAxis(normFlow, normModels.tables, hpSys, sysPeakPhase, -1.5, 0.55);
  assert.ok(
    upstreamOffAxis.along > 0 && upstreamOffAxis.dispersion < 0.12,
    `expected laminar forward LVOT upstream flow without sign reversal, got along=${upstreamOffAxis.along.toFixed(2)} disp=${upstreamOffAxis.dispersion.toFixed(2)}`,
  );

  // 3. Normal mitral inflow: smooth annulus-to-tips acceleration and strictly laminar (< VARIANCE_MIN)
  let ePeakPhase = 0.55;
  let maxQmv = -1;
  for (let p = 0.4; p < 0.8; p += 0.005) {
    const q = sampleTable(normModels.tables.mitralFlowMlps, p);
    if (q > maxQmv) {
      maxQmv = q;
      ePeakPhase = p;
    }
  }
  const hpE = computeHeartPose(
    normModels.heart,
    cycleStateAt(normModels.tables, ePeakPhase, normFlow.flowPeaks),
  );
  sampleFlow(normFlow, normModels.tables, hpE, ePeakPhase, normFlow.mvCenter.x, normFlow.mvCenter.y, hpE.zAnn + 0.02, sTmp);
  const vAnnulus = Math.hypot(sTmp.vx, sTmp.vy, sTmp.vz);
  sampleFlow(normFlow, normModels.tables, hpE, ePeakPhase, normFlow.mvCenter.x, normFlow.mvCenter.y, hpE.zAnn + 1.1, sTmp);
  const vTips = Math.hypot(sTmp.vx, sTmp.vy, sTmp.vz);
  const dispTips = sTmp.dispersion;
  assert.ok(
    vAnnulus < vTips && vAnnulus > vTips * 0.65,
    `expected smooth mitral annulus-to-tips acceleration, got annulus=${vAnnulus.toFixed(2)} tips=${vTips.toFixed(2)}`,
  );
  assert.ok(
    dispTips < Math.min(0.15, COLOR_MAP.VARIANCE_MIN),
    `expected normal mitral inflow free of false turbulence variance, got disp=${dispTips.toFixed(3)}`,
  );

  // 4. Severe AS post-stenotic turbulent jet persists past t = 2.0 cm into the ascending aorta
  const severeAs = loadCaseById('aortic-stenosis-severe');
  const asModels = buildCaseModels(severeAs, REST_PATIENT);
  const asFlow = buildFlowParams(severeAs, asModels.heart, asModels.tables, asModels.thorax);
  let asSysPhase = 0.28;
  let maxAsQ = -1;
  for (let p = 0.05; p < 0.45; p += 0.005) {
    const q = sampleTable(asModels.tables.aorticFlowMlps, p);
    if (q > maxAsQ) {
      maxAsQ = q;
      asSysPhase = p;
    }
  }
  const hpAs = computeHeartPose(
    asModels.heart,
    cycleStateAt(asModels.tables, asSysPhase, asFlow.flowPeaks),
  );
  const asDistal = sampleAvAxis(asFlow, asModels.tables, hpAs, asSysPhase, 2.4);
  assert.ok(
    asDistal.speed > 1.8 && asDistal.dispersion > 0.35,
    `expected Severe AS post-stenotic turbulent jet past t=2.0 cm, got v=${asDistal.speed.toFixed(2)} disp=${asDistal.dispersion.toFixed(2)}`,
  );

  // 5. Tissue write priority suppresses color on bright endocardium/valves (lum >= 172) while preserving dark blood lumen
  const lut: ScanLut = {
    key: 'twp-test',
    lines: 2,
    samples: 2,
    idx: new Int32Array([0, 0]),
    inPix: new Int32Array([0, 1]),
    inIdx: new Int32Array([0, 0]),
    inWl: new Uint16Array([512, 512]),
    inWs: new Uint16Array([0, 0]),
    inDl: new Uint8Array([1, 1]),
    inDs: new Uint8Array([0, 0]),
    wl: new Float32Array([0.5, 0.5]),
    ws: new Float32Array([0.0, 0.0]),
    li: new Int16Array([0, 0]),
    si: new Int16Array([0, 0]),
    rCm: new Float32Array([6, 6]),
    theta: new Float32Array([0, 0]),
  };
  // Pixel 0: dark blood (lum=24), Pixel 1: bright aortic wall / calcified leaflet (lum=210)
  const rgbaTwp = new Uint8ClampedArray([24, 24, 24, 255, 210, 210, 210, 255]);
  const velArr = new Float32Array([0.5, 0.5, 0.5, 0.5]);
  const varArr = new Float32Array(4);
  const powArr = new Float32Array([1, 1, 1, 1]);
  overlayColorField(rgbaTwp, lut, velArr, varArr, FULL_BOX, powArr);
  assert.ok(
    rgbaTwp[0]! > rgbaTwp[2]! + 120,
    `expected strong BART red inside dark blood lumen, got rgb=(${rgbaTwp[0]},${rgbaTwp[1]},${rgbaTwp[2]})`,
  );
  assert.equal(rgbaTwp[4], 210, 'bright wall/valve tissue must suppress Color Doppler bleed cleanly');
  assert.equal(rgbaTwp[5], 210, 'bright wall/valve tissue must remain grayscale');
  assert.equal(rgbaTwp[6], 210, 'bright wall/valve tissue must remain grayscale');
});

test('eccentric MR splits LA regurgitant jet and LV PISA cleanly along the anatomical mitral annular plane', () => {
  const mrCase = loadCaseById('mvp-primary-mr');
  const models = buildCaseModels(mrCase, REST_PATIENT);
  const flow = buildFlowParams(mrCase, models.heart, models.tables, models.thorax);
  let sysPhase = 0.2;
  let maxMrQ = -1;
  for (let p = 0.05; p < 0.45; p += 0.005) {
    const q = sampleTable(models.tables.mrFlowMlps, p);
    if (q > maxMrQ) {
      maxMrQ = q;
      sysPhase = p;
    }
  }
  const hp = computeHeartPose(
    models.heart,
    cycleStateAt(models.tables, sysPhase, flow.flowPeaks),
  );
  const ox = flow.mvCenter.x;
  const oy = flow.mvCenter.y;
  const oz = hp.zAnn - 0.1;
  const fs: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };

  // Symmetric LV-side points at equal distance from mitral orifice (dx = ±0.55, dz = +0.35):
  // Previously, the -35° tilted `along >= 0` cut clipped the dx = -0.55 LV side into the LA jet branch.
  sampleFlow(flow, models.tables, hp, sysPhase, ox - 0.55, oy, oz + 0.35, fs);
  const leftLvSpeed = Math.hypot(fs.vx, fs.vy, fs.vz);
  const leftLvDisp = fs.dispersion;
  const leftLvTowardsOrifice = fs.vz < 0 && fs.vx > 0;

  sampleFlow(flow, models.tables, hp, sysPhase, ox + 0.55, oy, oz + 0.35, fs);
  const rightLvSpeed = Math.hypot(fs.vx, fs.vy, fs.vz);
  const rightLvTowardsOrifice = fs.vz < 0 && fs.vx < 0;

  assert.ok(leftLvTowardsOrifice && rightLvTowardsOrifice, 'Both sides of LV PISA must converge radially toward the mitral orifice');
  assert.ok(
    leftLvSpeed > 0.25 && Math.abs(leftLvSpeed - rightLvSpeed) < 0.12,
    `Eccentric MR LV PISA should remain symmetric across the annular plane: left=${leftLvSpeed.toFixed(2)} right=${rightLvSpeed.toFixed(2)}`,
  );
  assert.ok(
    leftLvDisp <= 0.15,
    `LV PISA side must stay laminar (not evaluated as turbulent LA jet), got dispersion=${leftLvDisp.toFixed(2)}`,
  );
});
