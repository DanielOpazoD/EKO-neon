import test from 'node:test';
import assert from 'node:assert/strict';
import {
  allocColorField,
  colorSpectralWidthMps,
  computeColorField,
  DEFAULT_COLOR,
  DOPPLER_SHADOW_TRANSMISSION,
  overlayColorField,
  type ColorAcquisition,
  type ColorSettings,
} from './colorDoppler';
import type { ScanLut } from '@/simulator/renderer/scanConvert';
import {
  buildFlowParams,
  cavalVenousVelocityMps,
  coronarySinusVelocityMps,
  mrJetInitialDirection,
  sampleFlow,
  sampleLeafletVelocity,
  sampleTissueVelocity,
  type FlowSample,
} from '@/simulator/doppler/flow-primitives/flowField';
import { allocPolarFrame, type PolarFrameSpec } from '@/simulator/renderer/types';
import { Structure, Tissue } from '@/simulator/anatomy/tissue';
import { softTissueTransmission } from '@/simulator/renderer/acoustic/acoustics';
import { loadCaseById } from '@/cases';
import { buildCaseModels, REST_PATIENT } from '@/simulator/anatomy/caseModels';
import { aorticVelocityProfile } from '@/simulator/doppler/flow-primitives/aorticNetwork';
import { COLOR_MAP } from '@/simulator/renderer/postprocess/colorMap';
import { colorWallFilterRangeMps } from './kasaiEstimator';
import { computeHeartPose, heartAnchors } from '@/simulator/anatomy/heartModel';
import { regionalMeanFraction } from '@/simulator/anatomy/lvGeometry';
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

// the default wall filter (DEFAULT_COLOR's, bounded to 0.32 Nyquist at 0.6 m/s): its old 0.05 m/s meant a filter near
// it and, read as the −3 dB cutoff it now is, asks for less than the packet's lowest
const FULL_BOX: ColorSettings = {
  ...DEFAULT_COLOR,
  boxThetaMinRad: -0.6,
  boxThetaMaxRad: 0.6,
  boxRMinCm: 1,
  boxRMaxCm: 15,
  scaleMps: 0.6,
  baselineShiftMps: 0,
  gainDb: 0,
  persistence: 0,
  invert: false,
};

test('colorDoppler wraps velocities above Nyquist and respects baseline shift', () => {
  const frame = makeBloodFrame();
  const n = frame.spec.lines * frame.spec.samples;
  const out = allocColorField(n);
  const meanInteriorVel = (f: typeof out) => {
    let sum = 0;
    let count = 0;
    for (let li = 4; li < 12; li++) {
      for (let si = 8; si < 24; si++) {
        const v = f.vel[li * frame.spec.samples + si]!;
        if (Number.isFinite(v)) {
          sum += v;
          count++;
        }
      }
    }
    assert.ok(count > 80, `expected colored interior samples, got ${count}`);
    return sum / count;
  };

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
    { realization: 1, seed: 42 },
  );
  const aliasedMean = meanInteriorVel(out);
  assert.ok(
    Math.abs(aliasedMean - -0.3) < 0.04,
    `expected aliased mean near -0.30 m/s, got ${aliasedMean.toFixed(4)}`,
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
    { realization: 1, seed: 42 },
  );
  const shiftedMean = meanInteriorVel(out);
  assert.ok(
    Math.abs(shiftedMean - 0.9) < 0.04,
    `expected unaliased mean near +0.90 m/s after baseline shift, got ${shiftedMean.toFixed(4)}`,
  );
});

test('colorDoppler rejects wall-filtered slow flow and shadowed samples', () => {
  const frame = makeBloodFrame();
  const n = frame.spec.lines * frame.spec.samples;
  const centerIdx = 8 * frame.spec.samples + 16;
  const out = allocColorField(n);
  const interiorFill = (f: typeof out) => {
    let colored = 0;
    let total = 0;
    for (let li = 4; li < 12; li++) {
      for (let si = 8; si < 24; si++) {
        total++;
        if (Number.isFinite(f.vel[li * frame.spec.samples + si])) colored++;
      }
    }
    return colored / total;
  };

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
    { realization: 1, seed: 42 },
  );
  assert.ok(interiorFill(out) < 0.02, `slow flow below wall filter should be suppressed, got ${interiorFill(out)}`);

  // The Kasai regression filter cuts at the request bounded to what 8 pulses allow at this scale: asked under the
  // lowest cutoff (mean removal, 0.11 Nyquist) it cuts at that lowest; asked over the highest (0.32 Nyquist) at that highest
  const { min, max } = colorWallFilterRangeMps(FULL_BOX.scaleMps);
  const fillAt = (wallFilterMps: number, v: number) => {
    computeColorField(
      frame,
      { ...FULL_BOX, wallFilterMps, smoothing: 0 },
      null,
      (_idx, _li, _si, res) => {
        res.v = v;
        res.disp = 0.02;
        res.present = 1;
      },
      out,
      ACQ,
      { realization: 2, seed: 42 },
    );
    return interiorFill(out);
  };
  assert.ok(fillAt(0.02, 0.5 * min) < 0.1, 'asked under the lowest cutoff, flow well under that lowest is dark');
  assert.ok(fillAt(0.02, 1.8 * min) > 0.85, 'asked under the lowest cutoff, flow well over that lowest is coloured');
  assert.ok(fillAt(0.5, 0.5 * max) < 0.1, 'asked over the highest cutoff, flow well under that highest is dark');
  assert.ok(fillAt(0.5, 1.8 * max) > 0.85, 'asked over the highest cutoff, flow well over that highest is coloured');

  // Deep acoustic shadow -> no color even for fast flow at +12 dB gain
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
    { realization: 3, seed: 42 },
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
    { realization: 1, seed: 101 },
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
    { realization: 2, seed: 101 },
  );
  assert.ok(
    Math.abs(curr.vel[centerIdx]!) > 0.45,
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

test('overlayColorField circularly re-wraps stored velocities on frozen baseline shift and applies selected palette', () => {
  const lut: ScanLut = {
    key: 'frozen-baseline-test',
    lines: 2,
    samples: 2,
    idx: new Int32Array([0, 0]),
    inPix: new Int32Array([0, 1]),
    inIdx: new Int32Array([0, 0]),
    inWl: new Uint16Array([0, 0]),
    inWs: new Uint16Array([0, 0]),
    inDl: new Uint8Array([0, 0]),
    inDs: new Uint8Array([0, 0]),
    wl: new Float32Array([0, 0]),
    ws: new Float32Array([0, 0]),
    li: new Int16Array([0, 0]),
    si: new Int16Array([0, 0]),
    rCm: new Float32Array([6, 6]),
    theta: new Float32Array([0, 0]),
  };

  // Stored at scale = 0.68, acqBaseline = 0:
  // True velocity +0.86 m/s wrapped modulo 2*0.68 = 1.36 m/s to -0.50 m/s (blue when baseline=0).
  const storedVel = new Float32Array([-0.5, -0.5, -0.5, -0.5]);
  const storedVar = new Float32Array(4);
  const storedPow = new Float32Array([1, 1, 1, 1]);

  // At baseline = 0, stored -0.50 renders blue (away)
  const rgbaAtZero = new Uint8ClampedArray([16, 16, 16, 255, 16, 16, 16, 255]);
  overlayColorField(
    rgbaAtZero,
    lut,
    storedVel,
    storedVar,
    { ...FULL_BOX, scaleMps: 0.68, baselineShiftMps: 0, palette: 'vivid' },
    storedPow,
  );
  assert.ok(
    rgbaAtZero[2]! > rgbaAtZero[0]! + 120,
    `expected aliased blue at baseline=0, got rgb=(${rgbaAtZero[0]},${rgbaAtZero[1]},${rgbaAtZero[2]})`,
  );

  // Shifting frozen baseline to +0.30 (range [-0.38, +0.98]) unwraps -0.50 -> +0.86 m/s (bright toward red/yellow!)
  const rgbaShiftedVivid = new Uint8ClampedArray([16, 16, 16, 255, 16, 16, 16, 255]);
  overlayColorField(
    rgbaShiftedVivid,
    lut,
    storedVel,
    storedVar,
    { ...FULL_BOX, scaleMps: 0.68, baselineShiftMps: 0.3, palette: 'vivid' },
    storedPow,
  );
  assert.ok(
    rgbaShiftedVivid[0]! > rgbaShiftedVivid[2]! + 120 && rgbaShiftedVivid[1]! > 100,
    `expected unwrapped bright red-yellow (+0.86 m/s) at baseline=+0.30, got rgb=(${rgbaShiftedVivid[0]},${rgbaShiftedVivid[1]},${rgbaShiftedVivid[2]})`,
  );

  // Switching frozen palette to ice-fire lifts blue quiet channel toward white-gold at +0.86 m/s
  const rgbaShiftedIceFire = new Uint8ClampedArray([16, 16, 16, 255, 16, 16, 16, 255]);
  overlayColorField(
    rgbaShiftedIceFire,
    lut,
    storedVel,
    storedVar,
    { ...FULL_BOX, scaleMps: 0.68, baselineShiftMps: 0.3, palette: 'ice-fire' },
    storedPow,
  );
  assert.ok(
    rgbaShiftedIceFire[2]! > rgbaShiftedVivid[2]! + 40,
    `ice-fire palette should lift high-speed warm tip blue channel: vivid=${rgbaShiftedVivid[2]} ice-fire=${rgbaShiftedIceFire[2]}`,
  );
});

test('Color Doppler 90° black-band regression and transverse aperture geometric broadening (v_perp)', () => {
  // 1. 90° black-band regression: horizontal flow 0.80 m/s crossing sector center at θ = 90°. Its axial velocity
  //    0.8·sin θ falls under the wall filter within a band whose half-width follows the cutoff: about 4° at the packet's
  //    lowest (7.6 cm/s), 12–13° at the default (20 cm/s), as main's default drew it (the old 0.06 m/s these checks ran
  //    at asks now for the lowest)
  const lines = 64;
  const samples = 40;
  const frame = makeBloodFrame(lines, samples, 16);
  const box: ColorSettings = {
    ...DEFAULT_COLOR,
    boxThetaMinRad: -0.6,
    boxThetaMaxRad: 0.6,
    boxRMinCm: 4,
    boxRMaxCm: 12,
    scaleMps: 0.68,
    baselineShiftMps: 0,
    gainDb: 0,
    persistence: 0,
    smoothing: 0,
  };
  const vTotal = 0.8;
  /** Share of the samples coloured (%) at |θ − 90°| within [d0, d1] degrees, with this wall filter. */
  const bandFill = (wallFilterMps: number) => {
    const outBand = allocColorField(lines * samples);
    computeColorField(
      frame,
      { ...box, wallFilterMps },
      null,
      (_idx, li, _si, res) => {
        // Beam angle relative to vertical axis: theta=0 is perpendicular (90°) to horizontal vessel
        const theta = -frame.spec.sectorRad / 2 + (frame.spec.sectorRad * (li + 0.5)) / lines;
        res.v = vTotal * Math.sin(theta);
        res.vPerp = Math.abs(vTotal * Math.cos(theta));
        res.disp = 0.04;
        res.present = 1;
      },
      outBand,
      ACQ,
      { realization: 7, seed: 202 },
    );
    return (d0: number, d1: number) => {
      let total = 0;
      let filled = 0;
      for (let li = 0; li < lines; li++) {
        const theta = -frame.spec.sectorRad / 2 + (frame.spec.sectorRad * (li + 0.5)) / lines;
        const degFromPerp = Math.abs((theta * 180) / Math.PI);
        if (degFromPerp < d0 || degFromPerp > d1) continue;
        for (let si = 12; si < 28; si++) {
          total++;
          if (Number.isFinite(outBand.vel[li * samples + si])) filled++;
        }
      }
      assert.ok(total > 0);
      return (filled / total) * 100;
    };
  };
  const lowest = bandFill(colorWallFilterRangeMps(box.scaleMps).min);
  const byDefault = bandFill(DEFAULT_COLOR.wallFilterMps);
  assert.ok(
    lowest(0, 2) <= 5,
    `Expected 90° black band (|θ-90°| ≤ 2°) to stay ≤ 5% filled at the lowest wall filter, got ${lowest(0, 2).toFixed(1)}%`,
  );
  assert.ok(
    lowest(20, 30) >= 70,
    `Expected off-90° lobes (20°–30° off perpendicular) to fill strongly, got ${lowest(20, 30).toFixed(1)}%`,
  );
  // the default filter: the band is dark out to 10° (measured 1 %; 6–10° fill 97–100 % at the lowest cutoff), and the
  // lobes beyond its 20 cm/s cutoff (14.5°) fill as before
  assert.ok(
    byDefault(0, 10) <= 5,
    `Expected the default wall filter to keep |θ-90°| ≤ 10° dark (≤ 5%), got ${byDefault(0, 10).toFixed(1)}%`,
  );
  assert.ok(
    lowest(6, 10) >= 90,
    `Expected the lowest wall filter to colour 6°–10° off perpendicular (≥ 90%), got ${lowest(6, 10).toFixed(1)}%`,
  );
  assert.ok(
    byDefault(20, 30) >= 70,
    `Expected off-90° lobes to fill strongly at the default wall filter, got ${byDefault(20, 30).toFixed(1)}%`,
  );

  // 2. Transverse aperture broadening via absolute sigmaVMps: >3× width at θ=80° and elevated Kasai variance
  const sigma0 = colorSpectralWidthMps(0.14, 0.04, 0, 8);
  const sigma80 = colorSpectralWidthMps(0.14, 0.04, 0.79, 8);
  assert.ok(
    sigma80 > sigma0 * 3,
    `Expected sigmaVMps at θ=80° (${sigma80.toFixed(4)}) > 3× axial-only (${sigma0.toFixed(4)})`,
  );

  const frame2 = makeBloodFrame(20, 40, 16);
  const n2 = 20 * 40;
  const kasaiCfg: ColorSettings = {
    ...DEFAULT_COLOR,
    boxThetaMinRad: -0.5,
    boxThetaMaxRad: 0.5,
    boxRMinCm: 5,
    boxRMaxCm: 9,
    scaleMps: 0.68,
    // the packet's lowest cutoff (7.6 cm/s): the 14 cm/s axial flow below must pass for its variance to be measured
    wallFilterMps: colorWallFilterRangeMps(0.68).min,
    gainDb: 6,
    persistence: 0,
    smoothing: 0,
  };
  const meanVar = (f: ReturnType<typeof allocColorField>) => {
    let sum = 0;
    let count = 0;
    for (let i = 0; i < f.vel.length; i++) {
      if (Number.isFinite(f.vel[i])) {
        sum += f.variance[i]!;
        count++;
      }
    }
    return count > 0 ? sum / count : 0;
  };

  // Near-90° transverse jet (v_axial = 0.14, v_perp = 1.6 m/s at r = 5..9 cm) elevates Kasai variance measurably
  const noPerp = allocColorField(n2);
  const withPerp = allocColorField(n2);
  computeColorField(
    frame2,
    kasaiCfg,
    null,
    (_idx, _li, _si, res) => {
      res.v = 0.14;
      res.vPerp = 0;
      res.disp = 0.04;
      res.present = 1;
    },
    noPerp,
    ACQ,
    { realization: 4, seed: 55 },
  );
  computeColorField(
    frame2,
    kasaiCfg,
    null,
    (_idx, _li, _si, res) => {
      res.v = 0.14;
      res.vPerp = 1.6;
      res.disp = 0.04;
      res.present = 1;
    },
    withPerp,
    ACQ,
    { realization: 4, seed: 55 },
  );
  const vNoPerp = meanVar(noPerp);
  const vWithPerp = meanVar(withPerp);
  assert.ok(
    vWithPerp >= vNoPerp + 0.05,
    `Expected transverse v_perp to raise Kasai variance by >= 0.05 (noPerp=${vNoPerp.toFixed(3)}, withPerp=${vWithPerp.toFixed(3)})`,
  );

  // Pure axial laminar flow (θ = 0°, v_perp = 0) stays completely unaffected (exact zero delta)
  const axialDefault = allocColorField(n2);
  const axialZeroPerp = allocColorField(n2);
  computeColorField(
    frame2,
    kasaiCfg,
    null,
    (_idx, _li, _si, res) => {
      res.v = 0.45;
      res.disp = 0.04;
      res.present = 1;
    },
    axialDefault,
    ACQ,
    { realization: 4, seed: 55 },
  );
  computeColorField(
    frame2,
    kasaiCfg,
    null,
    (_idx, _li, _si, res) => {
      res.v = 0.45;
      res.vPerp = 0;
      res.disp = 0.04;
      res.present = 1;
    },
    axialZeroPerp,
    ACQ,
    { realization: 4, seed: 55 },
  );
  assert.ok(
    Math.abs(meanVar(axialDefault) - meanVar(axialZeroPerp)) < 1e-6,
    'Pure axial laminar flow (v_perp = 0) must remain unchanged',
  );
});

test('Color Doppler Fase 4: real myocardial clutter flash, tissue-write breakthrough, blood isolation, and zero pinhole fills on myocardium', () => {
  const lines = 20;
  const samples = 36;
  const frame = makeBloodFrame(lines, samples, 16);
  // Split grid: top half (li < 10) = Blood (+0.38 m/s), bottom half (li >= 10) = Myocardium (vWall = -0.08 m/s)
  for (let li = 10; li < lines; li++) {
    for (let si = 0; si < samples; si++) {
      frame.tissue[li * samples + si] = Tissue.Myocardium;
    }
  }

  const baseCfg: ColorSettings = {
    ...DEFAULT_COLOR,
    boxThetaMinRad: -0.55,
    boxThetaMaxRad: 0.55,
    boxRMinCm: 3,
    boxRMaxCm: 13,
    scaleMps: 0.68,
    baselineShiftMps: 0,
    gainDb: 0,
    persistence: 0.5,
    smoothing: 0.45,
  };

  const countRegion = (field: ReturnType<typeof allocColorField>, li0: number, li1: number) => {
    let colored = 0;
    let total = 0;
    let sumVel = 0;
    let maxPow = 0;
    for (let li = li0; li < li1; li++) {
      for (let si = 10; si < 26; si++) {
        total++;
        const v = field.vel[li * samples + si]!;
        if (Number.isFinite(v)) {
          colored++;
          sumVel += v;
          maxPow = Math.max(maxPow, field.power[li * samples + si]!);
        }
      }
    }
    return {
      frac: total > 0 ? colored / total : 0,
      meanVel: colored > 0 ? sumVel / colored : NaN,
      maxPow,
    };
  };

  // 1. Default wall filter (−3 dB at 20 cm/s; labelled "6 cm/s" before it was calibrated): 8 cm/s wall motion stays
  //    clean (< 5% flash)
  const outWfDefault = allocColorField(lines * samples);
  computeColorField(
    frame,
    { ...baseCfg, wallFilterMps: DEFAULT_COLOR.wallFilterMps },
    null,
    (_idx, li, _si, res) => {
      if (li < 10) {
        res.v = 0.38;
        res.disp = 0.03;
        res.present = 1;
      } else {
        res.vWall = -0.08;
      }
    },
    outWfDefault,
    ACQ,
    { realization: 3, seed: 88 },
  );
  const myoWfDefault = countRegion(outWfDefault, 12, 18);
  const bloodWfDefault = countRegion(outWfDefault, 3, 9);
  assert.ok(
    myoWfDefault.frac < 0.05,
    `Expected < 5% myocardial clutter flash at the default wall filter, got ${(myoWfDefault.frac * 100).toFixed(1)}%`,
  );
  assert.ok(
    bloodWfDefault.frac > 0.85 && Math.abs(bloodWfDefault.meanVel - 0.38) < 0.04,
    `Expected clean blood fill near +0.38 m/s at the default wall filter, got frac=${bloodWfDefault.frac.toFixed(2)} v=${bloodWfDefault.meanVel.toFixed(3)}`,
  );

  // 2. The packet's lowest wall filter (the mean removal: 7.6 cm/s at 0.68 m/s): moving myocardium produces real wall
  //    flash with breakthrough power > 1.0 while adjacent blood velocity stays uncontaminated and rawVel/rawPower
  //    clutter history stays cleared
  const wfMin = colorWallFilterRangeMps(baseCfg.scaleMps).min;
  const outWf2 = allocColorField(lines * samples);
  computeColorField(
    frame,
    { ...baseCfg, wallFilterMps: wfMin },
    null,
    (_idx, li, _si, res) => {
      if (li < 10) {
        res.v = 0.38;
        res.disp = 0.03;
        res.present = 1;
      } else {
        res.vWall = -0.08;
      }
    },
    outWf2,
    ACQ,
    { realization: 3, seed: 88 },
  );
  const myoWf2 = countRegion(outWf2, 12, 18);
  const bloodBoundaryWf2 = countRegion(outWf2, 8, 10);
  // the control is the same wall filter over still myocardium (no flash): against the default filter's field the
  // difference would be the two filters' own bias on 0.38 m/s blood (+5 % at 1.9× the 20 cm/s cutoff, −2 % at 5× the
  // 7.6 cm/s one), not contamination by the flash
  const outWf2Still = allocColorField(lines * samples);
  computeColorField(
    frame,
    { ...baseCfg, wallFilterMps: wfMin },
    null,
    (_idx, li, _si, res) => {
      if (li < 10) {
        res.v = 0.38;
        res.disp = 0.03;
        res.present = 1;
      }
    },
    outWf2Still,
    ACQ,
    { realization: 3, seed: 88 },
  );
  const bloodBoundaryStill = countRegion(outWf2Still, 8, 10);
  assert.ok(
    myoWf2.frac >= 0.6,
    `Expected >= 60% myocardial clutter flash at the lowest WF, got ${(myoWf2.frac * 100).toFixed(1)}%`,
  );
  assert.ok(
    myoWf2.maxPow > 1.05,
    `Expected surviving myocardial flash to encode breakthrough power > 1.05, got ${myoWf2.maxPow.toFixed(3)}`,
  );
  assert.ok(
    Math.abs(bloodBoundaryWf2.meanVel - bloodBoundaryStill.meanVel) < 0.02,
    `Adjacent blood velocity must not be biased by myocardial flash: still=${bloodBoundaryStill.meanVel.toFixed(3)} flash=${bloodBoundaryWf2.meanVel.toFixed(3)}`,
  );
  // Clutter temporal history must be cleared so moving myocardium never leaves stale ghost trails
  let clutterHistCount = 0;
  for (let li = 12; li < 18; li++) {
    for (let si = 10; si < 26; si++) {
      if ((outWf2.rawPower?.[li * samples + si] ?? 0) > 0) clutterHistCount++;
    }
  }
  assert.equal(clutterHistCount, 0, 'Clutter samples must never persist into rawPower temporal history');

  // 3. Breakthrough power > 1.0 visibly paints over bright myocardium (lum = 190) in overlayColorField,
  //    while normal blood power = 1.0 stays suppressed on lum = 190
  const lut: ScanLut = {
    key: 'bt-test',
    lines: 2,
    samples: 2,
    idx: new Int32Array([0, 0]),
    inPix: new Int32Array([0, 1]),
    inIdx: new Int32Array([0, 0]),
    inWl: new Uint16Array([0, 0]),
    inWs: new Uint16Array([0, 0]),
    inDl: new Uint8Array([0, 0]),
    inDs: new Uint8Array([0, 0]),
    wl: new Float32Array([0, 0]),
    ws: new Float32Array([0, 0]),
    li: new Int16Array([0, 0]),
    si: new Int16Array([0, 0]),
    rCm: new Float32Array([6, 6]),
    theta: new Float32Array([0, 0]),
  };
  const rgbaFlash = new Uint8ClampedArray([190, 190, 190, 255, 190, 190, 190, 255]);
  overlayColorField(
    rgbaFlash,
    lut,
    new Float32Array([-0.25, -0.25, -0.25, -0.25]),
    new Float32Array(4),
    baseCfg,
    new Float32Array([1.85, 1.85, 1.85, 1.85]),
  );
  assert.ok(
    rgbaFlash[2]! > rgbaFlash[0]! + 35,
    `Expected breakthrough flash (power=1.85) to paint visible blue over bright myocardium (lum=190), got rgb=(${rgbaFlash[0]},${rgbaFlash[1]},${rgbaFlash[2]})`,
  );

  // 4. Spatial pinhole closure must never bleed blood color into an isolated Tissue.Myocardium sample
  const pinholeFrame = makeBloodFrame(16, 32, 16);
  const pIdx = 8 * 32 + 16;
  pinholeFrame.tissue[pIdx] = Tissue.Myocardium;
  const outPinhole = allocColorField(16 * 32);
  computeColorField(
    pinholeFrame,
    { ...baseCfg, wallFilterMps: DEFAULT_COLOR.wallFilterMps, smoothing: 0.8 },
    null,
    (idx, _li, _si, res) => {
      if (idx === pIdx) {
        res.vWall = 0; // stationary myocardium surrounded by 8 colored blood neighbors
      } else {
        res.v = 0.42;
        res.disp = 0.03;
        res.present = 1;
      }
    },
    outPinhole,
    ACQ,
    { realization: 1, seed: 19 },
  );
  assert.ok(
    Number.isNaN(outPinhole.vel[pIdx]),
    'D3/CPU pinhole closure must never fill a stationary Tissue.Myocardium sample surrounded by blood',
  );
});

test('LV radial wall velocity and valve leaflet snap velocity follow cardiac physiology and regional segAmp', () => {
  const normal = loadCaseById('normal-excellent-window');
  const { heart, tables, thorax } = buildCaseModels(normal, REST_PATIENT);
  const flow = buildFlowParams(normal, heart, tables, thorax);

  // Mid-ejection systole: LV volume is shrinking rapidly -> inward radial velocity on mid-LV anterior/septal wall
  const { ejectionStartS, ejectionEndS } = tables.timings;
  const sysPhase = (ejectionStartS + 0.45 * (ejectionEndS - ejectionStartS)) / tables.rrS;
  const hpSys = computeHeartPose(heart, cycleStateAt(tables, sysPhase, flow.flowPeaks));
  const midZ = hpSys.zAnn + 0.45 * (heart.lv.lengthCm - hpSys.zAnn);
  const midR = heart.lv.rMax * 0.95;

  const vMidNorm = sampleTissueVelocity(heart, tables, sysPhase, midR, 0, midZ);
  assert.ok(
    vMidNorm.vx < -0.015,
    `Expected inward systolic LV radial velocity (vx < -0.015 m/s) at mid-LV lateral wall, got vx=${vMidNorm.vx.toFixed(4)}`,
  );

  // Akinetic segment (segAmp = 0) zeros radial velocity cleanly while preserving basal longitudinal TDI behavior; the
  // heart declares the field as createHeartModel does (the tissue velocity, as the classifier, reads it only then)
  const segAmp = heart.segAmp.map(() => 0);
  const akineticHeart = { ...heart, segAmp, regionalMeanFrac: regionalMeanFraction(segAmp, heart.lv.shape) };
  const vMidAkinetic = sampleTissueVelocity(
    akineticHeart,
    tables,
    sysPhase,
    midR,
    0,
    midZ,
  );
  assert.ok(
    Math.abs(vMidAkinetic.vx) < 1e-6 && Math.abs(vMidAkinetic.vy) < 1e-6,
    `Akinetic LV segment (segAmp=0) must have zero radial velocity, got (${vMidAkinetic.vx}, ${vMidAkinetic.vy})`,
  );

  // Valve leaflet snap velocity peaks briefly around mitral opening/closing and is near zero in mid-systole
  let peakMvSnap = 0;
  for (let i = 0; i < 200; i++) {
    const p = i / 200;
    const hp = computeHeartPose(heart, cycleStateAt(tables, p, flow.flowPeaks));
    const vLeaf = sampleLeafletVelocity(
      heart,
      tables,
      hp,
      p,
      flow.mvCenter.x,
      flow.mvCenter.y,
      hp.zAnn + 0.5,
      flow.flowPeaks,
    );
    peakMvSnap = Math.max(peakMvSnap, Math.hypot(vLeaf.vx, vLeaf.vy, vLeaf.vz));
  }
  const vMidSysLeaf = sampleLeafletVelocity(
    heart,
    tables,
    hpSys,
    sysPhase,
    flow.mvCenter.x,
    flow.mvCenter.y,
    hpSys.zAnn + 0.5,
    flow.flowPeaks,
  );
  const midSysLeafSpeed = Math.hypot(vMidSysLeaf.vx, vMidSysLeaf.vy, vMidSysLeaf.vz);
  assert.ok(
    peakMvSnap >= 0.08 && peakMvSnap <= 0.25,
    `Expected mitral leaflet snap velocity in [0.08, 0.25] m/s during opening/closing, got ${peakMvSnap.toFixed(3)}`,
  );
  assert.ok(
    midSysLeafSpeed < 0.02,
    `Expected closed mitral leaflet in mid-systole to stay quiescent (< 0.02 m/s), got ${midSysLeafSpeed.toFixed(4)}`,
  );
});

test('sampleFlow anatomical chamber containment blocks cross-septal and cross-wall jet bleed', () => {
  const fs: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };

  // 1. Eccentric MR (mvp-primary-mr) aimed at -35 deg toward the IAS must stay in LA and never leak into RA, SVC, or PV
  const mrCase = loadCaseById('mvp-primary-mr');
  const mrModels = buildCaseModels(mrCase, REST_PATIENT);
  const mrFlow = buildFlowParams(mrCase, mrModels.heart, mrModels.tables, mrModels.thorax);
  const mrHp = computeHeartPose(mrModels.heart, cycleStateAt(mrModels.tables, 0.2, mrFlow.flowPeaks));
  // Point along the 3D anteromedial MR jet trajectory 1.5 cm into the atrial level
  const mrDir = mrJetInitialDirection(mrFlow.mrJetDirRad);
  const jx = mrFlow.mvCenter.x + mrDir.x * 1.5;
  const jy = mrFlow.mvCenter.y + mrDir.y * 1.5;
  const jz = mrHp.zAnn - 0.1 + mrDir.z * 1.5;
  sampleFlow(mrFlow, mrModels.tables, mrHp, 0.2, jx, jy, jz, fs, Structure.LaCavity);
  const laSpeed = Math.hypot(fs.vx, fs.vy, fs.vz);
  assert.ok(laSpeed > 1.5, `Expected strong MR jet in LaCavity, got ${laSpeed.toFixed(2)} m/s`);

  sampleFlow(mrFlow, mrModels.tables, mrHp, 0.2, jx, jy, jz, fs, Structure.Svc);
  assert.equal(fs.present, 0, 'MR jet must never bleed into SVC');

  sampleFlow(mrFlow, mrModels.tables, mrHp, 0.2, jx, jy, jz, fs, Structure.PulmonaryVein);
  assert.ok(
    fs.dispersion < 0.15,
    `MR jet must never bleed turbulent flow into PulmonaryVein, got disp=${fs.dispersion.toFixed(2)}`,
  );

  // 2. Severe AS post-stenotic jet must stay inside AorticRoot/AscendingAorta and never bleed into RV or PA
  const asCase = loadCaseById('aortic-stenosis-severe');
  const asModels = buildCaseModels(asCase, REST_PATIENT);
  const asFlow = buildFlowParams(asCase, asModels.heart, asModels.tables, asModels.thorax);
  const asHp = computeHeartPose(asModels.heart, cycleStateAt(asModels.tables, 0.18, asFlow.flowPeaks));
  const ax = asFlow.avAxis;
  const asX = asFlow.avCenter.x + ax.x * 1.2;
  const asY = asFlow.avCenter.y + ax.y * 1.2;
  const asZ = asFlow.avCenter.z + asHp.zAnn * 0.65 + ax.z * 1.2;
  sampleFlow(asFlow, asModels.tables, asHp, 0.18, asX, asY, asZ, fs, Structure.AorticRoot);
  assert.ok(
    Math.hypot(fs.vx, fs.vy, fs.vz) > 2.5 && fs.dispersion > 0.35,
    'Expected high-velocity turbulent AS jet in AorticRoot',
  );
  sampleFlow(asFlow, asModels.tables, asHp, 0.18, asX, asY, asZ, fs, Structure.RvCavity);
  assert.ok(
    fs.dispersion < 0.15,
    `AS jet must never bleed turbulent flow into RvCavity, got disp=${fs.dispersion.toFixed(2)}`,
  );
  sampleFlow(asFlow, asModels.tables, asHp, 0.18, asX, asY, asZ, fs, Structure.LaCavity);
  assert.ok(
    fs.dispersion < 0.15,
    `AS jet must never bleed turbulent flow into LaCavity, got disp=${fs.dispersion.toFixed(2)}`,
  );

  // 3. Severe TR (pulmonary-hypertension-rv) must never bleed into PulmonaryVein or LaCavity
  const phCase = loadCaseById('pulmonary-hypertension-rv');
  const phModels = buildCaseModels(phCase, REST_PATIENT);
  const phFlow = buildFlowParams(phCase, phModels.heart, phModels.tables, phModels.thorax);
  const phHp = computeHeartPose(phModels.heart, cycleStateAt(phModels.tables, 0.2, phFlow.flowPeaks));
  const tx = phFlow.tvCenter.x;
  const ty = phFlow.tvCenter.y;
  const tz = phFlow.tvCenter.z + phHp.tvZ - 1.5;
  sampleFlow(phFlow, phModels.tables, phHp, 0.2, tx, ty, tz, fs, Structure.RaCavity);
  assert.ok(Math.hypot(fs.vx, fs.vy, fs.vz) > 1.5, 'Expected strong TR jet inside RaCavity');
  sampleFlow(phFlow, phModels.tables, phHp, 0.2, tx, ty, tz, fs, Structure.PulmonaryVein);
  assert.ok(
    fs.dispersion < 0.15,
    `TR jet must never bleed into PulmonaryVein, got disp=${fs.dispersion.toFixed(2)}`,
  );
  sampleFlow(phFlow, phModels.tables, phHp, 0.2, tx, ty, tz, fs, Structure.LaCavity);
  assert.ok(
    fs.dispersion < 0.15,
    `TR jet must never bleed into LaCavity, got disp=${fs.dispersion.toFixed(2)}`,
  );
});

test('pulmonary trunk, RPA, and LPA carry continuous mass-conserving systolic flow', () => {
  const fs: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };
  const c = loadCaseById('normal-excellent');
  const models = buildCaseModels(c, REST_PATIENT);
  const flow = buildFlowParams(c, models.heart, models.tables, models.thorax);
  const A = heartAnchors(models.heart);
  const phase = 0.16;
  const hp = computeHeartPose(models.heart, cycleStateAt(models.tables, phase, flow.flowPeaks));
  const qpv = sampleTable(models.tables.pulmonaryFlowMlps, phase);
  assert.ok(qpv > 150, `Expected brisk peak pulmonary flow at phase=${phase}, got ${qpv}`);

  const mid = (p0: { x: number; y: number; z: number }, p1: { x: number; y: number; z: number }, z0Off = 0) => ({
    x: 0.5 * (p0.x + p1.x),
    y: 0.5 * (p0.y + p1.y),
    z: 0.5 * (p0.z + z0Off + p1.z),
  });
  const mpaPt = mid(A.paStj, A.paEnd, hp.pvZ);
  const rpaPt = mid(A.paEnd, A.rpaEnd);
  const lpaPt = mid(A.paEnd, A.lpaEnd);

  sampleFlow(flow, models.tables, hp, phase, mpaPt.x, mpaPt.y, mpaPt.z, fs, Structure.PulmonaryArtery);
  const vMpa = Math.hypot(fs.vx, fs.vy, fs.vz);
  assert.ok(fs.present === 1 && vMpa > 0.6 && vMpa < 1.8, `Expected physiological MPA velocity, got ${vMpa.toFixed(2)} m/s`);
  assert.ok(fs.dispersion < 0.15, `Normal MPA flow should remain laminar, got disp=${fs.dispersion.toFixed(2)}`);

  sampleFlow(flow, models.tables, hp, phase, rpaPt.x, rpaPt.y, rpaPt.z, fs, Structure.PulmonaryArtery);
  const vRpa = Math.hypot(fs.vx, fs.vy, fs.vz);
  // Direction check: velocity must align with paEnd -> rpaEnd
  const rpaDx = A.rpaEnd.x - A.paEnd.x, rpaDy = A.rpaEnd.y - A.paEnd.y, rpaDz = A.rpaEnd.z - A.paEnd.z;
  const rpaCos = (fs.vx * rpaDx + fs.vy * rpaDy + fs.vz * rpaDz) / (vRpa * Math.hypot(rpaDx, rpaDy, rpaDz));
  assert.ok(fs.present === 1 && vRpa > 0.5, `Expected brisk RPA flow beneath aortic arch, got ${vRpa.toFixed(2)} m/s`);
  assert.ok(rpaCos > 0.95, `Expected RPA velocity aligned with RPA lumen, got cos=${rpaCos.toFixed(3)}`);

  sampleFlow(flow, models.tables, hp, phase, lpaPt.x, lpaPt.y, lpaPt.z, fs, Structure.PulmonaryArtery);
  const vLpa = Math.hypot(fs.vx, fs.vy, fs.vz);
  assert.ok(fs.present === 1 && vLpa > 0.5, `Expected brisk LPA flow, got ${vLpa.toFixed(2)} m/s`);

  // Mass conservation across the bifurcation: Q_mpa = Q_rpa + Q_lpa (centerline scales by 4/3 identically)
  const qMpa = (vMpa / (4 / 3)) * Math.PI * A.paR * A.paR * 100;
  const qBranches = (vRpa / (4 / 3)) * Math.PI * A.rpaR * A.rpaR * 100 + (vLpa / (4 / 3)) * Math.PI * A.lpaR * A.lpaR * 100;
  assert.ok(
    Math.abs(qMpa - qpv) / qpv < 0.08,
    `Expected MPA flux (${qMpa.toFixed(1)} mL/s) to match Q_pv (${qpv.toFixed(1)} mL/s)`,
  );
  assert.ok(
    Math.abs(qBranches - qpv) / qpv < 0.08,
    `Expected RPA+LPA flux (${qBranches.toFixed(1)} mL/s) to conserve Q_pv (${qpv.toFixed(1)} mL/s)`,
  );
});

test('diastolic pulmonary regurgitation reflects PA pressure gradient and stays contained to RVOT/PA', () => {
  const fs: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };
  const normCase = loadCaseById('normal-excellent');
  const normModels = buildCaseModels(normCase, REST_PATIENT);
  const normFlow = buildFlowParams(normCase, normModels.heart, normModels.tables, normModels.thorax);

  const phCase = loadCaseById('pulmonary-hypertension-rv');
  const phModels = buildCaseModels(phCase, REST_PATIENT);
  const phFlow = buildFlowParams(phCase, phModels.heart, phModels.tables, phModels.thorax);
  const phA = heartAnchors(phModels.heart);

  // Early diastole (right after PV closure): PH should show brisk high-velocity PR in RVOT
  const phDia = (phModels.tables.timings.pulmonaryCloseS + 0.06) / phModels.tables.rrS;
  const phHp = computeHeartPose(phModels.heart, cycleStateAt(phModels.tables, phDia, phFlow.flowPeaks));
  const rx = phA.rvotM.x - phA.rvotB.x;
  const ry = phA.rvotM.y - phA.rvotB.y;
  const rz = phA.rvotM.z - phA.rvotB.z;
  const rLen = Math.hypot(rx, ry, rz);
  // 0.8 cm retrograde into distal RVOT beneath the pulmonary valve
  const prX = phA.rvotB.x + (rx / rLen) * 0.8;
  const prY = phA.rvotB.y + (ry / rLen) * 0.8;
  const prZ = phA.rvotB.z + phHp.pvZ + (rz / rLen) * 0.8;

  sampleFlow(phFlow, phModels.tables, phHp, phDia, prX, prY, prZ, fs, Structure.Rvot);
  const vPh = Math.hypot(fs.vx, fs.vy, fs.vz);
  assert.ok(
    fs.present === 1 && vPh > 2.2 && vPh < 3.2,
    `Expected brisk PH early-diastolic PR velocity (2.2–3.2 m/s), got ${vPh.toFixed(2)} m/s`,
  );
  assert.ok(fs.dispersion > 0.3, `Expected turbulent mosaic in PH PR jet, got disp=${fs.dispersion.toFixed(2)}`);

  // Retrograde PR jet must not bleed into AorticRoot or LaCavity
  sampleFlow(phFlow, phModels.tables, phHp, phDia, prX, prY, prZ, fs, Structure.AorticRoot);
  assert.equal(fs.present, 0, 'PR jet must never bleed into AorticRoot');

  // PA-side proximal convergence (0.45 cm into pulmonary root) in PH
  const pisaX = phA.rvotB.x - (rx / rLen) * 0.45;
  const pisaY = phA.rvotB.y - (ry / rLen) * 0.45;
  const pisaZ = phA.rvotB.z + phHp.pvZ - (rz / rLen) * 0.45;
  sampleFlow(phFlow, phModels.tables, phHp, phDia, pisaX, pisaY, pisaZ, fs, Structure.PulmonaryArtery);
  assert.ok(
    fs.present === 1 && Math.hypot(fs.vx, fs.vy, fs.vz) > 0.2,
    `Expected PA-side PR proximal convergence in PH, got ${Math.hypot(fs.vx, fs.vy, fs.vz).toFixed(2)} m/s`,
  );

  // Normal heart: trace PR is much shorter & lower velocity than PH (and absent 2.0 cm deep in RVOT)
  const normA = heartAnchors(normModels.heart);
  const normDia = (normModels.tables.timings.pulmonaryCloseS + 0.06) / normModels.tables.rrS;
  const normHp = computeHeartPose(normModels.heart, cycleStateAt(normModels.tables, normDia, normFlow.flowPeaks));
  const nrx = normA.rvotM.x - normA.rvotB.x, nry = normA.rvotM.y - normA.rvotB.y, nrz = normA.rvotM.z - normA.rvotB.z;
  const nrL = Math.hypot(nrx, nry, nrz);
  sampleFlow(
    normFlow,
    normModels.tables,
    normHp,
    normDia,
    normA.rvotB.x + (nrx / nrL) * 2.0,
    normA.rvotB.y + (nry / nrL) * 2.0,
    normA.rvotB.z + normHp.pvZ + (nrz / nrL) * 2.0,
    fs,
    Structure.Rvot,
  );
  assert.equal(fs.present, 0, 'Normal physiological trace PR must not reach 2.0 cm into RVOT');
});

test('IVC and hepatic vein exhibit triphasic S/D/Ar return, TR systolic reversal, tamponade D blunting, and inspiratory augmentation', () => {
  const fs: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };

  // 1. Normal heart: antegrade S > D in IVC & hepatic vein, negative Ar, inspiratory augmentation
  const normCase = loadCaseById('normal-excellent');
  const normModels = buildCaseModels(normCase, REST_PATIENT);
  const normFlow = buildFlowParams(normCase, normModels.heart, normModels.tables, normModels.thorax);
  const normA = heartAnchors(normModels.heart);
  const tmN = normModels.tables.timings;

  const phSysN = (tmN.pulmonaryOpenS + 0.5 * (tmN.pulmonaryCloseS - tmN.pulmonaryOpenS)) / normModels.tables.rrS;
  const phDiaN = (tmN.tricuspidOpenS + tmN.eAccelS) / normModels.tables.rrS;
  const phArN = (tmN.aStartS + 0.5 * (tmN.aEndS - tmN.aStartS)) / normModels.tables.rrS;

  const hpSysN = computeHeartPose(normModels.heart, cycleStateAt(normModels.tables, phSysN, normFlow.flowPeaks));
  const hpDiaN = computeHeartPose(normModels.heart, cycleStateAt(normModels.tables, phDiaN, normFlow.flowPeaks));
  const hpArN = computeHeartPose(normModels.heart, cycleStateAt(normModels.tables, phArN, normFlow.flowPeaks));

  const vSysN = cavalVenousVelocityMps(normFlow, normModels.tables, hpSysN, phSysN);
  const vDiaN = cavalVenousVelocityMps(normFlow, normModels.tables, hpDiaN, phDiaN);
  const vArN = cavalVenousVelocityMps(normFlow, normModels.tables, hpArN, phArN);
  assert.ok(vSysN > 0.35 && vDiaN > 0.2 && vSysN > vDiaN, `Expected normal S > D > 0, got S=${vSysN.toFixed(2)}, D=${vDiaN.toFixed(2)}`);
  assert.ok(vArN < -0.1, `Expected retrograde Ar reversal during atrial kick, got ${vArN.toFixed(2)} m/s`);

  // Inspiratory augmentation with ivcCollapse > 0
  const vSysInsp = cavalVenousVelocityMps(normFlow, normModels.tables, { ...hpSysN, ivcCollapse: 0.5 }, phSysN);
  assert.ok(vSysInsp > vSysN * 1.15, `Expected inspiratory augmentation in IVC return (${vSysInsp.toFixed(2)} vs ${vSysN.toFixed(2)})`);

  // Spatial verification inside IVC and hepatic vein lumens (direction points toward RA / ivcA)
  const ivcMid = { x: 0.5 * (normA.ivcA.x + normA.ivcB.x), y: 0.5 * (normA.ivcA.y + normA.ivcB.y), z: 0.5 * (normA.ivcA.z + normA.ivcB.z) };
  const hvMid = { x: 0.5 * (normA.hvA.x + normA.hvB.x), y: 0.5 * (normA.hvA.y + normA.hvB.y), z: 0.5 * (normA.hvA.z + normA.hvB.z) };
  sampleFlow(normFlow, normModels.tables, hpSysN, phSysN, ivcMid.x, ivcMid.y, ivcMid.z, fs, Structure.Ivc);
  const ivcDx = normA.ivcA.x - normA.ivcB.x, ivcDy = normA.ivcA.y - normA.ivcB.y, ivcDz = normA.ivcA.z - normA.ivcB.z;
  const ivcDot = fs.vx * ivcDx + fs.vy * ivcDy + fs.vz * ivcDz;
  assert.ok(fs.present === 1 && ivcDot > 0 && Math.hypot(fs.vx, fs.vy, fs.vz) > 0.35, 'Expected antegrade systolic IVC flow toward RA');

  sampleFlow(normFlow, normModels.tables, hpSysN, phSysN, hvMid.x, hvMid.y, hvMid.z, fs, Structure.HepaticVein);
  const hvDx = normA.hvA.x - normA.hvB.x, hvDy = normA.hvA.y - normA.hvB.y, hvDz = normA.hvA.z - normA.hvB.z;
  const hvDot = fs.vx * hvDx + fs.vy * hvDy + fs.vz * hvDz;
  assert.ok(fs.present === 1 && hvDot > 0 && Math.hypot(fs.vx, fs.vy, fs.vz) > 0.38, 'Expected antegrade systolic hepatic vein flow toward IVC');

  // 2. Severe TR (pulmonary-hypertension-rv): holosystolic flow reversal in hepatic vein & IVC + brisk diastolic D
  const phCase = loadCaseById('pulmonary-hypertension-rv');
  const phModels = buildCaseModels(phCase, REST_PATIENT);
  const phFlow = buildFlowParams(phCase, phModels.heart, phModels.tables, phModels.thorax);
  const phA = heartAnchors(phModels.heart);
  const tmPh = phModels.tables.timings;
  const phSysPh = (tmPh.pulmonaryOpenS + 0.5 * (tmPh.pulmonaryCloseS - tmPh.pulmonaryOpenS)) / phModels.tables.rrS;
  const phDiaPh = (tmPh.tricuspidOpenS + tmPh.eAccelS) / phModels.tables.rrS;
  const hpSysPh = computeHeartPose(phModels.heart, cycleStateAt(phModels.tables, phSysPh, phFlow.flowPeaks));
  const hpDiaPh = computeHeartPose(phModels.heart, cycleStateAt(phModels.tables, phDiaPh, phFlow.flowPeaks));

  const vSysPh = cavalVenousVelocityMps(phFlow, phModels.tables, hpSysPh, phSysPh);
  const vDiaPh = cavalVenousVelocityMps(phFlow, phModels.tables, hpDiaPh, phDiaPh);
  assert.ok(vSysPh < -0.25, `Expected severe-TR systolic reversal (< -0.25 m/s), got ${vSysPh.toFixed(2)} m/s`);
  assert.ok(vDiaPh > 0.3, `Expected brisk diastolic forward D wave in severe TR, got ${vDiaPh.toFixed(2)} m/s`);

  const phHvMid = { x: 0.5 * (phA.hvA.x + phA.hvB.x), y: 0.5 * (phA.hvA.y + phA.hvB.y), z: 0.5 * (phA.hvA.z + phA.hvB.z) };
  sampleFlow(phFlow, phModels.tables, hpSysPh, phSysPh, phHvMid.x, phHvMid.y, phHvMid.z, fs, Structure.HepaticVein);
  const phHvDot = fs.vx * (phA.hvA.x - phA.hvB.x) + fs.vy * (phA.hvA.y - phA.hvB.y) + fs.vz * (phA.hvA.z - phA.hvB.z);
  assert.ok(fs.present === 1 && phHvDot < 0, 'Expected retrograde systolic flow in hepatic vein during severe TR');

  // 3. Cardiac tamponade: dominant systolic S with markedly blunted diastolic D
  const tampCase = loadCaseById('pericardial-effusion-tamponade');
  const tampModels = buildCaseModels(tampCase, REST_PATIENT);
  const tampFlow = buildFlowParams(tampCase, tampModels.heart, tampModels.tables, tampModels.thorax);
  const tmT = tampModels.tables.timings;
  const phSysT = (tmT.pulmonaryOpenS + 0.5 * (tmT.pulmonaryCloseS - tmT.pulmonaryOpenS)) / tampModels.tables.rrS;
  const phDiaT = (tmT.tricuspidOpenS + tmT.eAccelS) / tampModels.tables.rrS;
  const vSysT = cavalVenousVelocityMps(tampFlow, tampModels.tables, computeHeartPose(tampModels.heart, cycleStateAt(tampModels.tables, phSysT, tampFlow.flowPeaks)), phSysT);
  const vDiaT = cavalVenousVelocityMps(tampFlow, tampModels.tables, computeHeartPose(tampModels.heart, cycleStateAt(tampModels.tables, phDiaT, tampFlow.flowPeaks)), phDiaT);
  assert.ok(vSysT > 0.3 && vDiaT < 0.16 && vSysT > 2 * vDiaT, `Expected tamponade systolic-dominant venous return (S=${vSysT.toFixed(2)}, D=${vDiaT.toFixed(2)})`);

  // 4. AF: no atrial Ar reversal
  const afCase = loadCaseById('af-diastolic');
  const afModels = buildCaseModels(afCase, REST_PATIENT);
  const afFlow = buildFlowParams(afCase, afModels.heart, afModels.tables, afModels.thorax);
  const hpLateAf = computeHeartPose(afModels.heart, cycleStateAt(afModels.tables, 0.92, afFlow.flowPeaks));
  const vLateAf = cavalVenousVelocityMps(afFlow, afModels.tables, hpLateAf, 0.92);
  assert.ok(vLateAf >= -0.01, `AF must have no atrial-kick Ar reversal, got ${vLateAf.toFixed(3)} m/s`);
});

test('SVC and coronary sinus carry physiological antegrade venous return into the RA without leaking across chambers', () => {
  const fs: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };
  const c = loadCaseById('normal-excellent');
  const models = buildCaseModels(c, REST_PATIENT);
  const flow = buildFlowParams(c, models.heart, models.tables, models.thorax);
  const A = heartAnchors(models.heart);
  const tm = models.tables.timings;

  const phSys = (tm.pulmonaryOpenS + 0.5 * (tm.pulmonaryCloseS - tm.pulmonaryOpenS)) / models.tables.rrS;
  const phDia = (tm.mitralOpenS + tm.eAccelS) / models.tables.rrS;
  const hpSys = computeHeartPose(models.heart, cycleStateAt(models.tables, phSys, flow.flowPeaks));
  const hpDia = computeHeartPose(models.heart, cycleStateAt(models.tables, phDia, flow.flowPeaks));

  // 1. SVC antegrade systolic return toward RA roof (svcM -> svcA)
  const svcMid = { x: 0.5 * (A.svcM.x + A.svcA.x), y: 0.5 * (A.svcM.y + A.svcA.y), z: 0.5 * (A.svcM.z + A.svcA.z) };
  sampleFlow(flow, models.tables, hpSys, phSys, svcMid.x, svcMid.y, svcMid.z, fs, Structure.Svc);
  const svcDot = fs.vx * (A.svcA.x - A.svcM.x) + fs.vy * (A.svcA.y - A.svcM.y) + fs.vz * (A.svcA.z - A.svcM.z);
  assert.ok(
    fs.present === 1 && svcDot > 0 && Math.hypot(fs.vx, fs.vy, fs.vz) > 0.35,
    `Expected antegrade systolic SVC flow toward RA roof, got v=${Math.hypot(fs.vx, fs.vy, fs.vz).toFixed(2)} m/s`,
  );

  // SVC flow must never leak into LaCavity or PulmonaryVein
  sampleFlow(flow, models.tables, hpSys, phSys, svcMid.x, svcMid.y, svcMid.z, fs, Structure.LaCavity);
  assert.ok(Math.hypot(fs.vx, fs.vy, fs.vz) < 0.05, 'SVC flow must not bleed into LaCavity');

  // 2. Coronary sinus biphasic systolic + diastolic medial drainage toward RA ostium
  const vCsSys = coronarySinusVelocityMps(models.tables, phSys);
  const vCsDia = coronarySinusVelocityMps(models.tables, phDia);
  assert.ok(vCsSys > 0.25 && vCsDia > 0.15, `Expected biphasic CS drainage (S=${vCsSys.toFixed(2)}, D=${vCsDia.toFixed(2)})`);

  // Sample at the anatomical coronary sinus ostium entering the inferomedial RA (anteromedial: vx < 0, vy > 0)
  const csX = A.iasX - 1.2;
  const csY = -1.2;
  const csZ = hpSys.zAnn - 0.4;
  sampleFlow(flow, models.tables, hpSys, phSys, csX, csY, csZ, fs, Structure.CoronarySinus);
  assert.ok(
    fs.present === 1 && fs.vx < -0.15 && fs.vy > 0.1 && fs.dispersion < 0.1,
    `Expected laminar anteromedial coronary sinus flow at RA ostium, got vx=${fs.vx.toFixed(2)}, vy=${fs.vy.toFixed(2)} m/s`,
  );

  // 3. Severe TR (pulmonary-hypertension-rv) reverses SVC systolic flow as well
  const phCase = loadCaseById('pulmonary-hypertension-rv');
  const phModels = buildCaseModels(phCase, REST_PATIENT);
  const phFlow = buildFlowParams(phCase, phModels.heart, phModels.tables, phModels.thorax);
  const phA = heartAnchors(phModels.heart);
  const phSysPh =
    (phModels.tables.timings.pulmonaryOpenS +
      0.5 * (phModels.tables.timings.pulmonaryCloseS - phModels.tables.timings.pulmonaryOpenS)) /
    phModels.tables.rrS;
  const hpSysPh = computeHeartPose(phModels.heart, cycleStateAt(phModels.tables, phSysPh, phFlow.flowPeaks));
  const phSvcMid = { x: 0.5 * (phA.svcM.x + phA.svcA.x), y: 0.5 * (phA.svcM.y + phA.svcA.y), z: 0.5 * (phA.svcM.z + phA.svcA.z) };
  sampleFlow(phFlow, phModels.tables, hpSysPh, phSysPh, phSvcMid.x, phSvcMid.y, phSvcMid.z, fs, Structure.Svc);
  const phSvcDot = fs.vx * (phA.svcA.x - phA.svcM.x) + fs.vy * (phA.svcA.y - phA.svcM.y) + fs.vz * (phA.svcA.z - phA.svcM.z);
  assert.ok(fs.present === 1 && phSvcDot < 0, 'Expected systolic SVC flow reversal in severe TR');
});

test('TR PISA obeys 3D radial hemispheric continuity Q = 2·π·r²·v and jet curves gently toward IAS', () => {
  const fs: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };
  const phCase = loadCaseById('pulmonary-hypertension-rv');
  const models = buildCaseModels(phCase, REST_PATIENT);
  const flow = buildFlowParams(phCase, models.heart, models.tables, models.thorax);
  const phase = 0.22;
  const hp = computeHeartPose(models.heart, cycleStateAt(models.tables, phase, flow.flowPeaks));
  const ox = flow.tvCenter.x;
  const oy = flow.tvCenter.y;
  const oz = flow.tvCenter.z + hp.tvZ + 0.3;
  const vmax = sampleTable(models.tables.trVelocityMps, phase);
  const qTr = vmax * 100 * flow.trEroCm2!;

  // 1. Off-axis RV-side PISA point (d = 0.70 cm, 45 deg off-axis in x-z): must carry radial vx toward orifice!
  const d = 0.7;
  const off = d / Math.SQRT2;
  sampleFlow(flow, models.tables, hp, phase, ox + off, oy, oz + off, fs, Structure.RvCavity);
  const vMag = Math.hypot(fs.vx, fs.vy, fs.vz);
  const expectedV = qTr / (2 * Math.PI * d * d * 100);
  assert.ok(
    fs.present === 1 && fs.vx < -0.15 && fs.vz < -0.15,
    `Expected 3D radial TR PISA pointing toward orifice (vx<0, vz<0), got (${fs.vx.toFixed(2)}, ${fs.vy.toFixed(2)}, ${fs.vz.toFixed(2)})`,
  );
  assert.ok(
    Math.abs(vMag - expectedV) / expectedV < 0.12,
    `Expected TR PISA speed (${vMag.toFixed(2)} m/s) to conserve Q/(2πd²) (${expectedV.toFixed(2)} m/s) within 12%`,
  );
  assert.ok(fs.dispersion <= 0.1, `TR PISA convergence must stay laminar, got disp=${fs.dispersion.toFixed(2)}`);

  // 2. Orifice vena contracta keeps exact full vmax along -z; distal RA jet curves gently medially (+x toward IAS)
  sampleFlow(flow, models.tables, hp, phase, ox, oy, oz - 0.2, fs, Structure.RaCavity);
  assert.ok(
    Math.abs(-fs.vz - vmax) / vmax < 0.04,
    `Expected exact TR vena contracta velocity (${(-fs.vz).toFixed(2)} vs ${vmax.toFixed(2)} m/s)`,
  );
  sampleFlow(flow, models.tables, hp, phase, ox + 0.25, oy, oz - 2.2, fs, Structure.RaCavity);
  assert.ok(
    fs.present === 1 && fs.vx > 0.15 && fs.vz < -1.0 && fs.dispersion > 0.35,
    `Expected distal TR jet curving gently toward IAS (+x, -z) with turbulence, got (${fs.vx.toFixed(2)}, ${fs.vz.toFixed(2)}, d=${fs.dispersion.toFixed(2)})`,
  );
});

test('eccentric MR steers in 3D (anteromedial prolapse, posterolateral SAM) with Coanda roof curvature while central MR stays symmetric', () => {
  const fs: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };

  // 1. Posterior leaflet prolapse (mvp-primary-mr): anteromedial 3D jet (+y anterior toward aortic root, -x medial)
  const mvpCase = loadCaseById('mvp-primary-mr');
  const mvpModels = buildCaseModels(mvpCase, REST_PATIENT);
  const mvpFlow = buildFlowParams(mvpCase, mvpModels.heart, mvpModels.tables, mvpModels.thorax);
  const mvpHp = computeHeartPose(mvpModels.heart, cycleStateAt(mvpModels.tables, 0.2, mvpFlow.flowPeaks));
  const mvpDir = mrJetInitialDirection(mvpFlow.mrJetDirRad);
  assert.ok(mvpDir.x < -0.3 && mvpDir.y > 0.3 && mvpDir.z < -0.75, 'MVP MR should steer anteromedially (+y, -x, -z)');

  // Sample in anterior LA (PLAX anteromedial Coanda zone, y > mvCenter.y, z < zAnn)
  const pX = mvpFlow.mvCenter.x + mvpDir.x * 1.4;
  const pY = mvpFlow.mvCenter.y + mvpDir.y * 1.4;
  const pZ = mvpHp.zAnn - 0.1 + mvpDir.z * 1.4;
  sampleFlow(mvpFlow, mvpModels.tables, mvpHp, 0.2, pX, pY, pZ, fs, Structure.LaCavity);
  assert.ok(
    fs.present === 1 && fs.vy > 0.8 && fs.vx < -0.8 && fs.dispersion > 0.35,
    `Expected strong anteromedial turbulent MVP MR jet in anterior LA, got (${fs.vx.toFixed(2)}, ${fs.vy.toFixed(2)}, ${fs.vz.toFixed(2)})`,
  );

  // Coanda wall-hugging curvature distally (along = 3.2 cm): turns upward along LA dome (|vx/vz| decreases vs orifice)
  const bend = 0.42 * 0.9 * (3.2 - 1.0);
  const cX = mvpFlow.mvCenter.x + mvpDir.x * (3.2 - bend);
  const cY = mvpFlow.mvCenter.y + mvpDir.y * (3.2 - bend);
  const cZ = mvpHp.zAnn - 0.1 + mvpDir.z * 3.2 - 0.14 * bend;
  sampleFlow(mvpFlow, mvpModels.tables, mvpHp, 0.2, cX, cY, cZ, fs, Structure.LaCavity);
  assert.ok(
    fs.present === 1 && Math.hypot(fs.vx, fs.vy, fs.vz) > 1.5 && Math.abs(fs.vy / fs.vz) < Math.abs(mvpDir.y / mvpDir.z),
    `Expected Coanda wall-hugging curvature turning toward LA roof distally, got (${fs.vx.toFixed(2)}, ${fs.vy.toFixed(2)}, ${fs.vz.toFixed(2)})`,
  );

  // 2. SAM-related MR (hocm-sam): posterolateral jet (-y posterior toward posterior LA wall, +x lateral)
  const hocmCase = loadCaseById('hocm-sam');
  const hocmModels = buildCaseModels(hocmCase, REST_PATIENT);
  const hocmFlow = buildFlowParams(hocmCase, hocmModels.heart, hocmModels.tables, hocmModels.thorax);
  const hocmHp = computeHeartPose(hocmModels.heart, cycleStateAt(hocmModels.tables, 0.34, hocmFlow.flowPeaks));
  const samDir = mrJetInitialDirection(hocmFlow.mrJetDirRad);
  assert.ok(samDir.x > 0.3 && samDir.y < -0.3, 'SAM MR should steer posterolaterally (+x, -y, -z)');
  sampleFlow(
    hocmFlow,
    hocmModels.tables,
    hocmHp,
    0.34,
    hocmFlow.mvCenter.x + samDir.x * 1.4,
    hocmFlow.mvCenter.y + samDir.y * 1.4,
    hocmHp.zAnn - 0.1 + samDir.z * 1.4,
    fs,
    Structure.LaCavity,
  );
  assert.ok(
    fs.present === 1 && fs.vy < -0.8 && fs.vx > 0.6,
    `Expected posterolateral SAM MR jet in posterior LA, got (${fs.vx.toFixed(2)}, ${fs.vy.toFixed(2)}, ${fs.vz.toFixed(2)})`,
  );

  // 3. Functional central MR (hfref-severe-mr): symmetric central jet with zero lateral/AP bias
  const hfCase = loadCaseById('hfref-severe-mr');
  const hfModels = buildCaseModels(hfCase, REST_PATIENT);
  const hfFlow = buildFlowParams(hfCase, hfModels.heart, hfModels.tables, hfModels.thorax);
  const hfHp = computeHeartPose(hfModels.heart, cycleStateAt(hfModels.tables, 0.25, hfFlow.flowPeaks));
  sampleFlow(hfFlow, hfModels.tables, hfHp, 0.25, hfFlow.mvCenter.x, hfFlow.mvCenter.y, hfHp.zAnn - 1.6, fs, Structure.LaCavity);
  assert.ok(
    fs.present === 1 && Math.abs(fs.vx) < 0.05 && Math.abs(fs.vy) < 0.05 && fs.vz < -2.0,
    `Expected symmetric central functional MR jet along -z, got (${fs.vx.toFixed(2)}, ${fs.vy.toFixed(2)}, ${fs.vz.toFixed(2)})`,
  );
});

test('LV diastolic apical vortex redirects E-deceleration flow basally along the anteroseptal LVOT without perturbing mitral tips', () => {
  const fs: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };
  const c = loadCaseById('normal-excellent');
  const models = buildCaseModels(c, REST_PATIENT);
  const flow = buildFlowParams(c, models.heart, models.tables, models.thorax);
  const tm = models.tables.timings;
  // Mid-to-late E deceleration phase
  const phDecel = (tm.mitralOpenS + tm.eAccelS + 0.45 * tm.eDecelS) / models.tables.rrS;
  const hp = computeHeartPose(models.heart, cycleStateAt(models.tables, phDecel, flow.flowPeaks));

  // 1. Anteroseptal/LVOT side (toward avCenter): laminar basal return (-z) toward the closed aortic valve
  const ox = flow.avCenter.x - flow.mvCenter.x;
  const oy = flow.avCenter.y - flow.mvCenter.y;
  const oLen = Math.hypot(ox, oy);
  const retX = flow.mvCenter.x + (ox / oLen) * 1.6;
  const retY = flow.mvCenter.y + (oy / oLen) * 1.6;
  const retZ = hp.zAnn + 0.35 * hp.lengthNow;
  sampleFlow(flow, models.tables, hp, phDecel, retX, retY, retZ, fs, Structure.Lvot);
  assert.ok(
    fs.present === 1 && fs.vz < -0.18 && fs.dispersion < 0.12,
    `Expected laminar basal vortex return in LVOT during E deceleration, got vz=${fs.vz.toFixed(2)} m/s, d=${fs.dispersion.toFixed(2)}`,
  );

  // 2. Mitral leaflet tips centerline (sOut = 0, zr = 1.2 cm): pure apical inflow (+z), zero lateral vortex leak
  sampleFlow(flow, models.tables, hp, phDecel, flow.mvCenter.x, flow.mvCenter.y, hp.zAnn + 1.2, fs, Structure.LvCavity);
  assert.ok(
    fs.present === 1 && fs.vz > 0.25 && Math.abs(fs.vx) < 1e-4 && Math.abs(fs.vy) < 1e-4,
    `Mitral leaflet tips must remain pure axial inflow without vortex contamination, got (${fs.vx.toFixed(4)}, ${fs.vy.toFixed(4)}, ${fs.vz.toFixed(2)})`,
  );

  // 3. Vortex does not leak into LaCavity or AorticRoot
  sampleFlow(flow, models.tables, hp, phDecel, retX, retY, retZ, fs, Structure.AorticRoot);
  assert.equal(fs.present, 0, 'LV diastolic vortex must never leak into AorticRoot');
});

test('severe aortic stenosis shows laminar distal-LVOT convergence and post-stenotic sinus turbulent plume while preserving PW LVOT plane', () => {
  const fs: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };
  const asCase = loadCaseById('aortic-stenosis-severe');
  const models = buildCaseModels(asCase, REST_PATIENT);
  const flow = buildFlowParams(asCase, models.heart, models.tables, models.thorax);
  const phase = 0.2;
  const hp = computeHeartPose(models.heart, cycleStateAt(models.tables, phase, flow.flowPeaks));
  const ax = flow.avAxis;
  const cz = flow.avCenter.z + hp.zAnn * 0.65;
  // Build an orthonormal radial vector perpendicular to avAxis
  const pxRaw = -ax.y, pyRaw = ax.x;
  const pLen = Math.hypot(pxRaw, pyRaw);
  const rx = pxRaw / pLen, ry = pyRaw / pLen;

  // 1. PW LVOT plane (tAv = -0.5 cm, rho = 0): exact continuity speed and pure axial alignment
  const qao = sampleTable(models.tables.aorticFlowMlps, phase);
  const expectedLvotV = qao / flow.lvotAreaCm2 / 100;
  sampleFlow(
    flow,
    models.tables,
    hp,
    phase,
    flow.avCenter.x + ax.x * -0.5,
    flow.avCenter.y + ax.y * -0.5,
    cz + ax.z * -0.5,
    fs,
    Structure.Lvot,
  );
  const vLvot = Math.hypot(fs.vx, fs.vy, fs.vz);
  assert.ok(
    Math.abs(vLvot - expectedLvotV) / expectedLvotV < 0.03,
    `PW LVOT plane speed (${vLvot.toFixed(2)} m/s) must match exact continuity (${expectedLvotV.toFixed(2)} m/s)`,
  );

  // 2. Distal LVOT off-axis proximal convergence (tAv = -0.10 cm, rho = 0.55 cm): bends inward toward avAxis
  sampleFlow(
    flow,
    models.tables,
    hp,
    phase,
    flow.avCenter.x + ax.x * -0.1 + rx * 0.55,
    flow.avCenter.y + ax.y * -0.1 + ry * 0.55,
    cz + ax.z * -0.1,
    fs,
    Structure.Lvot,
  );
  const radialDot = fs.vx * rx + fs.vy * ry;
  assert.ok(
    fs.present === 1 && radialDot < -0.15,
    `Expected inward proximal flow convergence beneath stenotic AV (radialDot < -0.15), got ${radialDot.toFixed(2)} m/s`,
  );

  // 3. Post-stenotic outer aortic sinus (tAv = 0.9 cm, rho = 1.15 cm): entrained turbulent sinus plume
  sampleFlow(
    flow,
    models.tables,
    hp,
    phase,
    flow.avCenter.x + ax.x * 0.9 + rx * 1.15,
    flow.avCenter.y + ax.y * 0.9 + ry * 1.15,
    cz + ax.z * 0.9,
    fs,
    Structure.AorticRoot,
  );
  assert.ok(
    fs.present === 1 && Math.hypot(fs.vx, fs.vy, fs.vz) > 0.35 && fs.dispersion > 0.45,
    `Expected post-stenotic turbulent sinus entrainment plume in AorticRoot, got v=${Math.hypot(fs.vx, fs.vy, fs.vz).toFixed(2)}, d=${fs.dispersion.toFixed(2)}`,
  );
});

test('HOCM SAM stays laminar in early systole and erupts into subaortic LVOT turbulence in late systole (distinct from valvular AS)', () => {
  const fs: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };
  const hocmCase = loadCaseById('hocm-sam');
  const models = buildCaseModels(hocmCase, REST_PATIENT);
  const flow = buildFlowParams(hocmCase, models.heart, models.tables, models.thorax);
  const ax = flow.avAxis;
  const tm = models.tables.timings;

  // Early systole (u = 0.15, before SAM-septal contact): subaortic LVOT at tAv = -0.35 cm is laminar
  const phEarly = (tm.ejectionStartS + 0.15 * (tm.ejectionEndS - tm.ejectionStartS)) / models.tables.rrS;
  const hpEarly = computeHeartPose(models.heart, cycleStateAt(models.tables, phEarly, flow.flowPeaks));
  const czEarly = flow.avCenter.z + hpEarly.zAnn * 0.65;
  sampleFlow(
    flow,
    models.tables,
    hpEarly,
    phEarly,
    flow.avCenter.x + ax.x * -0.35,
    flow.avCenter.y + ax.y * -0.35,
    czEarly + ax.z * -0.35,
    fs,
    Structure.Lvot,
  );
  assert.ok(
    fs.present === 1 && fs.dispersion < 0.18,
    `Early-systolic HOCM LVOT before SAM contact should stay laminar, got disp=${fs.dispersion.toFixed(2)}`,
  );

  // Late systole (u = 0.80, peak SAM-septal contact): high-velocity subaortic turbulent mosaic BELOW the AV (tAv = -0.35 cm)
  const phLate = (tm.ejectionStartS + 0.8 * (tm.ejectionEndS - tm.ejectionStartS)) / models.tables.rrS;
  const hpLate = computeHeartPose(models.heart, cycleStateAt(models.tables, phLate, flow.flowPeaks));
  const czLate = flow.avCenter.z + hpLate.zAnn * 0.65;
  sampleFlow(
    flow,
    models.tables,
    hpLate,
    phLate,
    flow.avCenter.x + ax.x * -0.35,
    flow.avCenter.y + ax.y * -0.35,
    czLate + ax.z * -0.35,
    fs,
    Structure.Lvot,
  );
  const vLate = Math.hypot(fs.vx, fs.vy, fs.vz);
  assert.ok(
    fs.present === 1 && vLate > 2.5 && fs.dispersion > 0.65,
    `Late-systolic HOCM subaortic LVOT must show blistering turbulent mosaic (v>2.5, d>0.65), got v=${vLate.toFixed(2)}, d=${fs.dispersion.toFixed(2)}`,
  );

  // Valvular AS at the exact same subaortic level (tAv = -0.35 cm) stays laminar (turbulence begins at the valve tAv > 0)
  const asCase = loadCaseById('aortic-stenosis-severe');
  const asModels = buildCaseModels(asCase, REST_PATIENT);
  const asFlow = buildFlowParams(asCase, asModels.heart, asModels.tables, asModels.thorax);
  const asHp = computeHeartPose(asModels.heart, cycleStateAt(asModels.tables, 0.2, asFlow.flowPeaks));
  const asCz = asFlow.avCenter.z + asHp.zAnn * 0.65;
  sampleFlow(
    asFlow,
    asModels.tables,
    asHp,
    0.2,
    asFlow.avCenter.x + asFlow.avAxis.x * -0.35,
    asFlow.avCenter.y + asFlow.avAxis.y * -0.35,
    asCz + asFlow.avAxis.z * -0.35,
    fs,
    Structure.Lvot,
  );
  assert.ok(
    fs.dispersion < 0.25,
    `Valvular AS subaortic LVOT (-0.35 cm) must stay laminar vs HOCM SAM mosaic, got disp=${fs.dispersion.toFixed(2)}`,
  );
});
