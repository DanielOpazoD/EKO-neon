import test from 'node:test';
import assert from 'node:assert/strict';
import {
  allocColorField,
  computeColorField,
  DEFAULT_COLOR,
  DOPPLER_SHADOW_TRANSMISSION,
  wallFilterResponse,
  type ColorAcquisition,
  type ColorSettings,
} from './colorDoppler';
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
import { computeHeartPose } from '@/simulator/anatomy/heartModel';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';

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
});
