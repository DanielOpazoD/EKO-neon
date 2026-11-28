import test from 'node:test';
import assert from 'node:assert/strict';
import { loadCaseById } from '@/cases';
import { buildCaseModels, REST_PATIENT } from '@/simulator/anatomy/caseModels';
import { computeHeartPose } from '@/simulator/anatomy/heartModel';
import { Structure } from '@/simulator/anatomy/tissue';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import { baseInput } from '@/simulator/core/baseInput';
import type { CineBuffer } from '@/simulator/core/cineBuffer';
import { ColorFieldDriver, type ColorUpdateContext } from '@/simulator/core/colorFieldDriver';
import { SimulatorCore } from '@/simulator/core/simulatorCore';
import {
  buildFlowParams,
  sampleFlow,
  type FlowFieldParams,
  type FlowSample,
} from '@/simulator/doppler/flow-primitives/flowField';
import {
  lvVortexState,
  sampleLvDiastolicVortex,
} from '@/simulator/doppler/flow-primitives/jetProfiles';
import { computeDopplerBoxBounds } from '@/simulator/renderer/gpu/deferredDoppler';
import { acquisitionFrameRate, colorFrameTimeS } from '@/simulator/renderer/frameRate';
import { DEFAULT_ACQUISITION } from '@/simulator/renderer/types';
import { controlThroughCrux, getViewTarget } from '@/simulator/windows/viewTargets';
import { DEFAULT_COLOR } from './colorDoppler';
import { colorLinePhaseOffset, wrapPhase } from './colorSweep';

const empty = (): FlowSample => ({ vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 });

test('colour sweep: the colour frame lasts its packets across the box, centred on the frame phase', () => {
  const s = { ...DEFAULT_ACQUISITION };
  const sweep = colorFrameTimeS(s, DEFAULT_COLOR);
  // the colour share of the frame time, exactly what the acquisition frame rate adds for colour
  assert.ok(Math.abs(1 / acquisitionFrameRate(s, DEFAULT_COLOR) - 1 / acquisitionFrameRate(s) - sweep) < 1e-12);
  assert.ok(sweep > 0.04 && sweep < 0.12, `default colour sweep ${sweep} s`);

  // no sweep, no offset; the lines fire from liMin to liMax, the middle one at the frame's phase
  assert.equal(colorLinePhaseOffset(10, 0, 20, 0, 0.9), 0);
  assert.equal(colorLinePhaseOffset(10, 0, 20, sweep, 0.9), 0);
  const first = colorLinePhaseOffset(0, 0, 20, sweep, 0.9);
  const last = colorLinePhaseOffset(20, 0, 20, sweep, 0.9);
  assert.ok(first < 0 && last > 0 && Math.abs(first + last) < 1e-15);
  assert.ok(Math.abs(last - first - (sweep / 0.9) * (20 / 21)) < 1e-12);
  for (let li = 1; li <= 20; li++)
    assert.ok(colorLinePhaseOffset(li, 0, 20, sweep, 0.9) > colorLinePhaseOffset(li - 1, 0, 20, sweep, 0.9));
  assert.equal(wrapPhase(-0.25), 0.75);
  assert.equal(wrapPhase(1.25), 0.25);
});

test('colour sweep: at mitral opening the later colour lines already hold the inflow the earlier ones do not', () => {
  const caseDef = loadCaseById('normal-excellent');
  const models = buildCaseModels(caseDef, baseInput().patient);
  const probe = controlThroughCrux(getViewTarget('a4c'), models.heart, models.thorax, null);
  const live = baseInput({ modality: 'color', probe });
  const core = new SimulatorCore(caseDef, live);
  const cine = (core as unknown as { cine: CineBuffer }).cine;
  for (let k = 0; k < 6 && !cine.newest; k++) core.step(1 / 30);
  const cf = cine.newest!;
  const tables = core.models.tables;
  const tm = tables.timings;
  // just after the mitral valve opens, while the inflow builds up
  const phase = (tm.mitralOpenS + 0.4 * tm.eAccelS) / tables.rrS;
  const a = cf.acquisition;
  const ctx: ColorUpdateContext = {
    frame: core.lastFrame!,
    scene: cf.scene,
    heart: cf.scene.heart,
    flow: (core as unknown as { flow: FlowFieldParams }).flow,
    tables,
    beam: cf.beam,
    spec: cf.spec,
    phase,
    probe: a.probe,
    color: a.color,
    frequencyMHz: a.settings.frequencyMHz,
    harmonics: a.settings.harmonics,
    modelVersion: a.modelVersion,
    timeS: cf.timeS,
    frameIntervalS: 1 / 30,
    seed: 7,
  };
  const n = cf.spec.lines * cf.spec.samples;
  const snapshot = new ColorFieldDriver();
  snapshot.resize(n);
  const still = snapshot.step(ctx).vel!.slice();
  const swept = new ColorFieldDriver();
  swept.resize(n);
  const sweepS = colorFrameTimeS(a.settings, a.color);
  const sweeping = swept.step({ ...ctx, sweepS }).vel!;

  const box = computeDopplerBoxBounds(cf.spec, a.color);
  const coloured = (vel: Float32Array, l0: number, l1: number): number => {
    let c = 0;
    for (let li = l0; li <= l1; li++)
      for (let si = box.siMin; si <= box.siMax; si++) if (Number.isFinite(vel[li * cf.spec.samples + si])) c++;
    return c;
  };
  const q = Math.max(1, Math.floor((box.liMax - box.liMin + 1) / 4));
  const earlyStill = coloured(still, box.liMin, box.liMin + q - 1);
  const earlySwept = coloured(sweeping, box.liMin, box.liMin + q - 1);
  const lateStill = coloured(still, box.liMax - q + 1, box.liMax);
  const lateSwept = coloured(sweeping, box.liMax - q + 1, box.liMax);
  assert.ok(
    earlySwept < earlyStill || lateSwept > lateStill,
    `sweep should shift the inflow: early ${earlyStill}→${earlySwept}, late ${lateStill}→${lateSwept}`,
  );
  assert.ok(earlySwept <= earlyStill && lateSwept >= lateStill);
  core.dispose();
});

test('diastolic LV vortex ring: rolls up in E, keeps turning through diastasis, gone in systole', () => {
  const c = loadCaseById('normal-excellent');
  const m = buildCaseModels(c, REST_PATIENT);
  const tm = m.tables.timings;
  const rr = m.tables.rrS;
  const at = (t: number, aShare = 0.4) => lvVortexState(tm, rr, t, aShare).strength;
  // ejection, isovolumic contraction and relaxation: no vortex
  assert.equal(at(0.5 * tm.ejectionStartS), 0);
  assert.equal(at(tm.ejectionStartS + 0.01), 0);
  assert.equal(at(0.5 * (tm.ejectionStartS + tm.ejectionEndS)), 0);
  assert.equal(at(tm.mitralOpenS), 0);
  // complete in mid E deceleration, still turning in diastasis, weaker by then
  const peak = at(tm.mitralOpenS + tm.eAccelS + 0.55 * tm.eDecelS);
  const diastasis = at(0.5 * (tm.eEndS + tm.aStartS));
  assert.ok(peak > 0.9, `peak ${peak}`);
  assert.ok(diastasis > 0.3 && diastasis < peak, `diastasis ${diastasis}`);
  // the atrial contraction rolls up its own ring
  const afterA = tm.aStartS + 0.6 * (tm.aEndS - tm.aStartS);
  assert.ok(at(afterA, 0.6) > at(afterA, 0));
  // the ring travels from the tips toward the mid cavity
  const tr0 = lvVortexState(tm, rr, tm.mitralOpenS + tm.eAccelS, 0).travel;
  const tr1 = lvVortexState(tm, rr, tm.eEndS, 0).travel;
  assert.ok(tr1 > tr0 && tr1 <= 1);
});

test('diastolic LV vortex ring: basal return outside the jet, larger toward the LVOT, nothing in the jet core', () => {
  const c = loadCaseById('normal-excellent');
  const m = buildCaseModels(c, REST_PATIENT);
  const flow = buildFlowParams(c, m.heart, m.tables, m.thorax);
  const tm = m.tables.timings;
  const phase = (tm.mitralOpenS + tm.eAccelS + 0.6 * tm.eDecelS) / m.tables.rrS;
  const hp = computeHeartPose(m.heart, cycleStateAt(m.tables, phase, flow.flowPeaks));
  const ux = flow.avCenter.x - flow.mvCenter.x;
  const uy = flow.avCenter.y - flow.mvCenter.y;
  const ul = Math.hypot(ux, uy);
  const re = Math.sqrt(flow.mvAreaCm2 / Math.PI);
  const ring = (side: number, rho: number, zr: number): FlowSample => {
    const o = empty();
    sampleLvDiastolicVortex(
      m.tables,
      phase,
      flow.mvCenter.x + side * (ux / ul) * rho,
      flow.mvCenter.y + side * (uy / ul) * rho,
      hp.zAnn + zr,
      hp.zAnn,
      hp.lengthNow,
      flow.mvCenter,
      flow.avCenter,
      flow.avAxis,
      c.physiology.ePeakMps,
      c.physiology.ePrimeSeptalCmps,
      o,
      flow.mvAreaCm2,
      c.physiology.aPeakMps,
    );
    return o;
  };
  // inside the jet's core the ring adds nothing (the PW sample volume and the colour M-mode line stay exact)
  for (const zr of [0.6, 1.2, 2.4, 3.6]) {
    const o = ring(1, 0.5 * re, zr);
    assert.equal(o.vx, 0);
    assert.equal(o.vy, 0);
    assert.equal(o.vz, 0);
  }
  // outside the jet the blood returns toward the base on both sides; more strongly under the anterior leaflet
  let antBasal = 0,
    postBasal = 0;
  for (let zr = 0.4; zr < 0.8 * hp.lengthNow; zr += 0.2) {
    antBasal = Math.min(antBasal, ring(1, re + 0.8, zr).vz);
    postBasal = Math.min(postBasal, ring(-1, re + 0.8, zr).vz);
  }
  assert.ok(postBasal < -0.03, `posterior lobe basal return ${postBasal}`);
  assert.ok(antBasal < postBasal, `anterior lobe ${antBasal} stronger than posterior ${postBasal}`);
  // laminar
  assert.ok(ring(1, re + 0.8, 2).dispersion <= 0.1);
  // the full field still holds pure axial inflow at the leaflet tips
  const fs = empty();
  sampleFlow(flow, m.tables, hp, phase, flow.mvCenter.x, flow.mvCenter.y, hp.zAnn + 1.2, fs, Structure.LvCavity);
  assert.ok(fs.vz > 0.2 && Math.abs(fs.vx) < 1e-9 && Math.abs(fs.vy) < 1e-9);
});
