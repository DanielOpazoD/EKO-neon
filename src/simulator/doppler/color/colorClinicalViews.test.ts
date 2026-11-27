import test from 'node:test';
import assert from 'node:assert/strict';
import { loadCaseById } from '@/cases';
import { buildCaseModels, REST_PATIENT } from '@/simulator/anatomy/caseModels';
import { torsoToHeart } from '@/simulator/anatomy/heartFrame';
import { computeHeartPose } from '@/simulator/anatomy/heartModel';
import { Structure, Tissue } from '@/simulator/anatomy/tissue';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import {
  buildFlowParams,
  sampleFlow,
  type FlowFieldParams,
  type FlowSample,
} from '@/simulator/doppler/flow-primitives/flowField';
import { colorBoxForView } from '@/simulator/doppler/color/colorDoppler';
import { beamFrameFromPose, poseFromControl } from '@/simulator/probe/pose';
import { ProceduralSliceRenderer } from '@/simulator/renderer/procedural/sliceRenderer';
import { scenePhysicsFor } from '@/simulator/renderer/scenePhysics';
import {
  allocPolarFrame,
  DEFAULT_ACQUISITION,
  type PolarFrameSpec,
  type Scene,
} from '@/simulator/renderer/types';
import { canonicalControl, getViewTarget } from '@/simulator/windows/viewTargets';

/**
 * End-to-end view-level Color Doppler clinical fidelity regression suite.
 * Renders real canonical TTE slices via ProceduralSliceRenderer and evaluates anatomical
 * chamber-contained flow (`sampleFlow(..., structure)`) across representative pathologies
 * and cardiac phases so every per-view flow feature stays locked against regressions.
 */

interface StructureFlowSummary {
  samples: number;
  inBoxSamples: number;
  flowSamples: number;
  fillPct: number;
  inBoxPct: number;
  maxSpeed: number;
  maxAbsAxial: number;
  meanSignedAxial: number;
  maxDisp: number;
}

const SPEC: PolarFrameSpec = {
  lines: 72,
  samples: 128,
  sectorRad: (80 * Math.PI) / 180,
  depthCm: 16,
  focusCm: 9,
  elevationSamples: 1,
};

interface CaseBundle {
  caseDef: ReturnType<typeof loadCaseById>;
  models: ReturnType<typeof buildCaseModels>;
  flow: FlowFieldParams;
}

const renderer = new ProceduralSliceRenderer();
const frame = allocPolarFrame(SPEC);
const bundleCache = new Map<string, CaseBundle>();

function bundleFor(caseId: string): CaseBundle {
  let b = bundleCache.get(caseId);
  if (!b) {
    const caseDef = loadCaseById(caseId);
    const models = buildCaseModels(caseDef, REST_PATIENT);
    const flow = buildFlowParams(caseDef, models.heart, models.tables, models.thorax);
    b = { caseDef, models, flow };
    bundleCache.set(caseId, b);
  }
  return b;
}

function auditSlice(
  caseId: string,
  viewId: string,
  phase: number,
): Map<number, StructureFlowSummary> {
  const { caseDef, models, flow } = bundleFor(caseId);
  const { heart, thorax, tables } = models;
  const tgt = getViewTarget(viewId);
  const ctrl = canonicalControl(tgt, heart, thorax);
  const beam = beamFrameFromPose(poseFromControl(thorax, ctrl));
  const hp = computeHeartPose(heart, cycleStateAt(tables, phase, flow.flowPeaks));
  const scene: Scene = {
    heart,
    thorax,
    heartPose: hp,
    physics: scenePhysicsFor(caseDef, DEFAULT_ACQUISITION),
  };
  const hf = heart.frame;
  const oh = torsoToHeart(hf, beam.origin);
  const box = colorBoxForView(viewId, SPEC.depthCm, SPEC.sectorRad);
  const dr = SPEC.depthCm / SPEC.samples;

  renderer.render(scene, beam, SPEC, phase, frame);

  const fs: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };
  const acc = new Map<
    number,
    {
      samples: number;
      inBoxSamples: number;
      flowSamples: number;
      maxSpeed: number;
      maxAbsAxial: number;
      sumSignedAxial: number;
      maxDisp: number;
    }
  >();

  for (let li = 0; li < SPEC.lines; li++) {
    const theta = -0.5 * SPEC.sectorRad + ((li + 0.5) / SPEC.lines) * SPEC.sectorRad;
    const ct = Math.cos(theta);
    const sn = Math.sin(theta);
    const dx = beam.forward.x * ct + beam.lateral.x * sn;
    const dy = beam.forward.y * ct + beam.lateral.y * sn;
    const dz = beam.forward.z * ct + beam.lateral.z * sn;
    const dhx = dx * hf.ex.x + dy * hf.ex.y + dz * hf.ex.z;
    const dhy = dx * hf.ey.x + dy * hf.ey.y + dz * hf.ey.z;
    const dhz = dx * hf.ez.x + dy * hf.ez.y + dz * hf.ez.z;

    for (let si = 0; si < SPEC.samples; si++) {
      const idx = li * SPEC.samples + si;
      if (frame.tissue[idx] !== Tissue.Blood) continue;
      const st = frame.structure[idx]!;
      const r = (si + 0.5) * dr;
      const inBox =
        r >= box.boxRMinCm &&
        r <= box.boxRMaxCm &&
        theta >= box.boxThetaMinRad &&
        theta <= box.boxThetaMaxRad;

      sampleFlow(
        flow,
        tables,
        hp,
        phase,
        oh.x + dhx * r,
        oh.y + dhy * r,
        oh.z + dhz * r,
        fs,
        st,
      );
      const speed = fs.present ? Math.hypot(fs.vx, fs.vy, fs.vz) : 0;
      const vAxial = fs.present ? -(fs.vx * dhx + fs.vy * dhy + fs.vz * dhz) : 0;

      let entry = acc.get(st);
      if (!entry) {
        entry = {
          samples: 0,
          inBoxSamples: 0,
          flowSamples: 0,
          maxSpeed: 0,
          maxAbsAxial: 0,
          sumSignedAxial: 0,
          maxDisp: 0,
        };
        acc.set(st, entry);
      }
      entry.samples++;
      if (inBox) entry.inBoxSamples++;
      if (fs.present && speed >= 0.05) {
        entry.flowSamples++;
        entry.sumSignedAxial += vAxial;
      }
      if (speed > entry.maxSpeed) entry.maxSpeed = speed;
      if (Math.abs(vAxial) > entry.maxAbsAxial) entry.maxAbsAxial = Math.abs(vAxial);
      if (fs.dispersion > entry.maxDisp) entry.maxDisp = fs.dispersion;
    }
  }

  const out = new Map<number, StructureFlowSummary>();
  for (const [st, e] of acc) {
    out.set(st, {
      samples: e.samples,
      inBoxSamples: e.inBoxSamples,
      flowSamples: e.flowSamples,
      fillPct: (100 * e.flowSamples) / Math.max(1, e.samples),
      inBoxPct: (100 * e.inBoxSamples) / Math.max(1, e.samples),
      maxSpeed: e.maxSpeed,
      maxAbsAxial: e.maxAbsAxial,
      meanSignedAxial: e.flowSamples > 0 ? e.sumSignedAxial / e.flowSamples : 0,
      maxDisp: e.maxDisp,
    });
  }
  return out;
}

function requireStat(map: Map<number, StructureFlowSummary>, st: Structure, label: string): StructureFlowSummary {
  const s = map.get(st);
  assert.ok(s && s.samples > 0, `Expected visible blood samples for ${label}`);
  return s;
}

test('view-level Color fidelity: pulmonary trunk, RPA beneath arch, and PH diastolic PR in PSAX-AV', () => {
  const psaxNorm = auditSlice('normal-excellent', 'psax-av', 0.28);
  const pa = requireStat(psaxNorm, Structure.PulmonaryArtery, 'normal psax-av PulmonaryArtery');
  const rvot = requireStat(psaxNorm, Structure.Rvot, 'normal psax-av Rvot');
  assert.ok(pa.fillPct >= 90 && pa.maxSpeed >= 0.75 && pa.maxDisp <= 0.15, `Unexpected PA flow: ${JSON.stringify(pa)}`);
  assert.ok(rvot.fillPct >= 90, `Expected RVOT systolic fill >= 90%, got ${rvot.fillPct.toFixed(1)}%`);
  assert.ok(pa.inBoxPct >= 75, `View Color ROI must frame PA in PSAX-AV, got ${pa.inBoxPct.toFixed(1)}%`);

  const ssnNorm = auditSlice('normal-excellent', 'suprasternal-arch', 0.28);
  const rpa = requireStat(ssnNorm, Structure.PulmonaryArtery, 'suprasternal RPA');
  const arch = requireStat(ssnNorm, Structure.AorticArch, 'suprasternal AorticArch');
  const desc = requireStat(ssnNorm, Structure.DescendingAorta, 'suprasternal DescendingAorta');
  assert.ok(rpa.fillPct >= 90 && rpa.maxSpeed >= 0.7, `Unexpected suprasternal RPA: ${JSON.stringify(rpa)}`);
  assert.ok(arch.fillPct >= 90 && desc.fillPct >= 90, `Unexpected arch/desc fill: arch=${arch.fillPct.toFixed(1)}% desc=${desc.fillPct.toFixed(1)}%`);
  assert.ok(arch.inBoxPct >= 70, `View Color ROI must frame AorticArch in suprasternal view, got ${arch.inBoxPct.toFixed(1)}%`);

  // PH early diastole: brisk turbulent PR jet in RVOT with zero leak into AorticRoot
  const psaxPhDia = auditSlice('pulmonary-hypertension-rv', 'psax-av', 0.65);
  const prRvot = requireStat(psaxPhDia, Structure.Rvot, 'PH psax-av diastolic Rvot');
  const prAo = requireStat(psaxPhDia, Structure.AorticRoot, 'PH psax-av diastolic AorticRoot');
  assert.ok(
    prRvot.fillPct >= 70 && prRvot.maxSpeed >= 2.0 && prRvot.maxDisp >= 0.4,
    `Expected turbulent PH PR jet in RVOT, got ${JSON.stringify(prRvot)}`,
  );
  assert.equal(prAo.flowSamples, 0, 'Diastolic PR jet must never bleed into AorticRoot');
});

test('view-level Color fidelity: systemic veins (IVC, hepatic vein, SVC, coronary sinus) and PH systolic reversal', () => {
  const ivcNorm = auditSlice('normal-excellent', 'subcostal-ivc', 0.28);
  const ivcN = requireStat(ivcNorm, Structure.Ivc, 'normal subcostal IVC');
  const hvN = requireStat(ivcNorm, Structure.HepaticVein, 'normal subcostal HepaticVein');
  assert.ok(ivcN.fillPct >= 90 && ivcN.maxSpeed >= 0.35 && ivcN.maxDisp <= 0.1, `Unexpected normal IVC: ${JSON.stringify(ivcN)}`);
  assert.ok(hvN.fillPct >= 90 && hvN.maxSpeed >= 0.35, `Unexpected normal HV: ${JSON.stringify(hvN)}`);
  assert.ok(ivcN.inBoxPct >= 50, `View Color ROI must frame IVC in subcostal-ivc, got ${ivcN.inBoxPct.toFixed(1)}%`);

  // Severe TR in PH reverses systolic flow sign in IVC & hepatic vein compared to normal
  const ivcPh = auditSlice('pulmonary-hypertension-rv', 'subcostal-ivc', 0.28);
  const ivcP = requireStat(ivcPh, Structure.Ivc, 'PH subcostal IVC');
  const hvP = requireStat(ivcPh, Structure.HepaticVein, 'PH subcostal HepaticVein');
  assert.ok(ivcP.fillPct >= 90 && hvP.fillPct >= 90, 'PH IVC and HV should carry full systolic flow');
  assert.ok(
    ivcN.meanSignedAxial * ivcP.meanSignedAxial < 0,
    `Expected opposite systolic flow sign in PH IVC vs normal: norm=${ivcN.meanSignedAxial.toFixed(2)} ph=${ivcP.meanSignedAxial.toFixed(2)}`,
  );
  assert.ok(
    hvN.meanSignedAxial * hvP.meanSignedAxial < 0,
    `Expected opposite systolic flow sign in PH HV vs normal: norm=${hvN.meanSignedAxial.toFixed(2)} ph=${hvP.meanSignedAxial.toFixed(2)}`,
  );

  // SVC in A4C and CoronarySinus in A2C
  const a4cNorm = auditSlice('normal-excellent', 'a4c', 0.28);
  const svc = requireStat(a4cNorm, Structure.Svc, 'normal A4C SVC');
  assert.ok(svc.fillPct >= 90 && svc.maxSpeed >= 0.25, `Unexpected A4C SVC flow: ${JSON.stringify(svc)}`);

  const a2cNorm = auditSlice('normal-excellent', 'a2c', 0.65);
  const cs = requireStat(a2cNorm, Structure.CoronarySinus, 'normal A2C CoronarySinus');
  assert.ok(cs.fillPct >= 85 && cs.maxSpeed >= 0.15, `Unexpected A2C CoronarySinus flow: ${JSON.stringify(cs)}`);
});

test('view-level Color fidelity: eccentric Coanda MR, PV systolic reversal, and SVC/RA containment in MVP', () => {
  const plaxMr = auditSlice('mvp-primary-mr', 'plax', 0.28);
  const laPlax = requireStat(plaxMr, Structure.LaCavity, 'MVP PLAX LaCavity');
  const lvPlax = requireStat(plaxMr, Structure.LvCavity, 'MVP PLAX LvCavity');
  assert.ok(
    laPlax.fillPct >= 35 && laPlax.maxSpeed >= 4.0 && laPlax.maxDisp >= 0.5,
    `Expected eccentric Coanda MR jet in PLAX LA, got ${JSON.stringify(laPlax)}`,
  );
  assert.ok(lvPlax.maxDisp <= 0.15, `LV PISA side in PLAX must remain laminar, got d=${lvPlax.maxDisp.toFixed(2)}`);

  const a3cMr = auditSlice('mvp-primary-mr', 'a3c', 0.28);
  const laA3c = requireStat(a3cMr, Structure.LaCavity, 'MVP A3C LaCavity');
  assert.ok(
    laA3c.fillPct >= 35 && laA3c.maxSpeed >= 4.0 && laA3c.maxDisp >= 0.5,
    `Expected eccentric Coanda MR jet in A3C LA, got ${JSON.stringify(laA3c)}`,
  );

  const a4cMr = auditSlice('mvp-primary-mr', 'a4c', 0.28);
  const pvA4c = requireStat(a4cMr, Structure.PulmonaryVein, 'MVP A4C PulmonaryVein');
  const svcA4c = requireStat(a4cMr, Structure.Svc, 'MVP A4C Svc');
  assert.ok(
    pvA4c.fillPct >= 85 && pvA4c.maxDisp >= 0.1,
    `Expected severe MR systolic reversal in A4C PulmonaryVein, got ${JSON.stringify(pvA4c)}`,
  );
  assert.ok(
    svcA4c.maxDisp <= 0.1,
    `Eccentric MR jet must not contaminate SVC, got disp=${svcA4c.maxDisp.toFixed(2)}`,
  );
});

test('view-level Color fidelity: dynamic HOCM subaortic SAM mosaic and late-systolic posterior SAM-MR', () => {
  // Early systole before SAM contact: LVOT less turbulent and LA has no high-velocity MR mosaic
  const plaxEarly = auditSlice('hocm-sam', 'plax', 0.18);
  const laEarly = requireStat(plaxEarly, Structure.LaCavity, 'HOCM PLAX early LaCavity');
  assert.ok(laEarly.maxDisp <= 0.15, `Early-systolic HOCM LA should not show SAM-MR yet, got d=${laEarly.maxDisp.toFixed(2)}`);

  // Late systole with peak SAM obstruction: subaortic LVOT mosaic + posterior SAM-MR in PLAX & A3C
  const plaxLate = auditSlice('hocm-sam', 'plax', 0.28);
  const lvotLate = requireStat(plaxLate, Structure.Lvot, 'HOCM PLAX late Lvot');
  const laLate = requireStat(plaxLate, Structure.LaCavity, 'HOCM PLAX late LaCavity');
  assert.ok(
    lvotLate.fillPct >= 85 && lvotLate.maxDisp >= 0.5,
    `Expected late-systolic SAM mosaic in PLAX LVOT, got ${JSON.stringify(lvotLate)}`,
  );
  assert.ok(
    laLate.fillPct >= 25 && laLate.maxSpeed >= 3.5 && laLate.maxDisp >= 0.45,
    `Expected posterior late-systolic SAM-MR in PLAX LA, got ${JSON.stringify(laLate)}`,
  );

  const a3cLate = auditSlice('hocm-sam', 'a3c', 0.28);
  const lvotA3c = requireStat(a3cLate, Structure.Lvot, 'HOCM A3C late Lvot');
  const laA3c = requireStat(a3cLate, Structure.LaCavity, 'HOCM A3C late LaCavity');
  assert.ok(lvotA3c.fillPct >= 85 && lvotA3c.maxDisp >= 0.5, `Unexpected HOCM A3C LVOT: ${JSON.stringify(lvotA3c)}`);
  assert.ok(laA3c.fillPct >= 25 && laA3c.maxDisp >= 0.45, `Unexpected HOCM A3C SAM-MR: ${JSON.stringify(laA3c)}`);
});

test('view-level Color fidelity: severe AS post-stenotic turbulent plume and cross-chamber isolation', () => {
  for (const viewId of ['plax', 'a5c'] as const) {
    const m = auditSlice('aortic-stenosis-severe', viewId, 0.28);
    const root = requireStat(m, Structure.AorticRoot, `AS ${viewId} AorticRoot`);
    const asc = requireStat(m, Structure.AscendingAorta, `AS ${viewId} AscendingAorta`);
    const rv = requireStat(m, Structure.RvCavity, `AS ${viewId} RvCavity`);
    const la = requireStat(m, Structure.LaCavity, `AS ${viewId} LaCavity`);
    assert.ok(
      root.fillPct >= 80 && root.maxSpeed >= 3.5 && root.maxDisp >= 0.7,
      `Expected turbulent AS plume in ${viewId} AorticRoot, got ${JSON.stringify(root)}`,
    );
    assert.ok(
      asc.fillPct >= 90 && asc.maxDisp >= 0.7,
      `Expected turbulent AS plume in ${viewId} AscendingAorta, got ${JSON.stringify(asc)}`,
    );
    assert.ok(rv.maxDisp <= 0.15, `AS plume must not bleed into ${viewId} RvCavity (d=${rv.maxDisp.toFixed(2)})`);
    assert.ok(la.maxDisp <= 0.15, `AS plume must not bleed into ${viewId} LaCavity (d=${la.maxDisp.toFixed(2)})`);
  }

  const psaxAs = auditSlice('aortic-stenosis-severe', 'psax-av', 0.18);
  const rootPsax = requireStat(psaxAs, Structure.AorticRoot, 'AS psax-av AorticRoot');
  assert.ok(rootPsax.fillPct >= 80 && rootPsax.maxDisp >= 0.7, `Unexpected AS PSAX-AV root: ${JSON.stringify(rootPsax)}`);
});

test('view-level Color fidelity: severe TR septal-curving jet, 3D RV PISA, LV diastolic vortex, and aortic Windkessel runoff', () => {
  // Severe TR in A4C & RV-focused with clean left-heart isolation
  for (const viewId of ['a4c', 'rv-focused'] as const) {
    const ph = auditSlice('pulmonary-hypertension-rv', viewId, 0.28);
    const ra = requireStat(ph, Structure.RaCavity, `PH ${viewId} RaCavity`);
    const rv = requireStat(ph, Structure.RvCavity, `PH ${viewId} RvCavity`);
    const la = requireStat(ph, Structure.LaCavity, `PH ${viewId} LaCavity`);
    assert.ok(
      ra.fillPct >= 30 && ra.maxSpeed >= 3.0 && ra.maxDisp >= 0.45,
      `Expected severe TR jet in ${viewId} RA, got ${JSON.stringify(ra)}`,
    );
    assert.ok(
      rv.fillPct >= 5 && rv.maxDisp <= 0.12,
      `Expected laminar RV-side TR PISA in ${viewId}, got ${JSON.stringify(rv)}`,
    );
    assert.ok(la.maxDisp <= 0.12, `TR jet must not bleed into ${viewId} LaCavity (d=${la.maxDisp.toFixed(2)})`);
  }

  // LV diastolic apical vortex and anteroseptal LVOT redirection during E-deceleration
  for (const viewId of ['plax', 'a3c'] as const) {
    const dia = auditSlice('normal-excellent', viewId, 0.65);
    const lv = requireStat(dia, Structure.LvCavity, `normal ${viewId} E-decel LvCavity`);
    const lvot = requireStat(dia, Structure.Lvot, `normal ${viewId} E-decel Lvot`);
    assert.ok(lv.fillPct >= 85 && lv.maxDisp <= 0.15, `Expected LV diastolic vortex fill in ${viewId}, got ${JSON.stringify(lv)}`);
    assert.ok(lvot.fillPct >= 35 && lvot.maxDisp <= 0.15, `Expected anteroseptal LVOT redirection in ${viewId}, got ${JSON.stringify(lvot)}`);
  }

  // Elastic Windkessel early-diastolic runoff in aortic arch & descending aorta
  const ssnDia = auditSlice('normal-excellent', 'suprasternal-arch', 0.52);
  const archDia = requireStat(ssnDia, Structure.AorticArch, 'normal suprasternal early-diastole AorticArch');
  const descDia = requireStat(ssnDia, Structure.DescendingAorta, 'normal suprasternal early-diastole DescendingAorta');
  assert.ok(
    archDia.fillPct >= 80 && archDia.maxSpeed >= 0.12 && archDia.maxDisp <= 0.05,
    `Expected laminar Windkessel runoff in AorticArch, got ${JSON.stringify(archDia)}`,
  );
  assert.ok(
    descDia.fillPct >= 80 && descDia.maxSpeed >= 0.12 && descDia.maxDisp <= 0.05,
    `Expected laminar Windkessel runoff in DescendingAorta, got ${JSON.stringify(descDia)}`,
  );
});
