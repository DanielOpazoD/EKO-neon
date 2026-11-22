import test from 'node:test';
import assert from 'node:assert/strict';
import { CASE_INPUTS, loadCaseById } from '@/cases';
import { buildCaseModels, REST_PATIENT } from '@/simulator/anatomy/caseModels';
import { measureModel } from '@/simulator/anatomy/measureModel';
import { computeGroundTruth, type StructuredEchoTruth } from './groundTruth';
import {
  pisaEroaCm2,
  pisaFlowRateMlps,
  pisaRadiusCm,
  venaContractaFromEroaCm,
} from '@/clinical/formulas';
import { gradeMitralRegurgitation } from '@/clinical/guidelines/valveGrading';
import { assessDiastolic } from '@/clinical/guidelines/diastolicAssessment';
import { buildFlowParams } from '@/simulator/doppler/flow-primitives/flowField';
import { sampleRegurgitantJets } from '@/simulator/doppler/flow-primitives/regurgitantJets';
import { computeHeartPose } from '@/simulator/anatomy/heartModel';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import {
  GLSL_HEART,
  GLSL_HEART_CHAMBERS,
  GLSL_HEART_CONSTANTS,
  GLSL_HEART_VALVES,
} from '@/simulator/renderer/gpu/glslHeart';

function truthFor(id: string): StructuredEchoTruth {
  const c = loadCaseById(id);
  const { tables } = buildCaseModels(c, REST_PATIENT);
  return computeGroundTruth(c, tables);
}

test('all 12 clinical cases produce finite, physiologically consistent hemodynamic ground truth', () => {
  assert.equal(CASE_INPUTS.length, 12, 'expected 12 mandatory clinical cases');
  for (const input of CASE_INPUTS) {
    const t = truthFor(input.id);
    assert.ok(Number.isFinite(t.bsaM2) && t.bsaM2 > 1.0 && t.bsaM2 < 3.0, `${input.id}: BSA`);
    assert.ok(t.lv.edvMl > t.lv.esvMl && t.lv.esvMl > 0, `${input.id}: EDV > ESV > 0`);
    assert.ok(t.lv.efPct > 10 && t.lv.efPct < 88, `${input.id}: LVEF=${t.lv.efPct}`);
    assert.ok(t.lv.cardiacOutputLpm > 1.0 && t.lv.cardiacOutputLpm < 12.0, `${input.id}: CO`);
    assert.ok(t.aorticValve.vmaxMps > 0.5 && t.aorticValve.vmaxMps < 8.0, `${input.id}: AV Vmax`);
    assert.ok(t.aorticValve.continuityAvaCm2 > 0.3 && t.aorticValve.continuityAvaCm2 < 6.0, `${input.id}: AVA`);
    assert.ok(t.mitral.ePeakMps > 0.2 && t.mitral.ePeakMps < 3.0, `${input.id}: MV E peak`);
    assert.equal(t.wallMotion.scores.length, 16, `${input.id}: 16 wall-motion segments`);
  }
});

test('clinical pathology invariants hold across severity tiers and hemodynamic phenotypes', () => {
  const normal = truthFor('normal-excellent');
  const modAs = truthFor('aortic-stenosis-moderate');
  const sevAs = truthFor('aortic-stenosis-severe');
  const hfrefMr = truthFor('hfref-severe-mr');
  const mvpMr = truthFor('mvp-primary-mr');
  const hocm = truthFor('hocm-sam');
  const ph = truthFor('pulmonary-hypertension-rv');
  const tamp = truthFor('pericardial-effusion-tamponade');
  const af = truthFor('af-diastolic');
  const rwma = truthFor('inferior-rwma');

  // Normal baseline
  assert.ok(normal.lv.efPct >= 55 && normal.lv.efPct <= 72, `normal EF=${normal.lv.efPct}`);
  assert.ok(normal.aorticValve.vmaxMps < 2.0, `normal AV Vmax=${normal.aorticValve.vmaxMps}`);

  // Aortic stenosis progression: normal < moderate < severe
  assert.ok(
    normal.aorticValve.vmaxMps < modAs.aorticValve.vmaxMps &&
      modAs.aorticValve.vmaxMps < sevAs.aorticValve.vmaxMps,
    'AV Vmax must increase monotonically from normal -> moderate AS -> severe AS',
  );
  assert.ok(sevAs.aorticValve.vmaxMps >= 4.0, `severe AS Vmax=${sevAs.aorticValve.vmaxMps}`);
  assert.ok(
    sevAs.aorticValve.meanGradientMmHg >= 40,
    `severe AS MPG=${sevAs.aorticValve.meanGradientMmHg}`,
  );
  assert.ok(
    sevAs.aorticValve.effectiveAreaCm2 <= 1.0,
    `severe AS AVA=${sevAs.aorticValve.effectiveAreaCm2}`,
  );

  // HFrEF + functional secondary MR (EROA >= 0.20) vs primary MVP severe MR (EROA >= 0.40)
  assert.ok(hfrefMr.lv.efPct <= 35, `HFrEF EF=${hfrefMr.lv.efPct}`);
  assert.ok(hfrefMr.lv.eddCm > 5.8, `HFrEF dilated EDD=${hfrefMr.lv.eddCm}`);
  assert.ok(hfrefMr.regurgitation.mr !== null && hfrefMr.regurgitation.mr.eroaCm2 >= 0.2);
  assert.ok(mvpMr.lv.efPct >= 55, `MVP primary MR preserved EF=${mvpMr.lv.efPct}`);
  assert.ok(mvpMr.regurgitation.mr !== null && mvpMr.regurgitation.mr.eroaCm2 >= 0.4);

  // HOCM with SAM & dynamic LVOT obstruction
  assert.ok(hocm.lv.ivsdCm >= 1.5, `HOCM IVSd=${hocm.lv.ivsdCm}`);
  assert.equal(hocm.lvot.dynamicObstruction, true, 'HOCM must have dynamic LVOT obstruction');
  assert.ok(hocm.lvot.peakGradientMmHg >= 30, `HOCM LVOT PG=${hocm.lvot.peakGradientMmHg}`);

  // Pulmonary hypertension & RV dilation
  assert.ok(ph.rightHeart.trVmaxMps !== null && ph.rightHeart.trVmaxMps >= 3.4);
  assert.ok(ph.rightHeart.rvspMmHg !== null && ph.rightHeart.rvspMmHg >= 50);
  assert.ok(ph.rv.basalDiameterCm > 4.1, `PH RV basal=${ph.rv.basalDiameterCm}`);

  // Pericardial effusion & tamponade
  assert.ok(tamp.pericardium.effusionCm >= 2.0, `tamponade effusion=${tamp.pericardium.effusionCm}`);
  assert.ok(tamp.pericardium.tamponade >= 0.5, `tamponade severity=${tamp.pericardium.tamponade}`);

  // Atrial fibrillation loses A wave
  assert.equal(af.rhythm, 'atrial-fibrillation');
  assert.equal(af.mitral.aPeakMps, 0);
  assert.equal(af.mitral.eOverA, null);

  // Inferior RWMA has regional abnormal segments & WMSI > 1
  assert.ok(rwma.wallMotion.abnormalSegments.length >= 2);
  assert.ok(rwma.wallMotion.wmsi > 1.05);
});

test('anatomical measureModel matches pathological chamber & wall remodeling', () => {
  const rowVal = (m: ReturnType<typeof measureModel>, id: string): number => {
    const r = m.rows.find((x) => x.id === id);
    assert.ok(r, `missing measurement row ${id}`);
    return r.value;
  };

  // 12,000 Monte Carlo samples gives ~1% volume precision in <150ms per case
  const mNormal = measureModel(loadCaseById('normal-excellent'), REST_PATIENT, 12000);
  const mHocm = measureModel(loadCaseById('hocm-sam'), REST_PATIENT, 12000);
  const mHfref = measureModel(loadCaseById('hfref-severe-mr'), REST_PATIENT, 12000);

  const normalEf = rowVal(mNormal, 'lv-ef');
  const hfrefEf = rowVal(mHfref, 'lv-ef');
  assert.ok(normalEf >= 50 && normalEf <= 76, `normal geometric EF=${normalEf.toFixed(1)}%`);
  assert.ok(hfrefEf < 40, `HFrEF geometric EF=${hfrefEf.toFixed(1)}% should be <40%`);

  const normalIvs = rowVal(mNormal, 'ivsd');
  const hocmIvs = rowVal(mHocm, 'ivsd');
  assert.ok(
    hocmIvs > normalIvs + 0.6 && hocmIvs >= 1.5,
    `HOCM IVSd (${hocmIvs.toFixed(2)} cm) must exceed normal (${normalIvs.toFixed(2)} cm)`,
  );

  const normalIdd = rowVal(mNormal, 'lv-idd');
  const hfrefIdd = rowVal(mHfref, 'lv-idd');
  assert.ok(
    hfrefIdd > normalIdd + 0.8,
    `HFrEF LVIDd (${hfrefIdd.toFixed(2)} cm) must exceed normal (${normalIdd.toFixed(2)} cm)`,
  );
});

test('modular GLSL_HEART assembles all chunks cleanly with balanced syntax and no NaN/undefined', () => {
  assert.ok(GLSL_HEART.startsWith(GLSL_HEART_CONSTANTS));
  assert.ok(GLSL_HEART.includes(GLSL_HEART_VALVES));
  assert.ok(GLSL_HEART.includes(GLSL_HEART_CHAMBERS));
  assert.ok(!/\bNaN\b|\bundefined\b/.test(GLSL_HEART), 'GLSL_HEART must not contain NaN/undefined');

  const requiredSymbols = [
    'struct Sample',
    'void setSample(',
    'float skirtDistance(',
    'float mitralDistance(',
    'float aorticCuspDistance(',
    'float lvCavitySdf(',
    'vec3 rvCrescent(',
    'bool classifyAorticExtension(',
    'float pulmonaryTrunkDist(',
    'float coronarySinusDistance(',
    'bool classifyHeart(',
  ];
  for (const sym of requiredSymbols) {
    assert.ok(GLSL_HEART.includes(sym), `missing GLSL symbol: ${sym}`);
  }

  let depth = 0;
  for (const ch of GLSL_HEART) {
    if (ch === '{') depth++;
    else if (ch === '}') depth--;
    assert.ok(depth >= 0, 'unbalanced closing brace in GLSL_HEART');
  }
  assert.equal(depth, 0, 'unbalanced braces in GLSL_HEART');
});

test('MR PISA/vena contracta formulas, ground truth, integrative grading and proximal convergence flow field agree physically', () => {
  // Exact inverse consistency between PISA radius and EROA
  const eroa = 0.45;
  const vmax = 5.2;
  const va = 0.4;
  const r = pisaRadiusCm(eroa, vmax, va);
  assert.ok(r > 0.9 && r < 1.1, `severe MR PISA radius at 40 cm/s should be ~0.96 cm, got ${r}`);
  assert.ok(Math.abs(pisaEroaCm2(r, va, vmax) - eroa) < 1e-9);
  assert.ok(Math.abs(pisaFlowRateMlps(r, va) - eroa * vmax * 100) < 1e-9);

  // Vena contracta width physically aligns with ASE 2017 thresholds
  assert.ok(venaContractaFromEroaCm(0.07) < 0.3);
  assert.ok(venaContractaFromEroaCm(0.4) >= 0.7);

  // Ground truth populates VC & PISA radius for severe primary MR
  const mvpTruth = truthFor('mvp-primary-mr');
  assert.ok(mvpTruth.regurgitation.mr !== null);
  assert.ok((mvpTruth.regurgitation.mr!.venaContractaCm ?? 0) >= 0.7);
  assert.ok((mvpTruth.regurgitation.mr!.pisaRadius40Cm ?? 0) >= 0.9);
  assert.equal(gradeMitralRegurgitation(mvpTruth.regurgitation.mr!), 'severe');

  // Integrative grading with VC respects ASE 2017 mild/severe cutoffs
  assert.equal(
    gradeMitralRegurgitation({
      eroaCm2: 0.15,
      regurgitantVolumeMl: 20,
      regurgitantFractionPct: 22,
      venaContractaCm: 0.25,
    }),
    'mild',
  );
  assert.equal(
    gradeMitralRegurgitation({
      eroaCm2: 0.42,
      regurgitantVolumeMl: 65,
      regurgitantFractionPct: 54,
      venaContractaCm: 0.74,
    }),
    'severe',
  );

  // Flow field produces a real unclipped hemispheric PISA convergence zone on the LV side at mid-systole
  const c = loadCaseById('mvp-primary-mr');
  const { heart, thorax, tables } = buildCaseModels(c, REST_PATIENT);
  const flow = buildFlowParams(c, heart, tables, thorax);
  const sysPhase =
    (0.5 * (tables.timings.ejectionStartS + tables.timings.ejectionEndS)) / tables.rrS;
  const hp = computeHeartPose(heart, cycleStateAt(tables, sysPhase, flow.flowPeaks));
  const dirX = Math.sin(flow.mrJetDirRad);
  const dirZ = -Math.cos(flow.mrJetDirRad);
  const ox = flow.mvCenter.x;
  const oy = flow.mvCenter.y;
  const oz = hp.zAnn - 0.1;

  // Sample 1.0 cm upstream of the mitral orifice along the LV convergence axis (inside PISA hemisphere)
  const pisaSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };
  sampleRegurgitantJets(
    flow,
    tables,
    hp,
    sysPhase,
    ox - dirX * 1.0,
    oy,
    oz - dirZ * 1.0,
    pisaSample,
  );
  const speed1cm = Math.hypot(pisaSample.vx, pisaSample.vy, pisaSample.vz);
  assert.equal(pisaSample.present, 1, 'PISA convergence should be present 1.0 cm upstream');
  assert.ok(speed1cm > 0.3 && speed1cm < 1.2, `expected PISA velocity ~0.4–0.6 m/s at 1 cm, got ${speed1cm}`);
  assert.ok(pisaSample.dispersion <= 0.1, 'proximal PISA convergence should remain laminar');
});

test('ASE 2025 sinus-rhythm LAP tie-breaker uses PV S/D < 1 in non-MR ties and stays indeterminate when LA/PV is confounded by MR', () => {
  // Missing TR velocity (null) -> only E/e' (>14, positive) and LAVI (<=34, negative) available -> 1-1 tie
  const baseTie = {
    ageYears: 62,
    rhythm: 'sinus',
    eMps: 0.95,
    eOverA: 1.2,
    septalEPrimeCmps: 5.5,
    lateralEPrimeCmps: 6.5, // avg e' = 6 -> E/e' = 15.8 > 14 (positive)
    laviMlM2: 31, // <= 34 (negative)
    laSizeReliable: true,
    trMps: null,
    paspMmHg: null,
    decelerationTimeMs: 190,
  };
  const elevatedByPv = assessDiastolic({ ...baseTie, pulmonaryVeinSd: 0.75 });
  assert.equal(elevatedByPv.fillingPressure, 'elevated');
  assert.ok(elevatedByPv.pressureEvidence.includes('pulmonary-vein-sd'));

  const normalByPv = assessDiastolic({ ...baseTie, pulmonaryVeinSd: 1.3 });
  assert.equal(normalByPv.fillingPressure, 'normal');

  // When LA size is unreliable (e.g. severe MR), a 1-1 tie between E/e' and TR must NOT use MR-blunted PV S/D
  const mrTie = assessDiastolic({
    ...baseTie,
    laSizeReliable: false,
    trMps: 2.5, // negative
    paspMmHg: 30,
    pulmonaryVeinSd: 0.4, // blunted by MR jet, not isolated LAP
  });
  assert.equal(mrTie.fillingPressure, 'indeterminate');
});

