import test from 'node:test';
import assert from 'node:assert/strict';
import { CASE_INPUTS, loadCaseById } from '@/cases';
import { buildCaseModels, REST_PATIENT } from '@/simulator/anatomy/caseModels';
import { measureModel } from '@/simulator/anatomy/measureModel';
import { computeGroundTruth, type StructuredEchoTruth } from './groundTruth';
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
