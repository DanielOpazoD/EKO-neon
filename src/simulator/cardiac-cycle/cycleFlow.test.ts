import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { baseInput } from '@/simulator/core/baseInput';
import { SimulatorCore } from '@/simulator/core/simulatorCore';
import { computeGroundTruth } from '@/simulator/hemodynamics/groundTruth';
import { cycleStateAt, valveEventTimes } from './cycleModel';

describe('Cardiac cycle model & Doppler hemodynamic physiology', () => {
  it('orders phase landmarks and valve events physiologically, and suppresses A-wave in atrial fibrillation', () => {
    const inp = baseInput();
    inp.rendererBackend = 'procedural';
    inp.quality = 'low';

    const normalCore = new SimulatorCore(loadCaseById('normal-excellent-window'), inp);
    const normalMarks = normalCore.phaseMarks();
    const normalValves = valveEventTimes(normalCore.models.tables);

    assert.ok(normalMarks.ejectionStart > 0 && normalMarks.ejectionStart < normalMarks.ejectionEnd);
    assert.ok(normalMarks.ejectionEnd < normalMarks.mitralOpen);
    assert.ok(normalMarks.mitralOpen < normalMarks.eEnd);
    assert.ok(normalMarks.eEnd <= normalMarks.aStart);
    assert.ok(normalMarks.aStart < normalMarks.aEnd);
    assert.equal(normalMarks.hasAWave, true);
    assert.ok(normalValves.aortic[0] < normalValves.aortic[1]);
    assert.ok(normalValves.mitral[0] > normalValves.aortic[1]);
    normalCore.dispose();

    const afCore = new SimulatorCore(loadCaseById('af-diastolic'), inp);
    const afMarks = afCore.phaseMarks();
    assert.equal(afMarks.hasAWave, false);
    afCore.dispose();
  });

  it('contracts LV volume from end-diastole to end-systole with distinct EF in normal vs HFrEF', () => {
    const inp = baseInput();
    inp.rendererBackend = 'procedural';
    inp.quality = 'low';

    const normalCore = new SimulatorCore(loadCaseById('normal-excellent-window'), inp);
    const normalMarks = normalCore.phaseMarks();
    const edNormal = cycleStateAt(normalCore.models.tables, 0);
    const esNormal = cycleStateAt(normalCore.models.tables, normalMarks.endSystole);
    assert.ok(edNormal.lvVolumeMl > esNormal.lvVolumeMl);
    const efNormal = ((edNormal.lvVolumeMl - esNormal.lvVolumeMl) / edNormal.lvVolumeMl) * 100;
    assert.ok(efNormal >= 52 && efNormal <= 74, `Unexpected normal EF: ${efNormal.toFixed(1)}%`);
    normalCore.dispose();

    const hfCore = new SimulatorCore(loadCaseById('hfref-severe-mr'), inp);
    const hfMarks = hfCore.phaseMarks();
    const edHf = cycleStateAt(hfCore.models.tables, 0);
    const esHf = cycleStateAt(hfCore.models.tables, hfMarks.endSystole);
    const efHf = ((edHf.lvVolumeMl - esHf.lvVolumeMl) / edHf.lvVolumeMl) * 100;
    assert.ok(efHf < 36, `Expected reduced EF (<36%) in HFrEF, got ${efHf.toFixed(1)}%`);
    assert.ok(edHf.lvVolumeMl > edNormal.lvVolumeMl, 'HFrEF LV EDV should be dilated vs normal');
    hfCore.dispose();
  });

  it('differentiates pathological hemodynamic jets across severe AS, primary MR, and obstructive HOCM', () => {
    const normalTruth = computeGroundTruth(loadCaseById('normal-excellent-window'));
    const asTruth = computeGroundTruth(loadCaseById('aortic-stenosis-severe'));
    const mrTruth = computeGroundTruth(loadCaseById('mvp-primary-mr'));
    const hocmTruth = computeGroundTruth(loadCaseById('hocm-sam'));

    // Severe AS: Vmax >= 4.0 m/s, mean gradient >= 40 mmHg, AVA <= 1.0 cm2
    assert.ok(normalTruth.aorticValve.vmaxMps < 1.8);
    assert.ok(
      asTruth.aorticValve.vmaxMps >= 4.0,
      `Expected severe AS Vmax >= 4.0, got ${asTruth.aorticValve.vmaxMps}`,
    );
    assert.ok(asTruth.aorticValve.meanGradientMmHg >= 40);
    assert.ok(asTruth.aorticValve.continuityAvaCm2 <= 1.0);

    // Primary severe MR: high-velocity regurgitant jet and large EROA
    assert.ok(mrTruth.regurgitation.mr !== null);
    assert.ok(mrTruth.regurgitation.mr.eroaCm2 >= 0.4);
    assert.ok(mrTruth.regurgitation.mr.regurgitantVolumeMl >= 60);

    // Obstructive HOCM: dynamic LVOT obstruction with peak gradient >= 30 mmHg and asymmetric septal hypertrophy
    assert.equal(hocmTruth.lvot.dynamicObstruction, true);
    assert.ok(hocmTruth.lvot.peakGradientMmHg >= 30);
    assert.ok(hocmTruth.lv.ivsdCm >= 1.5);
  });
});
