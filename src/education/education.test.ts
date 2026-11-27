import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { CASE_INPUTS, loadCaseById } from '@/cases';
import { Structure } from '@/simulator/anatomy/tissue';
import { computeGroundTruth } from '@/simulator/hemodynamics/groundTruth';
import { getMeasurementSpec } from '@/simulator/measurements/protocol';
import type { Measurement } from '@/simulator/measurements/types';
import { getViewTarget } from '@/simulator/windows/viewTargets';
import { allTasks, evaluateTasks, isUnlocked, type LearnerSnapshot } from './curriculum';
import { subcostalPostureHint } from './causes';
import { expectedFindings, scoreImpression } from './impression';
import { loadProgress, PROGRESS_KEY, summarizeProgress } from './progress';
import { biplaneVolume, deriveCalculations, pathologyImpressions, rightAtrialPressure } from './report';
import { truthFor } from './scoring/scoring';
import { evaluateTechnique, structureName } from './technique';

function ellipseContour(cx: number, cy: number, rx: number, ry: number, n = 48) {
  return Array.from({ length: n }, (_, i) => {
    const a = (i / n) * Math.PI * 2;
    return { x: cx + rx * Math.cos(a), y: cy + ry * Math.sin(a) };
  });
}

function fakeSimpsonMeasurement(
  id: string,
  view: 'a4c' | 'a2c',
  rxCm: number,
  ryCm: number,
): Measurement {
  const pxPerCm = 20;
  return {
    id: `${id}-${view}`,
    measurementId: id,
    kind: 'volume',
    label: id,
    value: 100,
    units: 'mL',
    modality: '2d',
    createdAt: '2026-01-01T00:00:00Z',
    frameId: 1,
    phase: 0,
    timeS: 0,
    sourceViewId: view,
    viewScore: 95,
    imageQualityScore: 95,
    userAssisted: false,
    referenceGuidelineIds: ['ase-tte-2019'],
    geometry: ellipseContour(200, 200, rxCm * pxPerCm, ryCm * pxPerCm),
    captureSector: {
      width: 400,
      height: 400,
      apexX: 200,
      apexY: 10,
      pxPerCm,
      depthCm: 16,
      sectorRad: 1.4,
      invertLR: false,
    },
  };
}

describe('Educational grading, ASE clinical rules & curriculum', () => {
  it('estimates right atrial pressure from IVC diameter and inspiratory collapse per ASE guidelines', () => {
    // <= 2.1 cm & > 50% collapse => 3 mmHg
    assert.equal(rightAtrialPressure(1.8, 0.6).rapMmHg, 3);
    // > 2.1 cm & < 50% collapse => 15 mmHg
    assert.equal(rightAtrialPressure(2.4, 2.0).rapMmHg, 15);
    // Discordant (> 2.1 cm but > 50% collapse) => intermediate 8 mmHg
    assert.equal(rightAtrialPressure(2.4, 1.0).rapMmHg, 8);
    // Exact 50% collapse boundary => intermediate 8 mmHg
    assert.equal(rightAtrialPressure(1.8, 0.9).rapMmHg, 8);
    // Missing inspiratory or expiratory IVC => intermediate 8 mmHg
    assert.equal(rightAtrialPressure(1.9, null).rapMmHg, 8);
    assert.equal(rightAtrialPressure(null, null).rapMmHg, 8);
  });

  it('derives accurate expected clinical findings and F1 impression scores across key pathologies', () => {
    const normalExp = expectedFindings(computeGroundTruth(loadCaseById('normal-excellent-window')));
    assert.ok(normalExp.includes('normal-study'));
    assert.ok(normalExp.includes('ef-normal'));

    const asExp = expectedFindings(computeGroundTruth(loadCaseById('aortic-stenosis-severe')));
    assert.ok(asExp.includes('as-severe'));
    assert.ok(!asExp.includes('normal-study'));

    const hfExp = expectedFindings(computeGroundTruth(loadCaseById('hfref-severe-mr')));
    assert.ok(hfExp.includes('lv-dilated'));
    assert.ok(hfExp.some((f) => f === 'ef-moderate' || f === 'ef-severe'));

    const tampTruth = computeGroundTruth(loadCaseById('pericardial-effusion-tamponade'));
    const tampExp = expectedFindings(tampTruth);
    assert.ok(tampExp.includes('effusion'));
    assert.ok(tampExp.includes('tamponade'));
    assert.ok(pathologyImpressions(tampTruth).length > 0);

    // F1 scoring: exact = 100, disjoint = 0, partial with 1 wrong = exact harmonic mean
    assert.equal(scoreImpression(asExp, asExp).score, 100);
    assert.equal(scoreImpression(['tamponade'], normalExp).score, 0);
    const partial = scoreImpression(['a', 'b', 'x'], ['a', 'b', 'c', 'd']);
    // precision = 2/3, recall = 2/4 = 0.5 => F1 = 2*(2/3*0.5)/(2/3+0.5) = 4/7 ≈ 57
    assert.equal(partial.score, 57);
    assert.deepEqual(partial.correct, ['a', 'b']);
    assert.deepEqual(partial.wrong, ['x']);
    assert.deepEqual(partial.missed, ['c', 'd']);
  });

  it('computes biplane Simpson volume from A4C + A2C contours and gates curriculum tasks on prerequisites', () => {
    // Prolate ellipsoid with a=4 cm (long axis 8 cm), b=2.5 cm (A4C), c=2.0 cm (A2C)
    // Theoretical volume = (4/3) * pi * a * b * c = (4/3) * pi * 4 * 2.5 * 2.0 ≈ 83.78 mL
    const m4 = fakeSimpsonMeasurement('lv-edv', 'a4c', 2.5, 4.0);
    const m2 = fakeSimpsonMeasurement('lv-edv', 'a2c', 2.0, 4.0);
    assert.equal(biplaneVolume([m4], 'lv-edv'), null, 'Needs both A4C and A2C traces');
    const bp = biplaneVolume([m4, m2], 'lv-edv');
    assert.ok(bp !== null);
    const expectedMl = (4 / 3) * Math.PI * 4.0 * 2.5 * 2.0;
    assert.ok(
      Math.abs(bp.volumeMl - expectedMl) / expectedMl < 0.03,
      `Expected biplane volume ~${expectedMl.toFixed(1)} mL, got ${bp.volumeMl.toFixed(1)} mL`,
    );

    const tasks = allTasks();
    assert.ok(tasks.length >= 10, 'Expected populated curriculum task catalogue');
    const rootTask = tasks.find((t) => !t.requires || t.requires.length === 0);
    const gatedTask = tasks.find((t) => t.requires && t.requires.length > 0);
    assert.ok(rootTask, 'Expected at least one root curriculum task');
    assert.equal(isUnlocked(rootTask, {}), true);
    if (gatedTask) {
      assert.equal(isUnlocked(gatedTask, {}), false);
      const completed = Object.fromEntries(gatedTask.requires!.map((id) => [id, true]));
      assert.equal(isUnlocked(gatedTask, completed), true);
    }
  });

  it('recovers valid learner progress while individually dropping malformed events from localStorage', () => {
    const store = new Map<string, string>();
    const storage = {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => {
        store.set(k, v);
      },
    };

    // Corrupt JSON -> empty fallback
    store.set(PROGRESS_KEY, '{not-json');
    assert.deepEqual(loadProgress(storage), { version: 1, completedTasks: {}, events: [] });

    // Mixed valid and malformed events -> only valid events survive
    store.set(
      PROGRESS_KEY,
      JSON.stringify({
        version: 1,
        completedTasks: { 'task-1': 1700000000000 },
        events: [
          { t: 100, kind: 'view', caseId: 'normal-excellent', viewId: 'plax', score: 92 },
          { t: 101, kind: 'view', caseId: 'normal-excellent', viewId: 'plax', score: 'bad' },
          { t: 102, kind: 'measurement', caseId: 'normal-excellent', measurementId: 'lv-edd', techniqueScore: null, value: 4.8 },
          { t: 103, kind: 'exam', caseId: 'aortic-stenosis-severe', total: 88, acquisition: 90, measurements: 85, impression: null },
          { t: 104, kind: 'unknown-legacy-kind', payload: 123 },
          null,
        ],
      }),
    );

    const loaded = loadProgress(storage);
    assert.equal(loaded.events.length, 3);
    assert.deepEqual(loaded.completedTasks, { 'task-1': 1700000000000 });
    const summary = summarizeProgress(loaded);
    assert.equal(summary.bestViewScores.plax, 92);
    assert.equal(summary.measurementsCount, 1);
    assert.equal(summary.exams.length, 1);
    assert.equal(summary.completedTasks, 1);
  });

  it('enforces Color-only modality, jet caliper rules, aortic blood-pool structures, and regurgitation protocol truths', () => {
    const marks = {
      ejectionStart: 0.08,
      ejectionEnd: 0.38,
      mitralOpen: 0.45,
      eEnd: 0.68,
      aStart: 0.82,
      aEnd: 0.96,
      hasAWave: true,
    };

    // 1. Color-only vena contracta rejects plain 2D and passes on systolic Color over the mitral coaptation zone
    const vcSpec = getMeasurementSpec('mr-vena-contracta')!;
    assert.ok(vcSpec);
    const vcOn2d = evaluateTechnique(vcSpec, {
      modality: '2d',
      viewId: 'plax',
      viewScore: 90,
      phase: 0.2,
      phaseMarks: marks,
      gate: null,
      segmentStructures: [Structure.LaCavity, Structure.MitralAnterior, Structure.LaCavity, Structure.MitralPosterior],
      segmentEndsOutside: [Structure.LaCavity, Structure.LaCavity],
    });
    assert.ok(
      vcOn2d.findings.some((f) => f.code === 'modality' && f.level === 'invalid'),
      'Color-only vena contracta must reject plain 2D',
    );

    const vcOnColor = evaluateTechnique(vcSpec, {
      modality: 'color',
      viewId: 'plax',
      viewScore: 90,
      phase: 0.2,
      phaseMarks: marks,
      gate: null,
      segmentStructures: [Structure.LaCavity, Structure.MitralAnterior, Structure.LaCavity, Structure.MitralPosterior],
      segmentEndsOutside: [Structure.LaCavity, Structure.LaCavity],
    });
    assert.equal(vcOnColor.score, 1, `Expected perfect technique score on systolic Color VC, got ${JSON.stringify(vcOnColor.findings)}`);

    // 2. 2D caliper measurements still accept Color mode and recognise AscendingAorta/AorticArch as blood pool
    const rootSpec = getMeasurementSpec('aortic-root')!;
    const rootInColor = evaluateTechnique(rootSpec, {
      modality: 'color',
      viewId: 'plax',
      viewScore: 88,
      phase: 0.0,
      phaseMarks: marks,
      gate: null,
      segmentStructures: [
        Structure.AorticRoot,
        Structure.AorticRoot,
        Structure.AorticRoot,
        Structure.AscendingAorta,
        Structure.AorticRoot,
      ],
      segmentEndsOutside: [Structure.Pericardium, Structure.LaWall],
    });
    assert.equal(
      rootInColor.score,
      1,
      `AscendingAorta blood samples should not trigger wall warning: ${JSON.stringify(rootInColor.findings)}`,
    );
    assert.equal(structureName(Structure.AscendingAorta), 'aorta ascendente');
    assert.equal(structureName(Structure.AorticArch), 'arco aórtico');
    assert.equal(structureName(Structure.BrachiocephalicArtery), 'tronco braquiocefálico');

    // 3. Expanded clinical views & structures: subcostal-4c for TR Vmax and AscendingAorta for AV CW
    const trSpec = getMeasurementSpec('tr-vmax')!;
    assert.ok(trSpec.views.includes('subcostal-4c'));
    const avSpec = getMeasurementSpec('av-vmax')!;
    assert.ok(avSpec.placement?.structures.includes(Structure.AscendingAorta));

    // 4. Regurgitation specs resolve cleanly from StructuredEchoTruth (non-null on MVP, null on normal)
    const mrTruth = computeGroundTruth(loadCaseById('mvp-primary-mr'));
    const normTruth = computeGroundTruth(loadCaseById('normal-excellent-window'));
    const mrVmaxT = truthFor('mr-vmax', mrTruth);
    const mrVtiT = truthFor('mr-vti', mrTruth);
    const mrVcT = truthFor('mr-vena-contracta', mrTruth);
    assert.ok(mrVmaxT && mrVmaxT.value > 4.0, `Expected MVP MR Vmax > 4 m/s, got ${mrVmaxT?.value}`);
    assert.ok(mrVtiT && mrVtiT.value > 80, `Expected MVP MR VTI > 80 cm, got ${mrVtiT?.value}`);
    assert.ok(mrVcT && mrVcT.value >= 0.6, `Expected MVP MR VC >= 0.6 cm, got ${mrVcT?.value}`);
    assert.equal(truthFor('mr-vmax', normTruth), null);
    assert.equal(truthFor('mr-vena-contracta', normTruth), null);
    assert.equal(truthFor('ar-vmax', normTruth), null);

    // 5. Derived MR peak gradient, PISA EROA/RVol, MVA by PHT, and PV S/D ratio
    const baseMeas = {
      createdAt: '2026-01-01T00:00:00Z',
      frameId: 1,
      phase: 0.2,
      timeS: 0.2,
      sourceViewId: 'a4c',
      viewScore: 92,
      imageQualityScore: 90,
      userAssisted: false,
      referenceGuidelineIds: ['ase-scmr-regurgitation-2017'],
      geometry: [],
    };
    const derived = deriveCalculations([
      {
        ...baseMeas,
        id: 'm-mr',
        measurementId: 'mr-vmax',
        kind: 'velocity',
        label: 'Vmax IM',
        value: 5.0,
        units: 'm/s',
        modality: 'cw',
      },
      {
        ...baseMeas,
        id: 'm-mr-vti',
        measurementId: 'mr-vti',
        kind: 'vti',
        label: 'VTI IM',
        value: 140,
        units: 'cm',
        modality: 'cw',
      },
      {
        ...baseMeas,
        id: 'm-pisa',
        measurementId: 'mr-pisa-radius',
        kind: 'linear',
        label: 'PISA IM',
        value: 0.9,
        units: 'cm',
        modality: 'color',
        derived: { aliasingVelocityMps: 0.4 },
      },
      {
        ...baseMeas,
        id: 'm-pht',
        measurementId: 'mitral-pht',
        kind: 'time',
        label: 'PHT mitral',
        value: 200,
        units: 'ms',
        modality: 'cw',
      },
      {
        ...baseMeas,
        id: 'm-pvs',
        measurementId: 'pv-s',
        kind: 'velocity',
        label: 'VP S',
        value: 0.6,
        units: 'm/s',
        modality: 'pw',
      },
      {
        ...baseMeas,
        id: 'm-pvd',
        measurementId: 'pv-d',
        kind: 'velocity',
        label: 'VP D',
        value: 0.4,
        units: 'm/s',
        modality: 'pw',
      },
    ]);
    const mrGrad = derived.find((d) => d.id === 'mr-peak-gradient');
    assert.ok(mrGrad && mrGrad.value === '100 mmHg', `Expected 100 mmHg MR gradient, got ${mrGrad?.value}`);
    const mrEroa = derived.find((d) => d.id === 'mr-eroa-pisa');
    // 2 * pi * 0.9^2 * 40 / 500 = 0.407 cm² -> 0.41 cm²
    assert.ok(mrEroa && mrEroa.value === '0.41 cm²', `Expected 0.41 cm² PISA EROA, got ${mrEroa?.value}`);
    const mrRvol = derived.find((d) => d.id === 'mr-rvol-pisa');
    // 0.40715 * 140 = 57.0 mL
    assert.ok(mrRvol && mrRvol.value === '57 mL', `Expected 57 mL PISA RVol, got ${mrRvol?.value}`);
    const mva = derived.find((d) => d.id === 'mva-pht');
    assert.ok(mva && mva.value === '1.10 cm²', `Expected 1.10 cm² MVA, got ${mva?.value}`);
    const pvSd = derived.find((d) => d.id === 'pv-sd-ratio');
    assert.ok(pvSd && pvSd.value === '1.50', `Expected 1.50 PV S/D ratio, got ${pvSd?.value}`);

    // Truth resolution for new protocol specs
    const mrPisaT = truthFor('mr-pisa-radius', mrTruth);
    assert.ok(mrPisaT && mrPisaT.value > 0.7 && mrPisaT.value < 1.3, `Expected MVP PISA@40 in (0.7, 1.3) cm, got ${mrPisaT?.value}`);
    const pvST = truthFor('pv-s', normTruth);
    const pvDT = truthFor('pv-d', normTruth);
    assert.ok(pvST && pvST.value > 0.3 && pvDT && pvDT.value > 0.25);
    const mitralPhtT = truthFor('mitral-pht', normTruth);
    assert.ok(mitralPhtT && mitralPhtT.value > 40 && mitralPhtT.value < 90);

    // 6. Non-intrusive subcostal posture hint appears only for subcostal views outside supine-subcostal
    assert.ok(subcostalPostureHint('subcostal-ivc', 'plax', 'left-lateral')?.includes('Supino (subcostal)'));
    assert.ok(subcostalPostureHint(null, 'subcostal-4c', 'supine')?.includes('Supino (subcostal)'));
    assert.equal(subcostalPostureHint('subcostal-ivc', 'subcostal-ivc', 'subcostal-supine'), null);
    assert.equal(subcostalPostureHint('plax', 'plax', 'left-lateral'), null);
  });

  it('validates curriculum graph integrity and unlocks/completes valvular and pulmonary venous tasks', () => {
    const tasks = allTasks();
    const ids = new Set(tasks.map((t) => t.id));
    assert.equal(ids.size, tasks.length, 'Curriculum task ids must be unique');
    const caseIds = new Set(CASE_INPUTS.map((c) => c.id));
    for (const t of tasks) {
      for (const req of t.requires ?? []) {
        assert.ok(ids.has(req), `Task ${t.id} requires unknown task ${req}`);
      }
      if (t.caseId) assert.ok(caseIds.has(t.caseId), `Task ${t.id} points to unknown case ${t.caseId}`);
      if (t.viewId) assert.ok(getViewTarget(t.viewId), `Task ${t.id} points to unknown view ${t.viewId}`);
    }

    const mkMeas = (measurementId: string, score = 0.9): Measurement => ({
      id: `m-${measurementId}`,
      measurementId,
      kind: 'velocity',
      label: measurementId,
      value: 1,
      units: 'm/s',
      modality: 'cw',
      createdAt: '2026-01-01T00:00:00Z',
      frameId: 1,
      phase: 0.2,
      timeS: 0.2,
      sourceViewId: 'a4c',
      viewScore: 90,
      imageQualityScore: 90,
      userAssisted: false,
      referenceGuidelineIds: ['ase-tte-2019'],
      geometry: [],
      technique: { score, findings: [] },
    });

    const baseSnap = (overrides: Partial<LearnerSnapshot>): LearnerSnapshot => ({
      caseId: 'normal-excellent-window',
      mode: 'guided',
      viewProgress: {},
      handViewProgress: {},
      bestView: null,
      modality: '2d',
      colorScaleMps: 0.62,
      colorGainDb: 0,
      gateStructure: null,
      gateFlowAngleDeg: null,
      measurements: [],
      impressionScore: null,
      settings: { depthCm: 16, gainDb: 0, frequencyMHz: 2.5, harmonics: true },
      ...overrides,
    });

    // 1. MR VC/PISA + CW task requires colour-normal-scale
    const mrSnap = baseSnap({
      caseId: 'mvp-primary-mr',
      measurements: [mkMeas('mr-pisa-radius'), mkMeas('mr-vmax')],
    });
    assert.deepEqual(evaluateTasks(mrSnap, {}), []);
    assert.ok(evaluateTasks(mrSnap, { 'colour-normal-scale': true }).includes('mr-vc-pisa'));

    // 2. Pulmonary venous S & D task requires mitral-e-ok and both waves with adequate technique
    const pvPartial = baseSnap({ measurements: [mkMeas('pv-s')] });
    assert.deepEqual(evaluateTasks(pvPartial, { 'mitral-e-ok': true }), []);
    const pvFull = baseSnap({ measurements: [mkMeas('pv-s'), mkMeas('pv-d')] });
    assert.ok(evaluateTasks(pvFull, { 'mitral-e-ok': true }).includes('pv-flow-ok'));

    // 3. Severe AR task requires cw-as-vmax, ar-pht/ar-vmax, and impression >= 70
    const arSnap = baseSnap({
      caseId: 'aortic-regurgitation-severe',
      measurements: [mkMeas('ar-pht')],
      impressionScore: 85,
    });
    assert.deepEqual(evaluateTasks(arSnap, {}), []);
    assert.ok(evaluateTasks(arSnap, { 'cw-as-vmax': true }).includes('impression-ar'));

    // 4. Severe MS task requires mitral-e-ok, mitral-pht, and impression >= 70
    const msSnap = baseSnap({
      caseId: 'mitral-stenosis-severe',
      measurements: [mkMeas('mitral-pht')],
      impressionScore: 80,
    });
    assert.deepEqual(evaluateTasks(msSnap, {}), []);
    assert.ok(evaluateTasks(msSnap, { 'mitral-e-ok': true }).includes('impression-ms'));
  });
});
