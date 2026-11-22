import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { computeGroundTruth } from '@/simulator/hemodynamics/groundTruth';
import type { Measurement } from '@/simulator/measurements/types';
import { allTasks, isUnlocked } from './curriculum';
import { expectedFindings, scoreImpression } from './impression';
import { loadProgress, PROGRESS_KEY, summarizeProgress } from './progress';
import { biplaneVolume, pathologyImpressions, rightAtrialPressure } from './report';

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
});
