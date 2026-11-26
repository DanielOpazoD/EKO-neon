import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { computeGroundTruth } from '@/simulator/hemodynamics/groundTruth';
import type { Measurement } from '@/simulator/measurements/types';
import {
  examSummaryOf,
  impressionCorrected,
  showsScoredTruth,
  type ExamInputs,
} from './examSummary';

describe('the score table of the report and the answer key (decision 257)', () => {
  const caseId = 'aortic-stenosis-severe';
  const truth = computeGroundTruth(loadCaseById(caseId));
  const practice: ExamInputs = {
    caseId,
    mode: 'sandbox',
    truth,
    viewProgress: {},
    measurements: [],
    impressionSelection: [],
    impressionChecked: false,
  };
  const shown = (s: ExamInputs, examFinished = false) => {
    const corrected = impressionCorrected({ ...s, examFinished });
    return examSummaryOf(s)!
      .measurements.rows.filter((r) => showsScoredTruth(r, corrected))
      .map((r) => r.measurementId);
  };

  it('keeps the model value of a required measurement not yet taken until «Corregir»', () => {
    const rows = examSummaryOf(practice)!.measurements.rows;
    assert.ok(rows.length >= 3, 'the stenosis requires its LVOT and aortic measurements');
    assert.ok(rows.some((r) => r.measurementId === 'av-vmax'));
    // the table gave the severe stenosis its Vmax and VTIs before anything was measured
    assert.deepEqual(shown(practice), []);
    assert.deepEqual(
      shown({ ...practice, impressionChecked: true }),
      rows.map((r) => r.measurementId),
    );
  });

  it('compares a value the learner measured at once, as the «Modelo (verdad)» column does', () => {
    const vmax: Measurement = {
      id: 'm1',
      kind: 'velocity',
      label: 'Velocidad máxima aórtica',
      measurementId: 'av-vmax',
      technique: null,
      value: 4.3,
      units: 'm/s',
      modality: 'cw',
      sourceViewId: null,
      viewScore: 90,
      frameId: 1,
      phase: 0,
      timeS: 0,
      geometry: [],
      imageQualityScore: null,
      userAssisted: false,
      referenceGuidelineIds: [],
      createdAt: '2026-01-01T00:00:00Z',
    };
    assert.deepEqual(shown({ ...practice, measurements: [vmax] }), ['av-vmax']);
  });

  it('shows the whole key in the exam once it is finished, the only time its table is shown', () => {
    const exam: ExamInputs = { ...practice, mode: 'exam' };
    const rows = examSummaryOf(exam)!.measurements.rows;
    assert.deepEqual(shown(exam, true), rows.map((r) => r.measurementId));
  });
});
