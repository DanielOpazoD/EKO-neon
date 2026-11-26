import assert from 'node:assert/strict';
import { beforeEach, describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import type { SimOutput } from '@/simulator/core/protocol';
import { computeGroundTruth } from '@/simulator/hemodynamics/groundTruth';
import type { Measurement } from '@/simulator/measurements/types';
import { examClosed } from './modePolicy';
import { useHudStore, useSimStore } from './store';

const get = () => useSimStore.getState();

function vmax(id: string, value: number): Measurement {
  return {
    id,
    kind: 'velocity',
    label: 'Velocidad máxima aórtica',
    measurementId: 'av-vmax',
    technique: null,
    value,
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
}

/** An exam on the severe stenosis with one measurement, finished with the next one's tool still armed. */
function finishedExam() {
  get().setMode('exam', 'aortic-stenosis-severe');
  get().setTruth(computeGroundTruth(loadCaseById('aortic-stenosis-severe')));
  get().setModality('cw');
  get().addMeasurement(vmax('before', 4.4));
  get().setActiveMeasurement('av-vti');
  get().finishExam();
}

describe('a finished exam is closed, as its confirmation says', () => {
  beforeEach(() => {
    get().setMode('sandbox');
    get().loadCase('normal-excellent-window');
    get().setModality('2d');
    useHudStore.getState().setHud(null);
  });

  it('keeps the frozen image it was finished on: neither a resume nor a modality acquires', () => {
    finishedExam();
    assert.equal(examClosed(get()), true);
    assert.equal(get().frozen, true);
    get().toggleFreeze(); // Space, the freeze button, a measurement click on the live image
    assert.equal(get().frozen, true);
    get().setModality('pw'); // the modality keys and buttons
    assert.equal(get().modality, 'cw');
    assert.equal(get().frozen, true);
    // its cine is still there to review
    useHudStore.getState().setHud({ cineLength: 10 } as SimOutput);
    get().setCineOffset(-3);
    assert.equal(get().cineOffset, -3);
  });

  it('keeps the measurements it was scored on, and arms no tool over them', () => {
    finishedExam();
    assert.equal(get().activeTool, 'none', 'the tool armed when finishing is put down');
    assert.equal(get().activeMeasurementId, null);
    get().setActiveTool('caliper');
    assert.equal(get().activeTool, 'none');
    get().setActiveMeasurement('av-vmax');
    assert.equal(get().activeTool, 'none');
    assert.equal(get().activeMeasurementId, null);
    get().addMeasurement(vmax('after', 4.6));
    get().removeMeasurement('before');
    get().clearMeasurements();
    assert.deepEqual(get().measurements.map((m) => m.id), ['before']);
    const row = get().examResult?.measurements.rows.find((r) => r.measurementId === 'av-vmax');
    assert.equal(row?.measured, 4.4, 'the list is the one the recorded score was computed on');
  });

  it('closes nothing before the finish, and a mode change reopens the case for practice', () => {
    get().setMode('exam', 'aortic-stenosis-severe');
    get().toggleFreeze();
    assert.equal(get().frozen, true);
    get().toggleFreeze();
    assert.equal(get().frozen, false, 'the running exam freezes and resumes');
    get().finishExam();
    get().setMode('sandbox');
    assert.equal(examClosed(get()), false);
    get().toggleFreeze();
    assert.equal(get().frozen, false);
    get().addMeasurement(vmax('practice', 4.5));
    assert.ok(get().measurements.some((m) => m.id === 'practice'));
  });

  it('is never closed outside the exam', () => {
    assert.equal(examClosed({ mode: 'sandbox', examFinished: true }), false);
    assert.equal(examClosed({ mode: 'guided', examFinished: true }), false);
    assert.equal(examClosed({ mode: 'exam', examFinished: false }), false);
    assert.equal(examClosed({ mode: 'exam', examFinished: true }), true);
  });
});
