import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { exportCaseTag, modePolicy, screenInMode, showsCaseIdentity } from './modePolicy';

describe('Product mode policy & blind exam guards', () => {
  it('enables learning/dev aids in sandbox and guided modes, and locks them down in exam mode', () => {
    const sandbox = modePolicy('sandbox');
    const guided = modePolicy('guided');
    const exam = modePolicy('exam');

    for (const training of [sandbox, guided]) {
      assert.equal(training.showViewFeedback, true);
      assert.equal(training.presetsEnabled, true);
      assert.equal(training.hintsEnabled, true);
      assert.equal(training.devToolsAllowed, true);
      assert.equal(training.navigatorAnatomy, true);
      assert.equal(training.impressionRequired, false);
      assert.equal(training.assignsCase, false);
    }

    assert.equal(exam.showViewFeedback, false);
    assert.equal(exam.presetsEnabled, false);
    assert.equal(exam.hintsEnabled, false);
    assert.equal(exam.learningScreensEnabled, false);
    assert.equal(exam.devToolsAllowed, false);
    assert.equal(exam.freeMeasurementsScored, false);
    assert.equal(exam.navigatorAnatomy, false);
    assert.equal(exam.impressionRequired, true);
    assert.equal(exam.assignsCase, true);
  });

  it('hides case identity and export tags during an active exam and reveals them upon completion', () => {
    assert.equal(showsCaseIdentity('sandbox', false), true);
    assert.equal(showsCaseIdentity('guided', false), true);
    assert.equal(showsCaseIdentity('exam', false), false);
    assert.equal(showsCaseIdentity('exam', true), true);

    assert.equal(exportCaseTag('exam', false, 'severe-as'), 'examen');
    assert.equal(exportCaseTag('exam', true, 'severe-as'), 'severe-as');
    assert.equal(exportCaseTag('sandbox', false, 'severe-as'), 'severe-as');
  });

  it('never shows a learning screen in the exam: it gives way to the simulator', () => {
    for (const screen of ['references', 'curriculum', 'progress'] as const) {
      assert.equal(screenInMode(screen, 'exam'), 'simulator', screen);
      assert.equal(screenInMode(screen, 'sandbox'), screen);
      assert.equal(screenInMode(screen, 'guided'), screen);
    }
    assert.equal(screenInMode('report', 'exam'), 'report', 'the exam is finished from its report');
    assert.equal(screenInMode('simulator', 'exam'), 'simulator');
  });
});
