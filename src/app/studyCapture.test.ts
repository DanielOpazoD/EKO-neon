import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { MAX_STUDY_CAPTURES } from './store/educationSlice';
import { useSimStore, type StudyCapture } from './store';
import { formatCaptureCaption } from './studyCapture';

function mkCapture(id: string, overrides: Partial<StudyCapture> = {}): StudyCapture {
  return {
    id,
    dataUrl: 'data:image/jpeg;base64,/9j/4AAQSkZJRg==',
    modality: '2d',
    viewId: 'plax',
    viewScore: 91,
    phase: 0.12,
    frameId: 42,
    frozen: true,
    createdAt: '2026-01-01T00:00:00Z',
    ...overrides,
  };
}

describe('Study capture store & exam privacy', () => {
  it('hides recognized view name in captions during an active exam and reveals it once finished', () => {
    const cap = mkCapture('c1', { modality: 'color', viewId: 'a5c' });
    assert.equal(formatCaptureCaption(cap, false), 'Color');
    assert.ok(formatCaptureCaption(cap, true).includes('A5C'));
    assert.ok(formatCaptureCaption(cap, true).includes('Color'));
    assert.equal(formatCaptureCaption(mkCapture('c2', { modality: 'pw', viewId: null }), true), 'PW');
  });

  it('stores, caps FIFO at MAX_STUDY_CAPTURES, clears on case load, and freezes after exam completion', () => {
    const st = useSimStore.getState();
    st.setMode('sandbox');
    st.loadCase('normal-excellent-window');
    assert.equal(useSimStore.getState().captures.length, 0);

    // FIFO cap at MAX_STUDY_CAPTURES
    for (let i = 0; i < MAX_STUDY_CAPTURES + 3; i++) {
      useSimStore.getState().addCapture(mkCapture(`cap-${i}`));
    }
    const stored = useSimStore.getState().captures;
    assert.equal(stored.length, MAX_STUDY_CAPTURES);
    assert.equal(stored[0]?.id, 'cap-3');
    assert.equal(stored[stored.length - 1]?.id, `cap-${MAX_STUDY_CAPTURES + 2}`);

    // Remove individual capture
    useSimStore.getState().removeCapture('cap-3');
    assert.equal(useSimStore.getState().captures.length, MAX_STUDY_CAPTURES - 1);

    // Fresh case load wipes prior session captures
    useSimStore.getState().loadCase('aortic-stenosis-severe');
    assert.equal(useSimStore.getState().captures.length, 0);

    // Exam mode: captures allowed during exam, locked once finished
    useSimStore.getState().setMode('exam', 'normal-excellent-window');
    useSimStore.getState().addCapture(mkCapture('exam-cap-1', { viewId: 'a4c' }));
    assert.equal(useSimStore.getState().captures.length, 1);

    useSimStore.getState().finishExam();
    useSimStore.getState().addCapture(mkCapture('exam-cap-2'));
    useSimStore.getState().removeCapture('exam-cap-1');
    useSimStore.getState().clearCaptures();
    assert.equal(useSimStore.getState().captures.length, 1);
    assert.equal(useSimStore.getState().captures[0]?.id, 'exam-cap-1');

    // Clean up store back to sandbox
    useSimStore.getState().setMode('sandbox');
    useSimStore.getState().loadCase('normal-excellent-window');
  });
});
