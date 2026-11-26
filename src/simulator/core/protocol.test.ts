import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { baseInput } from './baseInput';
import { frameTransferList } from './protocol';
import { SimulatorCore } from './simulatorCore';

describe('Worker protocol conditional structure/segment transfer', () => {
  it('omits structure and segment copies on live frames when disabled, and restores them when enabled or frozen', () => {
    const caseDef = loadCaseById('normal-excellent-window');
    const inp = baseInput();
    inp.rendererBackend = 'procedural';
    inp.quality = 'low';
    inp.needSegments = false;
    inp.needStructure = false;

    const core = new SimulatorCore(caseDef, inp);
    const liveSkip = core.step(0.05);
    assert.ok(liveSkip !== null);
    assert.equal(liveSkip.segment.length, 0);
    assert.equal(liveSkip.structure.length, 0);
    const skipTransfers = frameTransferList(liveSkip);
    assert.ok(!skipTransfers.includes(liveSkip.segment.buffer));
    assert.ok(!skipTransfers.includes(liveSkip.structure.buffer));
    assert.ok(skipTransfers.includes(liveSkip.ecg.buffer));

    // Enabling needSegments + needStructure ships populated polar maps
    core.setInput({ ...inp, needSegments: true, needStructure: true });
    const liveFull = core.step(0.05);
    assert.ok(liveFull !== null);
    const expectedLen = liveFull.polar.lines * liveFull.polar.samples;
    assert.equal(liveFull.segment.length, expectedLen);
    assert.equal(liveFull.structure.length, expectedLen);
    const fullTransfers = frameTransferList(liveFull);
    assert.ok(fullTransfers.includes(liveFull.segment.buffer));
    assert.ok(fullTransfers.includes(liveFull.structure.buffer));

    // Frozen frames always include structure and segment maps so post-freeze tools work immediately
    core.setInput({ ...inp, needSegments: false, needStructure: false, frozen: true });
    const frozenOut = core.step(0.05);
    assert.ok(frozenOut !== null);
    assert.equal(frozenOut.structure.length, expectedLen);
    assert.equal(frozenOut.segment.length, expectedLen);

    core.dispose();
  });
});

describe('Session identity of the frames', () => {
  it('stamps case and session on each frame; a new session drops the last one’s view and cine', () => {
    const caseDef = loadCaseById('normal-excellent-window');
    const inp = baseInput({ rendererBackend: 'procedural', quality: 'low', sessionEpoch: 3 });
    const core = new SimulatorCore(caseDef, inp);
    let practice = core.step(0.05);
    for (let i = 0; i < 6; i++) practice = core.step(0.05) ?? practice;
    assert.ok(practice);
    assert.equal(practice.acquisition?.caseId, 'normal-excellent-window');
    assert.equal(practice.acquisition?.sessionEpoch, 3);
    assert.ok(practice.view, 'the practice frames carry their view analysis');
    assert.ok(practice.cineLength > 1);

    // an exam begins on the same case: a new session, same pose and controls
    core.setInput({ ...inp, sessionEpoch: 4 });
    const exam = core.step(0.05);
    assert.ok(exam);
    assert.equal(exam.acquisition?.sessionEpoch, 4);
    assert.notEqual(exam.view, practice.view, 'the view attached is analysed in the new session');
    assert.equal(exam.cineLength, 1, 'the practice frames are not in the exam’s cine to scroll back to');
    core.dispose();
  });
});
