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
