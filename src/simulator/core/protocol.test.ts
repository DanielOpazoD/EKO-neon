import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { ProceduralSliceRenderer } from '@/simulator/renderer/procedural/sliceRenderer';
import type { Webgl2Renderer } from '@/simulator/renderer/gpu/webgl2Renderer';
import type {
  DisplayConsole,
  PolarFrame,
  PolarFrameSpec,
  RenderHints,
  Scene,
} from '@/simulator/renderer/types';
import type { BeamFrame } from '@/simulator/probe/pose';
import { controlThroughCrux, getViewTarget } from '@/simulator/windows/viewTargets';
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

  it('populates exact AHA segment maps on every GPU cine frame even when live needSegments is false', () => {
    const caseDef = loadCaseById('inferior-rwma');
    const cpuRef = new ProceduralSliceRenderer();
    let skippedReadbacks = 0;
    const fakeGpu = {
      id: 'webgl2' as const,
      contextLost: false,
      gbuffer: { ready: false },
      hasDeferredDoppler: false,
      stats: () => ({ gpu: 'webgl2' }),
      dispose: () => {},
      present: () => null,
      resetDopplerHistory: () => {},
      render: (
        scene: Scene,
        beam: BeamFrame,
        spec: PolarFrameSpec,
        phase: number,
        out: PolarFrame,
      ) => cpuRef.render(scene, beam, spec, phase, out),
      renderDisplay: (
        scene: Scene,
        beam: BeamFrame,
        spec: PolarFrameSpec,
        phase: number,
        out: PolarFrame,
        _hints: RenderHints | undefined,
        con: DisplayConsole,
        display: Uint8ClampedArray,
      ) => {
        // Mirror Webgl2Renderer.renderDisplay: populate structure & tissue, but leave segment zeroed when needSegments is false
        cpuRef.render(scene, beam, spec, phase, out);
        display.fill(128);
        if (con.needSegments === false) {
          out.segment.fill(0);
          skippedReadbacks++;
        }
        return true;
      },
    } as unknown as Webgl2Renderer;

    const inp = baseInput({
      rendererBackend: 'webgl2',
      quality: 'low',
      needSegments: false,
      needStructure: false,
    });
    const core = new SimulatorCore(caseDef, inp, {
      createGpu: () => ({ renderer: fakeGpu, reason: 'ok' }),
    });
    const a2cProbe = controlThroughCrux(
      getViewTarget('a2c'),
      core.models.heart,
      core.models.thorax,
      null,
    );
    const live = { ...inp, probe: a2cProbe };
    core.setInput(live);

    // Step enough ticks to capture at least 6 GPU cine frames (4 of 5 skip GPU segment readback)
    for (let i = 0; i < 18; i++) core.step(1 / 30);
    assert.ok(skippedReadbacks >= 4, `Expected skipped GPU segment readbacks, got ${skippedReadbacks}`);

    // Freeze and scrub across 5 consecutive cine frames: EVERY frame must have exact non-empty segment codes
    const refFrame: PolarFrame = {
      spec: core.lastFrame!.spec,
      amplitude: new Float32Array(core.lastFrame!.amplitude.length),
      transmission: new Float32Array(core.lastFrame!.transmission.length),
      structure: new Uint8Array(core.lastFrame!.structure.length),
      tissue: new Uint8Array(core.lastFrame!.tissue.length),
      segment: new Uint8Array(core.lastFrame!.segment.length),
    };
    for (let offset = 0; offset >= -4; offset--) {
      core.setInput({ ...live, frozen: true, cineOffset: offset });
      const out = core.step(0.05);
      assert.ok(out !== null, `Expected frozen output at offset ${offset}`);
      let nonZero = 0;
      for (let i = 0; i < out.segment.length; i++) if (out.segment[i]! > 0) nonZero++;
      assert.ok(
        nonZero > 50,
        `GPU cine frame at offset ${offset} has empty segment map (nonZero=${nonZero})`,
      );
      // Compare bit-for-bit against direct CPU classification of the same cine frame
      const cf = (core as unknown as { cine: { select: (o: number) => { scene: Scene; beam: BeamFrame; spec: PolarFrameSpec; phase: number } } }).cine.select(offset);
      cpuRef.render(cf.scene, cf.beam, cf.spec, cf.phase, refFrame);
      assert.deepEqual(
        Array.from(out.segment),
        Array.from(refFrame.segment),
        `GPU cine frame at offset ${offset} segment map must match direct classifier bit-for-bit`,
      );
    }
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
