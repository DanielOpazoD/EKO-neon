import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { simulatedGpuFactory, type SimulatedCanvas } from '@/simulator/renderer/gpu/simulatedGl';
import type { Webgl2Renderer } from '@/simulator/renderer/gpu/webgl2Renderer';
import { CALIBRATED_TIER } from '@/simulator/renderer/types';
import { baseInput } from './baseInput';
import { GPU_RETRY_DELAYS_MS, GPU_STABLE_MS, GpuRecovery } from './gpuRecovery';
import type { SimOutput } from './protocol';
import { SimulatorCore } from './simulatorCore';

/** A recovery over simulated canvases and a clock the test moves. */
function setup() {
  const gpu = simulatedGpuFactory();
  const clock = { t: 0 };
  const recovery = new GpuRecovery(gpu.create, () => clock.t);
  const first = gpu.create().renderer!;
  return { gpu, clock, recovery, first, canvas: gpu.canvases[0]! };
}

/** The renderer's context is lost as in a browser: the loss and its event, then the core hands the renderer over. */
function lose(recovery: GpuRecovery, r: Webgl2Renderer): void {
  (r.canvas as unknown as SimulatedCanvas).lose();
  recovery.lost(r);
}

describe('GpuRecovery', () => {
  it('builds the port again at once and on the same canvas when the browser restores the context', () => {
    const { gpu, recovery, first, canvas } = setup();
    lose(recovery, first);
    assert.equal(recovery.poll(), null, 'nothing to build on before the restoration or the first retry');
    assert.match(recovery.reason!, /intento 1 de 5/);
    assert.equal(canvas.restore(), true, 'the kept renderer prevented the default of the loss');
    const back = recovery.poll();
    assert.ok(back, 'rebuilt on the restoration, without waiting for the retry');
    assert.equal(back.canvas, canvas as unknown, 'on the restored canvas');
    assert.equal(back.contextLost, false);
    assert.deepEqual(gpu.calls, ['new', 'restored'], 'no new context');
    assert.equal(canvas.listenerCount('webglcontextlost'), 1, "the lost renderer's listeners are gone");
    assert.equal(canvas.listenerCount('webglcontextrestored'), 1);
    assert.equal(recovery.reason, null);
    back.dispose();
  });

  it('without a restoration, waits for the first retry and builds on a new canvas', () => {
    const { gpu, clock, recovery, first, canvas } = setup();
    lose(recovery, first);
    clock.t = GPU_RETRY_DELAYS_MS[0]! - 1;
    assert.equal(recovery.poll(), null);
    assert.deepEqual(gpu.calls, ['new'], 'no attempt before the first delay');
    clock.t = GPU_RETRY_DELAYS_MS[0]!;
    const back = recovery.poll();
    assert.ok(back);
    assert.equal(back.canvas, gpu.canvases[1] as unknown, 'a new canvas');
    assert.equal(canvas.listenerCount(), 0, 'the lost renderer is disposed once replaced');
    back.dispose();
    assert.equal(gpu.canvases[1]!.listenerCount(), 0);
  });

  it('gives up after five attempts at 1, 3, 7, 15 and 31 s when the GPU never comes back', () => {
    const { gpu, clock, recovery, first, canvas } = setup();
    gpu.refuse = true; // new canvases have no WebGL2, and the lost context is not restored
    lose(recovery, first);
    const attempts: number[] = [];
    for (clock.t = 0; clock.t <= 120_000; clock.t += 100) {
      const before = gpu.calls.length;
      assert.equal(recovery.poll(), null);
      if (gpu.calls.length > before) attempts.push(clock.t);
    }
    assert.deepEqual(attempts, [1000, 3000, 7000, 15000, 31000]);
    assert.match(recovery.reason!, /no volvió tras 5 intentos \(WebGL2 unavailable\); recarga el caso/);
    assert.equal(canvas.listenerCount(), 0, 'the lost renderer is disposed on giving up');
    // and for good: an hour later, or a restoration nobody listens to any more, builds nothing
    assert.equal(canvas.restore(), true);
    clock.t += 3_600_000;
    assert.equal(recovery.poll(), null);
    assert.equal(gpu.calls.length, 1 + 5);
  });

  it('a context lost again soon after each rebuild keeps counting its attempts, with the backoff, until the cap', () => {
    const { gpu, clock, recovery, first } = setup();
    let r = first;
    const rebuiltAt: number[] = [];
    for (let i = 0; i < 10; i++) {
      lose(recovery, r);
      if (recovery.reason?.includes('no volvió')) break;
      (r.canvas as unknown as SimulatedCanvas).restore();
      let back = recovery.poll();
      while (!back) {
        clock.t += 100;
        back = recovery.poll();
      }
      rebuiltAt.push(clock.t);
      r = back;
      clock.t += 100; // lost again 100 ms later
    }
    assert.deepEqual(rebuiltAt, [0, 2100, 6200, 14300, 30400], 'at once, then spaced by the backoff');
    assert.match(recovery.reason!, /no volvió tras 5 intentos/);
    assert.equal(gpu.calls.length, 1 + 5, 'five rebuilds, not one per restoration');
  });

  it(`a port that kept its context ${GPU_STABLE_MS / 1000} s has recovered: the next loss has all its attempts`, () => {
    const { clock, recovery, first } = setup();
    lose(recovery, first);
    (first.canvas as unknown as SimulatedCanvas).restore();
    const back = recovery.poll()!;
    clock.t += GPU_STABLE_MS;
    lose(recovery, back);
    assert.match(recovery.reason!, /intento 1 de 5/);
    (back.canvas as unknown as SimulatedCanvas).restore();
    assert.ok(recovery.poll(), 'rebuilt at once again, as on a first loss');
  });

  it('dispose lets go of the lost renderer', () => {
    const { recovery, first, canvas } = setup();
    lose(recovery, first);
    recovery.dispose();
    assert.equal(canvas.listenerCount(), 0);
    assert.equal(recovery.poll(), null);
  });
});

describe('SimulatorCore after a lost WebGL context', () => {
  const caseDef = loadCaseById('normal-excellent-window');
  const step = (core: SimulatorCore): SimOutput => {
    const out = core.step(0.05);
    assert.ok(out);
    return out;
  };

  it('returns to the GPU port at the calibrated tier when the browser restores the context', () => {
    const gpu = simulatedGpuFactory();
    const clock = { t: 0 };
    const core = new SimulatorCore(caseDef, baseInput({ rendererBackend: 'atlas', quality: 'auto' }), {
      createGpu: gpu.create,
      now: () => clock.t,
    });
    let out = step(core);
    assert.equal(out.stats['gpu'], 'ok');
    assert.equal(out.stats['tier'], CALIBRATED_TIER);
    assert.equal(out.stats['console'], 'gpu');
    const canvas = gpu.canvases[0]!;

    // as in a browser, the core may see isContextLost() before the event is dispatched
    canvas.loseSilently();
    out = step(core);
    assert.equal(out.stats['console'], 'cpu', 'the frame is formed by the CPU tracer');
    assert.equal(out.stats['tier'], 'medium', 'from its first frame at medium, not traced at the calibrated tier');
    assert.match(String(out.stats['gpu']), /Contexto WebGL perdido/);
    assert.equal(canvas.dispatchLost(), true, 'the kept renderer still asks the browser to restore');
    out = step(core);
    assert.equal(out.stats['console'], 'cpu', 'the CPU tracer meanwhile');

    canvas.restore();
    out = step(core);
    assert.equal(out.stats['gpu'], 'ok', "the HUD's warning goes away");
    assert.equal(out.stats['tier'], CALIBRATED_TIER, 'the image is at the calibrated tier again');
    assert.equal(out.stats['console'], 'gpu');
    assert.equal(out.stats['source'], 'webgl2-procedural', 'a new atlas over the port');
    assert.deepEqual(gpu.calls, ['new', 'restored'], 'rebuilt on the restored context, no new one');
    assert.equal(canvas.listenerCount('webglcontextlost'), 1, 'only the new renderer listens');
    assert.equal(canvas.listenerCount('webglcontextrestored'), 1);
    core.dispose();
    assert.equal(canvas.listenerCount(), 0);
  });

  it('a context the browser restored before the next step costs no CPU frame', () => {
    const gpu = simulatedGpuFactory();
    const core = new SimulatorCore(caseDef, baseInput({ rendererBackend: 'atlas', quality: 'auto' }), {
      createGpu: gpu.create,
      now: () => 0,
    });
    step(core);
    const canvas = gpu.canvases[0]!;
    canvas.lose();
    canvas.restore();
    const out = step(core);
    assert.equal(out.stats['gpu'], 'ok');
    assert.equal(out.stats['console'], 'gpu');
    assert.equal(out.stats['tier'], CALIBRATED_TIER);
    assert.deepEqual(gpu.calls, ['new', 'restored']);
    core.dispose();
    assert.equal(canvas.listenerCount(), 0);
  });

  it('keeps the CPU tracer after five failed attempts, says so, and tries no more', () => {
    const gpu = simulatedGpuFactory();
    const clock = { t: 0 };
    const core = new SimulatorCore(caseDef, baseInput({ rendererBackend: 'atlas', quality: 'low' }), {
      createGpu: gpu.create,
      now: () => clock.t,
    });
    step(core);
    gpu.refuse = true;
    gpu.canvases[0]!.lose();
    let out = step(core);
    for (const t of [1000, 3000, 7000, 15000, 31000, 3_600_000]) {
      clock.t = t;
      out = step(core);
    }
    assert.equal(gpu.calls.filter((c) => c === 'new').length, 1 + 5);
    assert.match(String(out.stats['gpu']), /no volvió tras 5 intentos/);
    assert.equal(out.stats['console'], 'cpu');
    assert.equal(gpu.canvases[0]!.listenerCount(), 0, 'the lost renderer was disposed');
    core.dispose();
  });

  it('the gpuContext dev request loses and restores the context through WEBGL_lose_context', () => {
    const gpu = simulatedGpuFactory();
    const core = new SimulatorCore(caseDef, baseInput({ rendererBackend: 'atlas', quality: 'low' }), {
      createGpu: gpu.create,
      now: () => 0,
    });
    step(core);
    assert.deepEqual(core.request({ kind: 'gpuContext', action: 'lose' }), { kind: 'gpuContext', ok: true });
    assert.equal(step(core).stats['console'], 'cpu');
    assert.deepEqual(core.request({ kind: 'gpuContext', action: 'restore' }), { kind: 'gpuContext', ok: true });
    const out = step(core);
    assert.equal(out.stats['gpu'], 'ok');
    assert.equal(out.stats['console'], 'gpu');
    assert.deepEqual(core.request({ kind: 'gpuContext', action: 'restore' }), { kind: 'gpuContext', ok: false });
    core.dispose();
  });
});
