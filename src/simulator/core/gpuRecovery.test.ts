import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { simulatedGpuFactory, type SimulatedCanvas } from '@/simulator/renderer/gpu/simulatedGl';
import type { Webgl2Renderer } from '@/simulator/renderer/gpu/webgl2Renderer';
import { CALIBRATED_TIER } from '@/simulator/renderer/types';
import { gpuNotice } from '@/ui/gpuNotice';
import { baseInput } from './baseInput';
import { GPU_LOST, GPU_RETRY_DELAYS_MS, GPU_STABLE_MS, GpuRecovery } from './gpuRecovery';
import type { SimInput, SimOutput } from './protocol';
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

/**
 * What a lost renderer let go of leaves on its canvas: none of its listeners (nothing asks the browser for another
 * restoration), only the one-shot that gives the context back should the browser still restore it.
 */
function disposedLost(canvas: SimulatedCanvas, message: string): void {
  assert.equal(canvas.listenerCount('webglcontextlost'), 0, message);
  assert.equal(canvas.listenerCount('webglcontextrestored'), 1, `${message}: the give-back listener`);
}

describe('GpuRecovery', () => {
  it('builds the port again at once and on the same canvas when the browser restores the context', () => {
    const { gpu, recovery, first, canvas } = setup();
    lose(recovery, first);
    assert.equal(recovery.poll(), null, 'nothing to build on before the restoration or the first retry');
    assert.equal(recovery.status, 'GPU perdida (intento 1 de 5)');
    assert.equal(canvas.restore(), true, 'the kept renderer prevented the default of the loss');
    const back = recovery.poll();
    assert.ok(back, 'rebuilt on the restoration, without waiting for the retry');
    assert.equal(back.canvas, canvas as unknown, 'on the restored canvas');
    assert.equal(back.contextLost, false);
    assert.deepEqual(gpu.calls, ['new', 'restored'], 'no new context');
    assert.equal(canvas.listenerCount('webglcontextlost'), 1, "the lost renderer's listeners are gone");
    assert.equal(canvas.listenerCount('webglcontextrestored'), 1);
    assert.equal(recovery.status, null);
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
    disposedLost(canvas, 'the lost renderer is disposed once replaced');
    back.dispose();
    assert.equal(gpu.canvases[1]!.listenerCount(), 0);
  });

  it('the lost context it replaced on a new canvas is given back when the browser restores it later', () => {
    const { gpu, clock, recovery, first, canvas } = setup();
    // where present() left it: a restoration allocated a drawing buffer of the screen's size
    canvas.width = 1024;
    canvas.height = 820;
    lose(recovery, first);
    clock.t = GPU_RETRY_DELAYS_MS[0]!;
    const back = recovery.poll()!;
    assert.equal(back.canvas, gpu.canvases[1] as unknown, 'replaced on a new canvas');
    assert.deepEqual([canvas.width, canvas.height], [1, 1], 'a later restoration allocates one pixel');
    assert.equal(canvas.restore(), true, 'the old renderer had prevented the default of the loss');
    assert.equal(canvas.contextLost, true, 'given back on its restoration, not left alive with no owner');
    assert.equal(canvas.calls.loseContext, 1);
    assert.equal(canvas.listenerCount(), 0, 'a one-shot listener');
    assert.equal(canvas.restore(), false, 'and for good');
    assert.equal(back.contextLost, false, 'the new port is untouched');
    back.dispose();
  });

  it('a restored context the port cannot be built on is given back, and the next attempt is on a new canvas', () => {
    const { gpu, clock, recovery, first, canvas } = setup();
    lose(recovery, first);
    canvas.failShaders = true; // the driver of the restarted GPU process fails a shader
    assert.equal(canvas.restore(), true);
    assert.equal(recovery.poll(), null, 'the attempt on the restored canvas fails');
    assert.deepEqual(gpu.calls, ['new', 'restored']);
    assert.equal(recovery.status, 'GPU perdida (intento 2 de 5)');
    assert.equal(canvas.contextLost, true, 'the restored context is given back, not left alive with no owner');
    assert.equal(canvas.calls.loseContext, 1);
    assert.equal(canvas.listenerCount(), 0);
    clock.t = GPU_RETRY_DELAYS_MS[1]!;
    const back = recovery.poll();
    assert.ok(back, 'the next attempt, two seconds later');
    assert.equal(back.canvas, gpu.canvases[1] as unknown, 'on a new canvas');
    back.dispose();
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
    assert.equal(recovery.status, 'Sin GPU tras 5 intentos; recarga la página');
    disposedLost(canvas, 'the lost renderer is disposed on giving up');
    // and for good: an hour later, or a restoration (given back at once), builds nothing
    assert.equal(canvas.restore(), true);
    assert.equal(canvas.contextLost, true);
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
      if (recovery.status?.startsWith('Sin GPU')) break; // given up
      (r.canvas as unknown as SimulatedCanvas).restore();
      let back = recovery.poll();
      // a bounded wait: a recovery that never rebuilds fails here instead of hanging the run
      for (let waited = 0; !back && waited < 60_000; waited += 100) {
        clock.t += 100;
        back = recovery.poll();
      }
      assert.ok(back, `rebuilt after loss ${i + 1}`);
      rebuiltAt.push(clock.t);
      r = back;
      clock.t += 100; // lost again 100 ms later
    }
    assert.deepEqual(rebuiltAt, [0, 2100, 6200, 14300, 30400], 'at once, then spaced by the backoff');
    assert.equal(recovery.status, 'Sin GPU tras 5 intentos; recarga la página');
    assert.equal(gpu.calls.length, 1 + 5, 'five rebuilds, not one per restoration');
  });

  it(`a port that kept its context ${GPU_STABLE_MS / 1000} s has recovered: the next loss has all its attempts`, () => {
    const { clock, recovery, first } = setup();
    lose(recovery, first);
    (first.canvas as unknown as SimulatedCanvas).restore();
    const back = recovery.poll()!;
    clock.t += GPU_STABLE_MS;
    lose(recovery, back);
    assert.equal(recovery.status, 'GPU perdida (intento 1 de 5)');
    (back.canvas as unknown as SimulatedCanvas).restore();
    assert.ok(recovery.poll(), 'rebuilt at once again, as on a first loss');
  });

  it('dispose lets go of the lost renderer', () => {
    const { recovery, first, canvas } = setup();
    lose(recovery, first);
    recovery.dispose();
    disposedLost(canvas, 'the lost renderer is disposed');
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
    assert.equal(out.stats['gpu'], GPU_LOST, 'why');
    assert.equal(out.stats['gpuLive'], 'GPU perdida (intento 1 de 5)', "and the port's state, apart");
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
    assert.equal(out.stats['gpu'], GPU_LOST);
    assert.equal(out.stats['gpuLive'], 'Sin GPU tras 5 intentos; recarga la página');
    assert.equal(out.stats['console'], 'cpu');
    disposedLost(gpu.canvases[0]!, 'the lost renderer was disposed');
    core.dispose();
  });

  /** A core over a simulated GPU, a step formed live, and frozen on its frame: the frozen frame first shown. */
  function frozenOverGpu() {
    const gpu = simulatedGpuFactory();
    const clock = { t: 0 };
    const live = baseInput({ rendererBackend: 'atlas', quality: 'auto' });
    const core = new SimulatorCore(caseDef, live, { createGpu: gpu.create, now: () => clock.t });
    step(core);
    core.setInput({ ...live, frozen: true });
    const shown = core.step(0.08);
    assert.ok(shown?.frozen);
    assert.equal(shown.stats['gpu'], 'ok');
    core.recycle(shown.rgba);
    assert.equal(core.step(0.08), null, 'nothing new to show');
    return { gpu, clock, live, core, shown, canvas: gpu.canvases[0]! };
  }

  it('a context lost while frozen shows in the HUD at once, and its attempt comes a second after the loss', () => {
    const { gpu, clock, live, core, shown, canvas } = frozenOverGpu();
    // the learner measures on the frozen image when the GPU process restarts, and the browser does not restore it
    canvas.lose();
    const lost = core.step(0.08);
    assert.ok(lost?.frozen, 'the frozen frame is composed again, for its HUD');
    assert.equal(lost.frameId, shown.frameId, 'the same frame');
    assert.equal(lost.stats['gpuLive'], 'GPU perdida (intento 1 de 5)', "the port's state, apart from the frame's");
    core.recycle(lost.rgba);
    assert.equal(core.step(0.08), null, 'once');
    clock.t = GPU_RETRY_DELAYS_MS[0]! - 1;
    assert.equal(core.step(0.08), null);
    assert.deepEqual(gpu.calls, ['new'], 'nothing tried before a second after the loss');
    clock.t = GPU_RETRY_DELAYS_MS[0]!;
    const back = core.step(0.08);
    assert.deepEqual(gpu.calls, ['new', 'new'], 'a second after the loss, still frozen, on a new canvas');
    assert.equal(back?.stats['gpuLive'], 'ok', 'and the HUD says the port is back');
    assert.equal(back.frameId, shown.frameId);
    // unfrozen a minute later, the first frame is on the port (a second of the CPU tracer at medium came first before)
    clock.t = 60_000;
    core.setInput(live);
    const out = step(core);
    assert.equal(out.stats['console'], 'gpu');
    assert.equal(out.stats['tier'], CALIBRATED_TIER);
    core.dispose();
  });

  it('unfrozen before its first attempt, a context lost while frozen still comes back a second after the loss', () => {
    const { gpu, clock, live, core, canvas } = frozenOverGpu();
    canvas.lose();
    core.step(0.08); // a frozen tick, at the loss
    clock.t = 500;
    core.setInput(live);
    let out = step(core);
    assert.equal(out.stats['console'], 'cpu', 'the CPU tracer meanwhile');
    assert.equal(out.stats['tier'], 'medium');
    clock.t = GPU_RETRY_DELAYS_MS[0]!;
    out = step(core);
    assert.equal(out.stats['console'], 'gpu', 'a second after the loss, not after the unfreeze');
    assert.equal(out.stats['tier'], CALIBRATED_TIER);
    assert.deepEqual(gpu.calls, ['new', 'new']);
    core.dispose();
  });

  it('a context the browser restores while frozen brings the port back at once, and the HUD says so', () => {
    const { gpu, live, core, shown, canvas } = frozenOverGpu();
    canvas.lose();
    const lost = core.step(0.08);
    assert.match(String(lost?.stats['gpuLive']), /GPU perdida/);
    assert.equal(canvas.restore(), true);
    const back = core.step(0.08);
    assert.ok(back?.frozen);
    assert.equal(back.frameId, shown.frameId, 'the frozen frame stands');
    assert.equal(back.stats['gpuLive'], 'ok', 'without waiting for the unfreeze');
    assert.deepEqual(gpu.calls, ['new', 'restored'], 'on the restored context');
    assert.equal(canvas.listenerCount('webglcontextlost'), 1, 'only the new renderer listens');
    assert.equal(canvas.listenerCount('webglcontextrestored'), 1);
    core.setInput(live);
    const out = step(core);
    assert.equal(out.stats['console'], 'gpu', 'the first live frame is on the port');
    assert.equal(out.stats['tier'], CALIBRATED_TIER);
    core.dispose();
  });

  /** What the stats say of how the frame was formed: the Dev panel's rows, and the tier a review report keeps. */
  const formation = (o: SimOutput) => {
    const s = o.stats;
    return {
      gpu: s['gpu'],
      tier: s['tier'],
      backend: s['backend'],
      source: s['source'],
      console: s['console'],
      dopplerPipe: s['dopplerPipe'],
      anatomy: s['anatomy'],
      imagen: s['imagen'],
    };
  };

  it('frozen, a context lost and brought back leaves what the Dev panel says of the frame as it was formed', () => {
    for (const rendererBackend of ['atlas', 'webgl2'] as const) {
      const gpu = simulatedGpuFactory();
      const live = baseInput({ rendererBackend, quality: 'auto', modality: 'color' });
      const core = new SimulatorCore(caseDef, live, { createGpu: gpu.create, now: () => 0 });
      for (let i = 0; i < 12; i++) {
        const o = core.step(1 / 30);
        if (o) core.recycle(o.rgba);
      }
      core.setInput({ ...live, frozen: true });
      const shown = core.step(0.08)!;
      core.recycle(shown.rgba);
      assert.deepEqual(
        formation(shown),
        {
          gpu: 'ok',
          tier: CALIBRATED_TIER,
          backend: rendererBackend === 'atlas' ? 'atlas' : 'webgl2-procedural',
          source: rendererBackend === 'atlas' ? 'webgl2-procedural' : undefined,
          console: 'gpu',
          dopplerPipe: 'gpu-deferred',
          anatomy: 'analítica',
          imagen: 'red',
        },
        `${rendererBackend}: formed on the port, its console and its colour Doppler too`,
      );
      // before, the stats followed the pipeline: console and Doppler on the CPU, the source gone (a new atlas), and with
      // the webgl2 backend the procedural one, the frame unchanged meanwhile
      gpu.canvases[0]!.lose();
      const lost = core.step(0.08);
      assert.equal(lost?.frameId, shown.frameId);
      assert.deepEqual(formation(lost), formation(shown), `${rendererBackend}: lost, the frame is as it was formed`);
      assert.equal(lost.stats['gpuLive'], 'GPU perdida (intento 1 de 5)', "the port's state goes apart");
      core.recycle(lost.rgba);
      assert.equal(gpu.canvases[0]!.restore(), true);
      const back = core.step(0.08);
      assert.equal(back?.frameId, shown.frameId);
      assert.deepEqual(formation(back), formation(shown), `${rendererBackend}: and back`);
      assert.equal(back.stats['gpuLive'], 'ok');
      core.dispose();
    }
  });

  it('a frame scrolled back to in the cine speaks of itself: formed on the GPU before a loss, by the CPU tracer after', () => {
    const gpu = simulatedGpuFactory();
    const clock = { t: 0 };
    const live = baseInput({ rendererBackend: 'atlas', quality: 'auto' });
    const core = new SimulatorCore(caseDef, live, { createGpu: gpu.create, now: () => clock.t });
    let onGpu = 0;
    for (let i = 0; i < 3; i++) onGpu = step(core).frameId;
    gpu.canvases[0]!.lose(); // and not restored: the attempt a second later is on a new canvas
    let newest = 0;
    for (let i = 0; i < 3; i++) newest = step(core).frameId;
    const frozen = { ...live, frozen: true };
    core.setInput(frozen);
    const cpuFrame = core.step(0.08)!;
    core.recycle(cpuFrame.rgba);
    const traced = {
      gpu: GPU_LOST,
      tier: 'medium',
      backend: 'atlas',
      source: 'procedural',
      console: 'cpu',
      dopplerPipe: 'off',
      anatomy: 'analítica',
      imagen: 'red',
    };
    assert.deepEqual(formation(cpuFrame), traced, 'the newest frame, by the CPU tracer at medium');
    assert.equal(cpuFrame.stats['gpuLive'], 'GPU perdida (intento 1 de 5)');
    // before, a frame scrolled back to took the newest's tier and GPU: «calidad media», the CPU tracer and its source
    core.setInput({ ...frozen, cineOffset: onGpu - newest });
    const gpuFrame = core.step(0.08)!;
    core.recycle(gpuFrame.rgba);
    assert.equal(gpuFrame.frameId, onGpu);
    const onPort = { ...traced, gpu: 'ok', tier: CALIBRATED_TIER, source: 'webgl2-procedural', console: 'gpu' };
    assert.deepEqual(formation(gpuFrame), onPort, 'one before the loss, on the port at the calibrated tier');
    assert.equal(gpuFrame.stats['gpuLive'], 'GPU perdida (intento 1 de 5)');
    // the port back while frozen: what each frame says of itself stays
    clock.t = GPU_RETRY_DELAYS_MS[0]!;
    const back = core.step(0.08)!;
    core.recycle(back.rgba);
    assert.deepEqual(gpu.calls, ['new', 'new']);
    assert.equal(back.frameId, onGpu);
    assert.deepEqual(formation(back), onPort);
    assert.equal(back.stats['gpuLive'], 'ok');
    core.setInput(frozen);
    const cpuBack = core.step(0.08)!;
    assert.equal(cpuBack.frameId, newest);
    assert.deepEqual(formation(cpuBack), traced, 'still the CPU tracer at medium: the frame shown is that one');
    assert.equal(cpuBack.stats['gpuLive'], 'ok');
    core.dispose();
  });

  it('paused, a loss is seen on waking: built on at once if the browser restored it, retried a second later if not', () => {
    // the worker schedules no tick while paused (another screen, a hidden tab), so the core steps nothing meanwhile
    for (const restored of [true, false]) {
      const gpu = simulatedGpuFactory();
      const clock = { t: 0 };
      const live = baseInput({ rendererBackend: 'atlas', quality: 'auto' });
      const core = new SimulatorCore(caseDef, live, { createGpu: gpu.create, now: () => clock.t });
      step(core);
      core.setInput({ ...live, paused: true });
      gpu.canvases[0]!.lose(); // a suspended laptop
      if (restored) assert.equal(gpu.canvases[0]!.restore(), true);
      clock.t = 60_000;
      core.setInput(live);
      let out = step(core);
      if (restored) {
        assert.equal(out.stats['gpu'], 'ok', 'no CPU frame on waking');
        assert.equal(out.stats['console'], 'gpu');
        assert.equal(out.stats['tier'], CALIBRATED_TIER);
        assert.deepEqual(gpu.calls, ['new', 'restored']);
      } else {
        assert.equal(out.stats['gpu'], GPU_LOST);
        assert.equal(out.stats['gpuLive'], 'GPU perdida (intento 1 de 5)', 'the HUD says so from the first frame');
        assert.equal(out.stats['console'], 'cpu');
        assert.equal(out.stats['tier'], 'medium');
        clock.t = 60_000 + GPU_RETRY_DELAYS_MS[0]! - 1;
        assert.equal(step(core).stats['console'], 'cpu');
        assert.deepEqual(gpu.calls, ['new'], 'the attempts count from waking, when the loss was seen');
        clock.t = 60_000 + GPU_RETRY_DELAYS_MS[0]!;
        out = step(core);
        assert.equal(out.stats['console'], 'gpu');
        assert.deepEqual(gpu.calls, ['new', 'new']);
      }
      core.dispose();
    }
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

/**
 * A core over a simulated GPU and a clock the test moves, as the worker runs it: `out` steps once (live 50 ms, frozen an
 * 80 ms tick) and returns the output it composed, which it must have; `hud` is what the HUD paints over that output.
 */
function scanner(options: { refuse?: boolean } = {}) {
  const gpu = simulatedGpuFactory();
  gpu.refuse = options.refuse ?? false;
  const clock = { t: 0 };
  const live = baseInput({ rendererBackend: 'atlas', quality: 'auto' });
  const core = new SimulatorCore(loadCaseById('normal-excellent-window'), live, {
    createGpu: gpu.create,
    now: () => clock.t,
  });
  const frozen = { ...live, frozen: true };
  const out = (dt = 0.05): SimOutput => {
    const o = core.step(dt);
    assert.ok(o, 'a frame composed');
    core.recycle(o.rgba);
    return o;
  };
  const hud = (dt = 0.05) => gpuNotice(out(dt).stats);
  return { gpu, clock, live, frozen, core, out, hud };
}

const TRACED = 'Sin GPU · trazador CPU, calidad media';
/** The tip over a frame the CPU tracer formed after a loss: why, and where the port stands now. */
const tipOver = (port: string) => `${GPU_LOST}: cuadro formado por el trazador CPU · ${port}`;

describe("the HUD's GPU notice over the frame shown, as the port moves on", () => {
  it('frozen over a frame the GPU formed at high quality, a lost context says the port is lost, not the CPU tracer', () => {
    const { gpu, clock, live, frozen, core, hud } = scanner();
    assert.equal(hud(), null);
    core.setInput(frozen);
    assert.equal(hud(0.08), null);
    gpu.canvases[0]!.lose();
    // round 1: «Sin GPU · trazador CPU, calidad alta», over a frame the GPU formed and with nothing traced on the CPU
    assert.deepEqual(hud(0.08), { text: 'GPU perdida (intento 1 de 5)' });
    clock.t = GPU_RETRY_DELAYS_MS[0]!;
    assert.equal(hud(0.08), null, 'the port back, still frozen');
    core.setInput(live);
    assert.equal(hud(), null, 'live on the port');
    core.dispose();
  });

  it('frozen over a frame the GPU formed, past the five attempts it says the port did not come back', () => {
    const { gpu, clock, frozen, core, hud } = scanner();
    hud();
    core.setInput(frozen);
    hud(0.08);
    gpu.refuse = true;
    gpu.canvases[0]!.lose();
    const seen = [hud(0.08)];
    // the attempts 1, 3, 7, 15 and 31 s after the loss, each failed one composing the frozen frame again for its HUD
    for (const t of [1000, 3000, 7000, 15000, 31000]) {
      clock.t = t;
      seen.push(hud(0.08));
    }
    assert.deepEqual(
      seen.map((n) => n?.text),
      [1, 2, 3, 4, 5].map((k) => `GPU perdida (intento ${k} de 5)`).concat('Sin GPU tras 5 intentos; recarga la página'),
    );
    core.dispose();
  });

  it('B1: frozen over a frame the CPU tracer formed, the notice stays when the GPU comes back, and its tip says so', () => {
    const { gpu, clock, live, frozen, core, out, hud } = scanner();
    out();
    gpu.canvases[0]!.lose(); // live, and not restored
    clock.t = 100;
    const traced = out();
    assert.equal(traced.stats['gpu'], GPU_LOST, 'the frame keeps why the CPU tracer formed it, and only that');
    assert.deepEqual(gpuNotice(traced.stats), { text: TRACED, tip: tipOver('GPU perdida (intento 1 de 5)') });
    core.setInput(frozen);
    assert.deepEqual(hud(0.08), gpuNotice(traced.stats), 'frozen, the port still lost');
    clock.t = GPU_RETRY_DELAYS_MS[0]! + 100;
    // round 2: «Contexto WebGL perdido: trazador CPU hasta que vuelva la GPU (intento 1 de 5)», with the GPU back
    const back = hud(0.08);
    assert.deepEqual(back, { text: TRACED, tip: tipOver('GPU de vuelta') }, "the frame shown is the CPU tracer's");
    assert.doesNotMatch(String(back?.tip), /hasta que vuelva/);
    assert.deepEqual(gpu.calls, ['new', 'new']);
    core.setInput(live);
    assert.equal(hud(), null, 'live on the port again');
    core.dispose();
  });

  it('B2: frozen over a frame the CPU tracer formed, the tip follows every attempt that fails, and the giving up', () => {
    const { gpu, clock, frozen, core, out } = scanner();
    out();
    gpu.canvases[0]!.lose();
    clock.t = 100;
    out();
    core.setInput(frozen);
    const shown = out(0.08);
    gpu.refuse = true;
    const tips: string[] = [];
    for (const t of [1100, 3100, 7100, 15100, 31100]) {
      clock.t = t;
      const o = out(0.08); // composed again at each attempt, for its HUD
      assert.equal(o.frameId, shown.frameId);
      assert.equal(o.stats['gpu'], GPU_LOST);
      const notice = gpuNotice(o.stats);
      assert.equal(notice?.text, TRACED);
      tips.push(String(notice?.tip));
    }
    // round 2: all five said «intento 1 de 5», the attempt of the frame's capture, and the advice came on unfreezing
    assert.deepEqual(
      tips,
      [2, 3, 4, 5]
        .map((k) => tipOver(`GPU perdida (intento ${k} de 5)`))
        .concat(tipOver('Sin GPU tras 5 intentos; recarga la página')),
    );
    core.dispose();
  });

  it("without WebGL2 the frame is the CPU tracer's, live and frozen, and the port's state is that same reason", () => {
    const { frozen, core, out } = scanner({ refuse: true });
    const live = out();
    assert.equal(live.stats['gpu'], 'WebGL2 unavailable');
    assert.equal(live.stats['gpuLive'], 'WebGL2 unavailable', "not 'ok': there is no port");
    assert.deepEqual(gpuNotice(live.stats), { text: TRACED, tip: 'WebGL2 unavailable' });
    core.setInput(frozen);
    const shown = out(0.08);
    assert.equal(shown.stats['gpuLive'], 'WebGL2 unavailable');
    assert.deepEqual(gpuNotice(shown.stats), gpuNotice(live.stats));
    core.dispose();
  });
});

/** Stats of an output that tell of its own composition and presentation, or of the port now: not of the frame shown. */
const OUTPUT_STATS = new Set(['gpuLive', 'compositeMs', 'present', 'presentMs']);
const formedAs = (o: SimOutput) => Object.fromEntries(Object.entries(o.stats).filter(([k]) => !OUTPUT_STATS.has(k)));

describe('a frozen frame tells of its formation what its live output told', () => {
  const caseDef = loadCaseById('normal-excellent-window');

  it('a loss and a return crossed live in 2D and Color: every frame of the cine says, frozen, what it said live', () => {
    for (const modality of ['2d', 'color'] as SimInput['modality'][]) {
      const gpu = simulatedGpuFactory();
      const clock = { t: 0 };
      const live = baseInput({ rendererBackend: 'atlas', quality: 'auto', modality });
      const core = new SimulatorCore(caseDef, live, { createGpu: gpu.create, now: () => clock.t });
      /** What the live output that formed each frame said of it (Dev panel rows, tier, notice), by frameId. */
      const said = new Map<number, Record<string, unknown>>();
      const form = (frames: number) => {
        for (let formed = 0, steps = 0; formed < frames; steps++) {
          assert.ok(steps < 200, `${modality}: ${frames} frames formed`);
          const o = core.step(0.05);
          if (!o) continue;
          core.recycle(o.rgba);
          if (said.has(o.frameId)) continue; // the colour re-shaded over a frame formed before
          said.set(o.frameId, formedAs(o));
          formed++;
        }
      };
      form(3); // on the port
      gpu.canvases[0]!.lose(); // not restored: the CPU tracer until the attempt a second later
      form(3);
      clock.t = GPU_RETRY_DELAYS_MS[0]!;
      form(3); // back, on a new canvas
      assert.deepEqual(gpu.calls, ['new', 'new']);
      // three on the port, three by the CPU tracer, three on the port again: the first CPU frame and the first one back
      // are formed by the step that sees the change, the frames a formation taken before forming the frame misdescribes
      assert.deepEqual(
        [...said.values()].map((s) => `${s['console']} ${s['source']} ${s['tier']}`),
        [0, 0, 0, 1, 1, 1, 0, 0, 0].map((cpu) =>
          cpu ? 'cpu procedural medium' : `gpu webgl2-procedural ${CALIBRATED_TIER}`,
        ),
        modality,
      );
      const ids = [...said.keys()];
      const newest = Math.max(...ids);
      const frozen = { ...live, frozen: true };
      const traverse = (order: number[]) => {
        for (const id of order) {
          core.setInput({ ...frozen, cineOffset: id - newest });
          const o = core.step(0.08);
          assert.ok(o?.frozen, `${modality}: frame ${id} composed`);
          assert.equal(o.frameId, id);
          core.recycle(o.rgba);
          // the colour of a frame on the deferred path is the GPU's own only for the newest, read back as it was seen;
          // any other is estimated again on the CPU on its first frozen display, and says so
          const told = { ...said.get(id)! };
          if (told['dopplerPipe'] === 'gpu-deferred' && id !== newest)
            told['dopplerPipe'] = 'gpu-deferred (reestimado en CPU)';
          // round 2: renderFrameMs, consoleMs, persistence, analysisMs and cineMs of the last live frame on every frame
          assert.deepEqual(formedAs(o), told, `${modality}: frame ${id}`);
        }
      };
      traverse([...ids].sort((a, b) => b - a)); // the newest first, as freezing shows it
      // and the port lost again while frozen: each frame still says how it was formed, the port's state apart
      gpu.canvases[1]!.lose();
      traverse([...ids].sort((a, b) => a - b));
      core.dispose();
    }
  });

  it('Color frozen: a frame whose colour the CPU estimated again says so; the newest, read back as it was seen, not', () => {
    for (const shown of ['newest', 'older', 'newest, lost before shown'] as const) {
      const gpu = simulatedGpuFactory();
      const live = baseInput({ rendererBackend: 'atlas', quality: 'auto', modality: 'color' });
      const core = new SimulatorCore(caseDef, live, { createGpu: gpu.create, now: () => 0 });
      let last: SimOutput | null = null;
      for (let i = 0; i < 20; i++) {
        const o = core.step(1 / 30);
        if (!o) continue;
        last = o;
        core.recycle(o.rgba);
      }
      assert.equal(last?.stats['dopplerPipe'], 'gpu-deferred', 'acquired on the GPU deferred path');
      const frozen = { ...live, frozen: true };
      core.setInput(shown === 'older' ? { ...frozen, cineOffset: -3 } : frozen);
      if (shown === 'newest, lost before shown') gpu.canvases[0]!.lose();
      const o = core.step(0.08);
      assert.ok(o?.frozen);
      // round 2: 'gpu-deferred' over the three, the colour of the last two being the CPU's estimate
      const pipe = shown === 'newest' ? 'gpu-deferred' : 'gpu-deferred (reestimado en CPU)';
      assert.equal(o.stats['dopplerPipe'], pipe, shown);
      core.dispose();
    }
  });
});
