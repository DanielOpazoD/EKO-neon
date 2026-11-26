import test from 'node:test';
import assert from 'node:assert/strict';
import type { HeartPose } from '@/simulator/anatomy/heartModel';
import type { VolumeHeart } from '@/simulator/anatomy/volumeHeart';
import { SimulatedCanvas, withSimulatedOffscreenCanvas } from './simulatedGl';
import { volumeHeartGpu } from './volumeHeartGpu';
import { createWebgl2Renderer, Webgl2Renderer } from './webgl2Renderer';

const SPEC = { lines: 4, samples: 4, sectorRad: 1, depthCm: 10, focusCm: 6, elevationSamples: 1 };

test('Webgl2Renderer intercepts webglcontextlost, short-circuits rendering, and cleans up on dispose', () => {
  const canvas = new SimulatedCanvas();
  const renderer = new Webgl2Renderer(canvas.gl);
  assert.equal(canvas.listenerCount('webglcontextlost'), 1, 'must subscribe to webglcontextlost on canvas');
  assert.equal(renderer.contextLost, false);

  // Fire webglcontextlost event (the event alone: isContextLost() is not set)
  const prevented = canvas.dispatchLost();
  assert.equal(prevented, true, 'must call preventDefault() on webglcontextlost');
  assert.equal(renderer.contextLost, true, 'contextLost must flip to true immediately');

  // Calling render / renderDisplay / present after loss must safely no-op without touching GL
  renderer.render({} as never, {} as never, SPEC, 0, {} as never);
  const displayOk = renderer.renderDisplay(
    {} as never,
    {} as never,
    SPEC,
    0,
    {} as never,
    undefined,
    {} as never,
    new Uint8ClampedArray(16),
  );
  const bitmap = renderer.present({
    lut: {} as never,
    width: 64,
    height: 64,
    color: null,
  });
  assert.equal(displayOk, false);
  assert.equal(bitmap, null);
  assert.equal(canvas.calls.readPixels, 0, 'must not issue readPixels after context loss');

  // Dispose must remove the event listeners
  renderer.dispose();
  assert.equal(canvas.listenerCount('webglcontextlost'), 0, 'dispose() must remove webglcontextlost listener');
  assert.equal(canvas.listenerCount(), 0, 'nor leave the webglcontextrestored listener');
});

test('Webgl2Renderer also detects driver-side gl.isContextLost() without a DOM event', () => {
  const canvas = new SimulatedCanvas();
  const renderer = new Webgl2Renderer(canvas.gl);
  assert.equal(renderer.contextLost, false);
  canvas.loseSilently();
  assert.equal(renderer.contextLost, true);
  renderer.dispose();
});

test('the lost renderer flags the restoration and lets go of the restored context untouched', () => {
  const canvas = new SimulatedCanvas();
  const lost = new Webgl2Renderer(canvas.gl);
  assert.equal(canvas.listenerCount('webglcontextrestored'), 1, 'listens for the restoration');
  assert.equal(canvas.lose(), true, 'the loss is prevented, so the browser may restore the context');
  assert.equal(lost.contextRestored, false);
  assert.equal(canvas.restore(), true);
  assert.equal(lost.contextRestored, true, 'the restoration is flagged');
  assert.equal(lost.contextLost, true, 'but its programs and textures died with the loss: it stays unusable');
  assert.equal(lost.canvas, canvas as unknown, 'the canvas to build the next renderer on');

  // the recovery builds the next renderer on the restored context and then lets go of the lost one (`keepContext`): that
  // must not touch the restored context. Deleting dead objects in it only raises INVALID_OPERATION, and releasing it
  // would lose the context the next renderer is built on
  const rebuilt = new Webgl2Renderer(canvas.gl);
  const before = { ...canvas.calls };
  lost.dispose(true);
  assert.deepEqual(canvas.calls, before, 'no GL call on the restored context');
  assert.equal(canvas.contextLost, false, 'the restored context is not released');
  assert.deepEqual([canvas.width, canvas.height], [4, 4], 'nor its canvas resized');
  assert.equal(rebuilt.contextLost, false, 'a new renderer works on the restored context');
  assert.equal(canvas.listenerCount('webglcontextlost'), 1, "the lost renderer's listeners removed");
  assert.equal(canvas.listenerCount('webglcontextrestored'), 1);
  rebuilt.dispose();
  assert.equal(canvas.listenerCount(), 0);
});

test('a lost renderer nobody builds on gives its context back: on the restoration, or at once if restored already', () => {
  const canvas = new SimulatedCanvas();
  const lost = new Webgl2Renderer(canvas.gl);
  // where present() leaves it: a restoration allocated a drawing buffer of the screen's size, 3.4 MB
  canvas.width = 1024;
  canvas.height = 820;
  assert.equal(canvas.lose(), true, 'its loss prevented, the browser may restore the context');
  lost.dispose();
  assert.deepEqual([canvas.width, canvas.height], [1, 1], 'a restoration allocates one pixel');
  assert.equal(canvas.listenerCount('webglcontextlost'), 0, 'nothing asks for another restoration');
  assert.equal(canvas.calls.loseContext, 0, 'nothing to release while it is lost');
  assert.equal(canvas.restore(), true);
  assert.equal(canvas.calls.loseContext, 1, 'released on its restoration');
  assert.equal(canvas.contextLost, true);
  assert.equal(canvas.listenerCount(), 0, 'by a one-shot listener');
  assert.equal(canvas.restore(), false, 'and for good: its loss is not prevented any more');

  const restored = new SimulatedCanvas();
  const r = new Webgl2Renderer(restored.gl);
  restored.lose();
  restored.restore();
  assert.equal(r.contextRestored, true);
  r.dispose();
  assert.equal(restored.calls.loseContext, 1, 'restored and nobody builds on it: released at once');
  assert.equal(restored.contextLost, true);
  assert.equal(restored.listenerCount(), 0);
});

test("a context createWebgl2Renderer made and could not build on is given back; one on the caller's canvas stays", () => {
  const { result, canvases } = withSimulatedOffscreenCanvas({ floatTargets: false }, () =>
    createWebgl2Renderer(),
  );
  assert.equal(result.renderer, null);
  assert.match(result.reason, /EXT_color_buffer_float/, 'the constructor threw after getContext');
  assert.equal(canvases.length, 1, 'on a canvas of its own, as in the worker');
  assert.equal(canvases[0]!.calls.loseContext, 1, 'given back at once, not at the GC');
  assert.equal(canvases[0]!.contextLost, true);

  const own = new SimulatedCanvas({ floatTargets: false });
  assert.equal(createWebgl2Renderer(own as unknown as OffscreenCanvas).renderer, null);
  assert.equal(own.calls.loseContext, 0, "the caller's to keep or release");
  assert.equal(own.contextLost, false);
});

test('a live renderer still deletes its objects and releases its context on dispose', () => {
  const canvas = new SimulatedCanvas();
  const renderer = new Webgl2Renderer(canvas.gl);
  renderer.dispose();
  assert.ok(canvas.calls.deleteProgram >= 12, `programs deleted: ${canvas.calls.deleteProgram}`);
  assert.ok(canvas.calls.deleteTexture >= 4, `textures deleted: ${canvas.calls.deleteTexture}`);
  assert.equal(canvas.calls.loseContext, 1, 'the context is given back (decision 172)');
  assert.equal(canvas.listenerCount(), 0);
});

test('a constructor that throws leaves no listener on the canvas', () => {
  const canvas = new SimulatedCanvas({ floatTargets: false });
  assert.throws(() => new Webgl2Renderer(canvas.gl), /EXT_color_buffer_float/);
  assert.equal(canvas.listenerCount(), 0, 'a renderer that was never built listens for nothing');
  // nor on a context lost again while the recovery builds on it, whose extensions are null
  const lostCanvas = new SimulatedCanvas();
  lostCanvas.loseSilently();
  assert.throws(() => new Webgl2Renderer(lostCanvas.gl), /EXT_color_buffer_float/);
  assert.equal(lostCanvas.listenerCount(), 0);
});

test('the dev hook loses and restores the context through WEBGL_lose_context, taken while it lives', () => {
  const canvas = new SimulatedCanvas();
  const renderer = new Webgl2Renderer(canvas.gl);
  const ext = renderer.loseContextExtension();
  assert.ok(ext, 'available while the context lives');
  ext.loseContext();
  assert.equal(renderer.contextLost, true);
  assert.equal(renderer.loseContextExtension(), null, 'null once lost: it has to be kept from before');
  ext.restoreContext();
  assert.equal(renderer.contextRestored, true, 'restored, since the renderer prevented the default of the loss');
  renderer.dispose();
  assert.equal(canvas.listenerCount(), 0);
});

test('the volume heart uploads its textures again on a restored context instead of binding dead ones', () => {
  const canvas = new SimulatedCanvas();
  const vh = {
    gpu: () => ({
      dims: [2, 2, 2],
      labels: new Uint8Array(8),
      fields: new Float32Array(32),
      nSlices: 1,
      slices: new Float32Array(16),
    }),
  } as unknown as VolumeHeart;
  const hv = volumeHeartGpu(vh, {} as HeartPose);
  const made = () => canvas.calls.createTexture;
  hv.upload(canvas.gl);
  assert.equal(made(), 3, 'labels, fields and the slice table');
  hv.upload(canvas.gl);
  assert.equal(made(), 3, 'kept while the context lives');
  canvas.lose();
  canvas.restore();
  hv.upload(canvas.gl);
  assert.equal(made(), 6, 'the textures of the lost context are made again');
  // and on another context (the recovery's new canvas when no restoration comes)
  const other = new SimulatedCanvas();
  hv.upload(other.gl);
  assert.equal(other.calls.createTexture, 3);
});
