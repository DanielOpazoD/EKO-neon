import test from 'node:test';
import assert from 'node:assert/strict';
import { Webgl2Renderer } from './webgl2Renderer';

/** Minimal Proxy-backed WebGL2 mock sufficient to exercise Webgl2Renderer lifecycle & context-loss guards. */
function makeMockGl() {
  const listeners = new Map<string, Set<(e: Event) => void>>();
  let lost = false;
  let readPixelsCalls = 0;

  const canvas = {
    width: 4,
    height: 4,
    addEventListener(type: string, fn: (e: Event) => void) {
      let set = listeners.get(type);
      if (!set) {
        set = new Set();
        listeners.set(type, set);
      }
      set.add(fn);
    },
    removeEventListener(type: string, fn: (e: Event) => void) {
      listeners.get(type)?.delete(fn);
    },
  };

  const target: Record<string, unknown> = {
    canvas,
    COMPILE_STATUS: 0x8b81,
    LINK_STATUS: 0x8b82,
    VERTEX_SHADER: 0x8b31,
    FRAGMENT_SHADER: 0x8b30,
    isContextLost: () => lost,
    getExtension: (name: string) =>
      name === 'EXT_color_buffer_float'
        ? {}
        : name === 'WEBGL_lose_context'
          ? { loseContext: () => { lost = true; } }
          : null,
    createShader: () => ({}),
    createProgram: () => ({}),
    createTexture: () => ({}),
    createFramebuffer: () => ({}),
    createVertexArray: () => ({}),
    getShaderParameter: () => true,
    getProgramParameter: () => true,
    getUniformLocation: () => ({}),
    readPixels: () => {
      readPixelsCalls++;
    },
  };

  const gl = new Proxy(target, {
    get(obj, prop) {
      if (prop in obj) return obj[prop as string];
      return () => {};
    },
  }) as unknown as WebGL2RenderingContext;

  return {
    gl,
    get listenerCount() {
      return listeners.get('webglcontextlost')?.size ?? 0;
    },
    get readPixelsCalls() {
      return readPixelsCalls;
    },
    fireContextLost() {
      let prevented = false;
      const evt = {
        preventDefault() {
          prevented = true;
        },
      } as unknown as Event;
      for (const fn of listeners.get('webglcontextlost') ?? []) fn(evt);
      return prevented;
    },
    setDriverLost(v: boolean) {
      lost = v;
    },
  };
}

test('Webgl2Renderer intercepts webglcontextlost, short-circuits rendering, and cleans up on dispose', () => {
  const mock = makeMockGl();
  const renderer = new Webgl2Renderer(mock.gl);
  assert.equal(mock.listenerCount, 1, 'must subscribe to webglcontextlost on canvas');
  assert.equal(renderer.contextLost, false);

  // Fire webglcontextlost event
  const prevented = mock.fireContextLost();
  assert.equal(prevented, true, 'must call preventDefault() on webglcontextlost');
  assert.equal(renderer.contextLost, true, 'contextLost must flip to true immediately');

  // Calling render / renderDisplay / present after loss must safely no-op without touching GL
  renderer.render(
    {} as never,
    {} as never,
    { lines: 4, samples: 4, sectorRad: 1, depthCm: 10, focusCm: 6, elevationSamples: 1 },
    0,
    {} as never,
  );
  const displayOk = renderer.renderDisplay(
    {} as never,
    {} as never,
    { lines: 4, samples: 4, sectorRad: 1, depthCm: 10, focusCm: 6, elevationSamples: 1 },
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
  assert.equal(mock.readPixelsCalls, 0, 'must not issue readPixels after context loss');

  // Dispose must remove the event listener
  renderer.dispose();
  assert.equal(mock.listenerCount, 0, 'dispose() must remove webglcontextlost listener');
});

test('Webgl2Renderer also detects driver-side gl.isContextLost() without a DOM event', () => {
  const mock = makeMockGl();
  const renderer = new Webgl2Renderer(mock.gl);
  assert.equal(renderer.contextLost, false);
  mock.setDriverLost(true);
  assert.equal(renderer.contextLost, true);
  renderer.dispose();
});
