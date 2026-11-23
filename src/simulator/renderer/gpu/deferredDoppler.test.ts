import test from 'node:test';
import assert from 'node:assert/strict';
import { Tissue } from '@/simulator/anatomy/tissue';
import {
  COLOR_THRESHOLD,
  DEFAULT_COLOR,
  DOPPLER_SHADOW_TRANSMISSION,
} from '@/simulator/doppler/color/colorDoppler';
import {
  COLOR_PACKET,
  NOISE_COHERENCE,
  NOISE_POWER,
  NOISE_POWER_GATE,
  TRANSIT_JITTER_RAD,
} from '@/simulator/doppler/color/kasaiEstimator';
import { ATTEN_NP_PER_DB, SOFT_TISSUE_ATTEN_DB } from '../acoustic/acoustics';
import { buildScanLut, computeSectorMapping } from '../scanConvert';
import type { PolarFrameSpec } from '../types';
import {
  computeDopplerBoxBounds,
  DeferredDopplerPipeline,
  type DeferredDopplerSettings,
  type PolarGBuffer,
} from './deferredDoppler';
import {
  ST_HYSTERESIS_RATIO,
  ST_SPATIAL_LAMINAR_MIX,
  ST_SPATIAL_TURB_MIX,
} from '@/simulator/doppler/color/spatiotemporalFilter';
import {
  dopplerDefinesGlsl,
  GLSL_DOPPLER_AUTOCORR_FRAG,
  GLSL_DOPPLER_SHADE_FRAG,
  GLSL_DOPPLER_SPATIAL_FRAG,
} from './glslDoppler';
import { Webgl2Renderer } from './webgl2Renderer';

interface GlCallLog {
  programsCreated: number;
  programsDeleted: number;
  texturesCreated: number;
  texturesDeleted: number;
  fbosCreated: number;
  fbosDeleted: number;
  scissorCalls: Array<[number, number, number, number]>;
  scissorEnabled: number;
  scissorDisabled: number;
  subImageCalls: Array<{ x: number; y: number; w: number; h: number }>;
  texImageCalls: Array<{ internalformat: number; w: number; h: number }>;
  clearBufferCalls: number;
  drawCalls: number;
  uniform1i: Map<string, number>;
  boundTexturesByUnit: Map<number, unknown>;
}

function makeInstrumentedGl(readbackPixels?: Float32Array) {
  let lost = false;
  let activeUnit = 0;
  let idSeq = 1;
  const log: GlCallLog = {
    programsCreated: 0,
    programsDeleted: 0,
    texturesCreated: 0,
    texturesDeleted: 0,
    fbosCreated: 0,
    fbosDeleted: 0,
    scissorCalls: [],
    scissorEnabled: 0,
    scissorDisabled: 0,
    subImageCalls: [],
    texImageCalls: [],
    clearBufferCalls: 0,
    drawCalls: 0,
    uniform1i: new Map(),
    boundTexturesByUnit: new Map(),
  };

  const canvas = {
    width: 64,
    height: 64,
    addEventListener() {},
    removeEventListener() {},
  };

  const target: Record<string, unknown> = {
    canvas,
    COMPILE_STATUS: 0x8b81,
    LINK_STATUS: 0x8b82,
    VERTEX_SHADER: 0x8b31,
    FRAGMENT_SHADER: 0x8b30,
    FRAMEBUFFER_COMPLETE: 0x8cd5,
    SCISSOR_TEST: 0x0c11,
    TEXTURE0: 0x84c0,
    COLOR_ATTACHMENT0: 0x8ce0,
    COLOR: 0x1800,
    RGBA16F: 0x881a,
    RGBA32F: 0x8814,
    RGBA8: 0x8058,
    isContextLost: () => lost,
    getExtension: (name: string) =>
      name === 'EXT_color_buffer_float'
        ? {}
        : name === 'WEBGL_lose_context'
          ? { loseContext: () => { lost = true; } }
          : null,
    createShader: () => ({ id: idSeq++ }),
    createProgram: () => {
      log.programsCreated++;
      return { id: idSeq++ };
    },
    deleteProgram: () => {
      log.programsDeleted++;
    },
    createTexture: () => {
      log.texturesCreated++;
      return { id: idSeq++ };
    },
    deleteTexture: () => {
      log.texturesDeleted++;
    },
    createFramebuffer: () => {
      log.fbosCreated++;
      return { id: idSeq++ };
    },
    deleteFramebuffer: () => {
      log.fbosDeleted++;
    },
    createVertexArray: () => ({ id: idSeq++ }),
    getShaderParameter: () => true,
    getProgramParameter: () => true,
    checkFramebufferStatus: () => 0x8cd5,
    getUniformLocation: (_prog: unknown, name: string) => ({ name }),
    uniform1i: (loc: { name?: string } | null, v: number) => {
      if (loc?.name) log.uniform1i.set(loc.name, v);
    },
    activeTexture: (unit: number) => {
      activeUnit = unit - 0x84c0;
    },
    bindTexture: (_target: number, tex: unknown) => {
      log.boundTexturesByUnit.set(activeUnit, tex);
    },
    enable: (cap: number) => {
      if (cap === 0x0c11) log.scissorEnabled++;
    },
    disable: (cap: number) => {
      if (cap === 0x0c11) log.scissorDisabled++;
    },
    scissor: (x: number, y: number, w: number, h: number) => {
      log.scissorCalls.push([x, y, w, h]);
    },
    texImage2D: (
      _target: number,
      _level: number,
      internalformat: number,
      w: number,
      h: number,
    ) => {
      log.texImageCalls.push({ internalformat, w, h });
    },
    texSubImage2D: (
      _target: number,
      _level: number,
      x: number,
      y: number,
      w: number,
      h: number,
    ) => {
      log.subImageCalls.push({ x, y, w, h });
    },
    clearBufferfv: () => {
      log.clearBufferCalls++;
    },
    drawArrays: () => {
      log.drawCalls++;
    },
    readPixels: (
      _x: number,
      _y: number,
      _w: number,
      _h: number,
      _fmt: number,
      _type: number,
      dst: ArrayBufferView,
    ) => {
      if (readbackPixels && dst instanceof Float32Array) {
        dst.set(readbackPixels.subarray(0, dst.length));
      }
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
    log,
    setLost(v: boolean) {
      lost = v;
    },
  };
}

const TEST_SPEC: PolarFrameSpec = {
  lines: 100,
  samples: 200,
  sectorRad: Math.PI / 2,
  depthCm: 16,
  focusCm: 8,
  elevationSamples: 1,
};

const TEST_DOPPLER: DeferredDopplerSettings = {
  ...DEFAULT_COLOR,
  boxThetaMinRad: -0.25,
  boxThetaMaxRad: 0.25,
  boxRMinCm: 4,
  boxRMaxCm: 12,
};

test('computeDopplerBoxBounds computes exact inclusive polar scissor rect and coverage', () => {
  const b = computeDopplerBoxBounds(TEST_SPEC, TEST_DOPPLER);
  assert.equal(b.empty, false);
  // 4..12 cm of 16 cm across 200 samples -> 50..150 inclusive => 101 samples
  assert.equal(b.siMin, 50);
  assert.equal(b.siMax, 150);
  assert.equal(b.boxW, 101);
  // -0.25..+0.25 rad of pi/2 across 100 lines -> ~34..66 inclusive
  assert.ok(b.liMin >= 33 && b.liMin <= 35);
  assert.ok(b.liMax >= 65 && b.liMax <= 67);
  assert.equal(b.boxH, b.liMax - b.liMin + 1);
  assert.equal(b.totalTexels, b.boxW * b.boxH);
  assert.ok(b.coveragePct > 10 && b.coveragePct < 25, `expected ~16.7% ROI coverage, got ${b.coveragePct}`);

  // Degenerate box outside sector produces empty=true
  const empty = computeDopplerBoxBounds(TEST_SPEC, {
    boxThetaMinRad: 1.2,
    boxThetaMaxRad: 1.5,
    boxRMinCm: 18,
    boxRMaxCm: 20,
  });
  assert.equal(empty.empty, true);
  assert.equal(empty.totalTexels, 0);
});

test('GLSL deferred Doppler shaders synchronize all physical constants with TypeScript modules', () => {
  const defs = dopplerDefinesGlsl();
  assert.ok(defs.includes(`#define T_BLOOD ${Tissue.Blood}`));
  assert.ok(defs.includes(`#define DOPPLER_SHADOW_TRANSMISSION ${DOPPLER_SHADOW_TRANSMISSION}`));
  assert.ok(defs.includes(`#define COLOR_THRESHOLD ${COLOR_THRESHOLD}`));
  assert.ok(defs.includes(`#define COLOR_PACKET ${COLOR_PACKET}`));
  assert.ok(defs.includes(`#define NOISE_POWER ${NOISE_POWER}`));
  assert.ok(defs.includes(`#define TRANSIT_JITTER_RAD ${TRANSIT_JITTER_RAD}`));
  assert.ok(defs.includes(`#define NOISE_COHERENCE ${NOISE_COHERENCE}`));
  assert.ok(defs.includes(`#define NOISE_POWER_GATE ${NOISE_POWER_GATE}`));
  assert.ok(defs.includes(`#define ATTEN_NP_PER_DB ${ATTEN_NP_PER_DB}`));
  assert.ok(defs.includes(`#define SOFT_TISSUE_ATTEN_DB ${SOFT_TISSUE_ATTEN_DB}`));
  assert.ok(defs.includes(`#define ST_HYSTERESIS_RATIO ${ST_HYSTERESIS_RATIO}`));
  assert.ok(defs.includes(`#define ST_SPATIAL_LAMINAR_MIX ${ST_SPATIAL_LAMINAR_MIX}`));
  assert.ok(defs.includes(`#define ST_SPATIAL_TURB_MIX ${ST_SPATIAL_TURB_MIX}`));
  assert.ok(defs.includes('const float KASAI_LIN_BASIS[COLOR_PACKET]'));

  assert.ok(GLSL_DOPPLER_AUTOCORR_FRAG.startsWith('#version 300 es'));
  assert.ok(GLSL_DOPPLER_SHADE_FRAG.startsWith('#version 300 es'));
  assert.ok(GLSL_DOPPLER_SPATIAL_FRAG.startsWith('#version 300 es'));
  assert.ok(GLSL_DOPPLER_SHADE_FRAG.includes('layout(location = 0) out vec4 outShade;'));
  assert.ok(GLSL_DOPPLER_SHADE_FRAG.includes('layout(location = 1) out vec4 outHist;'));
  assert.ok(GLSL_DOPPLER_SPATIAL_FRAG.includes('layout(location = 0) out vec4 outDisplay;'));
  assert.ok(GLSL_DOPPLER_SHADE_FRAG.includes('safeAtan2'));
});

test('DeferredDopplerPipeline allocates lazily, clips to ROI scissor box, uploads only box sub-rect, and tracks independent passes', () => {
  const mock = makeInstrumentedGl();
  const pipe = new DeferredDopplerPipeline(mock.gl);
  assert.equal(pipe.allocated, false, 'must not allocate shaders/FBOs before first Color Doppler pass');
  assert.equal(mock.log.programsCreated, 0);
  assert.equal(mock.log.texturesCreated, 0);

  const gbuf: PolarGBuffer = {
    ready: true,
    w: TEST_SPEC.samples,
    h: TEST_SPEC.lines,
    version: 7,
    envTex: { id: 'env' } as unknown as WebGLTexture,
    idsTex: { id: 'ids' } as unknown as WebGLTexture,
    packedTex: { id: 'packed' } as unknown as WebGLTexture,
  };
  const vao = { id: 'vao' } as unknown as WebGLVertexArrayObject;
  const box = computeDopplerBoxBounds(TEST_SPEC, TEST_DOPPLER);
  const rawFlow = new Float32Array(box.boxW * box.boxH * 4);

  // First deferred pass: lazy allocation + sub-rect upload + 2 scissor-clipped draws (D1 + D2)
  const ok1 = pipe.shade(gbuf, vao, {
    spec: TEST_SPEC,
    settings: TEST_DOPPLER,
    frequencyMHz: 2.5,
    harmonics: true,
    rawFlowBox: rawFlow,
    estimate: { realization: 0, seed: 42 },
  });
  assert.equal(ok1, true);
  assert.equal(pipe.allocated, true);
  assert.equal(pipe.valid, true);
  assert.equal(mock.log.programsCreated, 3, 'Pass D1 + Pass D2 + Pass D3 spatial programs');
  assert.equal(mock.log.subImageCalls.length, 1, 'uploads only the ROI box sub-rectangle');
  assert.deepEqual(mock.log.subImageCalls[0], {
    x: box.siMin,
    y: box.liMin,
    w: box.boxW,
    h: box.boxH,
  });
  assert.equal(mock.log.scissorCalls.length, 3, 'all 3 passes (D1, D2, D3) must be scissor-clipped');
  assert.deepEqual(mock.log.scissorCalls[0], [box.siMin, box.liMin, box.boxW, box.boxH]);
  assert.deepEqual(mock.log.scissorCalls[1], [box.siMin, box.liMin, box.boxW, box.boxH]);
  assert.deepEqual(mock.log.scissorCalls[2], [box.siMin, box.liMin, box.boxW, box.boxH]);
  assert.equal(mock.log.drawCalls, 3);
  assert.equal(mock.log.uniform1i.get('uUseHistory'), 0, 'first frame has no prior phasor history');
  assert.equal(pipe.stats().independentPasses, 0);

  // Second pass on the SAME Polar G-Buffer version WITHOUT rawFlowBox (settings-only re-shade!)
  const ok2 = pipe.shade(gbuf, vao, {
    spec: TEST_SPEC,
    settings: { ...TEST_DOPPLER, gainDb: 4 },
    frequencyMHz: 2.5,
    harmonics: true,
    estimate: { realization: 1, seed: 42 },
  });
  assert.equal(ok2, true);
  assert.equal(mock.log.subImageCalls.length, 1, 'settings-only re-shade reuses GPU-resident texFlowRaw with 0 uploads');
  assert.equal(mock.log.uniform1i.get('uUseHistory'), 1, 'identical scale/baseline/invert preserves circular phasor history');
  assert.equal(pipe.stats().independentPasses, 1, 'tracks independent Doppler pass without B-mode rebuild');

  // Changing Nyquist scale invalidates circular phasor history without touching G-Buffer
  const ok3 = pipe.shade(gbuf, vao, {
    spec: TEST_SPEC,
    settings: { ...TEST_DOPPLER, scaleMps: 0.9 },
    frequencyMHz: 2.5,
    harmonics: true,
    estimate: { realization: 2, seed: 42 },
  });
  assert.equal(ok3, true);
  assert.equal(mock.log.uniform1i.get('uUseHistory'), 0, 'scale change must invalidate circular phasor history');
  assert.equal(pipe.stats().independentPasses, 2);

  // Dispose releases every program, texture and framebuffer allocated by the pipeline
  const createdProgs = mock.log.programsCreated;
  const createdTex = mock.log.texturesCreated;
  const createdFbos = mock.log.fbosCreated;
  pipe.dispose();
  assert.equal(mock.log.programsDeleted, createdProgs);
  assert.equal(mock.log.texturesDeleted, createdTex);
  assert.equal(mock.log.fbosDeleted, createdFbos);
  assert.equal(pipe.valid, false);
});

test('DeferredDopplerPipeline.readbackBox decodes complex phasors back into aliased CPU velocities', () => {
  const box = computeDopplerBoxBounds(TEST_SPEC, TEST_DOPPLER);
  const scale = 0.62;
  const vExpected = 0.45;
  const phi = (Math.PI * vExpected) / scale;
  const fakeReadback = new Float32Array(box.boxW * box.boxH * 4);
  // Populate texel (dx=2, dy=3) inside box with valid unit phasor + variance 0.35
  const idx = (3 * box.boxW + 2) * 4;
  fakeReadback[idx] = Math.cos(phi);
  fakeReadback[idx + 1] = Math.sin(phi);
  fakeReadback[idx + 2] = 0.35;
  fakeReadback[idx + 3] = 1;

  const mock = makeInstrumentedGl(fakeReadback);
  const pipe = new DeferredDopplerPipeline(mock.gl);
  const gbuf: PolarGBuffer = {
    ready: true,
    w: TEST_SPEC.samples,
    h: TEST_SPEC.lines,
    version: 1,
    envTex: {} as WebGLTexture,
    idsTex: {} as WebGLTexture,
    packedTex: {} as WebGLTexture,
  };
  pipe.shade(gbuf, {} as WebGLVertexArrayObject, {
    spec: TEST_SPEC,
    settings: { ...TEST_DOPPLER, scaleMps: scale, baselineShiftMps: 0 },
    frequencyMHz: 2.5,
    harmonics: true,
    rawFlowBox: new Float32Array(box.boxW * box.boxH * 4),
  });

  const velOut = new Float32Array(TEST_SPEC.lines * TEST_SPEC.samples);
  const varOut = new Float32Array(TEST_SPEC.lines * TEST_SPEC.samples);
  assert.equal(pipe.readbackBox(TEST_SPEC, scale, 0, velOut, varOut), true);

  const polarIdx = (box.liMin + 3) * TEST_SPEC.samples + (box.siMin + 2);
  assert.ok(Math.abs(velOut[polarIdx]! - vExpected) < 1e-5);
  assert.ok(Math.abs(varOut[polarIdx]! - 0.35) < 1e-6);
  assert.ok(Number.isNaN(velOut[0]!), 'outside ROI box stays NaN');

  // Verify non-zero baselineShiftMps decodes baseline-centered phase without wrap distortion
  const shift = 0.25;
  const vShiftedExpected = 0.82; // inside [-0.35, +0.85] at scale=0.60, baseline=+0.25
  const phiShifted = (Math.PI * (vShiftedExpected - shift)) / scale;
  fakeReadback[idx] = Math.cos(phiShifted);
  fakeReadback[idx + 1] = Math.sin(phiShifted);
  fakeReadback[idx + 3] = 0.78; // valid power-modulated alpha >= 0.5
  assert.equal(pipe.readbackBox(TEST_SPEC, scale, shift, velOut, varOut), true);
  assert.ok(
    Math.abs(velOut[polarIdx]! - vShiftedExpected) < 1e-5,
    `expected shifted readback ${vShiftedExpected}, got ${velOut[polarIdx]}`,
  );
  pipe.dispose();
});

test('Webgl2Renderer defers Doppler allocation in B-mode and binds GPU-resident deferred texture in present() with zero CPU color upload', () => {
  const mock = makeInstrumentedGl();
  const renderer = new Webgl2Renderer(mock.gl);
  assert.equal(renderer.dopplerStats().allocated, false, 'zero Doppler VRAM in 2D B-mode');
  assert.equal(renderer.hasDeferredDoppler, false);
  assert.equal(renderer.stats()['doppler'], 'idle');

  // Calling shadeDoppler before G-Buffer is ready returns false safely
  assert.equal(
    renderer.shadeDoppler({
      spec: TEST_SPEC,
      settings: TEST_DOPPLER,
      frequencyMHz: 2.5,
      harmonics: true,
      rawFlowBox: new Float32Array(4),
    }),
    false,
  );

  // Simulate ready Polar G-Buffer on renderer targets to verify deferred present() zero-upload path
  (renderer as unknown as { targets: { ensure: (w: number, h: number) => boolean } }).targets.ensure(
    TEST_SPEC.samples,
    TEST_SPEC.lines,
  );
  (renderer as unknown as { gbufferReady: boolean; gbufferVersion: number; packedFresh: boolean }).gbufferReady =
    true;
  (renderer as unknown as { gbufferVersion: number }).gbufferVersion = 3;
  (renderer as unknown as { packedFresh: boolean }).packedFresh = true;

  const box = computeDopplerBoxBounds(TEST_SPEC, TEST_DOPPLER);
  const shaded = renderer.shadeDoppler({
    spec: TEST_SPEC,
    settings: TEST_DOPPLER,
    frequencyMHz: 2.5,
    harmonics: true,
    rawFlowBox: new Float32Array(box.boxW * box.boxH * 4),
    estimate: { realization: 0, seed: 7 },
  });
  assert.equal(shaded, true);
  assert.equal(renderer.hasDeferredDoppler, true);
  assert.equal(renderer.gbuffer.version, 3, 'shadeDoppler must NOT re-run or mutate B-mode G-Buffer');
  assert.equal(renderer.stats()['doppler'], 'gpu-deferred');

  // Present with deferred color must NOT upload colorTex via texImage2D(RGBA16F)
  const mapping = computeSectorMapping(TEST_SPEC, 64, 64, false, 1);
  const lut = buildScanLut(TEST_SPEC, mapping);
  const texImagesBefore = mock.log.texImageCalls.length;
  renderer.present({
    lut,
    width: 64,
    height: 64,
    color: { deferred: true, settings: TEST_DOPPLER },
  });
  const newUploads = mock.log.texImageCalls.slice(texImagesBefore);
  const colorUploads = newUploads.filter((u) => u.internalformat === 0x881a); // RGBA16F
  assert.equal(colorUploads.length, 0, 'deferred present() must bypass CPU RGBA16F color upload completely');
  assert.equal(mock.log.uniform1i.get('uColorOn'), 1);

  // Context loss safely short-circuits shadeDoppler
  mock.setLost(true);
  assert.equal(
    renderer.shadeDoppler({
      spec: TEST_SPEC,
      settings: TEST_DOPPLER,
      frequencyMHz: 2.5,
      harmonics: true,
    }),
    false,
  );

  renderer.dispose();
});
