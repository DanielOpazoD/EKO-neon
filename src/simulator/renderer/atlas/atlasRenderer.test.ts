import assert from 'node:assert/strict';
import test from 'node:test';
import type { BeamFrame } from '@/simulator/probe/pose';
import {
  allocPolarFrame,
  DEFAULT_ACQUISITION,
  type PolarFrame,
  type PolarFrameSpec,
  type RendererBackend,
  type Scene,
} from '../types';
import { applyConsole, createConsoleState, NOISE_FLOOR } from '../postprocess/consolePipeline';
import { AtlasRenderer, ENTER_AFTER_FRAMES } from './atlasRenderer';

test('AtlasRenderer preserves weak echoes below the former 16-bit linear floor and high specular amplitudes without quantization loss', () => {
  const spec: PolarFrameSpec = {
    lines: 8,
    samples: 16,
    elevationSamples: 1,
    sectorRad: 1.2,
    depthCm: 16,
    focusCm: 8,
  };
  const n = spec.lines * spec.samples;
  // Values spanning weak blood/far-field echoes below 1/2048 (~4.88e-4) up to strong specular peaks (>32)
  const pattern = new Float32Array(n);
  const probeAmps = [0, 5e-6, 2e-5, 1e-4, 2.5e-4, NOISE_FLOOR, 8e-4, 0.015, 1.2, 48.5];
  for (let i = 0; i < n; i++) pattern[i] = probeAmps[i % probeAmps.length]!;

  let clockMs = 0;
  let sourceRenders = 0;
  const slowSource: RendererBackend = {
    id: 'procedural',
    render(_scene, _beam, _spec, _phase, out) {
      clockMs += 50; // over budget to trigger cache mode
      sourceRenders++;
      out.amplitude.set(pattern);
      out.transmission.fill(0.5);
      out.structure.fill(1);
      out.tissue.fill(2);
      out.segment.fill(3);
    },
    stats: () => ({}),
    dispose: () => {},
  };

  const atlas = new AtlasRenderer(slowSource, 7, () => clockMs);
  const beam: BeamFrame = {
    origin: { x: 0, y: 0, z: 0 },
    forward: { x: 0, y: 0, z: 1 },
    lateral: { x: 1, y: 0, z: 0 },
    normal: { x: 0, y: 1, z: 0 },
    contact: 1,
  };
  const scene = {
    physics: {
      frequencyMHz: 3,
      harmonics: true,
      clutterLevel: 0.2,
      windowAttenuation: 0,
      seed: 7,
    },
  } as unknown as Scene;

  const out: PolarFrame = allocPolarFrame(spec);
  const hints = { stationary: true, budgetMs: 10, sceneAtPhase: () => scene };

  // Enter cache mode after ENTER_AFTER_FRAMES slow direct renders, then store & serve slot 0
  for (let k = 0; k < ENTER_AFTER_FRAMES + 1; k++) {
    atlas.render(scene, beam, spec, 0, out, hints);
  }
  const rendersBeforeServe = sourceRenders;
  out.amplitude.fill(-1);
  atlas.render(scene, beam, spec, 0, out, hints);

  assert.equal(sourceRenders, rendersBeforeServe, 'slot 0 should be served from the atlas cache');
  assert.equal(atlas.stats()['served'], 'cache');
  for (let i = 0; i < n; i++) {
    assert.equal(
      out.amplitude[i],
      pattern[i],
      `sample ${i} (amp=${pattern[i]}): cached amplitude must match source Float32 without weak-echo truncation`,
    );
  }

  // Console output with high gain (+15 dB) matches direct render identically on weak echoes
  const directFrame = allocPolarFrame(spec);
  directFrame.amplitude.set(pattern);
  directFrame.transmission.fill(0.5);
  const settings = { ...DEFAULT_ACQUISITION, gainDb: 15, persistence: 0, sri: 0, edgeEnhance: 0 };
  const dispDirect = new Uint8ClampedArray(n);
  const dispCached = new Uint8ClampedArray(n);
  applyConsole(directFrame, settings, createConsoleState(7), dispDirect);
  applyConsole(out, settings, createConsoleState(7), dispCached);
  assert.deepEqual(Array.from(dispCached), Array.from(dispDirect));
});
