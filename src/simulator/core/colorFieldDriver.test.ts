import assert from 'node:assert/strict';
import test from 'node:test';
import { loadCaseById } from '@/cases';
import { buildCaseModels } from '@/simulator/anatomy/caseModels';
import type { BeatTables } from '@/simulator/cardiac-cycle/cycleModel';
import type { FlowFieldParams } from '@/simulator/doppler/flow-primitives/flowField';
import type { Webgl2Renderer } from '@/simulator/renderer/gpu/webgl2Renderer';
import { encodeTransmission } from '@/simulator/renderer/transmissionCode';
import type { PolarFrameSpec } from '@/simulator/renderer/types';
import { controlThroughCrux, getViewTarget } from '@/simulator/windows/viewTargets';
import { baseInput } from './baseInput';
import type { CineBuffer, CineFrame } from './cineBuffer';
import { ColorFieldDriver, type ColorUpdateContext } from './colorFieldDriver';
import type { SimInput } from './protocol';
import { SimulatorCore } from './simulatorCore';

/** Colour in the apical four-chamber view, where the default box holds the mitral inflow. */
function apicalColourInput(caseId: string): SimInput {
  const caseDef = loadCaseById(caseId);
  const models = buildCaseModels(caseDef, baseInput().patient);
  const probe = controlThroughCrux(getViewTarget('a4c'), models.heart, models.thorax, null);
  return baseInput({ modality: 'color', probe });
}

const cineOf = (core: SimulatorCore): CineBuffer => (core as unknown as { cine: CineBuffer }).cine;

/** Cine offset (−(length−1)..0) of a frame still in the cine. */
function offsetOf(cine: CineBuffer, cf: CineFrame): number {
  for (let o = 0; o > -cine.length; o--) if (cine.select(o) === cf) return o;
  throw new Error(`frame ${cf.frameId} left the cine`);
}

/**
 * Turn the newest frame into what renderFrame keeps of it on the GPU deferred path (zero readback): no colour field, the
 * frame's tissue map and its 8-bit transmission codes. Node has no WebGL, so the core forms colour on the CPU path.
 */
function asGpuDeferred(cf: CineFrame, core: SimulatorCore): void {
  const frame = core.lastFrame!;
  cf.colorVel = null;
  cf.colorVar = null;
  cf.colorPower = null;
  cf.colorTissue = frame.tissue.slice();
  cf.colorTransCode = Uint8Array.from(frame.transmission, (t) => encodeTransmission(t));
}

/** Show cine frame `offset` frozen for one tick, then resume live acquisition. */
function showFrozen(core: SimulatorCore, live: SimInput, offset: number): void {
  core.setInput({ ...live, frozen: true, cineOffset: offset });
  core.step(1 / 30);
  core.setInput(live);
}

/** Step live until the cine holds a new frame, and return it. */
function nextFrame(core: SimulatorCore): CineFrame {
  const cine = cineOf(core);
  const before = cine.newest?.frameId;
  for (let k = 0; k < 20; k++) {
    core.step(1 / 30);
    if (cine.newest && cine.newest.frameId !== before) return cine.newest;
  }
  throw new Error('no frame formed');
}

function sameField(a: Float32Array, b: Float32Array): number {
  let differing = 0;
  for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) differing++;
  return differing;
}

test('a frozen GPU-deferred colour frame is formed from its own beat, whichever beat is live (atrial fibrillation)', () => {
  // In AF every beat has its own RR and tables, and a colour cine (96 frames at 10 frames/s here) spans 13 beats. Before,
  // frozenOutput formed a scrolled-back frame's colour with the LIVE beat's tables and flow at the frame's phase: a
  // frame of a beat in diastasis showed the live beat's mitral inflow jet (1231 coloured samples where its own beat has
  // none), and 82 of the 88 frames of older beats changed. Only frames acquired in the live beat came out right.
  const caseDef = loadCaseById('af-diastolic');
  const live = apicalColourInput(caseDef.id);
  const core = new SimulatorCore(caseDef, live);
  const cine = cineOf(core);

  // 1. Two beats: right after its capture each frame's beat is still live, so the frozen colour formed then is its own.
  const formedInItsBeat = new Map<CineFrame, Float32Array>();
  const ownTables = new Map<CineFrame, BeatTables>();
  const firstBeat = nextFrame(core).beatIndex;
  while (cine.newest!.beatIndex < firstBeat + 2) {
    const cf = nextFrame(core);
    asGpuDeferred(cf, core);
    showFrozen(core, live, 0);
    assert.ok(cf.colorVel, `frame ${cf.frameId}: frozen colour formed`);
    formedInItsBeat.set(cf, cf.colorVel);
    ownTables.set(cf, core.models.tables);
    cf.colorVel = cf.colorVar = cf.colorPower = null; // form it again on the next freeze
  }
  assert.ok(formedInItsBeat.size >= 10, `${formedInItsBeat.size} frames`);

  // 2. Three more beats, then scroll back to the same frames: the same colour.
  const lastBeat = cine.newest!.beatIndex + 3;
  while (cine.newest!.beatIndex < lastBeat) nextFrame(core);
  for (const [cf, formed] of formedInItsBeat) {
    showFrozen(core, live, offsetOf(cine, cf));
    assert.ok(cf.colorVel, `frame ${cf.frameId}: frozen colour formed`);
    assert.equal(
      sameField(cf.colorVel, formed),
      0,
      `frame ${cf.frameId} (beat ${cf.beatIndex}, phase ${cf.phase.toFixed(3)}): frozen colour depends on the live beat`,
    );
    assert.notEqual(ownTables.get(cf), core.models.tables, 'frame of an older beat');
  }

  // 3. Not vacuous: formed with the live beat instead, as frozenOutput did, nearly every one of them comes out different.
  const liveBeat = cine.newest!;
  const driver = new ColorFieldDriver();
  let differ = 0;
  for (const [cf, formed] of formedInItsBeat) {
    const withLiveBeat: CineFrame = {
      ...cf,
      tables: liveBeat.tables,
      flow: liveBeat.flow,
      colorVel: null,
      colorVar: null,
      colorPower: null,
    };
    driver.reconstructCineColor(withLiveBeat, caseDef.seed);
    if (sameField(withLiveBeat.colorVel!, formed) > 0) differ++;
  }
  assert.ok(differ >= 0.8 * formedInItsBeat.size, `${differ}/${formedInItsBeat.size} differ with the live beat`);
  core.dispose();
});

/** A fake deferred GPU that presents `shown`: what a frame looked like on screen, recognisable from any CPU estimate. */
function fakeDeferredGpu(spec: PolarFrameSpec) {
  const n = spec.lines * spec.samples;
  const shown = {
    vel: new Float32Array(n).fill(Number.NaN),
    variance: new Float32Array(n),
    power: new Float32Array(n),
  };
  for (let i = 0; i < n; i += 7) {
    shown.vel[i] = 0.3141;
    shown.variance[i] = 0.27;
    shown.power[i] = 0.42;
  }
  let readbacks = 0;
  const gpu = {
    hasDeferredDoppler: true,
    shadeDoppler: () => true,
    readbackDopplerBox: (
      s: PolarFrameSpec,
      _scaleMps: number,
      _baselineShiftMps: number,
      vel: Float32Array,
      variance: Float32Array,
      power?: Float32Array,
    ) => {
      assert.equal(s.lines * s.samples, n);
      readbacks++;
      vel.set(shown.vel);
      variance.set(shown.variance);
      power?.set(shown.power);
      return true;
    },
  } as unknown as Webgl2Renderer;
  return {
    gpu,
    shown,
    get readbacks() {
      return readbacks;
    },
  };
}

/**
 * The colour update the newest cine frame was acquired with, as renderFrame builds it: the core's own context, which is
 * that frame's (sinus rhythm at expiration keeps one beat context).
 */
function updateContextOf(cf: CineFrame, core: SimulatorCore): ColorUpdateContext {
  const a = cf.acquisition;
  return {
    frame: core.lastFrame!,
    scene: cf.scene,
    heart: cf.scene.heart,
    flow: (core as unknown as { flow: FlowFieldParams }).flow,
    tables: core.models.tables,
    beam: cf.beam,
    spec: cf.spec,
    phase: cf.phase,
    probe: a.probe,
    color: a.color,
    frequencyMHz: a.settings.frequencyMHz,
    harmonics: a.settings.harmonics,
    modelVersion: a.modelVersion,
    timeS: cf.timeS,
    frameIntervalS: 1 / 30,
    seed: 7,
  };
}

test('freezing on the GPU deferred path shows the newest frame as it was on screen and re-forms older ones', () => {
  const live = apicalColourInput('normal-excellent');
  const core = new SimulatorCore(loadCaseById('normal-excellent'), live);
  const cine = cineOf(core);
  const older = nextFrame(core);
  asGpuDeferred(older, core);
  const newest = nextFrame(core);
  asGpuDeferred(newest, core);
  const fake = fakeDeferredGpu(newest.spec);
  const internals = core as unknown as {
    gpu: Webgl2Renderer | null;
    dopplerOnGpu: boolean;
    colorDriver: ColorFieldDriver;
  };
  // the deferred shade the GPU is presenting over the newest frame
  assert.equal(internals.colorDriver.stepDeferredGpu(updateContextOf(newest, core), fake.gpu), true);
  internals.gpu = fake.gpu;
  internals.dopplerOnGpu = true;

  showFrozen(core, live, 0);
  assert.equal(fake.readbacks, 1);
  assert.equal(sameField(newest.colorVel!, fake.shown.vel), 0, 'frozen colour is the field on screen');
  assert.equal(sameField(newest.colorVar!, fake.shown.variance), 0);
  assert.equal(sameField(newest.colorPower!, fake.shown.power), 0);

  // an older frame has no GPU field left: it is re-estimated on the CPU, from its own beat
  showFrozen(core, live, offsetOf(cine, older));
  assert.equal(fake.readbacks, 1);
  assert.ok(older.colorVel);
  assert.ok(sameField(older.colorVel, fake.shown.vel) > 0);

  // the snapshot is kept: back on the newest frame there is no second readback
  showFrozen(core, live, 0);
  assert.equal(fake.readbacks, 1);
  internals.gpu = null;
  core.dispose();
});

test('the GPU field is not taken for a frame it was not shaded for', () => {
  const live = apicalColourInput('normal-excellent');
  const core = new SimulatorCore(loadCaseById('normal-excellent'), live);
  const cf = nextFrame(core);
  asGpuDeferred(cf, core);
  const fake = fakeDeferredGpu(cf.spec);
  const driver = new ColorFieldDriver();
  // nothing shaded yet
  assert.equal(driver.readbackDeferredGpu(cf, fake.gpu), false);
  // a settings re-shade after the frame (gain turned up): the resident field is no longer the frame's
  const ctx = updateContextOf(cf, core);
  const louder = { ...ctx, color: { ...ctx.color, gainDb: ctx.color.gainDb + 3 } };
  assert.equal(driver.stepDeferredGpu(louder, fake.gpu), true);
  assert.equal(driver.readbackDeferredGpu(cf, fake.gpu), false);
  // the probe moved since the frame: another geometry
  driver.resetHistory();
  const turned = { ...ctx, probe: { ...ctx.probe, rotationDeg: ctx.probe.rotationDeg + 5 } };
  assert.equal(driver.stepDeferredGpu(turned, fake.gpu), true);
  assert.equal(driver.readbackDeferredGpu(cf, fake.gpu), false);
  assert.equal(fake.readbacks, 0);
  assert.equal(cf.colorVel, null, 'frame untouched');
  // shaded for exactly this frame: taken
  assert.equal(driver.stepDeferredGpu(ctx, fake.gpu), true);
  assert.equal(driver.readbackDeferredGpu(cf, fake.gpu), true);
  assert.equal(sameField(cf.colorVel!, fake.shown.vel), 0);
  core.dispose();
});
