import assert from 'node:assert/strict';
import test from 'node:test';
import { loadCaseById } from '@/cases';
import { buildScanLut, computeSectorMapping, scanConvertLut } from '@/simulator/renderer/scanConvert';
import { baseInput } from './baseInput';
import type { CineBuffer } from './cineBuffer';
import type { SimInput } from './protocol';
import { SimulatorCore } from './simulatorCore';
import type { StripEngine } from './stripEngine';

const caseDef = loadCaseById('normal-excellent-window');

function internals(core: SimulatorCore): { cine: CineBuffer; strips: StripEngine } {
  return core as unknown as { cine: CineBuffer; strips: StripEngine };
}

function stepLive(core: SimulatorCore, ticks: number): void {
  for (let i = 0; i < ticks; i++) {
    const out = core.step(1 / 30);
    if (out) core.recycle(out.rgba);
  }
}

test('frozen, an unchanged image is not composed again; a cine scroll or a resize answers at the next tick', () => {
  const live = baseInput();
  const core = new SimulatorCore(caseDef, live);
  stepLive(core, 8);
  const newest = internals(core).cine.newest!.frameId;
  const frozen = { ...live, frozen: true };
  core.setInput(frozen);
  const first = core.step(0.08);
  assert.ok(first?.frozen);
  assert.equal(first.frameId, newest);
  core.recycle(first.rgba);
  // the learner measures on it while the worker ticks every 80 ms: 12.5 whole frames a second were composed and sent
  for (let i = 0; i < 25; i++) assert.equal(core.step(0.08), null, `tick ${i}: nothing changed`);

  core.setInput({ ...frozen, cineOffset: -3 });
  const scrolled = core.step(0.08);
  assert.ok(scrolled?.frozen);
  assert.equal(scrolled.frameId, newest - 3);
  assert.equal(scrolled.cineOffset, -3);
  core.recycle(scrolled.rgba);
  assert.equal(core.step(0.08), null);

  core.setInput({ ...frozen, cineOffset: -3, display: { width: 400, height: 300 } });
  const resized = core.step(0.08);
  assert.equal(resized?.frameId, newest - 3);
  assert.equal(resized.width, 400);
  core.recycle(resized.rgba);
  assert.equal(core.step(0.08), null);

  // live again, then frozen: the newest frame is sent again
  core.setInput(live);
  stepLive(core, 3);
  core.setInput(frozen);
  assert.equal(core.step(0.08)?.frameId, internals(core).cine.newest!.frameId);
  core.dispose();
});

test('frozen over a strip, the frame is formed again when the strip history it overlays is reset', () => {
  const live = baseInput({ modality: 'pw' });
  const core = new SimulatorCore(caseDef, live);
  stepLive(core, 12);
  core.setInput({ ...live, frozen: true });
  const first = core.step(0.08);
  assert.equal(first?.strip.kind, 'spectral');
  core.recycle(first.rgba);
  assert.equal(core.step(0.08), null);
  // an experimental asset arriving resets the strips (SimulatorCore's loader callback): the screen must follow the core
  internals(core).strips.reset();
  const after = core.step(0.08);
  assert.ok(after?.frozen);
  assert.notEqual(after.strip.acquisitionId, first.strip.acquisitionId);
  core.dispose();
});

test('with no room to present, the core acquires on without composing, and a frozen change waits for room', () => {
  const live = baseInput({ modality: 'pw' });
  const core = new SimulatorCore(caseDef, live);
  stepLive(core, 6);
  const { cine } = internals(core);
  const before = cine.newest!.frameId;
  const head = core.spectralStrip.head;
  // the worker composed these frames, bitmap and all, and then dropped them
  for (let i = 0; i < 10; i++) assert.equal(core.step(1 / 30, false), null, `tick ${i}`);
  assert.ok(cine.newest!.frameId > before, 'the 2D went on being acquired');
  assert.notEqual(core.spectralStrip.head, head, 'and the sweep went on');
  assert.equal(core.droppedFrames, 10);
  const next = core.step(1 / 30);
  assert.ok(next);
  assert.equal(next.frameId, cine.newest!.frameId, 'the next frame with room is the newest');
  core.recycle(next.rgba);

  core.setInput({ ...live, frozen: true });
  assert.equal(core.step(0.08, false), null);
  const shown = core.step(0.08);
  assert.ok(shown?.frozen, 'the frozen frame that found no room is sent at the next tick');
  assert.equal(core.droppedFrames, 10, 'waiting to send a frozen frame is not dropping one');
  core.dispose();
});

test('the sector above a strip is the scan conversion of the frame it shows, on every tick', () => {
  // the sector is converted once per 2D frame and copied on the ticks that only move the sweep; a new frame, tint map or
  // size converts it again
  const live = baseInput({ modality: 'pw', display: { width: 480, height: 400 } });
  const core = new SimulatorCore(caseDef, live);
  const { cine } = internals(core);
  let sweepOnly = 0;
  let lastFrameId = -1;
  const check = (inp: SimInput, ticks: number): void => {
    for (let i = 0; i < ticks; i++) {
      const out = core.step(1 / 30);
      assert.ok(out);
      const cf = cine.newest!;
      assert.equal(out.frameId, cf.frameId);
      if (cf.frameId === lastFrameId) sweepOnly++;
      lastFrameId = cf.frameId;
      const m = computeSectorMapping(
        cf.spec,
        out.width,
        out.sector.height,
        inp.settings.invertLR,
        inp.settings.zoom,
      );
      const expected = new Uint8ClampedArray(out.width * out.sector.height * 4);
      scanConvertLut(cf.display, buildScanLut(cf.spec, m), expected, inp.settings.tintMap);
      const got = new Uint8ClampedArray(out.rgba, 0, expected.length);
      let differing = 0;
      for (let k = 0; k < expected.length; k++) if (got[k] !== expected[k]) differing++;
      assert.equal(differing, 0, `frame ${cf.frameId}, ${out.width}×${out.height}: sector bytes differ`);
      core.recycle(out.rgba);
    }
  };
  check(live, 30);
  const tinted = { ...live, settings: { ...live.settings, tintMap: 'sepia' as const } };
  core.setInput(tinted);
  check(tinted, 10);
  const resized = { ...tinted, display: { width: 560, height: 420 } };
  core.setInput(resized);
  check(resized, 10);
  assert.ok(sweepOnly >= 10, `${sweepOnly} of 50 ticks moved only the sweep`);
  core.dispose();
});
