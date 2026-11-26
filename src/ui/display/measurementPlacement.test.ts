import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import type { SimOutput, StripInfo } from '@/simulator/core/protocol';
import type { Measurement } from '@/simulator/measurements/types';
import { ecgTimeAtX, ecgLayoutOf } from '../ecgTrace';
import {
  measurementGeometryOn,
  pendingAcquisitionKey,
  reprojectStripGeometry,
} from './measurementPlacement';

/** A frozen PW strip of 2 s over 800 px, its sweep head at column 500, newest column at t = 10 s. */
const STRIP: StripInfo = {
  x: 0,
  y: 300,
  width: 800,
  height: 200,
  secondsPerColumn: 2 / 800,
  headColumn: 500,
  columns: 800,
  topValue: 1,
  bottomValue: -1,
  kind: 'spectral',
  acquisitionId: 7,
  headTimeS: 10,
};
/** The same acquisition redrawn on a narrower display (the strip keeps its seconds, not its pixels). */
const NARROW: StripInfo = {
  ...STRIP,
  y: 250,
  width: 600,
  height: 150,
  secondsPerColumn: 2 / 600,
  headColumn: 375,
  columns: 600,
};

function hud(over: Partial<SimOutput>): SimOutput {
  return { frozen: true, frameId: 40, strip: STRIP, width: 800, height: 520, ...over } as SimOutput;
}

function measurement(over: Partial<Measurement>): Measurement {
  return {
    id: 'm1',
    kind: 'velocity',
    label: 'Velocidad',
    value: 0.5,
    units: 'm/s',
    modality: 'pw',
    sourceViewId: null,
    viewScore: null,
    frameId: 40,
    phase: 0.2,
    timeS: 9,
    geometry: [{ x: 100, y: 350 }],
    imageQualityScore: null,
    userAssisted: false,
    referenceGuidelineIds: [],
    createdAt: '2026-11-26T00:00:00.000Z',
    ...over,
  };
}

const timeOn = (strip: StripInfo, x: number) =>
  ecgTimeAtX(x, ecgLayoutOf(hud({ strip }), 'pw'));
const valueOn = (strip: StripInfo, y: number) =>
  strip.topValue + ((y - strip.y) / strip.height) * (strip.bottomValue - strip.topValue);

describe('measurements stay on the acquisition they were taken on', () => {
  it('moves strip points with what they stand for when the strip is redrawn at another size', () => {
    const points = [
      { x: 100, y: 350 },
      { x: 499, y: 420 },
      { x: 520, y: 310 },
    ];
    const same = reprojectStripGeometry(points, STRIP, STRIP);
    assert.ok(same);
    for (const [i, q] of same.entries()) {
      assert.ok(Math.abs(q.x - points[i]!.x) < 1e-9 && Math.abs(q.y - points[i]!.y) < 1e-9);
    }
    const moved = reprojectStripGeometry(points, STRIP, NARROW);
    assert.ok(moved);
    for (const [i, p] of points.entries()) {
      const q = moved[i]!;
      assert.ok(Math.abs(timeOn(NARROW, q.x) - timeOn(STRIP, p.x)) < 1e-9, `time of point ${i}`);
      assert.ok(Math.abs(valueOn(NARROW, q.y) - valueOn(STRIP, p.y)) < 1e-9, `velocity of point ${i}`);
    }
  });

  it('drops a strip measurement once the sweep has written over its columns, frozen again later', () => {
    const points = [
      { x: 100, y: 350 },
      { x: 499, y: 420 },
    ];
    // resumed for 0.5 s and frozen again: the sweep wrote columns 500–700 only, the measured beat is where it was
    const shortResume: StripInfo = { ...STRIP, headTimeS: 10.5, headColumn: 700 };
    const kept = reprojectStripGeometry(points, STRIP, shortResume);
    assert.ok(kept);
    assert.ok(Math.abs(kept[0]!.x - 100) < 1e-6 && Math.abs(kept[1]!.x - 499) < 1e-6);
    // resumed for 1.2 s: the older end has been written over
    assert.equal(reprojectStripGeometry(points, STRIP, { ...STRIP, headTimeS: 11.2, headColumn: 180 }), null);
    // resumed for 5 s: all of it, and the same acquisition id is no reason to draw it at the sweep head
    const later: StripInfo = { ...STRIP, headTimeS: 15, headColumn: 300 };
    assert.equal(reprojectStripGeometry(points, STRIP, later), null);
    const ms = measurement({ kind: 'time', geometry: points, captureStrip: { ...STRIP } });
    assert.equal(measurementGeometryOn(ms, hud({ strip: later })), null);
  });

  it('keeps a deceleration-time intercept beyond the strip’s edge beyond it, never wrapped round', () => {
    // a peak and a slope point near the right edge, and their baseline intercept past it (x 820 of 800)
    const points = [
      { x: 700, y: 320 },
      { x: 760, y: 360 },
      { x: 820, y: 400 },
    ];
    const same = reprojectStripGeometry(points, STRIP, STRIP);
    assert.ok(same);
    assert.ok(Math.abs(same[2]!.x - 820) < 1e-6, 'drawn where it was taken');
    const narrow = reprojectStripGeometry(points, STRIP, NARROW);
    assert.ok(narrow);
    assert.ok(narrow[2]!.x > NARROW.width, 'still past the edge of the narrower strip');
    assert.ok(Math.abs(narrow[2]!.x - narrow[0]!.x - 120 * (600 / 800)) < 1e-6);
  });

  it('draws a strip measurement only on its own frozen strip, at the strip’s current size', () => {
    const ms = measurement({ captureStrip: { ...STRIP } });
    assert.equal(measurementGeometryOn(ms, hud({ frozen: false })), null, 'live: the sweep moves on');
    assert.equal(
      measurementGeometryOn(ms, hud({ strip: { ...STRIP, acquisitionId: 8 } })),
      null,
      'another acquisition',
    );
    const own = measurementGeometryOn(ms, hud({ strip: NARROW, frameId: 41 }));
    assert.ok(own, 'any cine frame of the same strip');
    assert.ok(Math.abs(timeOn(NARROW, own[0]!.x) - timeOn(STRIP, 100)) < 1e-9);
  });

  it('draws a sector measurement only on the frozen frame it was taken on', () => {
    const ms = measurement({ kind: 'linear', modality: '2d', geometry: [{ x: 1, y: 2 }] });
    assert.deepEqual(measurementGeometryOn(ms, hud({})), [{ x: 1, y: 2 }]);
    assert.equal(measurementGeometryOn(ms, hud({ frameId: 41 })), null, 'another phase or view');
    assert.equal(measurementGeometryOn(ms, hud({ frozen: false })), null);
  });

  it('starts a measurement in progress over when its acquisition changes', () => {
    assert.equal(pendingAcquisitionKey(hud({ frozen: false }), true), 'live');
    assert.notEqual(
      pendingAcquisitionKey(hud({ frameId: 40 }), true),
      pendingAcquisitionKey(hud({ frameId: 41 }), true),
    );
    // a strip tool keeps its points across the cine frames of one strip, not across a redraw at another size
    assert.equal(
      pendingAcquisitionKey(hud({ frameId: 40 }), false),
      pendingAcquisitionKey(hud({ frameId: 41 }), false),
    );
    assert.notEqual(
      pendingAcquisitionKey(hud({}), false),
      pendingAcquisitionKey(hud({ width: 600 }), false),
    );
  });
});
