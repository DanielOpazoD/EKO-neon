import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { computeSectorMapping, polarToPixel } from '@/simulator/renderer/scanConvert';
import { DEFAULT_ACQUISITION, polarSpecFor } from '@/simulator/renderer/types';
import { computeMmodeDepthTicks, orientationMarker } from './overlayDraw';

describe('orientation marker on the sector', () => {
  const spec = polarSpecFor(DEFAULT_ACQUISITION, 'medium');

  for (const invertLR of [false, true]) {
    it(`marks the side the probe's notch images (+lateral), invertLR ${invertLR}`, () => {
      const m = computeSectorMapping(spec, 791, 600, invertLR);
      // the notch is the +x side of the probe: +θ in the sector (navigatorProbe.ts, pose.ts)
      const notchSide = Math.sign(polarToPixel(m, 8, 0.3).x - m.apexX);
      assert.equal(notchSide, invertLR ? -1 : 1, 'the inversion mirrors the image');
      const mark = orientationMarker(m);
      assert.equal(Math.sign(mark.x - m.apexX), notchSide, 'the dot is on the notch side');
      assert.equal(Math.sign(mark.labelX - mark.x), notchSide, 'its label is outside the dot');
      assert.equal(mark.labelAlign, notchSide > 0 ? 'left' : 'right');
    });
  }

  it('mirrors across the apex with the image, at the same height', () => {
    const plain = orientationMarker(computeSectorMapping(spec, 791, 600, false));
    const m = computeSectorMapping(spec, 791, 600, true);
    const mirrored = orientationMarker(m);
    assert.equal(mirrored.x, 2 * m.apexX - plain.x);
    assert.equal(mirrored.labelX, 2 * m.apexX - plain.labelX);
    assert.equal(mirrored.y, plain.y);
  });

  it('keeps the orientation marker inside the visible top border under 2× focalized zoom', () => {
    const mZoom = computeSectorMapping(spec, 791, 600, false, 2);
    assert.ok(mZoom.apexY < 0, '2× focalized zoom lifts apex above the top border');
    const mark = orientationMarker(mZoom);
    assert.ok(mark.y >= 14, `orientation marker must remain visible, got y=${mark.y}`);
  });

  it('generates clinical 1 cm minor and 5 cm major depth ticks for M-mode strips', () => {
    const ticks = computeMmodeDepthTicks(16);
    assert.equal(ticks.length, 16);
    const majors = ticks.filter((t) => t.major);
    assert.deepEqual(
      majors.map((t) => t.label),
      ['5 cm', '10', '15'],
    );
    assert.equal(ticks[0]!.major, false);
    assert.equal(ticks[0]!.label, undefined);
    assert.ok(Math.abs(majors[0]!.frac - 5 / 16) < 1e-6);
  });
});
