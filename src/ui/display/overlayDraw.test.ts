import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { computeSectorMapping, polarToPixel } from '@/simulator/renderer/scanConvert';
import { DEFAULT_ACQUISITION, polarSpecFor } from '@/simulator/renderer/types';
import { orientationMarker } from './overlayDraw';

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
});
