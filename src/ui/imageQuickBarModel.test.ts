import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  quickBarGroupFor,
  stepColorBaseline,
  stepColorScale,
  stepSpectralBaseline,
  stepSpectralScale,
  toggleHdZoom,
} from './imageQuickBarModel';

describe('contextual ImageQuickBar modality stepping & clamping', () => {
  it('maps each modality to its clinical quick-bar control family', () => {
    assert.equal(quickBarGroupFor('2d'), 'bmode');
    assert.equal(quickBarGroupFor('m-mode'), 'bmode');
    assert.equal(quickBarGroupFor('color'), 'color');
    assert.equal(quickBarGroupFor('cmm'), 'color');
    assert.equal(quickBarGroupFor('pw'), 'spectral');
    assert.equal(quickBarGroupFor('cw'), 'spectral');
    assert.equal(quickBarGroupFor('tdi'), 'spectral');
  });

  it('steps Color Nyquist scale within depth limit and clamps baseline shift symmetrically', () => {
    const down = stepColorScale(0.6, 0.45, 0.8, -1);
    assert.equal(down.scaleMps, 0.55);
    assert.equal(down.baselineShiftMps, 0.44);

    const capped = stepColorScale(0.7, 0, 0.72, 1);
    assert.equal(capped.scaleMps, 0.72);

    assert.equal(stepColorBaseline(0.5, 0.38, 1), 0.4);
    assert.equal(stepColorBaseline(0.5, -0.38, -1), -0.4);
  });

  it('steps PW, CW, and TDI velocity scales & baselines with clinical step sizes', () => {
    const pw = stepSpectralScale('pw', 0.9, -0.7, 0.95, -1);
    assert.equal(pw.scaleMps, 0.8);
    assert.equal(pw.baselineShiftMps, -0.64);

    const cw = stepSpectralScale('cw', 3.0, 0, 10, 1);
    assert.equal(cw.scaleMps, 3.5);
    assert.equal(stepSpectralBaseline('cw', 3.0, 0, -1), -0.4);

    const tdi = stepSpectralScale('tdi', 0.2, 0, 0.5, 1);
    assert.equal(tdi.scaleMps, 0.24);
    assert.equal(stepSpectralBaseline('tdi', 0.2, 0, -1), -0.02);
  });

  it('toggles 1-click HD Write-Zoom cleanly between 1× and 2×', () => {
    assert.equal(toggleHdZoom(1), 2);
    assert.equal(toggleHdZoom(1.04), 2);
    assert.equal(toggleHdZoom(1.5), 1);
    assert.equal(toggleHdZoom(2), 1);
  });
});
