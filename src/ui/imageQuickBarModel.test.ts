import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  cycleRespiration,
  cycleSweepSpeed,
  quickBarGroupFor,
  RESPIRATION_SHORT_LABEL,
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

  it('cycles strip sweep speed through 25 -> 50 -> 100 -> 25 mm/s', () => {
    assert.equal(cycleSweepSpeed(25), 50);
    assert.equal(cycleSweepSpeed(50), 100);
    assert.equal(cycleSweepSpeed(100), 25);
  });

  it('cycles patient respiration smoothly across bedside phases with Spanish short labels', () => {
    assert.equal(cycleRespiration('free-breathing'), 'inspiration');
    assert.equal(cycleRespiration('inspiration'), 'expiration');
    assert.equal(cycleRespiration('expiration'), 'breath-hold');
    assert.equal(cycleRespiration('breath-hold'), 'free-breathing');
    assert.equal(RESPIRATION_SHORT_LABEL.inspiration, 'Insp');
    assert.equal(RESPIRATION_SHORT_LABEL['free-breathing'], 'Libre');
  });
});
