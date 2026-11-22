import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { baseInput } from '@/simulator/core/baseInput';
import { SimulatorCore } from '@/simulator/core/simulatorCore';
import { getViewTarget } from '@/simulator/windows/viewDefinitions';
import { canonicalControl } from '@/simulator/windows/viewTargets';
import { windowFromSkin } from './viewQuality';

describe('View recognition & canonical window classification', () => {
  it('classifies canonical acoustic skin coordinates into parasternal, apical, subcostal and suprasternal windows', () => {
    assert.equal(windowFromSkin(2.5, 0.5), 'parasternal');
    assert.equal(windowFromSkin(8.0, -3.0), 'apical');
    assert.equal(windowFromSkin(0.0, -9.0), 'subcostal');
    assert.equal(windowFromSkin(0.0, 11.0), 'suprasternal');
    assert.equal(windowFromSkin(-6.0, 5.0), 'none');
  });

  it('recognizes canonical TTE views with high clinical quality scores', () => {
    const caseDef = loadCaseById('normal-excellent-window');
    const inp = baseInput();
    inp.rendererBackend = 'procedural';
    inp.quality = 'low';
    const core = new SimulatorCore(caseDef, inp);

    // Core canonical transthoracic views spanning all 3 main acoustic windows
    const viewsToVerify = ['plax', 'psax-mv', 'psax-pm', 'a4c', 'a2c', 'a3c', 'subcostal-4c'];
    for (const viewId of viewsToVerify) {
      const target = getViewTarget(viewId);
      const ctrl = canonicalControl(target, core.models.heart, core.models.thorax);
      const midDepth = Math.round(
        (target.recommendedDepthRangeCm[0] + target.recommendedDepthRangeCm[1]) / 2,
      );
      core.setInput({
        ...inp,
        probe: ctrl,
        settings: { ...inp.settings, depthCm: midDepth },
      });
      // Advance >= 0.5s simulated time so throttled view analysis runs fresh on the new pose
      for (let i = 0; i < 6; i++) core.step(0.1);
      const analysis = core.lastView;
      assert.ok(analysis, `Expected ViewAnalysis for ${viewId}`);
      assert.equal(
        analysis.bestViewId,
        viewId,
        `Canonical pose for ${viewId} was misclassified as ${analysis.bestViewId} (score ${analysis.score})`,
      );
      assert.ok(
        analysis.score >= 70,
        `Expected high quality score (>=70) for canonical ${viewId}, got ${analysis.score}`,
      );
    }

    core.dispose();
  });
});
