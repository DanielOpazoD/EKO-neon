import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { baseInput } from '@/simulator/core/baseInput';
import { SimulatorCore } from '@/simulator/core/simulatorCore';
import { getViewTarget } from '@/simulator/windows/viewDefinitions';
import { canonicalControl } from '@/simulator/windows/viewTargets';
import { expectedTransmission, windowFromSkin } from './viewQuality';

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

  it('separates diffuse chest-wall habitus attenuation from focal rib/lung/calcium shadowing across difficult and subcostal views', () => {
    // Pure unit check on expectedTransmission
    const base = expectedTransmission(8, 2.5, true);
    const habitus = expectedTransmission(8, 2.5, true, 0.7, 3.4);
    assert.ok(habitus < base * 0.5, 'habitus chest-wall attenuation should lower the diffuse shadow baseline');
    assert.equal(expectedTransmission(8, 2.5, true, 0, 3.4), base);

    // 1. HFrEF subcostal-ivc (clear liver acoustic window, zero rib/lung) must not misclassify ivc/hepatic-vein as shadowed
    const hfref = loadCaseById('hfref-severe-mr');
    const inp = baseInput();
    inp.rendererBackend = 'procedural';
    inp.quality = 'low';
    const hfCore = new SimulatorCore(hfref, inp);
    try {
      const vt = getViewTarget('subcostal-ivc');
      const ctrl = canonicalControl(vt, hfCore.models.heart, hfCore.models.thorax);
      const midDepth = Math.round(
        (vt.recommendedDepthRangeCm[0] + vt.recommendedDepthRangeCm[1]) / 2,
      );
      hfCore.setInput({ ...inp, probe: ctrl, settings: { ...inp.settings, depthCm: midDepth } });
      for (let i = 0; i < 6; i++) hfCore.step(0.1);
      const a = hfCore.lastView!;
      assert.equal(a.bestViewId, 'subcostal-ivc');
      assert.ok(a.score >= 75, `Expected HFrEF subcostal-ivc score >= 75, got ${a.score}`);
      assert.ok(
        a.visibleLandmarks.includes('ivc') && a.visibleLandmarks.includes('hepatic-vein'),
        `Expected ivc and hepatic-vein visible on clear liver window, missing=${a.missingLandmarks.join(',')}`,
      );
    } finally {
      hfCore.dispose();
    }

    // 2. Difficult-window curriculum targets (PLAX & A4C >= 55) are achievable in expiration + left-lateral,
    //    while genuine focal lung/rib shadowing (PSAX-PM) is still flagged with high shadowFraction
    const diff = loadCaseById('normal-difficult-window');
    const diffCore = new SimulatorCore(diff, inp);
    try {
      for (const viewId of ['plax', 'a4c'] as const) {
        const vt = getViewTarget(viewId);
        const ctrl = canonicalControl(vt, diffCore.models.heart, diffCore.models.thorax);
        const midDepth = Math.round(
          (vt.recommendedDepthRangeCm[0] + vt.recommendedDepthRangeCm[1]) / 2,
        );
        diffCore.setInput({
          ...inp,
          probe: ctrl,
          settings: { ...inp.settings, depthCm: midDepth, frequencyMHz: 2.0 },
        });
        for (let i = 0; i < 6; i++) diffCore.step(0.1);
        const a = diffCore.lastView!;
        assert.equal(a.bestViewId, viewId);
        assert.ok(
          a.score >= 55,
          `Difficult-window ${viewId} should reach curriculum target >= 55, got ${a.score}`,
        );
      }
      const pmVt = getViewTarget('psax-pm');
      const pmCtrl = canonicalControl(pmVt, diffCore.models.heart, diffCore.models.thorax);
      diffCore.setInput({
        ...inp,
        probe: pmCtrl,
        settings: { ...inp.settings, depthCm: 15 },
      });
      for (let i = 0; i < 6; i++) diffCore.step(0.1);
      assert.ok(
        diffCore.lastView!.shadowFraction > 0.5,
        `Focal rib/lung-shadowed difficult-window psax-pm must still report high shadowFraction, got ${diffCore.lastView!.shadowFraction}`,
      );
    } finally {
      diffCore.dispose();
    }
  });
});
