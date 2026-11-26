import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { BREATH_PERIOD_S, respiratoryDepth } from '@/simulator/cardiac-cycle/respiration';
import { baseInput } from '@/simulator/core/baseInput';
import { SimulatorCore } from '@/simulator/core/simulatorCore';
import { canonicalSolveCount, getViewTarget } from '@/simulator/windows/viewTargets';
import type { ViewAnalysis } from './viewQuality';

/** A free-breathing core held at a view's preset (the pose the simulator solves for it), at the view's middle depth. */
function freeBreathingAtPreset(viewId: string): SimulatorCore {
  const inp = baseInput({
    patient: { position: 'left-lateral', respiration: 'free-breathing', headElevationDeg: 0 },
  });
  const core = new SimulatorCore(loadCaseById('normal-excellent-window'), inp);
  const preset = core.request({ kind: 'canonicalControl', viewId });
  assert.ok(preset?.kind === 'canonicalControl');
  const [lo, hi] = getViewTarget(viewId).recommendedDepthRangeCm;
  core.setInput({
    ...inp,
    probe: preset.control,
    settings: { ...inp.settings, depthCm: Math.round((lo + hi) / 2) },
  });
  return core;
}

describe('View reference in free breathing', () => {
  it('keeps the reference of the patient at rest through a whole breath, solved once, in every window', () => {
    // The probe stays on the preset while the lung border slides 1.15 cm and the diaphragm 0.5 cm with the breath. The
    // reference was resolved against the thorax of each frame: in the apical window it left the axis in deep
    // inspiration (the held preset 4.8° and 1.2 cm off it, A4C 97 → 68), and every analysis solved the references of
    // its window again (the apical window and its five beams, or the suprasternal plane and its beam: 22-44 ms).
    for (const viewId of ['a4c', 'suprasternal-arch', 'plax', 'subcostal-4c']) {
      const core = freeBreathingAtPreset(viewId);
      try {
        core.step(0.1); // the first analysis solves the reference of every view in the window
        const solved = canonicalSolveCount();
        const analyses: ViewAnalysis[] = [core.lastView!];
        let deepest = 0;
        for (let timeS = 0.2; timeS < BREATH_PERIOD_S + 1e-9; timeS += 0.1) {
          core.step(0.1);
          const a = core.lastView!;
          if (a === analyses[analyses.length - 1]) continue;
          analyses.push(a);
          deepest = Math.max(deepest, respiratoryDepth(timeS));
        }
        assert.ok(deepest > 0.9, `${viewId}: no analysis in deep inspiration (${deepest.toFixed(2)})`);
        for (const a of analyses) {
          assert.equal(a.bestViewId, viewId);
          assert.ok(
            a.planeAngleDeg < 1e-3 && a.inPlaneRotationDeg < 1e-3 && a.offsetCm < 1e-4,
            `${viewId}: the held preset left its reference (${a.planeAngleDeg.toFixed(2)}°, ` +
              `${a.inPlaneRotationDeg.toFixed(2)}°, ${a.offsetCm.toFixed(2)} cm)`,
          );
        }
        assert.equal(
          canonicalSolveCount() - solved,
          0,
          `${viewId}: reference poses solved again during the breath`,
        );
      } finally {
        core.dispose();
      }
    }
  });
});
