import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { evaluateTechnique } from '@/education/technique';
import { buildCaseModels, REST_PATIENT } from '@/simulator/anatomy/caseModels';
import { Structure } from '@/simulator/anatomy/tissue';
import { baseInput } from '@/simulator/core/baseInput';
import type { SimInput } from '@/simulator/core/protocol';
import { SimulatorCore } from '@/simulator/core/simulatorCore';
import { computeGroundTruth } from '@/simulator/hemodynamics/groundTruth';
import { canonicalControl, getViewTarget } from '@/simulator/windows/viewTargets';
import { MEASUREMENT_SPECS } from './protocol';

function coreAtView(caseId: string, viewId: string) {
  const c = loadCaseById(caseId);
  const { heart, thorax } = buildCaseModels(c, REST_PATIENT);
  const ctrl = canonicalControl(getViewTarget(viewId), heart, thorax);
  const inp: SimInput = {
    ...baseInput(),
    probe: ctrl,
    modality: '2d',
    quality: 'low',
    rendererBackend: 'procedural',
    display: { width: 480, height: 380 },
  };
  const core = new SimulatorCore(c, inp);
  return { core, inp, c, truth: computeGroundTruth(c) };
}

describe('Automatic LVOT PW gate placement (A5C/A3C) and 1-click Auto-VTI beat tracing', () => {
  it('locates the LVOT gate in A5C across diverse pathologies, aligns PW flow, and extracts accurate LVOT VTI', () => {
    const lvotVtiSpec = MEASUREMENT_SPECS.find((m) => m.id === 'lvot-vti')!;
    for (const caseId of [
      'normal-excellent-window',
      'aortic-stenosis-severe',
      'hfref-severe-mr',
    ] as const) {
      const { core, inp, truth } = coreAtView(caseId, 'a5c');
      try {
        // 1. Render >= 1 beat in 2D A5C (capturing mid-systole) and request autoLvotGate purely from polar structure.
        for (let i = 0; i < 16; i++) core.step(0.05);
        const gateRes = core.request({ kind: 'autoLvotGate' });
        assert.equal(gateRes?.kind, 'autoLvotGate');
        assert.equal(
          gateRes.result.ok,
          true,
          `${caseId} A5C should detect LVOT gate: ${gateRes.result.ok ? '' : gateRes.result.message}`,
        );
        if (!gateRes.result.ok) continue;
        assert.ok(gateRes.result.rCm > 4 && gateRes.result.rCm < 14, 'plausible LVOT gate depth');
        assert.ok(
          gateRes.result.angle2dDeg < 35,
          `2D beam-LVOT angle (${gateRes.result.angle2dDeg}°)`,
        );

        // 2. Snap cursor/gate to the detected LVOT position, switch to PW (with negative-flow headroom), and sweep ~2.2 s.
        const pwInp: SimInput = {
          ...inp,
          cursorThetaRad: gateRes.result.thetaRad,
          gateDepthCm: gateRes.result.rCm,
          modality: 'pw',
          spectral: { ...inp.spectral, baselineShiftMps: -0.25 },
        };
        core.setInput(pwInp);
        let out = core.step(0.05);
        for (let i = 0; i < 44; i++) out = core.step(0.05);
        assert.ok(out, 'expected PW SimOutput');

        assert.equal(
          out.gate?.structure === Structure.Lvot ||
            out.gate?.peakFlowStructure === Structure.Lvot,
          true,
          `${caseId}: PW gate must sample Structure.Lvot at peak flow (got structure=${out.gate?.structure}, peakFlowStructure=${out.gate?.peakFlowStructure})`,
        );
        assert.ok(
          (out.gate?.flowAngleDeg ?? 90) <= 30,
          `${caseId}: 3D beam-flow angle <= 30° (got ${out.gate?.flowAngleDeg})`,
        );

        // 3. Freeze and invoke 1-click autoTraceBeat on the acquired PW strip.
        core.setInput({ ...pwInp, frozen: true });
        out = core.step(0.05);
        assert.ok(out, 'expected frozen PW SimOutput');
        const vtiRes = core.request({ kind: 'autoTraceBeat' });
        assert.equal(vtiRes?.kind, 'autoTraceBeat');
        assert.equal(
          vtiRes.result.ok,
          true,
          `${caseId} autoTraceBeat should trace LVOT systolic VTI: ${vtiRes.result.ok ? '' : vtiRes.result.message}`,
        );
        if (!vtiRes.result.ok) continue;

        const relErr = Math.abs(vtiRes.result.vtiCm - truth.lvot.vtiCm) / truth.lvot.vtiCm;
        assert.ok(
          relErr < 0.22,
          `${caseId} auto LVOT VTI ${vtiRes.result.vtiCm.toFixed(1)} cm vs truth ${truth.lvot.vtiCm.toFixed(1)} cm`,
        );

        // 4. Confirm the technique engine grades this auto-placed LVOT PW capture cleanly (no invalid findings).
        const tq = evaluateTechnique(lvotVtiSpec, {
          modality: 'pw',
          viewId: out.view?.bestViewId ?? 'a5c',
          viewScore: out.view?.score ?? 90,
          phase: out.phase,
          phaseMarks: out.acquisition?.phaseMarks ?? null,
          gate: out.gate,
          userAssisted: true,
        });
        assert.equal(
          tq.findings.some((f) => f.level === 'invalid'),
          false,
          `${caseId} auto LVOT VTI must have no invalid technique findings: ${tq.findings.map((f) => f.message).join('; ')}`,
        );
      } finally {
        core.dispose();
      }
    }
  });

  it('also locates the LVOT gate cleanly on A3C and rejects PLAX, A4C, and PSAX-PM geometrically', () => {
    // A3C succeeds
    const a3c = coreAtView('normal-excellent-window', 'a3c');
    try {
      a3c.core.step(0.05);
      const a3cRes = a3c.core.request({ kind: 'autoLvotGate' });
      assert.equal(a3cRes?.kind, 'autoLvotGate');
      assert.equal(a3cRes.result.ok, true, 'A3C should locate aligned LVOT gate');
    } finally {
      a3c.core.dispose();
    }

    // PLAX is nearly perpendicular to the ultrasound beam -> rejected as misaligned
    const plax = coreAtView('normal-excellent-window', 'plax');
    try {
      plax.core.step(0.05);
      const plaxRes = plax.core.request({ kind: 'autoLvotGate' });
      assert.equal(plaxRes?.kind, 'autoLvotGate');
      assert.equal(
        plaxRes.result.ok,
        false,
        'PLAX LVOT is perpendicular to beam and must be rejected',
      );
      if (!plaxRes.result.ok) {
        assert.equal(plaxRes.result.reason, 'misaligned-view');
      }
    } finally {
      plax.core.dispose();
    }

    // A4C & PSAX-PM have no LVOT in plane -> rejected cleanly
    for (const viewId of ['a4c', 'psax-pm'] as const) {
      const v = coreAtView('normal-excellent-window', viewId);
      try {
        v.core.step(0.05);
        const r = v.core.request({ kind: 'autoLvotGate' });
        assert.equal(r?.kind, 'autoLvotGate');
        assert.equal(r.result.ok, false, `${viewId} should reject autoLvotGate`);
      } finally {
        v.core.dispose();
      }
    }
  });

  it('reports honest failure when autoTraceBeat runs in 2D or with a PW gate placed outside flow', () => {
    const { core, inp } = coreAtView('normal-excellent-window', 'a5c');
    try {
      core.step(0.05);
      const in2d = core.request({ kind: 'autoTraceBeat' });
      assert.equal(in2d?.kind, 'autoTraceBeat');
      assert.equal(in2d.result.ok, false);
      if (!in2d.result.ok) assert.equal(in2d.result.reason, 'wrong-modality');

      // Place PW gate far lateral in the chest wall / outside cardiac blood -> no systolic flow
      core.setInput({
        ...inp,
        modality: 'pw',
        cursorThetaRad: -0.58,
        gateDepthCm: 3.0,
      });
      for (let i = 0; i < 40; i++) core.step(0.05);
      const noFlow = core.request({ kind: 'autoTraceBeat' });
      assert.equal(noFlow?.kind, 'autoTraceBeat');
      assert.equal(noFlow.result.ok, false);
      if (!noFlow.result.ok) assert.equal(noFlow.result.reason, 'no-flow');
    } finally {
      core.dispose();
    }
  });
});
