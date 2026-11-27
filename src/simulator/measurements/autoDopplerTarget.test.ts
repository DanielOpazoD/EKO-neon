import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { evaluateTechnique } from '@/education/technique';
import { buildCaseModels, REST_PATIENT } from '@/simulator/anatomy/caseModels';
import { baseInput } from '@/simulator/core/baseInput';
import type { SimInput } from '@/simulator/core/protocol';
import { SimulatorCore } from '@/simulator/core/simulatorCore';
import { canonicalControl, getViewTarget } from '@/simulator/windows/viewTargets';
import type { DopplerTargetId } from './autoDopplerTarget';
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
  return { core, inp };
}

function specById(id: string) {
  const spec = MEASUREMENT_SPECS.find((m) => m.id === id);
  assert.ok(spec, `missing measurement spec ${id}`);
  return spec;
}

describe('Pure-2D guided Doppler & M-mode auto-targeting', () => {
  it('places Mitral PW, TDI septal/lateral, TAPSE M-mode, and TR CW cleanly on canonical A4C and passes technique grading', () => {
    const targets: { target: DopplerTargetId; specId: string; expectedModality: string }[] = [
      { target: 'mitral-inflow', specId: 'mitral-e', expectedModality: 'pw' },
      { target: 'tdi-septal', specId: 'e-prime-septal', expectedModality: 'tdi' },
      { target: 'tdi-lateral', specId: 'e-prime-lateral', expectedModality: 'tdi' },
      { target: 'tapse', specId: 'tapse', expectedModality: 'm-mode' },
      { target: 'tr-cw', specId: 'tr-vmax', expectedModality: 'cw' },
    ];

    for (const caseId of ['normal-excellent-window', 'hfref-severe-mr'] as const) {
      for (const { target, specId, expectedModality } of targets) {
        const { core, inp } = coreAtView(caseId, 'a4c');
        try {
          // Run >= 1 cardiac cycle in 2D so cine holds diastolic + ED frames
          for (let i = 0; i < 18; i++) core.step(0.05);
          const res = core.request({ kind: 'autoDopplerTarget', target });
          assert.equal(res?.kind, 'autoDopplerTarget');
          assert.equal(
            res.result.ok,
            true,
            `${caseId} A4C ${target} should succeed: ${res.result.ok ? '' : res.result.message}`,
          );
          if (!res.result.ok) continue;
          assert.equal(res.result.modality, expectedModality);
          assert.ok(res.result.rCm > 2 && res.result.rCm < 15, `${target} plausible depth ${res.result.rCm}`);

          // Apply cursor/gate and target modality and run 2 beats so peakFlowStructure & strip settle
          const nextInp: SimInput = {
            ...inp,
            modality: res.result.modality,
            cursorThetaRad: res.result.thetaRad,
            gateDepthCm: res.result.rCm,
            spectral: {
              ...inp.spectral,
              ...(res.result.recommendedBaselineShiftMps !== undefined
                ? { baselineShiftMps: res.result.recommendedBaselineShiftMps }
                : {}),
            },
          };
          core.setInput(nextInp);
          let out = core.step(0.05);
          for (let i = 0; i < 36; i++) out = core.step(0.05);
          assert.ok(out, `${caseId} ${target}: expected SimOutput`);

          // Confirm evaluateTechnique accepts the auto-targeted placement with zero invalid findings
          const spec = specById(specId);
          const tq = evaluateTechnique(spec, {
            modality: res.result.modality,
            viewId: out.view?.bestViewId ?? 'a4c',
            viewScore: out.view?.score ?? 90,
            phase: out.phase,
            phaseMarks: out.acquisition?.phaseMarks ?? null,
            gate: out.gate,
            userAssisted: true,
          });
          assert.equal(
            tq.findings.some((f) => f.level === 'invalid'),
            false,
            `${caseId} ${target} (${specId}) must have no invalid findings: ${tq.findings
              .map((f) => `[${f.level}] ${f.message}`)
              .join('; ')}`,
          );
        } finally {
          core.dispose();
        }
      }
    }
  });

  it('places TAPSE M-mode and TR CW cleanly on canonical rv-focused and delegates LVOT cleanly on A5C', () => {
    for (const { viewId, target, specId } of [
      { viewId: 'rv-focused', target: 'tapse' as const, specId: 'tapse' },
      { viewId: 'rv-focused', target: 'tr-cw' as const, specId: 'tr-vmax' },
      { viewId: 'a5c', target: 'lvot' as const, specId: 'lvot-vti' },
    ]) {
      const { core, inp } = coreAtView('normal-excellent-window', viewId);
      try {
        for (let i = 0; i < 18; i++) core.step(0.05);
        const res = core.request({ kind: 'autoDopplerTarget', target });
        assert.equal(res?.kind, 'autoDopplerTarget');
        assert.equal(
          res.result.ok,
          true,
          `${viewId} ${target} should succeed: ${res.result.ok ? '' : res.result.message}`,
        );
        if (!res.result.ok) continue;

        core.setInput({
          ...inp,
          modality: res.result.modality,
          cursorThetaRad: res.result.thetaRad,
          gateDepthCm: res.result.rCm,
        });
        let out = core.step(0.05);
        for (let i = 0; i < 36; i++) out = core.step(0.05);
        assert.ok(out);

        const tq = evaluateTechnique(specById(specId), {
          modality: res.result.modality,
          viewId: out.view?.bestViewId ?? viewId,
          viewScore: out.view?.score ?? 90,
          phase: out.phase,
          phaseMarks: out.acquisition?.phaseMarks ?? null,
          gate: out.gate,
          userAssisted: true,
        });
        assert.equal(
          tq.findings.some((f) => f.level === 'invalid'),
          false,
          `${viewId} ${target} invalid findings: ${tq.findings.map((f) => f.message).join('; ')}`,
        );
      } finally {
        core.dispose();
      }
    }
  });

  it('honestly rejects misaligned or anatomically absent views (PLAX and PSAX-PM)', () => {
    const allTargets: DopplerTargetId[] = [
      'lvot',
      'mitral-inflow',
      'tdi-septal',
      'tdi-lateral',
      'tapse',
      'tr-cw',
    ];
    for (const viewId of ['plax', 'psax-pm'] as const) {
      const { core } = coreAtView('normal-excellent-window', viewId);
      try {
        for (let i = 0; i < 10; i++) core.step(0.05);
        for (const target of allTargets) {
          const res = core.request({ kind: 'autoDopplerTarget', target });
          assert.equal(res?.kind, 'autoDopplerTarget');
          assert.equal(
            res.result.ok,
            false,
            `${viewId} must reject ${target} (got ok=true)`,
          );
        }
      } finally {
        core.dispose();
      }
    }
  });
});
