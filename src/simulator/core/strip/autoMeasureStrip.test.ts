import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { evaluateTechnique } from '@/education/technique';
import { buildCaseModels, REST_PATIENT } from '@/simulator/anatomy/caseModels';
import { baseInput } from '@/simulator/core/baseInput';
import type { SimInput } from '@/simulator/core/protocol';
import { SimulatorCore } from '@/simulator/core/simulatorCore';
import { spectralDefaultsFor } from '@/simulator/doppler/spectral/spectrum';
import { computeGroundTruth } from '@/simulator/hemodynamics/groundTruth';
import { MEASUREMENT_SPECS } from '@/simulator/measurements/protocol';
import { canonicalControl, getViewTarget } from '@/simulator/windows/viewTargets';

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
  return { core, inp, truth: computeGroundTruth(c) };
}

function specById(id: string) {
  const spec = MEASUREMENT_SPECS.find((m) => m.id === id);
  assert.ok(spec, `missing spec ${id}`);
  return spec;
}

describe('Pure strip-signal auto-measurements (Mitral E/A/TD, TDI e′, TR Vmax, M-mode TAPSE)', () => {
  it('measures Mitral E, A, and DT on sinus rhythm and omits A wave cleanly in Atrial Fibrillation', () => {
    // 1. Sinus rhythm (normal): detects E, A, and DT cleanly
    {
      const { core, inp, truth } = coreAtView('normal-excellent-window', 'a4c');
      try {
        for (let i = 0; i < 18; i++) core.step(0.05);
        const tgt = core.request({ kind: 'autoDopplerTarget', target: 'mitral-inflow' });
        assert.equal(tgt?.kind, 'autoDopplerTarget');
        assert.equal(tgt.result.ok, true);
        if (!tgt.result.ok) return;

        const pwInp: SimInput = {
          ...inp,
          modality: 'pw',
          cursorThetaRad: tgt.result.thetaRad,
          gateDepthCm: tgt.result.rCm,
          spectral: {
            ...spectralDefaultsFor('pw'),
            baselineShiftMps: tgt.result.recommendedBaselineShiftMps ?? 0.25,
          },
        };
        core.setInput(pwInp);
        let out = core.step(0.05);
        for (let i = 0; i < 46; i++) out = core.step(0.05);
        assert.ok(out);

        core.setInput({ ...pwInp, frozen: true });
        out = core.step(0.05);
        assert.ok(out);

        const res = core.request({ kind: 'autoMeasureStrip', target: 'mitral-inflow' });
        assert.equal(res?.kind, 'autoMeasureStrip');
        assert.equal(
          res.result.ok,
          true,
          `normal mitral-inflow should succeed: ${res.result.ok ? '' : res.result.message}`,
        );
        if (!res.result.ok || res.result.target !== 'mitral-inflow') return;

        const eErr = Math.abs(Math.abs(res.result.e.velocityMps) - truth.mitral.ePeakMps) / truth.mitral.ePeakMps;
        assert.ok(
          eErr < 0.2,
          `normal mitral E ${Math.abs(res.result.e.velocityMps).toFixed(2)} vs truth ${truth.mitral.ePeakMps.toFixed(2)}`,
        );
        assert.ok(res.result.a, 'sinus rhythm should detect A wave');
        const aErr =
          Math.abs(Math.abs(res.result.a!.velocityMps) - truth.mitral.aPeakMps!) /
          truth.mitral.aPeakMps!;
        assert.ok(
          aErr < 0.25,
          `normal mitral A ${Math.abs(res.result.a!.velocityMps).toFixed(2)} vs truth ${truth.mitral.aPeakMps!.toFixed(2)}`,
        );
        assert.ok(res.result.dt, 'should compute E-wave deceleration time');
        assert.ok(
          res.result.dt!.dtMs > 110 && res.result.dt!.dtMs < 320,
          `normal mitral DT plausible (${res.result.dt!.dtMs.toFixed(0)} ms vs truth ${truth.mitral.decelerationTimeMs} ms)`,
        );
      } finally {
        core.dispose();
      }
    }

    // 2. Atrial fibrillation: detects E wave, and omits A wave strictly from signal (no phantom A wave!)
    {
      const { core, inp, truth } = coreAtView('af-diastolic', 'a4c');
      try {
        for (let i = 0; i < 18; i++) core.step(0.05);
        const tgt = core.request({ kind: 'autoDopplerTarget', target: 'mitral-inflow' });
        assert.equal(tgt?.kind, 'autoDopplerTarget');
        assert.equal(tgt.result.ok, true);
        if (!tgt.result.ok) return;

        const pwInp: SimInput = {
          ...inp,
          modality: 'pw',
          cursorThetaRad: tgt.result.thetaRad,
          gateDepthCm: tgt.result.rCm,
          spectral: {
            ...spectralDefaultsFor('pw'),
            scaleMps: 1.5,
            baselineShiftMps: 0.35,
          },
        };
        core.setInput(pwInp);
        for (let i = 0; i < 50; i++) core.step(0.05);
        core.setInput({ ...pwInp, frozen: true });
        core.step(0.05);

        const res = core.request({ kind: 'autoMeasureStrip', target: 'mitral-inflow' });
        assert.equal(res?.kind, 'autoMeasureStrip');
        assert.equal(
          res.result.ok,
          true,
          `AF mitral-inflow should succeed: ${res.result.ok ? '' : res.result.message}`,
        );
        if (!res.result.ok || res.result.target !== 'mitral-inflow') return;
        const eErr = Math.abs(Math.abs(res.result.e.velocityMps) - truth.mitral.ePeakMps) / truth.mitral.ePeakMps;
        assert.ok(
          eErr < 0.25,
          `AF mitral E ${Math.abs(res.result.e.velocityMps).toFixed(2)} vs truth ${truth.mitral.ePeakMps.toFixed(2)}`,
        );
        assert.equal(res.result.a, null, 'Atrial fibrillation must not emit a phantom A wave');
      } finally {
        core.dispose();
      }
    }
  });

  it('measures TDI septal and lateral e′ accurately on normal and HFrEF cases', () => {
    for (const [caseId, target, specId] of [
      ['normal-excellent-window', 'tdi-septal', 'e-prime-septal'],
      ['normal-excellent-window', 'tdi-lateral', 'e-prime-lateral'],
      ['hfref-severe-mr', 'tdi-septal', 'e-prime-septal'],
    ] as const) {
      const { core, inp, truth } = coreAtView(caseId, 'a4c');
      try {
        for (let i = 0; i < 18; i++) core.step(0.05);
        const tgt = core.request({ kind: 'autoDopplerTarget', target });
        assert.equal(tgt?.kind, 'autoDopplerTarget');
        assert.equal(tgt.result.ok, true);
        if (!tgt.result.ok) continue;

        const tdiInp: SimInput = {
          ...inp,
          modality: 'tdi',
          cursorThetaRad: tgt.result.thetaRad,
          gateDepthCm: tgt.result.rCm,
          spectral: spectralDefaultsFor('tdi'),
        };
        core.setInput(tdiInp);
        let out = core.step(0.05);
        for (let i = 0; i < 46; i++) out = core.step(0.05);
        core.setInput({ ...tdiInp, frozen: true });
        out = core.step(0.05);
        assert.ok(out);

        const res = core.request({ kind: 'autoMeasureStrip', target: 'tdi-e-prime' });
        assert.equal(res?.kind, 'autoMeasureStrip');
        assert.equal(
          res.result.ok,
          true,
          `${caseId} ${target} autoMeasureStrip should succeed: ${res.result.ok ? '' : res.result.message}`,
        );
        if (!res.result.ok || res.result.target !== 'tdi-e-prime') continue;

        const measuredCmps = Math.abs(res.result.ePrime.velocityMps) * 100;
        const truthCmps =
          target === 'tdi-septal' ? truth.mitral.ePrimeSeptalCmps : truth.mitral.ePrimeLateralCmps;
        const relErr = Math.abs(measuredCmps - truthCmps) / truthCmps;
        assert.ok(
          relErr < 0.3,
          `${caseId} ${target} measured ${measuredCmps.toFixed(1)} cm/s vs truth ${truthCmps.toFixed(1)} cm/s`,
        );

        const tq = evaluateTechnique(specById(specId), {
          modality: 'tdi',
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
          `${caseId} ${specId} invalid findings: ${tq.findings.map((f) => f.message).join('; ')}`,
        );
      } finally {
        core.dispose();
      }
    }
  });

  it('measures TR Vmax in pulmonary hypertension CW and TAPSE excursion in M-mode', () => {
    // 1. TR Vmax in PH (truth ≈ 3.77 m/s)
    {
      const { core, inp, truth } = coreAtView('pulmonary-hypertension-rv', 'a4c');
      try {
        for (let i = 0; i < 18; i++) core.step(0.05);
        const tgt = core.request({ kind: 'autoDopplerTarget', target: 'tr-cw' });
        assert.equal(tgt?.kind, 'autoDopplerTarget');
        assert.equal(tgt.result.ok, true);
        if (!tgt.result.ok) return;

        const cwInp: SimInput = {
          ...inp,
          modality: 'cw',
          cursorThetaRad: tgt.result.thetaRad,
          gateDepthCm: tgt.result.rCm,
          spectral: { ...spectralDefaultsFor('cw'), baselineShiftMps: -0.5 },
        };
        core.setInput(cwInp);
        let out = core.step(0.05);
        for (let i = 0; i < 46; i++) out = core.step(0.05);
        core.setInput({ ...cwInp, frozen: true });
        out = core.step(0.05);
        assert.ok(out);

        const res = core.request({ kind: 'autoMeasureStrip', target: 'tr-vmax' });
        assert.equal(res?.kind, 'autoMeasureStrip');
        assert.equal(
          res.result.ok,
          true,
          `PH TR Vmax should succeed: ${res.result.ok ? '' : res.result.message}`,
        );
        if (!res.result.ok || res.result.target !== 'tr-vmax') return;
        const measured = Math.abs(res.result.vmax.velocityMps);
        const truthTr = truth.rightHeart.trVmaxMps!;
        const relErr = Math.abs(measured - truthTr) / truthTr;
        assert.ok(
          relErr < 0.22,
          `PH TR Vmax ${measured.toFixed(2)} m/s vs truth ${truthTr.toFixed(2)} m/s`,
        );
      } finally {
        core.dispose();
      }
    }

    // 2. TAPSE in normal A4C (truth ≈ 2.2 cm)
    {
      const { core, inp, truth } = coreAtView('normal-excellent-window', 'a4c');
      try {
        for (let i = 0; i < 18; i++) core.step(0.05);
        const tgt = core.request({ kind: 'autoDopplerTarget', target: 'tapse' });
        assert.equal(tgt?.kind, 'autoDopplerTarget');
        assert.equal(tgt.result.ok, true);
        if (!tgt.result.ok) return;

        const mmInp: SimInput = {
          ...inp,
          modality: 'm-mode',
          cursorThetaRad: tgt.result.thetaRad,
          gateDepthCm: tgt.result.rCm,
        };
        core.setInput(mmInp);
        let out = core.step(0.05);
        for (let i = 0; i < 48; i++) out = core.step(0.05);
        core.setInput({ ...mmInp, frozen: true });
        out = core.step(0.05);
        assert.ok(out);

        const res = core.request({ kind: 'autoMeasureStrip', target: 'tapse' });
        assert.equal(res?.kind, 'autoMeasureStrip');
        assert.equal(
          res.result.ok,
          true,
          `normal TAPSE should succeed: ${res.result.ok ? '' : res.result.message}`,
        );
        if (!res.result.ok || res.result.target !== 'tapse') return;
        const relErr = Math.abs(res.result.tapseCm - truth.rightHeart.tapseCm) / truth.rightHeart.tapseCm;
        assert.ok(
          relErr < 0.25,
          `normal TAPSE ${res.result.tapseCm.toFixed(2)} cm vs truth ${truth.rightHeart.tapseCm.toFixed(2)} cm`,
        );
      } finally {
        core.dispose();
      }
    }

    // 3. Honest rejection when called on wrong modality (2D)
    {
      const { core } = coreAtView('normal-excellent-window', 'a4c');
      try {
        core.step(0.05);
        for (const target of ['mitral-inflow', 'tdi-e-prime', 'tr-vmax', 'tapse'] as const) {
          const r = core.request({ kind: 'autoMeasureStrip', target });
          assert.equal(r?.kind, 'autoMeasureStrip');
          assert.equal(r.result.ok, false);
          if (!r.result.ok) assert.equal(r.result.reason, 'wrong-modality');
        }
      } finally {
        core.dispose();
      }
    }
  });
});
