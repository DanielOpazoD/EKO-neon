import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { deriveCalculations, rightAtrialPressure } from '@/education/report';
import { evaluateTechnique } from '@/education/technique';
import { buildCaseModels, REST_PATIENT } from '@/simulator/anatomy/caseModels';
import { baseInput } from '@/simulator/core/baseInput';
import { SimulatorCore } from '@/simulator/core/simulatorCore';
import { computeGroundTruth } from '@/simulator/hemodynamics/groundTruth';
import { canonicalControl, getViewTarget } from '@/simulator/windows/viewTargets';
import { MEASUREMENT_SPECS } from './protocol';
import type { Measurement } from './types';

function specById(id: string) {
  const s = MEASUREMENT_SPECS.find((m) => m.id === id);
  assert.ok(s, `Missing protocol spec for ${id}`);
  return s;
}

describe('Automatic 2D & Doppler Right-Heart + IVC measurements (Paquete 2)', () => {
  it('measures subcostal IVC expiratory & inspiratory diameters, collapse %, and ASE RAP accurately across normal, PH, and tamponade', () => {
    const cases = [
      { id: 'normal-excellent-window', expectedRap: 3 },
      { id: 'pulmonary-hypertension-rv', expectedRap: 15 },
      { id: 'pericardial-effusion-tamponade', expectedRap: 15 },
    ] as const;

    for (const { id: caseId, expectedRap } of cases) {
      const c = loadCaseById(caseId);
      const truth = computeGroundTruth(c);
      const runResp = (respiration: 'expiration' | 'inspiration') => {
        const patient = { ...REST_PATIENT, position: 'subcostal-supine' as const, respiration };
        const { heart, thorax } = buildCaseModels(c, patient);
        const ctrl = canonicalControl(getViewTarget('subcostal-ivc'), heart, thorax);
        const core = new SimulatorCore(c, {
          ...baseInput(),
          probe: ctrl,
          patient,
          modality: '2d',
          quality: 'low',
          rendererBackend: 'procedural',
          display: { width: 640, height: 480 },
        });
        try {
          for (let i = 0; i < 8; i++) core.step(0.032);
          const resp = core.request({ kind: 'autoIvc' });
          assert.ok(
            resp && resp.kind === 'autoIvc' && resp.result.ok,
            `${caseId} (${respiration}): autoIvc should succeed, got ${JSON.stringify(resp)}`,
          );
          const r = resp.result;
          const expectedId =
            respiration === 'inspiration' ? 'ivc-diameter-inspiration' : 'ivc-diameter';
          assert.equal(r.measurementId, expectedId);

          const tech = evaluateTechnique(specById(expectedId), {
            modality: r.acquisition.modality,
            viewId: r.viewId,
            viewScore: r.viewScore,
            phase: r.phase,
            phaseMarks: r.acquisition.phaseMarks,
            gate: null,
            segmentStructures: r.segmentStructures,
            segmentEndsOutside: r.segmentEndsOutside,
            userAssisted: true,
          });
          const invalids = tech.findings.filter((f) => f.level === 'invalid');
          assert.equal(
            invalids.length,
            0,
            `${caseId} (${respiration}): invalid technique: ${invalids.map((f) => f.message).join('; ')}`,
          );
          return r;
        } finally {
          core.dispose();
        }
      };

      const exp = runResp('expiration');
      const insp = runResp('inspiration');

      const trueExp = truth.rightHeart.ivcCm;
      const trueInsp = trueExp * (1 - truth.rightHeart.ivcCollapsePct / 100);
      const expErrPct = (Math.abs(exp.valueCm - trueExp) / trueExp) * 100;
      const inspErrPct = (Math.abs(insp.valueCm - trueInsp) / Math.max(0.3, trueInsp)) * 100;
      assert.ok(
        expErrPct <= 15,
        `${caseId} IVC exp: measured ${exp.valueCm.toFixed(2)} cm vs truth ${trueExp.toFixed(2)} cm (${expErrPct.toFixed(1)}%)`,
      );
      assert.ok(
        inspErrPct <= 25,
        `${caseId} IVC insp: measured ${insp.valueCm.toFixed(2)} cm vs truth ${trueInsp.toFixed(2)} cm (${inspErrPct.toFixed(1)}%)`,
      );

      const rap = rightAtrialPressure(exp.valueCm, insp.valueCm);
      assert.equal(
        rap.rapMmHg,
        expectedRap,
        `${caseId}: expected ASE RAP ${expectedRap} mmHg, got ${rap.rapMmHg} (${rap.basis})`,
      );

      // Verify derived rap-ivc and rvsp rows populate in report
      const mkLin = (id: string, value: number): Measurement => ({
        id,
        measurementId: id,
        kind: 'linear',
        label: id,
        value,
        units: 'cm',
        modality: '2d',
        sourceViewId: 'subcostal-ivc',
        viewScore: 95,
        frameId: 1,
        phase: 0.4,
        timeS: 0.5,
        geometry: [
          { x: 100, y: 100 },
          { x: 100, y: 150 },
        ],
        imageQualityScore: 90,
        userAssisted: true,
        referenceGuidelineIds: ['ase-right-heart-2025'],
        createdAt: '2026-01-01T00:00:00Z',
      });
      const derived = deriveCalculations([
        mkLin('ivc-diameter', exp.valueCm),
        mkLin('ivc-diameter-inspiration', insp.valueCm),
      ]);
      assert.ok(derived.some((d) => d.id === 'rap-ivc'));
    }
  });

  it('rejects autoIvc cleanly on PLAX and PW Doppler', () => {
    const c = loadCaseById('normal-excellent-window');
    const { heart, thorax } = buildCaseModels(c, REST_PATIENT);
    const ctrl = canonicalControl(getViewTarget('plax'), heart, thorax);
    const inp = {
      ...baseInput(),
      probe: ctrl,
      modality: '2d' as const,
      quality: 'low' as const,
      rendererBackend: 'procedural' as const,
      display: { width: 640, height: 480 },
    };
    const core = new SimulatorCore(c, inp);
    try {
      for (let i = 0; i < 6; i++) core.step(0.032);
      const plaxRes = core.request({ kind: 'autoIvc' });
      assert.ok(plaxRes && plaxRes.kind === 'autoIvc' && !plaxRes.result.ok);
      assert.equal(plaxRes.result.reason, 'not-ivc');

      core.setInput({ ...inp, modality: 'pw' });
      core.step(0.032);
      const pwRes = core.request({ kind: 'autoIvc' });
      assert.ok(pwRes && pwRes.kind === 'autoIvc' && !pwRes.result.ok);
      assert.equal(pwRes.result.reason, 'wrong-modality');
    } finally {
      core.dispose();
    }
  });

  it('extracts RV basal diameter cleanly alongside RV Auto-FE in normal vs dilated PH right ventricle', () => {
    for (const caseId of ['normal-excellent-window', 'pulmonary-hypertension-rv'] as const) {
      const c = loadCaseById(caseId);
      const truth = computeGroundTruth(c);
      const { heart, thorax } = buildCaseModels(c, REST_PATIENT);
      const ctrl = canonicalControl(getViewTarget('rv-focused'), heart, thorax);
      const core = new SimulatorCore(c, {
        ...baseInput(),
        probe: ctrl,
        modality: '2d',
        quality: 'low',
        rendererBackend: 'procedural',
        display: { width: 640, height: 480 },
      });
      try {
        for (let i = 0; i < 40; i++) core.step(0.032);
        const resp = core.request({ kind: 'autoChamberVolume', chamber: 'rv' });
        assert.ok(
          resp && resp.kind === 'autoChamberVolume' && resp.result.ok,
          `${caseId}: RV autoChamberVolume should succeed`,
        );
        const basal = resp.result.rvBasal;
        assert.ok(basal, `${caseId}: expected rvBasal caliper from RV Auto-FE`);

        const tech = evaluateTechnique(specById('rv-basal'), {
          modality: resp.result.ed.acquisition.modality,
          viewId: resp.result.ed.viewId,
          viewScore: resp.result.ed.viewScore,
          phase: resp.result.ed.phase,
          phaseMarks: resp.result.ed.acquisition.phaseMarks,
          gate: null,
          segmentStructures: basal.segmentStructures,
          segmentEndsOutside: basal.segmentEndsOutside,
          userAssisted: true,
        });
        const invalids = tech.findings.filter((f) => f.level === 'invalid');
        assert.equal(
          invalids.length,
          0,
          `${caseId} rv-basal invalid technique: ${invalids.map((f) => f.message).join('; ')}`,
        );

        const errPct = (Math.abs(basal.valueCm - truth.rv.basalDiameterCm) / truth.rv.basalDiameterCm) * 100;
        assert.ok(
          errPct <= 14,
          `${caseId} rv-basal: measured ${basal.valueCm.toFixed(2)} cm vs truth ${truth.rv.basalDiameterCm.toFixed(2)} cm (${errPct.toFixed(1)}%)`,
        );
      } finally {
        core.dispose();
      }
    }
  });

  it('locates RVOT PW gate on PSAX-AV and measures Pulmonary Acceleration Time accurately (normal >105 ms vs PH <105 ms)', () => {
    for (const caseId of ['normal-excellent-window', 'pulmonary-hypertension-rv'] as const) {
      const c = loadCaseById(caseId);
      const truth = computeGroundTruth(c);
      const { heart, thorax } = buildCaseModels(c, REST_PATIENT);
      const ctrl = canonicalControl(getViewTarget('psax-av'), heart, thorax);
      const base = baseInput();
      const core = new SimulatorCore(c, {
        ...base,
        probe: ctrl,
        modality: '2d',
        quality: 'low',
        rendererBackend: 'procedural',
        display: { width: 640, height: 480 },
      });
      try {
        for (let i = 0; i < 20; i++) core.step(0.032);
        const dop = core.request({ kind: 'autoDopplerTarget', target: 'rvot-pw' });
        assert.ok(
          dop && dop.kind === 'autoDopplerTarget' && dop.result.ok,
          `${caseId}: rvot-pw target should succeed, got ${JSON.stringify(dop)}`,
        );
        assert.equal(dop.result.modality, 'pw');

        core.setInput({
          ...base,
          probe: ctrl,
          modality: 'pw',
          cursorThetaRad: dop.result.thetaRad,
          gateDepthCm: dop.result.rCm,
          spectral: {
            ...base.spectral,
            baselineShiftMps: dop.result.recommendedBaselineShiftMps ?? -0.25,
          },
          quality: 'low',
          rendererBackend: 'procedural',
          display: { width: 640, height: 480 },
        });
        let lastOut = core.step(0.032);
        for (let i = 0; i < 65; i++) lastOut = core.step(0.032);
        assert.ok(lastOut, `${caseId}: expected PW output`);

        const atRes = core.request({ kind: 'autoMeasureStrip', target: 'rvot-at' });
        assert.ok(
          atRes &&
            atRes.kind === 'autoMeasureStrip' &&
            atRes.result.ok &&
            atRes.result.target === 'rvot-at',
          `${caseId}: rvot-at should succeed, got ${JSON.stringify(atRes)}`,
        );
        const measuredAt = atRes.result.atMs;
        const truthAt = truth.rightHeart.rvotAccelerationTimeMs;
        const errPct = (Math.abs(measuredAt - truthAt) / truthAt) * 100;
        assert.ok(
          errPct <= 15,
          `${caseId} RVOT AT: measured ${measuredAt.toFixed(0)} ms vs truth ${truthAt.toFixed(0)} ms (${errPct.toFixed(1)}%)`,
        );
        if (caseId === 'pulmonary-hypertension-rv') {
          assert.ok(measuredAt < 105, `PH TAc must be < 105 ms, got ${measuredAt.toFixed(0)} ms`);
        } else {
          assert.ok(measuredAt > 110, `Normal TAc must be > 110 ms, got ${measuredAt.toFixed(0)} ms`);
        }

        const tech = evaluateTechnique(specById('rvot-acceleration-time'), {
          modality: 'pw',
          viewId: lastOut.view?.bestViewId ?? null,
          viewScore: lastOut.view?.score ?? null,
          phase: lastOut.phase,
          phaseMarks: lastOut.acquisition?.phaseMarks ?? null,
          gate: lastOut.gate,
          userAssisted: true,
        });
        const invalids = tech.findings.filter((f) => f.level === 'invalid');
        assert.equal(
          invalids.length,
          0,
          `${caseId} rvot-at invalid technique: ${invalids.map((f) => f.message).join('; ')}`,
        );
      } finally {
        core.dispose();
      }
    }
  });

  it('locates lateral tricuspid annulus TDI gate and measures RV S′ accurately in normal vs PH', () => {
    for (const caseId of ['normal-excellent-window', 'pulmonary-hypertension-rv'] as const) {
      const c = loadCaseById(caseId);
      const truth = computeGroundTruth(c);
      const { heart, thorax } = buildCaseModels(c, REST_PATIENT);
      const ctrl = canonicalControl(getViewTarget('rv-focused'), heart, thorax);
      const base = baseInput();
      const core = new SimulatorCore(c, {
        ...base,
        probe: ctrl,
        modality: '2d',
        quality: 'low',
        rendererBackend: 'procedural',
        display: { width: 640, height: 480 },
      });
      try {
        for (let i = 0; i < 20; i++) core.step(0.032);
        const dop = core.request({ kind: 'autoDopplerTarget', target: 'rv-s-prime' });
        assert.ok(
          dop && dop.kind === 'autoDopplerTarget' && dop.result.ok,
          `${caseId}: rv-s-prime target should succeed, got ${JSON.stringify(dop)}`,
        );
        assert.equal(dop.result.modality, 'tdi');

        core.setInput({
          ...base,
          probe: ctrl,
          modality: 'tdi',
          cursorThetaRad: dop.result.thetaRad,
          gateDepthCm: dop.result.rCm,
          spectral: { ...base.spectral, scaleMps: 0.25, wallFilterMps: 0.02, baselineShiftMps: 0 },
          quality: 'low',
          rendererBackend: 'procedural',
          display: { width: 640, height: 480 },
        });
        let lastOut = core.step(0.032);
        for (let i = 0; i < 65; i++) lastOut = core.step(0.032);
        assert.ok(lastOut, `${caseId}: expected PW output`);

        const spRes = core.request({ kind: 'autoMeasureStrip', target: 'rv-s-prime' });
        assert.ok(
          spRes &&
            spRes.kind === 'autoMeasureStrip' &&
            spRes.result.ok &&
            spRes.result.target === 'rv-s-prime',
          `${caseId}: rv-s-prime strip should succeed, got ${JSON.stringify(spRes)}`,
        );
        const measuredCmps = Math.abs(spRes.result.sPrime.velocityMps) * 100;
        const truthCmps = truth.rightHeart.sPrimeCmps;
        const errPct = (Math.abs(measuredCmps - truthCmps) / truthCmps) * 100;
        assert.ok(
          errPct <= 12,
          `${caseId} RV S′: measured ${measuredCmps.toFixed(1)} cm/s vs truth ${truthCmps.toFixed(1)} cm/s (${errPct.toFixed(1)}%)`,
        );

        const tech = evaluateTechnique(specById('rv-s-prime'), {
          modality: 'tdi',
          viewId: lastOut.view?.bestViewId ?? null,
          viewScore: lastOut.view?.score ?? null,
          phase: lastOut.phase,
          phaseMarks: lastOut.acquisition?.phaseMarks ?? null,
          gate: lastOut.gate,
          userAssisted: true,
        });
        // Modality, view, and lateral tricuspid annulus placement must all be strictly ok
        // (canonical RV-focused preset naturally has ~33° beam angle to lateral free wall until slid medially)
        for (const code of ['modality', 'view', 'placement'] as const) {
          const f = tech.findings.find((x) => x.code === code);
          assert.equal(
            f?.level,
            'ok',
            `${caseId} rv-s-prime expected ${code}=ok, got ${JSON.stringify(f)}`,
          );
        }
      } finally {
        core.dispose();
      }
    }
  });

  it('anchors M-mode TAPSE strictly to the lateral tricuspid annulus and distinguishes normal vs PH accurately', () => {
    for (const caseId of ['normal-excellent-window', 'pulmonary-hypertension-rv'] as const) {
      const c = loadCaseById(caseId);
      const truth = computeGroundTruth(c);
      const { heart, thorax } = buildCaseModels(c, REST_PATIENT);
      const ctrl = canonicalControl(getViewTarget('rv-focused'), heart, thorax);
      const base = baseInput();
      const core = new SimulatorCore(c, {
        ...base,
        probe: ctrl,
        modality: '2d',
        quality: 'low',
        rendererBackend: 'procedural',
        display: { width: 640, height: 480 },
      });
      try {
        for (let i = 0; i < 20; i++) core.step(0.032);
        const dop = core.request({ kind: 'autoDopplerTarget', target: 'tapse' });
        assert.ok(
          dop && dop.kind === 'autoDopplerTarget' && dop.result.ok,
          `${caseId}: tapse target should succeed`,
        );
        assert.equal(dop.result.modality, 'm-mode');

        core.setInput({
          ...base,
          probe: ctrl,
          modality: 'm-mode',
          cursorThetaRad: dop.result.thetaRad,
          gateDepthCm: dop.result.rCm,
          quality: 'low',
          rendererBackend: 'procedural',
          display: { width: 640, height: 480 },
        });
        let lastOut = core.step(0.032);
        for (let i = 0; i < 65; i++) lastOut = core.step(0.032);
        assert.ok(lastOut);

        const mRes = core.request({ kind: 'autoMeasureStrip', target: 'tapse' });
        assert.ok(
          mRes && mRes.kind === 'autoMeasureStrip' && mRes.result.ok && mRes.result.target === 'tapse',
          `${caseId}: autoMeasureStrip tapse should succeed, got ${JSON.stringify(mRes)}`,
        );
        const measured = mRes.result.tapseCm;
        const truthVal = truth.rightHeart.tapseCm;
        const errPct = (Math.abs(measured - truthVal) / truthVal) * 100;
        assert.ok(
          errPct <= 16,
          `${caseId} TAPSE: measured ${measured.toFixed(2)} cm vs truth ${truthVal.toFixed(2)} cm (${errPct.toFixed(1)}%)`,
        );
        if (caseId === 'pulmonary-hypertension-rv') {
          assert.ok(measured < 1.6, `PH TAPSE must be reduced (<1.6 cm), got ${measured.toFixed(2)} cm`);
        } else {
          assert.ok(measured >= 1.8, `Normal TAPSE must be >=1.8 cm, got ${measured.toFixed(2)} cm`);
        }

        const tech = evaluateTechnique(specById('tapse'), {
          modality: 'm-mode',
          viewId: lastOut.view?.bestViewId ?? null,
          viewScore: lastOut.view?.score ?? null,
          phase: lastOut.phase,
          phaseMarks: lastOut.acquisition?.phaseMarks ?? null,
          gate: lastOut.gate,
          userAssisted: true,
        });
        assert.equal(
          tech.findings.filter((f) => f.level === 'invalid').length,
          0,
          `${caseId} tapse invalid technique: ${JSON.stringify(tech.findings)}`,
        );
      } finally {
        core.dispose();
      }
    }
  });

  it('measures septal TDI e′ reliably in atrial fibrillation (short irregular RR) without incomplete-beat failure', () => {
    const c = loadCaseById('af-diastolic');
    const truth = computeGroundTruth(c);
    const { heart, thorax } = buildCaseModels(c, REST_PATIENT);
    const ctrl = canonicalControl(getViewTarget('a4c'), heart, thorax);
    const base = baseInput();
    const core = new SimulatorCore(c, {
      ...base,
      probe: ctrl,
      modality: '2d',
      quality: 'low',
      rendererBackend: 'procedural',
      display: { width: 640, height: 480 },
    });
    try {
      for (let i = 0; i < 20; i++) core.step(0.032);
      const dop = core.request({ kind: 'autoDopplerTarget', target: 'tdi-septal' });
      assert.ok(dop && dop.kind === 'autoDopplerTarget' && dop.result.ok);

      core.setInput({
        ...base,
        probe: ctrl,
        modality: 'tdi',
        cursorThetaRad: dop.result.thetaRad,
        gateDepthCm: dop.result.rCm,
        spectral: { ...base.spectral, scaleMps: 0.25, wallFilterMps: 0.02, baselineShiftMps: 0 },
        quality: 'low',
        rendererBackend: 'procedural',
        display: { width: 640, height: 480 },
      });
      for (let i = 0; i < 70; i++) core.step(0.032);

      const epRes = core.request({ kind: 'autoMeasureStrip', target: 'tdi-e-prime' });
      assert.ok(
        epRes && epRes.kind === 'autoMeasureStrip' && epRes.result.ok && epRes.result.target === 'tdi-e-prime',
        `af-diastolic tdi-e-prime should succeed, got ${JSON.stringify(epRes)}`,
      );
      const measuredCmps = Math.abs(epRes.result.ePrime.velocityMps) * 100;
      const errPct = (Math.abs(measuredCmps - truth.mitral.ePrimeSeptalCmps) / truth.mitral.ePrimeSeptalCmps) * 100;
      assert.ok(
        errPct <= 15,
        `af-diastolic e′ septal: measured ${measuredCmps.toFixed(1)} vs truth ${truth.mitral.ePrimeSeptalCmps.toFixed(1)} cm/s (${errPct.toFixed(1)}%)`,
      );
    } finally {
      core.dispose();
    }
  });
});
