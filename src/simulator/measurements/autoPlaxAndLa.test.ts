import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import {
  fractionalShorteningPct,
  lvMassDevereuxG,
  relativeWallThickness,
} from '@/clinical/formulas';
import { biplaneVolume, deriveCalculations } from '@/education/report';
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

function runCoreCine(
  caseId: string,
  viewId: string,
  opts?: { depthCm?: number; steps?: number; stepS?: number },
) {
  const c = loadCaseById(caseId);
  const { heart, thorax } = buildCaseModels(c, REST_PATIENT);
  const ctrl = canonicalControl(getViewTarget(viewId), heart, thorax);
  const base = baseInput();
  const inp = {
    ...base,
    probe: ctrl,
    modality: '2d' as const,
    quality: 'low' as const,
    rendererBackend: 'procedural' as const,
    display: { width: 640, height: 480 },
    settings: {
      ...base.settings,
      ...(opts?.depthCm !== undefined ? { depthCm: opts.depthCm } : {}),
    },
  };
  const core = new SimulatorCore(c, inp);
  const steps = opts?.steps ?? 42;
  const stepS = opts?.stepS ?? 0.032;
  for (let i = 0; i < steps; i++) core.step(stepS);
  return { core, inp, truth: computeGroundTruth(c) };
}

describe('Automatic 2D PLAX linear dimensions & LA volume (Paquete 1)', () => {
  it('computes Devereux LV mass, LVMI, RWT, and FS accurately and surfaces them in deriveCalculations', () => {
    // Classic ASE reference: LVEDD 4.8 cm, IVSd 0.9 cm, LVPWd 0.9 cm, LVESD 3.0 cm, BSA 1.9 m²
    // Devereux: 0.8 * (1.04 * (6.6³ - 4.8³)) + 0.6 = 147.8 g
    const mass = lvMassDevereuxG(4.8, 0.9, 0.9);
    assert.ok(Math.abs(mass - 147.8) < 0.5, `Expected Devereux mass ~147.8 g, got ${mass.toFixed(1)}`);
    assert.ok(Math.abs(relativeWallThickness(4.8, 0.9) - 0.375) < 1e-6);
    assert.ok(Math.abs(fractionalShorteningPct(4.8, 3.0) - 37.5) < 1e-6);

    const mkLin = (id: string, value: number): Measurement => ({
      id,
      measurementId: id,
      kind: 'linear',
      label: id,
      value,
      units: 'cm',
      modality: '2d',
      sourceViewId: 'plax',
      viewScore: 95,
      frameId: 1,
      phase: 0,
      timeS: 0,
      geometry: [
        { x: 100, y: 100 },
        { x: 100, y: 200 },
      ],
      imageQualityScore: 90,
      userAssisted: true,
      referenceGuidelineIds: ['ase-eacvi-chamber-2015'],
      createdAt: '2026-01-01T00:00:00Z',
    });

    const derived = deriveCalculations(
      [mkLin('ivsd', 0.9), mkLin('lv-edd', 4.8), mkLin('lvpwd', 0.9), mkLin('lv-esd', 3.0)],
      1.9,
    );
    const byId = new Map(derived.map((d) => [d.id, d]));
    assert.ok(byId.has('lv-fs'), 'expected lv-fs');
    assert.ok(byId.has('lv-rwt'), 'expected lv-rwt');
    assert.ok(byId.has('lv-mass'), 'expected lv-mass');
    assert.ok(byId.has('lv-mass-index'), 'expected lv-mass-index');
    assert.equal(byId.get('lv-fs')!.value, '38 %');
    assert.equal(byId.get('lv-rwt')!.value, '0.38');
  });

  it('extracts all 7 PLAX linear dimensions accurately with valid technique across normal, HFrEF, HCM, and severe AS', () => {
    const cases = [
      'normal-excellent-window',
      'hfref-severe-mr',
      'hocm-sam',
      'aortic-stenosis-severe',
    ] as const;

    for (const caseId of cases) {
      const { core, truth } = runCoreCine(caseId, 'plax');
      try {
        const resp = core.request({ kind: 'autoPlax' });
        assert.ok(
          resp && resp.kind === 'autoPlax' && resp.result.ok,
          `${caseId}: autoPlax should succeed, got ${JSON.stringify(resp)}`,
        );
        const res = resp.result;
        const byId = new Map(res.calipers.map((c) => [c.measurementId, c]));

        for (const expectedId of [
          'ivsd',
          'lv-edd',
          'lvpwd',
          'lv-esd',
          'lvot-diameter',
          'aortic-root',
          'la-ap',
        ] as const) {
          const cal = byId.get(expectedId);
          assert.ok(cal, `${caseId}: missing PLAX caliper ${expectedId}`);

          // Verify zero invalid technique findings for every committed caliper
          const spec = specById(expectedId);
          const tech = evaluateTechnique(spec, {
            modality: cal.acquisition.modality,
            viewId: cal.viewId,
            viewScore: cal.viewScore,
            phase: cal.phase,
            phaseMarks: cal.acquisition.phaseMarks,
            gate: null,
            segmentStructures: cal.segmentStructures,
            segmentEndsOutside: cal.segmentEndsOutside,
            userAssisted: true,
          });
          const invalids = tech.findings.filter((f) => f.level === 'invalid');
          assert.equal(
            invalids.length,
            0,
            `${caseId}/${expectedId}: invalid technique findings: ${invalids.map((f) => f.message).join('; ')}`,
          );
        }

        // Compare every PLAX dimension against case ground truth within clinical ASE tolerances
        const checkTol = (id: keyof typeof expectedTruth, tolPct: number) => {
          const measured = byId.get(id)!.valueCm;
          const target = expectedTruth[id];
          const errPct = (Math.abs(measured - target) / target) * 100;
          assert.ok(
            errPct <= tolPct,
            `${caseId}/${id}: measured ${measured.toFixed(2)} cm vs truth ${target.toFixed(2)} cm (err ${errPct.toFixed(1)}% > ${tolPct}%)`,
          );
        };
        const expectedTruth = {
          ivsd: truth.lv.ivsdCm,
          'lv-edd': truth.lv.eddCm,
          lvpwd: truth.lv.lvpwdCm,
          'lv-esd': truth.lv.esdCm,
          'lvot-diameter': truth.lvot.diameterCm,
          'aortic-root': truth.aorta.sinusCm,
          'la-ap': truth.la.apDiameterCm,
        } as const;

        checkTol('ivsd', 15);
        checkTol('lv-edd', 10);
        checkTol('lvpwd', 15);
        checkTol('lv-esd', 12);
        checkTol('lvot-diameter', 12);
        checkTol('aortic-root', 10);
        // HFrEF's canonical 2D PLAX slice is anatomically clipped posteriorly by the descending aorta (~3.9 cm vs 4.7 cm 3D AP)
        checkTol('la-ap', caseId === 'hfref-severe-mr' ? 20 : 12);
      } finally {
        core.dispose();
      }
    }
  });

  it('rejects autoPlax cleanly on A4C and on non-2D Doppler modalities', () => {
    const { core, inp } = runCoreCine('normal-excellent-window', 'a4c');
    try {
      const a4cRes = core.request({ kind: 'autoPlax' });
      assert.ok(a4cRes && a4cRes.kind === 'autoPlax' && !a4cRes.result.ok);
      assert.equal(a4cRes.result.reason, 'not-plax');

      core.setInput({ ...inp, modality: 'pw' });
      core.step(0.04);
      const pwRes = core.request({ kind: 'autoPlax' });
      assert.ok(pwRes && pwRes.kind === 'autoPlax' && !pwRes.result.ok);
      assert.equal(pwRes.result.reason, 'wrong-modality');
    } finally {
      core.dispose();
    }
  });

  it('traces end-systolic LA volume & biplane LAVI accurately across normal, AF diastolic, and severe MS', () => {
    const cases = [
      { id: 'normal-excellent-window', depthCm: 16 },
      { id: 'af-diastolic', depthCm: 18 },
      { id: 'mitral-stenosis-severe', depthCm: 18 },
    ] as const;

    for (const { id: caseId, depthCm } of cases) {
      const a4cRun = runCoreCine(caseId, 'a4c', { depthCm });
      const a2cRun = runCoreCine(caseId, 'a2c', { depthCm });
      try {
        const r4 = a4cRun.core.request({ kind: 'autoLaVolume' });
        const r2 = a2cRun.core.request({ kind: 'autoLaVolume' });
        assert.ok(
          r4 && r4.kind === 'autoLaVolume' && r4.result.ok,
          `${caseId} A4C: autoLaVolume should succeed, got ${JSON.stringify(r4)}`,
        );
        assert.ok(
          r2 && r2.kind === 'autoLaVolume' && r2.result.ok,
          `${caseId} A2C: autoLaVolume should succeed, got ${JSON.stringify(r2)}`,
        );

        const spec = specById('la-volume');
        for (const [planeName, res] of [
          ['a4c', r4.result],
          ['a2c', r2.result],
        ] as const) {
          const tech = evaluateTechnique(spec, {
            modality: res.acquisition.modality,
            viewId: res.viewId,
            viewScore: res.viewScore,
            phase: res.phase,
            phaseMarks: res.acquisition.phaseMarks,
            gate: null,
            segmentStructures: res.interiorStructures,
            cutByDepth: res.cutByDepth,
            userAssisted: true,
          });
          const invalids = tech.findings.filter((f) => f.level === 'invalid');
          assert.equal(
            invalids.length,
            0,
            `${caseId}/${planeName} LA: invalid technique findings: ${invalids.map((f) => f.message).join('; ')}`,
          );
        }

        const mkLa = (view: 'a4c' | 'a2c', res: typeof r4.result): Measurement => ({
          id: `la-${view}`,
          measurementId: 'la-volume',
          kind: 'volume',
          label: 'Volumen de AI',
          value: res.volumeMl,
          units: 'mL',
          modality: '2d',
          sourceViewId: view,
          viewScore: res.viewScore ?? 95,
          frameId: res.frameId,
          phase: res.phase,
          timeS: res.timeS,
          geometry: res.contourPx,
          captureSector: res.sector,
          derived: { areaCm2: res.areaCm2, longAxisCm: res.longAxisCm },
          imageQualityScore: res.gainScore ?? 90,
          userAssisted: true,
          referenceGuidelineIds: ['ase-eacvi-chamber-2015'],
          createdAt: '2026-01-01T00:00:00Z',
        });

        const ms = [mkLa('a4c', r4.result), mkLa('a2c', r2.result)];
        const bi = biplaneVolume(ms, 'la-volume');
        assert.ok(bi, `${caseId}: biplane LA volume should compute from auto A4C + A2C contours`);

        const truthVol = a4cRun.truth.la.volumeMl;
        const errPct = (Math.abs(bi.volumeMl - truthVol) / truthVol) * 100;
        assert.ok(
          errPct <= 20,
          `${caseId}: biplane LA vol ${bi.volumeMl.toFixed(1)} mL (A4C ${r4.result.volumeMl.toFixed(1)}, A2C ${r2.result.volumeMl.toFixed(1)}) vs truth ${truthVol.toFixed(1)} mL (err ${errPct.toFixed(1)}%)`,
        );

        const derived = deriveCalculations(ms, a4cRun.truth.bsaM2);
        assert.ok(derived.some((d) => d.id === 'la-volume-biplane'));
        assert.ok(derived.some((d) => d.id === 'la-volume-index'));
      } finally {
        a4cRun.core.dispose();
        a2cRun.core.dispose();
      }
    }
  });

  it('rejects autoLaVolume cleanly on PLAX and CW modality', () => {
    const { core, inp } = runCoreCine('normal-excellent-window', 'plax');
    try {
      const plaxRes = core.request({ kind: 'autoLaVolume' });
      assert.ok(plaxRes && plaxRes.kind === 'autoLaVolume' && !plaxRes.result.ok);
      assert.equal(plaxRes.result.reason, 'chamber-not-visible');

      core.setInput({ ...inp, modality: 'cw' });
      core.step(0.04);
      const cwRes = core.request({ kind: 'autoLaVolume' });
      assert.ok(cwRes && cwRes.kind === 'autoLaVolume' && !cwRes.result.ok);
      assert.equal(cwRes.result.reason, 'wrong-modality');
    } finally {
      core.dispose();
    }
  });
});
