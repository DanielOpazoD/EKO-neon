import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { biplaneVolume, deriveCalculations } from '@/education/report';
import { buildCaseModels, REST_PATIENT } from '@/simulator/anatomy/caseModels';
import { computeHeartPose } from '@/simulator/anatomy/heartModel';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import { baseInput } from '@/simulator/core/baseInput';
import { SimulatorCore } from '@/simulator/core/simulatorCore';
import { computeGroundTruth } from '@/simulator/hemodynamics/groundTruth';
import { beamFrameFromPose, poseFromControl } from '@/simulator/probe/pose';
import { ProceduralSliceRenderer } from '@/simulator/renderer/procedural/sliceRenderer';
import { scenePhysicsFor } from '@/simulator/renderer/scenePhysics';
import { allocPolarFrame, DEFAULT_ACQUISITION } from '@/simulator/renderer/types';
import { canonicalControl, getViewTarget } from '@/simulator/windows/viewTargets';
import { extractChamberContour, polygonAreaCm2, type ChamberId } from './autoVolume';
import type { Measurement } from './types';

const SPEC = {
  lines: 112,
  samples: 180,
  sectorRad: (80 * Math.PI) / 180,
  depthCm: 16,
  focusCm: 9,
  elevationSamples: 1,
};

function renderSlice(
  caseId: string,
  viewId: string,
  phase: number,
  probeOverride?: (c: ReturnType<typeof canonicalControl>) => ReturnType<typeof canonicalControl>,
) {
  const c = loadCaseById(caseId);
  const { heart, thorax, tables } = buildCaseModels(c, REST_PATIENT);
  const ctrl0 = canonicalControl(getViewTarget(viewId), heart, thorax);
  const ctrl = probeOverride ? probeOverride(ctrl0) : ctrl0;
  const beam = beamFrameFromPose(poseFromControl(thorax, ctrl));
  const pose = computeHeartPose(heart, cycleStateAt(tables, phase));
  const rdr = new ProceduralSliceRenderer();
  const frame = allocPolarFrame(SPEC);
  rdr.render(
    { heart, thorax, heartPose: pose, physics: scenePhysicsFor(c, DEFAULT_ACQUISITION) },
    beam,
    SPEC,
    phase,
    frame,
  );
  return { c, heart, thorax, tables, frame };
}

function measurePairOnView(caseId: string, viewId: string, chamber: ChamberId) {
  const c = loadCaseById(caseId);
  const { tables } = buildCaseModels(c, REST_PATIENT);
  const ed = renderSlice(caseId, viewId, 0);
  const es = renderSlice(caseId, viewId, tables.endSystoleS / tables.rrS);
  const extEd = extractChamberContour(ed.frame.structure, SPEC, chamber);
  const extEs = extractChamberContour(es.frame.structure, SPEC, chamber);
  return { ed: extEd, es: extEs };
}

describe('Automatic 2D LV and RV volume & ejection fraction', () => {
  it('traces LV endocardial contours on A4C/A2C accurately across normal, HFrEF, and volume-loaded MVP (and biplane)', () => {
    for (const id of ['normal-excellent-window', 'hfref-severe-mr', 'mvp-primary-mr'] as const) {
      const truth = computeGroundTruth(loadCaseById(id));
      const a4c = measurePairOnView(id, 'a4c', 'lv');
      const a2c = measurePairOnView(id, 'a2c', 'lv');
      assert.ok(a4c.ed && a4c.es, `${id}: expected LV contour on A4C`);
      assert.ok(a2c.ed && a2c.es, `${id}: expected LV contour on A2C`);

      const ef4 = ((a4c.ed.volumeMl - a4c.es.volumeMl) / a4c.ed.volumeMl) * 100;
      assert.ok(
        Math.abs(a4c.ed.volumeMl - truth.lv.edvMl) / truth.lv.edvMl < 0.08,
        `${id}: A4C EDV ${a4c.ed.volumeMl.toFixed(0)} vs truth ${truth.lv.edvMl.toFixed(0)}`,
      );
      assert.ok(
        Math.abs(ef4 - truth.lv.efPct) <= 4,
        `${id}: A4C EF ${ef4.toFixed(1)}% vs truth ${truth.lv.efPct.toFixed(1)}%`,
      );

      // Verify contours feed the report's biplane Simpson calculation seamlessly
      const mk = (
        measurementId: string,
        view: 'a4c' | 'a2c',
        pts: { x: number; y: number }[],
        val: number,
      ): Measurement => ({
        id: `${measurementId}-${view}`,
        measurementId,
        kind: 'volume',
        label: measurementId,
        value: val,
        units: 'mL',
        modality: '2d',
        createdAt: '2026-01-01T00:00:00Z',
        frameId: 1,
        phase: 0,
        timeS: 0,
        sourceViewId: view,
        viewScore: 95,
        imageQualityScore: 90,
        userAssisted: true,
        referenceGuidelineIds: ['ase-eacvi-chamber-2015'],
        geometry: pts.map((p) => ({ x: 300 + p.x * 20, y: 20 + p.y * 20 })),
        captureSector: {
          width: 600,
          height: 480,
          apexX: 300,
          apexY: 20,
          pxPerCm: 20,
          depthCm: 16,
          sectorRad: SPEC.sectorRad,
          invertLR: false,
        },
      });
      const ms = [
        mk('lv-edv-simpson', 'a4c', a4c.ed.contourCm, a4c.ed.volumeMl),
        mk('lv-esv-simpson', 'a4c', a4c.es.contourCm, a4c.es.volumeMl),
        mk('lv-edv-simpson', 'a2c', a2c.ed.contourCm, a2c.ed.volumeMl),
        mk('lv-esv-simpson', 'a2c', a2c.es.contourCm, a2c.es.volumeMl),
      ];
      const edBi = biplaneVolume(ms, 'lv-edv-simpson');
      const esBi = biplaneVolume(ms, 'lv-esv-simpson');
      assert.ok(edBi && esBi, `${id}: biplane volume should resolve from auto A4C + A2C contours`);
      const efBi = ((edBi.volumeMl - esBi.volumeMl) / edBi.volumeMl) * 100;
      assert.ok(
        Math.abs(edBi.volumeMl - truth.lv.edvMl) / truth.lv.edvMl < 0.06,
        `${id}: biplane EDV ${edBi.volumeMl.toFixed(0)} vs truth ${truth.lv.edvMl.toFixed(0)}`,
      );
      assert.ok(
        Math.abs(efBi - truth.lv.efPct) <= 3,
        `${id}: biplane EF ${efBi.toFixed(1)}% vs truth ${truth.lv.efPct.toFixed(1)}%`,
      );
    }
  });

  it('tracks real foreshortening on off-axis apical views instead of reading ground truth', () => {
    const onAxis = renderSlice('normal-excellent-window', 'a4c', 0);
    const foreshortened = renderSlice('normal-excellent-window', 'a4c', 0, (ctrl) => ({
      ...ctrl,
      v: ctrl.v + 2.2,
      tiltDeg: ctrl.tiltDeg - 14,
    }));
    const extGood = extractChamberContour(onAxis.frame.structure, SPEC, 'lv');
    const extShort = extractChamberContour(foreshortened.frame.structure, SPEC, 'lv');
    assert.ok(extGood && extShort, 'Both on-axis and foreshortened A4C should produce contours');
    assert.ok(
      extShort.longAxisCm < extGood.longAxisCm - 0.3 && extShort.volumeMl < extGood.volumeMl * 0.95,
      `Foreshortened view must reduce traced length (${extShort.longAxisCm.toFixed(1)} vs ${extGood.longAxisCm.toFixed(1)} cm) and volume (${extShort.volumeMl.toFixed(0)} vs ${extGood.volumeMl.toFixed(0)} mL)`,
    );
  });

  it('traces RV 2D area, FAC, volume, and RVEF accurately in normal vs dilated PH right ventricle', () => {
    const norm = measurePairOnView('normal-excellent-window', 'rv-focused', 'rv');
    const ph = measurePairOnView('pulmonary-hypertension-rv', 'rv-focused', 'rv');
    assert.ok(norm.ed && norm.es, 'Expected normal RV-focused contours');
    assert.ok(ph.ed && ph.es, 'Expected PH RV-focused contours');

    const normFac = ((norm.ed.areaCm2 - norm.es.areaCm2) / norm.ed.areaCm2) * 100;
    const normEf = ((norm.ed.volumeMl - norm.es.volumeMl) / norm.ed.volumeMl) * 100;
    assert.ok(normFac >= 40 && normFac <= 60, `Normal RV FAC expected 40–60%, got ${normFac.toFixed(1)}%`);
    assert.ok(normEf >= 55 && normEf <= 75, `Normal RVEF expected 55–75%, got ${normEf.toFixed(1)}%`);

    const phFac = ((ph.ed.areaCm2 - ph.es.areaCm2) / ph.ed.areaCm2) * 100;
    const phEf = ((ph.ed.volumeMl - ph.es.volumeMl) / ph.ed.volumeMl) * 100;
    assert.ok(ph.ed.volumeMl > norm.ed.volumeMl + 45, `PH RV EDV (${ph.ed.volumeMl.toFixed(0)}) should clearly exceed normal (${norm.ed.volumeMl.toFixed(0)})`);
    assert.ok(phFac < 35, `PH RV FAC must be reduced (< 35%), got ${phFac.toFixed(1)}%`);
    assert.ok(phEf < 45, `PH RVEF must be reduced (< 45%), got ${phEf.toFixed(1)}%`);
  });

  it('rejects non-apical slices (PLAX, PSAX-PM) from pure 2D slice geometry and computes polygonAreaCm2 accurately', () => {
    const plax = renderSlice('normal-excellent-window', 'plax', 0);
    const psax = renderSlice('normal-excellent-window', 'psax-pm', 0);
    assert.equal(extractChamberContour(plax.frame.structure, SPEC, 'lv'), null);
    assert.equal(extractChamberContour(plax.frame.structure, SPEC, 'rv'), null);
    assert.equal(extractChamberContour(psax.frame.structure, SPEC, 'lv'), null);
    assert.equal(extractChamberContour(psax.frame.structure, SPEC, 'rv'), null);

    // 4 cm x 3 cm rectangle at 20 px/cm => exact area 12 cm²
    const rect = [
      { x: 0, y: 0 },
      { x: 80, y: 0 },
      { x: 80, y: 60 },
      { x: 0, y: 60 },
    ];
    assert.ok(Math.abs(polygonAreaCm2(rect, 20) - 12) < 1e-9);
  });

  it('executes end-to-end autoChamberVolume RPC on SimulatorCore and resolves exact ED/ES cine frameIds', () => {
    const c = loadCaseById('normal-excellent-window');
    const { heart, thorax } = buildCaseModels(c, REST_PATIENT);
    const a4cCtrl = canonicalControl(getViewTarget('a4c'), heart, thorax);
    const inp = {
      ...baseInput(),
      probe: a4cCtrl,
      modality: '2d' as const,
      quality: 'low' as const,
      rendererBackend: 'procedural' as const,
      display: { width: 640, height: 480 },
    };
    const core = new SimulatorCore(c, inp);
    try {
      // Before a full beat is captured: incomplete-cycle rejection
      core.step(0.05);
      const early = core.request({ kind: 'autoChamberVolume', chamber: 'lv' });
      assert.ok(early && early.kind === 'autoChamberVolume' && !early.result.ok);
      assert.equal(early.result.reason, 'incomplete-cycle');

      // Step ~1.2 s (> 1 full beat at 65 bpm) so cine captures both ED and ES at A4C
      for (let i = 0; i < 36; i++) core.step(0.035);

      const lvRes = core.request({ kind: 'autoChamberVolume', chamber: 'lv' });
      assert.ok(lvRes && lvRes.kind === 'autoChamberVolume' && lvRes.result.ok, 'Expected LV Auto-FE to succeed');
      assert.ok(lvRes.result.edvMl > 95 && lvRes.result.edvMl < 145, `LV EDV=${lvRes.result.edvMl}`);
      assert.ok(lvRes.result.efPct >= 54 && lvRes.result.efPct <= 72, `LV EF=${lvRes.result.efPct}%`);
      assert.ok(lvRes.result.ed.contourPx.length >= 20 && lvRes.result.es.contourPx.length >= 20);

      // Exact frameId cine jump locates the exact traced ED & ES frames
      const jumpEd = core.request({
        kind: 'cinePhaseJump',
        target: 'ed',
        frameId: lvRes.result.ed.frameId,
      });
      const jumpEs = core.request({
        kind: 'cinePhaseJump',
        target: 'es',
        frameId: lvRes.result.es.frameId,
      });
      assert.ok(jumpEd && jumpEd.kind === 'cinePhaseJump' && jumpEd.offset === lvRes.result.ed.offset);
      assert.ok(jumpEs && jumpEs.kind === 'cinePhaseJump' && jumpEs.offset === lvRes.result.es.offset);

      // RV Auto-FE works on the same A4C cine loop too
      const rvRes = core.request({ kind: 'autoChamberVolume', chamber: 'rv' });
      assert.ok(rvRes && rvRes.kind === 'autoChamberVolume' && rvRes.result.ok, 'Expected RV Auto-FE to succeed');
      assert.ok(rvRes.result.edvMl > 70 && rvRes.result.edvMl < 170, `RV EDV=${rvRes.result.edvMl}`);
      assert.ok(rvRes.result.facPct >= 35, `RV FAC=${rvRes.result.facPct}%`);

      // Derived calculations integrate RV EF, SV, and FAC cleanly
      const derived = deriveCalculations([
        {
          id: 'r1',
          measurementId: 'rv-edv',
          kind: 'volume',
          label: 'VTD VD',
          value: rvRes.result.edvMl,
          units: 'mL',
          modality: '2d',
          sourceViewId: 'a4c',
          viewScore: 90,
          frameId: rvRes.result.ed.frameId,
          phase: rvRes.result.ed.phase,
          timeS: rvRes.result.ed.timeS,
          geometry: rvRes.result.ed.contourPx,
          captureSector: rvRes.result.ed.sector,
          derived: { areaCm2: rvRes.result.edAreaCm2 },
          imageQualityScore: 90,
          userAssisted: true,
          referenceGuidelineIds: ['ase-right-heart-2025'],
          createdAt: '2026-01-01T00:00:00Z',
        },
        {
          id: 'r2',
          measurementId: 'rv-esv',
          kind: 'volume',
          label: 'VTS VD',
          value: rvRes.result.esvMl,
          units: 'mL',
          modality: '2d',
          sourceViewId: 'a4c',
          viewScore: 90,
          frameId: rvRes.result.es.frameId,
          phase: rvRes.result.es.phase,
          timeS: rvRes.result.es.timeS,
          geometry: rvRes.result.es.contourPx,
          captureSector: rvRes.result.es.sector,
          derived: { areaCm2: rvRes.result.esAreaCm2 },
          imageQualityScore: 90,
          userAssisted: true,
          referenceGuidelineIds: ['ase-right-heart-2025'],
          createdAt: '2026-01-01T00:00:00Z',
        },
      ]);
      assert.ok(derived.some((d) => d.id === 'ef-rv'));
      assert.ok(derived.some((d) => d.id === 'sv-rv'));
      assert.ok(derived.some((d) => d.id === 'rv-fac'));

      // Wrong modality rejection (CW)
      core.setInput({ ...inp, modality: 'cw' });
      core.step(0.04);
      const cwRes = core.request({ kind: 'autoChamberVolume', chamber: 'lv' });
      assert.ok(cwRes && cwRes.kind === 'autoChamberVolume' && !cwRes.result.ok);
      assert.equal(cwRes.result.reason, 'wrong-modality');
    } finally {
      core.dispose();
    }
  });
});
