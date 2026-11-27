import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { MEASUREMENT_SPECS } from '@/simulator/measurements/protocol';
import type { Measurement } from '@/simulator/measurements/types';
import {
  captureModalityCheck,
  computeQuickStudySteps,
  isFreeToolCompatible,
  isTimeInCineWindow,
  phaseTag,
  specsForView,
  summarizeAutoEf,
} from './measureContext';

function makeSimp(
  measurementId: 'lv-edv-simpson' | 'lv-esv-simpson' | 'rv-edv' | 'rv-esv',
  viewId: string,
  value: number,
  frameId: number,
  areaCm2?: number,
): Measurement {
  // Simple symmetric LV/RV contour around a 7 cm long axis so biplaneVolume() computes real discs.
  const pxPerCm = 20;
  const contour = [
    { x: 160, y: 200 },
    { x: 150, y: 150 },
    { x: 160, y: 100 },
    { x: 190, y: 60 },
    { x: 220, y: 100 },
    { x: 230, y: 150 },
    { x: 220, y: 200 },
  ];
  return {
    id: `${measurementId}-${viewId}-${frameId}`,
    kind: 'volume',
    measurementId,
    technique: null,
    label: measurementId,
    value,
    units: 'mL',
    modality: '2d',
    sourceViewId: viewId,
    viewScore: 92,
    frameId,
    phase: measurementId.includes('edv') ? 0 : 0.42,
    timeS: 1,
    geometry: contour,
    captureSector: {
      apexX: 200,
      apexY: 10,
      pxPerCm,
      depthCm: 16,
      sectorRad: 1.4,
      width: 400,
      height: 360,
      invertLR: false,
    },
    derived: areaCm2 !== undefined ? { areaCm2 } : undefined,
    imageQualityScore: 90,
    userAssisted: true,
    referenceGuidelineIds: [],
    createdAt: '2026-01-01T00:00:00Z',
  };
}

describe('measurement context & compatibility helpers', () => {
  it('detects modality compatibility for protocol specs and free tools with 1-click switch targets', () => {
    const lvotVti = MEASUREMENT_SPECS.find((m) => m.id === 'lvot-vti')!;
    const lvEdd = MEASUREMENT_SPECS.find((m) => m.id === 'lv-edd')!;

    assert.deepEqual(captureModalityCheck(lvotVti, 'auto-vti', '2d'), {
      ok: false,
      suggestedModality: 'pw',
    });
    assert.deepEqual(captureModalityCheck(lvotVti, 'auto-vti', 'pw'), {
      ok: true,
      suggestedModality: null,
    });
    assert.deepEqual(captureModalityCheck(lvEdd, 'caliper', '2d'), {
      ok: true,
      suggestedModality: null,
    });
    assert.deepEqual(captureModalityCheck(undefined, 'velocity', '2d'), {
      ok: false,
      suggestedModality: 'pw',
    });
    assert.deepEqual(captureModalityCheck(undefined, 'tapse', '2d'), {
      ok: false,
      suggestedModality: 'm-mode',
    });
    assert.equal(isFreeToolCompatible('caliper', '2d'), true);
    assert.equal(isFreeToolCompatible('velocity', 'cw'), true);
    assert.equal(isFreeToolCompatible('velocity', '2d'), false);
    assert.equal(isFreeToolCompatible('tapse', 'm-mode'), true);
    assert.equal(isFreeToolCompatible('time', 'pw'), true);
    assert.equal(isFreeToolCompatible('time', '2d'), false);
  });

  it('orders view-matched measurements with active-modality specs first and required specs prioritized', () => {
    const plax2d = specsForView('plax', '2d', ['lv-edd', 'lvot-diameter']);
    assert.ok(plax2d.length >= 6);
    assert.ok(plax2d.every((s) => s.views.includes('plax')));
    // Required 2D specs come ahead of non-required 2D specs, and Color-only specs (VC IM) go after 2D specs.
    const idxEdd = plax2d.findIndex((s) => s.id === 'lv-edd');
    const idxRoot = plax2d.findIndex((s) => s.id === 'aortic-root');
    const idxVc = plax2d.findIndex((s) => s.id === 'mr-vena-contracta');
    assert.ok(idxEdd < idxRoot, 'required 2D PLAX spec precedes optional 2D spec');
    assert.ok(idxRoot < idxVc, '2D specs precede Color-only spec while in 2D');

    // Switching to Color promotes Color specs in PLAX to the top.
    const plaxColor = specsForView('plax', 'color', ['lv-edd']);
    assert.equal(plaxColor[0]!.modalities.includes('color'), true);
  });

  it('tracks Auto-FE single-plane -> biplane progression, RV FAC, and hides view readiness in exam mode', () => {
    const a4cOnly = [
      makeSimp('lv-edv-simpson', 'a4c', 122, 10),
      makeSimp('lv-esv-simpson', 'a4c', 45, 22),
      makeSimp('rv-edv', 'rv-focused', 115, 30, 17.2),
      makeSimp('rv-esv', 'rv-focused', 36, 42, 7.9),
    ];
    const s1 = summarizeAutoEf(a4cOnly, 'a4c', '2d', true);
    assert.equal(s1.is2dReady, true);
    assert.equal(s1.lv.a4cDone, true);
    assert.equal(s1.lv.a2cDone, false);
    assert.equal(s1.lv.biplane, false);
    assert.equal(s1.lv.viewReady, true);
    assert.equal(s1.lv.edFrameId, 10);
    assert.equal(s1.lv.esFrameId, 22);
    assert.ok(Math.abs((s1.lv.efPct ?? 0) - 63.1) < 0.5);
    assert.equal(s1.rv.hasResult, true);
    assert.ok((s1.rv.facPct ?? 0) > 50 && (s1.rv.facPct ?? 0) < 58);

    // Adding A2C upgrades automatically to biplane and exposes explicit A4C vs A2C breakdown.
    const withA2c = [
      ...a4cOnly,
      makeSimp('lv-edv-simpson', 'a2c', 118, 50),
      makeSimp('lv-esv-simpson', 'a2c', 44, 62),
    ];
    const sBi = summarizeAutoEf(withA2c, 'plax', '2d', true);
    assert.equal(sBi.lv.a4cDone, true);
    assert.equal(sBi.lv.a2cDone, true);
    assert.equal(sBi.lv.biplane, true);
    assert.equal(sBi.lv.viewReady, false);
    assert.equal(sBi.lv.a4c?.edvMl, 122);
    assert.equal(sBi.lv.a4c?.esvMl, 45);
    assert.equal(sBi.lv.a2c?.edvMl, 118);
    assert.equal(sBi.lv.a2c?.esvMl, 44);
    assert.equal(sBi.lv.lengthMismatch, false);

    // Foreshortened A2C contour (>10% shorter long axis) triggers ASE lengthMismatch alert.
    const shortA2cEd = makeSimp('lv-edv-simpson', 'a2c', 90, 70);
    shortA2cEd.geometry = [
      { x: 160, y: 200 },
      { x: 155, y: 170 },
      { x: 165, y: 140 },
      { x: 190, y: 110 },
      { x: 215, y: 140 },
      { x: 225, y: 170 },
      { x: 220, y: 200 },
    ];
    const sShort = summarizeAutoEf(
      [...a4cOnly, shortA2cEd, makeSimp('lv-esv-simpson', 'a2c', 36, 75)],
      'a2c',
      '2d',
      true,
    );
    assert.equal(sShort.lv.lengthMismatch, true);

    // In exam mode (`showViewFeedback = false`), view readiness stays strictly null.
    const sExam = summarizeAutoEf(withA2c, 'a4c', '2d', false);
    assert.equal(sExam.lv.viewReady, null);
    assert.equal(sExam.rv.viewReady, null);
    assert.equal(sExam.plax.viewReady, null);
    assert.equal(sExam.la.viewReady, null);
  });

  it('summarizes PLAX linear dimensions (with Devereux mass/RWT/FS) and LA monoplane/biplane progression', () => {
    const mkLin = (
      id: string,
      value: number,
      frameId: number,
      phase: number,
    ): Measurement => ({
      ...makeSimp('lv-edv-simpson', 'plax', value, frameId),
      kind: 'linear',
      measurementId: id,
      units: 'cm',
      phase,
      geometry: [
        { x: 120, y: 120 },
        { x: 120, y: 200 },
      ],
    });
    const mkLa = (view: 'a4c' | 'a2c', val: number, frameId: number, lenPx = 100): Measurement => ({
      ...makeSimp('lv-esv-simpson', view, val, frameId),
      measurementId: 'la-volume',
      geometry: [
        { x: 170, y: 200 },
        { x: 165, y: 200 + lenPx * 0.5 },
        { x: 190, y: 200 + lenPx },
        { x: 215, y: 200 + lenPx * 0.5 },
        { x: 210, y: 200 },
      ],
      derived: { areaCm2: 18.5, longAxisCm: lenPx / 20 },
    });

    const ms = [
      mkLin('ivsd', 1.0, 11, 0),
      mkLin('lv-edd', 4.8, 11, 0),
      mkLin('lvpwd', 0.9, 11, 0),
      mkLin('lvot-diameter', 2.1, 15, 0.2),
      mkLin('lv-esd', 3.1, 19, 0.42),
      mkLin('aortic-root', 3.2, 11, 0),
      mkLin('la-ap', 3.6, 19, 0.42),
      mkLa('a4c', 52, 25, 100),
    ];
    const sPlax = summarizeAutoEf(ms, 'plax', '2d', true);
    assert.equal(sPlax.plax.viewReady, true);
    assert.equal(sPlax.plax.hasResult, true);
    assert.equal(sPlax.plax.lvEddCm, 4.8);
    assert.equal(sPlax.plax.lvotCm, 2.1);
    assert.equal(sPlax.plax.edFrameId, 11);
    assert.equal(sPlax.plax.midSysFrameId, 15);
    assert.equal(sPlax.plax.esFrameId, 19);
    assert.ok((sPlax.plax.massG ?? 0) > 140 && (sPlax.plax.massG ?? 0) < 170);
    assert.ok(Math.abs((sPlax.plax.rwt ?? 0) - 0.375) < 0.005);
    assert.ok(Math.abs((sPlax.plax.fsPct ?? 0) - 35.4) < 0.5);

    // Monoplane LA before A2C
    assert.equal(sPlax.la.hasResult, true);
    assert.equal(sPlax.la.a4cDone, true);
    assert.equal(sPlax.la.a2cDone, false);
    assert.equal(sPlax.la.biplane, false);
    assert.equal(sPlax.la.volumeMl, 52);

    // Add A2C LA with matching length -> biplane LA
    const sLaBi = summarizeAutoEf([...ms, mkLa('a2c', 48, 35, 104)], 'a2c', '2d', true);
    assert.equal(sLaBi.la.viewReady, true);
    assert.equal(sLaBi.la.biplane, true);
    assert.equal(sLaBi.la.a4cDone, true);
    assert.equal(sLaBi.la.a2cDone, true);
    assert.equal(sLaBi.la.lengthMismatch, false);

    // Add foreshortened A2C LA (>0.5 cm shorter -> ASE lengthMismatch warning)
    const sLaShort = summarizeAutoEf([...ms, mkLa('a2c', 40, 40, 80)], 'a2c', '2d', true);
    assert.equal(sLaShort.la.lengthMismatch, true);
  });

  it('formats phase tags concisely in Spanish and checks rolling cine window membership', () => {
    assert.equal(phaseTag('ed'), 'TD');
    assert.equal(phaseTag('es'), 'TS');
    assert.equal(phaseTag('mid-systole'), 'Mesosíst.');
    assert.equal(phaseTag('any'), null);
    assert.equal(isTimeInCineWindow(10.5, { startS: 9.0, endS: 12.0 }), true);
    assert.equal(isTimeInCineWindow(7.2, { startS: 9.0, endS: 12.0 }), false);
    assert.equal(isTimeInCineWindow(null, { startS: 9.0, endS: 12.0 }), false);
  });

  it('derives the 4-step Quick Study clinical checklist purely from measurements', () => {
    const empty = computeQuickStudySteps([], 'plax');
    assert.equal(empty.completedCount, 0);
    assert.equal(empty.steps[0]!.active, true);
    assert.equal(empty.steps[1]!.targetViewId, 'a4c');

    // Add PLAX + A4C only -> PLAX done, VI FE partial (1/2) targeting A2C next
    const mid: Measurement[] = [
      { ...makeSimp('lv-edv-simpson', 'plax', 4.8, 1), kind: 'linear', measurementId: 'lv-edd', units: 'cm' },
      makeSimp('lv-edv-simpson', 'a4c', 120, 10),
      makeSimp('lv-esv-simpson', 'a4c', 45, 20),
    ];
    const sMid = computeQuickStudySteps(mid, 'a4c');
    assert.equal(sMid.completedCount, 1);
    assert.equal(sMid.steps[0]!.status, 'done');
    assert.equal(sMid.steps[1]!.status, 'partial');
    assert.equal(sMid.steps[1]!.detail, '1/2');
    assert.equal(sMid.steps[1]!.targetViewId, 'a2c');

    // Complete A2C + LVOT VTI + RV TAPSE -> 4/4 done
    const full: Measurement[] = [
      ...mid,
      makeSimp('lv-edv-simpson', 'a2c', 118, 30),
      makeSimp('lv-esv-simpson', 'a2c', 44, 40),
      { ...makeSimp('lv-edv-simpson', 'a5c', 19.2, 50), kind: 'vti', measurementId: 'lvot-vti', modality: 'pw', units: 'cm' },
      { ...makeSimp('rv-edv', 'a4c', 2.2, 60), kind: 'linear', measurementId: 'tapse', modality: 'm-mode', units: 'cm' },
    ];
    const sFull = computeQuickStudySteps(full, 'rv-focused');
    assert.equal(sFull.completedCount, 4);
    assert.ok(sFull.steps.every((st) => st.status === 'done'));
    assert.equal(sFull.steps[3]!.active, true);
  });
});
