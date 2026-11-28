import assert from 'node:assert/strict';
import { beforeEach, describe, it } from 'node:test';
import { useHudStore, useSimStore } from '@/app/store';
import type { SimOutput } from '@/simulator/core/protocol';
import type { ImagingModality } from '@/simulator/renderer/types';
import {
  AUTO_MEASURE_SPECS,
  cancelGuidedAutoMeasure,
  prepareCleanAcquisition,
  runGuidedAutoMeasure,
  type OrchestratorDeps,
} from './autoMeasureOrchestrator';

function makeHud(overrides: Partial<SimOutput> = {}): SimOutput {
  const st = useSimStore.getState();
  return {
    frameId: 1,
    timeS: 0,
    phase: 0,
    rrS: 0.8,
    beatIndex: 1,
    frozen: st.frozen,
    cineLength: 30,
    cineWindow: { startS: 0, endS: 1.5 },
    view: {
      bestViewId: 'plax',
      score: 94,
      components: { plane: 1, landmarks: 1, geometry: 1, centering: 1, depth: 1, gain: 0.9, artifacts: 1 },
      hints: [],
      probe: st.probe,
    },
    sector: {
      apexX: 200,
      apexY: 10,
      pxPerCm: 20,
      depthCm: 16,
      sectorRad: 1.4,
      width: 400,
      height: 360,
      invertLR: false,
    },
    strip: {
      kind: null,
      x: 0,
      y: 360,
      width: 400,
      height: 120,
      columns: 200,
      headColumn: 100,
      secondsPerColumn: 0.015,
      topValue: 1.5,
      bottomValue: -1.5,
    },
    acquisition: {
      modality: st.modality,
      probe: st.probe,
      patient: st.patient,
      settings: st.settings,
      color: st.color,
      spectral: st.spectral,
      cursorThetaRad: st.cursorThetaRad,
      gateDepthCm: st.gateDepthCm,
      stripAcquisitionId: 1,
      phaseMarks: {
        endSystole: 0.36,
        ejectionStart: 0.06,
        ejectionEnd: 0.36,
        mitralOpen: 0.44,
        eEnd: 0.72,
        aPeak: 0.88,
      },
    },
    gate: null,
    ...overrides,
  } as SimOutput;
}

function stubDeps(calls: string[]): OrchestratorDeps {
  return {
    runAutoPlax: async () => {
      calls.push('runAutoPlax');
      return {
        ok: true,
        edFrameId: 10,
        midSysFrameId: 12,
        esFrameId: 14,
        calipers: [
          { measurementId: 'lv-edd', valueCm: 4.5 },
          { measurementId: 'lv-esd', valueCm: 3.1 },
          { measurementId: 'lvot-diameter', valueCm: 2.1 },
        ],
      } as never;
    },
    runAutoChamberVolume: async (chamber) => {
      calls.push(`runAutoChamberVolume:${chamber}`);
      return {
        ok: true,
        chamber,
        efPct: 62,
        facPct: 48,
        ed: { volumeMl: 118 },
        es: { volumeMl: 45 },
        rvBasal: chamber === 'rv' ? { valueCm: 3.4 } : undefined,
      } as never;
    },
    runAutoLaVolume: async () => {
      calls.push('runAutoLaVolume');
      return { ok: true, volumeMl: 48, cutByDepth: false } as never;
    },
    runAutoIvc: async () => {
      const resp = useSimStore.getState().patient.respiration;
      calls.push(`runAutoIvc:${resp}`);
      useSimStore.setState({ frozen: true });
      return {
        ok: true,
        measurementId: resp === 'inspiration' ? 'ivc-diameter-inspiration' : 'ivc-diameter',
        valueCm: resp === 'inspiration' ? 0.55 : 1.7,
      } as never;
    },
    runAutoLvotGate: async () => {
      calls.push('runAutoLvotGate');
      useSimStore.getState().setModality('pw');
      return { ok: true, thetaRad: 0.05, rCm: 8.5, distToAvCm: 0.5 } as never;
    },
    runAutoDopplerTarget: async (target) => {
      calls.push(`runAutoDopplerTarget:${target}`);
      const mod: ImagingModality =
        target === 'tapse'
          ? 'm-mode'
          : target === 'av-cw' || target === 'tr-cw'
            ? 'cw'
            : target === 'tdi-septal' || target === 'tdi-lateral' || target === 'rv-s-prime'
              ? 'tdi'
              : 'pw';
      useSimStore.getState().setModality(mod);
      return { ok: true, target, modality: mod, thetaRad: 0.1, rCm: 9.0, summary: 'OK' } as never;
    },
    runAutoSpectralVti: async (id) => {
      calls.push(`runAutoSpectralVti:${id}`);
      return { ok: true, vtiCm: 20.5, vmaxMps: 1.05, meanGradientMmHg: 3 } as never;
    },
    runAutoStripMeasure: async (target, opts) => {
      calls.push(`runAutoStripMeasure:${target}:${opts?.tdiSite ?? ''}`);
      if (target === 'tapse') return { ok: true, target, tapseCm: 2.2 } as never;
      return { ok: true, target, beatIndex: 2, e: { xPx: 20, velocityMps: 0.8 } } as never;
    },
  };
}

describe('1-click guided auto-measurement orchestrator', () => {
  beforeEach(() => {
    cancelGuidedAutoMeasure();
    useSimStore.setState({
      mode: 'sandbox',
      examFinished: false,
      modality: '2d',
      frozen: false,
      cinePlaying: false,
      activeTool: 'none',
      activeMeasurementId: null,
      teachingPointerOn: false,
      teachingArrows: [],
      presetAnim: null,
      patient: { position: 'left-lateral', respiration: 'expiration', headElevationDeg: 0 },
    });
    useHudStore.getState().setHud(makeHud());
  });

  it('cleanly disarms active manual tools, teaching pointer, playback, and review mode', () => {
    useSimStore.setState({
      activeTool: 'caliper',
      activeMeasurementId: 'lv-edd',
      teachingPointerOn: true,
      teachingArrows: [{ id: 'a1', x0: 1, y0: 1, x1: 2, y1: 2, onStrip: false }],
      cinePlaying: true,
      ui: { ...useSimStore.getState().ui, reviewMode: true },
    });

    prepareCleanAcquisition();

    const s = useSimStore.getState();
    assert.equal(s.activeTool, 'none');
    assert.equal(s.activeMeasurementId, null);
    assert.equal(s.teachingPointerOn, false);
    assert.equal(s.teachingArrows.length, 0);
    assert.equal(s.cinePlaying, false);
    assert.equal(s.ui.reviewMode, false);
  });

  it('switches off conflicting modes, moves to target view, activates target Doppler modality, waits 1 beat, and measures', async () => {
    const calls: string[] = [];
    const deps = stubDeps(calls);
    let presetRequested: string | null = null;

    // Start in Color Doppler on PLAX; click VTI TSVI (requires A5C + PW)
    useSimStore.setState({
      modality: 'color',
      startPresetView: (viewId: string) => {
        presetRequested = viewId;
        useSimStore.setState({
          targetViewId: viewId,
          presetAnim: {
            from: useSimStore.getState().probe,
            to: useSimStore.getState().probe,
            startMs: 0,
            durationMs: 10,
            viewId,
          },
        });
      },
    });

    let frame = 10;
    let timeS = 1.0;
    const pump = setInterval(() => {
      const s = useSimStore.getState();
      frame++;
      timeS += 0.08;
      if (s.presetAnim) {
        useSimStore.setState({ presetAnim: null });
      }
      const mod = s.modality;
      const ph = (timeS % 0.8) / 0.8;
      const beat = Math.floor(timeS / 0.8) + 1;
      useHudStore.getState().setHud(
        makeHud({
          frameId: frame,
          timeS,
          phase: ph,
          beatIndex: beat,
          view: {
            bestViewId: presetRequested ?? 'plax',
            score: 95,
            components: { plane: 1, landmarks: 1, geometry: 1, centering: 1, depth: 1, gain: 0.9, artifacts: 1 },
            hints: [],
            probe: s.probe,
          } as unknown as SimOutput['view'],
          strip: {
            kind: mod === 'pw' || mod === 'cw' || mod === 'tdi' ? 'spectral' : mod === 'm-mode' ? 'm-mode' : null,
            x: 0,
            y: 360,
            width: 400,
            height: 120,
            columns: 200,
            headColumn: 100,
            secondsPerColumn: 0.015,
            topValue: 1.5,
            bottomValue: -1.5,
          },
          acquisition: {
            ...makeHud().acquisition!,
            modality: mod,
            stripAcquisitionId: mod === 'pw' ? 7 : 1,
          },
        }),
      );
    }, 8);

    try {
      const res = await runGuidedAutoMeasure('lvot-vti', undefined, deps);
      assert.equal(res.ok, true, res.message);
      assert.equal(presetRequested, 'a5c');
      assert.equal(useSimStore.getState().modality, 'pw');
      assert.deepEqual(calls, ['runAutoLvotGate', 'runAutoSpectralVti:lvot-vti']);
    } finally {
      clearInterval(pump);
    }
  });

  it('orchestrates full IVC expiration + inspiratory sniff in subcostal-supine posture in 1 click', async () => {
    const calls: string[] = [];
    const deps = stubDeps(calls);
    let presetRequested: string | null = null;
    useSimStore.setState({
      modality: 'cw',
      startPresetView: (viewId: string) => {
        presetRequested = viewId;
        useSimStore.setState({ targetViewId: viewId, presetAnim: null });
      },
    });

    let frame = 1;
    let timeS = 0;
    const pump = setInterval(() => {
      frame++;
      timeS += 0.05;
      const s = useSimStore.getState();
      useHudStore.getState().setHud(
        makeHud({
          frameId: frame,
          timeS,
          frozen: s.frozen,
          view: {
            bestViewId: presetRequested ?? 'plax',
            score: 94,
            components: { plane: 1, landmarks: 1, geometry: 1, centering: 1, depth: 1, gain: 0.9, artifacts: 1 },
            hints: [],
            probe: s.probe,
          } as unknown as SimOutput['view'],
        }),
      );
    }, 6);

    try {
      const res = await runGuidedAutoMeasure('ivc-full', undefined, deps);
      assert.equal(res.ok, true, res.message);
      assert.equal(presetRequested, 'subcostal-ivc');
      assert.equal(useSimStore.getState().patient.position, 'subcostal-supine');
      assert.equal(useSimStore.getState().modality, '2d');
      assert.deepEqual(calls, ['runAutoIvc:expiration', 'runAutoIvc:inspiration']);
      assert.match(res.message, /colapso 68 %/);
    } finally {
      clearInterval(pump);
    }
  });

  it('strictly preserves exam mode privacy: never moves probe, never changes posture/modality, never auto-gates', async () => {
    const calls: string[] = [];
    const deps = stubDeps(calls);
    let presetCalled = false;
    useSimStore.setState({
      mode: 'exam',
      modality: 'm-mode',
      patient: { position: 'left-lateral', respiration: 'expiration', headElevationDeg: 0 },
      startPresetView: () => {
        presetCalled = true;
      },
    });

    const res = await runGuidedAutoMeasure('tapse', undefined, deps);
    assert.equal(res.ok, true);
    assert.equal(presetCalled, false, 'Exam mode must never invoke startPresetView');
    assert.equal(useSimStore.getState().modality, 'm-mode');
    assert.deepEqual(calls, ['runAutoStripMeasure:tapse:']);
  });

  it('supports immediate cancellation while waiting for view transition', async () => {
    const calls: string[] = [];
    const deps = stubDeps(calls);
    useSimStore.setState({
      modality: 'pw',
      startPresetView: (viewId: string) => {
        useSimStore.setState({
          presetAnim: {
            from: useSimStore.getState().probe,
            to: useSimStore.getState().probe,
            startMs: 0,
            durationMs: 5000,
            viewId,
          },
        });
      },
    });

    const promise = runGuidedAutoMeasure('rv-2d', undefined, deps);
    setTimeout(() => cancelGuidedAutoMeasure(), 15);
    const res = await promise;
    assert.equal(res.ok, false);
    assert.equal(res.cancelled, true);
    assert.equal(calls.length, 0);
  });

  it('defines all 16 clinical 1-click measurement specs with valid views and modality badges', () => {
    const ids = Object.keys(AUTO_MEASURE_SPECS);
    assert.equal(ids.length, 16);
    for (const spec of Object.values(AUTO_MEASURE_SPECS)) {
      assert.ok(spec.label.length > 0);
      assert.ok(spec.targetViewId.length > 0);
      assert.ok(['2D', 'PW', 'CW', 'TDI', 'M'].includes(spec.modalityBadge));
    }
  });
});
