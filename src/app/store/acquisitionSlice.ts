import { constrainAcquisition } from '@/simulator/core/acquisitionInput';
import type { ProbeControl } from '@/simulator/probe/pose';
import { DEFAULT_ACQUISITION } from '@/simulator/renderer/types';
import { DEFAULT_COLOR } from '@/simulator/doppler/color/colorDoppler';
import {
  DEFAULT_SPECTRAL,
  SPECTRAL_MODE_DEFAULTS,
  spectralVelocitySetup,
  type SpectralModality,
} from '@/simulator/doppler/spectral/spectrum';
import { DEFAULT_QUALITY } from '@/simulator/core/protocol';
import { easeInOut, lerpControl, presetDurationMs } from '@/simulator/probe/interpolate';
import { isSpectralModality } from '@/simulator/renderer/modality';
import { frameBus } from '../frameBus';
import {
  START_PROBE,
  START_VIEW_ID,
  atStartProbe,
  cancelPendingPresets,
  clampProbe,
  clampSettings,
  currentPresetToken,
  nextPresetToken,
} from './helpers';
import { useHudStore } from './uiSlice';
import type { AcquisitionSlice, SimSliceCreator, SimStore } from './types';

export const createAcquisitionSlice: SimSliceCreator<AcquisitionSlice> = (set, get) => ({
  probe: { ...START_PROBE },
  patient: { position: 'left-lateral', respiration: 'expiration', headElevationDeg: 0 },
  settings: { ...DEFAULT_ACQUISITION, tgcDb: [...DEFAULT_ACQUISITION.tgcDb] },
  modality: '2d',
  frozen: false,
  cineOffset: 0,
  cinePlaying: false,
  cineSpeed: 1,
  color: { ...DEFAULT_COLOR },
  spectral: { ...DEFAULT_SPECTRAL },
  spectralByMode: {},
  cursorThetaRad: 0.05,
  gateDepthCm: 9,
  quality: DEFAULT_QUALITY,
  rendererBackend: 'atlas',
  anatomy: 'analytic',
  imageFormation: 'lattice',
  artifactLab: null,
  presetAnim: null,

  setProbe: (p) => set((s) => ({ probe: clampProbe({ ...s.probe, ...p }), presetAnim: null })),
  nudgeProbe: (p) =>
    set((s) => {
      const n = { ...s.probe };
      for (const k of Object.keys(p) as (keyof ProbeControl)[]) n[k] = (n[k] ?? 0) + (p[k] ?? 0);
      return { probe: clampProbe(n), presetAnim: null };
    }),
  setPatient: (p) => set((s) => ({ patient: { ...s.patient, ...p } })),
  setSettings: (st) =>
    set((s) => constrainAcquisition({ ...s, settings: clampSettings({ ...s.settings, ...st }) })),
  setTgc: (band, db) =>
    set((s) => {
      const tgc = [...s.settings.tgcDb];
      tgc[band] = Math.max(-15, Math.min(15, db));
      return { settings: { ...s.settings, tgcDb: tgc } };
    }),
  resetTgc: () =>
    set((s) => ({
      settings: { ...s.settings, tgcDb: [0, 0, 0, 0, 0, 0, 0, 0] },
    })),
  autoOptimize: () => {
    void frameBus
      .request({ kind: 'autoOpt' })
      .then((res) => {
        if (res && res.kind === 'autoOpt') {
          get().setSettings({ gainDb: res.gainDb, tgcDb: res.tgcDb });
        }
      })
      .catch((e: unknown) => {
        console.warn('autoOpt request failed', e instanceof Error ? e.message : e);
      });
  },
  setModality: (m) =>
    set((s) => {
      const out: Partial<SimStore> = {
        modality: m,
        frozen: false,
        cineOffset: 0,
        cinePlaying: false,
      };
      // each spectral mode opens with its own velocity window, or with the one the learner left it with (decision 230)
      const from = isSpectralModality(s.modality) ? (s.modality as SpectralModality) : null;
      const to = isSpectralModality(m) ? (m as SpectralModality) : null;
      if (from && from !== to)
        out.spectralByMode = { ...s.spectralByMode, [from]: spectralVelocitySetup(s.spectral) };
      if (to && to !== from) {
        const memory = (out.spectralByMode ?? s.spectralByMode)[to];
        out.spectral = { ...s.spectral, ...(memory ?? SPECTRAL_MODE_DEFAULTS[to]) };
      }
      return constrainAcquisition({ ...s, ...out });
    }),
  toggleFreeze: () => set((s) => ({ frozen: !s.frozen, cineOffset: 0, cinePlaying: false })),
  setCineOffset: (o) =>
    set(() => ({
      cineOffset: Math.min(0, Math.max(-((useHudStore.getState().hud?.cineLength ?? 1) - 1), o)),
      cinePlaying: false,
    })),
  stepCine: () =>
    set((s) => {
      const length = useHudStore.getState().hud?.cineLength ?? 1;
      return { cineOffset: s.cineOffset >= 0 ? -(length - 1) : s.cineOffset + 1 };
    }),
  setCinePlaying: (p) => set({ cinePlaying: p }),
  setCineSpeed: (sp) => set({ cineSpeed: sp }),
  jumpCinePhase: (target) => {
    void frameBus
      .request({ kind: 'cinePhaseJump', target })
      .then((res) => {
        if (res && res.kind === 'cinePhaseJump') {
          get().setCineOffset(res.offset);
        }
      })
      .catch((e: unknown) => {
        console.warn('cinePhaseJump request failed', e instanceof Error ? e.message : e);
      });
  },
  setColor: (c) => set((s) => constrainAcquisition({ ...s, color: { ...s.color, ...c } })),
  setSpectral: (sp) =>
    set((s) => constrainAcquisition({ ...s, spectral: { ...s.spectral, ...sp } })),
  setCursor: (theta, depth) =>
    set((s) =>
      constrainAcquisition({ ...s, cursorThetaRad: theta, gateDepthCm: depth ?? s.gateDepthCm }),
    ),
  setQuality: (q) => set({ quality: q }),
  setBackend: (b) => set({ rendererBackend: b }),
  setAnatomy: (a) => set({ anatomy: a }),
  setImageFormation: (f) => set({ imageFormation: f }),
  setArtifactLab: (v) => set({ artifactLab: v }),

  startPresetView: (viewId, opts) => {
    if (get().mode === 'exam') return;
    // from now on this view's score is the preset's, not the learner's (decision 174). The opening jump is not a preset
    // the learner used: it holds its view back only until the learner loses it (`openingView`), or free practice, which
    // makes the jump on every case it opens, left the first view of the curriculum out of reach for good
    if (!opts?.instant && !get().presetViews.includes(viewId))
      set((s) => ({ presetViews: [...s.presetViews, viewId] }));
    const token = nextPresetToken();
    const begin = (to: ProbeControl) =>
      set((s) => {
        // the pose arrives asynchronously: a session started or a preset cancelled meanwhile does not get it, even if it
        // went back to practice by then
        if (token !== currentPresetToken() || s.mode === 'exam') return s;
        // the opening jump (decision 343) yields to a probe moved while its pose was on the way
        if (opts?.instant)
          return atStartProbe(s.probe)
            ? {
                probe: to,
                presetAnim: null,
                targetViewId: viewId,
                frozen: false,
                openingView: { viewId, pose: to, released: false },
              }
            : s;
        const from = { ...s.probe };
        return {
          presetAnim: {
            from,
            to,
            startMs: performance.now(),
            durationMs: presetDurationMs(from, to),
            viewId,
          },
          targetViewId: viewId,
          frozen: false,
        };
      });
    // the pose comes from the worker's own models — the ones the image is traced from (audit B7); without a
    // worker (tests, inline mode) the main-thread copies answer, as they did before
    void frameBus
      .request({ kind: 'canonicalControl', viewId })
      .then((res) => {
        if (res && res.kind === 'canonicalControl') return begin(clampProbe(res.control));
        // no worker answered (tests, inline mode): the main-thread copies compute it, loaded on demand so the
        // anatomy engine stays out of the entry chunk
        return Promise.all([
          import('../caseModels'),
          import('@/simulator/windows/viewTargets'),
        ]).then(([cm, vt]) => {
          const { heart, thorax } = cm.getCaseModels(get().caseId, get().patient);
          begin(clampProbe(vt.canonicalControl(vt.getViewTarget(viewId), heart, thorax)));
        });
      })
      .catch((e: unknown) => {
        console.warn('preset view request failed', e instanceof Error ? e.message : e);
      });
  },
  tickPresetAnimation: (nowMs) =>
    set((s) => {
      const a = s.presetAnim;
      if (!a) return s;
      const t = (nowMs - a.startMs) / a.durationMs;
      const probe = lerpControl(a.from, a.to, easeInOut(t));
      return t >= 1 ? { probe: a.to, presetAnim: null } : { probe };
    }),
  resetProbe: () => {
    // a preset pose still on its way does not undo the reset. From the start pose any view reached is the learner's, the
    // one free practice opened on included; analyses made at the opening pose still do not count (`appPlacedPose`), and
    // the frames on their way carry them
    cancelPendingPresets();
    set((s) => ({
      probe: { ...START_PROBE },
      presetAnim: null,
      openingView: s.openingView && { ...s.openingView, released: true },
    }));
  },
  openOnStartView: () => {
    const s = get();
    if (s.mode !== 'sandbox' || s.presetAnim || s.targetViewId) return;
    if (atStartProbe(s.probe)) s.startPresetView(START_VIEW_ID, { instant: true });
  },
});
