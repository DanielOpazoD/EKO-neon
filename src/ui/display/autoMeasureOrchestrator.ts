import { examClosed, modePolicy } from '@/app/modePolicy';
import { useHudStore, useSimStore } from '@/app/store';
import type { AutoStripMeasureTarget } from '@/simulator/core/strip/autoMeasureStrip';
import type { DopplerTargetId } from '@/simulator/measurements/autoDopplerTarget';
import type { ImagingModality } from '@/simulator/renderer/types';
import { viewLabel } from '@/simulator/windows/viewLabels';
import {
  runAutoChamberVolume,
  runAutoDopplerTarget,
  runAutoIvc,
  runAutoLaVolume,
  runAutoLvotGate,
  runAutoPlax,
  runAutoSpectralVti,
  runAutoStripMeasure,
} from './measurementTools';

/** Every clinical 1-click automatic measurement supported by the guided orchestrator. */
export type AutoMeasureActionId =
  | 'plax-all'
  | 'lv-a4c'
  | 'lv-a2c'
  | 'la-a4c'
  | 'la-a2c'
  | 'lvot-vti'
  | 'av-vti'
  | 'mitral-ea'
  | 'tdi-septal'
  | 'tdi-lateral'
  | 'rv-2d'
  | 'tapse'
  | 'rv-s-prime'
  | 'tr-vmax'
  | 'rvot-at'
  | 'ivc-full';

export interface AutoMeasureSpec {
  id: AutoMeasureActionId;
  label: string;
  shortLabel: string;
  modalityBadge: '2D' | 'PW' | 'CW' | 'TDI' | 'M';
  targetViewId: string;
  targetModality: ImagingModality;
  posture: 'left-lateral' | 'subcostal-supine';
  tip: string;
}

export const AUTO_MEASURE_SPECS: Record<AutoMeasureActionId, AutoMeasureSpec> = {
  'plax-all': {
    id: 'plax-all',
    label: 'Dimensiones PLAX',
    shortLabel: 'PLAX completo',
    modalityBadge: '2D',
    targetViewId: 'plax',
    targetModality: '2d',
    posture: 'left-lateral',
    tip: 'Cambia a PLAX 2D y mide SIVd, DTDVI, PPd, raíz Ao (TD), TSVI ⌀ (mesosístole) y DTSVI + AI (TS) en la misma línea perpendicular.',
  },
  'lv-a4c': {
    id: 'lv-a4c',
    label: 'FEVI · A4C',
    shortLabel: 'FEVI A4C',
    modalityBadge: '2D',
    targetViewId: 'a4c',
    targetModality: '2d',
    posture: 'left-lateral',
    tip: 'Cambia a A4C 2D y traza el endocardio del VI en TD y TS (Simpson).',
  },
  'lv-a2c': {
    id: 'lv-a2c',
    label: 'FEVI · A2C',
    shortLabel: 'FEVI A2C',
    modalityBadge: '2D',
    targetViewId: 'a2c',
    targetModality: '2d',
    posture: 'left-lateral',
    tip: 'Cambia a A2C 2D y traza el endocardio del VI en TD y TS para completar Simpson biplano.',
  },
  'la-a4c': {
    id: 'la-a4c',
    label: 'Vol AI · A4C',
    shortLabel: 'Vol AI A4C',
    modalityBadge: '2D',
    targetViewId: 'a4c',
    targetModality: '2d',
    posture: 'left-lateral',
    tip: 'Cambia a A4C 2D y mide el volumen de la aurícula izquierda en telesístole.',
  },
  'la-a2c': {
    id: 'la-a2c',
    label: 'Vol AI · A2C',
    shortLabel: 'Vol AI A2C',
    modalityBadge: '2D',
    targetViewId: 'a2c',
    targetModality: '2d',
    posture: 'left-lateral',
    tip: 'Cambia a A2C 2D y completa el volumen biplano indexado de la AI (LAVI).',
  },
  'lvot-vti': {
    id: 'lvot-vti',
    label: 'VTI TSVI',
    shortLabel: 'VTI TSVI',
    modalityBadge: 'PW',
    targetViewId: 'a5c',
    targetModality: 'pw',
    posture: 'left-lateral',
    tip: 'Cambia a A5C, coloca el gate PW en el TSVI (~0,5 cm de la VAo), adquiere 1 latido y traza el VTI.',
  },
  'av-vti': {
    id: 'av-vti',
    label: 'VTI / Vmáx VAo',
    shortLabel: 'VTI VAo',
    modalityBadge: 'CW',
    targetViewId: 'a5c',
    targetModality: 'cw',
    posture: 'left-lateral',
    tip: 'Cambia a A5C, alinea Doppler continuo (CW) sobre la válvula aórtica, adquiere 1 latido y mide VTI, Vmáx y ΔPm.',
  },
  'mitral-ea': {
    id: 'mitral-ea',
    label: 'Mitral E / A / TD',
    shortLabel: 'E / A / TD',
    modalityBadge: 'PW',
    targetViewId: 'a4c',
    targetModality: 'pw',
    posture: 'left-lateral',
    tip: 'Cambia a A4C, coloca el gate PW en las puntas mitrales, adquiere 1 latido y mide onda E, onda A y tiempo de desaceleración.',
  },
  'tdi-septal': {
    id: 'tdi-septal',
    label: 'e′ Septal',
    shortLabel: 'e′ septal',
    modalityBadge: 'TDI',
    targetViewId: 'a4c',
    targetModality: 'tdi',
    posture: 'left-lateral',
    tip: 'Cambia a A4C, coloca el gate TDI en el anillo mitral septal, adquiere 1 latido y mide e′ septal.',
  },
  'tdi-lateral': {
    id: 'tdi-lateral',
    label: 'e′ Lateral',
    shortLabel: 'e′ lateral',
    modalityBadge: 'TDI',
    targetViewId: 'a4c',
    targetModality: 'tdi',
    posture: 'left-lateral',
    tip: 'Cambia a A4C, coloca el gate TDI en el anillo mitral lateral, adquiere 1 latido y mide e′ lateral.',
  },
  'rv-2d': {
    id: 'rv-2d',
    label: 'FEVD / CAF / Basal',
    shortLabel: 'VD 2D + basal',
    modalityBadge: '2D',
    targetViewId: 'rv-focused',
    targetModality: '2d',
    posture: 'left-lateral',
    tip: 'Cambia a apical enfocada en VD 2D y mide FEVD, cambio de área fraccional (CAF) y diámetro basal del VD.',
  },
  tapse: {
    id: 'tapse',
    label: 'TAPSE',
    shortLabel: 'TAPSE',
    modalityBadge: 'M',
    targetViewId: 'rv-focused',
    targetModality: 'm-mode',
    posture: 'left-lateral',
    tip: 'Cambia a apical enfocada en VD, alinea Modo M en el anillo tricuspídeo lateral, adquiere 1 latido y mide TAPSE.',
  },
  'rv-s-prime': {
    id: 'rv-s-prime',
    label: 'S′ VD',
    shortLabel: 'S′ VD',
    modalityBadge: 'TDI',
    targetViewId: 'rv-focused',
    targetModality: 'tdi',
    posture: 'left-lateral',
    tip: 'Cambia a apical enfocada en VD, coloca el gate TDI en el anillo tricuspídeo lateral, adquiere 1 latido y mide S′ VD.',
  },
  'tr-vmax': {
    id: 'tr-vmax',
    label: 'Vmáx IT',
    shortLabel: 'Vmáx IT',
    modalityBadge: 'CW',
    targetViewId: 'rv-focused',
    targetModality: 'cw',
    posture: 'left-lateral',
    tip: 'Cambia a apical enfocada en VD, alinea CW sobre la válvula tricúspide, adquiere 1 latido y mide Vmáx IT y ΔP (para PSVD).',
  },
  'rvot-at': {
    id: 'rvot-at',
    label: 'TAc Pulmonar',
    shortLabel: 'TAc pulm.',
    modalityBadge: 'PW',
    targetViewId: 'psax-av',
    targetModality: 'pw',
    posture: 'left-lateral',
    tip: 'Cambia a PSAX válvula aórtica, coloca el gate PW en el TSVD / válvula pulmonar y mide el tiempo de aceleración pulmonar.',
  },
  'ivc-full': {
    id: 'ivc-full',
    label: 'VCI (Esp + Insp)',
    shortLabel: 'VCI + PAD',
    modalityBadge: '2D',
    targetViewId: 'subcostal-ivc',
    targetModality: '2d',
    posture: 'subcostal-supine',
    tip: 'Cambia a subcostal VCI en decúbito supino y mide automáticamente espiración + inspiración (olfateo) para calcular colapso y PAD.',
  },
};

export interface OrchestratorProgress {
  actionId: AutoMeasureActionId;
  phaseLabel: string;
}

export interface OrchestratorResult {
  ok: boolean;
  cancelled?: boolean;
  message: string;
}

export interface OrchestratorDeps {
  runAutoPlax: typeof runAutoPlax;
  runAutoChamberVolume: typeof runAutoChamberVolume;
  runAutoLaVolume: typeof runAutoLaVolume;
  runAutoIvc: typeof runAutoIvc;
  runAutoLvotGate: typeof runAutoLvotGate;
  runAutoDopplerTarget: typeof runAutoDopplerTarget;
  runAutoSpectralVti: typeof runAutoSpectralVti;
  runAutoStripMeasure: typeof runAutoStripMeasure;
}

const DEFAULT_DEPS: OrchestratorDeps = {
  runAutoPlax,
  runAutoChamberVolume,
  runAutoLaVolume,
  runAutoIvc,
  runAutoLvotGate,
  runAutoDopplerTarget,
  runAutoSpectralVti,
  runAutoStripMeasure,
};

let activeToken = 0;
const cancelWaiters = new Set<() => void>();

/** Cancel any in-flight 1-click guided auto-measurement immediately. */
export function cancelGuidedAutoMeasure(): void {
  activeToken++;
  for (const fn of [...cancelWaiters]) fn();
  cancelWaiters.clear();
}

function isCancelled(token: number): boolean {
  return token !== activeToken;
}

/**
 * Cleanly disarm any conflicting manual tools, annotations, playback, or overlays before starting
 * a guided auto-measurement. Safe in every mode (never moves the probe or changes view).
 */
export function prepareCleanAcquisition(): void {
  const st = useSimStore.getState();
  if (st.activeMeasurementId !== null) st.setActiveMeasurement(null);
  if (st.activeTool !== 'none') st.setActiveTool('none');
  if (st.teachingPointerOn) st.setTeachingPointer(false);
  if (st.teachingArrows.length > 0) st.clearTeachingArrows();
  if (st.cinePlaying) st.setCinePlaying(false);
  if (st.ui.reviewMode) st.setUi({ reviewMode: false });
}

/** Event-driven wait on HUD + SimStore updates with instant cancellation and hard timeout. */
function waitUntil(
  predicate: () => boolean,
  timeoutMs: number,
  token: number,
): Promise<boolean> {
  if (isCancelled(token)) return Promise.resolve(false);
  if (predicate()) return Promise.resolve(true);

  return new Promise<boolean>((resolve) => {
    let settled = false;
    let unsubHud: (() => void) | null = null;
    let unsubSim: (() => void) | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const finish = (ok: boolean) => {
      if (settled) return;
      settled = true;
      cancelWaiters.delete(onCancel);
      if (timer !== null) clearTimeout(timer);
      unsubHud?.();
      unsubSim?.();
      resolve(ok);
    };

    const onCancel = () => finish(false);
    const check = () => {
      if (isCancelled(token)) {
        finish(false);
        return;
      }
      if (predicate()) finish(true);
    };

    cancelWaiters.add(onCancel);
    unsubHud = useHudStore.subscribe(check);
    unsubSim = useSimStore.subscribe(check);
    timer = setTimeout(() => finish(predicate() && !isCancelled(token)), timeoutMs);
  });
}

async function waitNextHudFrames(count: number, token: number, timeoutMs = 2500): Promise<boolean> {
  const startId = useHudStore.getState().hud?.frameId ?? 0;
  let seen = 0;
  let lastId = startId;
  return waitUntil(
    () => {
      const h = useHudStore.getState().hud;
      if (!h) return false;
      if (h.frameId !== lastId) {
        lastId = h.frameId;
        seen++;
      }
      return seen >= count;
    },
    timeoutMs,
    token,
  );
}

/**
 * Switch cleanly out of any conflicting modality into 2D, adjust patient posture/respiration,
 * animate the probe to `targetViewId`, and wait for either a full cardiac cycle (`needsFullBeat`)
 * or a target cardiac phase (`desiredPhaseBand`) so the subsequent pure-2D detector sees crisp anatomy.
 */
async function ensureViewReady(
  targetViewId: string,
  posture: 'left-lateral' | 'subcostal-supine',
  respiration: 'expiration' | 'inspiration',
  opts: {
    needsFullBeat?: boolean;
    desiredPhase?: 'ed' | 'mid-systole' | 'early-diastole' | 'any';
  },
  token: number,
  emit: (phaseLabel: string) => void,
): Promise<boolean> {
  const st0 = useSimStore.getState();
  const hud0 = useHudStore.getState().hud;

  const postureMatches =
    st0.patient.position === posture && st0.patient.respiration === respiration;
  const viewMatches =
    hud0 !== null &&
    hud0.view?.bestViewId === targetViewId &&
    (hud0.view.score ?? 0) >= 65 &&
    st0.presetAnim === null;
  const modalityIs2d = st0.modality === '2d';

  // Fast path: already cleanly on target 2D view with matching posture
  if (postureMatches && viewMatches && modalityIs2d) {
    if (opts.needsFullBeat) {
      const rr = Math.max(0.55, hud0.rrS || 0.85);
      const cineSpan = hud0.cineWindow ? hud0.cineWindow.endS - hud0.cineWindow.startS : 0;
      if (hud0.cineLength >= 16 && cineSpan >= rr * 0.95) {
        return true;
      }
      // Need a bit more live cine on this view
      if (st0.frozen) st0.toggleFreeze();
      emit(`Capturando 1 ciclo en ${viewLabel(targetViewId)}…`);
      const t0 = useHudStore.getState().hud?.timeS ?? hud0.timeS;
      return waitUntil(
        () => {
          const h = useHudStore.getState().hud;
          if (!h || h.frozen) return false;
          const span = h.cineWindow ? h.cineWindow.endS - h.cineWindow.startS : 0;
          return h.timeS - t0 >= Math.max(0.75, rr * 1.05) || (h.cineLength >= 18 && span >= rr * 1.05);
        },
        3800,
        token,
      );
    }
    return true;
  }

  // Deactivate any conflicting Doppler / M-mode / Color mode and unfreeze cleanly
  emit(`Cambiando a ${viewLabel(targetViewId)}…`);
  if (st0.modality !== '2d') {
    st0.setModality('2d');
  } else if (st0.frozen) {
    st0.toggleFreeze();
  }

  // Update patient posture/respiration BEFORE canonicalControl so worker rebuilds thorax first
  if (!postureMatches) {
    useSimStore.getState().setPatient({ position: posture, respiration });
    const settled = await waitNextHudFrames(2, token, 1800);
    if (!settled || isCancelled(token)) return false;
  }

  // Animate probe smoothly to target canonical window if not already centered on it
  const hudAfterPosture = useHudStore.getState().hud;
  const alreadyOnView =
    hudAfterPosture !== null &&
    hudAfterPosture.view?.bestViewId === targetViewId &&
    (hudAfterPosture.view.score ?? 0) >= 72 &&
    useSimStore.getState().presetAnim === null;

  if (!alreadyOnView) {
    const startFrameId = hudAfterPosture?.frameId ?? 0;
    useSimStore.getState().startPresetView(targetViewId);

    // Wait for presetAnim to appear and complete (plus 2 settled frames after animation ends),
    // OR for the view to recognize cleanly if the pose change was instantaneous.
    let sawAnim = false;
    let animEndFrameId: number | null = null;
    const arrived = await waitUntil(
      () => {
        const s = useSimStore.getState();
        const h = useHudStore.getState().hud;
        if (s.presetAnim !== null) {
          sawAnim = true;
          animEndFrameId = null;
          return false;
        }
        if (!h || h.frozen) return false;
        if (sawAnim) {
          if (animEndFrameId === null) {
            animEndFrameId = h.frameId;
            return false;
          }
          return h.frameId >= animEndFrameId + 2;
        }
        return h.view?.bestViewId === targetViewId && (h.view.score ?? 0) >= 65 && h.frameId >= startFrameId + 4;
      },
      3600,
      token,
    );
    if (!arrived || isCancelled(token)) return false;
  }

  // Wait for 2D cine frames to settle at the new probe pose
  const arrivedHud = useHudStore.getState().hud;
  const arrivedTimeS = arrivedHud?.timeS ?? 0;
  const arrivedFrameId = arrivedHud?.frameId ?? 0;
  const rr = Math.max(0.55, arrivedHud?.rrS || 0.85);

  if (opts.needsFullBeat) {
    emit(`Capturando ciclo cardíaco en ${viewLabel(targetViewId)}…`);
    return waitUntil(
      () => {
        const h = useHudStore.getState().hud;
        if (!h || h.frozen) return false;
        return h.timeS - arrivedTimeS >= Math.max(1.0, rr * 1.15) && h.frameId >= arrivedFrameId + 10;
      },
      4200,
      token,
    );
  }

  // Wait for at least 3 settled frames at the target view AND (when requested) the ideal anatomical phase
  const phaseOk = (ph: number): boolean => {
    switch (opts.desiredPhase) {
      case 'ed':
        return ph <= 0.12 || ph >= 0.88;
      case 'mid-systole':
        return ph >= 0.1 && ph <= 0.34;
      case 'early-diastole':
        return ph >= 0.44 && ph <= 0.72;
      default:
        return true;
    }
  };
  let seenDesiredPhase = false;
  return waitUntil(
    () => {
      const h = useHudStore.getState().hud;
      if (!h || h.frozen) return false;
      if (h.frameId < arrivedFrameId + 3) return false;
      if (phaseOk(h.phase)) seenDesiredPhase = true;
      return seenDesiredPhase || h.timeS - arrivedTimeS >= Math.max(0.9, rr * 1.05);
    },
    2800,
    token,
  );
}

/**
 * Wait event-driven until a newly opened live Doppler or M-mode strip has recorded at least ONE
 * complete cardiac cycle covering `[reqMinPhase, reqMaxPhase]` on the same `beatIndex`.
 */
async function waitForCompleteStripBeat(
  expectedModality: ImagingModality,
  expectedKind: 'spectral' | 'm-mode',
  needsDiastole: boolean,
  token: number,
): Promise<boolean> {
  const reqMinPhase = 0.08;
  const reqMaxPhase = needsDiastole ? 0.9 : 0.5;
  let stripAcqId: number | null = null;
  let stripStartTimeS: number | null = null;
  const beatSpan = new Map<number, { minP: number; maxP: number; count: number }>();

  return waitUntil(
    () => {
      const h = useHudStore.getState().hud;
      if (!h || h.frozen) return false;
      const acqMod = h.acquisition?.modality;
      if (acqMod !== expectedModality || h.strip.kind !== expectedKind) return false;

      const curId = h.acquisition?.stripAcquisitionId ?? 0;
      if (stripAcqId === null || curId !== stripAcqId) {
        stripAcqId = curId;
        stripStartTimeS = h.timeS;
        beatSpan.clear();
      }
      const b = beatSpan.get(h.beatIndex) ?? { minP: 1, maxP: 0, count: 0 };
      b.minP = Math.min(b.minP, h.phase);
      b.maxP = Math.max(b.maxP, h.phase);
      b.count++;
      beatSpan.set(h.beatIndex, b);

      const elapsed = stripStartTimeS !== null ? h.timeS - stripStartTimeS : 0;
      if (elapsed < 0.45) return false;

      for (const sp of beatSpan.values()) {
        if (sp.count >= 6 && sp.minP <= reqMinPhase && sp.maxP >= reqMaxPhase) {
          return true;
        }
      }
      const rr = Math.max(0.6, h.rrS || 0.85);
      return elapsed >= Math.max(2.1, rr * 2.25);
    },
    5200,
    token,
  );
}

interface StripPipelineConfig {
  dopplerTarget: DopplerTargetId;
  desired2dPhase: 'ed' | 'mid-systole' | 'early-diastole';
  expectedModality: ImagingModality;
  expectedKind: 'spectral' | 'm-mode';
  needsDiastole: boolean;
  stripAction:
    | { kind: 'vti'; id: 'lvot-vti' | 'av-vti' }
    | { kind: 'strip'; target: AutoStripMeasureTarget; tdiSite?: 'septal' | 'lateral' };
  placingLabel: string;
  capturingLabel: string;
}

const STRIP_PIPELINES: Partial<Record<AutoMeasureActionId, StripPipelineConfig>> = {
  'lvot-vti': {
    dopplerTarget: 'lvot',
    desired2dPhase: 'mid-systole',
    expectedModality: 'pw',
    expectedKind: 'spectral',
    needsDiastole: false,
    stripAction: { kind: 'vti', id: 'lvot-vti' },
    placingLabel: 'Alineando gate PW en TSVI…',
    capturingLabel: 'Capturando espectro PW (1 latido)…',
  },
  'av-vti': {
    dopplerTarget: 'av-cw',
    desired2dPhase: 'mid-systole',
    expectedModality: 'cw',
    expectedKind: 'spectral',
    needsDiastole: false,
    stripAction: { kind: 'vti', id: 'av-vti' },
    placingLabel: 'Alineando CW sobre válvula aórtica…',
    capturingLabel: 'Capturando espectro CW (1 latido)…',
  },
  'mitral-ea': {
    dopplerTarget: 'mitral-inflow',
    desired2dPhase: 'early-diastole',
    expectedModality: 'pw',
    expectedKind: 'spectral',
    needsDiastole: true,
    stripAction: { kind: 'strip', target: 'mitral-inflow' },
    placingLabel: 'Colocando gate PW en puntas mitrales…',
    capturingLabel: 'Capturando llenado mitral PW (1 latido)…',
  },
  'tdi-septal': {
    dopplerTarget: 'tdi-septal',
    desired2dPhase: 'early-diastole',
    expectedModality: 'tdi',
    expectedKind: 'spectral',
    needsDiastole: true,
    stripAction: { kind: 'strip', target: 'tdi-e-prime', tdiSite: 'septal' },
    placingLabel: 'Colocando gate TDI en anillo septal…',
    capturingLabel: 'Capturando TDI septal (1 latido)…',
  },
  'tdi-lateral': {
    dopplerTarget: 'tdi-lateral',
    desired2dPhase: 'early-diastole',
    expectedModality: 'tdi',
    expectedKind: 'spectral',
    needsDiastole: true,
    stripAction: { kind: 'strip', target: 'tdi-e-prime', tdiSite: 'lateral' },
    placingLabel: 'Colocando gate TDI en anillo lateral…',
    capturingLabel: 'Capturando TDI lateral (1 latido)…',
  },
  tapse: {
    dopplerTarget: 'tapse',
    desired2dPhase: 'ed',
    expectedModality: 'm-mode',
    expectedKind: 'm-mode',
    needsDiastole: true,
    stripAction: { kind: 'strip', target: 'tapse' },
    placingLabel: 'Alineando Modo M en anillo tricuspídeo lateral…',
    capturingLabel: 'Capturando excursión Modo M (1 latido)…',
  },
  'rv-s-prime': {
    dopplerTarget: 'rv-s-prime',
    desired2dPhase: 'mid-systole',
    expectedModality: 'tdi',
    expectedKind: 'spectral',
    needsDiastole: false,
    stripAction: { kind: 'strip', target: 'rv-s-prime' },
    placingLabel: 'Colocando gate TDI en anillo tricuspídeo…',
    capturingLabel: 'Capturando onda S′ VD (1 latido)…',
  },
  'tr-vmax': {
    dopplerTarget: 'tr-cw',
    desired2dPhase: 'ed',
    expectedModality: 'cw',
    expectedKind: 'spectral',
    needsDiastole: false,
    stripAction: { kind: 'strip', target: 'tr-vmax' },
    placingLabel: 'Alineando CW sobre válvula tricúspide…',
    capturingLabel: 'Capturando chorro IT CW (1 latido)…',
  },
  'rvot-at': {
    dopplerTarget: 'rvot-pw',
    desired2dPhase: 'ed',
    expectedModality: 'pw',
    expectedKind: 'spectral',
    needsDiastole: false,
    stripAction: { kind: 'strip', target: 'rvot-at' },
    placingLabel: 'Colocando gate PW en TSVD / válvula pulmonar…',
    capturingLabel: 'Capturando flujo pulmonar PW (1 latido)…',
  },
};

function formatStripSuccess(
  cfg: StripPipelineConfig,
  res: Awaited<ReturnType<typeof runAutoSpectralVti>> | Awaited<ReturnType<typeof runAutoStripMeasure>>,
): string {
  if ('vtiCm' in res && !('target' in res)) {
    return cfg.stripAction.kind === 'vti' && cfg.stripAction.id === 'av-vti'
      ? `VAo: VTI ${res.vtiCm.toFixed(1)} cm · Vmáx ${res.vmaxMps.toFixed(2)} m/s · ΔPm ${res.meanGradientMmHg.toFixed(0)} mmHg.`
      : `VTI TSVI ${res.vtiCm.toFixed(1)} cm (Vmáx ${Math.round(res.vmaxMps * 100)} cm/s).`;
  }
  if (!('target' in res) || !res.ok) return 'Medición completada.';
  switch (res.target) {
    case 'mitral-inflow': {
      const parts = [`E ${Math.abs(res.e.velocityMps).toFixed(2)} m/s`];
      if (res.a) parts.push(`A ${Math.abs(res.a.velocityMps).toFixed(2)} m/s`);
      if (res.dt) parts.push(`TD ${Math.round(res.dt.dtMs)} ms`);
      return `Llenado mitral: ${parts.join(' · ')}.`;
    }
    case 'tdi-e-prime': {
      const site =
        cfg.stripAction.kind === 'strip' && cfg.stripAction.tdiSite === 'lateral'
          ? 'lateral'
          : 'septal';
      return `e′ ${site}: ${(Math.abs(res.ePrime.velocityMps) * 100).toFixed(1)} cm/s.`;
    }
    case 'rv-s-prime':
      return `S′ VD: ${(Math.abs(res.sPrime.velocityMps) * 100).toFixed(1)} cm/s.`;
    case 'rvot-at':
      return `TAc pulmonar: ${Math.round(res.atMs)} ms${res.atMs < 105 ? ' (<105 ms sugiere HTP)' : ''}.`;
    case 'tr-vmax':
      return `Vmáx IT: ${Math.abs(res.vmax.velocityMps).toFixed(2)} m/s (ΔP ${res.vmax.peakGradientMmHg.toFixed(0)} mmHg).`;
    case 'tapse':
      return `TAPSE: ${res.tapseCm.toFixed(2)} cm.`;
  }
}

/**
 * Execute ANY clinical auto-measurement in 1 click:
 * - Cleanly deactivates conflicting tools, modes, and overlays.
 * - In Practice / Guided mode (`presetsEnabled`): switches posture, moves smoothly to the canonical
 *   window, activates the exact Doppler/M-mode modality when required, waits event-driven for 1 full
 *   cardiac cycle, freezes, and runs the pure-signal measurement engine.
 * - In Exam mode (`!presetsEnabled`): NEVER moves the probe, NEVER changes view/posture/modality —
 *   measures strictly over whatever the student manually acquired.
 */
export async function runGuidedAutoMeasure(
  actionId: AutoMeasureActionId,
  onProgress?: (p: OrchestratorProgress | null) => void,
  deps: OrchestratorDeps = DEFAULT_DEPS,
): Promise<OrchestratorResult> {
  const st0 = useSimStore.getState();
  if (examClosed(st0)) {
    return { ok: false, message: 'El examen ha finalizado.' };
  }

  cancelGuidedAutoMeasure();
  const token = activeToken;
  const emit = (phaseLabel: string) => {
    if (!isCancelled(token)) onProgress?.({ actionId, phaseLabel });
  };

  prepareCleanAcquisition();
  const spec = AUTO_MEASURE_SPECS[actionId];
  const presetsEnabled = modePolicy(st0.mode).presetsEnabled;

  try {
    // Exam mode: strictly measure on the student's manually acquired frame/strip — zero navigation or auto-gating.
    if (!presetsEnabled) {
      emit('Analizando imagen actual…');
      if (actionId === 'plax-all') {
        const res = await deps.runAutoPlax();
        return res.ok
          ? { ok: true, message: `PLAX: ${res.calipers.length} dimensiones medidas.` }
          : { ok: false, message: res.message };
      }
      if (actionId === 'lv-a4c' || actionId === 'lv-a2c') {
        const res = await deps.runAutoChamberVolume('lv');
        return res.ok
          ? { ok: true, message: `VI trazado (FEVI ${res.efPct.toFixed(0)} %).` }
          : { ok: false, message: res.message };
      }
      if (actionId === 'la-a4c' || actionId === 'la-a2c') {
        const res = await deps.runAutoLaVolume();
        return res.ok
          ? { ok: true, message: `Volumen AI: ${res.volumeMl.toFixed(0)} mL.` }
          : { ok: false, message: res.message };
      }
      if (actionId === 'rv-2d') {
        const res = await deps.runAutoChamberVolume('rv');
        return res.ok
          ? { ok: true, message: `VD trazado (FEVD ${res.efPct.toFixed(0)} %, CAF ${res.facPct.toFixed(0)} %).` }
          : { ok: false, message: res.message };
      }
      if (actionId === 'ivc-full') {
        const res = await deps.runAutoIvc();
        return res.ok
          ? { ok: true, message: `VCI medida: ${res.valueCm.toFixed(2)} cm.` }
          : { ok: false, message: res.message };
      }
      const pipe = STRIP_PIPELINES[actionId]!;
      if (pipe.stripAction.kind === 'vti') {
        const res = await deps.runAutoSpectralVti(pipe.stripAction.id);
        return res.ok
          ? { ok: true, message: formatStripSuccess(pipe, res) }
          : { ok: false, message: res.message };
      }
      const res = await deps.runAutoStripMeasure(
        pipe.stripAction.target,
        pipe.stripAction.tdiSite ? { tdiSite: pipe.stripAction.tdiSite } : undefined,
      );
      return res.ok
        ? { ok: true, message: formatStripSuccess(pipe, res) }
        : { ok: false, message: res.message };
    }

    // ---- Practice / Guided 1-click pipelines ----

    // 1. Subcostal IVC full expiratory + inspiratory sniff collapse workflow
    if (actionId === 'ivc-full') {
      const readyExp = await ensureViewReady(
        'subcostal-ivc',
        'subcostal-supine',
        'expiration',
        { needsFullBeat: false, desiredPhase: 'any' },
        token,
        emit,
      );
      if (isCancelled(token)) return { ok: false, cancelled: true, message: 'Medición cancelada.' };
      if (!readyExp) {
        return { ok: false, message: 'No se pudo estabilizar la vista subcostal VCI en espiración.' };
      }

      emit('Midiendo VCI en espiración…');
      const expRes = await deps.runAutoIvc();
      if (isCancelled(token)) return { ok: false, cancelled: true, message: 'Medición cancelada.' };
      if (!expRes.ok) return { ok: false, message: expRes.message };

      // Switch live to inspiratory sniff so IVC collapses, wait a few frames, and measure again
      emit('Simulando olfateo (inspiración)…');
      const cur = useSimStore.getState();
      if (cur.frozen) cur.toggleFreeze();
      useSimStore.getState().setPatient({ position: 'subcostal-supine', respiration: 'inspiration' });
      const inspSettled = await waitNextHudFrames(6, token, 2200);
      if (isCancelled(token)) return { ok: false, cancelled: true, message: 'Medición cancelada.' };
      if (!inspSettled) {
        return {
          ok: true,
          message: `VCI espiración: ${expRes.valueCm.toFixed(2)} cm (no se completó inspiración).`,
        };
      }

      emit('Midiendo colapso inspiratorio…');
      const inspRes = await deps.runAutoIvc();
      if (isCancelled(token)) return { ok: false, cancelled: true, message: 'Medición cancelada.' };
      if (!inspRes.ok) {
        return {
          ok: true,
          message: `VCI espiración: ${expRes.valueCm.toFixed(2)} cm (${inspRes.message})`,
        };
      }

      const collapsePct =
        expRes.valueCm > 0 ? Math.max(0, (1 - inspRes.valueCm / expRes.valueCm) * 100) : 0;
      return {
        ok: true,
        message: `VCI: ${expRes.valueCm.toFixed(2)} → ${inspRes.valueCm.toFixed(2)} cm (colapso ${collapsePct.toFixed(0)} %).`,
      };
    }

    // 2. Pure-2D full-beat pipelines (PLAX, LV Simpson, LA Volume, RV 2D + basal)
    if (
      actionId === 'plax-all' ||
      actionId === 'lv-a4c' ||
      actionId === 'lv-a2c' ||
      actionId === 'la-a4c' ||
      actionId === 'la-a2c' ||
      actionId === 'rv-2d'
    ) {
      const ready = await ensureViewReady(
        spec.targetViewId,
        spec.posture,
        'expiration',
        { needsFullBeat: true },
        token,
        emit,
      );
      if (isCancelled(token)) return { ok: false, cancelled: true, message: 'Medición cancelada.' };
      if (!ready) {
        return {
          ok: false,
          message: `No se pudo estabilizar ${viewLabel(spec.targetViewId)} en 2D.`,
        };
      }

      emit(`Midiendo ${spec.shortLabel}…`);
      if (actionId === 'plax-all') {
        const res = await deps.runAutoPlax();
        if (isCancelled(token)) return { ok: false, cancelled: true, message: 'Medición cancelada.' };
        if (!res.ok) return { ok: false, message: res.message };
        const byId = new Map(res.calipers.map((c) => [c.measurementId, c.valueCm]));
        const parts: string[] = [];
        if (byId.has('lv-edd')) parts.push(`DTDVI ${byId.get('lv-edd')!.toFixed(1)}`);
        if (byId.has('lv-esd')) parts.push(`DTSVI ${byId.get('lv-esd')!.toFixed(1)}`);
        if (byId.has('lvot-diameter')) parts.push(`TSVI ${byId.get('lvot-diameter')!.toFixed(2)} cm`);
        return {
          ok: true,
          message: `PLAX: ${res.calipers.length} medidas (${parts.join(' · ')}).`,
        };
      }
      if (actionId === 'lv-a4c' || actionId === 'lv-a2c') {
        const res = await deps.runAutoChamberVolume('lv');
        if (isCancelled(token)) return { ok: false, cancelled: true, message: 'Medición cancelada.' };
        if (!res.ok) return { ok: false, message: res.message };
        return {
          ok: true,
          message: `VI ${actionId === 'lv-a4c' ? 'A4C' : 'A2C'}: FEVI ${res.efPct.toFixed(0)} % (VTD ${res.ed.volumeMl.toFixed(0)} / VTS ${res.es.volumeMl.toFixed(0)} mL).`,
        };
      }
      if (actionId === 'la-a4c' || actionId === 'la-a2c') {
        const res = await deps.runAutoLaVolume();
        if (isCancelled(token)) return { ok: false, cancelled: true, message: 'Medición cancelada.' };
        if (!res.ok) return { ok: false, message: res.message };
        const depthWarn = res.cutByDepth ? ' ⚠ Sube Profundidad para incluir todo el techo de la AI.' : '';
        return {
          ok: true,
          message: `AI ${actionId === 'la-a4c' ? 'A4C' : 'A2C'}: ${res.volumeMl.toFixed(0)} mL.${depthWarn}`,
        };
      }
      const res = await deps.runAutoChamberVolume('rv');
      if (isCancelled(token)) return { ok: false, cancelled: true, message: 'Medición cancelada.' };
      if (!res.ok) return { ok: false, message: res.message };
      return {
        ok: true,
        message: `VD: FEVD ${res.efPct.toFixed(0)} % · CAF ${res.facPct.toFixed(0)} %${res.rvBasal ? ` · Basal ${res.rvBasal.valueCm.toFixed(1)} cm` : ''}.`,
      };
    }

    // 3. Doppler & M-mode 1-click pipelines: 2D target view -> auto gate/cursor + modality -> 1 full beat -> auto measure
    const pipe = STRIP_PIPELINES[actionId]!;
    const ready2d = await ensureViewReady(
      spec.targetViewId,
      spec.posture,
      'expiration',
      { needsFullBeat: false, desiredPhase: pipe.desired2dPhase },
      token,
      emit,
    );
    if (isCancelled(token)) return { ok: false, cancelled: true, message: 'Medición cancelada.' };
    if (!ready2d) {
      return {
        ok: false,
        message: `No se pudo estabilizar ${viewLabel(spec.targetViewId)} para colocar el cursor.`,
      };
    }

    emit(pipe.placingLabel);
    const gateRes =
      pipe.dopplerTarget === 'lvot'
        ? await deps.runAutoLvotGate()
        : await deps.runAutoDopplerTarget(pipe.dopplerTarget);
    if (isCancelled(token)) return { ok: false, cancelled: true, message: 'Medición cancelada.' };
    if (!gateRes.ok) return { ok: false, message: gateRes.message };

    emit(pipe.capturingLabel);
    const beatReady = await waitForCompleteStripBeat(
      pipe.expectedModality,
      pipe.expectedKind,
      pipe.needsDiastole,
      token,
    );
    if (isCancelled(token)) return { ok: false, cancelled: true, message: 'Medición cancelada.' };
    if (!beatReady) {
      return { ok: false, message: 'No se completó un latido en la tira. Intenta de nuevo.' };
    }

    emit(`Midiendo ${spec.shortLabel}…`);
    if (pipe.stripAction.kind === 'vti') {
      const mRes = await deps.runAutoSpectralVti(pipe.stripAction.id);
      if (isCancelled(token)) return { ok: false, cancelled: true, message: 'Medición cancelada.' };
      if (!mRes.ok) return { ok: false, message: mRes.message };
      return { ok: true, message: formatStripSuccess(pipe, mRes) };
    }
    const mRes = await deps.runAutoStripMeasure(
      pipe.stripAction.target,
      pipe.stripAction.tdiSite ? { tdiSite: pipe.stripAction.tdiSite } : undefined,
    );
    if (isCancelled(token)) return { ok: false, cancelled: true, message: 'Medición cancelada.' };
    if (!mRes.ok) return { ok: false, message: mRes.message };
    return { ok: true, message: formatStripSuccess(pipe, mRes) };
  } catch (e: unknown) {
    return {
      ok: false,
      message: e instanceof Error ? e.message : 'Error durante la automedición.',
    };
  } finally {
    if (!isCancelled(token)) onProgress?.(null);
  }
}
