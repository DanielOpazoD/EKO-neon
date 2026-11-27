import type { ProbeControl } from '@/simulator/probe/pose';
import type { FrameAcquisition } from '@/simulator/core/protocol';
import type { AcquisitionSettings } from '@/simulator/renderer/types';
import { listCases } from '@/cases';
import { modePolicy } from '../modePolicy';
import { parsePrefs, pickPrefs } from '../prefs';
import type { NavigatorLayers, SimStore, UiPrefs } from './types';

const PREF_KEY = 'echotwin.prefs.v1';

/** The preset pose request in flight: a later request, a cancellation or a fresh session supersedes it. */
let presetToken = 0;

export function nextPresetToken(): number {
  return ++presetToken;
}

export function currentPresetToken(): number {
  return presetToken;
}

export function cancelPendingPresets(): void {
  presetToken++;
}

/** The view free practice opens on (decision 343). */
export const START_VIEW_ID = 'plax';

/**
 * A deliberately imperfect starting pose near the parasternal window (never a canonical view): where guided practice and
 * the exam start, so the learner acquires the view (decision 235). Free practice moves on to START_VIEW_ID (decision 343).
 */
export const START_PROBE: ProbeControl = {
  u: 3.4,
  v: 0.4,
  rotationDeg: 25,
  tiltDeg: 6,
  rockDeg: -4,
  pressure: 0.55,
};

/** Whether a probe is still at START_PROBE, untouched. */
export function atStartProbe(p: ProbeControl): boolean {
  return samePose(p, START_PROBE);
}

/** Whether two probe controls are the same pose, value for value. */
export function samePose(a: ProbeControl, b: ProbeControl): boolean {
  return (Object.keys(START_PROBE) as (keyof ProbeControl)[]).every((k) => a[k] === b[k]);
}

/**
 * Below this score the view free practice opened on counts as lost: the red band of the view guide, and the score under
 * which a measurement's view is not valid (scoring.ts). A view the learner has lost and found again is theirs.
 */
export const OPENING_VIEW_LOST_BELOW = 50;

/**
 * Whether a probe pose is one the app placed, not one the learner reached: the start pose, or the pose of the jump free
 * practice opens with (`openingView`). A view analysed at either adds nothing to what the learner acquired by hand,
 * however the probe got back there, and whatever frame carries the analysis.
 */
export function appPlacedPose(pose: ProbeControl, opening: { pose: ProbeControl } | null): boolean {
  return atStartProbe(pose) || (opening !== null && samePose(pose, opening.pose));
}

/**
 * Whether a frame was acquired for this case in this session (`sessionEpoch`): only such a frame is scored. The frames
 * still in flight when a case loads or an exam starts carry the view of the session before (an A4C credited at 91 in an
 * exam just begun).
 */
export function frameOfSession(
  acq: FrameAcquisition | undefined,
  s: { caseId: string; sessionEpoch: number },
): boolean {
  return acq !== undefined && acq.caseId === s.caseId && acq.sessionEpoch === s.sessionEpoch;
}

export function loadPrefs(): Partial<UiPrefs> {
  try {
    const raw = localStorage.getItem(PREF_KEY);
    if (!raw) return {};
    // validated key by key (prefs.ts): a stale or edited entry cannot put arbitrary keys or types into `ui`. The view
    // guide is not persisted, so it always starts hidden (decision 188)
    return parsePrefs(JSON.parse(raw));
  } catch {
    return {};
  }
}

export function savePrefs(ui: UiPrefs): void {
  try {
    localStorage.setItem(PREF_KEY, JSON.stringify(pickPrefs(ui)));
  } catch {
    /* storage unavailable */
  }
}

/**
 * The LV segment layer (decision 152) is shown on the cut map and the 3D heart only where hints are allowed: in exam
 * mode it is off whatever the saved preference, since its toggle lives in a panel exam mode hides.
 */
export function segmentLayerOn(s: Pick<SimStore, 'mode' | 'ui'>): boolean {
  return s.ui.navSegments && modePolicy(s.mode).hintsEnabled;
}

/** The segment layer of the ultrasound image (decision 153), off in exam mode like the others. */
export function imageSegmentsOn(s: Pick<SimStore, 'mode' | 'ui'>): boolean {
  return s.ui.imageSegments && modePolicy(s.mode).hintsEnabled;
}

/** On-image 2D anatomical structure labels: teaching overlay strictly disabled in exam mode. */
export function imageAnatomyLabelsOn(s: Pick<SimStore, 'mode' | 'ui'>): boolean {
  return s.ui.imageAnatomyLabels && modePolicy(s.mode).hintsEnabled;
}

/** Projector / classroom presentation mode: clean maximized scanner plus dimmed dark-room chrome. */
export function isClassModeOn(ui: Pick<UiPrefs, 'minimal' | 'roomMode'>): boolean {
  return ui.minimal && ui.roomMode;
}

/** Toggle 1-click classroom presentation mode cleanly on or off. */
export function nextClassModeUi(
  ui: Pick<UiPrefs, 'minimal' | 'roomMode'>,
): Pick<UiPrefs, 'minimal' | 'roomMode'> {
  const next = !isClassModeOn(ui);
  return { minimal: next, roomMode: next };
}

export function navigatorLayers(s: Pick<SimStore, 'mode' | 'ui'>): NavigatorLayers {
  if (!modePolicy(s.mode).navigatorAnatomy)
    return {
      skin: true,
      skeleton: true,
      heart: false,
      chambers: false,
      valves: false,
      vessels: false,
      cut: false,
      split: false,
      windows: false,
      axes: false,
    };
  return {
    skin: s.ui.navSkin,
    skeleton: s.ui.showSkeleton,
    heart: s.ui.navHeart,
    chambers: s.ui.navChambers,
    valves: s.ui.navValves,
    vessels: s.ui.navVessels,
    cut: s.ui.navCut,
    split: s.ui.navSplit,
    windows: s.ui.navWindows,
    axes: s.ui.navAxes,
  };
}

/** A case drawn at random for an exam. */
export function drawExamCase(): string {
  const ids = listCases().map((c) => c.id);
  return ids[Math.floor(Math.random() * ids.length)] ?? ids[0]!;
}

/** The sessions started so far: each one's number travels with its frames (`SimInput.sessionEpoch`). */
let sessionCounter = 0;

/**
 * What a session starts from, when a case is loaded and when an exam begins (decision 235): one list, so that the two
 * cannot drift apart (the exam start left the artifact lab on and the case load the target view).
 */
export function freshSession() {
  cancelPendingPresets();
  return {
    sessionEpoch: ++sessionCounter,
    measurements: [],
    captures: [],
    frozen: false,
    cineOffset: 0,
    probe: { ...START_PROBE },
    viewProgress: {},
    handViewProgress: {},
    presetViews: [],
    openingView: null,
    examFinished: false,
    examResult: null,
    presetAnim: null,
    targetViewId: null,
    impressionSelection: [],
    impressionChecked: false,
    artifactLab: null,
    colorBoxCustom: false,
  };
}

export function clampProbe(p: ProbeControl): ProbeControl {
  return {
    u: Math.max(-14, Math.min(14, p.u)),
    v: Math.max(-12, Math.min(11, p.v)),
    rotationDeg: ((((p.rotationDeg + 180) % 360) + 360) % 360) - 180,
    tiltDeg: Math.max(-70, Math.min(70, p.tiltDeg)),
    rockDeg: Math.max(-60, Math.min(60, p.rockDeg)),
    pressure: Math.max(0, Math.min(1, p.pressure)),
  };
}

export function clampSettings(s: AcquisitionSettings): AcquisitionSettings {
  const depthCm = Math.max(6, Math.min(30, s.depthCm));
  return {
    ...s,
    depthCm,
    sectorDeg: Math.max(30, Math.min(100, s.sectorDeg)),
    gainDb: Math.max(-30, Math.min(30, s.gainDb)),
    dynamicRangeDb: Math.max(30, Math.min(90, s.dynamicRangeDb)),
    frequencyMHz: Math.max(1.5, Math.min(5, s.frequencyMHz)),
    focusCm: Math.max(2, Math.min(depthCm, s.focusCm)),
    persistence: Math.max(0, Math.min(0.9, s.persistence)),
    sri: s.sri === undefined ? undefined : Math.max(0, Math.min(1, s.sri)),
    spatialCompound:
      s.spatialCompound === undefined ? undefined : Math.max(0, Math.min(1, s.spatialCompound)),
    zoom: Math.max(1, Math.min(2.5, s.zoom)),
  };
}
