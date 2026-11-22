import type { StateCreator } from 'zustand';
import type { ProbeControl } from '@/simulator/probe/pose';
import type { PatientState } from '@/simulator/anatomy/thoraxModel';
import type { AcquisitionSettings, ImagingModality } from '@/simulator/renderer/types';
import type { ColorSettings } from '@/simulator/doppler/color/colorDoppler';
import type {
  SpectralModality,
  SpectralSettings,
  SpectralVelocitySetup,
} from '@/simulator/doppler/spectral/spectrum';
import type {
  AnatomyChoice,
  ImageFormationChoice,
  PhaseMarks,
  QualityChoice,
  RendererBackendChoice,
} from '@/simulator/core/protocol';
import type { StructuredEchoTruth } from '@/simulator/hemodynamics/groundTruth';
import type { Measurement } from '@/simulator/measurements/types';
import type { ReviewMarker, ReviewReport } from '../review';
import type { ProgressState } from '@/education/progress';
import type { ExamSummary } from '@/education/scoring/scoring';
import type { ProductMode } from '../modePolicy';

export type { ProductMode };

/** Right-console tabs (PR: tabbed console). Persisted so a reload restores the working context. */
export type ConsoleTab = 'adquirir' | 'imagen' | 'doppler' | 'medir' | 'lab' | 'revisar';

export interface UiPrefs {
  showTorso: boolean;
  showSkeleton: boolean;
  /** 3D navigator layers and aids (spec 4.3): each one can be turned off to read the anatomy underneath. */
  navSkin: boolean;
  navHeart: boolean;
  navChambers: boolean;
  navValves: boolean;
  navVessels: boolean;
  navAxes: boolean;
  navCut: boolean;
  /** Second navigator view under the torso: the heart cut by the imaging plane, seen face-on (decision 137). */
  navSplit: boolean;
  /** Chamber and valve names on the cut face. */
  navLabels: boolean;
  /** The cut map colours the LV myocardium by segment, numbered (decision 152). */
  navSegments: boolean;
  /** The ultrasound image shows the LV segments of the tissue in it, translucent and numbered (decision 153). */
  imageSegments: boolean;
  /** The LV segment panel (polar map and inspector) under the navigator. */
  segmentsOpen: boolean;
  /** Segment model shown: the anatomical AHA 17 or the 16-segment wall-motion model. */
  segmentModel: 'LV_AHA17' | 'LV_16';
  /** Segment selected on the cut map or the polar map, shared by both. Not persisted. */
  selectedSegment: number | null;
  /**
   * The view guide (score, hints, details, causes) under the navigator; hidden until asked for (decision 141) and
   * hidden again at every load: it is not persisted (decision 188).
   */
  guidanceOpen: boolean;
  /** The keyboard shortcuts sheet over the simulator («?» or the ⋯ menu, decision 185). Not persisted. */
  shortcutsOpen: boolean;
  /** Rings on the skin where each canonical view is acquired (decision 132). */
  navWindows: boolean;
  showHints: boolean;
  showPhysics: boolean;
  /** Developer mode (off by default): exposes engineering tools (Lab, Rodero anatomy, review mode, DevPanel). */
  devMode: boolean;
  devPanel: boolean;
  /** Review mode (decision 134): clicks on the image place numbered markers for a feedback report. Not persisted. */
  reviewMode: boolean;
  /** Freeze the image when the first marker is placed on a live frame, so the markers keep their frame (decision 135). */
  reviewFreezeOnMark: boolean;
  /** What a press on the image does in review mode (decision 340): a point that identifies, or the hand-drawn lasso. */
  reviewTool: 'point' | 'lasso';
  showEcg: boolean;
  tutorialDone: boolean;
  /** Machine-style telemetry overlay on the image corners (case, vitals, acquisition params). */
  showHud: boolean;
  /** Left navigation rail collapsed to the mini strip (score + expander). */
  railMini: boolean;
  /** Clean-interface mode: the whole left rail hides so only the image and console remain. */
  minimal: boolean;
  /** Room mode (decision 205): the chrome dims and only the image keeps its light, as in a dark echo room. */
  roomMode: boolean;
  consoleTab: ConsoleTab;
  screen: 'simulator' | 'references' | 'report' | 'curriculum' | 'progress';
}

export interface PresetAnimationState {
  from: ProbeControl;
  to: ProbeControl;
  startMs: number;
  durationMs: number;
  viewId: string;
}

export interface ArtifactLabOverrides {
  sideLobe: number;
  mirror: number;
  beamWidth: number;
  clutter: number;
}

export interface AcquisitionSlice {
  probe: ProbeControl;
  patient: PatientState;
  settings: AcquisitionSettings;
  modality: ImagingModality;
  frozen: boolean;
  cineOffset: number;
  /**
   * The frozen cine replays as a loop (decision 191). Any frame the learner picks by hand, a tool, review mode, the
   * freeze and the modality stop it, so a measurement or a marker never lands on a frame the loop chose (decision 197).
   */
  cinePlaying: boolean;
  /** Playback speed multiplier for frozen cine loops (0.25×, 0.5×, 1×). */
  cineSpeed: 0.25 | 0.5 | 1;
  color: ColorSettings;
  /** The spectral settings of the mode on screen; each spectral mode keeps its own velocity window (decision 230). */
  spectral: SpectralSettings;
  /** The velocity settings each spectral mode had when the learner left it, restored on return (decision 230). */
  spectralByMode: Partial<Record<SpectralModality, SpectralVelocitySetup>>;
  cursorThetaRad: number;
  gateDepthCm: number;
  quality: QualityChoice;
  rendererBackend: RendererBackendChoice;
  /** The heart the frames draw (decision 352): the analytic model or the experimental Rodero heart. */
  anatomy: AnatomyChoice;
  /** How the frames form the image (decision 354): the lattice phasor or the experimental convolution. */
  imageFormation: ImageFormationChoice;
  /** Artifact laboratory overrides (null = case defaults). */
  artifactLab: ArtifactLabOverrides | null;
  /** Preset view in progress: the probe is moved continuously to the canonical pose (never teleported). */
  presetAnim: PresetAnimationState | null;
  setProbe: (p: Partial<ProbeControl>) => void;
  nudgeProbe: (p: Partial<ProbeControl>) => void;
  setPatient: (p: Partial<PatientState>) => void;
  setSettings: (s: Partial<AcquisitionSettings>) => void;
  setTgc: (band: number, db: number) => void;
  resetTgc: () => void;
  autoOptimize: () => void;
  setModality: (m: ImagingModality) => void;
  toggleFreeze: () => void;
  setCineOffset: (o: number) => void;
  /** One step of the cine loop: the next frame, the newest wrapping to the oldest; it keeps the loop playing. */
  stepCine: () => void;
  setCinePlaying: (p: boolean) => void;
  setCineSpeed: (sp: 0.25 | 0.5 | 1) => void;
  jumpCinePhase: (target: 'ed' | 'es') => void;
  setColor: (c: Partial<ColorSettings>) => void;
  setSpectral: (s: Partial<SpectralSettings>) => void;
  setCursor: (theta: number, depth?: number) => void;
  setQuality: (q: QualityChoice) => void;
  setBackend: (b: RendererBackendChoice) => void;
  setAnatomy: (a: AnatomyChoice) => void;
  setImageFormation: (f: ImageFormationChoice) => void;
  setArtifactLab: (v: ArtifactLabOverrides | null) => void;
  /** Start moving the probe to a predefined view (disabled in exam mode). */
  startPresetView: (viewId: string, opts?: { instant?: boolean }) => void;
  /**
   * Free practice opens on the canonical parasternal long axis (decision 343): when a case has loaded in sandbox mode and
   * the probe is still at START_PROBE, it goes there at once, as the PLAX preset (it does not count as acquired by hand).
   * Guided practice and the exam keep the imperfect pose the learner starts from.
   */
  openOnStartView: () => void;
  /** Advance the preset animation; called from the animation loop. */
  tickPresetAnimation: (nowMs: number) => void;
  resetProbe: () => void;
}

export interface EducationSlice {
  caseId: string;
  mode: ProductMode;
  targetViewId: string | null;
  truth: StructuredEchoTruth | null;
  measurements: Measurement[];
  activeTool:
    | 'none'
    | 'caliper'
    | 'velocity'
    | 'vti'
    | 'auto-vti'
    | 'time'
    | 'slope'
    | 'simpson'
    | 'tapse';
  /** Semantic measurement being captured (protocol id) or null for a free measurement. */
  activeMeasurementId: string | null;
  /** Local learning progress (events + completed curriculum tasks), persisted in localStorage. */
  progress: ProgressState;
  /** Structured impression selected by the learner for the current case. */
  impressionSelection: string[];
  /** The learner pressed «Corregir» on the impression (practice; the exam corrects it when finished, decision 257). */
  impressionChecked: boolean;
  /** Best view-quality score reached per view id during this case (drives acquisition scoring). */
  viewProgress: Record<string, number>;
  /**
   * Best score of each view reached by moving the probe (decision 174): a view whose preset the learner used in this case
   * keeps its score in `viewProgress` but adds nothing here until the case is loaded again. The curriculum reads this.
   */
  handViewProgress: Record<string, number>;
  /** Views whose preset was used since the case was loaded. */
  presetViews: string[];
  examFinished: boolean;
  /**
   * The exam's summary as it was when the learner finished, the one recorded for the teacher and the one the report shows
   * (decision 236): recomputed from the live session, it drifted from the record as soon as anything changed after.
   */
  examResult: ExamSummary | null;
  completeTasks: (taskIds: string[]) => void;
  resetLearningProgress: () => void;
  toggleFinding: (id: string) => void;
  /** «Corregir»: shows which findings were right; the sheet stays closed until «Rehacer». */
  checkImpression: () => void;
  /** «Rehacer»: an empty sheet, not yet corrected. */
  redoImpression: () => void;
  /** `examCaseId` fixes the case an exam starts on; without it the exam draws one at random (`modePolicy.assignsCase`). */
  setMode: (m: ProductMode, examCaseId?: string) => void;
  setTargetView: (id: string | null) => void;
  /**
   * «Abrir tarea» of the curriculum: the task's case (loaded only when it is not the current one, so the session is kept)
   * and its view as the target of the guided mode, with the view guide open on the simulator.
   */
  openTask: (task: { caseId?: string; viewId?: string }) => void;
  setTruth: (t: StructuredEchoTruth | null, caseId: string) => void;
  addMeasurement: (m: Measurement) => void;
  removeMeasurement: (id: string) => void;
  clearMeasurements: () => void;
  setActiveTool: (t: EducationSlice['activeTool']) => void;
  /** Start capturing a protocol measurement: selects its tool and remembers the semantic id. */
  setActiveMeasurement: (id: string | null) => void;
  recordViewScore: (viewId: string, score: number) => void;
  finishExam: () => void;
  resetProgress: () => void;
  loadCase: (id: string) => void;
}

export interface ReviewSlice {
  /** Review mode markers and the free note of the report being prepared (decision 134). */
  reviewMarkers: ReviewMarker[];
  reviewNote: string;
  /** Marker highlighted on the image, the model and the panel; Delete removes it (decision 135). */
  reviewSelectedId: string | null;
  /** Removed markers as batches (a primary goes with its secondaries), newest last, restored by undo. */
  reviewUndo: ReviewMarker[][];
  /** Primary marker the next clicks add secondary points to, until Esc (decision 136). */
  reviewLinkParentId: string | null;
  /** Who wrote the report the current markers came from, when they were loaded from one. */
  reviewSource: { author: ReviewReport['author']; createdAt: string } | null;
  armReviewLink: (id: string | null) => void;
  selectReviewMarker: (id: string | null) => void;
  undoReviewRemove: () => void;
  addReviewMarker: (m: ReviewMarker) => void;
  updateReviewMarker: (id: string, patch: Partial<ReviewMarker>) => void;
  removeReviewMarker: (id: string) => void;
  clearReview: () => void;
  setReviewNote: (note: string) => void;
  /** Restore the state a report describes (case, patient, probe, console) and show its markers. */
  loadReviewReport: (r: ReviewReport) => void;
}

export interface UiSlice {
  ui: UiPrefs;
  /** Cardiac phase landmarks and LV length of the loaded case (from the simulator). */
  phaseMarks: PhaseMarks | null;
  lvLengthCm: number | null;
  error: string | null;
  workerMode: 'worker' | 'inline' | 'starting';
  /** The 3D navigator has its model (decision 203): the start screen ticks it off. Not persisted. */
  navigatorReady: boolean;
  fpsUi: number;
  setUi: (u: Partial<UiPrefs>) => void;
  setCycleInfo: (marks: PhaseMarks, lvLengthCm: number) => void;
  setError: (e: string | null) => void;
  setWorkerMode: (m: UiSlice['workerMode']) => void;
  setNavigatorReady: (ready: boolean) => void;
  setFpsUi: (f: number) => void;
}

export type SimStore = AcquisitionSlice & EducationSlice & ReviewSlice & UiSlice;

export type SimSliceCreator<T> = StateCreator<SimStore, [], [], T>;

/** The layers the 3D navigator draws: the saved ones, or in exam mode only the skin and the ribs (decision 256). */
export interface NavigatorLayers {
  skin: boolean;
  skeleton: boolean;
  heart: boolean;
  chambers: boolean;
  valves: boolean;
  vessels: boolean;
  cut: boolean;
  split: boolean;
  windows: boolean;
  axes: boolean;
}
