import { loadCaseById } from '@/cases';
import { getMeasurementSpec } from '@/simulator/measurements/protocol';
import {
  addEvent,
  completeTask,
  emptyProgress,
  loadProgress,
  saveProgress,
} from '@/education/progress';
import { expectedFindings, getFinding, scoreImpression } from '@/education/impression';
import { examSummaryOf, impressionCorrected } from '../examSummary';
import { examClosed, modePolicy, screenInMode } from '../modePolicy';
import { OPENING_VIEW_LOST_BELOW, appPlacedPose, drawExamCase, freshSession } from './helpers';
import type { EducationSlice, SimSliceCreator } from './types';

export const createEducationSlice: SimSliceCreator<EducationSlice> = (set, get) => ({
  caseId: 'normal-excellent-window',
  sessionEpoch: 0,
  mode: 'sandbox',
  targetViewId: null,
  truth: null,
  measurements: [],
  activeTool: 'none',
  activeMeasurementId: null,
  progress: loadProgress(typeof localStorage !== 'undefined' ? localStorage : null),
  impressionSelection: [],
  impressionChecked: false,
  viewProgress: {},
  handViewProgress: {},
  presetViews: [],
  openingView: null,
  examFinished: false,
  examResult: null,

  setMode: (m, examCaseId) =>
    set((s) => {
      const policy = modePolicy(m);
      // the exam draws its case: chosen by the learner, the case whose name the exam hides was known beforehand
      let caseId = s.caseId;
      let progress = s.progress;
      if (policy.assignsCase) {
        caseId = loadCaseById(examCaseId ?? drawExamCase()).id;
        progress = addEvent(progress, { t: Date.now(), kind: 'exam-start', caseId });
        saveProgress(typeof localStorage !== 'undefined' ? localStorage : null, progress);
      }
      // an exam starts from scratch, as a case load does: the pose, the impression, the preset and the artifact lab left
      // behind by the practice would otherwise be scored as the learner's (A4C scored 97 the instant the exam began)
      return {
        mode: m,
        caseId,
        // another case's truth waits for its own core's `ready`: until then the previous case's would be read as its
        ...(caseId !== s.caseId ? { truth: null } : {}),
        progress,
        examFinished: false,
        examResult: null,
        ...(m === 'exam' ? freshSession() : {}),
        ui: {
          ...s.ui,
          screen: screenInMode(s.ui.screen, m),
          showHints: policy.hintsEnabled,
          showPhysics: policy.devToolsAllowed ? s.ui.showPhysics : false,
          devPanel: policy.devToolsAllowed ? s.ui.devPanel : false,
          reviewMode: policy.devToolsAllowed ? s.ui.reviewMode : false,
        },
      };
    }),
  setTargetView: (id) => set({ targetViewId: id }),
  openTask: (task) => {
    const st = get();
    if (task.caseId && task.caseId !== st.caseId) st.loadCase(task.caseId);
    if (get().mode !== 'guided') get().setMode('guided');
    set((s) => ({
      targetViewId: task.viewId ?? null,
      ui: { ...s.ui, screen: 'simulator', guidanceOpen: Boolean(task.viewId) || s.ui.guidanceOpen },
    }));
  },
  setTruth: (t) => set({ truth: t }),
  caseReady: (r) => {
    // the store's case is the learner's choice: a core built for a case picked and then left while it loaded is not the
    // one on screen, and its truth, phase marks and opening view are not this case's
    if (r.caseId !== get().caseId) return;
    get().setTruth(r.truth);
    get().setCycleInfo(r.phaseMarks, r.lvLengthCm);
    get().setWorkerMode(r.workerMode);
    // free practice opens on the canonical PLAX once the case's models exist (decision 343)
    get().openOnStartView();
  },
  addMeasurement: (m) =>
    set((s) => {
      // the measurements of a finished exam are the ones it was scored on
      if (examClosed(s)) return s;
      const progress = addEvent(s.progress, {
        t: Date.now(),
        kind: 'measurement',
        caseId: s.caseId,
        measurementId: m.measurementId ?? m.label,
        techniqueScore: m.technique?.score ?? null,
        value: m.value,
      });
      saveProgress(typeof localStorage !== 'undefined' ? localStorage : null, progress);
      return { measurements: [...s.measurements, m], progress };
    }),
  updateMeasurement: (id, patch) =>
    set((s) => {
      if (examClosed(s)) return s;
      let changed = false;
      const measurements = s.measurements.map((m) => {
        if (m.id !== id) return m;
        changed = true;
        return { ...m, ...patch };
      });
      return changed ? { measurements } : s;
    }),
  removeMeasurement: (id) =>
    set((s) =>
      examClosed(s) ? s : { measurements: s.measurements.filter((m) => m.id !== id) },
    ),
  clearMeasurements: () => set((s) => (examClosed(s) ? s : { measurements: [] })),
  setActiveTool: (t) =>
    set((s) =>
      t !== 'none' && examClosed(s)
        ? s
        : {
            activeTool: t,
            activeMeasurementId: null,
            ...(t !== 'none' ? { cinePlaying: false } : {}),
          },
    ),
  setActiveMeasurement: (id) => {
    if (!id) return set({ activeMeasurementId: null, activeTool: 'none' });
    const spec = getMeasurementSpec(id);
    if (!spec || examClosed(get())) return set({ activeMeasurementId: null, activeTool: 'none' });
    set({ activeMeasurementId: id, activeTool: spec.tool, cinePlaying: false });
  },
  completeTasks: (ids) =>
    set((s) => {
      let progress = s.progress;
      const now = Date.now();
      for (const id of ids) progress = completeTask(progress, id, now);
      if (progress === s.progress) return s;
      saveProgress(typeof localStorage !== 'undefined' ? localStorage : null, progress);
      return { progress };
    }),
  resetLearningProgress: () => {
    const progress = emptyProgress();
    saveProgress(typeof localStorage !== 'undefined' ? localStorage : null, progress);
    set({ progress });
  },
  toggleFinding: (id) =>
    set((s) => {
      // the answer sheet closes when the exam is finished (decision 236) and, in practice, once corrected (decision 257)
      if (impressionCorrected(s)) return s;
      const f = getFinding(id);
      let sel = s.impressionSelection.filter((x) => x !== id);
      if (!s.impressionSelection.includes(id)) {
        // exclusive groups: selecting one deselects the others of the group
        if (f?.exclusive) sel = sel.filter((x) => getFinding(x)?.exclusive !== f.exclusive);
        sel.push(id);
      }
      return { impressionSelection: sel };
    }),
  checkImpression: () =>
    set((s) => {
      if (s.mode === 'exam' || !s.impressionSelection.length || s.impressionChecked) return s;
      // every corrected impression is recorded; the curriculum counts the first one on each case (firstImpressionScore)
      let progress = s.progress;
      if (s.truth) {
        progress = addEvent(progress, {
          t: Date.now(),
          kind: 'impression',
          caseId: s.caseId,
          score: scoreImpression(s.impressionSelection, expectedFindings(s.truth)).score,
        });
        saveProgress(typeof localStorage !== 'undefined' ? localStorage : null, progress);
      }
      return { impressionChecked: true, progress };
    }),
  redoImpression: () =>
    set((s) => (s.mode === 'exam' ? s : { impressionSelection: [], impressionChecked: false })),
  recordViewScore: (viewId, score, pose) =>
    set((s) => {
      // the same state when nothing changes: a new one, even empty, notifies every subscriber (decision 173)
      const overall = (s.viewProgress[viewId] ?? 0) < score;
      // not acquired by hand: a view reached with its preset (decision 174), the view free practice opened on until the
      // learner has lost it (`openingView`), and any view analysed at a pose the app placed the probe at
      const o = s.openingView;
      const handBefore = s.handViewProgress[viewId] ?? 0;
      const byHand =
        !s.presetViews.includes(viewId) &&
        !(o && !o.released && o.viewId === viewId) &&
        !(pose && appPlacedPose(pose, o)) &&
        handBefore < score;
      if (!overall && !byHand) return s;
      // the learner's record keeps only what they reached by hand, and only the first score and each rise of 5 points
      // or more over the best so far, so that a preset or a score that wavers does not fill it (decision 236)
      const record = byHand && (score >= handBefore + 5 || handBefore === 0);
      const progress = record
        ? addEvent(s.progress, { t: Date.now(), kind: 'view', caseId: s.caseId, viewId, score })
        : s.progress;
      if (record) saveProgress(typeof localStorage !== 'undefined' ? localStorage : null, progress);
      return {
        viewProgress: overall ? { ...s.viewProgress, [viewId]: score } : s.viewProgress,
        handViewProgress: byHand ? { ...s.handViewProgress, [viewId]: score } : s.handViewProgress,
        progress,
      };
    }),
  observeView: (viewId, score, pose) =>
    set((s) => {
      const o = s.openingView;
      // the opening pose and the start pose are the app's: nothing the learner did shows there
      if (!o || o.released || !pose || appPlacedPose(pose, o)) return s;
      if (viewId === o.viewId && score >= OPENING_VIEW_LOST_BELOW) return s;
      // lost away from where the app put the probe: whatever the learner finds from here on is theirs
      return { openingView: { ...o, released: true } };
    }),
  finishExam: () =>
    set((s) => {
      let progress = s.progress;
      // the summary the report shows, not a second computation of it (decision 236)
      const summary = examSummaryOf(s);
      if (summary) {
        progress = addEvent(progress, {
          t: Date.now(),
          kind: 'exam',
          caseId: s.caseId,
          total: summary.total,
          acquisition: summary.acquisition.total,
          measurements: summary.measurements.total,
          impression: summary.impression,
        });
        saveProgress(typeof localStorage !== 'undefined' ? localStorage : null, progress);
      }
      // closed from here on (`examClosed`): the image stays frozen and no tool stays armed over it
      return {
        examFinished: true,
        examResult: summary,
        frozen: true,
        activeTool: 'none',
        activeMeasurementId: null,
        progress,
        ui: { ...s.ui, screen: 'report' },
      };
    }),
  resetProgress: () =>
    set({
      viewProgress: {},
      handViewProgress: {},
      presetViews: [],
      examFinished: false,
      examResult: null,
      measurements: [],
    }),
  loadCase: (id) =>
    set((s) => {
      // the case the simulator draws: an unknown id loads the normal case (`loadCaseById`), and the store says so
      const caseId = loadCaseById(id).id;
      const progress = addEvent(s.progress, { t: Date.now(), kind: 'case', caseId });
      saveProgress(typeof localStorage !== 'undefined' ? localStorage : null, progress);
      return {
        caseId,
        ...freshSession(),
        progress,
        // another case's truth waits for its own core's `ready`
        ...(caseId !== s.caseId ? { truth: null } : {}),
      };
    }),
});
