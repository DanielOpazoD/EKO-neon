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
import { modePolicy } from '../modePolicy';
import { drawExamCase, freshSession } from './helpers';
import type { EducationSlice, SimSliceCreator } from './types';

export const createEducationSlice: SimSliceCreator<EducationSlice> = (set, get) => ({
  caseId: 'normal-excellent-window',
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
  examFinished: false,
  examResult: null,

  setMode: (m, examCaseId) =>
    set((s) => {
      const policy = modePolicy(m);
      // the exam draws its case: chosen by the learner, the case whose name the exam hides was known beforehand
      let caseId = s.caseId;
      let progress = s.progress;
      if (policy.assignsCase) {
        caseId = examCaseId ?? drawExamCase();
        progress = addEvent(progress, { t: Date.now(), kind: 'exam-start', caseId });
        saveProgress(typeof localStorage !== 'undefined' ? localStorage : null, progress);
      }
      // an exam starts from scratch, as a case load does: the pose, the impression, the preset and the artifact lab left
      // behind by the practice would otherwise be scored as the learner's (A4C scored 97 the instant the exam began)
      return {
        mode: m,
        caseId,
        progress,
        examFinished: false,
        examResult: null,
        ...(m === 'exam' ? freshSession() : {}),
        ui: {
          ...s.ui,
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
  setTruth: (t, caseId) => set({ truth: t, caseId }),
  addMeasurement: (m) =>
    set((s) => {
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
  removeMeasurement: (id) =>
    set((s) => ({ measurements: s.measurements.filter((m) => m.id !== id) })),
  clearMeasurements: () => set({ measurements: [] }),
  setActiveTool: (t) =>
    set({
      activeTool: t,
      activeMeasurementId: null,
      ...(t !== 'none' ? { cinePlaying: false } : {}),
    }),
  setActiveMeasurement: (id) => {
    if (!id) return set({ activeMeasurementId: null, activeTool: 'none' });
    const spec = getMeasurementSpec(id);
    if (!spec) return set({ activeMeasurementId: null, activeTool: 'none' });
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
  recordViewScore: (viewId, score) =>
    set((s) => {
      // the same state when nothing changes: a new one, even empty, notifies every subscriber (decision 173)
      const overall = (s.viewProgress[viewId] ?? 0) < score;
      // a view reached with its preset does not count as acquired by hand (decision 174)
      const handBefore = s.handViewProgress[viewId] ?? 0;
      const byHand = !s.presetViews.includes(viewId) && handBefore < score;
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
      return {
        examFinished: true,
        examResult: summary,
        frozen: true,
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
      const progress = addEvent(s.progress, { t: Date.now(), kind: 'case', caseId: id });
      saveProgress(typeof localStorage !== 'undefined' ? localStorage : null, progress);
      return { caseId: id, ...freshSession(), progress };
    }),
});
