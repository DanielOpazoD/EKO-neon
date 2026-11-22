/**
 * Local learning analytics (spec 40, proposal 7): events and progress live only in the learner's
 * browser (localStorage) and can be exported as JSON by the learner for research use. No telemetry.
 */
export type ProgressEvent =
  | { t: number; kind: 'view'; caseId: string; viewId: string; score: number }
  | {
      t: number;
      kind: 'measurement';
      caseId: string;
      measurementId: string;
      techniqueScore: number | null;
      value: number;
    }
  | { t: number; kind: 'task'; taskId: string }
  | { t: number; kind: 'case'; caseId: string }
  /** an exam began on the case it drew: with the 'exam' record it gives the time the exam took */
  | { t: number; kind: 'exam-start'; caseId: string }
  | {
      t: number;
      kind: 'exam';
      caseId: string;
      total: number;
      acquisition: number;
      measurements: number;
      /** null when no impression was marked (it then weighs nothing in the total) */
      impression: number | null;
    }
  | { t: number; kind: 'impression'; caseId: string; score: number };

export interface ProgressState {
  version: 1;
  completedTasks: Record<string, number>; // taskId → timestamp
  events: ProgressEvent[];
}

export const PROGRESS_KEY = 'echotwin.progress.v1';
const MAX_EVENTS = 2000;

export function emptyProgress(): ProgressState {
  return { version: 1, completedTasks: {}, events: [] };
}

export interface ProgressStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

const isFiniteNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const isNullableNum = (v: unknown): v is number | null => v === null || isFiniteNum(v);
const isStr = (v: unknown): v is string => typeof v === 'string';

/** Parse a single stored ProgressEvent; returns null for malformed or unknown shapes. */
function parseEvent(raw: unknown): ProgressEvent | null {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return null;
  const e = raw as Record<string, unknown>;
  if (!isFiniteNum(e.t) || !isStr(e.kind)) return null;
  switch (e.kind) {
    case 'view':
      return isStr(e.caseId) && isStr(e.viewId) && isFiniteNum(e.score)
        ? { t: e.t, kind: 'view', caseId: e.caseId, viewId: e.viewId, score: e.score }
        : null;
    case 'measurement':
      return isStr(e.caseId) &&
        isStr(e.measurementId) &&
        isNullableNum(e.techniqueScore) &&
        isFiniteNum(e.value)
        ? {
            t: e.t,
            kind: 'measurement',
            caseId: e.caseId,
            measurementId: e.measurementId,
            techniqueScore: e.techniqueScore,
            value: e.value,
          }
        : null;
    case 'task':
      return isStr(e.taskId) ? { t: e.t, kind: 'task', taskId: e.taskId } : null;
    case 'case':
      return isStr(e.caseId) ? { t: e.t, kind: 'case', caseId: e.caseId } : null;
    case 'exam-start':
      return isStr(e.caseId) ? { t: e.t, kind: 'exam-start', caseId: e.caseId } : null;
    case 'exam':
      return isStr(e.caseId) &&
        isFiniteNum(e.total) &&
        isFiniteNum(e.acquisition) &&
        isFiniteNum(e.measurements) &&
        isNullableNum(e.impression)
        ? {
            t: e.t,
            kind: 'exam',
            caseId: e.caseId,
            total: e.total,
            acquisition: e.acquisition,
            measurements: e.measurements,
            impression: e.impression,
          }
        : null;
    case 'impression':
      return isStr(e.caseId) && isFiniteNum(e.score)
        ? { t: e.t, kind: 'impression', caseId: e.caseId, score: e.score }
        : null;
    default:
      return null;
  }
}

function parseCompletedTasks(raw: unknown): Record<string, number> {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return {};
  const entries = Object.entries(raw as Record<string, unknown>);
  const out: Record<string, number> = {};
  for (const [k, v] of entries) {
    if (!isFiniteNum(v)) return {};
    out[k] = v;
  }
  return out;
}

export function loadProgress(storage: ProgressStorage | null): ProgressState {
  if (!storage) return emptyProgress();
  try {
    const raw = storage.getItem(PROGRESS_KEY);
    if (!raw) return emptyProgress();
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== 'object' || parsed === null) return emptyProgress();
    const { version, events, completedTasks } = parsed as Record<string, unknown>;
    if (version !== 1 || !Array.isArray(events)) return emptyProgress();
    // a malformed event (old build, hand edit) is dropped on its own instead of reaching the summaries with the
    // wrong shape; the rest of the history survives
    const valid: ProgressEvent[] = [];
    for (const e of events) {
      const ev = parseEvent(e);
      if (ev) valid.push(ev);
    }
    return {
      version: 1,
      completedTasks: parseCompletedTasks(completedTasks ?? {}),
      events: valid.slice(-MAX_EVENTS),
    };
  } catch {
    return emptyProgress();
  }
}

export function saveProgress(storage: ProgressStorage | null, p: ProgressState): void {
  if (!storage) return;
  try {
    storage.setItem(PROGRESS_KEY, JSON.stringify({ ...p, events: p.events.slice(-MAX_EVENTS) }));
  } catch {
    /* quota or private mode: progress stays in memory */
  }
}

export function addEvent(p: ProgressState, e: ProgressEvent): ProgressState {
  const events = [...p.events, e].slice(-MAX_EVENTS);
  return { ...p, events };
}

/**
 * The score of the first impression the learner had corrected on a case, or null if none. A later attempt follows the
 * revealed answer key («Corregir» shows which findings are right, «Rehacer» clears the sheet), so it cannot show that the
 * learner can read the case: only the first counts toward the curriculum.
 */
export function firstImpressionScore(p: ProgressState, caseId: string): number | null {
  for (const e of p.events) if (e.kind === 'impression' && e.caseId === caseId) return e.score;
  return null;
}

export function completeTask(p: ProgressState, taskId: string, t: number): ProgressState {
  if (p.completedTasks[taskId]) return p;
  return addEvent(
    { ...p, completedTasks: { ...p.completedTasks, [taskId]: t } },
    { t, kind: 'task', taskId },
  );
}

export interface ProgressSummary {
  bestViewScores: Record<string, number>;
  /** How many times the view's best score by hand was first set or rose by 5 points or more: improvements, not attempts (decision 236). */
  viewImprovements: Record<string, number>;
  measurementsCount: number;
  meanTechniqueScore: number | null;
  techniqueTrend: { t: number; score: number }[];
  casesOpened: string[];
  exams: { t: number; caseId: string; total: number }[];
  completedTasks: number;
}

export function summarizeProgress(p: ProgressState): ProgressSummary {
  const bestViewScores: Record<string, number> = {};
  const viewImprovements: Record<string, number> = {};
  const trend: { t: number; score: number }[] = [];
  const cases = new Set<string>();
  const exams: { t: number; caseId: string; total: number }[] = [];
  let tSum = 0,
    tN = 0,
    mN = 0;
  for (const e of p.events) {
    if (e.kind === 'view') {
      bestViewScores[e.viewId] = Math.max(bestViewScores[e.viewId] ?? 0, e.score);
      viewImprovements[e.viewId] = (viewImprovements[e.viewId] ?? 0) + 1;
    } else if (e.kind === 'measurement') {
      mN++;
      if (e.techniqueScore !== null) {
        tSum += e.techniqueScore;
        tN++;
        trend.push({ t: e.t, score: e.techniqueScore });
      }
    } else if (e.kind === 'case') cases.add(e.caseId);
    else if (e.kind === 'exam') exams.push({ t: e.t, caseId: e.caseId, total: e.total });
  }
  return {
    bestViewScores,
    viewImprovements,
    measurementsCount: mN,
    meanTechniqueScore: tN ? tSum / tN : null,
    techniqueTrend: trend,
    casesOpened: [...cases],
    exams,
    completedTasks: Object.keys(p.completedTasks).length,
  };
}

/** Anonymous export for the validation protocol (docs/VALIDATION_PROTOCOL.md): no identifiers, only events. */
export function exportProgressJson(p: ProgressState, appVersion: string): string {
  return JSON.stringify({ exportedAt: new Date().toISOString(), appVersion, progress: p }, null, 2);
}
