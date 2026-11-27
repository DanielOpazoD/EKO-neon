import { fractionalAreaChange } from '@/clinical/formulas';
import { biplaneVolume } from '@/education/report';
import {
  MEASUREMENT_SPECS,
  type MeasurementSpec,
  type MeasurementTool,
  type PhaseRequirement,
} from '@/simulator/measurements/protocol';
import type { Measurement } from '@/simulator/measurements/types';
import { isSpectralModality, isStripModality } from '@/simulator/renderer/modality';
import type { ImagingModality } from '@/simulator/renderer/types';

/** Compact Spanish clinical label for a required cardiac phase, or null when phase-agnostic. */
export function phaseTag(phase: PhaseRequirement): string | null {
  switch (phase) {
    case 'ed':
      return 'TD';
    case 'es':
      return 'TS';
    case 'mid-systole':
      return 'Mesosíst.';
    case 'early-diastole':
      return 'Diást. temp.';
    case 'late-diastole':
      return 'Diást. tard.';
    case 'systole':
      return 'Sístole';
    case 'any':
      return null;
  }
}

/** Whether a free measurement tool can meaningfully capture on the active imaging modality. */
export function isFreeToolCompatible(
  tool: MeasurementTool | 'none',
  modality: ImagingModality,
): boolean {
  switch (tool) {
    case 'caliper':
    case 'simpson':
      return modality === '2d' || modality === 'color';
    case 'velocity':
    case 'vti':
    case 'auto-vti':
    case 'slope':
      return isSpectralModality(modality);
    case 'tapse':
      return modality === 'm-mode' || modality === 'cmm';
    case 'time':
      return isStripModality(modality);
    default:
      return true;
  }
}

/** Recommended modality to switch to when a free tool is armed on an incompatible modality. */
export function defaultModalityForTool(tool: MeasurementTool): ImagingModality {
  switch (tool) {
    case 'caliper':
    case 'simpson':
      return '2d';
    case 'tapse':
      return 'm-mode';
    case 'velocity':
    case 'vti':
    case 'auto-vti':
    case 'slope':
    case 'time':
      return 'pw';
  }
}

/**
 * Compatibility of an armed capture session (protocol spec or free tool) with the current modality.
 * Returns `ok: true` when ready, or `ok: false` with the best target modality to switch to in 1 click.
 */
export function captureModalityCheck(
  spec: MeasurementSpec | undefined,
  activeTool: MeasurementTool | 'none',
  modality: ImagingModality,
): { ok: boolean; suggestedModality: ImagingModality | null } {
  if (spec) {
    if (spec.modalities.includes(modality)) return { ok: true, suggestedModality: null };
    return { ok: false, suggestedModality: spec.modalities[0] ?? '2d' };
  }
  if (activeTool === 'none') return { ok: true, suggestedModality: null };
  if (isFreeToolCompatible(activeTool, modality)) return { ok: true, suggestedModality: null };
  return { ok: false, suggestedModality: defaultModalityForTool(activeTool) };
}

/**
 * Protocol measurements clinically applicable to the currently recognized view, ordered with
 * required measurements first, then measurements matching the active modality, then other
 * modalities on the same view (so e.g. on A5C in 2D, TSVI ⌀ appears before PW/CW VTI).
 */
export function specsForView(
  viewId: string | null,
  modality: ImagingModality,
  requiredIds: readonly string[],
): MeasurementSpec[] {
  if (!viewId) return [];
  const reqSet = new Set(requiredIds);
  const matching = MEASUREMENT_SPECS.filter((spec) => spec.views.includes(viewId));
  return matching.sort((a, b) => {
    const aMod = a.modalities.includes(modality) ? 0 : 1;
    const bMod = b.modalities.includes(modality) ? 0 : 1;
    if (aMod !== bMod) return aMod - bMod;
    const aReq = reqSet.has(a.id) ? 0 : 1;
    const bReq = reqSet.has(b.id) ? 0 : 1;
    if (aReq !== bReq) return aReq - bReq;
    return 0;
  });
}

export interface AutoEfChamberSummary {
  hasResult: boolean;
  edvMl: number | null;
  esvMl: number | null;
  svMl: number | null;
  efPct: number | null;
  edFrameId: number | null;
  esFrameId: number | null;
  edTimeS: number | null;
  esTimeS: number | null;
  /** Ready on the current 2D apical view (always null in exam mode so view identity stays hidden). */
  viewReady: boolean | null;
}

/** Whether a captured measurement's timestamp still lies within the active rolling cine buffer. */
export function isTimeInCineWindow(
  timeS: number | null | undefined,
  cineWindow: { startS: number; endS: number } | null,
): boolean {
  if (timeS == null || !cineWindow) return false;
  if (!(cineWindow.endS >= cineWindow.startS)) return false;
  return timeS >= cineWindow.startS - 0.02 && timeS <= cineWindow.endS + 0.02;
}

export interface PlaneSimpsonSummary {
  edvMl: number;
  esvMl: number;
  efPct: number;
  edLengthCm: number | null;
  esLengthCm: number | null;
  edFrameId: number;
  esFrameId: number;
  edTimeS: number;
  esTimeS: number;
}

export interface AutoEfPanelState {
  is2dReady: boolean;
  lv: AutoEfChamberSummary & {
    a4cDone: boolean;
    a2cDone: boolean;
    biplane: boolean;
    a4c: PlaneSimpsonSummary | null;
    a2c: PlaneSimpsonSummary | null;
    /** True when A4C and A2C ED long axes differ by >10% (ASE/EACVI foreshortening alert). */
    lengthMismatch: boolean;
  };
  rv: AutoEfChamberSummary & {
    facPct: number | null;
    edAreaCm2: number | null;
    esAreaCm2: number | null;
  };
}

function latestByMeasurementId(ms: readonly Measurement[], id: string): Measurement | null {
  for (let i = ms.length - 1; i >= 0; i--) {
    if (ms[i]!.measurementId === id) return ms[i]!;
  }
  return null;
}

function latestByIdAndView(
  ms: readonly Measurement[],
  id: string,
  view: string,
): Measurement | null {
  for (let i = ms.length - 1; i >= 0; i--) {
    const m = ms[i]!;
    if (m.measurementId === id && m.sourceViewId === view) return m;
  }
  return null;
}

function planeSimpsonSummary(
  ms: readonly Measurement[],
  view: 'a4c' | 'a2c',
): PlaneSimpsonSummary | null {
  const ed = latestByIdAndView(ms, 'lv-edv-simpson', view);
  const es = latestByIdAndView(ms, 'lv-esv-simpson', view);
  if (!ed || !es || !(ed.value > 0)) return null;
  return {
    edvMl: ed.value,
    esvMl: es.value,
    efPct: ((ed.value - es.value) / ed.value) * 100,
    edLengthCm: ed.derived?.longAxisCm ?? null,
    esLengthCm: es.derived?.longAxisCm ?? null,
    edFrameId: ed.frameId,
    esFrameId: es.frameId,
    edTimeS: ed.timeS,
    esTimeS: es.timeS,
  };
}

/** Structured summary for the Auto-FE VI/VD workspace, respecting exam privacy when `showViewFeedback` is false. */
export function summarizeAutoEf(
  measurements: readonly Measurement[],
  currentViewId: string | null,
  modality: ImagingModality,
  showViewFeedback: boolean,
): AutoEfPanelState {
  const is2dReady = modality === '2d' || modality === 'color';

  const lvEdvM = latestByMeasurementId(measurements, 'lv-edv-simpson');
  const lvEsvM = latestByMeasurementId(measurements, 'lv-esv-simpson');
  const lvEdvBi = biplaneVolume(measurements as Measurement[], 'lv-edv-simpson');
  const lvEsvBi = biplaneVolume(measurements as Measurement[], 'lv-esv-simpson');
  const biplane = Boolean(lvEdvBi && lvEsvBi && lvEdvBi.volumeMl > 0);
  const lvEdv = biplane ? lvEdvBi!.volumeMl : (lvEdvM?.value ?? null);
  const lvEsv = biplane ? lvEsvBi!.volumeMl : (lvEsvM?.value ?? null);
  const lvEf = lvEdv !== null && lvEsv !== null && lvEdv > 0 ? ((lvEdv - lvEsv) / lvEdv) * 100 : null;
  const a4c = planeSimpsonSummary(measurements, 'a4c');
  const a2c = planeSimpsonSummary(measurements, 'a2c');
  const a4cDone = a4c !== null;
  const a2cDone = a2c !== null;
  const lengthMismatch = Boolean(
    lvEdvBi &&
      Math.abs(lvEdvBi.a4cLongAxisCm - lvEdvBi.a2cLongAxisCm) >
        0.1 * Math.max(lvEdvBi.a4cLongAxisCm, lvEdvBi.a2cLongAxisCm),
  );

  const rvEdvM = latestByMeasurementId(measurements, 'rv-edv');
  const rvEsvM = latestByMeasurementId(measurements, 'rv-esv');
  const rvEdv = rvEdvM?.value ?? null;
  const rvEsv = rvEsvM?.value ?? null;
  const rvEf = rvEdv !== null && rvEsv !== null && rvEdv > 0 ? ((rvEdv - rvEsv) / rvEdv) * 100 : null;
  const rvEdArea = rvEdvM?.derived?.areaCm2 ?? null;
  const rvEsArea = rvEsvM?.derived?.areaCm2 ?? null;
  const rvFac =
    rvEdArea !== null && rvEsArea !== null && rvEdArea > 0
      ? fractionalAreaChange(rvEdArea, rvEsArea)
      : null;

  const lvViewReady = showViewFeedback
    ? is2dReady && (currentViewId === 'a4c' || currentViewId === 'a2c')
    : null;
  const rvViewReady = showViewFeedback
    ? is2dReady && (currentViewId === 'rv-focused' || currentViewId === 'a4c')
    : null;

  return {
    is2dReady,
    lv: {
      hasResult: lvEf !== null && lvEdv !== null && lvEsv !== null,
      edvMl: lvEdv,
      esvMl: lvEsv,
      svMl: lvEdv !== null && lvEsv !== null ? lvEdv - lvEsv : null,
      efPct: lvEf,
      edFrameId: lvEdvM?.frameId ?? null,
      esFrameId: lvEsvM?.frameId ?? null,
      edTimeS: lvEdvM?.timeS ?? null,
      esTimeS: lvEsvM?.timeS ?? null,
      viewReady: lvViewReady,
      a4cDone,
      a2cDone,
      biplane,
      a4c,
      a2c,
      lengthMismatch,
    },
    rv: {
      hasResult: rvEf !== null && rvEdv !== null && rvEsv !== null,
      edvMl: rvEdv,
      esvMl: rvEsv,
      svMl: rvEdv !== null && rvEsv !== null ? rvEdv - rvEsv : null,
      efPct: rvEf,
      edFrameId: rvEdvM?.frameId ?? null,
      esFrameId: rvEsvM?.frameId ?? null,
      edTimeS: rvEdvM?.timeS ?? null,
      esTimeS: rvEsvM?.timeS ?? null,
      viewReady: rvViewReady,
      facPct: rvFac,
      edAreaCm2: rvEdArea,
      esAreaCm2: rvEsArea,
    },
  };
}

export interface QuickStudyStep {
  id: 'plax' | 'lv-ef' | 'lvot' | 'rv';
  label: string;
  targetViewId: string;
  status: 'pending' | 'partial' | 'done';
  /** Compact progress tag when partially complete (e.g. "1/2" when only A4C of biplane is done). */
  detail: string | null;
  active: boolean;
  tip: string;
}

const PLAX_MEASUREMENT_IDS = new Set([
  'lv-edd',
  'lv-esd',
  'ivsd',
  'lvpwd',
  'lvot-diameter',
  'aortic-root',
  'la-ap',
]);
const LVOT_MEASUREMENT_IDS = new Set(['lvot-vti', 'av-vti', 'av-vmax', 'lvot-peak-velocity']);
const RV_SINGLE_MEASUREMENT_IDS = new Set(['tapse', 'tr-vmax', 'rv-basal', 'rv-s-prime']);

/**
 * Derive the 4-step «Estudio rápido» clinical workflow checklist purely from captured measurements
 * (zero extra store state): 1. PLAX -> 2. VI FE (A4C+A2C) -> 3. TSVI (A5C) -> 4. VD.
 */
export function computeQuickStudySteps(
  measurements: readonly Measurement[],
  currentViewId: string | null,
): { steps: QuickStudyStep[]; completedCount: number } {
  const ids = new Set<string>();
  for (const m of measurements) {
    if (m.measurementId) ids.add(m.measurementId);
  }
  const hasAny = (set: Set<string>) => {
    for (const id of set) if (ids.has(id)) return true;
    return false;
  };

  const plaxDone = hasAny(PLAX_MEASUREMENT_IDS);

  const a4cDone = planeSimpsonSummary(measurements, 'a4c') !== null;
  const a2cDone = planeSimpsonSummary(measurements, 'a2c') !== null;
  const anyLvPair = ids.has('lv-edv-simpson') && ids.has('lv-esv-simpson');
  const lvEfStatus: QuickStudyStep['status'] =
    a4cDone && a2cDone ? 'done' : a4cDone || a2cDone || anyLvPair ? 'partial' : 'pending';
  const lvEfTarget = a4cDone && !a2cDone ? 'a2c' : 'a4c';

  const lvotDone = hasAny(LVOT_MEASUREMENT_IDS);

  const rvVolDone = ids.has('rv-edv') && ids.has('rv-esv');
  const rvDone = rvVolDone || hasAny(RV_SINGLE_MEASUREMENT_IDS);

  const steps: QuickStudyStep[] = [
    {
      id: 'plax',
      label: '1. PLAX',
      targetViewId: 'plax',
      status: plaxDone ? 'done' : 'pending',
      detail: null,
      active: currentViewId === 'plax',
      tip: 'PLAX: mide DTDVI o diámetro del TSVI',
    },
    {
      id: 'lv-ef',
      label: '2. VI FE',
      targetViewId: lvEfTarget,
      status: lvEfStatus,
      detail: lvEfStatus === 'partial' ? '1/2' : null,
      active: currentViewId === 'a4c' || currentViewId === 'a2c',
      tip:
        lvEfStatus === 'partial' && lvEfTarget === 'a2c'
          ? 'A4C listo · salta a A2C para completar Simpson biplano del VI'
          : 'Apical A4C + A2C: Simpson biplano del VI (Auto-FE VI)',
    },
    {
      id: 'lvot',
      label: '3. TSVI',
      targetViewId: 'a5c',
      status: lvotDone ? 'done' : 'pending',
      detail: null,
      active: currentViewId === 'a5c' || currentViewId === 'a3c',
      tip: 'A5C: Gate TSVI + Auto-VTI en Doppler pulsado',
    },
    {
      id: 'rv',
      label: '4. VD',
      targetViewId: 'rv-focused',
      status: rvDone ? 'done' : 'pending',
      detail: null,
      active: currentViewId === 'rv-focused',
      tip: 'Apical enfocada en VD: Auto-FE VD, TAPSE o Vmax IT',
    },
  ];
  const completedCount = steps.filter((s) => s.status === 'done').length;
  return { steps, completedCount };
}
