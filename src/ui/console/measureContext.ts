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

export interface AutoEfPanelState {
  is2dReady: boolean;
  lv: AutoEfChamberSummary & {
    a4cDone: boolean;
    a2cDone: boolean;
    biplane: boolean;
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

function hasPlanePair(ms: readonly Measurement[], edId: string, esId: string, view: string): boolean {
  let hasEd = false;
  let hasEs = false;
  for (const m of ms) {
    if (m.sourceViewId !== view) continue;
    if (m.measurementId === edId) hasEd = true;
    if (m.measurementId === esId) hasEs = true;
  }
  return hasEd && hasEs;
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
  const a4cDone = hasPlanePair(measurements, 'lv-edv-simpson', 'lv-esv-simpson', 'a4c');
  const a2cDone = hasPlanePair(measurements, 'lv-edv-simpson', 'lv-esv-simpson', 'a2c');

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
