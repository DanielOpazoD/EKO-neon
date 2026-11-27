import type { PatientState } from '@/simulator/anatomy/thoraxModel';
import { SPECTRAL_SCALE_RANGE, type SpectralModality } from '@/simulator/doppler/spectral/spectrum';
import { isSpectralModality } from '@/simulator/renderer/modality';
import type { ImagingModality } from '@/simulator/renderer/types';

export type QuickBarGroup = 'bmode' | 'color' | 'spectral';

/** Which quick-bar control family matches the active imaging modality. */
export function quickBarGroupFor(modality: ImagingModality): QuickBarGroup {
  if (modality === 'color' || modality === 'cmm') return 'color';
  if (isSpectralModality(modality)) return 'spectral';
  return 'bmode';
}

/** Step Color/CMM Nyquist scale safely within physical depth/frequency limits and clamp baseline shift. */
export function stepColorScale(
  currentScaleMps: number,
  currentBaselineMps: number,
  maxScaleMps: number,
  dir: -1 | 1,
): { scaleMps: number; baselineShiftMps: number } {
  const hi = Math.max(0.15, Math.min(1.2, maxScaleMps));
  const next = Math.round(Math.max(0.15, Math.min(hi, currentScaleMps + dir * 0.05)) * 100) / 100;
  const maxShift = Math.round(next * 80) / 100;
  const baselineShiftMps = Math.max(-maxShift, Math.min(maxShift, currentBaselineMps));
  return { scaleMps: next, baselineShiftMps };
}

/** Step Color/CMM baseline shift symmetrically within ±80% of Nyquist scale. */
export function stepColorBaseline(
  currentScaleMps: number,
  currentBaselineMps: number,
  dir: -1 | 1,
): number {
  const maxShift = Math.round(currentScaleMps * 80) / 100;
  return (
    Math.round(Math.max(-maxShift, Math.min(maxShift, currentBaselineMps + dir * 0.05)) * 100) / 100
  );
}

/** Step PW/CW/TDI velocity scale safely within modality & depth limits and clamp baseline shift. */
export function stepSpectralScale(
  modality: SpectralModality,
  currentScaleMps: number,
  currentBaselineMps: number,
  maxScaleMps: number,
  dir: -1 | 1,
): { scaleMps: number; baselineShiftMps: number } {
  const range = SPECTRAL_SCALE_RANGE[modality];
  const step = modality === 'cw' ? 0.5 : modality === 'tdi' ? 0.04 : 0.1;
  const lo = modality === 'cw' ? range.min : Math.min(range.min, maxScaleMps);
  const hi = modality === 'cw' ? range.max : Math.min(range.max, maxScaleMps);
  const next = Math.round(Math.max(lo, Math.min(hi, currentScaleMps + dir * step)) * 100) / 100;
  const maxShift = Math.round(next * 80) / 100;
  const baselineShiftMps = Math.max(-maxShift, Math.min(maxShift, currentBaselineMps));
  return { scaleMps: next, baselineShiftMps };
}

/** Toggle 1-click HD Write-Zoom between normal (1×) and focalized 2× magnification. */
export function toggleHdZoom(currentZoom: number): number {
  return currentZoom > 1.05 ? 1 : 2;
}

/** Step PW/CW/TDI baseline shift cleanly within ±80% of active velocity scale. */
export function stepSpectralBaseline(
  modality: SpectralModality,
  currentScaleMps: number,
  currentBaselineMps: number,
  dir: -1 | 1,
): number {
  const step = modality === 'cw' ? 0.4 : modality === 'tdi' ? 0.02 : 0.1;
  const maxShift = Math.round(currentScaleMps * 80) / 100;
  return (
    Math.round(Math.max(-maxShift, Math.min(maxShift, currentBaselineMps + dir * step)) * 100) / 100
  );
}

export const SWEEP_SPEED_STEPS = [25, 50, 100] as const;

/** Cycle clinical strip sweep speed (25 -> 50 -> 100 -> 25 mm/s) for 1-click teaching adjustments. */
export function cycleSweepSpeed(currentMmPerS: number): 25 | 50 | 100 {
  if (currentMmPerS < 37.5) return 50;
  if (currentMmPerS < 75) return 100;
  return 25;
}

export const RESPIRATION_CYCLE: readonly PatientState['respiration'][] = [
  'free-breathing',
  'inspiration',
  'expiration',
  'breath-hold',
];

export const RESPIRATION_SHORT_LABEL: Record<PatientState['respiration'], string> = {
  'free-breathing': 'Libre',
  inspiration: 'Insp',
  expiration: 'Esp',
  'breath-hold': 'Apnea',
};

/** Cycle patient respiratory phase cleanly for bedside physiology demos (IVC sniff, tamponade inflow). */
export function cycleRespiration(
  current: PatientState['respiration'],
): PatientState['respiration'] {
  const idx = RESPIRATION_CYCLE.indexOf(current);
  return RESPIRATION_CYCLE[(idx + 1) % RESPIRATION_CYCLE.length] ?? 'free-breathing';
}
