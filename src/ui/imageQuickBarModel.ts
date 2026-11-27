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
