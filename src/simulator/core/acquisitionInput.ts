import { DUPLEX_BMODE_SHARE, pulsedAcquisition } from '@/simulator/renderer/pulseTiming';
import type { SimInput } from './protocol';

type AcquisitionInput = Pick<
  SimInput,
  'settings' | 'color' | 'spectral' | 'modality' | 'gateDepthCm'
>;

/** Shared by the console store and the headless/worker boundary. No UI-only physics. */
export function acquisitionLimits(input: AcquisitionInput) {
  const { settings, spectral, color } = input;
  const gateDepthCm = Math.max(0.1, Math.min(settings.depthCm, input.gateDepthCm));
  return {
    gateDepthCm,
    color: pulsedAcquisition(
      input.modality === 'cmm' ? settings.depthCm : Math.min(settings.depthCm, color.boxRMaxCm),
      settings.frequencyMHz,
      color.scaleMps,
    ),
    spectral: pulsedAcquisition(
      gateDepthCm + spectral.gateLengthCm / 2,
      settings.frequencyMHz,
      spectral.scaleMps,
      1 - DUPLEX_BMODE_SHARE,
    ),
  };
}

function clampBaselineShift(baselineShiftMps: number, scaleMps: number): number {
  const maxShift = 0.85 * Math.max(0.01, scaleMps);
  return Math.max(-maxShift, Math.min(maxShift, baselineShiftMps));
}

export function constrainAcquisition<T extends AcquisitionInput>(input: T): T {
  const limits = acquisitionLimits(input);
  const pulsed = input.modality === 'pw' || input.modality === 'tdi';
  const colorScale = limits.color.scaleMps;
  const colorShift = clampBaselineShift(input.color.baselineShiftMps, colorScale);
  const spectralScale = pulsed ? limits.spectral.scaleMps : input.spectral.scaleMps;
  const spectralShift = clampBaselineShift(input.spectral.baselineShiftMps, spectralScale);
  return {
    ...input,
    gateDepthCm: limits.gateDepthCm,
    color:
      colorScale === input.color.scaleMps && colorShift === input.color.baselineShiftMps
        ? input.color
        : { ...input.color, scaleMps: colorScale, baselineShiftMps: colorShift },
    spectral:
      spectralScale === input.spectral.scaleMps &&
      spectralShift === input.spectral.baselineShiftMps
        ? input.spectral
        : { ...input.spectral, scaleMps: spectralScale, baselineShiftMps: spectralShift },
  };
}
