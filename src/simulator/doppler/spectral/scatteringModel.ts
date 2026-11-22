import { SPEED_OF_SOUND_MPS } from '@/core/units';
import {
  beamHalfWidthCm,
  ELEVATION_FOCUS_CM,
  focusingGain,
  SLICE_HALF_BASE_CM,
  SLICE_HALF_SLOPE,
} from '@/simulator/renderer/acoustic/acoustics';
import { DUPLEX_BMODE_SHARE, pulsedAcquisition } from '@/simulator/renderer/pulseTiming';

/** Reference transmit fundamental (MHz) against which Rayleigh scattering gain is normalised. */
export const RAYLEIGH_REF_MHZ = 2.5;

/**
 * Relative backscatter intensity scaling with transmit fundamental f0.
 * Erythrocytes (a << lambda) lie in the Rayleigh regime (sigma_b ~ f0^4), whereas myocardial
 * parenchyma (TDI) exhibits a weaker frequency slope (~ f0^1.4).
 */
export function rayleighBackscatterGain(frequencyMHz: number, isTissue = false): number {
  const ratio = Math.max(0.25, frequencyMHz / RAYLEIGH_REF_MHZ);
  if (isTissue) return Math.min(3.0, Math.max(0.35, Math.pow(ratio, 1.4)));
  return Math.min(6.0, Math.max(0.15, Math.pow(ratio, 4)));
}

/** Elevational slice half-width (cm) at depth rCm around the fixed cylindrical lens elevation focus. */
export function sliceHalfWidthCm(depthCm: number): number {
  return SLICE_HALF_BASE_CM + SLICE_HALF_SLOPE * Math.abs(depthCm - ELEVATION_FOCUS_CM);
}

/**
 * Two-way acoustic sensitivity weight of a scatterer at axial offset `drCm` and transverse offsets
 * `(duCm, dvCm)` from the cursor ray at depth `depthCm`. Combines focal waist sensitivity,
 * depth-dependent lateral/elevational Gaussian beam profiles, and (for PW/TDI) the rounded axial
 * transmit-pulse/receive-gate sensitivity envelope.
 */
export function sampleVolumeWeight(
  depthCm: number,
  focusCm: number,
  drCm: number,
  duCm: number,
  dvCm: number,
  gateLengthCm: number | null,
): number {
  const wl = Math.max(0.05, beamHalfWidthCm(depthCm, focusCm));
  const we = Math.max(0.05, sliceHalfWidthCm(depthCm));
  const focal = focusingGain(depthCm, focusCm);
  const transverse = Math.exp(-2 * ((duCm * duCm) / (wl * wl) + (dvCm * dvCm) / (we * we)));
  const axial =
    gateLengthCm !== null && gateLengthCm > 0
      ? Math.exp(-2 * (drCm / (0.6 * gateLengthCm)) ** 2)
      : 1;
  return focal * focal * transverse * axial;
}

/**
 * Characteristic residence (transit) time (s) of a scatterer crossing the finite acoustic sample
 * volume at depth `depthCm` with axial speed `vAxialMps` and cross-beam speed `vPerpMps`.
 * In CW (`gateLengthCm === null`), axial transit is unbounded and only lateral beam crossing acts.
 */
export function scattererTransitTimeS(
  vAxialMps: number,
  vPerpMps: number,
  depthCm: number,
  focusCm: number,
  gateLengthCm: number | null,
): number {
  const axialRate =
    gateLengthCm !== null && gateLengthCm > 0
      ? Math.abs(vAxialMps) / Math.max(5e-4, gateLengthCm / 100)
      : 0;
  const beamDiamM = Math.max(1e-3, (2 * beamHalfWidthCm(depthCm, focusCm)) / 100);
  const lateralRate = Math.abs(vPerpMps) / beamDiamM;
  const rate = Math.hypot(axialRate, lateralRate);
  return rate > 1e-4 ? 1 / rate : Infinity;
}

/**
 * Intrinsic transit-time spectral broadening standard deviation (m/s) caused by finite scatterer
 * dwell time inside the sample volume (Newhouse et al. 1980 / Evans & McDicken):
 *   sigma_v = (lambda / (2 * sqrt(2*pi))) / tau_transit.
 */
export function transitBroadeningMps(
  vAxialMps: number,
  vPerpMps: number,
  depthCm: number,
  focusCm: number,
  frequencyMHz: number,
  gateLengthCm: number | null,
): number {
  const tau = scattererTransitTimeS(vAxialMps, vPerpMps, depthCm, focusCm, gateLengthCm);
  if (!Number.isFinite(tau) || tau <= 0) return 0;
  const lambdaM = SPEED_OF_SOUND_MPS / Math.max(1e5, frequencyMHz * 1e6);
  return lambdaM / (2 * Math.sqrt(2 * Math.PI) * tau);
}

/** Depth-dependent conventional range-unambiguous pulsed Doppler timing and Nyquist ceiling. */
export interface DepthPulsedAcquisition {
  effectiveScaleMps: number;
  maxScaleMps: number;
  prfHz: number;
  maxPrfHz: number;
  nyquistHz: number;
  depthLimited: boolean;
  effectiveDepthCm: number;
}

/**
 * Computes the physical depth-dependent PRF and Nyquist ceiling for a PW/TDI gate centred at
 * `gateDepthCm` with axial length `gateLengthCm`. Moving the gate deeper increases round-trip echo
 * time 2*(z + L_g/2)/c, reducing maxPrfHz and forcing aliasing at progressively lower velocities.
 */
export function depthLimitedPulsedAcquisition(
  gateDepthCm: number,
  gateLengthCm: number,
  frequencyMHz: number,
  requestedScaleMps: number,
  activeFraction = 1 - DUPLEX_BMODE_SHARE,
): DepthPulsedAcquisition {
  const effectiveDepthCm = Math.max(0.5, gateDepthCm + Math.max(0, gateLengthCm) / 2);
  const acq = pulsedAcquisition(effectiveDepthCm, frequencyMHz, requestedScaleMps, activeFraction);
  return {
    effectiveScaleMps: acq.scaleMps,
    maxScaleMps: acq.maxScaleMps,
    prfHz: acq.prfHz,
    maxPrfHz: acq.maxPrfHz,
    nyquistHz: acq.prfHz / 2,
    depthLimited: requestedScaleMps > acq.maxScaleMps + 1e-4,
    effectiveDepthCm,
  };
}

/**
 * Smooth 2nd-order high-pass clutter (wall) filter power response |H(v)|^2.
 * Preserves a strict stop-band below `0.8 * wallFilterMps` to suppress stationary myocardium/valve
 * clutter while transitioning smoothly (-3 dB at `wallFilterMps`) instead of carving a hard stripe.
 */
export function clutterFilterPowerResponse(velocityMps: number, wallFilterMps: number): number {
  if (!(wallFilterMps > 0)) return 1;
  const r = Math.abs(velocityMps) / wallFilterMps;
  if (r < 0.8) return 0;
  const u = Math.min(1, (r - 0.8) / 0.2);
  const ramp = u * u * (3 - 2 * u);
  const r4 = r * r * r * r;
  return ramp * (r4 / (1 + r4));
}

/**
 * Signed Doppler shift frequency (Hz) after optional discrete-time PRF aliasing.
 * - CW (`aliasing === false`): unaliased 2*f0*v/c.
 * - PW/TDI (`aliasing === true`): wrapped modulo `prfHz` into `[-prfHz/2, +prfHz/2]`.
 */
export function aliasedDopplerShiftHz(
  velocityMps: number,
  f0Hz: number,
  prfHz: number,
  aliasing: boolean,
): number {
  const fd = (2 * f0Hz * velocityMps) / SPEED_OF_SOUND_MPS;
  if (!aliasing || !(prfHz > 0)) return fd;
  const half = prfHz / 2;
  return ((((fd + half) % prfHz) + prfHz) % prfHz) - half;
}

/**
 * Relative acoustic SNR factor from column peak power so deeply attenuated or poorly coupled gates
 * show a genuinely noisier baseline rather than normalising thermal noise to full display white.
 */
export function acousticSnrScale(columnPeakPower: number): number {
  if (!(columnPeakPower > 0)) return 0;
  // Soft saturation: strong cardiac flows (>= 0.15) keep full contrast; deep/attenuated weak flows
  // (< 0.02) progressively drop closer to the thermal noise floor.
  return columnPeakPower / (columnPeakPower + 0.015);
}
