import { axialWallFactor } from './lvShape';
import { latticeNoise3 } from '@/core/noise';
import { RV_GROOVE_ANTERIOR_RAD, RV_GROOVE_INFERIOR_RAD } from './lvSegments';
import type { HeartModel } from './heartModel';

/**
 * Share of the septal thickness in the wall at azimuth `az` (rad): 1 on the septum, the wall between the RV insertions
 * `rvAzA` and `rvAzP` (the anteroseptal and inferoseptal walls of lvSegments.ts), 0 on the free wall, and a smoothstep
 * across ±0.2 rad (11.5°) of each insertion. It was 0.5 − 0.5·cos(az), centred on the septum of the A4C: ivsd only at the
 * inferoseptal 180° and lvpwd only at the anterolateral 0°, while the PLAX caliper crosses the anteroseptal wall at
 * 120° and the inferolateral at −60°, where it gave 0.75 and 0.25. HOCM (2.1 / 1.1 cm declared) then measured 1.88 cm of
 * septum and 1.36 cm of posterior wall at the level of the mitral tips in PLAX, SIV/PP 1.38 instead of 1.91.
 * The transition is narrower than the 28° from the anterior insertion to the PLAX (and PSAX) caliper line and the 20°
 * from the inferior insertion to the A4C one. The smoothstep is antisymmetric about each insertion, so the share
 * integrates to the septal arc exactly (meanWallThicknessCm). Translated to GLSL (glslGenerated.ts).
 */
export function septalShare(az: number, rvAzA: number, rvAzP: number): number {
  const d = az - (rvAzA + rvAzP) * 0.5;
  // angular distance from the septal centre, in [0, π]
  const off = Math.abs(d - 2 * Math.PI * Math.floor(d / (2 * Math.PI) + 0.5));
  const t = Math.min(1, Math.max(0, ((rvAzP - rvAzA) * 0.5 + 0.2 - off) / 0.4));
  return t * t * (3 - 2 * t);
}

/**
 * End-diastolic base thickness (cm) averaged around the circumference: the free wall, and the septum over its share of
 * it ((rvAzP − rvAzA) / 2π = 0.33). The shell volume that the systolic thickening conserves is the wall as drawn; with
 * the septal share centred on the A4C the mean was (ivsd + lvpwd) / 2.
 */
export function meanWallThicknessCm(ivsd: number, lvpwd: number): number {
  return (
    lvpwd + ((ivsd - lvpwd) * (RV_GROOVE_INFERIOR_RAD - RV_GROOVE_ANTERIOR_RAD)) / (2 * Math.PI)
  );
}

/**
 * Gain of the low-frequency thickness modulation: the wall varies by up to ±WALL_MODULATION_GAIN / 2 at the base, fading
 * to none at the apical cap. It was 0.28 (±14 %), more than the 12 % the protocol grants SIVd and PPd: at the caliper
 * sites of the 12 cases it moved the drawn wall by −11.4 … +10.8 % (rms 5.8 % over the basal ring), so that an exact
 * caliper could fail a case. At 0.08 (at most ±3.6 % at those sites, rms 1.6 %) it takes under a third of the
 * tolerance and leaves the rest to the caliper. Shared with GLSL (glslHeartChambers.ts).
 */
export const WALL_MODULATION_GAIN = 0.08;

/**
 * Local LV wall thickness (cm) at azimuth/level: the end-diastolic thickness of the septum (`ivsd`) between the RV
 * insertions and of the free wall (`lvpwd`) elsewhere (`septalShare`), systolic thickening from wall incompressibility,
 * and a low-frequency modulation around the wall.
 *
 * The thickening follows the local radial excursion: `motion` is the share of the normal excursion the wall makes here
 * (`lvRegionalMotion`; `regionalCavityOffset` holds its endocardium back by the rest), and an incompressible wall
 * thickens, to first order, in proportion to how far its endocardium moves in (δt ≈ δr·t/(r + t)), so it gains that share
 * of the normal thickening thickK − 1. An akinetic wall keeps its end-diastolic thickness and its epicardium stays; a
 * dyskinetic one (motion < 0), pushed out, thins. It kept 0.35 + 0.65·amplitude of the thickening: the akinetic
 * mid inferior segment of the inferior infarction (amplitude 0.1) thickened 18 % (0.90 → 1.06 cm at end-systole) with
 * 0.5 mm of endocardial excursion, and its epicardium moved 1.1 mm outward; it now thickens 4.4 % with 0.8 mm (a tenth
 * of the normal walls' 42 % and 8 mm), and its epicardium moves 0.4 mm in.
 */
export function wallThicknessAt(
  m: HeartModel,
  thickK: number,
  az: number,
  levelFrac: number,
  motion: number,
): number {
  const lv = m.lv;
  const tBase =
    lv.lvpwd +
    (lv.ivsd - lv.lvpwd) * septalShare(az, RV_GROOVE_ANTERIOR_RAD, RV_GROOVE_INFERIOR_RAD);
  // end-diastolic thickness: septal / free-wall value over the basal half, tapering to the apical thickness
  const tED = tBase * axialWallFactor(levelFrac, lv.apexT / tBase);
  // low-frequency thickness modulation (at most ±4 % at the base, fading to none at the apical cap)
  const wallMod =
    1 +
    WALL_MODULATION_GAIN *
      (latticeNoise3(
        Math.cos(az) * 1.6 + 7.3,
        Math.sin(az) * 1.6 + 2.1,
        levelFrac * 2.4,
        m.wallNoise,
      ) -
        0.5) *
      (1 - levelFrac * levelFrac);
  return tED * Math.max(0.6, 1 + (thickK - 1) * motion) * wallMod;
}

/** Septal shift toward the LV (cm) at azimuth/level: maximal at the mid septum, zero at the free wall, base and apex. */
export function septalShiftAt(shiftCm: number, az: number, levelFrac: number): number {
  if (shiftCm <= 0) return 0;
  const c = -Math.cos(az); // 1 at the septum (az = π)
  if (c <= 0) return 0;
  const zw = 1 - Math.pow((levelFrac - 0.45) / 0.45, 2);
  if (zw <= 0) return 0;
  return shiftCm * c * c * zw;
}

/**
 * The septal crest under the membranous septum (decision 223). Beneath the commissure of the non-coronary and right
 * coronary sinuses the muscular septum narrows to the thin membranous septum: by CT 3.0-3.4 mm at 1 mm below it, 4.1-4.7
 * at 2 mm, 6.1-6.9 at 5 mm and 9.7-10.7 at 10 mm (Schamroth Pravda et al., Europace 2024;26:euae109). The model's septum
 * kept its full thickness up to its basal edge, 1.0-1.1 cm right under the aortic annulus. `septalCrestFactor` is the
 * share of the wall thickness kept at azimuth `az` (around the LV axis) and height `zRel` below the mitral annulus: the
 * membranous part (0.2) above SEPTAL_CREST_Z_CM, the CT profile below it (as shares of the thickness 1 cm down), over
 * ±SEPTAL_CREST_HALF_WIDTH around the commissure's azimuth.
 */
export const SEPTAL_CREST_AZ = 2.85;
export const SEPTAL_CREST_HALF_WIDTH = 0.7;
export const SEPTAL_CREST_Z_CM = 0.05;
export function septalCrestFactor(az: number, zRel: number): number {
  let da = Math.abs(az - SEPTAL_CREST_AZ);
  if (da > Math.PI) da = 2 * Math.PI - da;
  if (da >= SEPTAL_CREST_HALF_WIDTH) return 1;
  const cw = Math.cos((da / SEPTAL_CREST_HALF_WIDTH) * (Math.PI / 2));
  const d = zRel - SEPTAL_CREST_Z_CM;
  let f = 1;
  if (d <= 0) f = 0.2;
  else if (d < 0.1) f = 0.2 + 1.1 * d;
  else if (d < 0.2) f = 0.31 + 1.2 * (d - 0.1);
  else if (d < 0.5) f = 0.43 + 0.7 * (d - 0.2);
  else if (d < 1) f = 0.64 + 0.72 * (d - 0.5);
  return 1 - cw * cw * (1 - f);
}
