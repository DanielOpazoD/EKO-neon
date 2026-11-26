import type { AnatomyConfig } from '@/cases/schema';
import { smoothstep } from '@/core/vec3';
import {
  aha17FromCode,
  lvSegmentCode,
  RV_GROOVE_ANTERIOR_RAD,
  RV_GROOVE_INFERIOR_RAD,
} from './lvSegments';
import { meanWallThicknessCm } from './lvWall';
import { MITRAL_CENTRE_X } from './heartFrame';
import {
  allocLvProfileTable,
  axialWallFactor,
  buildLvProfile,
  ellipseFactor,
  lvNeckGain,
  lvProfileG,
  lvShapeFor,
  lvShellVolume,
  type LvProfileTable,
  type LvShape,
} from './lvShape';

/** End-diastolic LV geometry derived from the case volume, length and sphericity. */
export interface LvGeometry {
  lengthCm: number; // annulus → endocardial apex at ED
  rMax: number; // maximal lateral cavity radius at ED
  shape: LvShape; // bullet profile (see lvShape.ts)
  wallVolumeMl: number; // myocardial shell volume at ED (conserved through the cycle)
  ivsd: number;
  lvpwd: number;
  apexT: number;
  /** ED profile table (annulus at z = 0), for tools that need the resting geometry. */
  edProfile: LvProfileTable;
}

export function lvGeometryFromVolume(
  edvMl: number,
  lengthCm: number,
  sphericity: number,
  anatomy: AnatomyConfig['lv'],
  annulusR = Infinity,
): LvGeometry {
  const shape = lvShapeFor(sphericity);
  // cavity volume between the annulus plane and the apex: π·ratio·R²·L·∫g²
  const rMax = Math.sqrt(edvMl / (Math.PI * shape.ratio * lengthCm * shape.I));
  const edProfile = buildLvProfile(
    shape,
    rMax,
    lengthCm,
    0,
    allocLvProfileTable(),
    lvNeckGain(shape, rMax, annulusR),
    Number.isFinite(annulusR) ? MITRAL_CENTRE_X : 0,
  );
  // the shell carries the wall as drawn: its mean around the circumference, the septum a third of it
  const tBase = meanWallThicknessCm(anatomy.ivsdCm, anatomy.lvpwdCm);
  const tMean = (zeta: number): number =>
    tBase * axialWallFactor(zeta, anatomy.apexWallThicknessCm / tBase);
  const wallVolumeMl = Math.max(20, lvShellVolume(edProfile, shape.ratio, tMean, 1));
  return {
    lengthCm,
    rMax,
    shape,
    wallVolumeMl,
    ivsd: anatomy.ivsdCm,
    lvpwd: anatomy.lvpwdCm,
    apexT: anatomy.apexWallThicknessCm,
    edProfile,
  };
}

/** Wall-motion amplitude per AHA segment (1 normal). */
export type SegmentAmplitudes = Float32Array; // length 18 (index 1..17)

export function segmentAmplitudes(anatomy: AnatomyConfig): SegmentAmplitudes {
  const s = new Float32Array(18).fill(1);
  for (const wm of anatomy.wallMotion) s[wm.segment] = wm.amplitude;
  return s;
}

/**
 * AHA 17-segment id from heart-frame azimuth (rad, 0 = lateral, π/2 = anterior) and level fraction 0 (annulus) → 1 (end
 * of the cavity; the cap beyond it). The rules and the model choices are in `lvSegments.ts` (decision 152); until then
 * the cap began at 93 % of the cavity length and the azimuth was shifted by a fixed 28°.
 */
export function ahaSegment(azimuthRad: number, levelFrac: number): number {
  return aha17FromCode(
    lvSegmentCode(azimuthRad, levelFrac, RV_GROOVE_ANTERIOR_RAD, RV_GROOVE_INFERIOR_RAD),
  );
}

/**
 * Half-widths of the borders of the regional wall-motion field (`lvRegionalMotion`): 15° of azimuth on either side of a
 * segment border and 8 % of the cavity length on either side of a level border. At the mid cavity (end-diastolic radius
 * about 2.5 cm, length about 8 cm) both are about 0.65 cm of wall, so a border is as wide in short axis as in long axis:
 * the tethered border zone of an infarct, not a cut.
 */
export const WALL_MOTION_BORDER_DEG = 15;
export const WALL_MOTION_BORDER_LEVEL = 0.08;

/** A segment's share of the normal motion: its amplitude, no more than normal (hyperkinesia is not modelled). */
function segmentMotion(amp: SegmentAmplitudes, id: number): number {
  return Math.min(1, amp[id] ?? 1);
}

/**
 * One ring of the field at AHA angle `deg`: `n` segments of 360/n°, ids first … first + n − 1, the segment of id
 * `first` starting at AHA angle 360/n − offsetDeg (the six basal or mid segments: n 6, offset 0, anterior from 60°; the
 * four apical ones: n 4, offset 45°, anterior from 45°). Within WALL_MOTION_BORDER_DEG of a border the two segments
 * blend by a smoothstep centred on it: continuous with a continuous slope, and the smoothstep being antisymmetric about
 * its centre, the ring's mean is still the mean of its segments.
 */
function ringMotion(
  amp: SegmentAmplitudes,
  deg: number,
  n: number,
  offsetDeg: number,
  first: number,
): number {
  const span = 360 / n;
  const u = (deg + offsetDeg) / span;
  const k = Math.floor(u);
  const f = u - k;
  const hw = WALL_MOTION_BORDER_DEG / span;
  const here = segmentMotion(amp, first + (k - 1 - n * Math.floor((k - 1) / n)));
  if (f < hw) {
    const prev = segmentMotion(amp, first + (k - 2 - n * Math.floor((k - 2) / n)));
    return prev + (here - prev) * smoothstep(-hw, hw, f);
  }
  if (f > 1 - hw) {
    const next = segmentMotion(amp, first + (k - n * Math.floor(k / n)));
    return here + (next - here) * smoothstep(1 - hw, 1 + hw, f);
  }
  return here;
}

/**
 * Regional wall motion at heart-frame azimuth `az` (rad) and level fraction `levelFrac` (0 annulus → 1 end of the
 * cavity; the cap beyond it): the share of the normal radial excursion that the wall makes there, and of the normal
 * thickening (`wallThicknessAt`). 1 is normal, 0 akinetic, below 0 dyskinetic. The case declares one amplitude per AHA
 * segment (`segmentAmplitudes`, segments as in `lvSegmentCode`); read per segment it was constant by pieces, and an
 * akinetic segment beside a normal one stepped by the whole systolic excursion at their border: in the inferior
 * infarction at end-systole the endocardium jumped 7.2 mm and the epicardium 5.0 mm within 0.5° (from the inferoseptal
 * to the inferior wall, AHA 240°), and 5.7 and 4.0 mm within 0.25 % of the length from the mid to the apical level of
 * the inferolateral wall. Here the segments blend across each border, over ±WALL_MOTION_BORDER_DEG between the segments
 * of a ring and over ±WALL_MOTION_BORDER_LEVEL between the basal, mid and apical rings; the apical ring fades into the cap
 * (17) over the last WALL_MOTION_BORDER_LEVEL of the length, so that the tissue beyond the cavity, which reads level 1,
 * is the cap's alone. The largest steps left are 0.2 mm per 0.5° and 0.3 mm per 0.25 % of the length, the wall's own
 * curvature and trabeculae. Translated to GLSL (`lvRegionalMotion` in glslHeartChambers.ts, with the same constants).
 */
export function lvRegionalMotion(amp: SegmentAmplitudes, az: number, levelFrac: number): number {
  // the AHA angle as lvSegmentCode reads it: the septal centre at 180°, lateral near 0°, anterior near 90°
  const raw =
    ((az - (RV_GROOVE_ANTERIOR_RAD + RV_GROOVE_INFERIOR_RAD) * 0.5) * 180) / Math.PI + 180;
  const deg = raw - 360 * Math.floor(raw / 360);
  const h = WALL_MOTION_BORDER_LEVEL;
  if (levelFrac < 1 / 3 + h) {
    const basal = ringMotion(amp, deg, 6, 0, 1);
    if (levelFrac <= 1 / 3 - h) return basal;
    const mid = ringMotion(amp, deg, 6, 0, 7);
    return basal + (mid - basal) * smoothstep(1 / 3 - h, 1 / 3 + h, levelFrac);
  }
  if (levelFrac < 2 / 3 + h) {
    const mid = ringMotion(amp, deg, 6, 0, 7);
    if (levelFrac <= 2 / 3 - h) return mid;
    const apical = ringMotion(amp, deg, 4, 45, 13);
    return mid + (apical - mid) * smoothstep(2 / 3 - h, 2 / 3 + h, levelFrac);
  }
  const apical = ringMotion(amp, deg, 4, 45, 13);
  if (levelFrac <= 1 - h) return apical;
  // weighted so that level 1 reads the cap exactly
  const t = smoothstep(1 - h, 1, levelFrac);
  return apical * (1 - t) + segmentMotion(amp, 17) * t;
}

/**
 * Outward offset (cm, along the cavity normal) of the endocardium at azimuth `az` and level `levelFrac` where the wall
 * makes `motion` of the normal excursion (`lvRegionalMotion`): the wall moves from its end-diastolic radius only that
 * share of the way to the radius the normal wall has now (`rMaxNow`), so an akinetic wall (0) stays at its end-diastolic
 * radius and a dyskinetic one bulges past it. The excursion is that of the elliptical cross-section at this azimuth
 * (ellipseFactor): without it the akinetic inferior and anterior walls, where the section is 5-6 % narrower than
 * laterally, bulged 0.4 mm past their end-diastolic radius at end-systole. Exact in the body of the ventricle; an
 * approximation in the basal fifth, whose neck narrows less in systole (lvNeckGain), and at the apex, where the cavity
 * normal leaves the radial direction. Translated to GLSL (glslHeartChambers.ts).
 */
export function regionalCavityOffset(
  lv: LvGeometry,
  rMaxNow: number,
  az: number,
  levelFrac: number,
  motion: number,
): number {
  return motion < 1
    ? (1 - motion) *
        (lv.rMax - rMaxNow) *
        lvProfileG(lv.shape, levelFrac) *
        ellipseFactor(lv.shape.ratio, az)
    : 0;
}

/**
 * The share F of the systolic excursion that regional abnormalities withhold, for the compensation of the normal walls
 * (computeChamberPose: R²·(1 − F) + F·R_ED² = R0², which keeps the cavity volume on the tables). With the offset of
 * `regionalCavityOffset` the cavity's radius at (az, ζ) is e(az)·g(ζ)·(motion·R + (1 − motion)·R_ED), so the formula
 * holds with F the mean of 1 − motion weighted by the area of the cross-section there, e²·g² (g the profile, e the
 * ellipseFactor), exactly where the motion is 0 or 1 and to first order in R_ED − R between: F is integrated over the
 * same field the classifier draws. It was the fraction of the cavity surface by segment (basal and mid 1/18 each, apical
 * 1/12), which counted the narrow apical third like the basal and mid ones and could not see the borders. The cap (17),
 * beyond the end of the cavity, holds none of it (g = 0 there). 0 exactly when the field is 1 everywhere.
 */
export function regionalMeanFraction(amp: SegmentAmplitudes, shape: LvShape): number {
  let normal = true;
  for (let seg = 1; seg <= 17; seg++) if (segmentMotion(amp, seg) < 1) normal = false;
  if (normal) return 0;
  // midpoint rule over 1° of azimuth and 1/150 of the length, much finer than the borders
  const NA = 360,
    NZ = 150;
  let withheld = 0,
    area = 0;
  for (let i = 0; i < NZ; i++) {
    const zeta = (i + 0.5) / NZ;
    const g2 = lvProfileG(shape, zeta) ** 2;
    for (let j = 0; j < NA; j++) {
      const az = (2 * Math.PI * (j + 0.5)) / NA - Math.PI;
      const w = g2 * ellipseFactor(shape.ratio, az) ** 2;
      withheld += (1 - lvRegionalMotion(amp, az, zeta)) * w;
      area += w;
    }
  }
  return withheld / area;
}
