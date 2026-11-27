import { makeSample, Structure, Tissue } from '../tissue';
import { sdCapsule, sdEllipsoid, smax, smin } from '../sdf';
import { ctx, setSample, type ClassifyCtx } from './context';
import { AORTIC_ROOT_WALL_CM } from '../aorticValve';
import {
  DESC_AORTA_MAX_OUTER_R,
  ESOPHAGUS_RX,
  ESOPHAGUS_RZ,
  ESOPHAGUS_Z,
  SPINE_R,
  SPINE_Z,
} from '../thoraxModel';
import { pulmonaryTrunkEpiDistance, trunkFrontDistance } from '../greatVessels';
import { rvCrescent, rvInflowSdf } from '../rv';
import { rootCoordinates } from './root';
import { classifyLeftVentricle } from './leftVentricle';
import { rvFreeWallNow } from './rightVentricle';
import { leftAppendageDistance, rightAppendageLumen } from '../atrialAppendages';
import type { AnchorsCached } from '../anchors';
import type { HeartModel } from '../heartModel';
import type { ChamberPose } from '../heartPose';

/** Left atrial wall (cm) and the appendage's. */
/**
 * The mediastinal fat in front of the pulmonary trunk that the lung does not cross (cm; decision 336), straight in front
 * of the trunk and less as the point turns to its sides.
 */
export const PA_ANTERIOR_FAT_CM = 1.0;
export const LA_WALL_CM = 0.25;
export const LAA_WALL_CM = 0.18;
/** Thickness (cm) of the pericardium the sac draws outside the epicardium. */
export const PERICARDIUM_CM = 0.12;
/**
 * How far (cm) above the mitral hinge the left ventricle's epicardium reaches for the sac (decision 380), and from how far
 * (cm) beyond the aortic root's wall it ends there: around the root its cap stays, the fill between the root, the atria
 * and the ventricular base.
 */
export const LV_SAC_ABOVE_HINGE_CM = 0.3;
export const LV_SAC_ROOT_KEEP_CM = 1.2;
/**
 * The blend (cm) of the ventricles' epicardium with the atria's, which fills the atrioventricular groove with fat, and how
 * it narrows from AV_GROOVE_BLEND_TOP_CM above the mitral hinge over AV_GROOVE_BLEND_FADE_CM to AV_GROOVE_BLEND_MIN_CM
 * (decision 387). Above the groove the left ventricle's cap kept beside the root (decision 380) still lay 1-2 mm from the
 * left atrium's wall, and the full blend filled the space between them with a band of fat 1-2 mm thick that the
 * five-chamber view drew under the atrium.
 */
export const AV_GROOVE_BLEND_CM = 0.8;
export const AV_GROOVE_BLEND_TOP_CM = 1;
export const AV_GROOVE_BLEND_FADE_CM = 0.5;
export const AV_GROOVE_BLEND_MIN_CM = 0.2;
/** The blend of the atrioventricular groove `h` cm above the mitral hinge (decision 387). Shared with GLSL. */
export function avGrooveBlend(h: number): number {
  const t = Math.min(1, Math.max(0, (h - AV_GROOVE_BLEND_TOP_CM) / AV_GROOVE_BLEND_FADE_CM));
  return Math.max(AV_GROOVE_BLEND_MIN_CM, AV_GROOVE_BLEND_CM * (1 - t));
}

/**
 * Distance (cm) from a heart-frame point to the surface of the posterior column (decision 273): the descending aorta at
 * its largest systolic size, the vertebral body, and the esophagus in front of the spine (`ESOPHAGUS_Z`, `ESOPHAGUS_RX`,
 * `ESOPHAGUS_RZ` in `thoraxModel.ts`). `u` is torso +y (vertical), `a` and `s` lie on the descending-aorta and spine
 * axes, and `pn` is `laPostN` (torso −z, posterior). Keeping the esophagus in the column prevents a large pericardial
 * effusion from invading the esophagus behind the left atrium. Mirrored in `glslGenerated.ts`.
 */
export function posteriorColumnDistance(
  x: number,
  y: number,
  z: number,
  ux: number,
  uy: number,
  uz: number,
  ax: number,
  ay: number,
  az: number,
  sx: number,
  sy: number,
  sz: number,
  pnx: number,
  pny: number,
  pnz: number,
): number {
  const dax = x - ax,
    day = y - ay,
    daz = z - az;
  const ta = dax * ux + day * uy + daz * uz;
  const dA =
    Math.sqrt(Math.max(0, dax * dax + day * day + daz * daz - ta * ta)) - DESC_AORTA_MAX_OUTER_R;
  const dsx = x - sx,
    dsy = y - sy,
    dsz = z - sz;
  const ts = dsx * ux + dsy * uy + dsz * uz;
  const dS = Math.sqrt(Math.max(0, dsx * dsx + dsy * dsy + dsz * dsz - ts * ts)) - SPINE_R;
  // Esophagus elliptic cylinder along u, centred (ESOPHAGUS_Z − SPINE_Z) anterior (−pn) of the spine (ESOPHAGUS_X = 0)
  const ezOff = ESOPHAGUS_Z - SPINE_Z;
  const dex = x - (sx - ezOff * pnx),
    dey = y - (sy - ezOff * pny),
    dez = z - (sz - ezOff * pnz);
  const te = dex * ux + dey * uy + dez * uz;
  const pex = dex - te * ux,
    pey = dey - te * uy,
    pez = dez - te * uz;
  const ezProj = pex * pnx + pey * pny + pez * pnz;
  const ex2 = Math.max(0, pex * pex + pey * pey + pez * pez - ezProj * ezProj);
  const nx2 = ex2 / (ESOPHAGUS_RX * ESOPHAGUS_RX);
  const nz2 = (ezProj * ezProj) / (ESOPHAGUS_RZ * ESOPHAGUS_RZ);
  const k0 = Math.sqrt(nx2 + nz2);
  const k1 = Math.sqrt(
    nx2 / (ESOPHAGUS_RX * ESOPHAGUS_RX) + nz2 / (ESOPHAGUS_RZ * ESOPHAGUS_RZ),
  );
  const dE = k0 > 1e-6 ? (k0 * (k0 - 1)) / k1 : -ESOPHAGUS_RZ;
  return Math.min(dA, Math.min(dS, dE));
}

/** Effusion (cm) the sac holds over the left atrium (decision 255). */
export const OBLIQUE_SINUS_EFFUSION_CM = 0.3;
/** The oblique sinus behind the left atrium along torso posterior (`laPostN`): the fluid thins from its centre plane
 * to OBLIQUE_SINUS_BEHIND behind it, within the atrium's posterior transverse footprint (normalised radius squared
 * from OBLIQUE_SINUS_FOOTPRINT_FROM to OBLIQUE_SINUS_FOOTPRINT_TO). */
export const OBLIQUE_SINUS_BEHIND = 0.6;
export const OBLIQUE_SINUS_FOOTPRINT_FROM = 0.6;
export const OBLIQUE_SINUS_FOOTPRINT_TO = 2.2;
/** Blend (cm) of the parietal layer against the descending aorta, esophagus and vertebral body (decision 383). */
export const PARIETAL_COLUMN_BLEND_CM = 0.2;

/**
 * Signed distance (cm) to the inner face of the still parietal pericardium over an effusion (decision 383):
 * the pre-tabulated still sac `dSac` (which already incorporates both the full effusion and the oblique-sinus
 * thinning behind the left atrium, clearing the full beating+swinging heart envelope across the beat) held off
 * the posterior column (`colDist`: descending aorta, esophagus and vertebral body). Because `dSac` and `colDist`
 * are both fixed in the torso, the parietal sheet stays completely still over the oblique sinus as well as the
 * ventricles, and its outer face (`par = PERICARDIUM_CM`) stays tangent to `colDist = 0` without cutting inside
 * the posterior left atrial wall. Mirrored in `glslGenerated.ts`.
 */
export function parietalSdf(dSac: number, colDist: number): number {
  return smax(dSac, PERICARDIUM_CM - colDist, PARIETAL_COLUMN_BLEND_CM);
}

/**
 * The fibrous pericardium over an effusion is a sac that does not beat: it is tied to the great vessels, the sternum and
 * the diaphragm and barely stretches over a beat, and the heart contracts and swings inside it. The fluid is what lies
 * between the beating epicardium and that still sheet, so its thickness changes with the phase, the parietal line of the
 * M-mode is flat and the swinging heart of tamponade moves inside a sac that stays. The sheet is the end-diastolic
 * heart's envelope pushed out by the declared effusion (decision 383), at the heart's largest, which is when an effusion
 * is measured, with the atria's appendages at theirs (heartEnvelope). The sac used to be rebuilt every sample from the
 * epicardium of the moment and moved with the swing: in the tamponade case the fluid was 2.20 cm thick in every phase on
 * fixed rays from the left ventricle's axis, and the sac's outer face travelled 0.27 cm on the inferior ray and 0.96 cm
 * on the lateral one over the beat. Now the face does not move, and the fluid goes from 2.19 cm at end-diastole to 2.44
 * cm at end-systole on the inferior ray and from 1.88 to 2.84 cm on the lateral one as the heart swings toward it and
 * away. Over the oblique sinus alone, where the layer is reflected onto the left atrium, it rests on the heart and moves
 * with it (`parietalSdf`).
 *
 * The sac is tabulated once per case (anchors.ts) as its radius from the centre of the heart's bounding sphere along
 * PARIETAL_AZ_BINS azimuths (around z, from +x toward +y) by PARIETAL_POLAR_BINS polar angles (from +z, the apex, to −z,
 * poles included), and read bilinearly at the point before the swing. In the tamponade case the full-effusion sac is
 * star-shaped from that centre but for one direction that grazes a 0.2 mm crease; the table puts the parietal within
 * 0.6 mm of the analytic surface on the rays from the left ventricle, 1.0-1.3 mm outside it behind the left atrium (the
 * oblique sinus holds 0.4 cm instead of 0.3), and within 0.2-0.5 mm (median) and 4 mm (worst) around the great vessels,
 * where the envelope turns fastest: 1.5 % of the sac's 1 800 mL changes side. The column and the oblique sinus
 * (`parietalSdf`) are applied to it, the sinus resting on the heart of the moment. A read costs about 50 ns on the CPU
 * against the 20 ns of the terms it replaces, within the noise of the 1.8-1.9 µs the classifier takes per sample in the
 * tamponade case.
 */
export const PARIETAL_AZ_BINS = 48;
export const PARIETAL_POLAR_BINS = 25;
export interface ParietalSac {
  /** Centre (heart frame, cm) the directions leave from: the heart's bounding-sphere centre. */
  cx: number;
  cy: number;
  cz: number;
  /** Radius (cm) of the full-effusion sac per direction, `r[j·PARIETAL_AZ_BINS + i]` for polar row j and azimuth i. */
  r: Float64Array;
}

const sacSample = makeSample();
/**
 * The heart's smooth envelope (cm, negative inside) the sac is pushed out from (decision 383): the left ventricle's
 * epicardium, the right ventricle's free wall over its crescent and inflow, the atria at their largest sizes with their
 * appendages, and the outflow tract and the trunk where the pericardium is anchored to them, smoothly united, the
 * ventricles in pose `hp` and the appendages at end-systole. It runs the left ventricle's block for the epicardium the
 * block leaves in the context, whatever it classifies.
 *
 * The atria's bodies are their anchor ellipsoids over their largest sizes: the outer faces their blocks draw are
 * distances only near the wall (their clipping planes and the walls rising from the hinges are bounds of it), and their
 * level set the effusion away lay up to 3.1 cm beyond the true one in the tamponade case, past the right ventricle's
 * inferior wall, where no atrium is. The appendages are a capsule and a round cone, distances at any range, at their largest and
 * without the clips that keep the right one off the atrial floor and the aortic root (they only take from it): their
 * radial scales (atrialScale, appendageScale) are 1 at end-systole and less in every other phase, and the left one hangs
 * lowest then, the atrium at its longest. Without the right one, which reaches past the right side of the aortic root,
 * 0.11-0.90 mL of it, of the right atrium's wall and of the visceral layer over them lay past the inner face of the
 * tamponade case's parietal layer, up to 0.61 cm, in every phase and in five of the six hearts the app's postures and
 * breathing build; none does now, and the sac moves out by up to 2.9 cm (0.07 cm on average) around the appendage. The
 * left one lies within its body's envelope: it moves the sac 1 mm at most.
 */
function heartEnvelope(
  m: HeartModel,
  A: AnchorsCached,
  hp: ChamberPose,
  x: number,
  y: number,
  z: number,
): number {
  const c = ctx;
  c.m = m;
  c.hp = hp;
  c.A = A;
  c.x = x;
  c.y = y;
  c.z = z;
  c.out = sacSample;
  rootCoordinates(c);
  classifyLeftVentricle(c);
  rvCrescent(m, hp, A, x, y, z, c.az, c.rvSdf);
  const fw = rvFreeWallNow(m.anatomy.rv.freeWallThicknessCm, hp.state.contraction);
  const dLvEpi = c.dEllR - c.wallT - c.crestLoss;
  const dRvEpi = smin(c.rvSdf[0]!, rvInflowSdf(x, y, z, hp.valves.tv, hp.tvZ, c.rvSdf[2]!), 0.3) - fw;
  const la = A.laCenter,
    lr = A.laR,
    ra = A.raCenter,
    rar = A.raR;
  // the left atrium's height centre as its block places it (classify/atria.ts) at end-systole, the annulus down the whole
  // MAPSE then (zAnn = mapseCm·longitudinal, longitudinal 1): read from the model, no end-systolic pose built for it
  const czL = (la.z - lr.z + m.physiology.mapseCm + 0.25) / 2;
  const dAppendages = smin(
    leftAppendageDistance(la, lr, czL, 1, 0, x, y, z) - LAA_WALL_CM,
    rightAppendageLumen(A.raaBase, A.avCenter, 1, x, y, z) - LAA_WALL_CM,
    0.8,
  );
  return smin(
    smin(
      smin(dLvEpi, dRvEpi, 0.8),
      smin(
        smin(
          sdEllipsoid(x, y, z, la.x, la.y, la.z, lr.x + 0.25, lr.y + 0.25, lr.z + 0.25),
          sdEllipsoid(x, y, z, ra.x, ra.y, ra.z, rar.x + 0.22, rar.y + 0.22, rar.z + 0.22),
          0.8,
        ),
        dAppendages,
        0.8,
      ),
      0.8,
    ),
    smin(rvotEpiDistance(A, fw, x, y, z), pulmonaryTrunkEpiDistance(A, x, y, z), 0.8),
    0.8,
  );
}

/**
 * The sac around the outflow tract and the trunk stays where the pericardium is anchored (sternopericardial ligaments in
 * front, the arterial reflection on the trunk) while they descend in systole (decision 111). One envelope stands for the
 * epicardial fat, the pericardium and the effusion here; moved with the tract, the effusion of the tamponade case, which
 * reaches the transducer face in the parasternal views, changed its near field with every beat.
 */
function rvotEpiDistance(A: AnchorsCached, fw: number, x: number, y: number, z: number): number {
  return sdCapsule(
    x,
    y,
    z,
    A.rvotA.x,
    A.rvotA.y,
    A.rvotA.z,
    A.rvotB.x,
    A.rvotB.y,
    A.rvotB.z,
    A.rvotRa + fw,
  );
}

/**
 * Smooth thinning weight 0..1 of the oblique sinus and superior posterior pericardial reflection behind the left
 * atrium along torso posterior (`A.laPostN`). Decomposing the transverse offset into superior (`A.colU`) and
 * left-right components keeps the full effusion around the inferior/lateral ventricles and right atrium while
 * tapering smoothly up behind the left atrial roof and aortic root where the pericardium reflects onto the
 * pulmonary veins and great vessels.
 */
export function obliqueSinusWeight(x0: number, y: number, z: number, A: AnchorsCached): number {
  const lc = A.laCentroid;
  const lr = A.laR;
  const pn = A.laPostN;
  const u = A.colU;
  const dx = x0 - lc.x;
  const dy = y - lc.y;
  const dz = z - lc.z;
  const post = dx * pn.x + dy * pn.y + dz * pn.z;
  if (post <= 0) return 0;
  const rPost = Math.hypot(pn.x * lr.x, pn.y * lr.y, pn.z * lr.z);
  const behind = post / Math.max(0.5, rPost);
  // Project transverse vector onto torso superior (u) and left-right (perpendicular to both pn and u).
  // Keep thinning active superiorly behind the LA roof / arterial reflection, while tapering inferiorly at the AV groove.
  const sup = dx * u.x + dy * u.y + dz * u.z;
  const lat2 = Math.max(0, dx * dx + dy * dy + dz * dz - post * post - sup * sup);
  const supNeg = Math.min(0, sup);
  const supExcess = Math.max(0, sup - 2.6);
  const rLat = Math.max(1.2, lr.x);
  const rInf = Math.max(1.0, lr.z);
  const footprint =
    lat2 / (rLat * rLat) + (supNeg * supNeg + supExcess * supExcess) / (rInf * rInf);
  const tb = Math.min(1, Math.max(0, behind / OBLIQUE_SINUS_BEHIND));
  const tf = Math.min(
    1,
    Math.max(
      0,
      (footprint - OBLIQUE_SINUS_FOOTPRINT_FROM) /
        (OBLIQUE_SINUS_FOOTPRINT_TO - OBLIQUE_SINUS_FOOTPRINT_FROM),
    ),
  );
  return tb * tb * (3 - 2 * tb) * (1 - tf * tf * (3 - 2 * tf));
}

/**
 * Tabulate the still parietal pericardial sac around the heart once per case with an effusion:
 * around the ventricles and free walls it holds the full effusion (`heartEnvelope` at ED + effusion);
 * over the posterior left atrium (oblique sinus, along `A.laPostN`) it blends the sac radius smoothly
 * between the full effusion and the oblique-sinus radius (`OBLIQUE_SINUS_EFFUSION_CM`), staying clear
 * of the beating and swinging heart across all `poses` (ED, ES, and swing extremes). Blending the radii
 * directly with `obliqueSinusWeight` evaluated on the heart surface avoids any radial-cliff discontinuities
 * in the 48×25 spherical table while keeping the parietal sheet 100% still over the beat.
 */
export function buildParietalSac(
  m: HeartModel,
  A: AnchorsCached,
  ed: ChamberPose,
  poses: readonly ChamberPose[] = [ed],
  epicardiumAt?: (hp: ChamberPose, x0: number, y: number, z: number) => number,
): ParietalSac {
  const eff = m.anatomy.pericardium.effusionCm;
  const thin = Math.min(eff, OBLIQUE_SINUS_EFFUSION_CM);
  const reachFull = PERICARDIUM_CM + eff;
  const reachThin = PERICARDIUM_CM + thin;
  // Margin over the bilinear 48×25 table's cell slack so the interpolated still face keeps the full sinus film
  const sinusFilm = PERICARDIUM_CM + thin + 0.12;
  const C = m.boundCenter;
  const r = new Float64Array(PARIETAL_AZ_BINS * PARIETAL_POLAR_BINS);
  const envAt = (dx: number, dy: number, dz: number, t: number): number =>
    heartEnvelope(m, A, ed, C.x + dx * t, C.y + dy * t, C.z + dz * t);
  const heartAt = (dx: number, dy: number, dz: number, t: number): number => {
    const px = C.x + dx * t,
      py = C.y + dy * t,
      pz = C.z + dz * t;
    let dMin = Infinity;
    for (let k = 0; k < poses.length; k++) {
      const d = epicardiumAt!(poses[k]!, px, py, pz);
      if (d < dMin) dMin = d;
    }
    return dMin;
  };
  const findRoot = (
    fn: (t: number) => number,
    tStart: number,
    tMax: number,
  ): number => {
    let t = tStart,
      step = 0,
      f = fn(t);
    while (f < 0 && t < tMax) {
      step = Math.max(0.5 * -f, 0.02);
      t += step;
      f = fn(t);
    }
    let lo = Math.max(tStart, t - step),
      hi = Math.min(tMax, t);
    for (let k = 0; k < 10; k++) {
      const mid = 0.5 * (lo + hi);
      if (fn(mid) < 0) lo = mid;
      else hi = mid;
    }
    return 0.5 * (lo + hi);
  };
  for (let j = 0; j < PARIETAL_POLAR_BINS; j++) {
    const phi = (Math.PI * j) / (PARIETAL_POLAR_BINS - 1);
    const pole = j === 0 || j === PARIETAL_POLAR_BINS - 1;
    for (let i = 0; i < PARIETAL_AZ_BINS; i++) {
      // the poles are one direction for every azimuth
      if (pole && i > 0) {
        r[j * PARIETAL_AZ_BINS + i] = r[j * PARIETAL_AZ_BINS]!;
        continue;
      }
      const th = (2 * Math.PI * i) / PARIETAL_AZ_BINS;
      const dx = Math.sin(phi) * Math.cos(th),
        dy = Math.sin(phi) * Math.sin(th),
        dz = Math.cos(phi);
      const rThinEnv = findRoot((t) => envAt(dx, dy, dz, t) - reachThin, 0, m.boundRadius);
      const rFull = findRoot((t) => envAt(dx, dy, dz, t) - reachFull, rThinEnv, m.boundRadius);
      const w = epicardiumAt
        ? obliqueSinusWeight(C.x + dx * rThinEnv, C.y + dy * rThinEnv, C.z + dz * rThinEnv, A)
        : 0;
      if (w <= 0) {
        r[j * PARIETAL_AZ_BINS + i] = rFull;
        continue;
      }
      const rSinus = findRoot(
        (t) => heartAt(dx, dy, dz, t) - sinusFilm,
        rThinEnv,
        rFull,
      );
      r[j * PARIETAL_AZ_BINS + i] = rFull - (rFull - rSinus) * w;
    }
  }
  return { cx: C.x, cy: C.y, cz: C.z, r };
}

/**
 * Signed distance (cm, negative inside) to the full-effusion sac at a point before the swing: the distance from the
 * centre less the bilinearly read radius of its direction, tilted by the slope of the table (first order, as the left
 * ventricle's polar profile, lvShape.ts). Mirrored by `parietalSacDistance` in glslHeartChambers.ts.
 */
export function parietalSacDistance(sac: ParietalSac, x: number, y: number, z: number): number {
  const dx = x - sac.cx,
    dy = y - sac.cy,
    dz = z - sac.cz;
  const rho = Math.sqrt(dx * dx + dy * dy);
  const rad = Math.sqrt(rho * rho + dz * dz);
  const th = rho > 1e-6 ? Math.atan2(dy, dx) : 0;
  const phi = Math.atan2(rho, dz);
  const fi = ((th < 0 ? th + 2 * Math.PI : th) * PARIETAL_AZ_BINS) / (2 * Math.PI);
  const fj = (phi * (PARIETAL_POLAR_BINS - 1)) / Math.PI;
  const i0 = Math.min(PARIETAL_AZ_BINS - 1, Math.floor(fi));
  const i1 = i0 === PARIETAL_AZ_BINS - 1 ? 0 : i0 + 1;
  const j0 = Math.min(PARIETAL_POLAR_BINS - 2, Math.floor(fj));
  const u = fi - i0,
    v = fj - j0;
  const T = sac.r,
    row0 = j0 * PARIETAL_AZ_BINS,
    row1 = row0 + PARIETAL_AZ_BINS;
  const r00 = T[row0 + i0]!,
    r10 = T[row0 + i1]!,
    r01 = T[row1 + i0]!,
    r11 = T[row1 + i1]!;
  const r0 = r00 + (r10 - r00) * u,
    r1 = r01 + (r11 - r01) * u;
  const R = r0 + (r1 - r0) * v;
  // the slope along the azimuth and the polar angle, per unit length of the surface
  const sa =
    ((r10 - r00 + (r11 - r01 - r10 + r00) * v) * PARIETAL_AZ_BINS) /
    (2 * Math.PI) /
    (R * Math.max(rho / Math.max(rad, 1e-6), 1e-3));
  const sp = ((r1 - r0) * (PARIETAL_POLAR_BINS - 1)) / Math.PI / R;
  return (rad - R) / Math.sqrt(1 + Math.min(9, sa * sa + sp * sp));
}

/** Pericardium & effusion: the outer envelope of all epicardial surfaces. True when the point is in the sac. */
export function classifyPericardium(c: ClassifyCtx): boolean {
  const { m, hp, A, x, y, z, out, dEllR, wallT, nx0, ny0, nz0, rvSdf, raSleeve, raBody } = c;
  // the epicardium is the outer face of the wall shell; the sac keeps the septum's full thickness where its crest narrows
  // (decision 223), so no hole opens between the root, the crest and the atrium
  const dLvEpi = dEllR - wallT - c.crestLoss;
  // the ventricle's epicardium ends at its base (decision 380): the profile it is the outer face of closes 2-2.5 cm above
  // the mitral hinge, and where no atrium covered that cap the sac held it as fat, 1-1.4 cm thick in the left
  // atrioventricular groove beside the atrium, 3-8 cm³ in the hypertrophic case; around the root the cap stays
  const rootGap = c.rootT > -90 ? c.rootRr - c.rootR : 1e3;
  const dLvSac = smax(
    dLvEpi,
    Math.min(c.zHinge - LV_SAC_ABOVE_HINGE_CM - z, rootGap - LV_SAC_ROOT_KEEP_CM),
    0.2,
  );
  // The right ventricle's epicardium is the outer face of the wall its block draws now (rightVentricle.ts): around the
  // cavity with the outflow tract, which descends with the base, and as thick as the wall is, 35 % more in systole. With
  // the crescent alone and the end-diastolic thickness, the wall covered the visceral pericardium from a contraction of
  // 0.86 in the normal case and of 0.49 with pulmonary hypertension and stood out of the sac in end-systole, on 15 and on
  // 20 of 20 rays through the free wall; over ten phases of the twelve cases 44-391 of ~690 such rays per case met the
  // lung or the fluid. None do now.
  const fw = m.anatomy.rv.freeWallThicknessCm;
  const dRvEpi = rvSdf[0]! - rvFreeWallNow(fw, hp.state.contraction); // the RV block's cavity, computed just before
  // the atrium's own outer face (decision 273): a fixed ellipsoid 0.25 cm over its largest size, it left up to 1.7 cm of
  // fat behind the posterior wall, where the cavity is flattened and, at end-diastole, smaller
  const dLaEpi = c.laEpi;
  // the right atrium's outer face is its own wall's (decision 378), as the left's is: the anchor ellipsoid 0.22 cm over its
  // largest size laid 8-10 cm³ of fat over its free walls, a layer 4 mm thick at the median and up to 1.6 cm, that the
  // subcostal view showed as a band around the atrium
  const dRaEpi = Math.min(
    raSleeve + 0.22, // the lengthened atrium over the vacated base (decision 133)
    raBody - 0.22, // the outer face of its body's wall, with the lateral wall over the tricuspid hinge (decision 367)
    c.raaEpi, // and its appendage (decision 323)
    c.raColumn - 0.22, // and the vestibule's column over the ring (decision 379)
  );
  const dRvotEpi = rvotEpiDistance(A, fw, x, y, z);
  const dPaEpi = pulmonaryTrunkEpiDistance(A, x, y, z);
  // the cardiac silhouette is the smooth union of the epicardial surfaces: the grooves between chambers and
  // the space between outflow and root are filled with epicardial fat, and one pericardium wraps the whole heart
  // the sac lies against the descending aorta, esophagus and vertebral body and does not run into them (decision 273)
  const rawEpi = smin(
    smin(smin(dLvSac, dRvEpi, 0.8), smin(dLaEpi, dRaEpi, 0.8), avGrooveBlend(c.zHinge - z)),
    smin(dRvotEpi, dPaEpi, 0.8),
    0.8,
  );
  c.rawEpi = rawEpi;
  const dEpi = smax(rawEpi, PERICARDIUM_CM - c.colDist, 0.2);
  // the parietal layer over an effusion is the fixed sac (ParietalSac), read at the point before the tamponade swing: the
  // heart beats and swings inside it, including over the oblique sinus where the still sac clears all cycle extremes.
  const sac = A.parietal;
  const x0 = x + hp.swingX;
  const par =
    hp.effusion > 0 && sac ? parietalSdf(parietalSacDistance(sac, x0, y, z), c.colDist) : 1e3;
  c.parietal = par;
  if (dEpi < 0) {
    setSample(out, Tissue.Fat, dEpi, nx0, ny0, nz0, x, y, z, 0, Structure.EpicardialFat);
    return true;
  }
  if (dEpi < PERICARDIUM_CM) {
    setSample(
      out,
      Tissue.Pericardium,
      -Math.min(Math.max(dEpi, 0), 0.12 - Math.max(dEpi, 0)),
      nx0,
      ny0,
      nz0,
      x,
      y,
      z,
      0,
      Structure.Pericardium,
    );
    return true;
  }
  if (par < 0) {
    setSample(
      out,
      Tissue.Fluid,
      Math.max(par, PERICARDIUM_CM - dEpi),
      nx0,
      ny0,
      nz0,
      x,
      y,
      z,
      0,
      Structure.PericardialEffusion,
    );
    return true;
  }
  if (par < PERICARDIUM_CM) {
    setSample(out, Tissue.Pericardium, 0, nx0, ny0, nz0, x, y, z, 0, Structure.Pericardium);
    return true;
  }
  // outside the sac: how far beyond the parietal pericardium, or beyond the wall of the ascending aorta, which leaves
  // the sac — what the thorax classifier needs to wrap the lungs around the heart (decision 144)
  const dSac = Math.min(dEpi, par) - PERICARDIUM_CM;
  const dRoot = c.rootT > -90 ? c.rootRr - c.rootR - AORTIC_ROOT_WALL_CM : dSac;
  // in front of the pulmonary trunk the mediastinum's fat stands between the trunk and the lung, as far as the point lies
  // in front of the trunk's middle, the torso's anterior direction read in the heart frame (decision 336)
  const dTrunkFront = trunkFrontDistance(m, A, x, y, z, dPaEpi);
  out.sdf = Math.min(dSac, dRoot, dTrunkFront);
  return false;
}
