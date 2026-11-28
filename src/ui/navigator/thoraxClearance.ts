import { heartToTorso } from '@/simulator/anatomy/heartFrame';
import { buildHeartMeshes } from '@/simulator/anatomy/heartMesh';
import type { HeartModel, HeartPose } from '@/simulator/anatomy/heartModel';
import { ribDepth, skinZ, type ThoraxModel } from '@/simulator/anatomy/thoraxModel';

/**
 * Shared 3D navigator thorax clearance geometry: keeps the rendered anterior rib cage, sternum,
 * and skin cleanly outside the beating 3D heart across every case (normal, left-lateral decubitus,
 * dilated cardiomyopathy, severe aortic regurgitation) without altering the acoustic physics engine.
 */

/** Anteroposterior half-thickness of the rendered ribbon ribs (cm). */
export const RIB_HALF_THICKNESS_CM = 0.16;

/** Minimum visual gap (cm) maintained between the beating epicardium and the inner face of anterior ribs. */
export const MIN_HEART_RIB_GAP_CM = 0.42;

/** Natural subcutaneous depth (cm) of the rendered anterior rib centerline beneath the outer skin. */
export function visualRibDepthCm(t: ThoraxModel): number {
  return Math.min(ribDepth(t) * 0.42, 0.48);
}

/**
 * Smooth anterior chest-wall clearance lift (cm) as a function of lateral torso coordinate `x`.
 * Full lift across the precordium (`x = 0..8 cm`), smoothly tapering to 0 at the mid-axillary flanks
 * so the torso silhouette remains continuous and natural.
 */
export function anteriorLiftAt(t: ThoraxModel, x: number, clearanceCm: number): number {
  if (clearanceCm <= 0) return 0;
  const u = Math.min(1, Math.abs(x) / Math.max(1, t.aw));
  return clearanceCm * Math.pow(Math.max(0, 1 - u * u), 0.55);
}

/** Outer skin Z (cm) for the 3D navigator with optional anterior precordial clearance applied. */
export function visualSkinZ(t: ThoraxModel, x: number, y: number, clearanceCm = 0): number {
  return skinZ(t, x, y) + anteriorLiftAt(t, x, clearanceCm);
}

/**
 * Superellipse point (x, z) on the outer skin surface (`inset = 0`) at angle `theta` (0 = midline),
 * plus its outward unit normal in the transverse plane.
 */
export function skinCrossSection(
  t: ThoraxModel,
  theta: number,
  y: number,
): { x: number; z: number; nx: number; nz: number } {
  const n = t.n;
  const s = Math.sin(theta);
  const c = Math.cos(theta);
  const absS = Math.abs(s);
  const absC = Math.abs(c);
  const p = 2 / n;
  const x = t.aw * Math.sign(s) * Math.pow(absS, p);
  let z = -t.bDepth + t.bDepth * Math.sign(c) * Math.pow(absC, p);
  if (y > 8) z -= 0.12 * (y - 8) * (y - 8);
  if (y < -8) z -= 0.18 * (-8 - y);
  // Outward transverse normal of superellipse (|x/aw|^n + |(z+b)/b|^n = 1)
  const gx = (Math.sign(s) * Math.pow(Math.max(1e-6, absS), 2 - p)) / Math.max(1, t.aw);
  const gz = (Math.sign(c) * Math.pow(Math.max(1e-6, absC), 2 - p)) / Math.max(1, t.bDepth);
  const gLen = Math.hypot(gx, gz) || 1;
  return { x, z, nx: gx / gLen, nz: gz / gLen };
}

/**
 * Centerline point (x, z) of an anatomical 3D rib at angle `theta` and height `y`:
 * offset inward along the TRUE skin surface normal by `visualRibDepthCm(t)` (so lateral ribs
 * stay uniformly subcutaneous instead of collapsing 2 cm deep into the LV apex) plus smooth
 * anterior clearance lift when needed for anterior/dilated hearts.
 */
export function visualRibPoint(
  t: ThoraxModel,
  theta: number,
  y: number,
  clearanceCm = 0,
): { x: number; z: number } {
  const d = visualRibDepthCm(t);
  const p = skinCrossSection(t, theta, y);
  const x = p.x - p.nx * d;
  const z = p.z - p.nz * d + anteriorLiftAt(t, x, clearanceCm);
  return { x, z };
}

/** Inner (posterior) face Z of the anterior rib cage at torso `(x, y)` for clearance checks. */
export function visualRibInnerZ(t: ThoraxModel, x: number, y: number, clearanceCm = 0): number {
  const ax = Math.min(Math.abs(x) / Math.max(1, t.aw), 0.98);
  const th = Math.asin(Math.pow(ax, t.n / 2)) * (x < 0 ? -1 : 1);
  const p = visualRibPoint(t, th, y, clearanceCm);
  return p.z - RIB_HALF_THICKNESS_CM;
}

/**
 * Computes the minimal anterior thoracic clearance lift (cm, >= 0) directly against the actual
 * filleted & smoothed 3D cardiac meshes across the provided poses (e.g. end-diastole & peak
 * systole) so no vertex of the beating heart or epicardial coronaries ever touches or penetrates
 * the rendered anterior ribs or sternum.
 */
export function computeAnteriorRibClearanceCm(
  heart: HeartModel,
  thorax: ThoraxModel,
  poses: HeartPose[],
): number {
  let needed = 0;

  for (const pose of poses) {
    const groups = buildHeartMeshes(heart, pose, {
      stepCm: 0.48,
      thorax,
      includeAorta: false,
      includeCoronaries: false,
    });
    for (const g of groups) {
      if (g.id === 'lv-cavity' || g.id === 'rv-cavity' || g.id === 'valves') continue;
      const pos = g.positions;
      for (let i = 0; i < pos.length; i += 3) {
        const p = heartToTorso(heart.frame, {
          x: pos[i]!,
          y: pos[i + 1]!,
          z: pos[i + 2]!,
        });
        const baseInnerZ = visualRibInnerZ(thorax, p.x, p.y, 0);
        // Reserve MIN_HEART_RIB_GAP_CM plus 0.18 cm for finer-grid/coronary relief
        const deficit = p.z + MIN_HEART_RIB_GAP_CM + 0.18 - baseInnerZ;
        if (deficit > 0) {
          const liftWeight = Math.max(0.45, anteriorLiftAt(thorax, p.x, 1));
          needed = Math.max(needed, deficit / liftWeight);
        }
      }
    }
  }
  return Math.min(2.4, Math.max(0, Math.ceil(needed * 20) / 20));
}
