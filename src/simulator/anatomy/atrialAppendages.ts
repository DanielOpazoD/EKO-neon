import { sdCapsule, sdRoundCone } from './sdf';

/**
 * The atrial appendages' geometry, read by the atria's classifier (classify/atria.ts, classify/atriaVessels.ts) and by
 * the pericardial sac, which holds them at their largest (classify/pericardium.ts). A leaf module: the sac's module is
 * imported by the atria's, which read its wall thicknesses when they load, so it cannot import theirs.
 */

interface Point3 {
  x: number;
  y: number;
  z: number;
}

/** The right atrial appendage (decisions 323, 388, 389): its tip from the aortic valve's centre and its radii (cm). */
export const RAA_TIP_FROM_AV: readonly [number, number, number] = [-2.7, 1.6, -2.4];
export const RAA_BASE_R_CM = 1.1;
export const RAA_TIP_R_CM = 1;

/**
 * Distance (cm) to the right atrial appendage's lumen at radial scale `bo` (decision 323): a round cone from its base on
 * the atrium (`base`, an anchor) to its tip beside the aortic root, placed from the valve's centre `av`.
 */
export function rightAppendageLumen(
  base: Point3,
  av: Point3,
  bo: number,
  x: number,
  y: number,
  z: number,
): number {
  return sdRoundCone(
    x,
    y,
    z,
    base.x,
    base.y,
    base.z,
    av.x + RAA_TIP_FROM_AV[0],
    av.y + RAA_TIP_FROM_AV[1],
    av.z + RAA_TIP_FROM_AV[2],
    RAA_BASE_R_CM * bo,
    RAA_TIP_R_CM * bo,
  );
}

/**
 * Distance (cm) to the left atrial appendage's pouch at radial scale `bo`, its lobes adding `lobe` to the radius: a
 * capsule on the anterolateral body of the atrium (centre `la`, semi-axes `lr`, height of its centre `czL` now) pointing
 * anteriorly (A2C/PSAX-AV). The lumen is lobed; the epicardium and the sac read the smooth pouch (`lobe` 0).
 */
export function leftAppendageDistance(
  la: Point3,
  lr: Point3,
  czL: number,
  bo: number,
  lobe: number,
  x: number,
  y: number,
  z: number,
): number {
  return sdCapsule(
    x,
    y,
    z,
    la.x + lr.x * 0.55,
    la.y + lr.y * 0.55,
    czL + 0.4,
    la.x + lr.x * 0.95,
    la.y + lr.y * 0.55 + 2.0,
    czL + 0.9,
    0.55 * bo + lobe,
  );
}
