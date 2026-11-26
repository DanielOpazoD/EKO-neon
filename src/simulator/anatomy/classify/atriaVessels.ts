import { LA_POSTERIOR_FLOOR_DROP_CM, LA_POSTERIOR_RATIO } from '../laGeometry';
import { Structure, Tissue } from '../tissue';
import { sdCapsule, sdRoundCone, smax } from '../sdf';
import { latticeNoise3 } from '@/core/noise';
import { setSample, type ClassifyCtx } from './context';
import { LA_WALL_CM, LAA_WALL_CM } from './pericardium';
import { IVC_HEPATIC_SEGMENT_CM, type AnchorsCached } from '../anchors';
import { PV_RADIUS, PV_WALL_CM, pulmonaryVeinSegment } from '../pulmonaryVeins';
import { leftAppendageDistance, rightAppendageLumen } from '../atrialAppendages';

export { RAA_BASE_R_CM, RAA_TIP_FROM_AV, RAA_TIP_R_CM } from '../atrialAppendages';

/** Scratch for the pulmonary vein segment of the sample being classified (no allocation per sample). */
const PV_SEG = new Float64Array(6);

/** The posterior wall plane of the anchors, in the shape the vein placement takes (decision 330). */
export function laWallOf(A: AnchorsCached): { nx: number; ny: number; nz: number; d: number } {
  return { nx: A.laPostN.x, ny: A.laPostN.y, nz: A.laPostN.z, d: A.laPostD };
}

/**
 * Share of the tricuspid annular descent the right atrial roof follows (decision 149). With the roof fixed the atrium
 * lengthened by the whole TAPSE of the free wall, 2.15 cm in the normal case, and measured 4.5 cm at end-diastole and
 * 6.7 at end-systole against a normal 3.5–4 and 4.5–5.3: the annular centre descends less than its lateral hinge, the
 * septal part being tied to the fibrous skeleton, and the atrium's upper wall follows the base a little. Half the TAPSE
 * is the effective elongation that brings the maximal volume to the case's declared one (47 against 45 mL in the normal
 * case) with a total emptying fraction of 38–43 % in sinus rhythm.
 */
export const RA_ROOF_DESCENT_SHARE = 0.5;

/**
 * The right atrium's lateral wall (decision 367), the left atrium's of decision 366 on the tricuspid side: it stands
 * RA_LATERAL_HINGE_CM beyond the lateral tricuspid hinge, over the top of the ventricle's free wall, from the shelf up to
 * RA_LATERAL_WALL_HOLD (a fraction of the body's half length about its centre, below it) and eases back into the body
 * by RA_LATERAL_WALL_TOP, above which it adds nothing.
 */
export const RA_LATERAL_HINGE_CM = 0.5;
export const RA_LATERAL_WALL_HOLD = -0.3;
export const RA_LATERAL_WALL_TOP = 0.9;

/**
 * The base the right ventricle vacates in systole becomes atrium only over the tricuspid orifice and a vestibule of
 * this width around it (decision 149).
 */
export const RA_SLEEVE_MARGIN_CM = 0.5;

/**
 * The vestibule opens onto the septal side of the tricuspid orifice (decision 226): the orifice column and the cavity
 * are joined with a smooth minimum of RA_VESTIBULE_UNION_CM where the direction from the orifice centre points toward the
 * septum (its cosine above RA_VESTIBULE_SEPTAL_COS, full RA_VESTIBULE_SEPTAL_RAMP beyond), and with a sharp one on the
 * free-wall side, which keeps its vestibule wall down to the annulus.
 */
export const RA_VESTIBULE_UNION_CM = 0.6;
export const RA_VESTIBULE_SEPTAL_COS = 0.2;
export const RA_VESTIBULE_SEPTAL_RAMP = 0.3;

/**
 * The anteromedial right atrium (decision 218): the atrial wall that wraps the right side of the aortic root.
 */
export const RA_ANTEROMEDIAL_FROM: readonly [number, number, number] = [0.6, 0.6, 0.4];
export const RA_ANTEROMEDIAL_TO: readonly [number, number, number] = [-2.1, -0.2, -0.3];
export const RA_ANTEROMEDIAL_R_CM = 0.8;
export const RA_ANTEROMEDIAL_BLEND_CM = 0.6;

/**
 * The atrium beside the membranous septum (decision 222).
 */
export const RA_SEPTAL_RELEASE_LA_CM = 0.6;
export const RA_SEPTAL_RELEASE_RAMP_CM = 0.6;

/**
 * Out to which fossa radius (1 = its rim, 1.3 = the limbus's outer edge) the left atrium meets the septum (decision 319).
 */
export const FOSSA_REACH = 1.0;

/** How far past the interatrial plane (cm) the released atrium may reach: its own shape bounds it. */
export const RA_SEPTAL_RELEASE_CM = 4;
export const RA_MEMBRANOUS_TO: readonly [number, number, number] = [-1.45, -0.85, 0.05];
export const RA_MEMBRANOUS_R_CM = 0.5;
/** Reach (cm) from the ventricular epicardium over which the atrium follows the septal crest (decision 223). */
export const RA_CREST_REACH_CM = 1.5;

/** Distance to the superior vena cava, which enters the roof of the atrium from above (decision 219). */
export function svcDistance(A: AnchorsCached, x: number, y: number, z: number): number {
  // its entry segment and the rest that rises (decision 324)
  return Math.min(
    sdCapsule(x, y, z, A.svcA.x, A.svcA.y, A.svcA.z, A.svcM.x, A.svcM.y, A.svcM.z, A.svcR),
    sdCapsule(x, y, z, A.svcM.x, A.svcM.y, A.svcM.z, A.svcB.x, A.svcB.y, A.svcB.z, A.svcR),
  );
}

/** Distance to the inferior vena cava and its intrahepatic segment, narrowed by its respiratory collapse (decision 113). */
export function ivcDistance(
  A: AnchorsCached,
  ivcCollapse: number,
  x: number,
  y: number,
  z: number,
): number {
  const rI = A.ivcR * (1 - ivcCollapse);
  return Math.min(
    sdCapsule(x, y, z, A.ivcA.x, A.ivcA.y, A.ivcA.z, A.ivcB.x, A.ivcB.y, A.ivcB.z, rI),
    sdCapsule(
      x,
      y,
      z,
      A.ivcB.x,
      A.ivcB.y,
      A.ivcB.z,
      A.ivcB.x + IVC_HEPATIC_SEGMENT_CM[0],
      A.ivcB.y + IVC_HEPATIC_SEGMENT_CM[1],
      A.ivcB.z + IVC_HEPATIC_SEGMENT_CM[2],
      rI,
    ),
  );
}

/**
 * Distance to the lumen of the veins and the appendage that open into the left atrium (decision 320), negative inside one
 * of them.
 */
export function leftAtrialMouthLumen(
  c: ClassifyCtx,
  x: number,
  y: number,
  z: number,
  la: { x: number; y: number; z: number },
  lr: { x: number; y: number; z: number },
  bo: number,
  czL: number,
): number {
  let d = leftAppendageDistance(
    la,
    lr,
    czL,
    bo,
    0.12 * (latticeNoise3(x * 2.3 + 1.7, y * 2.3 + 4.2, z * 2.3 + 8.8, c.m.wallNoise) - 0.5),
    x,
    y,
    z,
  );
  const g = c.veinSeg;
  for (let i = 0; i < 24; i += 6) {
    const dPv = smax(
      sdCapsule(x, y, z, g[i]!, g[i + 1]!, g[i + 2]!, g[i + 3]!, g[i + 4]!, g[i + 5]!, PV_RADIUS),
      0.12 - c.colDist,
      0.2,
    );
    d = Math.min(d, dPv);
  }
  return d;
}

/**
 * Clearance (cm) the right atrial cavity keeps from the lumen of the right pulmonary veins (decision 320).
 */
export const RA_RIGHT_VEIN_CLEARANCE_CM = PV_WALL_CM + 0.22;

/**
 * Clearance (cm) the superior vena cava's lumen keeps from the right pulmonary veins' (decision 370).
 */
export const PV_SVC_CLEARANCE_CM = PV_WALL_CM + 0.12;

/** Distance to the lumen of the right pulmonary veins (decision 320), from the per-pose vein segments. */
export function rightVeinDistance(c: ClassifyCtx, x: number, y: number, z: number): number {
  const g = c.veinSeg;
  return Math.min(
    sdCapsule(x, y, z, g[0]!, g[1]!, g[2]!, g[3]!, g[4]!, g[5]!, PV_RADIUS),
    sdCapsule(x, y, z, g[12]!, g[13]!, g[14]!, g[15]!, g[16]!, g[17]!, PV_RADIUS),
  );
}

/**
 * Fill the per-pose atrial cache (decisions 320-321): the vein segments and the coronary sinus course depend only on the
 * pose and the anchors, and computing them per sample cost a quarter of the CPU tracer's time.
 */
export function atrialPoseCache(
  c: ClassifyCtx,
  la: { x: number; y: number; z: number },
  lr: { x: number; y: number; z: number },
  bo: number,
  czL: number,
  rzL: number,
): void {
  if (c.atrialHp === c.hp && c.atrialA === c.A) return;
  c.atrialHp = c.hp;
  c.atrialA = c.A;
  const wall = laWallOf(c.A);
  for (let i = 0; i < 4; i++) {
    pulmonaryVeinSegment(i, la, lr, bo, czL, rzL, PV_SEG, wall);
    c.veinSeg.set(PV_SEG, 6 * i);
  }
  const zAnn = c.hp.zAnn;
  const mv = c.hp.valves.mitral;
  // the sinus lies in the groove, under the body's full posterior surface, whatever the radial scale (decision 330)
  const rx = lr.x,
    ry = lr.y;
  const zs = zAnn - CS_ABOVE_ANNULUS_CM;
  const x0 = c.A.iasX + CS_CRUX_FROM_IAS_CM;
  const g = c.csGeo;
  g[0] = zs;
  g[1] = x0;
  g[2] = Math.min(CS_LATERAL_X_CM, la.x + 0.85 * rx);
  g[3] = rx;
  g[4] = ry;
  g[5] = czL;
  g[6] = rzL;
  g[7] = Math.max(
    laPosteriorY(Math.max(x0, la.x), zs, la, rx, ry, czL, rzL) -
      LA_POSTERIOR_FLOOR_DROP_CM -
      CS_GROOVE_OFFSET_CM,
    csWallFloor(c, x0, zs),
  );
  g[8] = c.A.iasX - CS_OSTIUM_FROM_IAS_CM;
  g[9] = CS_OSTIUM_Y_CM;
  g[10] = zAnn - CS_OSTIUM_ABOVE_ANNULUS_CM;
  g[11] = mv.cx;
  g[12] = mv.cy;
  g[13] = mv.R;
}

/**
 * The coronary sinus (decision 321) runs in the posterior atrioventricular groove on the atrial side of the mitral annulus.
 */
export const CS_ABOVE_ANNULUS_CM = 0.5;
export const CS_LATERAL_X_CM = 2.2;
export const CS_CRUX_FROM_IAS_CM = 0.4;
export const CS_OSTIUM_FROM_IAS_CM = 1.2;
export const CS_OSTIUM_Y_CM = -1.2;
export const CS_OSTIUM_ABOVE_ANNULUS_CM = 0.4;
export const CS_RADIUS_CM = 0.33;
export const CS_OSTIUM_RADIUS_CM = 0.5;

/**
 * Above the annulus the left atrium comes up to the root's outer wall less its own wall and this gap (cm), within
 * LA_ROOT_NEAR_CM of the root (decision 335).
 */
export const LA_ROOT_GAP_CM = 0.05;
export const LA_ROOT_NEAR_CM = 1.2;
/** Up to how far above the annulus, along the root's axis, the septum's anterior rim rests on the root: the sinuses (cm). */
export const LA_ROOT_TOP_CM = 1.5;
/** How fast the right atrium's release takes the root's ground back from the left atrium (cm per unit of release). */
export const LA_ROOT_RELEASE_CM = RA_SEPTAL_RELEASE_CM;
export const CS_WALL_CM = 0.1;
/** From the atrium's posterior surface to the sinus's axis: the atrial wall, the sinus's wall and its radius (cm). */
export const CS_GROOVE_OFFSET_CM = LA_WALL_CM + CS_WALL_CM + CS_RADIUS_CM;
/** From the posterior mitral annulus to the sinus's axis (cm): the ring, the sinus's wall and its radius. */
export const CS_ANNULUS_OFFSET_CM = 0.15 + CS_WALL_CM + CS_RADIUS_CM;

/** Lowest y the sinus's axis may take at (x, z) to stay its wall in front of the posterior wall plane (decision 330). */
function csWallFloor(c: ClassifyCtx, x: number, z: number): number {
  const n = c.A.laPostN;
  return (c.A.laPostD - CS_RADIUS_CM - CS_WALL_CM - x * n.x - z * n.z) / n.y;
}

/** Posterior surface (y) of the left atrial cavity at (x, z), the flattened ellipsoid of `leftAtrialFreeSdf`. */
function laPosteriorY(
  x: number,
  z: number,
  la: { x: number; y: number },
  rx: number,
  ry: number,
  cz: number,
  rz: number,
): number {
  const u = (x - la.x) / rx,
    w = (z - cz) / rz;
  const q = Math.max(0, 1 - u * u - w * w);
  return Math.max(la.y - ry * Math.sqrt(q), la.y - LA_POSTERIOR_RATIO * ry);
}

/**
 * Distance to the coronary sinus lumen (decision 321). Far from its course it may return a lower bound (the distance to
 * the box that holds it) where that already decides every use of it.
 */
export function coronarySinusDistance(c: ClassifyCtx, x: number, y: number, z: number): number {
  const g = c.csGeo;
  const zs = g[0]!,
    x0 = g[1]!,
    x1 = g[2]!;
  // the lumen lies within 0.5 cm of zs (the ostium's funnel reaches zs + 0.6) and between the ostium and the lateral end
  const far = Math.max(
    Math.abs(z - zs - 0.05) - 0.65,
    g[8]! - CS_OSTIUM_RADIUS_CM - x,
    x - x1 - CS_RADIUS_CM,
  );
  // a lower bound is enough where it already decides every use: the wall and lumen tests, and the sac's minimum
  if (far > 0.3 && far - CS_WALL_CM >= c.laEpi) return far;
  const la = c.A.laCenter;
  const xc = Math.min(x1, Math.max(x0, x));
  const yc = Math.max(
    Math.min(
      laPosteriorY(Math.max(xc, la.x), zs, la, g[3]!, g[4]!, g[5]!, g[6]!) -
        LA_POSTERIOR_FLOOR_DROP_CM -
        CS_GROOVE_OFFSET_CM,
      g[12]! - Math.sqrt(Math.max(0, g[13]! * g[13]! - (xc - g[11]!) ** 2)) - CS_ANNULUS_OFFSET_CM,
    ),
    csWallFloor(c, xc, zs),
  );
  const body = Math.hypot(x - xc, y - yc, z - zs) - CS_RADIUS_CM;
  const ostium = sdRoundCone(
    x,
    y,
    z,
    x0,
    g[7]!,
    zs,
    g[8]!,
    g[9]!,
    g[10]!,
    CS_RADIUS_CM,
    CS_OSTIUM_RADIUS_CM,
  );
  // clear of the left atrial cavity by both walls, so the sinus is never unroofed into it
  return Math.max(Math.min(body, ostium), LA_WALL_CM + CS_WALL_CM - c.laCav);
}

/**
 * Height (cm) over the atrial floor below which the right atrial appendage does not reach (decision 324); its tip and
 * radii are in atrialAppendages.ts (decisions 323, 388, 389).
 */
export const RAA_ABOVE_FLOOR_CM = 0.5;

/**
 * The appendage's own emptying (decision 389).
 */
export const RAA_PASSIVE_SHARE = 0.5;
export const RAA_BOOSTER_GAIN = 3.5;

/** The appendage's radial scale from its atrium's booster and reservoir (decision 389). Shared with GLSL. */
export function appendageScale(booster: number, reservoir: number, contraction: number): number {
  const res = 1 - (1 - reservoir) * RAA_PASSIVE_SHARE;
  return (1 - (1 - booster) * RAA_BOOSTER_GAIN) * (res + (1 - res) * contraction);
}

/** Distance to the right atrial appendage's lumen (decision 323). */
export function rightAppendageDistance(
  c: ClassifyCtx,
  bo: number,
  x: number,
  y: number,
  z: number,
): number {
  return rightAppendageLumen(c.A.raaBase, c.A.avCenter, bo, x, y, z);
}

/** Radial scale of the atria: reservoir/conduit/booster with the LV contraction, shrunk by the atrial kick and its hold. */
export function atrialScale(booster: number, reservoir: number, contraction: number): number {
  return booster * (reservoir + (1 - reservoir) * contraction);
}

/** Right atrial radial scale under tamponade: late-diastolic collapse of `raCollapse` (0..1). */
export function raCollapseScale(raCollapse: number): number {
  return 1 - 0.35 * raCollapse;
}

/**
 * Classifies the left atrial appendage, four pulmonary veins, superior/inferior venae cavae, hepatic vein,
 * and coronary sinus once the atrial chambers and interatrial septum have been evaluated.
 */
export function classifyAtrialVessels(
  c: ClassifyCtx,
  la: { x: number; y: number; z: number },
  lr: { x: number; y: number; z: number },
  bo: number,
  czL: number,
  dSvc: number,
  dIvc: number,
  dCs: number,
): boolean {
  const { m, A, x, y, z, out } = c;
  // left atrial appendage: lobulated pouch on the anterolateral LA, pointing anteriorly (A2C/PSAX-AV)
  {
    const lob =
      0.12 * (latticeNoise3(x * 2.3 + 1.7, y * 2.3 + 4.2, z * 2.3 + 8.8, m.wallNoise) - 0.5);
    const dApp = leftAppendageDistance(la, lr, czL, bo, lob, x, y, z);
    if (dApp < 0) {
      setSample(out, Tissue.Blood, dApp, 0, 1, 0, x, y, z, 0, Structure.LaAppendage);
      return true;
    }
    if (dApp < LAA_WALL_CM) {
      setSample(
        out,
        Tissue.Myocardium,
        -Math.min(dApp, LAA_WALL_CM - dApp),
        0,
        1,
        0,
        x,
        y,
        z,
        0,
        Structure.LaWall,
      );
      return true;
    }
  }
  // pulmonary veins: towards the hila from the lateral wall (left) and the posteromedial corner (right); the
  // geometry lives in pulmonaryVeins.ts, shared with the venous flow sampler (decision 143)
  for (let i = 0; i < 24; i += 6) {
    const g = c.veinSeg;
    // the veins pass beside the descending aorta and the vertebral body, not through them (decision 273)
    const dPv = smax(
      sdCapsule(x, y, z, g[i]!, g[i + 1]!, g[i + 2]!, g[i + 3]!, g[i + 4]!, g[i + 5]!, PV_RADIUS),
      0.12 - c.colDist,
      0.2,
    );
    if (dPv < 0) {
      setSample(out, Tissue.Blood, dPv, 0, -1, 0, x, y, z, 0, Structure.PulmonaryVein);
      return true;
    }
    // the wall scatters as the atrial wall it continues (decision 370)
    if (dPv < PV_WALL_CM) {
      setSample(
        out,
        Tissue.Myocardium,
        -Math.min(dPv, PV_WALL_CM - dPv),
        0,
        -1,
        0,
        x,
        y,
        z,
        0,
        Structure.PulmonaryVein,
      );
      return true;
    }
  }
  // venae cavae: the superior enters the RA roof from above (torso superior), the inferior its floor from
  // below and behind through the liver, joined by a hepatic vein (subcostal views); the IVC narrows with the sniff
  {
    // the superior vena cava's lumen keeps clear of the right pulmonary veins' by both walls (decision 370)
    const dSvcLumen = Math.max(dSvc, PV_SVC_CLEARANCE_CM - rightVeinDistance(c, x, y, z));
    if (dSvcLumen < 0) {
      setSample(out, Tissue.Blood, dSvcLumen, 0, 0, -1, x, y, z, 0, Structure.Svc);
      return true;
    }
    if (dSvcLumen < 0.12) {
      setSample(
        out,
        Tissue.VesselWall,
        -Math.min(dSvcLumen, 0.12 - dSvcLumen),
        0,
        0,
        -1,
        x,
        y,
        z,
        0,
        Structure.Svc,
      );
      return true;
    }
    const dHv = sdCapsule(x, y, z, A.hvA.x, A.hvA.y, A.hvA.z, A.hvB.x, A.hvB.y, A.hvB.z, 0.4);
    if (dIvc < 0) {
      setSample(out, Tissue.Blood, dIvc, 0, 0, 1, x, y, z, 0, Structure.Ivc);
      return true;
    }
    // nor the caval wall across the mouth of the hepatic vein (it drew 0.14 cm there)
    if (dIvc < 0.12 && dHv >= 0) {
      setSample(
        out,
        Tissue.VesselWall,
        -Math.min(dIvc, 0.12 - dIvc),
        0,
        0,
        1,
        x,
        y,
        z,
        0,
        Structure.Ivc,
      );
      return true;
    }
    if (dHv < 0) {
      setSample(out, Tissue.Blood, dHv, 0, 0, 1, x, y, z, 0, Structure.HepaticVein);
      return true;
    }
    if (dHv < 0.08) {
      setSample(
        out,
        Tissue.VesselWall,
        -Math.min(dHv, 0.08 - dHv),
        0,
        0,
        1,
        x,
        y,
        z,
        0,
        Structure.HepaticVein,
      );
      return true;
    }
  }
  // coronary sinus: in the posterior atrioventricular groove above the mitral annulus, into the RA (decision 321)
  {
    if (dCs < 0) {
      setSample(out, Tissue.Blood, dCs, 0, -1, 0, x, y, z, 0, Structure.CoronarySinus);
      return true;
    }
    if (dCs < CS_WALL_CM) {
      setSample(
        out,
        Tissue.VesselWall,
        -Math.min(dCs, CS_WALL_CM - dCs),
        0,
        -1,
        0,
        x,
        y,
        z,
        0,
        Structure.CoronarySinus,
      );
      return true;
    }
  }
  return false;
}
