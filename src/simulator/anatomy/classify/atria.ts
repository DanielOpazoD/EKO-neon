import {
  IAS_LEFT_FACE_CM,
  iasThickness,
  iasBow,
  IAS_FOSSA_DESCENT_SHARE,
  LA_CORNER_ANTERIOR_MARGIN_CM,
  LA_CORNER_HEIGHT_CM,
  LA_BASE_BOUND_READ_CM,
  laBaseLateralShare,
  laBaseSdf,
  laLateralWallSdf,
  LA_LATERAL_WALL_HOLD,
  LA_LATERAL_WALL_TOP,
  LA_LATERAL_HINGE_CM,
  LA_LATERAL_WALL_AHEAD_CM,
  LA_LATERAL_WALL_FRONT_CM,
  laLateralWallBlend,
  LA_CORNER_FLAT_CM,
  LA_FLOOR_Y_CM,
  laFootprintScale,
  laRootHold,
  LA_SEPTAL_FLAT_CM,
  LA_TOUCH_CM,
  laMedialFaceOffset,
  leftAtrialFreeSdf,
  RA_LA_MIN_CM,
} from '../laGeometry';
import { Structure, Tissue } from '../tissue';
import { sdCapsule, sdEllipsoid, smax, smin } from '../sdf';
import { ROOT_EXCURSION } from '../heartFrame';
import { mitralOutlineSdf } from '../mitralValve';
import { rvFloorZ, rvRadii } from '../rv';
import { skirtOffsetAt, skirtWarpScale } from '../valveSkirt';
import { setSample, type ClassifyCtx } from './context';
import { AORTIC_ROOT_WALL_CM } from '../aorticValve';
import { LA_WALL_CM, LAA_WALL_CM } from './pericardium';
import {
  appendageScale,
  atrialPoseCache,
  atrialScale,
  classifyAtrialVessels,
  coronarySinusDistance,
  CS_WALL_CM,
  FOSSA_REACH,
  ivcDistance,
  LA_ROOT_GAP_CM,
  LA_ROOT_NEAR_CM,
  LA_ROOT_RELEASE_CM,
  LA_ROOT_TOP_CM,
  leftAtrialMouthLumen,
  RA_ANTEROMEDIAL_BLEND_CM,
  RA_ANTEROMEDIAL_FROM,
  RA_ANTEROMEDIAL_R_CM,
  RA_ANTEROMEDIAL_TO,
  RA_CREST_REACH_CM,
  RA_LATERAL_HINGE_CM,
  RA_LATERAL_WALL_HOLD,
  RA_LATERAL_WALL_TOP,
  RA_MEMBRANOUS_R_CM,
  RA_MEMBRANOUS_TO,
  RA_RIGHT_VEIN_CLEARANCE_CM,
  RA_ROOF_DESCENT_SHARE,
  RA_SEPTAL_RELEASE_CM,
  RA_SEPTAL_RELEASE_LA_CM,
  RA_SEPTAL_RELEASE_RAMP_CM,
  RA_SLEEVE_MARGIN_CM,
  RA_VESTIBULE_SEPTAL_COS,
  RA_VESTIBULE_SEPTAL_RAMP,
  RA_VESTIBULE_UNION_CM,
  RAA_ABOVE_FLOOR_CM,
  raCollapseScale,
  rightAppendageDistance,
  rightVeinDistance,
  svcDistance,
} from './atriaVessels';

export { iasThickness, LA_CORNER_HEIGHT_CM } from '../laGeometry';
export {
  appendageScale,
  atrialScale,
  coronarySinusDistance,
  CS_ABOVE_ANNULUS_CM,
  CS_ANNULUS_OFFSET_CM,
  CS_CRUX_FROM_IAS_CM,
  CS_GROOVE_OFFSET_CM,
  CS_LATERAL_X_CM,
  CS_OSTIUM_ABOVE_ANNULUS_CM,
  CS_OSTIUM_FROM_IAS_CM,
  CS_OSTIUM_RADIUS_CM,
  CS_OSTIUM_Y_CM,
  CS_RADIUS_CM,
  CS_WALL_CM,
  FOSSA_REACH,
  ivcDistance,
  LA_ROOT_GAP_CM,
  LA_ROOT_NEAR_CM,
  LA_ROOT_RELEASE_CM,
  LA_ROOT_TOP_CM,
  laWallOf,
  leftAtrialMouthLumen,
  PV_SVC_CLEARANCE_CM,
  RA_ANTEROMEDIAL_BLEND_CM,
  RA_ANTEROMEDIAL_FROM,
  RA_ANTEROMEDIAL_R_CM,
  RA_ANTEROMEDIAL_TO,
  RA_CREST_REACH_CM,
  RA_LATERAL_HINGE_CM,
  RA_LATERAL_WALL_HOLD,
  RA_LATERAL_WALL_TOP,
  RA_MEMBRANOUS_R_CM,
  RA_MEMBRANOUS_TO,
  RA_RIGHT_VEIN_CLEARANCE_CM,
  RA_ROOF_DESCENT_SHARE,
  RA_SEPTAL_RELEASE_CM,
  RA_SEPTAL_RELEASE_LA_CM,
  RA_SEPTAL_RELEASE_RAMP_CM,
  RA_SLEEVE_MARGIN_CM,
  RA_VESTIBULE_SEPTAL_COS,
  RA_VESTIBULE_SEPTAL_RAMP,
  RA_VESTIBULE_UNION_CM,
  RAA_ABOVE_FLOOR_CM,
  RAA_BASE_R_CM,
  RAA_BOOSTER_GAIN,
  RAA_PASSIVE_SHARE,
  RAA_TIP_FROM_AV,
  RAA_TIP_R_CM,
  raCollapseScale,
  rightAppendageDistance,
  rightVeinDistance,
  svcDistance,
} from './atriaVessels';

/**
 * Atria (lengthen in systole as the annulus descends), interatrial septum, left atrial appendage, pulmonary veins,
 * venae cavae with the hepatic vein, and the coronary sinus. True when the point is one of them.
 */
export function classifyAtria(c: ClassifyCtx): boolean {
  const { m, hp, A, x, y, z, out } = c;
  const zAnn = hp.zAnn;
  const V = hp.valves;
  const la = A.laCenter,
    lr = A.laR;
  const zTop = la.z - lr.z; // fixed superior boundary (roof, under the pulmonary bifurcation)
  const zBottom = zAnn + 0.25;
  const czL = (zTop + zBottom) / 2,
    rzL = (zBottom - zTop) / 2;
  // reservoir / conduit / booster: radial size follows LV contraction (maximal at end-systole); the atrial
  // kick and its hold until ejection shrink it further (laBooster)
  const bo = atrialScale(hp.laBooster, A.laReservoir, hp.state.contraction);
  // the right atrium's own (decision 330)
  const boR = atrialScale(hp.laBooster, A.raReservoir, hp.state.contraction);
  atrialPoseCache(c, la, lr, bo, czL, rzL);
  // interatrial septum: muscular septum ~0.55 cm thinning to the fossa ovalis membrane. Its left face is flat (decision
  // 327): the fossa sinks on the right atrial side only. The fossa follows the base down, and the septum bows toward the
  // right atrium as the left fills (decision 369)
  const fo = Math.hypot(
    (y - A.fossaY) / 0.6,
    (z - A.fossaZ - IAS_FOSSA_DESCENT_SHARE * zAnn) / 0.7,
  );
  const xIas = A.iasX - iasBow(fo, hp.state.contraction);
  const tIas = iasThickness(fo);
  const xFace = xIas + IAS_LEFT_FACE_CM;
  // LA: ellipsoid flattened against the septum (medial clip), against the oesophagus / descending aorta
  // (posterior clip) and under the pulmonary bifurcation (roof clip)
  // beside the aortic root the body keeps its end-systolic size (decision 341): the scale at this point
  const rootOuter = c.rootRr - c.rootR - AORTIC_ROOT_WALL_CM;
  const boL = bo + (1 - bo) * laRootHold(c.rootT, rootOuter, x - (A.iasX + IAS_LEFT_FACE_CM));
  const dFreeLa = leftAtrialFreeSdf(x, y, z, la.x, la.y, czL, lr.x * boL, lr.y * boL, rzL, zTop);
  // the body at the scale of the moment, without the hold: what the right atrium's release reads (decision 222), which
  // the hold would otherwise keep from the membranous septum beside the root
  const dFreeLaNow =
    boL === bo
      ? dFreeLa
      : leftAtrialFreeSdf(x, y, z, la.x, la.y, czL, lr.x * bo, lr.y * bo, rzL, zTop);
  // the atrium's epicardium, for the pericardial sac: the outer face of the appendage's wall here, and of the body's below,
  // once its clips are known (decision 273)
  c.laEpi =
    sdCapsule(
      x,
      y,
      z,
      la.x + lr.x * 0.55,
      la.y + lr.y * 0.55,
      czL + 0.4,
      la.x + lr.x * 0.95,
      la.y + lr.y * 0.55 + 2.0,
      czL + 0.9,
      0.55 * bo,
    ) - LAA_WALL_CM;
  // The corner between the septum and the annulus belongs to the atrium (decision 318)
  const dCrossLa = leftAtrialFreeSdf(
    la.x,
    y,
    z,
    la.x,
    la.y,
    czL,
    lr.x * boL,
    lr.y * boL,
    rzL,
    zTop,
  );
  // the gap between the plane and the ellipsoid's medial chord at this (y, z) (decision 327)
  const uL = (y - la.y) / (lr.y * boL),
    wL = (z - czL) / rzL;
  const gap = la.x - lr.x * boL * Math.sqrt(Math.max(0, 1 - uL * uL - wL * wL)) - xFace;
  const zRef = zAnn - LA_CORNER_HEIGHT_CM;
  const wRef = (zRef - czL) / rzL;
  const gapRef = la.x - lr.x * boL * Math.sqrt(Math.max(0, 1 - uL * uL - wRef * wRef)) - xFace;
  const cornerOff = laMedialFaceOffset(gapRef, LA_CORNER_FLAT_CM);
  const dCrossRef = leftAtrialFreeSdf(la.x, y, zRef, la.x, la.y, czL, lr.x, lr.y, rzL, zTop);
  const zHingeHere = c.zHinge;
  const mvA = V.mitral;
  const rootClearFaceLine =
    (xFace - mvA.cx) * mvA.ux + (y - mvA.cy) * mvA.uy - (mvA.D + LA_CORNER_ANTERIOR_MARGIN_CM);
  const xRead = Math.max(xFace, x - LA_BASE_BOUND_READ_CM);
  const rootClearBaseLine =
    (xRead - mvA.cx) * mvA.ux + (y - mvA.cy) * mvA.uy - (mvA.D + LA_CORNER_ANTERIOR_MARGIN_CM);
  // the right atrium is released from the septal plane in front of the left atrium (decision 222): by how much
  const qRel = Math.min(
    1,
    Math.max(
      0,
      (Math.max(dFreeLaNow, rootClearFaceLine) - RA_SEPTAL_RELEASE_LA_CM) /
        RA_SEPTAL_RELEASE_RAMP_CM,
    ),
  );
  const relL = qRel * qRel * (3 - 2 * qRel);
  const atSinuses = c.rootT >= 0 && c.rootT <= LA_ROOT_TOP_CM;
  const rootWall = atSinuses
    ? Math.max(LA_WALL_CM + LA_ROOT_GAP_CM - rootOuter, rootOuter - LA_ROOT_NEAR_CM) +
      LA_ROOT_RELEASE_CM * relL
    : 1e3;
  const rootClearFace = Math.min(rootClearFaceLine, rootWall);
  const rootClearBase = Math.min(rootClearBaseLine, rootWall);
  const dCrossEq = leftAtrialFreeSdf(la.x, y, czL, la.x, la.y, czL, lr.x, lr.y, rzL, zTop);
  const dCorner = laBaseSdf(
    x,
    la.x,
    xFace,
    cornerOff,
    z,
    zRef,
    zHingeHere,
    dCrossRef,
    dCrossEq,
    rootClearBase,
    mitralOutlineSdf(x, y, mvA),
    laBaseLateralShare(x - mvA.cx, y - mvA.cy),
  );
  const dCrossLaFull = leftAtrialFreeSdf(la.x, y, z, la.x, la.y, czL, lr.x, lr.y, rzL, zTop);
  const dFossaLa = Math.max(xFace - x, x - la.x, dCrossLaFull, (fo - FOSSA_REACH) * 0.6);
  const faceOff = laMedialFaceOffset(gap, LA_SEPTAL_FLAT_CM);
  const dFlat = Math.max(xFace + faceOff - x, x - la.x, dCrossLa);
  const pn = A.laPostN;
  const tPost = x * pn.x + y * pn.y + z * pn.z - A.laPostD;
  const sPost = (x - la.x) * pn.x + (y - la.y) * pn.y + (z - czL) * pn.z;
  const sFront = (x - la.x) * pn.x + (y - la.y) * pn.y;
  let dPost: number;
  if (-sFront >= Math.max(dFreeLa, 0) + 0.6) dPost = dFreeLa + 0.6;
  else {
    const fp = laFootprintScale(bo);
    const dSil = sdEllipsoid(
      x - sPost * pn.x,
      Math.max(y - sPost * pn.y, la.y),
      Math.min(z - sPost * pn.z, czL),
      la.x,
      la.y,
      czL,
      lr.x * fp,
      lr.y * fp,
      rzL,
    );
    dPost = smax(
      smax(
        Math.max(tPost, -sFront, dSil, z - (c.zHinge - 0.5 * LA_WALL_CM)),
        LA_FLOOR_Y_CM - y,
        0.3,
      ),
      zTop + 0.15 * rzL - z,
      0.5,
    );
  }
  const raReach = Math.min(1, (RA_SEPTAL_RELEASE_CM * relL) / tIas);
  const holdShare = atSinuses && rootOuter < LA_ROOT_NEAR_CM ? raReach : 1;
  const frontClear = Math.min(rootClearFaceLine, xFace + RA_LA_MIN_CM * holdShare - x);
  const faceKeep = rootClearFaceLine > 0 ? xFace - x : -1e3;
  const dLatWall = Math.max(
    laLateralWallSdf(
      x,
      y,
      c.zHinge - z,
      (czL - z) / rzL,
      la.x,
      mvA.cx + mvA.R + LA_LATERAL_HINGE_CM,
      mvA.cy + LA_LATERAL_WALL_AHEAD_CM,
      LA_LATERAL_WALL_HOLD,
      LA_LATERAL_WALL_TOP,
      LA_LATERAL_WALL_FRONT_CM,
    ),
    czL - LA_LATERAL_WALL_TOP * rzL - z,
  );
  const dBodyPost = smin(smin(dFreeLa, dPost, 0.6), dLatWall, laLateralWallBlend(c.zHinge - z));
  const dLaBody = Math.max(
    Math.min(dBodyPost, dCorner, dFossaLa, dFlat),
    tPost,
    frontClear,
    faceKeep,
  );
  c.laEpi = Math.min(c.laEpi, dLaBody - LA_WALL_CM);
  const laTouch = Math.min(
    Math.max(dCrossLa, faceOff - LA_TOUCH_CM),
    Math.max(dCrossRef, zRef - z, z - zHingeHere, cornerOff - LA_TOUCH_CM, rootClearFace),
    Math.max(dCrossLaFull, (fo - FOSSA_REACH) * 0.6),
  );
  const nearLa =
    laTouch < 0 ||
    (rootClearFace < 0 &&
      sdEllipsoid(xIas, y, z, la.x, la.y, czL, lr.x * bo, lr.y * bo, rzL) < 0.45);
  const d = Math.max(
    Math.min(smax(dBodyPost, xFace - x, 0.3), dCorner, dFossaLa, dFlat),
    tPost,
    frontClear,
    faceKeep,
  );
  c.laCav = d;
  const nearRoot = atSinuses && rootOuter < LA_ROOT_NEAR_CM;
  const nearLaBody = nearLa || (nearRoot && x <= xFace && d < tIas + LA_TOUCH_CM);
  if (d < 0) {
    setSample(
      out,
      Tissue.Blood,
      d,
      (x - la.x) / lr.x,
      (y - la.y) / lr.y,
      (z - czL) / rzL,
      x,
      y,
      z,
      0,
      Structure.LaCavity,
    );
    return true;
  }
  // the coronary sinus, which the pericardial sac wraps with the atrium (decision 321)
  const dCs = coronarySinusDistance(c, x, y, z);
  c.laEpi = Math.min(c.laEpi, dCs - CS_WALL_CM);
  // not across the mouths of the pulmonary veins and the appendage (decision 320)
  if (dLaBody < LA_WALL_CM && x > xFace && leftAtrialMouthLumen(c, x, y, z, la, lr, bo, czL) >= 0) {
    setSample(
      out,
      Tissue.Myocardium,
      -Math.min(dLaBody, LA_WALL_CM - dLaBody),
      (x - la.x) / lr.x,
      (y - la.y) / lr.y,
      (z - czL) / rzL,
      x,
      y,
      z,
      0,
      Structure.LaWall,
    );
    return true;
  }
  // RA: rounder, flattened against the septum and posteriorly
  const ra = A.raCenter,
    rr = A.raR;
  const zTopR = ra.z - rr.z + RA_ROOF_DESCENT_SHARE * hp.tvZ;
  const tvOff = skirtOffsetAt(V.tv, x, y);
  const zBotR = A.tvCenter.z + hp.tvZ + tvOff + 0.03;
  const czR = (zTopR + zBotR) / 2,
    rzR = (zBotR - zTopR) / 2;
  const raC = raCollapseScale(hp.raCollapse);
  const dEllRa = sdEllipsoid(x, y, z, ra.x, ra.y, czR, rr.x * boR * raC, rr.y * boR * raC, rzR);
  let dFreeRa = smax(dEllRa, ra.y - 0.8 * rr.y * boR - y, 0.6);
  // the lateral wall rises from the lateral tricuspid hinge as the left atrium's does from the mitral one (decision 367)
  {
    const tvLat = V.tv.cx - V.tv.R - RA_LATERAL_HINGE_CM;
    const hR = zBotR - z;
    const dLatR = Math.max(
      laLateralWallSdf(
        -x,
        y,
        hR,
        (czR - z) / rzR,
        -ra.x,
        -tvLat,
        V.tv.cy,
        RA_LATERAL_WALL_HOLD,
        RA_LATERAL_WALL_TOP,
        0,
      ),
      czR - RA_LATERAL_WALL_TOP * rzR - z,
    );
    dFreeRa = smin(dFreeRa, dLatR, laLateralWallBlend(hR));
    c.raBody = dFreeRa;
  }
  // the anteromedial atrium beside the right side of the aortic root (decision 218)
  {
    const av = A.avCenter;
    const f = RA_ANTEROMEDIAL_FROM,
      t = RA_ANTEROMEDIAL_TO;
    const dAm = sdCapsule(
      x,
      y,
      z,
      ra.x + f[0],
      ra.y + f[1],
      ra.z + f[2],
      av.x + t[0],
      av.y + t[1],
      av.z + zAnn * ROOT_EXCURSION + t[2],
      RA_ANTEROMEDIAL_R_CM * boR,
    );
    dFreeRa = smin(dFreeRa, dAm, RA_ANTEROMEDIAL_BLEND_CM);
    const m2 = RA_MEMBRANOUS_TO;
    const zr = zAnn * ROOT_EXCURSION;
    let dMs = sdCapsule(
      x,
      y,
      z,
      av.x + t[0],
      av.y + t[1],
      av.z + zr + t[2],
      av.x + m2[0],
      av.y + m2[1],
      av.z + zr + m2[2],
      RA_MEMBRANOUS_R_CM * boR,
    );
    dMs = smax(dMs, z - zBotR, 0.1);
    dMs = smax(dMs, -Math.max(c.dEllR - c.wallT - 0.05, zAnn - 0.25 - z), 0.1);
    dFreeRa = smin(dFreeRa, dMs, 0.3);
  }
  // where the atrium borders the septal crest it takes the thickness the crest gave up (decision 223)
  if (c.crestLoss > 0) {
    const dEpi = c.dEllR - c.wallT;
    const w =
      Math.min(1, Math.max(0, dEpi / 0.05)) *
      Math.min(1, Math.max(0, 1 - dEpi / RA_CREST_REACH_CM));
    dFreeRa -= c.crestLoss * w;
  }
  let dSleeve = 1e3;
  if (hp.tvZ > 0.05) {
    const rad = c.sleeveRad;
    rvRadii(m, A, hp.prof, hp.thickK, zAnn, hp.lengthNow, 0, 0, hp.septalShiftCm, 0, c.az, z, rad);
    const u = rad[1]!;
    if (u > 0 && u < 1) {
      const r = Math.hypot(x, y);
      const rhoT = Math.hypot(x - V.tv.cx, y - V.tv.cy);
      const tvOffEd =
        tvOff - V.tv.lift - (rhoT > 1e-6 ? (V.tv.tiltC * (x - V.tv.cx)) / rhoT : V.tv.tiltC);
      dSleeve = Math.max(
        rad[0]! + 0.1 - r,
        r - (rad[2]! - m.anatomy.rv.freeWallThicknessCm),
        rvFloorZ(A.tvCenter.z, 0, 0, u, tvOffEd) - z,
        z - rvFloorZ(A.tvCenter.z, hp.tvZ, hp.pvZ, u, tvOff),
        Math.hypot(x - V.tv.cx, y - V.tv.cy) * skirtWarpScale(V.tv, x - V.tv.cx, y - V.tv.cy) -
          (V.tv.R + RA_SLEEVE_MARGIN_CM),
      );
    }
  }
  c.raSleeve = dSleeve;
  dFreeRa = Math.min(dFreeRa, dSleeve);
  const rightVeinClear = Math.min(
    RA_RIGHT_VEIN_CLEARANCE_CM - rightVeinDistance(c, x, y, z),
    (fo - FOSSA_REACH) * 0.6,
  );
  dFreeRa = smax(dFreeRa, rightVeinClear, 0.2);
  const xRaFace = xFace - tIas;
  const septumPlane = x - xRaFace - RA_SEPTAL_RELEASE_CM * relL;
  const uR = (y - ra.y) / (rr.y * boR * raC),
    wR = (z - czR) / rzR;
  const gapR = xRaFace - (ra.x + rr.x * boR * raC * Math.sqrt(Math.max(0, 1 - uR * uR - wR * wR)));
  const dCrossRa = smax(
    sdEllipsoid(ra.x, y, z, ra.x, ra.y, czR, rr.x * boR * raC, rr.y * boR * raC, rzR),
    ra.y - 0.8 * rr.y * boR - y,
    0.6,
  );
  const dFlatRa = Math.max(
    x - (xRaFace - laMedialFaceOffset(gapR, LA_SEPTAL_FLAT_CM)),
    ra.x - x,
    dCrossRa,
    laTouch,
    rightVeinClear,
  );
  const raLaMin = Math.min(tIas, RA_LA_MIN_CM);
  const dR = Math.max(Math.min(smax(dFreeRa, septumPlane, 0.3), dFlatRa), raLaMin - c.laCav);
  const inSeptum = x <= xFace && x >= xFace - tIas && nearLaBody;
  if (dR < 0 && !inSeptum) {
    setSample(
      out,
      Tissue.Blood,
      dR,
      (x - ra.x) / rr.x,
      (y - ra.y) / rr.y,
      (z - czR) / rzR,
      x,
      y,
      z,
      0,
      Structure.RaCavity,
    );
    return true;
  }
  const boA = appendageScale(hp.laBooster, A.raReservoir, hp.state.contraction);
  const dRaa = Math.max(
    smax(rightAppendageDistance(c, boA, x, y, z), z - (zBotR - RAA_ABOVE_FLOOR_CM), 0.2),
    LAA_WALL_CM - c.aaOuter,
    c.rootT > -90 ? LAA_WALL_CM - rootOuter : -1e3,
  );
  c.raaEpi = dRaa - LAA_WALL_CM;
  const dSvc = svcDistance(A, x, y, z);
  const dIvc = ivcDistance(A, hp.ivcCollapse, x, y, z);
  if (
    !inSeptum &&
    dFreeRa < 0.22 &&
    septumPlane < 0 &&
    dSvc >= 0 &&
    dIvc >= 0 &&
    dCs >= 0 &&
    dRaa >= 0
  ) {
    const oxT = x - V.tv.cx,
      oyT = y - V.tv.cy;
    const dOrifice = Math.hypot(oxT, oyT) - V.tv.R / skirtWarpScale(V.tv, oxT, oyT);
    const toward = oxT / Math.max(1e-6, Math.hypot(oxT, oyT));
    const septalSide = Math.min(
      1,
      Math.max(0, (toward - RA_VESTIBULE_SEPTAL_COS) / RA_VESTIBULE_SEPTAL_RAMP),
    );
    if (
      z > czR &&
      smin(dFreeRa, dOrifice, 1e-3 + RA_VESTIBULE_UNION_CM * septalSide) < 0 &&
      c.laCav > raLaMin
    ) {
      const past = z > A.tvCenter.z + hp.tvZ + tvOff;
      setSample(
        out,
        Tissue.Blood,
        dFreeRa - 0.22,
        (x - ra.x) / rr.x,
        (y - ra.y) / rr.y,
        (z - czR) / rzR,
        x,
        y,
        z,
        0,
        past ? Structure.RvCavity : Structure.RaCavity,
      );
      return true;
    }
    setSample(
      out,
      Tissue.Myocardium,
      -Math.min(dFreeRa, 0.22 - dFreeRa),
      (x - ra.x) / rr.x,
      (y - ra.y) / rr.y,
      (z - czR) / rzR,
      x,
      y,
      z,
      0,
      Structure.RaWall,
    );
    return true;
  }
  const zTvAnn = A.tvCenter.z + hp.tvZ + tvOff;
  const oxT = x - V.tv.cx,
    oyT = y - V.tv.cy;
  const dColumn = Math.max(Math.hypot(oxT, oyT) - V.tv.R / skirtWarpScale(V.tv, oxT, oyT), czR - z);
  c.raColumn = Math.max(dColumn, z - zTvAnn);
  if (
    !inSeptum &&
    z > czR - 0.22 &&
    z <= zTvAnn &&
    septumPlane < 0 &&
    dSvc >= 0 &&
    dIvc >= 0 &&
    dCs >= 0 &&
    dRaa >= 0 &&
    c.laCav > raLaMin
  ) {
    if (dColumn < 0) {
      setSample(
        out,
        Tissue.Blood,
        dColumn,
        (x - ra.x) / rr.x,
        (y - ra.y) / rr.y,
        (z - czR) / rzR,
        x,
        y,
        z,
        0,
        Structure.RaCavity,
      );
      return true;
    }
    if (dColumn < 0.22) {
      setSample(
        out,
        Tissue.Myocardium,
        -Math.min(dColumn, 0.22 - dColumn),
        oxT,
        oyT,
        0,
        x,
        y,
        z,
        0,
        Structure.RaWall,
      );
      return true;
    }
  }
  if (dRaa < 0 && !(nearRoot && (inSeptum || c.laCav <= raLaMin))) {
    setSample(out, Tissue.Blood, dRaa, 0, 1, 0, x, y, z, 0, Structure.RaAppendage);
    return true;
  }
  if (dRaa < LAA_WALL_CM) {
    setSample(
      out,
      Tissue.Myocardium,
      -Math.min(dRaa, LAA_WALL_CM - dRaa),
      0,
      1,
      0,
      x,
      y,
      z,
      0,
      Structure.RaWall,
    );
    return true;
  }
  // interatrial septum: slab between the clipped atria wherever either atrium reaches the septal plane
  if (x <= xFace && x >= xFace - tIas && z < zAnn + 0.4) {
    const nearRa =
      sdEllipsoid(xIas, y, z, ra.x, ra.y, czR, rr.x * boR * raC, rr.y * boR * raC, rzR) < 0.45;
    if (
      (nearLaBody || (nearRa && relL < 0.5)) &&
      dCs >= 0 &&
      leftAtrialMouthLumen(c, x, y, z, la, lr, bo, czL) >= 0
    ) {
      setSample(
        out,
        Tissue.Myocardium,
        -Math.min(xFace - x, x - (xFace - tIas)),
        1,
        0,
        0,
        x,
        y,
        z,
        0,
        Structure.InteratrialSeptum,
      );
      return true;
    }
  }
  return classifyAtrialVessels(c, la, lr, bo, czL, dSvc, dIvc, dCs);
}
