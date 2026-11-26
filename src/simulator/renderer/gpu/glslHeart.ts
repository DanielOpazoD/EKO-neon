/**
 * GLSL port of `classifyHeart` (heartModel.ts). Mirrors the CPU classifier block by block; the
 * equivalence test (e2e/gpu-equivalence.spec.ts) compares both on the canonical views.
 *
 * Modularized into domain chunks:
 * - `glslHeartConstants.ts` — projected TS anatomical constants & `Sample` struct
 * - `glslHeartValves.ts` — skirts, mitral apparatus, aortic root/cusps, pulmonary cusps, AHA segments
 * - `glslHeartChambers.ts` — LV/RV geometry, aortic extension, pulmonary trunk, veins, coronary sinus
 * - `glslHeart.ts` — `classifyHeart` assembly
 */
import { GLSL_HEART_CONSTANTS } from './glslHeartConstants';
import { GLSL_HEART_VALVES } from './glslHeartValves';
import { GLSL_HEART_CHAMBERS } from './glslHeartChambers';

export { GLSL_HEART_CONSTANTS, GLSL_HEART_VALVES, GLSL_HEART_CHAMBERS };

export const GLSL_HEART = `${GLSL_HEART_CONSTANTS}${GLSL_HEART_VALVES}${GLSL_HEART_CHAMBERS}
bool classifyHeart(vec3 p0, out Sample s) {
  s.segment = 0;
  s.sdf = FAR_FROM_HEART_CM;
#ifndef VOLUME_HEART
  // over a volume heart this classifier is asked for the valve apparatus only, which the arch never holds, and the great
  // vessels come from classifyGreatVessels: one inlined copy of the arch's loop, not two, which SwiftShader cannot take
  // (decision 375: the second copy left pass A of the volume variant empty)
  if (classifyAorticExtension(p0, s)) return true;
#endif
  float vascularDistance = s.sdf;
  // the point after the swing, the bounding sphere and the posterior column, which classify.ts computes before its
  // blocks (cpuGpuParity.test.ts runs both)
  // src/simulator/anatomy/classify.ts: classifyChambers
  vec3 p = vec3(p0.x - SWING_X, p0.y, p0.z);
  float x = p.x, y = p.y, z = p.z;
  vec3 bd = p - vec3(BOUND_CX, BOUND_CY, BOUND_CZ);
  if (dot(bd, bd) > BOUND_R * BOUND_R) { s.sdf = vascularDistance; return false; }
  // the descending aorta and the vertebral body, which the heart yields to (decision 273)
  float colDist = posteriorColumnDistance(p0.x, p0.y, p0.z, COL_UX, COL_UY, COL_UZ, COL_AX, COL_AY, COL_AZ, COL_SX, COL_SY, COL_SZ);
  float zAnn = ZANN;
  // end of classifyChambers

  // ---------- aortic root coordinates ----------
  // src/simulator/anatomy/classify/root.ts: rootCoordinates
  float rootT = -99.0, rootRr = 0.0, rootR = 0.0, rootPhi = 0.0;
  vec3 rootQ = vec3(0.0);
  vec3 avC = vec3(AV_CX, AV_CY, AV_CZ);
  vec3 ax = vec3(AV_AXX, AV_AXY, AV_AXZ);
  {
    float czz = avC.z + zAnn * ROOT_EXCURSION;
    vec3 d = vec3(x - avC.x, y - avC.y, z - czz);
    float t = dot(d, ax);
    if (t > -1.6 && t < AO_ROOT_END) {
      float bend = rootBend(t);
      rootQ = d - ax * t - vec3(AV_BX, AV_BY, AV_BZ) * bend;
      rootRr = length(rootQ);
      rootT = t;
      rootPhi = atan(dot(rootQ, vec3(AV_E2X, AV_E2Y, AV_E2Z)), dot(rootQ, vec3(AV_E1X, AV_E1Y, AV_E1Z)));
      rootR = rootRadiusAt(t, rootPhi);
    }
  }
  bool inRootLumen = rootT >= -0.05 && rootRr < rootR;
  bool inOutflowLumen = rootT > -1.6 && rootRr < rootR;
  // end of rootCoordinates

  // ---------- valves ----------
  {
    float dM, fr;
    int lf;
    vec3 nM;
    float t = mitralDistance(p, dM, fr, lf, nM);
    if (dM < t) {
      setSample(s, T_VALVE, dM - t, nM, p, MV_CALC, lf == 0 ? S_MV_ANT : S_MV_POST);
      return true;
    }
  }
  if (rootT > -0.5 && rootRr < rootR + 0.02) {
    float dA, frA;
    vec2 nA;
    float hA = aorticCuspDistance(rootT, rootRr, rootPhi, dA, frA, nA);
    if (dA < hA) {
      vec3 u = rootQ / max(rootRr, 1e-6);
      setSample(s, T_VALVE, dA - hA, u * nA.x + ax * nA.y, p, AV_CALC, S_AV);
      return true;
    }
  }
  if (AVC_OPEN < 1.0 && rootT > 0.0 && rootT < AVC_HCOMM && rootRr < rootR * 0.97) {
    vec2 band;
    float axialDistance = 1e3;
    if (aorticContactBand(rootT, rootRr, rootPhi, band)) axialDistance = max(max(band.x - rootT, rootT - band.y), 0.0);
    if (axialDistance < AV_COAPT_HALF) {
      float n = CUSP_COUNT;
      float per = TWO_PI / n;
      float dphi = mod(mod(rootPhi - 0.5 - PI / n, per) + per, per);
      if (dphi > PI / n) dphi = per - dphi;
      float dist = length(vec2(rootRr * sin(dphi), axialDistance));
      if (dist < AV_COAPT_HALF) {
        vec3 u = rootQ / max(rootRr, 1e-6);
        setSample(s, T_VALVE, dist - AV_COAPT_HALF, cross(ax, u), p, AV_CALC, S_AV);
        return true;
      }
    }
  }
  bool outsideAorticRoot = rootT <= -1.6 || rootRr > rootR + 0.22;
  if (outsideAorticRoot && sdCapsule(p, vec3(RVOT_MX, RVOT_MY, RVOT_MZ + PV_Z), vec3(PA_EX, PA_EY, PA_EZ), PA_R + 0.02) < 0.0) {
    for (int i = 0; i < 3; i++) {
      vec3 w = vec3(P(PV_W_BASE + i * 3), P(PV_W_BASE + i * 3 + 1), P(PV_W_BASE + i * 3 + 2));
      float fr;
      float dd = sdCuspChain(p, PV_SEGS_BASE + i * 12, PV_SEGLEN, w, PV_HALF, 0.75, fr);
      float t = PV_T * (1.0 - 0.3 * fr) * 0.5 + 0.03;
      if (dd < t) {
        int o = PV_SEGS_BASE + i * 12 + min(1, int(floor(fr * 2.0))) * 6;
        vec3 d = vec3(P(o + 3), P(o + 4), P(o + 5));
        setSample(s, T_VALVE, dd - t, cross(d, w), p, 0.0, S_PV);
        return true;
      }
    }
  }
  {
    float dS, fr;
    int zn;
    vec3 nS;
    vec3 c = vec3(TVS_CX, TVS_CY, TVS_CZ);
    float t = skirtDistance(p, c, TVS_R, TVS_ZONES_BASE, TVS_PROF_BASE, int(TVS_NZ + 0.5), TVS_CLOSED, TVS_BLEND, TVS_T, TVS_SADDLE, dS, fr, zn, nS);
    if (dS < t) {
      setSample(s, T_VALVE, dS - t, nS, p, 0.0, int(P(TVS_ZONES_BASE + zn * 6 + 5) + 0.5));
      return true;
    }
  }
  {
    float dR = mvAnnulusDistance(p, 0.11);
    if (dR < 0.0) {
      setSample(s, T_FIBROUS, dR, vec3(x - MVL_CX, y - MVL_CY, 0.0), p, 0.15 * MV_CALC, S_MV_ANN);
      return true;
    }
    vec3 q = vec3(TV_RING_X, TV_RING_Y, TV_RING_Z);
    vec2 dq = p.xy - q.xy;
    float dT = sdTorusZ(vec3(q.xy + dq * tvWarpScale(dq), z - tvOffsetAt(p.xy)), q, TV_RING_R, 0.09);
    if (dT < 0.0) {
      setSample(s, T_FIBROUS, dT, vec3(x - q.x, y - q.y, 0.0), p, 0.0, S_TV_ANN);
      return true;
    }
  }
  for (int i = 0; i < 10; i++) {
    int o = CHORDAE_BASE + i * 6;
    vec3 ca = vec3(P(o), P(o + 1), P(o + 2)), cb = vec3(P(o + 3), P(o + 4), P(o + 5));
    float d = sdCapsule(p, ca, cb, 0.045);
    if (d < 0.0) {
      // decision 227: the cord's axis
      setSample(s, T_CHORDAE, d, cb - ca, p, 0.0, S_CHORDAE);
      return true;
    }
  }

#ifndef VOLUME_HEART
  // over a volume heart only the valve apparatus above is asked for: the chambers below stay out of that variant, whose
  // pass A SwiftShader leaves empty past its size (decisions 375 and 381)
  // ---------- LV cavity & wall ----------
  // the azimuth exactly, where the CPU reads it with fastAtan2 (0.005 rad at most): what follows it mirrors
  // classify/leftVentricle.ts from the azimuth on (cpuGpuParity.test.ts runs both at the CPU's)
  float az = atan(y, x);
  // src/simulator/anatomy/classify/leftVentricle.ts: classifyLeftVentricle
  float levelFrac = clamp((z - zAnn) / max(LENGTH_NOW, 1.0), 0.0, 1.0);
  float septalShift = septalShiftAt(SEPTAL_SHIFT, az, levelFrac);
  float xs = x - septalShift;
  vec3 n0;
  float dProf = lvCavitySdf(xs, y, z, n0);
  // regional wall motion, continuous across the segment borders (lvRegionalMotion, as on the CPU): an akinetic wall keeps
  // its end-diastolic radius and thickness
  float motion = lvMotionAt(az, levelFrac);
  float regional = regionalCavityOffset(az, levelFrac, motion);
  float rs = RADIAL_SCALE, ls = LONG_SCALE;
  // the wall's own radial scale, so that its speckle and trabeculae stay with the tissue (leftVentricle.ts)
  float rsWall = rs + (1.0 - motion) * (1.0 - rs);
  float trab = levelFrac > 0.45 ? 0.2 * min(1.0, (levelFrac - 0.45) / 0.35) * (lat(vec3((x / rsWall) * 2.6 + 11.3, (y / rsWall) * 2.6 + 2.9, ((z - LV_LEN) / ls) * 1.1 + 6.1), 3) - 0.5) : 0.0;
  float dCavR = smax(dProf - regional + trab, zAnn - z, 0.6);
  float tFull = wallThicknessAt(az, levelFrac, motion);
  // decision 223
  float tNow = tFull * septalCrestFactor(az, z - zAnn);
  float crestLoss = tFull - tNow;
  // mitral inflow: cavity and wall are the smooth union of the profile with the narrowing annular outline
  bool inRootTube = rootT > -1.6 && rootRr < rootR + 0.2;
  float zHinge = mvHingeZ(p.xy);
  float dInflow = inRootTube ? 1e3 : mvOutlineSdf(p.xy) + 0.04 + mvInflowTaper(z - zHinge);
  float dLvBlood = smin(dCavR, dInflow, 0.3);
  if (dLvBlood < 0.0) {
    float dPa = sdRoundCone(p, vec3(P(PAPS_BASE), P(PAPS_BASE + 1), P(PAPS_BASE + 2)), vec3(P(PAPS_BASE + 3), P(PAPS_BASE + 4), P(PAPS_BASE + 5)), P(PAPS_BASE + 6), P(PAPS_BASE + 7));
    float dPm = sdRoundCone(p, vec3(P(PAPS_BASE + 8), P(PAPS_BASE + 9), P(PAPS_BASE + 10)), vec3(P(PAPS_BASE + 11), P(PAPS_BASE + 12), P(PAPS_BASE + 13)), P(PAPS_BASE + 14), P(PAPS_BASE + 15));
    float dPap = min(dPa, dPm);
    if (dPap < 0.0) {
      setSample(s, T_MYO, dPap, vec3(x, y, 0.0), vec3(x / rs, y / rs, z / ls), 0.0, S_PAP);
      return true;
    }
    setSample(s, T_BLOOD, dLvBlood, n0, vec3(x / rs, y / rs, (z - LV_LEN) / ls), 0.0, (dCavR >= 0.0 && z < zHinge) ? S_LA_CAV : S_LV_CAV);
    return true;
  }
  float wallT = tNow;
  // the regional offset moves the profile, not the inflow column, as in the cavity (leftVentricle.ts)
  float dEllR = smin(dProf - regional, dInflow, 0.3);
  // apical to the hinge at this azimuth, not to the mean annular plane (decision 327)
  if (dEllR + trab >= 0.0 && dEllR < wallT && z >= zHinge - LV_WALL_BASAL_OVERHANG_CM && !inOutflowLumen) {
    // the wall label follows the AHA wall of the segment, bounded by the RV insertions (lvSegments.ts, decision 154)
    float segCode = lvSegmentCode(az, levelFrac, RV_AZA, RV_AZP);
    int wall = int(lvWallKind(segCode) + 0.5);
    int structure = wall == 1 ? S_LV_SEPT : wall == 2 ? S_LV_ANT : wall == 3 ? S_LV_INF : S_LV_LAT;
    if (z > LV_LEN - 0.6) structure = S_LV_APEX;
    bool nearEpi = wallT - dEllR < dEllR;
    // only the smooth epicardium reflects coherently; the trabeculated endocardium scatters (decision 144)
    float dIn = nearEpi ? -(wallT - dEllR) : -wallT;
    float sg = nearEpi ? 1.0 : -1.0;
    setSample(s, T_MYO, dIn, sg * n0, vec3(x / rsWall, y / rsWall, (z - LV_LEN) / ls), 0.0, structure);
    s.transmural = clamp(dEllR / wallT, 0.0, 1.0);
    s.segment = int(segCode + 0.5);
    return true;
  }
  bool inAnnularRegion = dEllR < 0.0 && z < zAnn && !inRootLumen;
  if (inAnnularRegion) {
    if (mvInsideOutline(p.xy)) {
      setSample(s, T_BLOOD, -0.3, vec3(0.0, 0.0, 1.0), p, 0.0, S_LA_CAV);
      return true;
    }
  }
  // end of classifyLeftVentricle

  // ---------- aortic root / LVOT ----------
  // src/simulator/anatomy/classify/aorticRoot.ts: classifyAorticRoot
  if (rootT > -1.6) {
    float wall = 0.2;
    if (rootRr < rootR) {
      setSample(s, T_BLOOD, rootRr - rootR, rootQ / rootRr, vec3(x, y, z - zAnn * ROOT_EXCURSION), 0.0, rootT < 0.0 ? S_LVOT : S_AO_ROOT);
      return true;
    }
    if (rootRr < rootR + wall) {
      float dIn = -min(rootRr - rootR, rootR + wall - rootRr);
      setSample(s, T_VESSEL, dIn, rootQ / rootRr, p, 0.0, S_AO_ROOT);
      return true;
    }
  }
  // basal of the hinge at this azimuth only in front of the atrium's septal corner, the curtain itself (decision 327)
  // the curtain in front of the corner's bound is a sheet with its own faces and normal (decision 328)
  float inFront = (x - MVL_CX) * MVL_UX + (y - MVL_CY) * MVL_UY - (MVL_D + LA_CORNER_ANTERIOR_MARGIN_CM);
  if (inAnnularRegion && z > zAnn - 1.2) {
    if (inFront > 0.0 && rootT > -90.0) {
      float toLumen = rootRr - rootR - AORTIC_ROOT_WALL_CM;
      setSample(s, T_FIBROUS, -max(0.0, min(inFront, toLumen)), vec3(MVL_UX, MVL_UY, 0.0), p, 0.0, S_LV_SEPT);
      return true;
    }
    if (z >= zHinge - LV_WALL_BASAL_OVERHANG_CM) {
      setSample(s, T_FIBROUS, -0.15, vec3(0.0, 0.0, 1.0), p, 0.0, S_LV_SEPT);
      return true;
    }
  }
  // end of classifyAorticRoot

  // the atria and what they leave the pericardium (cpuGpuParity.test.ts runs both)
  // src/simulator/anatomy/classify/atria.ts: classifyAtria
  // the base the ventricle vacated as the annulus descended, atrium now (decision 133); read again by the pericardium
  float raSleeve = 1e3;
  float raBody = 1e3;
  // the left atrium's epicardium, read by the pericardium (decision 273)
  float laEpi = 1e3;
  // decision 323
  float raaEpi = 1e3;
  // decision 379
  float raColumn = 1e3;
  // ---------- atria ----------
  vec3 la = vec3(LA_CX, LA_CY, LA_CZ);
  vec3 lr = vec3(LA_RX, LA_RY, LA_RZ);
  vec3 ra = vec3(RA_CX, RA_CY, RA_CZ);
  vec3 rar = vec3(RA_RX, RA_RY, RA_RZ);
  float bo = atrialScale(LA_BOOSTER, LA_RESERVOIR, CONTRACTION);
  // the right atrium's own (decision 330)
  float boR = atrialScale(LA_BOOSTER, RA_RESERVOIR, CONTRACTION);
  float czL, rzL, czR, rzR;
  {
    float zTop = la.z - lr.z;
    float zBottom = zAnn + 0.25;
    czL = (zTop + zBottom) / 2.0;
    rzL = (zBottom - zTop) / 2.0;
    // decision 369: the fossa follows the base down and the septum bows toward the right atrium as the left fills
    float fo = length(vec2((y - FOSSA_Y) / 0.6, (z - FOSSA_Z - IAS_FOSSA_DESCENT_SHARE * zAnn) / 0.7));
    float xIas = IAS_X - iasBow(fo, CONTRACTION);
    float tIas = iasThickness(fo);
    // the septum's left face is flat; the fossa shapes its right side (decision 327)
    float xFace = xIas + IAS_LEFT_FACE_CM;
    // decision 341: beside the aortic root the body keeps its end-systolic size
    float rootOuter = rootRr - rootR - AORTIC_ROOT_WALL_CM;
    float boL = bo + (1.0 - bo) * laRootHold(rootT, rootOuter, x - (IAS_X + IAS_LEFT_FACE_CM));
    float dEllLa = sdEllipsoid(p, vec3(la.x, la.y, czL), vec3(lr.x * boL, lr.y * boL, rzL));
    float dFreeLa = smax(smax(dEllLa, la.y - LA_POSTERIOR_RATIO * lr.y * boL - y, 0.6), zTop + 0.15 * rzL - z, 0.5);
    // the body at the scale of the moment, without the hold: what the right atrium's release reads (decision 341)
    float dFreeLaNow = boL == bo ? dFreeLa : smax(smax(sdEllipsoid(p, vec3(la.x, la.y, czL), vec3(lr.x * bo, lr.y * bo, rzL)), la.y - LA_POSTERIOR_RATIO * lr.y * bo - y, 0.6), zTop + 0.15 * rzL - z, 0.5);
    laEpi = sdCapsule(p, vec3(la.x + lr.x * 0.55, la.y + lr.y * 0.55, czL + 0.4), vec3(la.x + lr.x * 0.95, la.y + lr.y * 0.55 + 2.0, czL + 0.9), 0.55 * bo) - LAA_WALL_CM; // the body's face below (decision 378)
    // the septal corner above the annulus belongs to the atrium (decision 318)
    // only toward the septum, where the atrium's own section reaches at this height
    vec3 pc = vec3(la.x, y, z);
    float dCross = smax(smax(sdEllipsoid(pc, vec3(la.x, la.y, czL), vec3(lr.x * boL, lr.y * boL, rzL)), la.y - LA_POSTERIOR_RATIO * lr.y * boL - y, 0.6), zTop + 0.15 * rzL - z, 0.5);
    float uL = (y - la.y) / (lr.y * boL);
    float wL = (z - czL) / rzL;
    float gap = la.x - lr.x * boL * sqrt(max(0.0, 1.0 - uL * uL - wL * wL)) - xFace;
    // in the corner the atrium keeps, down to the hinge at this azimuth, the section it has LA_CORNER_HEIGHT_CM above the
    // annulus (decision 327)
    float zRef = zAnn - LA_CORNER_HEIGHT_CM;
    float wRef = (zRef - czL) / rzL;
    float gapRef = la.x - lr.x * boL * sqrt(max(0.0, 1.0 - uL * uL - wRef * wRef)) - xFace;
    float cornerOff = laMedialFaceOffset(gapRef, LA_CORNER_FLAT_CM);
    // the base keeps the body's full sections (decision 330)
    float dCrossRef = smax(smax(sdEllipsoid(vec3(la.x, y, zRef), vec3(la.x, la.y, czL), vec3(lr.x, lr.y, rzL)), la.y - LA_POSTERIOR_RATIO * lr.y - y, 0.6), zTop + 0.15 * rzL - zRef, 0.5);
    float zHingeA = zHinge;
    // and in front it ends at the aortomitral curtain, the straight anterior edge of the annulus (decision 222)
    float rootClearFaceLine = (xFace - MVL_CX) * MVL_UX + (y - MVL_CY) * MVL_UY - (MVL_D + LA_CORNER_ANTERIOR_MARGIN_CM);
    // and the base flares from the hinge all around the annulus (decision 328)
    float xRead = max(xFace, x - LA_BASE_BOUND_READ_CM);
    float rootClearBaseLine = (xRead - MVL_CX) * MVL_UX + (y - MVL_CY) * MVL_UY - (MVL_D + LA_CORNER_ANTERIOR_MARGIN_CM);
    // decision 222: the right atrium's release, read at the curtain's straight bound
    float qRel = clamp((max(dFreeLaNow, rootClearFaceLine) - RA_SEPTAL_RELEASE_LA_CM) / RA_SEPTAL_RELEASE_RAMP_CM, 0.0, 1.0);
    float relL = qRel * qRel * (3.0 - 2.0 * qRel);
    // decision 335: above the annulus, near the round root, the atrium comes up to a wall of its outer wall, giving up the
    // ground as the right atrium is released
    bool atSinuses = rootT >= 0.0 && rootT <= LA_ROOT_TOP_CM;
    float rootWall = atSinuses ? max(LA_WALL_CM + LA_ROOT_GAP_CM - rootOuter, rootOuter - LA_ROOT_NEAR_CM) + LA_ROOT_RELEASE_CM * relL : 1e3;
    float rootClearFace = min(rootClearFaceLine, rootWall);
    float rootClearBase = min(rootClearBaseLine, rootWall);
    float dCrossEq = smax(smax(sdEllipsoid(vec3(la.x, y, czL), vec3(la.x, la.y, czL), vec3(lr.x, lr.y, rzL)), la.y - LA_POSTERIOR_RATIO * lr.y - y, 0.6), zTop + 0.15 * rzL - czL, 0.5);
    float dCorner = laBaseSdf(x, la.x, xFace, cornerOff, z, zRef, zHingeA, dCrossRef, dCrossEq, rootClearBase, mvOutlineSdf(p.xy), laBaseLateralShare(x - MVL_CX, y - MVL_CY));
    // and over the fossa ovalis it meets the septum (decision 319)
    // the fossa fill reads the body's full section (decision 330)
    float dCrossFull = smax(smax(sdEllipsoid(pc, vec3(la.x, la.y, czL), vec3(lr.x, lr.y, rzL)), la.y - LA_POSTERIOR_RATIO * lr.y - y, 0.6), zTop + 0.15 * rzL - z, 0.5);
    float dFossaLa = max(max(xFace - x, x - la.x), max(dCrossFull, (fo - FOSSA_REACH) * 0.6));
    // and wherever its body comes close to the septal plane (decision 327)
    float faceOff = laMedialFaceOffset(gap, LA_SEPTAL_FLAT_CM);
    float dFlat = max(max(xFace + faceOff - x, x - la.x), dCross);
    // the body with its fills, before the septal clip
    // decision 330: the posterior wall on the descending aorta, the prism of the body's full section behind its centre
    vec3 pn = vec3(LA_PNX, LA_PNY, LA_PNZ);
    float tPost = dot(p, pn) - LA_PD;
    float sPost = dot(p - vec3(la.x, la.y, czL), pn);
    float sFront = dot(p.xy - la.xy, pn.xy);
    // in front of the prism's front by more than the union's blend plus the body's distance the prism cannot change it
    float dPost;
    if (-sFront >= max(dFreeLa, 0.0) + 0.6) dPost = dFreeLa + 0.6;
    else {
      vec3 pSil = p - sPost * pn;
      float fp = laFootprintScale(bo);
      float dSil = sdEllipsoid(vec3(pSil.x, max(pSil.y, la.y), min(pSil.z, czL)), vec3(la.x, la.y, czL), vec3(lr.x * fp, lr.y * fp, rzL));
      dPost = smax(smax(max(max(max(tPost, -sFront), dSil), z - (zHinge - 0.5 * LA_WALL_CM)), LA_FLOOR_Y_CM - y, 0.3), zTop + 0.15 * rzL - z, 0.5);
    }
    // in front of the curtain the body keeps RA_LA_MIN_CM from the septal plane (decision 330)
    float raReach = min(1.0, RA_SEPTAL_RELEASE_CM * relL / tIas);
    float holdShare = atSinuses && rootOuter < LA_ROOT_NEAR_CM ? raReach : 1.0;
    float frontClear = min(rootClearFaceLine, xFace + RA_LA_MIN_CM * holdShare - x);
    float faceKeep = rootClearFaceLine > 0.0 ? xFace - x : -1e3;
    // decision 366: the lateral wall stands on the lateral hinge; above its top it adds nothing (decision 381)
    float dLatWall = max(laLateralWallSdf(x, y, zHinge - z, (czL - z) / rzL, la.x, MVL_CX + MVL_R + LA_LATERAL_HINGE_CM, MVL_CY + LA_LATERAL_WALL_AHEAD_CM, LA_LATERAL_WALL_HOLD, LA_LATERAL_WALL_TOP, LA_LATERAL_WALL_FRONT_CM), czL - LA_LATERAL_WALL_TOP * rzL - z);
    float dBodyPost = smin(smin(dFreeLa, dPost, 0.6), dLatWall, laLateralWallBlend(zHinge - z));
    float dLaBody = max(max(max(min(min(dBodyPost, dCorner), min(dFossaLa, dFlat)), tPost), frontClear), faceKeep);
    laEpi = min(laEpi, dLaBody - LA_WALL_CM);
    // negative where a fill leaves the atrium within LA_TOUCH_CM of the plane
    float laTouch = min(max(dCross, faceOff - LA_TOUCH_CM), min(max(max(max(dCrossRef, rootClearFace), cornerOff - LA_TOUCH_CM), max(zRef - z, z - zHingeA)), max(dCrossFull, (fo - FOSSA_REACH) * 0.6)));
    bool nearLa = laTouch < 0.0 || (rootClearFace < 0.0 && sdEllipsoid(vec3(xIas, y, z), vec3(la.x, la.y, czL), vec3(lr.x * bo, lr.y * bo, rzL)) < 0.45);
    float d = max(max(max(min(min(min(smax(dBodyPost, xFace - x, 0.3), dCorner), dFossaLa), dFlat), tPost), frontClear), faceKeep);
    // decision 335: and wherever the body with its fills comes within LA_TOUCH_CM of its face beside the point
    bool nearRoot = atSinuses && rootOuter < LA_ROOT_NEAR_CM;
    bool nearLaBody = nearLa || (nearRoot && x <= xFace && d < tIas + LA_TOUCH_CM);
    if (d < 0.0) {
      setSample(s, T_BLOOD, d, vec3((x - la.x) / lr.x, (y - la.y) / lr.y, (z - czL) / rzL), p, 0.0, S_LA_CAV);
      return true;
    }
    // decision 321
    float dCs = coronarySinusDistance(p, la, lr, bo, czL, rzL, zAnn, d);
    laEpi = min(laEpi, dCs - CS_WALL_CM);
    // decision 320
    if (dLaBody < LA_WALL_CM && x > xFace && leftAtrialMouthLumen(p, la, lr, bo, czL, rzL, colDist) >= 0.0) {
      setSample(s, T_MYO, -min(dLaBody, LA_WALL_CM - dLaBody), vec3((x - la.x) / lr.x, (y - la.y) / lr.y, (z - czL) / rzL), p, 0.0, S_LA_WALL);
      return true;
    }
    float zTopR = ra.z - rar.z + RA_ROOF_DESCENT_SHARE * TVZ;
    // the atrium ends at the annulus (decision 64): mirrors classifyHeart
    float tvOff = tvOffsetAt(p.xy);
    float zBotR = TV_CZ + TVZ + tvOff + 0.03;
    czR = (zTopR + zBotR) / 2.0;
    rzR = (zBotR - zTopR) / 2.0;
    float raC = raCollapseScale(RA_COLLAPSE);
    float dEllRa = sdEllipsoid(p, vec3(ra.x, ra.y, czR), vec3(rar.x * boR * raC, rar.y * boR * raC, rzR));
    float dFreeRa = smax(dEllRa, ra.y - 0.8 * rar.y * boR - y, 0.6);
    // decision 367: the lateral wall rises from the lateral tricuspid hinge (the ring shortens about its septal edge)
    float tvRNow = TV_R * (1.0 - tvShortening(CONTRACTION, RV_RADIAL_SCALE));
    float tvLat = TV_CX + TV_R - 2.0 * tvRNow - RA_LATERAL_HINGE_CM;
    float dLatR = max(laLateralWallSdf(-x, y, zBotR - z, (czR - z) / rzR, -ra.x, -tvLat, TV_CY, RA_LATERAL_WALL_HOLD, RA_LATERAL_WALL_TOP, 0.0), czR - RA_LATERAL_WALL_TOP * rzR - z);
    dFreeRa = smin(dFreeRa, dLatR, laLateralWallBlend(zBotR - z));
    raBody = dFreeRa;
    // decision 218
    vec3 amFrom = ra + vec3(RA_ANTEROMEDIAL_FROM_X, RA_ANTEROMEDIAL_FROM_Y, RA_ANTEROMEDIAL_FROM_Z);
    vec3 amTo = vec3(AV_CX + RA_ANTEROMEDIAL_TO_X, AV_CY + RA_ANTEROMEDIAL_TO_Y, AV_CZ + zAnn * ROOT_EXCURSION + RA_ANTEROMEDIAL_TO_Z);
    dFreeRa = smin(dFreeRa, sdCapsule(p, amFrom, amTo, RA_ANTEROMEDIAL_R_CM * boR), RA_ANTEROMEDIAL_BLEND_CM);
    // decision 222
    vec3 msTo = vec3(AV_CX + RA_MEMBRANOUS_TO_X, AV_CY + RA_MEMBRANOUS_TO_Y, AV_CZ + zAnn * ROOT_EXCURSION + RA_MEMBRANOUS_TO_Z);
    float dMs = smax(sdCapsule(p, amTo, msTo, RA_MEMBRANOUS_R_CM * boR), z - zBotR, 0.1);
    dMs = smax(dMs, -max(dEllR - wallT - 0.05, zAnn - 0.25 - z), 0.1);
    dFreeRa = smin(dFreeRa, dMs, 0.3);
    // decision 223
    if (crestLoss > 0.0) {
      float dEpiC = dEllR - wallT;
      dFreeRa -= crestLoss * clamp(dEpiC / 0.05, 0.0, 1.0) * clamp(1.0 - dEpiC / RA_CREST_REACH_CM, 0.0, 1.0);
    }
    if (TVZ > 0.05) {
      vec4 rr0 = rvRadii(az, z, 0.0, 0.0, 0.0);
      float u0 = rr0.y;
      if (u0 > 0.0 && u0 < 1.0) {
        float r0 = length(p.xy);
        // decision 220
        vec2 dT = vec2(x - TVS_CX, y - TVS_CY);
        float rhoT = length(dT);
        float tvOffEd = tvOff - TVS_LIFT - (rhoT > 1e-6 ? TVS_TILTC * (x - TVS_CX) / rhoT : TVS_TILTC);
        raSleeve = max(max(max(rr0.x + 0.1 - r0, r0 - (rr0.z - RV_FW)), max(rvFloorZ(TV_CZ, 0.0, 0.0, u0, tvOffEd) - z, z - rvFloorZ(TV_CZ, TVZ, PV_Z, u0, tvOff))), rhoT * tvWarpScale(dT) - (TVS_R + RA_SLEEVE_MARGIN_CM));
      }
    }
    dFreeRa = min(dFreeRa, raSleeve);
    // decision 320
    // decision 330: over the fossa the membrane stands between the right atrium and the veins' mouths
    float rightVeinClear = min(RA_RIGHT_VEIN_CLEARANCE_CM - rightVeinDistance(p, la, lr, bo, czL, rzL), (fo - FOSSA_REACH) * 0.6);
    dFreeRa = smax(dFreeRa, rightVeinClear, 0.2);
    // decision 222
    float xRaFace = xFace - tIas;
    float septumPlane = x - xRaFace - RA_SEPTAL_RELEASE_CM * relL;
    // the right atrium meets its face of the septum wherever its body comes close (decision 327)
    float uR = (y - ra.y) / (rar.y * boR * raC);
    float wR = (z - czR) / rzR;
    float gapR = xRaFace - (ra.x + rar.x * boR * raC * sqrt(max(0.0, 1.0 - uR * uR - wR * wR)));
    float dCrossRa = smax(sdEllipsoid(vec3(ra.x, y, z), vec3(ra.x, ra.y, czR), vec3(rar.x * boR * raC, rar.y * boR * raC, rzR)), ra.y - 0.8 * rar.y * boR - y, 0.6);
    float dFlatRa = max(max(max(x - (xRaFace - laMedialFaceOffset(gapR, LA_SEPTAL_FLAT_CM)), ra.x - x), max(dCrossRa, laTouch)), rightVeinClear);
    // and never within the septum's thickness of the left atrial blood (decision 327)
    float raLaMin = min(tIas, RA_LA_MIN_CM);
    float dR = max(min(smax(dFreeRa, septumPlane, 0.3), dFlatRa), raLaMin - d);
    // and not where the septum stands (decision 327)
    bool inSeptum = x <= xFace && x >= xFace - tIas && nearLaBody;
    if (dR < 0.0 && !inSeptum) {
      setSample(s, T_BLOOD, dR, vec3((x - ra.x) / rar.x, (y - ra.y) / rar.y, (z - czR) / rzR), p, 0.0, S_RA_CAV);
      return true;
    }
    // decision 323
    // decision 389: its own emptying, and against the ascending aorta and the root only its wall
    float boA = appendageScale(LA_BOOSTER, RA_RESERVOIR, CONTRACTION);
    float dRaa = sdRoundCone(p, vec3(RAA_BX, RAA_BY, RAA_BZ), vec3(AV_CX + RAA_TIP_FROM_AV_X, AV_CY + RAA_TIP_FROM_AV_Y, AV_CZ + RAA_TIP_FROM_AV_Z), RAA_BASE_R_CM * boA, RAA_TIP_R_CM * boA);
    dRaa = max(max(smax(dRaa, z - (zBotR - RAA_ABOVE_FLOOR_CM), 0.2), LAA_WALL_CM - vascularDistance), rootT > -90.0 ? LAA_WALL_CM - rootOuter : -1e3);
    raaEpi = dRaa - LAA_WALL_CM;
    // decision 219
    // decision 324: the entry segment and the rest that rises
    float dSvc = min(sdCapsule(p, vec3(SVC_AX, SVC_AY, SVC_AZ), vec3(SVC_MX, SVC_MY, SVC_MZ), SVC_R), sdCapsule(p, vec3(SVC_MX, SVC_MY, SVC_MZ), vec3(SVC_BX, SVC_BY, SVC_BZ), SVC_R));
    float dIvc = min(sdCapsule(p, vec3(IVC_AX, IVC_AY, IVC_AZ), vec3(IVC_BX, IVC_BY, IVC_BZ), IVC_R), sdCapsule(p, vec3(IVC_BX, IVC_BY, IVC_BZ), vec3(IVC_BX + IVC_HEPATIC_X, IVC_BY + IVC_HEPATIC_Y, IVC_BZ + IVC_HEPATIC_Z), IVC_R));
    if (!inSeptum && dFreeRa < 0.22 && septumPlane < 0.0 && dSvc >= 0.0 && dIvc >= 0.0 && dCs >= 0.0 && dRaa >= 0.0) {
      // no wall across the tricuspid orifice (decision 64): atrial blood up to the annular plane, ventricular past it
      // decision 219
      // decision 226
      vec2 dO = vec2(x - TVS_CX, y - TVS_CY);
      float sepS = clamp((dO.x / max(1e-6, length(dO)) - RA_VESTIBULE_SEPTAL_COS) / RA_VESTIBULE_SEPTAL_RAMP, 0.0, 1.0);
      if (z > czR && smin(dFreeRa, length(dO) - TVS_R / tvWarpScale(dO), 1e-3 + RA_VESTIBULE_UNION_CM * sepS) < 0.0 && d > raLaMin) {
        setSample(s, T_BLOOD, dFreeRa - 0.22, vec3((x - ra.x) / rar.x, (y - ra.y) / rar.y, (z - czR) / rzR), p, 0.0, z > TV_CZ + TVZ + tvOff ? S_RV_CAV : S_RA_CAV);
        return true;
      }
      setSample(s, T_MYO, -min(dFreeRa, 0.22 - dFreeRa), vec3((x - ra.x) / rar.x, (y - ra.y) / rar.y, (z - czR) / rzR), p, 0.0, S_RA_WALL);
      return true;
    }
    // the vestibule reaches the annulus, under a roof at the equator, with its wall rising from the ring (decision 379)
    vec2 dOC = vec2(x - TVS_CX, y - TVS_CY);
    float dColumn = max(length(dOC) - TVS_R / tvWarpScale(dOC), czR - z);
    raColumn = max(dColumn, z - (TV_CZ + TVZ + tvOff));
    if (!inSeptum && z > czR - 0.22 && z <= TV_CZ + TVZ + tvOff && septumPlane < 0.0 && dSvc >= 0.0 && dIvc >= 0.0 && dCs >= 0.0 && dRaa >= 0.0 && d > raLaMin) {
      vec2 dO = dOC;
      if (dColumn < 0.0) {
        setSample(s, T_BLOOD, dColumn, vec3((x - ra.x) / rar.x, (y - ra.y) / rar.y, (z - czR) / rzR), p, 0.0, S_RA_CAV);
        return true;
      }
      if (dColumn < 0.22) {
        setSample(s, T_MYO, -min(dColumn, 0.22 - dColumn), vec3(dO, 0.0), p, 0.0, S_RA_WALL);
        return true;
      }
    }
    // the right atrial appendage (decision 323)
    // decision 335: the appendage keeps out of the septum and off the left atrial blood
    if (dRaa < 0.0 && !(nearRoot && (inSeptum || d <= raLaMin))) {
      setSample(s, T_BLOOD, dRaa, vec3(0.0, 1.0, 0.0), p, 0.0, S_RAA);
      return true;
    }
    if (dRaa < LAA_WALL_CM) {
      setSample(s, T_MYO, -min(dRaa, LAA_WALL_CM - dRaa), vec3(0.0, 1.0, 0.0), p, 0.0, S_RA_WALL);
      return true;
    }
    if (x <= xFace && x >= xFace - tIas && z < zAnn + 0.4 && dCs >= 0.0 && leftAtrialMouthLumen(p, la, lr, bo, czL, rzL, colDist) >= 0.0) {
      if (nearLaBody || (relL < 0.5 && sdEllipsoid(vec3(xIas, y, z), vec3(ra.x, ra.y, czR), vec3(rar.x * boR * raC, rar.y * boR * raC, rzR)) < 0.45)) {
        setSample(s, T_MYO, -min(xFace - x, x - (xFace - tIas)), vec3(1.0, 0.0, 0.0), p, 0.0, S_IAS);
        return true;
      }
    }
    // appendage
    {
      vec3 a0 = vec3(la.x + lr.x * 0.55, la.y + lr.y * 0.55, czL + 0.4);
      vec3 a1 = vec3(la.x + lr.x * 0.95, a0.y + 2.0, czL + 0.9);
      float lob = 0.12 * (lat(vec3(x * 2.3 + 1.7, y * 2.3 + 4.2, z * 2.3 + 8.8), 3) - 0.5);
      float dApp = sdCapsule(p, a0, a1, 0.55 * bo + lob);
      if (dApp < 0.0) {
        setSample(s, T_BLOOD, dApp, vec3(0.0, 1.0, 0.0), p, 0.0, S_LAA);
        return true;
      }
      if (dApp < LAA_WALL_CM) {
        setSample(s, T_MYO, -min(dApp, LAA_WALL_CM - dApp), vec3(0.0, 1.0, 0.0), p, 0.0, S_LA_WALL);
        return true;
      }
    }
    // pulmonary veins: towards the hila from the lateral wall (left) and the posteromedial corner (right); mirror of
    // pulmonaryVeins.ts (decision 143)
    for (int i = 0; i < 4; i++) {
      float sx = (i == 0 || i == 2) ? -1.0 : 1.0;
      bool sup = i < 2;
      float dzN = sup ? PV_SUP_Z : PV_INF_Z;
      float kz = sqrt(1.0 - dzN * dzN);
      float t = sx > 0.0 ? (sup ? PV_LEFT_SUP_T : PV_LEFT_INF_T) : PV_RIGHT_T;
      float ea = lr.x * bo * kz, eb = lr.y * bo * kz;
      float pz = czL + dzN * rzL;
      if (sx > 0.0) t = veinOstiumAngle(t, la, ea, eb, pz);
      // decisions 330 and 337: a left vein's ostium stays on the atrium in front of the posterior wall plane (pulmonaryVeinSegment)
      vec3 o = vec3(la.x + sx * ea * cos(t), la.y - eb * sin(t), pz);
      float dPv = smax(sdCapsule(p, o, o + vec3(PV_COURSE[3 * i], PV_COURSE[3 * i + 1], PV_COURSE[3 * i + 2]), PV_RADIUS), 0.12 - colDist, 0.2);
      if (dPv < 0.0) {
        setSample(s, T_BLOOD, dPv, vec3(0.0, -1.0, 0.0), p, 0.0, S_PVEIN);
        return true;
      }
      // decision 370: the wall scatters as the atrial wall it continues
      if (dPv < PV_WALL_CM) {
        setSample(s, T_MYO, -min(dPv, PV_WALL_CM - dPv), vec3(0.0, -1.0, 0.0), p, 0.0, S_PVEIN);
        return true;
      }
    }
    // venae cavae and hepatic vein
    {
      // decision 370: the cava's lumen keeps clear of the right pulmonary veins' by both walls
      float dSvcLumen = max(dSvc, PV_SVC_CLEARANCE_CM - rightVeinDistance(p, la, lr, bo, czL, rzL));
      if (dSvcLumen < 0.0) { setSample(s, T_BLOOD, dSvcLumen, vec3(0.0, 0.0, -1.0), p, 0.0, S_SVC); return true; }
      if (dSvcLumen < 0.12) { setSample(s, T_VESSEL, -min(dSvcLumen, 0.12 - dSvcLumen), vec3(0.0, 0.0, -1.0), p, 0.0, S_SVC); return true; }
      float dHv = sdCapsule(p, vec3(HV_AX, HV_AY, HV_AZ), vec3(HV_BX, HV_BY, HV_BZ), 0.4);
      if (dIvc < 0.0) { setSample(s, T_BLOOD, dIvc, vec3(0.0, 0.0, 1.0), p, 0.0, S_IVC); return true; }
      if (dIvc < 0.12 && dHv >= 0.0) { setSample(s, T_VESSEL, -min(dIvc, 0.12 - dIvc), vec3(0.0, 0.0, 1.0), p, 0.0, S_IVC); return true; }
      if (dHv < 0.0) { setSample(s, T_BLOOD, dHv, vec3(0.0, 0.0, 1.0), p, 0.0, S_HV); return true; }
      if (dHv < 0.08) { setSample(s, T_VESSEL, -min(dHv, 0.08 - dHv), vec3(0.0, 0.0, 1.0), p, 0.0, S_HV); return true; }
    }
    // coronary sinus (decision 321)
    if (dCs < 0.0) {
      setSample(s, T_BLOOD, dCs, vec3(0.0, -1.0, 0.0), p, 0.0, S_CS);
      return true;
    }
    if (dCs < CS_WALL_CM) {
      setSample(s, T_VESSEL, -min(dCs, CS_WALL_CM - dCs), vec3(0.0, -1.0, 0.0), p, 0.0, S_CS);
      return true;
    }
  }
  // end of classifyAtria

  // ---------- RV ----------
  // src/simulator/anatomy/classify/rightVentricle.ts: classifyRightVentricle
  vec3 rvc = rvCrescent(p, az);
  float dRv = rvc.x;
  // decision 243: the inflow column inside the free wall below the hinges (rv.ts rvInflowSdf)
  float dIn = max(tvInflowSdf(p), length(p.xy) - rvc.z - rvInflowSlack(p.z - (TVS_CZ + tvOffsetAt(p.xy))));
  float dRvU = smin(dRv, dIn, 0.3);
  // the cavity the RV wall is drawn around, with the outflow tract: read again by the pericardium (rightVentricle.ts)
  float dCavRv = dRvU;
  {
    float sc = CONTRACTION;
    float fw = rvFreeWallNow(RV_FW, sc);
    float k = rvOutflowScale(sc);
    // the outflow tract and the pulmonary root move with the base; the bifurcation stays (decision 111)
    vec3 rvotA = vec3(RVOT_AX, RVOT_AY, RVOT_AZ + PV_Z);
    vec3 rvotM = vec3(RVOT_MX, RVOT_MY, RVOT_MZ + PV_Z);
    vec3 rvotB = vec3(RVOT_BX, RVOT_BY, RVOT_BZ + PV_Z);
    float dRvot = min(sdRoundCone(p, rvotA, rvotM, RVOT_RA * k, RVOT_RM * k), sdRoundCone(p, rvotM, rvotB, RVOT_RM * k, RVOT_R * k));
    float dTrunk = pulmonaryTrunkDist(p);
    vec3 v = p - rvotB;
    if (dTrunk < 0.0) {
      setSample(s, T_BLOOD, dTrunk, v, p, 0.0, S_PA);
      return true;
    }
    if (dTrunk < PA_WALL_CM && dRvot > 0.0 && dRvU > 0.0) {
      setSample(s, T_VESSEL, -min(dTrunk, PA_WALL_CM - dTrunk), v, p, 0.0, S_PA);
      return true;
    }
    dCavRv = smin(dRvU, dRvot, RV_OUTFLOW_BLEND_CM);
    float sc3 = 1.0 - 0.3 * sc;
    if (dCavRv < 0.0) {
      if (dRv < 0.0) {
        float L = LV_LEN;
        float rIn = rvc.y, rOut = rvc.z;
        vec3 b0 = vec3(-(rIn + 0.12), -0.2, L * 0.6);
        float rB = rOut - fw * 1.2;
        vec3 b1 = vec3(rB * cos(RV_PAP_AZ), rB * sin(RV_PAP_AZ), L * 0.68);
        float dBand = sdCapsule(p, b0, b1, 0.28);
        if (dBand < 0.0) {
          setSample(s, T_MYO, dBand, vec3(0.0, 0.0, 1.0), p, 0.0, S_MOD_BAND);
          return true;
        }
        float dRp = sdRoundCone(p, vec3(P(RVPAP_BASE), P(RVPAP_BASE + 1), P(RVPAP_BASE + 2)), vec3(P(RVPAP_BASE + 3), P(RVPAP_BASE + 4), P(RVPAP_BASE + 5)), P(RVPAP_BASE + 6), P(RVPAP_BASE + 7));
        if (dRp < 0.0) {
          setSample(s, T_MYO, dRp, vec3(x, y, 0.0), p, 0.0, S_RV_PAP);
          return true;
        }
      }
      float rr = length(p.xy); if (rr == 0.0) rr = 1.0;
      // the supraventricular crest between the atrial inflow column and the outflow tract (decision 334)
      float zAnnulus = TV_CZ + TVZ + tvOffsetAt(p.xy);
      if (dRvot >= dRvU && dRv >= 0.0 && z <= zAnnulus && dRvot < fw) {
        setSample(s, T_MYO, -min(fw - dRvot, dRvot - dRvU), vec3(x / rr, y / rr, 0.0), p, 0.0, S_RV_WALL);
        return true;
      }
      setSample(s, T_BLOOD, dCavRv, vec3(x / rr, y / rr, 0.0), vec3(x / sc3, y / sc3, z), 0.0, dRvot < dRvU ? S_RVOT : ((dRv >= 0.0 && z <= zAnnulus) ? S_RA_CAV : S_RV_CAV));
      return true;
    }
    // decision 367: the free wall ends at its tricuspid hinge
    float tvRH = TV_R * (1.0 - tvShortening(CONTRACTION, RV_RADIAL_SCALE));
    float ringH = length(vec2(x - (TV_CX + TV_R - tvRH), y - TV_CY)) - tvRH;
    bool aboveHinge = z < TV_CZ + TVZ + tvOffsetAt(p.xy) - LV_WALL_BASAL_OVERHANG_CM && dRvot >= fw && ringH > -RV_HINGE_INNER_CM && ringH < RV_HINGE_REACH_CM;
    if (dCavRv < fw && !aboveHinge) {
      float rr = length(p.xy); if (rr == 0.0) rr = 1.0;
      setSample(s, T_MYO, -min(dCavRv, fw - dCavRv), vec3(x / rr, y / rr, 0.0), vec3(x / sc3, y / sc3, z), 0.0, S_RV_WALL);
      return true;
    }
  }
  // end of classifyRightVentricle

  // ---------- pericardium & effusion ----------
  // the block classify/pericardium.ts mirrors, given what the earlier blocks leave (cpuGpuParity.test.ts runs both)
  // src/simulator/anatomy/classify/pericardium.ts: classifyPericardium
  {
    float dLvEpi = dEllR - wallT - crestLoss;
    // decision 380: the ventricle's epicardium ends at its base, but around the root
    float rootGap = rootT > -90.0 ? rootRr - rootR : 1e3;
    float dLvSac = smax(dLvEpi, min(zHinge - LV_SAC_ABOVE_HINGE_CM - z, rootGap - LV_SAC_ROOT_KEEP_CM), 0.2);
    float fw = RV_FW;
    // the outer face of the free wall as the RV block draws it now (pericardium.ts)
    float dRvEpi = dCavRv - rvFreeWallNow(fw, CONTRACTION);
    float dLaEpi = laEpi;
    float dRaEpi = min(min(min(raSleeve + 0.22, raBody - 0.22), raaEpi), raColumn - 0.22); // the atrium's own wall (decision 378), and the vestibule's column (decision 379)
    // the sac around the outflow tract and the trunk stays where the pericardium is anchored (decision 111)
    vec3 rvotA = vec3(RVOT_AX, RVOT_AY, RVOT_AZ);
    vec3 rvotB = vec3(RVOT_BX, RVOT_BY, RVOT_BZ);
    float dRvotEpi = sdCapsule(p, rvotA, rvotB, RVOT_RA + fw);
    float dPaEpi = pulmonaryTrunkEpiDist(p);
    // decision 387: the atrioventricular groove's blend narrows above the groove
    float dEpi = smax(smin(smin(smin(dLvSac, dRvEpi, 0.8), smin(dLaEpi, dRaEpi, 0.8), avGrooveBlend(zHinge - z)), smin(dRvotEpi, dPaEpi, 0.8), 0.8), PERICARDIUM_CM - colDist, 0.2);
    // over an effusion the parietal layer is the fixed sac (pericardium.ts ParietalSac), read before the swing, thin
    // behind the left atrium (decision 255), where it rests on the heart of the moment (parietalSdf)
    float par = EFFUSION > 0.0 ? parietalSdf(EFFUSION, parietalSacDistance(p0.x, p0.y, p0.z), dEpi, (la.y - y) / lr.y, ((p0.x - la.x) / lr.x) * ((p0.x - la.x) / lr.x) + ((z - la.z) / lr.z) * ((z - la.z) / lr.z), colDist) : 1e3;
    vec3 nEpi = n0;
    if (dEpi < 0.0) {
      setSample(s, T_FAT, dEpi, nEpi, p, 0.0, S_EPI_FAT);
      return true;
    }
    if (dEpi < PERICARDIUM_CM) {
      float de = max(dEpi, 0.0);
      setSample(s, T_PERI, -min(de, 0.12 - de), nEpi, p, 0.0, S_PERI);
      return true;
    }
    if (par < 0.0) {
      setSample(s, T_FLUID, max(par, PERICARDIUM_CM - dEpi), nEpi, p, 0.0, S_EFFUSION);
      return true;
    }
    if (par < PERICARDIUM_CM) {
      setSample(s, T_PERI, 0.0, nEpi, p, 0.0, S_PERI);
      return true;
    }
    // outside the sac: distance beyond the parietal pericardium or the wall of the ascending aorta (decision 144)
    float dSac = min(dEpi, par) - PERICARDIUM_CM;
    float dRoot = rootT > -90.0 ? rootRr - rootR - 0.2 : dSac;
    // decision 336: in front of the pulmonary trunk the mediastinum's fat stands between the trunk and the lung
    float dTrunkFront = trunkFrontDist(p, dPaEpi);
    s.sdf = min(min(min(dSac, dRoot), vascularDistance), dTrunkFront);
  }
#endif
  return false;
}
`;
