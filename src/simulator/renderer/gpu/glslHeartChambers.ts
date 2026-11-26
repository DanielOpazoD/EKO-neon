import {
  AORTIC_EXTENSION_WALL_CM,
  AORTIC_ASCENDING_SEGMENTS,
  AORTIC_EXTENSION_SEGMENTS,
} from '@/simulator/anatomy/aorticExtension';
import { VASCULAR_JOIN_EPS_CM } from '@/simulator/anatomy/vascularTube';
import { ROOT_TUBE_END_T } from '@/simulator/anatomy/classify/root';
import { PA_WALL_CM } from '@/simulator/anatomy/greatVessels';
import { WALL_MODULATION_GAIN } from '@/simulator/anatomy/lvWall';

/**
 * GLSL chamber & great-vessel geometry helpers (LV bullet profile, wall thickness, RV crescent
 * & inflow, aortic extension, pulmonary trunk, pulmonary veins, and coronary sinus).
 * Extracted verbatim from `glslHeart.ts`.
 */
export const GLSL_HEART_CHAMBERS = /* glsl */ `
// ---- LV bullet profile (lvShape.ts) ----
float lvProfileG(float zeta) {
  if (zeta < 0.0) {
    float v = zeta / LV_ZETATOP;
    return v >= 1.0 ? 0.0 : LV_G0 * sqrt(1.0 - v * v);
  }
  if (zeta <= LV_ZETAMAX) {
    float u = 1.0 - zeta / LV_ZETAMAX;
    return max(0.0, 1.0 - (1.0 - LV_G0) * u * u);
  }
  if (zeta >= 1.0) return 0.0;
  float s = (zeta - LV_ZETAMAX) / (1.0 - LV_ZETAMAX);
  return sqrt(max(0.0, 1.0 - pow(s, LV_N)));
}
float lvProfileDG(float zeta) {
  if (zeta < 0.0) {
    float v = zeta / LV_ZETATOP;
    if (v >= 0.999) return 6.0;
    return -LV_G0 * v / LV_ZETATOP / sqrt(1.0 - v * v);
  }
  if (zeta <= LV_ZETAMAX) {
    float u = 1.0 - zeta / LV_ZETAMAX;
    return 2.0 * (1.0 - LV_G0) * u / LV_ZETAMAX;
  }
  if (zeta >= 1.0) return -6.0;
  float s = (zeta - LV_ZETAMAX) / (1.0 - LV_ZETAMAX);
  float sn = pow(s, LV_N);
  float d = -((LV_N / 2.0) * pow(s, LV_N - 1.0)) / sqrt(max(1e-9, 1.0 - sn)) / (1.0 - LV_ZETAMAX);
  return max(-6.0, d);
}
float lvCavityRadius(float az, float z) {
  float zeta = (z - ZANN) / max(LENGTH_NOW, 1e-3);
  // decision 226
  float w = lvNeckWeight(zeta);
  float S = 1.0 + LV_NECK_K * w, ox = LV_NECK_X * w;
  float rho = LV_RMAX * lvProfileG(zeta);
  float c = cos(az), sn = sin(az) / LV_RATIO;
  float a = c * c * S * S + sn * sn, b = c * ox * S * S;
  return (b + sqrt(max(0.0, b * b - a * (ox * ox * S * S - rho * rho)))) / a;
}
float lvRadialOffsetFactor(float az, float z) {
  float zeta = (z - ZANN) / max(LENGTH_NOW, 1e-3);
  float drdz = LV_RMAX * lvProfileDG(zeta) * ellipseFactor(LV_RATIO, az) / max(LENGTH_NOW, 1e-3);
  return sqrt(1.0 + min(9.0, drdz * drdz));
}
// signed distance to the tabulated cavity surface (polar table from the centre LV_PZC); writes the normal
float lvCavitySdf(float xs, float y, float z, out vec3 n) {
  // decision 226
  float w = lvNeckWeight((z - ZANN) / max(LENGTH_NOW, 1e-3));
  float sx = 1.0 + LV_NECK_K * w;
  float xr = xs - LV_NECK_X * w;
  float xn = xr * sx;
  float ys = y / LV_RATIO;
  float rho2 = xn * xn + ys * ys;
  float rho = sqrt(rho2);
  float dz = z - LV_PZC;
  float phi = rho > 1e-9 ? atan(rho, dz) : (dz >= 0.0 ? 0.0 : PI);
  float rad = sqrt(rho2 + dz * dz);
  float nb = float(LV_PROF_BINS - 1);
  float fk = clamp(phi / PI * nb, 0.0, nb - 1.0001);
  int k = int(floor(fk));
  float t = fk - float(k);
  float R = P(LV_PROF_R_BASE + k) + (P(LV_PROF_R_BASE + k + 1) - P(LV_PROF_R_BASE + k)) * t;
  float S = P(LV_PROF_S_BASE + k) + (P(LV_PROF_S_BASE + k + 1) - P(LV_PROF_S_BASE + k)) * t;
  float f = 1.0 / sqrt(1.0 + S * S);
  float sinP = rad > 1e-9 ? rho / rad : 0.0;
  float cosP = rad > 1e-9 ? dz / rad : 1.0;
  float nr = sinP - S * cosP, nz = cosP + S * sinP;
  float ir = rho > 1e-9 ? 1.0 / rho : 0.0;
  n = vec3(nr * xn * ir * sx, nr * ys * ir / LV_RATIO, nz);
  float q = rho2 > 1e-12 ? sqrt((xr * xr + LV_RATIO * LV_RATIO * ys * ys) / rho2) : 1.0;
  float corr = 1.0 - (1.0 - q) * sinP * sinP;
  return (rad - R) * f * corr;
}

// lvWall.ts wallThicknessAt: ivsd between the RV insertions, lvpwd on the free wall (septalShare)
float wallThicknessAt(float az, float levelFrac, float amp) {
  float tBase = LV_LVPWD + (LV_IVSD - LV_LVPWD) * septalShare(az, RV_AZA, RV_AZP);
  float tED = tBase * axialWallFactor(levelFrac, APEX_T / tBase);
  float wallMod = 1.0 + ${WALL_MODULATION_GAIN.toFixed(8)} * (lat(vec3(cos(az) * 1.6 + 7.3, sin(az) * 1.6 + 2.1, levelFrac * 2.4), 3) - 0.5) * (1.0 - levelFrac * levelFrac);
  return tED * max(0.6, 1.0 + (LV_THICK_K - 1.0) * (0.35 + 0.65 * amp)) * wallMod;
}


// [rIn, u, rOut, t] without trabecular noise
vec4 rvRadii(float az, float z, float contraction, float tvZ, float rvCollapse) {
  float L = LV_LEN;
  float azN = az < 0.0 ? az + TWO_PI : az;
  float u = (azN - RV_AZA) / (RV_AZP - RV_AZA);
  float rCav = lvCavityRadius(az, z);
  float levelFrac = clamp((z - ZANN) / max(LENGTH_NOW, 1.0), 0.0, 1.0);
  float amp = P(SEG_AMP_BASE + ahaSegment(az, levelFrac));
  // decision 223: the crest's lost thickness goes to the cavity below the tricuspid plane, the free wall stays
  float wFull = wallThicknessAt(az, levelFrac, amp) * lvRadialOffsetFactor(az, z);
  float below = clamp((z - (TV_CZ + tvZ - 0.6)) / 0.3, 0.0, 1.0);
  float crestLossR = wFull * (1.0 - septalCrestFactor(az, z - ZANN)) * below * below * (3.0 - 2.0 * below);
  float rEpi = rCav + wFull - crestLossR;
  float rIn = rEpi - septalShiftAt(SEPTAL_SHIFT, az, levelFrac) + 0.05;
  if (u <= 0.0 || u >= 1.0) return vec4(rIn, u, rIn, 0.0);
  float tvPlane = TV_CZ + tvZ;
  float t = RV_T * rvAzProfile(RV_AZA, RV_AZP, u) * rvAxialTaper(tvPlane, RV_APEX_FRAC * L, z) * (1.0 - rvRadialContraction(u) * rvRadialState(tvZ, RV_TAPSE, contraction) * RV_RADIAL_SCALE);
  // decision 243: the free wall hangs from the annulus, which shortens about its septal edge
  float tvR = TV_R * (1.0 - tvShortening(contraction, RV_RADIAL_SCALE));
  float tvCx = TV_CX + (TV_R - tvR);
  float rc = length(vec2(tvCx, TV_CY));
  float dAz = az - atan(TV_CY, tvCx);
  dAz = abs(dAz - TWO_PI * floor(dAz / TWO_PI + 0.5));
  float wH = rvHingeWeight(dAz, asin(min(1.0, tvR / rc)), z - tvPlane);
  if (wH > 0.0) t += wH * max(0.0, rvHingeRadius(rc, tvR, dAz) - rIn - crestLossR - t);
  if (rvCollapse > 0.0 && u < 0.55) t *= 1.0 - 0.65 * rvCollapse * (1.0 - u / 0.55);
  return vec4(rIn, u, rIn + t + crestLossR, t + crestLossR);
}
// tricuspid inflow column (heartModel.ts tvInflowSdf): annular circle narrowing below the hinges, closed on the atrial side
// annulus offset above a point, by its azimuth around the tricuspid centre (valveSkirt.ts skirtOffsetAt)
float tvOffsetAt(vec2 q) {
  vec2 d = q - vec2(TVS_CX, TVS_CY);
  float phi = dot(d, d) > 1e-12 ? atan(d.y, d.x) : 0.0;
  return annulusOffset(phi, TVS_SADDLE_PHI, TVS_SADDLE, TVS_TILTC, TVS_TILTS, TVS_LIFT);
}
float tvInflowSdf(vec3 p) {
  vec2 d = p.xy - vec2(TVS_CX, TVS_CY);
  d *= tvWarpScale(d);
  float r2 = dot(d, d);
  float rho = sqrt(r2);
  float phi = r2 > 1e-12 ? atan(d.y, d.x) : 0.0;
  float h = p.z - (TVS_CZ + annulusOffset(phi, TVS_SADDLE_PHI, TVS_SADDLE, TVS_TILTC, TVS_TILTS, TVS_LIFT));
  float bulge = TV_INFLOW_BULGE_CM * 0.5 * (1.0 - cos(phi - P(TVS_ZONES_BASE + 6)));
  return rho - TVS_R + 0.04 + tvInflowTaper(h, 0.25 + 0.3 * TVZ, bulge);
}
// RV crescent: returns [signed distance, rIn, rOut]
vec3 rvCrescent(vec3 p, float az) {
  vec4 rr = rvRadii(az, p.z, CONTRACTION, TVZ, RV_COLLAPSE);
  float rIn = rr.x, u = rr.y;
  if (u <= 0.0 || u >= 1.0) return vec3(1e3, rIn, rIn);
  float L = LV_LEN;
  float zApex = RV_APEX_FRAC * L;
  float zBase = rvFloorZ(TV_CZ, TVZ, PV_Z, u, tvOffsetAt(p.xy));
  float t = rr.w;
  if (p.z > 0.25 * L) {
    float w = min(1.0, (p.z - 0.25 * L) / (0.35 * L));
    float rs = 1.0 - 0.3 * CONTRACTION;
    float n = lat(vec3((p.x / rs) * 1.4 + 3.1, (p.y / rs) * 1.4 + 9.7, p.z * 0.9 + 5.3), 3) - 0.5;
    t += (0.25 + 0.25 * w) * n - 0.12 * w * w;
  }
  float rOut = rIn + max(0.0, t);
  float r = length(p.xy);
  float d = max(max(rIn - r, r - rOut), max(zBase - p.z, p.z - zApex));
  return vec3(d, rIn, rOut);
}

// Same packed tube geometry as the CPU and navigator.
const float AO_JOIN_EPS = ${VASCULAR_JOIN_EPS_CM.toFixed(8)};
const int AO_SEGMENTS = ${AORTIC_EXTENSION_SEGMENTS};
const int AO_ASCENDING = ${AORTIC_ASCENDING_SEGMENTS};
const float AO_WALL = ${AORTIC_EXTENSION_WALL_CM.toFixed(8)};
const float AO_ROOT_END = ${ROOT_TUBE_END_T.toFixed(8)};
bool classifyAorticExtension(vec3 p, inout Sample s) {
  vec3 start = vec3(P(AO_JOIN_BASE), P(AO_JOIN_BASE+1), P(AO_JOIN_BASE+2));
  vec3 axis = vec3(P(AO_JOIN_BASE+3), P(AO_JOIN_BASE+4), P(AO_JOIN_BASE+5));
  vec3 mn = vec3(P(AO_BOUNDS_BASE), P(AO_BOUNDS_BASE+1), P(AO_BOUNDS_BASE+2));
  vec3 mx = vec3(P(AO_BOUNDS_BASE+3), P(AO_BOUNDS_BASE+4), P(AO_BOUNDS_BASE+5));
  if (any(lessThan(p,mn)) || any(greaterThan(p,mx))) return false;
  float best = FAR_FROM_HEART_CM;
  vec3 normal = vec3(0.0, 0.0, 1.0);
  int segment = 0;
  for (int i=0; i<AO_SEGMENTS; i++) {
    int j=AO_TUBES_BASE+i*8;
    vec3 a=vec3(P(j),P(j+1),P(j+2)), b=vec3(P(j+4),P(j+5),P(j+6));
    float ra=P(j+3), rb=P(j+7);
    vec3 delta=b-a;
    float len2=dot(delta,delta);
    float raw=len2>0.0 ? dot(p-a,delta)/len2 : 0.0;
    float u=clamp(raw,0.0,1.0);
    vec3 r=p-a-u*delta;
    float radial=length(r), dist=radial-mix(ra,rb,u);
    if (dist>=best) continue;
    best=dist; segment=i;
    float taper=raw>0.0 && raw<1.0 ? (rb-ra)/len2 : 0.0;
    normal=(radial>0.0 ? r/radial : vec3(0.0))-taper*delta;
  }
  if (segment < AO_ASCENDING && dot(p-start, axis) < -AO_JOIN_EPS) return false;
  if (best>=AO_WALL) { s.sdf=min(s.sdf,best-AO_WALL); return false; }
  float n=length(normal);
  setSample(s, best<0.0 ? T_BLOOD : T_VESSEL, best<0.0 ? best : -min(best,AO_WALL-best), n>0.0 ? normal/n : vec3(0.0), p, 0.0, int(P(AO_IDS_BASE+segment)));
  return true;
}
const float PA_WALL_CM = ${PA_WALL_CM.toFixed(8)};
/** Signed distance to the lumen of the pulmonary trunk and its branches (greatVessels.ts pulmonaryTrunkDistance). */
float pulmonaryTrunkDist(vec3 p) {
  vec3 rvotB = vec3(RVOT_BX, RVOT_BY, RVOT_BZ + PV_Z);
  vec3 paStj = vec3(PA_SX, PA_SY, PA_SZ + PV_Z);
  vec3 paEnd = vec3(PA_EX, PA_EY, PA_EZ);
  float dPa = min(sdRoundCone(p, rvotB, paStj, PA_ROOT_R, PA_R), sdCapsule(p, paStj, paEnd, PA_R));
  float dRpa = sdCapsule(p, paEnd, vec3(RPA_EX, RPA_EY, RPA_EZ), RPA_R);
  float dLpa = sdCapsule(p, paEnd, vec3(LPA_EX, LPA_EY, LPA_EZ), LPA_R);
  return min(dPa, min(dRpa, dLpa));
}
/** Distance beyond the epicardium of the pulmonary trunk (greatVessels.ts pulmonaryTrunkEpiDistance). */
float pulmonaryTrunkEpiDist(vec3 p) {
  vec3 rvotB = vec3(RVOT_BX, RVOT_BY, RVOT_BZ);
  vec3 paStj = vec3(PA_SX, PA_SY, PA_SZ);
  vec3 paEnd = vec3(PA_EX, PA_EY, PA_EZ);
  return min(sdRoundCone(p, rvotB, paStj, PA_ROOT_R + 0.2, PA_R + 0.2), sdCapsule(p, paStj, paEnd, PA_R + 0.2));
}
/** Beyond the mediastinal fat in front of the pulmonary trunk (greatVessels.ts trunkFrontDistance, decision 336). */
float trunkFrontDist(vec3 p, float dPaEpi) {
  vec3 tm = p - 0.5 * (vec3(RVOT_BX, RVOT_BY, RVOT_BZ) + vec3(PA_SX, PA_SY, PA_SZ));
  float tl = length(tm);
  float front = tl > 1e-6 ? dot(tm, vec3(HF_EXZ, HF_EYZ, HF_EZZ)) / tl : 0.0;
  return dPaEpi - PERICARDIUM_CM - PA_ANTERIOR_FAT_CM * max(0.0, front);
}

// Shared RV distance for the pericardium block (recomputed; cheap)
// decision 320: distance to the lumen of the pulmonary veins and the appendage, which the atrial wall and septum leave open
// decision 330: a left vein's ostium the posterior wall plane would leave behind it moves in front of the plane;
// decision 337: it turns forward along the atrial section (mirror of pulmonaryVeinSegment), staying on the atrium
float veinOstiumAngle(float t, vec3 la, float a, float b, float oz) {
  float lim = LA_PD - PV_OSTIUM_AHEAD_CM - LA_PNX * la.x - LA_PNY * la.y - LA_PNZ * oz;
  float ca = LA_PNX * a, cb = -LA_PNY * b;
  if (ca * cos(t) + cb * sin(t) <= lim) return t;
  float rr = length(vec2(ca, cb));
  return rr > abs(lim) ? atan(cb, ca) - acos(lim / rr) : t;
}
float leftAtrialMouthLumen(vec3 p, vec3 la, vec3 lr, float bo, float czL, float rzL, float colDist) {
  float x = p.x, y = p.y, z = p.z;
  vec3 a0 = vec3(la.x + lr.x * 0.55, la.y + lr.y * 0.55, czL + 0.4);
  vec3 a1 = vec3(la.x + lr.x * 0.95, a0.y + 2.0, czL + 0.9);
  float d = sdCapsule(p, a0, a1, 0.55 * bo + 0.12 * (lat(vec3(x * 2.3 + 1.7, y * 2.3 + 4.2, z * 2.3 + 8.8), 3) - 0.5));
  for (int i = 0; i < 4; i++) {
    float sx = (i == 0 || i == 2) ? -1.0 : 1.0;
    bool sup = i < 2;
    float dzN = sup ? PV_SUP_Z : PV_INF_Z;
    float kz = sqrt(1.0 - dzN * dzN);
    float t = sx > 0.0 ? (sup ? PV_LEFT_SUP_T : PV_LEFT_INF_T) : PV_RIGHT_T;
    float ea = lr.x * bo * kz, eb = lr.y * bo * kz;
    float pz = czL + dzN * rzL;
    if (sx > 0.0) t = veinOstiumAngle(t, la, ea, eb, pz);
    // decisions 330 and 337: a left vein's ostium stays on the atrium in front of the posterior wall plane
    vec3 o = vec3(la.x + sx * ea * cos(t), la.y - eb * sin(t), pz);
    d = min(d, smax(sdCapsule(p, o, o + vec3(PV_COURSE[3 * i], PV_COURSE[3 * i + 1], PV_COURSE[3 * i + 2]), PV_RADIUS), 0.12 - colDist, 0.2));
  }
  return d;
}

// decision 320: distance to the lumen of the right pulmonary veins, which the right atrium leaves room for
float rightVeinCapsule(int i, vec3 p, vec3 la, vec3 lr, float bo, float czL, float rzL) {
  float dzN = i < 2 ? PV_SUP_Z : PV_INF_Z;
  float kz = sqrt(1.0 - dzN * dzN);
  float px = la.x - lr.x * bo * kz * cos(PV_RIGHT_T);
  float py0 = la.y - lr.y * bo * kz * sin(PV_RIGHT_T);
  float pz = czL + dzN * rzL;
  vec3 o = vec3(px, py0, pz);
  return sdCapsule(p, o, o + vec3(PV_COURSE[3 * i], PV_COURSE[3 * i + 1], PV_COURSE[3 * i + 2]), PV_RADIUS);
}
float rightVeinDistance(vec3 p, vec3 la, vec3 lr, float bo, float czL, float rzL) {
  return min(rightVeinCapsule(0, p, la, lr, bo, czL, rzL), rightVeinCapsule(2, p, la, lr, bo, czL, rzL));
}

// decision 321: the coronary sinus along the posterior groove above the mitral annulus, opening into the right atrium
float laPosteriorY(float x, float z, vec3 la, float rx, float ry, float cz, float rz) {
  float u = (x - la.x) / rx, w = (z - cz) / rz;
  return max(la.y - ry * sqrt(max(0.0, 1.0 - u * u - w * w)), la.y - LA_POSTERIOR_RATIO * ry);
}
// decision 330: lowest y of the sinus's axis at (x, z) that keeps its wall in front of the posterior wall plane
float csWallFloor(float x, float z) {
  return (LA_PD - CS_RADIUS_CM - CS_WALL_CM - x * LA_PNX - z * LA_PNZ) / LA_PNY;
}
float coronarySinusDistance(vec3 p, vec3 la, vec3 lr, float bo, float czL, float rzL, float zAnn, float laCav) {
  // in the groove, under the body's full posterior surface (decision 330)
  float rx = lr.x, ry = lr.y;
  float zs = zAnn - CS_ABOVE_ANNULUS_CM;
  float x0 = IAS_X + CS_CRUX_FROM_IAS_CM;
  float x1 = min(CS_LATERAL_X_CM, la.x + 0.85 * rx);
  float xc = min(x1, max(x0, p.x));
  // under the atrium's floor at the back and in front of the posterior wall plane (decision 330)
  float yc = max(min(laPosteriorY(max(xc, la.x), zs, la, rx, ry, czL, rzL) - LA_POSTERIOR_FLOOR_DROP_CM - CS_GROOVE_OFFSET_CM, MVL_CY - sqrt(max(0.0, MVL_R * MVL_R - (xc - MVL_CX) * (xc - MVL_CX))) - CS_ANNULUS_OFFSET_CM), csWallFloor(xc, zs));
  float body = length(vec3(p.x - xc, p.y - yc, p.z - zs)) - CS_RADIUS_CM;
  vec3 a = vec3(x0, max(laPosteriorY(max(x0, la.x), zs, la, rx, ry, czL, rzL) - LA_POSTERIOR_FLOOR_DROP_CM - CS_GROOVE_OFFSET_CM, csWallFloor(x0, zs)), zs);
  vec3 b = vec3(IAS_X - CS_OSTIUM_FROM_IAS_CM, CS_OSTIUM_Y_CM, zAnn - CS_OSTIUM_ABOVE_ANNULUS_CM);
  float ostium = sdRoundCone(p, a, b, CS_RADIUS_CM, CS_OSTIUM_RADIUS_CM);
  return max(min(body, ostium), LA_WALL_CM + CS_WALL_CM - laCav);
}
`;
