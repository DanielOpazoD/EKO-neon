/**
 * GLSL valve apparatus & AHA segment helpers (skirts, mitral apparatus, aortic root & cusps,
 * pulmonary cusp chain, and AHA 17-segment mapping). Extracted verbatim from `glslHeart.ts`.
 */
export const GLSL_HEART_VALVES = /* glsl */ `
// ---- profile helpers (skirts) ----
float profAt(int base, int i) { return P(base + i); }
// tricuspid annulus extension (valveSkirt.ts skirtBumpAt / skirtWarpScale, decision 224)
float tvBumpAt(float phi) {
  float f = (phi - TV_BUMP_PHI0) / TV_BUMP_STEP_RAD;
  if (f <= 0.0 || f >= float(TV_BUMP_N - 1)) return 0.0;
  int i = int(floor(f));
  return mix(P(TVS_BUMP_BASE + i), P(TVS_BUMP_BASE + i + 1), f - float(i));
}
float tvWarpScale(vec2 d) {
  float b = dot(d, d) > 1e-12 ? tvBumpAt(atan(d.y, d.x)) : 0.0;
  return b > 0.0 ? TVS_R / (TVS_R + b) : 1.0;
}

// AV-valve skirt: minimum over leaflet zones (radial revolution or parallel-fibre sheets); writes distance, frac, zone, normal
float skirtDistance(vec3 p, vec3 c, float R, int zonesBase, int profBase, int nz, float closed, float blend, float thickness, float saddle, out float dOut, out float fracOut, out int zoneOut, out vec3 nOut) {
  vec2 d = p.xy - c.xy;
  d *= tvWarpScale(d);
  float zr0 = p.z - c.z;
  float rho = length(d);
  zoneOut = 0;
  nOut = vec3(0.0, 0.0, 1.0);
  if (zr0 > SKIRT_ABOVE_CM || zr0 < -SKIRT_BELOW_CM || rho > R + SKIRT_RADIAL_MARGIN_CM) { dOut = 1e3; fracOut = 0.0; return 0.0; }
  float phi = atan(d.y, d.x);
  // the annulus offset weighs by the distance from the centre, where the closed leaflets meet (valveSkirt.ts, decision 148)
  float zr = zr0 - annulusOffset(phi, TVS_SADDLE_PHI, saddle, TVS_TILTC, TVS_TILTS, TVS_LIFT) * min(1.0, rho / R);
  float best = 1e9, bestFrac = 0.0, bestW = 0.0;
  int bestZone = 0;
  float bestEx = 0.0, bestEz = 1.0, bestCa = 1.0, bestSa = 0.0, bestKind = 0.0;
  for (int zi = 0; zi < 3; zi++) {
    if (zi >= nz) break;
    int zb = zonesBase + zi * 6;
    float zphi = P(zb), zhalf = P(zb + 1), zkind = P(zb + 2), zlobes = P(zb + 3), zc = P(zb + 4);
    float w, rhoS, s;
    float ca = cos(zphi), sa = sin(zphi);
    if (zkind > 0.5) {
      float v = d.x * ca + d.y * sa;
      float u = -d.x * sa + d.y * ca;
      float t = abs(u) / R;
      if (t >= SKIRT_FIBRE_CLIP) continue;
      float vAtt = sqrt(R * R - u * u);
      rhoS = R - (vAtt - v);
      float tw = (t - SKIRT_TAPER_START) / SKIRT_TAPER_WIDTH;
      w = tw <= 0.0 ? 1.0 : 1.0 - tw * tw * (3.0 - 2.0 * tw);
      float sc = sqrt(1.0 - t * t) * (1.0 + zc * t * t);
      if (zlobes > 0.0) sc *= 1.0 + zlobes * cos(TWO_PI * t / SKIRT_LOBE_PERIOD);
      s = 1.0 + (sc - 1.0) * closed;
    } else {
      float dphi = abs(phi - zphi);
      if (dphi > PI) dphi = TWO_PI - dphi;
      float tw = (dphi - (zhalf - blend)) / (2.0 * blend);
      w = tw <= 0.0 ? 1.0 : (tw >= 1.0 ? 0.0 : 1.0 - tw * tw * (3.0 - 2.0 * tw));
      if (w <= 0.0) continue;
      rhoS = rho;
      float q = dphi / zhalf;
      s = 1.0 - zc * q * q * closed;
    }
    int pb = profBase + zi * 8;
    for (int i = 0; i < 3; i++) {
      float ax = R + (P(pb + i * 2) - R) * s, az = P(pb + i * 2 + 1) * s;
      float bx = R + (P(pb + i * 2 + 2) - R) * s, bz = P(pb + i * 2 + 3) * s;
      float ex = bx - ax, ez = bz - az;
      float l2 = ex * ex + ez * ez;
      float uu = l2 > 0.0 ? ((rhoS - ax) * ex + (zr - az) * ez) / l2 : 0.0;
      uu = clamp(uu, 0.0, 1.0);
      float qx = ax + ex * uu - rhoS;
      float qz = az + ez * uu - zr;
      float dd = sqrt(qx * qx + qz * qz);
      if (dd < best) { best = dd; bestFrac = (float(i) + uu) / 3.0; bestW = w; bestZone = zi; bestEx = ex; bestEz = ez; bestCa = ca; bestSa = sa; bestKind = zkind; }
    }
  }
  dOut = best;
  fracOut = bestFrac;
  zoneOut = bestZone;
  // surface normal from the profile edge (like the mitral valve and the CPU skirt): the edge
  // direction in the (rho, z) plane is (ex, ez), so the outward normal is (-ez, ex) rotated into
  // 3D by the radial direction at this point. For radial zones the radial direction is (dx, dy)/rho;
  // for parallel zones it is the zone's perpendicular (-sa, ca).
  float rx, ry;
  if (bestKind > 0.5) {
    rx = -bestSa;
    ry = bestCa;
  } else {
    rx = rho > 1e-6 ? d.x / rho : bestCa;
    ry = rho > 1e-6 ? d.y / rho : bestSa;
  }
  nOut = vec3(-bestEz * rx, -bestEz * ry, bestEx);
  return (thickness * (SKIRT_THICK_BASE + SKIRT_THICK_EDGE * bestFrac) * 0.5 + SKIRT_THICK_FLOOR_CM) * (SKIRT_THICK_COMMISSURE + SKIRT_THICK_BODY * bestW);
}

// ---- mitral apparatus (mitralValve.ts) ----
float mvBin(int tab, int col, float f) {
  float xx = clamp(f - 0.5, 0.0, float(MV_BINS - 1));
  int i = min(MV_BINS - 2, int(floor(xx)));
  float w = xx - float(i);
  int o = tab + col * MV_BINS + i;
  return P(o) * (1.0 - w) + P(o + 1) * w;
}
// annulus height above the hinge plane at (u, v) around the valve centre: saddle and curtain lift
float mvHingeHeight(float u, float v) {
  float theta = atan(abs(u), v);
  float thetaC = atan(sqrt(MVL_R * MVL_R - MVL_D * MVL_D), MVL_D) + AML_ARC_EXTENSION;
  float onCurtain = clamp((thetaC - theta) / AML_ARC_EXTENSION, 0.0, 1.0);
  float r2 = u * u + v * v;
  return MVL_SADDLE * (u * u / (r2 > 0.0 ? r2 : 1.0)) + MVL_LIFT * onCurtain;
}
float mvHingeZ(vec2 p) {
  vec2 d = p - vec2(MVL_CX, MVL_CY);
  return MVL_CZ + mvHingeHeight(-d.x * MVL_UY + d.y * MVL_UX, d.x * MVL_UX + d.y * MVL_UY);
}
float mvOutlineSdf(vec2 p) {
  vec2 d = p - vec2(MVL_CX, MVL_CY);
  return max(length(d) - MVL_R, dot(d, vec2(MVL_UX, MVL_UY)) - MVL_D);
}
bool mvInsideOutline(vec2 p) {
  vec2 d = p - vec2(MVL_CX, MVL_CY);
  return dot(d, d) < MVL_R * 0.98 * (MVL_R * 0.98) && dot(d, vec2(MVL_UX, MVL_UY)) < MVL_D * 0.98;
}
float mvInflowTaper(float h) {
  float beyond = h - MVL_INFLOW_DEPTH;
  return h < 0.0
    ? (h < -0.25 ? 2.0 * (-h - 0.25) : 0.0)
    : MVL_INFLOW_SLOPE * h + (beyond > 0.0 ? 3.0 * beyond * beyond : 0.0);
}
float mvAnnulusDistance(vec3 p, float tube) {
  vec2 d = p.xy - vec2(MVL_CX, MVL_CY);
  float v = dot(d, vec2(MVL_UX, MVL_UY));
  float u = -d.x * MVL_UY + d.y * MVL_UX;
  float uc = sqrt(MVL_R * MVL_R - MVL_D * MVL_D);
  float rho = length(vec2(u, v));
  if (rho == 0.0) rho = 1e-6;
  float qu = u / rho * MVL_R, qv = v / rho * MVL_R;
  float su = clamp(u, -uc, uc);
  if (qv > MVL_D || (u - su) * (u - su) + (v - MVL_D) * (v - MVL_D) < (u - qu) * (u - qu) + (v - qv) * (v - qv)) {
    qu = su;
    qv = MVL_D;
  }
  float dPlane = length(vec2(u - qu, v - qv));
  float dzz = p.z - (MVL_CZ + mvHingeHeight(qu, qv));
  float duT = abs(qu) - uc;
  float dvT = qv - MVL_D;
  float trig = max(0.0, 1.0 - (duT * duT + dvT * dvT) / 0.25);
  return sqrt(dPlane * dPlane + dzz * dzz) - (tube + 0.048 * trig * trig);
}
// share of the SAM a fibre takes at fan fraction q: the chordae hold the commissures (samShare, mitralValve.ts)
// src/simulator/anatomy/mitralValve.ts: samShare
float mvSamShare(float q) { return 1.0 - q * q; }
// one leaflet (fan of fibres from its focus): updates the nearest hit
void mvLeaflet(float v, float u, float zr0, int tab, int prof, float focusV, float axisSign, float halfSpan, int leaflet, float opening, float samBlend,
               inout float best, inout float bestFrac, inout int bestLeaflet, inout float bestW, inout vec3 bestN) {
  float dv = v - focusV;
  float rho = length(vec2(dv, u));
  if (rho < 1e-6) return;
  float q = atan(u, dv * axisSign) / halfSpan;
  if (q <= -1.0 || q >= 1.0) return;
  float openness = max(opening, samBlend * mvSamShare(q));
  float f = (q + 1.0) * 0.5 * float(MV_BINS);
  float hingeS = mvBin(tab, 0, f);
  float reach = mvBin(tab, 1, f);
  float tent = mvBin(tab, 2, f);
  float zr = zr0 - mvBin(tab, 3, f);
  float aq = abs(q);
  float tq = (aq - 0.9) / 0.1;
  float w = aq < 0.9 ? 1.0 : 1.0 - tq * tq * (3.0 - 2.0 * tq);
  float openScale = 0.55 + 0.45 * sqrt(max(0.0, 1.0 - q * q));
  float c = 1.0 - openness;
  float rot = openness > 0.0 ? mvBin(tab, 4, f) : 0.0;
  float cr = cos(rot), sr = sin(rot);
  float inw = hingeS - rho;
  float ix = -(dv * MVL_UX - u * MVL_UY) / rho, iy = -(dv * MVL_UY + u * MVL_UX) / rho;
  float ax = 0.0, az = 0.0;
  for (int i = 0; i < 3; i++) {
    float oa = P(prof + i * 2) * openScale, oz = P(prof + i * 2 + 1) * openScale;
    float bx = reach * MV_CLOSED_REACH[i] * c + (oa * cr + oz * sr) * openness;
    float bz = tent * MV_CLOSED_DEPTH[i] * c + (oz * cr - oa * sr) * openness;
    float ex = bx - ax, ez = bz - az;
    float l2 = ex * ex + ez * ez;
    float sg = l2 > 0.0 ? clamp(((inw - ax) * ex + (zr - az) * ez) / l2, 0.0, 1.0) : 0.0;
    float qx = ax + ex * sg - inw, qz = az + ez * sg - zr;
    float dd = sqrt(qx * qx + qz * qz);
    if (dd < best) {
      best = dd;
      bestFrac = (float(i) + sg) / 3.0;
      bestLeaflet = leaflet;
      bestW = w;
      bestN = vec3(-ez * ix, -ez * iy, ex);
    }
    ax = bx;
    az = bz;
  }
}
// distance to the mitral leaflets; returns the local half thickness
float mitralDistance(vec3 p, out float dOut, out float fracOut, out int leafletOut, out vec3 nOut) {
  dOut = 1e3;
  fracOut = 0.0;
  leafletOut = 0;
  nOut = vec3(0.0, 0.0, 1.0);
  float dx = p.x - MVL_CX, dy = p.y - MVL_CY;
  float zr0 = p.z - MVL_CZ;
  if (zr0 > 3.5 || zr0 < -2.5 || dx * dx + dy * dy > (MVL_R + 2.2) * (MVL_R + 2.2)) return 0.0;
  float v = dx * MVL_UX + dy * MVL_UY;
  float u = -dx * MVL_UY + dy * MVL_UX;
  float best = 1e9, bestFrac = 0.0, bestW = 1.0;
  int bestLeaflet = 0;
  vec3 bestN = vec3(0.0, 0.0, 1.0);
  mvLeaflet(v, u, zr0, MVL_A_TAB_BASE, MVL_A_PROF_BASE, MVL_A_FOCUS, MVL_A_AXIS, MVL_A_HALF, 0, MVL_OPEN, MVL_SAM, best, bestFrac, bestLeaflet, bestW, bestN);
  mvLeaflet(v, u, zr0, MVL_P_TAB_BASE, MVL_P_PROF_BASE, MVL_P_FOCUS, MVL_P_AXIS, MVL_P_HALF, 1, MVL_OPEN, 0.0, best, bestFrac, bestLeaflet, bestW, bestN);
  if (best >= 1e8) return 0.0;
  dOut = best;
  fracOut = bestFrac;
  leafletOut = bestLeaflet;
  nOut = bestN;
  return mitralHalfThickness(MVL_T, bestFrac, bestW);
}

// ---- aortic root profile and cusps (aorticValve.ts) ----
float rootRadiusAt(float t, float phi) {
  float sinusMax = SINUS_R * (1.0 + 0.06 * cos(CUSP_COUNT * (phi - AV_PHI0)) * ((t > 0.0 && t < ROOT_STJ_T) ? sin(PI * t / ROOT_STJ_T) : 0.0));
  float stjR = min(ASC_R, SINUS_R * 0.88);
  if (t < 0.0) { float u = min(1.0, -t / LVOT_TAPER_CM); return AV_R * 0.95 + (LVOT_D / 2.0 - AV_R * 0.95) * u * u * (3.0 - 2.0 * u); }
  if (t < ROOT_SINUS_T) return AV_R + (sinusMax - AV_R) * sin((PI / 2.0) * (t / ROOT_SINUS_T));
  if (t < ROOT_STJ_T) return stjR + (sinusMax - stjR) * 0.5 * (1.0 + cos(PI * (t - ROOT_SINUS_T) / (ROOT_STJ_T - ROOT_SINUS_T)));
  if (t < ROOT_ASC_T) return stjR + (ASC_R - stjR) * 0.5 * (1.0 - cos(PI * (t - ROOT_STJ_T) / (ROOT_ASC_T - ROOT_STJ_T)));
  return ASC_R;
}
// coaptation band on the line to a commissure: [bottom, top]
vec2 aorticBand(float rn) {
  float r = clamp(rn, 0.0, 1.0);
  float margin = CUSP_COUNT == 3.0 ? r * r * r : pow(r, 1.5);
  float top = AVC_EH + (AVC_HCOMM - AVC_EH) * margin;
  if (CUSP_COUNT != 3.0) return vec2(top - (AVC_CH * (1.0 - r) + 0.1 * r), top);
  bool inner = r <= AV_LATERAL_PROFILE_RADIUS;
  float u = inner ? r / AV_LATERAL_PROFILE_RADIUS : (r - AV_LATERAL_PROFILE_RADIUS) / (1.0 - AV_LATERAL_PROFILE_RADIUS);
  float blend = u * u * (3.0 - 2.0 * u);
  float height = inner ? AVC_CH + (AV_LATERAL_COAPTATION_HEIGHT - AVC_CH) * blend : AV_LATERAL_COAPTATION_HEIGHT + (0.1 - AV_LATERAL_COAPTATION_HEIGHT) * blend;
  return vec2(top - height, top);
}
bool aorticContactBand(float t, float r, float phi, out vec2 band) {
  float closed = 1.0 - AVC_OPEN;
  if (closed <= 0.0) return false;
  float hingeT = AVC_HCOMM - 0.1;
  float hingeR = rootRadiusAt(hingeT, phi) - 0.05;
  float restT = (t - AVC_OPEN * hingeT) / closed;
  float restR = (r - AVC_OPEN * hingeR) / closed;
  if (restT < 0.0 || restR < 0.0) return false;
  float wallR = rootRadiusAt(restT, phi);
  if (restR >= wallR * 0.97) return false;
  band = closed * aorticBand(restR / wallR) + AVC_OPEN * hingeT;
  return true;
}
float aorticCuspDistance(float t, float rr, float phi, out float dOut, out float fracOut, out vec2 nOut) {
  dOut = 1e3;
  fracOut = 0.0;
  nOut = vec2(0.0, 1.0);
  if (t < -0.5 || t > AVC_HCOMM + 0.3) return 0.0;
  float per = TWO_PI / CUSP_COUNT;
  float psi = mod(phi - AV_PHI0, per);
  if (psi > per / 2.0) psi -= per;
  float q = psi / (per / 2.0);
  float aq = min(1.0, abs(q));
  float k = 1.0 - sqrt(max(0.0, 1.0 - pow(aq, AV_CROWN_EXPONENT)));
  float tAtt = (AVC_HCOMM - 0.1) * k;
  float rw = rootRadiusAt(tAtt, phi) - 0.02;
  float tTopOpen = AVC_HCOMM - 0.35 + 0.25 * aq * aq;
  float centreWeight = 1.0 - aq * aq;
  float best = 1e9, bestFrac = 0.0;
  vec2 bestN = vec2(0.0, 1.0);
  vec2 a = vec2(0.0);
  for (int i = 0; i < 4; i++) {
    float rn = i == 0 ? 1.0 : (i == 1 ? AV_LATERAL_PROFILE_RADIUS : (i == 2 ? 0.29 : 0.0));
    float tMid = (AVC_EH - AVC_CH) * (1.0 - rn) - AVC_SAG * sin(PI * rn) * (1.0 - aq);
    float edge = aorticBand(rn).x;
    float rc = rw * rn, tc = tMid + (edge - tMid) * k;
    float fo = float(i) / 3.0;
    float to = tAtt + (tTopOpen - tAtt) * fo;
    float wallR = rootRadiusAt(to, phi) - AV_OPEN_WALL_GAP;
    float hangR = AV_R * (1.0 - (1.0 - AV_OPEN_EDGE_FRACTION) * fo);
    float ro = min(wallR, wallR + (hangR - wallR) * centreWeight);
    vec2 b = vec2(rc + (ro - rc) * AVC_OPEN, tc + (to - tc) * AVC_OPEN);
    if (i > 0) {
      vec2 e = b - a;
      float l2 = dot(e, e);
      float sg = l2 > 0.0 ? clamp(dot(vec2(rr, t) - a, e) / l2, 0.0, 1.0) : 0.0;
      float dd = length(a + e * sg - vec2(rr, t));
      if (dd < best) {
        best = dd;
        bestFrac = (float(i - 1) + sg) / 3.0;
        float l = sqrt(l2);
        if (l == 0.0) l = 1.0;
        bestN = vec2(-e.y / l, e.x / l);
      }
    }
    a = b;
  }
  dOut = best;
  fracOut = bestFrac;
  nOut = bestN;
  float tw = (aq - 0.85) / 0.15;
  float w = aq < 0.85 ? 1.0 : 1.0 - tw * tw * (3.0 - 2.0 * tw);
  return cuspHalfThickness(CUSP_T, bestFrac, w, aq);
}

// distance to a 2-segment cusp chain with tapered width
float sdCuspChain(vec3 p, int segBase, float segLen, vec3 w, float halfW, float taper, out float fracOut) {
  float best = 1e9;
  float bestFrac = 0.0;
  for (int i = 0; i < 2; i++) {
    float hw = halfW * (1.0 - taper * (float(i) + 0.5) / 2.0);
    int o = segBase + i * 6;
    vec3 s0 = vec3(P(o), P(o + 1), P(o + 2));
    vec3 d = vec3(P(o + 3), P(o + 4), P(o + 5));
    vec3 r = p - s0;
    float a = clamp(dot(r, d), 0.0, segLen);
    float b = clamp(dot(r, w), -hw, hw);
    vec3 q = s0 + d * a + w * b - p;
    float dd = length(q);
    if (dd < best) { best = dd; bestFrac = (float(i) + a / segLen) / 2.0; }
  }
  fracOut = bestFrac;
  return best;
}

// AHA 17 id of the tissue at (az, levelFrac): the shared segment code (lvSegments.ts), the apical cap codes 17–20 read as 17
int ahaSegment(float az, float levelFrac) {
  int code = int(lvSegmentCode(az, levelFrac, RV_AZA, RV_AZP) + 0.5);
  return code > 16 ? 17 : code;
}
`;
