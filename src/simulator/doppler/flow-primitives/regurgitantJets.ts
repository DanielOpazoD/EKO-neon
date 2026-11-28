import {
  heartAnchors,
  ROOT_EXCURSION,
  type HeartModel,
  type HeartPose,
} from '@/simulator/anatomy/heartModel';
import { lvRegionalMotion } from '@/simulator/anatomy/lvGeometry';
import { lvCompensationGain } from '@/simulator/anatomy/heartPose';
import type { BeatTables } from '@/simulator/cardiac-cycle/cycleModel';
import { cycleStateAt, sampleTable } from '@/simulator/cardiac-cycle/cycleModel';
import type { FlowPeaks } from '@/simulator/cardiac-cycle/flowPeaks';
import { tvShortening } from '@/simulator/anatomy/valveSkirt';
import { smoothstep } from '@/core/vec3';
import { Structure, Tissue } from '@/simulator/anatomy/tissue';
import { atrialScale, RA_ROOF_DESCENT_SHARE } from '@/simulator/anatomy/classify/atria';
import { PV_RADIUS, PV_REACH, pulmonaryVeinSegment } from '@/simulator/anatomy/pulmonaryVeins';
import {
  FLOW_MASK_ALL,
  FLOW_MASK_AR_JET,
  FLOW_MASK_AR_PISA,
  FLOW_MASK_MR_JET,
  FLOW_MASK_MR_PISA,
  type FlowFieldParams,
} from './flowField';
import {
  convergenceWeight,
  fadeOut,
  FAST_JET_SHEAR,
  freeJetCentreline,
  jetProfile,
  jetReach,
  type FlowSample,
} from './jetProfiles';

/** Scratch for the pulmonary vein being sampled (no allocation per sample). */
const PV_SEG = new Float64Array(6);
const PV_WALL = { nx: 0, ny: 0, nz: 0, d: 0 };

/** Pulmonary venous flow in the four vein stubs and their ostia (geometry mirrors heartModel's veins). */
export function samplePulmonaryVeins(
  p: FlowFieldParams,
  tables: BeatTables,
  hp: HeartPose,
  phase: number,
  x: number,
  y: number,
  z: number,
  out: FlowSample,
): void {
  if (p.enabled['pulmonary-vein'] === false) return;
  const pv = p.pulmonaryVeins;
  const la = pv.laCenter,
    lr = pv.laR;
  // quick reject: far from the atrium's sides and back
  if (y > la.y + lr.y * 0.3 + PV_REACH || Math.abs(x - la.x) > lr.x + PV_REACH + 1) return;
  const zTop = la.z - lr.z;
  const zBottom = hp.zAnn + 0.25;
  const czL = (zTop + zBottom) / 2;
  const tm = tables.timings;
  const t = phase * tables.rrS;
  // S: during ventricular systole (annular descent pulls blood into the LA; reverses in severe MR)
  const uS = (t - tm.ejectionStartS + 0.03) / (tm.ejectionEndS - tm.ejectionStartS + 0.08);
  const sWave = uS > 0 && uS < 1 ? Math.pow(Math.sin(Math.PI * uS), 0.9) : 0;
  // D: mirrors the mitral E wave (conduit), Ar: atrial contraction reversal
  const qmvMax = p.flowPeaks.mitral;
  const inE = t > tm.mitralOpenS && t < tm.mitralOpenS + tm.eAccelS + tm.eDecelS + 0.02;
  const dWave = inE ? sampleTable(tables.mitralFlowMlps, phase) / qmvMax : 0;
  const inA = tm.hasAWave && t > tm.aStartS && t < tm.aEndS + 0.03;
  const arWave = inA ? Math.sin((Math.PI * (t - tm.aStartS)) / (tm.aEndS + 0.03 - tm.aStartS)) : 0;
  const vAlong = pv.sMps * sWave + pv.dMps * dWave - pv.arMps * arWave; // + = into the LA, − = reversal
  if (Math.abs(vAlong) < 0.02) return;
  const rzL = (zBottom - zTop) / 2;
  const bo = atrialScale(hp.laBooster, pv.laReservoir, hp.state.contraction);
  const A = heartAnchors(p.heart);
  PV_WALL.nx = A.laPostN.x;
  PV_WALL.ny = A.laPostN.y;
  PV_WALL.nz = A.laPostN.z;
  PV_WALL.d = A.laPostD;
  const disp = vAlong < -0.22 ? 0.12 : (p.turbulence['pulmonary-vein'] ?? 0.06);
  for (let i = 0; i < 4; i++) {
    // exact same wall-clipped segment as classify/atriaVessels.ts (decisions 143, 330, 337)
    pulmonaryVeinSegment(i, la, lr, bo, czL, rzL, PV_SEG, PV_WALL);
    const ox = PV_SEG[0]!,
      oy = PV_SEG[1]!,
      oz = PV_SEG[2]!;
    const ex = PV_SEG[3]!,
      ey = PV_SEG[4]!,
      ez = PV_SEG[5]!;
    // vein axis from the distal end toward the ostium, extended 1.45 cm smoothly into the LA roof
    const ax = ox - ex,
      ay = oy - ey,
      az = oz - ez;
    const L = Math.hypot(ax, ay, az);
    const ux = ax / L,
      uy = ay / L,
      uz = az / L;
    const dx = x - ex,
      dy = y - ey,
      dz = z - ez;
    const along = dx * ux + dy * uy + dz * uz;
    if (along < 0 || along > L + 1.45) continue;
    const qx = dx - ux * along,
      qy = dy - uy * along,
      qz = dz - uz * along;
    const rho = Math.sqrt(qx * qx + qy * qy + qz * qz);
    const R = along <= L ? PV_RADIUS - 0.02 : PV_RADIUS - 0.02 + (along - L) * 0.55; // spreads into the atrium
    const sOut = Math.max(0, along - L);
    if (rho >= jetReach(0.84 * R, sOut)) continue;
    const prof = jetProfile(rho, 0.84 * R, sOut);
    const decay =
      (along <= L ? 1 : 1 / (1 + (along - L) / 0.72)) *
      smoothstep(0, 0.4, along) *
      fadeOut(along, L + 0.5, L + 1.45);
    const v = vAlong * prof * decay;
    out.vx += ux * v;
    out.vy += uy * v;
    out.vz += uz * v;
    out.dispersion = Math.max(out.dispersion, disp);
    out.present = 1;
  }
}

/**
 * Initial 3D unit direction (heart frame, into the LA) of a mitral regurgitant jet from its
 * declared eccentricity angle `mrJetDirRad`:
 * - `|angle| < 0.05`: symmetric central jet `(0, 0, -1)` (functional MR).
 * - `angle < 0` (posterior leaflet prolapse): anteromedial (`-x` septal, `+y` anterior toward aortic root).
 * - `angle > 0` (SAM-related MR): posterolateral (`+x` lateral, `-y` posterior toward posterior LA wall).
 * Preserves exact `z = -cos(angle)` so apical CW alignment stays consistent with the declared angle.
 */
export function mrJetInitialDirection(mrJetDirRad: number): { x: number; y: number; z: number } {
  const theta = Math.abs(mrJetDirRad);
  if (theta < 0.05) return { x: 0, y: 0, z: -1 };
  const sinT = Math.sin(theta);
  const cosT = Math.cos(theta);
  if (mrJetDirRad < 0) {
    // Posterior leaflet prolapse: anteromedial toward the aortic root (+y) and IAS (-x)
    return { x: -0.72 * sinT, y: 0.693974 * sinT, z: -cosT };
  }
  // SAM-related MR: posterolateral toward the posterior LA wall (-y) and lateral wall (+x)
  return { x: 0.64 * sinT, y: -0.768375 * sinT, z: -cosT };
}

/**
 * The left atrial wall an eccentric MR jet attaches to (Fase C): the classifier's atrial body at this instant (the
 * ellipsoid samplePulmonaryVeins rides on, inset to the endocardium), where the jet's free centreline meets it, and the
 * great ellipse it then follows in the plane of its initial direction.
 */
interface MrWallFrame {
  /** Body centre and radii (cm), heart frame. */
  cx: number;
  cy: number;
  cz: number;
  rx: number;
  ry: number;
  rz: number;
  /** Mean radius (cm): arc lengths and lateral distances on the wall. */
  rm: number;
  /** Distance along the jet's initial direction from the orifice to the wall (cm). */
  hitAlong: number;
  /** Impact point on the unit sphere of the body's normalized coordinates, the tangent the jet leaves it along, binormal. */
  hx: number;
  hy: number;
  hz: number;
  tx: number;
  ty: number;
  tz: number;
  bx: number;
  by: number;
  bz: number;
}

/** Share of the body's radii at which the wall jet runs: the endocardium inside the classifier's outer body surface. */
const MR_WALL_INSET = 0.94;
/** Wall-jet decay length (cm): U(s) = U₀·(s0 / (s + s0))^½ along the wall (Glauert's plane wall jet). */
const MR_WALL_S0_CM = 1.0;
/** Direction of the wall jet at the last sample (unit, heart frame): no allocation per sample. */
const MR_WALL_DIR = { x: 0, y: 0, z: 0 };

function mrWallFrame(
  p: FlowFieldParams,
  hp: HeartPose,
  ox: number,
  oy: number,
  oz: number,
  dir: { x: number; y: number; z: number },
): MrWallFrame | null {
  const pv = p.pulmonaryVeins;
  const la = pv.laCenter,
    lr = pv.laR;
  const zTop = la.z - lr.z;
  const zBottom = hp.zAnn + 0.25;
  const bo = atrialScale(hp.laBooster, pv.laReservoir, hp.state.contraction);
  const rx = lr.x * bo * MR_WALL_INSET,
    ry = lr.y * bo * MR_WALL_INSET,
    rz = ((zBottom - zTop) / 2) * MR_WALL_INSET;
  const cx = la.x,
    cy = la.y,
    cz = (zTop + zBottom) / 2;
  if (!(rx > 0.3 && ry > 0.3 && rz > 0.3)) return null;
  // the jet's free centreline against the body: |o + t·d| = 1 in normalized coordinates, its exit
  const qox = (ox - cx) / rx,
    qoy = (oy - cy) / ry,
    qoz = (oz - cz) / rz;
  const qdx = dir.x / rx,
    qdy = dir.y / ry,
    qdz = dir.z / rz;
  const a = qdx * qdx + qdy * qdy + qdz * qdz;
  const b = qox * qdx + qoy * qdy + qoz * qdz;
  const c = qox * qox + qoy * qoy + qoz * qoz - 1;
  const disc = b * b - a * c;
  if (!(disc > 0)) return null;
  const hitAlong = (-b + Math.sqrt(disc)) / a;
  if (!(hitAlong > 0.6) || hitAlong > 5.2) return null;
  const hx = qox + qdx * hitAlong,
    hy = qoy + qdy * hitAlong,
    hz = qoz + qdz * hitAlong;
  const hl = Math.hypot(hx, hy, hz) || 1;
  const ux = hx / hl,
    uy = hy / hl,
    uz = hz / hl;
  // the tangent the jet keeps at the wall: its direction (normalized coordinates) less the wall-normal part, or toward
  // the roof when it meets the wall head on
  const dl = Math.sqrt(a);
  let tx = qdx / dl,
    ty = qdy / dl,
    tz = qdz / dl;
  const dn = tx * ux + ty * uy + tz * uz;
  tx -= dn * ux;
  ty -= dn * uy;
  tz -= dn * uz;
  let tl = Math.hypot(tx, ty, tz);
  if (tl < 0.15) {
    // toward the roof (−z), less its wall-normal part
    tx = -uz * ux;
    ty = -uz * uy;
    tz = -1 + uz * uz;
    tl = Math.hypot(tx, ty, tz) || 1;
  }
  tx /= tl;
  ty /= tl;
  tz /= tl;
  return {
    cx,
    cy,
    cz,
    rx,
    ry,
    rz,
    rm: Math.cbrt(rx * ry * rz),
    hitAlong,
    hx: ux,
    hy: uy,
    hz: uz,
    tx,
    ty,
    tz,
    bx: uy * tz - uz * ty,
    by: uz * tx - ux * tz,
    bz: ux * ty - uy * tx,
  };
}

/**
 * Speed (m/s) of the MR wall jet at (x, y, z), its direction left in MR_WALL_DIR: a Glauert wall jet past the impact,
 * its centre `dc` off the wall in a sheet of half-thickness `delta`, spreading along the wall over a lateral half-width
 * `b` that grows five times faster than the sheet thickens, decaying as (s0 / (s + s0))^½ and gone 6 cm round the dome.
 */
function sampleMrWallJet(
  w: MrWallFrame,
  x: number,
  y: number,
  z: number,
  vImpact: number,
  rImpact: number,
): number {
  const qx = (x - w.cx) / w.rx,
    qy = (y - w.cy) / w.ry,
    qz = (z - w.cz) / w.rz;
  const rn = Math.hypot(qx, qy, qz);
  if (rn < 0.3 || rn > 1.25) return 0;
  const px = qx / rn,
    py = qy / rn,
    pz = qz / rn;
  const along = px * w.tx + py * w.ty + pz * w.tz;
  const toward = px * w.hx + py * w.hy + pz * w.hz;
  const phi = Math.atan2(along, toward);
  const s = phi * w.rm;
  if (s < -0.4 || s > 6.2) return 0;
  const sp = Math.max(0, s);
  // distance to the wall along the body's radius through the point, and across the jet's plane on the wall
  const radial = Math.hypot(x - w.cx, y - w.cy, z - w.cz);
  const dWall = ((1 - rn) * radial) / rn;
  const off = Math.asin(Math.max(-1, Math.min(1, px * w.bx + py * w.by + pz * w.bz))) * w.rm;
  const delta = 0.34 + 0.07 * sp;
  const dc = 0.3 + 0.05 * sp;
  const b = Math.max(0.35, rImpact) + 0.38 * sp;
  const fn = Math.exp(-((dWall - dc) * (dWall - dc)) / (delta * delta));
  const fl = Math.exp(-(off * off) / (b * b));
  const v =
    vImpact *
    Math.sqrt(MR_WALL_S0_CM / (sp + MR_WALL_S0_CM)) *
    fn *
    fl *
    smoothstep(-0.4, 0.2, s) *
    fadeOut(sp, 4.2, 6.2);
  if (v <= 0.03) return 0;
  // the great ellipse's tangent at the point, back to the heart frame
  const sinP = Math.sin(phi),
    cosP = Math.cos(phi);
  const gx = (-w.hx * sinP + w.tx * cosP) * w.rx,
    gy = (-w.hy * sinP + w.ty * cosP) * w.ry,
    gz = (-w.hz * sinP + w.tz * cosP) * w.rz;
  const gl = Math.hypot(gx, gy, gz) || 1;
  MR_WALL_DIR.x = gx / gl;
  MR_WALL_DIR.y = gy / gl;
  MR_WALL_DIR.z = gz / gl;
  return v;
}

/**
 * The centre of the MR wall jet `s` cm along the wall past its impact (heart frame), the inward-facing... outward wall
 * normal there (unit), and where the free jet met the wall: null for a central jet or one that never reaches the wall.
 * For tests and teaching overlays.
 */
export function mrWallJetCentre(
  p: FlowFieldParams,
  hp: HeartPose,
  s: number,
): { x: number; y: number; z: number; nx: number; ny: number; nz: number; hitAlong: number } | null {
  if (smoothstep(0.18, 0.55, Math.abs(p.mrJetDirRad)) <= 0) return null;
  const dir = mrJetInitialDirection(p.mrJetDirRad);
  const w = mrWallFrame(p, hp, p.mvCenter.x, p.mvCenter.y, hp.zAnn - 0.1, dir);
  if (!w) return null;
  const phi = s / w.rm;
  const ux = w.hx * Math.cos(phi) + w.tx * Math.sin(phi),
    uy = w.hy * Math.cos(phi) + w.ty * Math.sin(phi),
    uz = w.hz * Math.cos(phi) + w.tz * Math.sin(phi);
  const gx = ux * w.rx,
    gy = uy * w.ry,
    gz = uz * w.rz;
  const gl = Math.hypot(gx, gy, gz);
  const k = 1 - (0.3 + 0.05 * Math.max(0, s)) / gl;
  // the ellipsoid's outward normal: the gradient of |q|², (u / R) componentwise
  const nx = ux / w.rx,
    ny = uy / w.ry,
    nz = uz / w.rz;
  const nl = Math.hypot(nx, ny, nz) || 1;
  return {
    x: w.cx + gx * k,
    y: w.cy + gy * k,
    z: w.cz + gz * k,
    nx: nx / nl,
    ny: ny / nl,
    nz: nz / nl,
    hitAlong: w.hitAlong,
  };
}

/** Regurgitant jets (MR in systole into the LA, AR in diastole into the LV) with proximal flow convergence (PISA). */
export function sampleRegurgitantJets(
  p: FlowFieldParams,
  tables: BeatTables,
  hp: HeartPose,
  phase: number,
  x: number,
  y: number,
  z: number,
  out: FlowSample,
  mask = FLOW_MASK_ALL,
): void {
  const zAnn = hp.zAnn;
  // ---- MR jet (vena contracta neck + 3D eccentric Coanda LA jet + hemispheric LV PISA) ----
  const allowMrJet = (mask & FLOW_MASK_MR_JET) !== 0;
  const allowMrPisa = (mask & FLOW_MASK_MR_PISA) !== 0;
  if ((allowMrJet || allowMrPisa) && p.enabled['mr-jet'] !== false && p.mrEroCm2 > 0) {
    const q = sampleTable(tables.mrFlowMlps, phase);
    if (q > 1) {
      const vj = Math.max(1e-3, sampleTable(tables.mrVelocityMps, phase));
      const areaEff = Math.min(p.mrEroCm2, q / vj / 100);
      // True orifice radius: 2·rVc equals the circular-equivalent vena contracta width
      const rVc = Math.sqrt(areaEff / Math.PI);
      const r0 = rVc * 1.1;
      const ox = p.mvCenter.x,
        oy = p.mvCenter.y,
        oz = zAnn - 0.1;
      const dir = mrJetInitialDirection(p.mrJetDirRad);
      const dx = x - ox,
        dy = y - oy,
        dz = z - oz;
      const along = dx * dir.x + dy * dir.y + dz * dir.z; // initial projection into the LA
      const ecc = smoothstep(0.18, 0.55, Math.abs(p.mrJetDirRad));
      // An eccentric jet meets the atrial wall and runs along it (Coandă, Fase C): where the free jet reaches the wall
      const wall = ecc > 0 && allowMrJet ? mrWallFrame(p, hp, ox, oy, oz, dir) : null;
      const hitAlong = wall ? wall.hitAlong : Infinity;
      let fvx = 0,
        fvy = 0,
        fvz = 0,
        fDisp = 0,
        fSpeed = 0;
      if (dz <= 0 && allowMrJet && along >= 0 && along < Math.min(5.8, hitAlong + 0.5)) {
        // Before the wall the eccentric jet is a straight free jet; a central jet that never reaches it keeps the gentle
        // Coanda roll toward the roof of decision-era jets.
        const coandaU = ecc > 0 && !wall ? smoothstep(1.0, 3.6, along) : 0;
        const bend = ecc * 0.42 * coandaU * Math.max(0, along - 1.0);
        const sHoriz = along - bend;
        const cx = ox + dir.x * sHoriz;
        const cy = oy + dir.y * sHoriz;
        const cz = oz + dir.z * along - 0.14 * bend;
        const rho = Math.hypot(x - cx, y - cy, z - cz);

        // Vena contracta neck (first 0.35 cm beyond coaptation): collimated core matching rVc before
        // organic free-jet / Coanda shear expansion into the LA.
        const expandAlong = Math.max(0, along - 0.35);
        const spreadSlope = 0.28 * (1 - 0.055 * Math.min(4, expandAlong));
        const coandaBroaden = 1 + 0.26 * ecc * (wall ? 0 : 1) * smoothstep(1.2, 3.5, along);
        const rHalf = 0.84 * (rVc + expandAlong * spreadSlope) * coandaBroaden;
        if (rho < jetReach(rHalf, expandAlong, FAST_JET_SHEAR)) {
          const prof = jetProfile(rho, rHalf, expandAlong, FAST_JET_SHEAR);
          const decay =
            freeJetCentreline(along, 2 * rVc) *
            fadeOut(along, 4.0, 5.7) *
            (wall ? fadeOut(along, hitAlong - 0.25, hitAlong + 0.45) : 1);
          const v = vj * prof * decay;
          // Local tangent follows the Coanda wall curve toward the LA roof (central-wall roll only without a wall jet)
          const roll = wall ? 0 : ecc * smoothstep(1.0, 3.2, along);
          const hSlope = Math.max(0.18, 1 - 0.58 * roll);
          const txRaw = dir.x * hSlope;
          const tyRaw = dir.y * hSlope;
          const tzRaw = dir.z - 0.14 * roll;
          const tLen = Math.hypot(txRaw, tyRaw, tzRaw) || 1;
          fvx = (txRaw / tLen) * v;
          fvy = (tyRaw / tLen) * v;
          fvz = (tzRaw / tLen) * v;
          fSpeed = v;
          // Laminar core inside the vena contracta, ramping smoothly into turbulent variance distally
          const neckBlend = smoothstep(0.15, 0.7, along);
          const baseTurb = (p.turbulence['mr-jet'] ?? 0.3) * (0.35 + 0.65 * neckBlend);
          fDisp = baseTurb + 0.22 * neckBlend * Math.min(1, rho / Math.max(0.08, rHalf / 0.84));
        }
      }
      if (wall && dz <= 0) {
        // the wall jet: along the atrial wall past the impact, thin off the wall and spreading along it
        const rImpact =
          0.84 *
          (rVc +
            Math.max(0, wall.hitAlong - 0.35) *
              0.28 *
              (1 - 0.055 * Math.min(4, Math.max(0, wall.hitAlong - 0.35))));
        const vImpact = vj * freeJetCentreline(wall.hitAlong, 2 * rVc) * fadeOut(wall.hitAlong, 4.0, 5.7);
        const w = sampleMrWallJet(wall, x, y, z, vImpact, rImpact);
        if (w > fSpeed) {
          fvx = MR_WALL_DIR.x * w;
          fvy = MR_WALL_DIR.y * w;
          fvz = MR_WALL_DIR.z * w;
          fSpeed = w;
          fDisp = (p.turbulence['mr-jet'] ?? 0.3) + 0.2;
        }
      }
      if (fSpeed > 0.03) {
        out.vx += fvx;
        out.vy += fvy;
        out.vz += fvz;
        out.dispersion = Math.max(out.dispersion, fDisp);
        out.present = 1;
      } else if (dz > 0 && allowMrPisa) {
        // Proximal Isovelocity Surface Area (PISA): hemispheric continuity v(d) = Q / (2·π·d²·100)
        // on the LV side (dz > 0) of the anatomical mitral coaptation plane, even for eccentric jets.
        const d = Math.hypot(dx, dy, dz);
        const rShell30 = Math.sqrt(q / (2 * Math.PI * 30)); // radius where v = 0.30 m/s
        const rMax = Math.min(2.6, Math.max(1.8, rShell30 * 1.9));
        if (d < rMax && d > 0.05) {
          const vHemisphere = q / (2 * Math.PI * d * d * 100);
          const rhoAnn = Math.hypot(dx, dy);
          // Gentler equatorial roll-off than inflow funnels so the 0.35–0.45 m/s shell forms a clean hemisphere
          const hemiWeight = convergenceWeight(rhoAnn, dz, r0, 0.12, 0.38);
          const v = Math.min(vj * 0.85, vHemisphere) * fadeOut(d, rMax * 0.78, rMax) * hemiWeight;
          if (v > 0.03) {
            out.vx += (-dx / d) * v;
            out.vy += (-dy / d) * v;
            out.vz += (-dz / d) * v;
            // Proximal convergence is laminar (low spectral variance until the orifice)
            out.dispersion = Math.max(out.dispersion, 0.06);
            out.present = 1;
          }
        }
      }
    }
  }
  // ---- AR jet (vena contracta neck + anterior-mitral-directed LVOT jet + 3D aortic-side PISA) ----
  const allowArJet = (mask & FLOW_MASK_AR_JET) !== 0;
  const allowArPisa = (mask & FLOW_MASK_AR_PISA) !== 0;
  if ((allowArJet || allowArPisa) && p.enabled['ar-jet'] !== false && p.arEroCm2 > 0) {
    const q = sampleTable(tables.arFlowMlps, phase);
    if (q > 1) {
      const vj = q / p.arEroCm2 / 100;
      const rVc = Math.sqrt(p.arEroCm2 / Math.PI);
      const r0 = rVc * 1.1;
      const ax = p.avAxis;
      const cz = p.avCenter.z + zAnn * ROOT_EXCURSION;
      const dx = x - p.avCenter.x,
        dy = y - p.avCenter.y,
        dz = z - cz;
      const t = dx * ax.x + dy * ax.y + dz * ax.z; // along the root axis (negative = into the LV)
      const along = -t;
      if (along >= 0 && allowArJet && along < 6) {
        // Past the collimated vena contracta neck (0..0.35 cm), AR jets drift subtly posteriorly (-y)
        // along the anterior mitral leaflet toward the LV inflow while keeping orifice alignment exact.
        const amlBend = 0.11 * smoothstep(0.35, 2.2, along);
        const shiftY = -amlBend * Math.max(0, along - 0.35);
        const qx = dx - ax.x * t,
          qy = dy - ax.y * t - shiftY,
          qz = dz - ax.z * t;
        const rho = Math.sqrt(qx * qx + qy * qy + qz * qz);
        const expandAlong = Math.max(0, along - 0.35);
        const spreadSlope = 0.3 * (1 - 0.05 * Math.min(4, expandAlong));
        const rHalf = 0.84 * (rVc + expandAlong * spreadSlope);
        if (rho < jetReach(rHalf, expandAlong, FAST_JET_SHEAR)) {
          const prof = jetProfile(rho, rHalf, expandAlong, FAST_JET_SHEAR);
          const decay = freeJetCentreline(along, 2 * rVc) * fadeOut(along, 4.3, 5.9);
          const v = vj * prof * decay;
          const vxRaw = -ax.x;
          const vyRaw = -ax.y - amlBend;
          const vzRaw = -ax.z;
          const vLen = Math.hypot(vxRaw, vyRaw, vzRaw) || 1;
          out.vx += (vxRaw / vLen) * v;
          out.vy += (vyRaw / vLen) * v;
          out.vz += (vzRaw / vLen) * v;
          const neckBlend = smoothstep(0.15, 0.7, along);
          const baseTurb = (p.turbulence['ar-jet'] ?? 0.38) * (0.35 + 0.65 * neckBlend);
          out.dispersion = Math.max(
            out.dispersion,
            baseTurb + 0.22 * neckBlend * Math.min(1, rho / Math.max(0.08, rHalf / 0.84)),
          );
          out.present = 1;
        }
      } else if (along < 0 && allowArPisa) {
        // 3D hemispheric aortic-side PISA conserving Q_AR = 2·π·d²·v
        const d = Math.hypot(dx, dy, dz);
        const rShell30 = Math.sqrt(q / (2 * Math.PI * 30));
        const rMax = Math.min(2.4, Math.max(1.6, rShell30 * 1.9));
        if (d > 0.05 && d < rMax) {
          const qx = dx - ax.x * t,
            qy = dy - ax.y * t,
            qz = dz - ax.z * t;
          const rho = Math.sqrt(qx * qx + qy * qy + qz * qz);
          const vHemisphere = q / (2 * Math.PI * d * d * 100);
          const hemiWeight = convergenceWeight(rho, -along, r0, 0.12, 0.38);
          const v =
            Math.min(vj * 0.85, vHemisphere) * fadeOut(d, rMax * 0.78, rMax) * hemiWeight;
          if (v > 0.03) {
            out.vx += (-dx / d) * v;
            out.vy += (-dy / d) * v;
            out.vz += (-dz / d) * v;
            out.dispersion = Math.max(out.dispersion, 0.06);
            out.present = 1;
          }
        }
      }
    }
  }
}

/** The right atrial vestibule (cm above the tricuspid annulus): the smooth muscle the ring carries whole (decision 367). */
export const RA_VESTIBULE_CM = 1.5;

/** Structures of the right ventricle, whose tissue moves with the tricuspid annulus. */
const RV_TISSUE = new Set<number>([
  Structure.RvWall,
  Structure.TricuspidAnnulus,
  Structure.RvPapillary,
  Structure.ModeratorBand,
]);

/** Basal annular myocardial cuff thickness (cm) that moves rigidly at full annular velocity before tapering to the fixed apex. */
const BASAL_ANNULAR_CUFF_CM = 0.85;

/** Axial extent (cm) a valve's leaflets sweep from its annulus (mirrors valveClicks.ts). */
const LEAFLET_SWEEP_CM = 1.5;
/** Spatial reach (cm) of valve leaflet clutter velocity around each valve cylinder. */
const LEAFLET_REACH_CM = 0.85;
/** Effective leaflet excursion length (m) scaling d(open)/dt into physiological snap speed (~12–24 cm/s). */
const LEAFLET_EXCURSION_M = 0.012;
/** Peak physiological valve leaflet snap velocity clamp (m/s). */
const LEAFLET_MAX_SPEED_MPS = 0.24;

function clampLeafletSpeed(dOpenDt: number): number {
  const v = dOpenDt * LEAFLET_EXCURSION_M;
  return v > LEAFLET_MAX_SPEED_MPS
    ? LEAFLET_MAX_SPEED_MPS
    : v < -LEAFLET_MAX_SPEED_MPS
      ? -LEAFLET_MAX_SPEED_MPS
      : v;
}

/** Distance (cm) from heart-frame point (px,py,pz) to a valve's leaflet sweep cylinder. */
function valveCylinderDistance(
  px: number,
  py: number,
  pz: number,
  cx: number,
  cy: number,
  cz: number,
  ax: number,
  ay: number,
  az: number,
  radius: number,
): number {
  const dx = px - cx;
  const dy = py - cy;
  const dz = pz - cz;
  const t = dx * ax + dy * ay + dz * az;
  const radial = Math.max(0, Math.hypot(dx - ax * t, dy - ay * t, dz - az * t) - radius);
  const axial = t < 0 ? -t : t > LEAFLET_SWEEP_CM ? t - LEAFLET_SWEEP_CM : 0;
  return Math.hypot(radial, axial);
}

/**
 * Instantaneous LV radial cavity expansion rate dR/dt (cm/s; negative during systolic inward contraction) of the walls
 * that move normally. Where regional abnormalities withhold part of the excursion, the normal walls carry it
 * (compensatory hyperkinesia, computeChamberPose) and move lvCompensationGain times faster than the volume alone would
 * move them: in the inferior infarction ×1.3-1.5 through the ejection and the filling.
 */
function lvRadialRateCmps(heart: HeartModel, tables: BeatTables, phase: number): number {
  const dPhase = 0.005;
  const edv = Math.max(1, tables.edvMl);
  const vPlus = Math.max(1, sampleTable(tables.lvVolumeMl, (phase + dPhase) % 1));
  const vMinus = Math.max(1, sampleTable(tables.lvVolumeMl, (phase - dPhase + 1) % 1));
  const rate =
    (heart.lv.rMax * (Math.cbrt(vPlus / edv) - Math.cbrt(vMinus / edv))) /
    (2 * dPhase * Math.max(0.1, tables.rrS));
  return heart.regionalMeanFrac > 0
    ? rate *
        lvCompensationGain(
          heart,
          sampleTable(tables.lvVolumeMl, phase),
          heart.lv.lengthCm - heart.physiology.mapseCm * sampleTable(tables.longitudinal, phase),
        )
    : rate;
}

/**
 * Instantaneous LV isovolumic contraction (IVC) and relaxation (IVR) biphasic annular reshaping
 * transients (cm/s): distinct positive-then-negative pre-ejection IVC spike and smaller post-AVC
 * biphasic IVR deflection delineating S′ from e′ on pulsed-wave TDI while staying well below e′/S′.
 */
export function lvIsoTransientCmps(heart: HeartModel, tables: BeatTables, phase: number): number {
  const t = phase * tables.rrS;
  const tm = tables.timings;
  const ivcDur = Math.max(0.025, tm.ejectionStartS);
  if (t > 0.004 && t < tm.ejectionStartS - 0.002) {
    const u = (t - 0.004) / Math.max(0.015, ivcDur - 0.006);
    const amp = Math.min(5.2, 0.44 * heart.physiology.ePrimeSeptalCmps);
    return amp * Math.sin(2 * Math.PI * u) * Math.sin(Math.PI * u);
  }
  if (t > tm.ejectionEndS + 0.004 && t < tm.mitralOpenS - 0.006) {
    const ivrDur = tm.mitralOpenS - tm.ejectionEndS - 0.01;
    if (ivrDur > 0.016) {
      const u = (t - (tm.ejectionEndS + 0.004)) / ivrDur;
      const amp = Math.min(3.8, 0.32 * heart.physiology.ePrimeSeptalCmps);
      return -amp * Math.sin(2 * Math.PI * u) * Math.sin(Math.PI * u);
    }
  }
  return 0;
}

/**
 * Instantaneous RV tricuspid annular IVC (myocardial isovolumic acceleration) and IVR biphasic
 * transients (cm/s), driven strictly by right-heart pulmonary/tricuspid valve timings.
 */
export function rvIsoTransientCmps(heart: HeartModel, tables: BeatTables, phase: number): number {
  const t = phase * tables.rrS;
  const tm = tables.timings;
  const ivcDur = Math.max(0.022, tm.pulmonaryOpenS);
  if (t > 0.004 && t < tm.pulmonaryOpenS - 0.002) {
    const u = (t - 0.004) / Math.max(0.015, ivcDur - 0.006);
    const amp = Math.min(5.4, 0.4 * heart.physiology.sPrimeTricuspidCmps);
    return amp * Math.sin(2 * Math.PI * u) * Math.sin(Math.PI * u);
  }
  if (t > tm.pulmonaryCloseS + 0.004 && t < tm.tricuspidOpenS - 0.006) {
    const ivrDur = tm.tricuspidOpenS - tm.pulmonaryCloseS - 0.01;
    if (ivrDur > 0.014) {
      const u = (t - (tm.pulmonaryCloseS + 0.004)) / ivrDur;
      const amp = Math.min(3.4, 0.24 * heart.physiology.sPrimeTricuspidCmps);
      return -amp * Math.sin(2 * Math.PI * u) * Math.sin(Math.PI * u);
    }
  }
  return 0;
}

/**
 * Tissue (myocardial) velocity for TDI and Color wall clutter (m/s, heart frame) at heart-frame
 * point (x, y, z): combines longitudinal annular excursion (preserving basal annular cuff velocity
 * before tapering toward the apex, with IVC/IVR transients) and LV radial systolic contraction /
 * diastolic expansion derived from `tables.lvVolumeMl` (with the normal walls' compensation, lvRadialRateCmps) and
 * modulated by the regional wall motion (`lvRegionalMotion`).
 */
export function sampleTissueVelocity(
  heart: HeartModel,
  tables: BeatTables,
  phase: number,
  x: number,
  y: number,
  z: number,
  structure = -1,
): { vx: number; vy: number; vz: number } {
  const A = heartAnchors(heart);
  if (structure === Structure.RaWall) {
    const free = Math.min(1, Math.max(0, (A.tvCenter.x + 0.5 * A.tvR - x) / A.tvR));
    if (free > 0) {
      const annulusZ =
        A.tvCenter.z + heart.physiology.tapseCm * sampleTable(tables.rvLongitudinal, phase);
      const above = Math.min(
        1,
        Math.max(0, (annulusZ - RA_VESTIBULE_CM - z) / (2 * A.raR.z - RA_VESTIBULE_CM)),
      );
      const vRing =
        (heart.physiology.tapseCm *
          sampleTable(tables.rvLongitudinalVelocity, phase) *
          (1 - (1 - RA_ROOF_DESCENT_SHARE) * above)) /
        100;
      const vSeptal = free < 1 ? sampleTissueVelocity(heart, tables, phase, x, y, z).vz : 0;
      return { vx: 0, vy: 0, vz: free * vRing + (1 - free) * vSeptal };
    }
  }
  if (RV_TISSUE.has(structure)) {
    const apexZ = A.rvApexFrac * heart.lv.lengthCm;
    const annulusZ =
      A.tvCenter.z + heart.physiology.tapseCm * sampleTable(tables.rvLongitudinal, phase);
    const rvLevel = Math.min(
      1,
      Math.max(
        0,
        (z - annulusZ - BASAL_ANNULAR_CUFF_CM) /
          Math.max(1, apexZ - annulusZ - BASAL_ANNULAR_CUFF_CM),
      ),
    );
    const dPhase = 0.004;
    const shortening = (ph: number) =>
      tvShortening(cycleStateAt(tables, ph).contraction, A.rvRadialScale);
    const rate =
      (shortening((phase + dPhase) % 1) - shortening((phase - dPhase + 1) % 1)) /
      (2 * dPhase * tables.rrS);
    const lateral = Math.min(1, Math.max(0, (A.tvCenter.x + A.tvR - x) / (2 * A.tvR)));
    const rvIsoCmps = rvIsoTransientCmps(heart, tables, phase);
    return {
      vx: (2 * A.tvR * rate * lateral * (1 - rvLevel)) / 100,
      vy: 0,
      vz:
        ((heart.physiology.tapseCm * sampleTable(tables.rvLongitudinalVelocity, phase) + rvIsoCmps) *
          (1 - rvLevel)) /
        100,
    };
  }
  const longVel = sampleTable(tables.longitudinalVelocity, phase); // fraction of MAPSE per s
  const mapse = heart.physiology.mapseCm;
  const annulusZ = mapse * sampleTable(tables.longitudinal, phase);
  const level = Math.min(
    1,
    Math.max(
      0,
      (z - annulusZ - BASAL_ANNULAR_CUFF_CM) /
        Math.max(1, heart.lv.lengthCm - annulusZ - BASAL_ANNULAR_CUFF_CM),
    ),
  );
  // Septal (-x) to lateral (+x) velocity gradient referenced to the mitral annulus centre.
  const dxMv = x - A.mvCenter.x;
  const dyMv = y - A.mvCenter.y;
  const rho = Math.hypot(dxMv, dyMv);
  const lateralness = rho > 1e-6 ? 0.5 * (1 + dxMv / rho) : 0.5;
  const wall =
    1 + (heart.physiology.ePrimeLateralCmps / heart.physiology.ePrimeSeptalCmps - 1) * lateralness;

  const isoCmps = lvIsoTransientCmps(heart, tables, phase);
  const vz = ((mapse * longVel + isoCmps) * (1 - level) * wall) / 100; // cm/s → m/s, +z = toward the apex (systole)

  // LV radial contraction / relaxation velocity projected onto (x, y) / rhoLv, scaled by the regional field the
  // classifier draws the wall with (lvRegionalMotion: the endocardium makes that share of the normal excursion, so it
  // moves at that share of the normal speed, outward where dyskinetic), continuous across the segment borders.
  // Tapers smoothly to 0 at the fibrous mitral annulus (levelFrac <= 0) so basal TDI annular velocities remain exact.
  let vx = 0;
  let vy = 0;
  const rhoLv = Math.hypot(x, y);
  if (z > annulusZ - 0.3 && z < heart.lv.lengthCm + 1.2 && rhoLv > 0.2 && rhoLv < heart.lv.rMax * 2.5) {
    const levelFrac = Math.min(1, Math.max(0, (z - annulusZ) / Math.max(1, heart.lv.lengthCm - annulusZ)));
    const radialEnv = smoothstep(0, 0.18, levelFrac) * (1 - 0.55 * levelFrac * levelFrac);
    if (radialEnv > 1e-4) {
      // the field is read only where the case declares regional motion, as the classifier reads it
      const motion =
        heart.regionalMeanFrac > 0 ? lvRegionalMotion(heart.segAmp, Math.atan2(y, x), levelFrac) : 1;
      if (Math.abs(motion) > 1e-4) {
        const vRadMps = (lvRadialRateCmps(heart, tables, phase) * radialEnv * motion) / 100;
        vx = (x / rhoLv) * vRadMps;
        vy = (y / rhoLv) * vRadMps;
      }
    }
  }
  return { vx, vy, vz };
}

/** Precomputed per-frame myocardial and valve-leaflet velocity rates for zero-allocation Color clutter sampling. */
export interface ClutterContext {
  annulusZ: number;
  lvLongVelCmps: number;
  lvIsoCmps: number;
  lvRadialRateCmps: number;
  wallRatioDelta: number;
  rvAnnulusZ: number;
  rvApexZ: number;
  rvLongVelCmps: number;
  rvShortRate: number;
  vMvMps: number;
  vAvMps: number;
  vTvMps: number;
  vPvMps: number;
  anyLeafletActive: boolean;
  mvCx: number;
  mvCy: number;
  mvCz: number;
  mvR: number;
  avCx: number;
  avCy: number;
  avCz: number;
  avAx: number;
  avAy: number;
  avAz: number;
  avR: number;
  tvCx: number;
  tvCy: number;
  tvCz: number;
  tvR: number;
  pvCx: number;
  pvCy: number;
  pvCz: number;
  pvAx: number;
  pvAy: number;
  pvAz: number;
  pvR: number;
  raRz: number;
}

/**
 * Precompute frame-constant myocardial and valve opening/closing rates once per Color Doppler update
 * so sampling `vWall` over myocardial/valve ROI texels incurs zero heap allocations and minimal math.
 */
export function prepareClutterContext(
  heart: HeartModel,
  tables: BeatTables,
  hp: HeartPose,
  phase: number,
  flowPeaks?: FlowPeaks,
): ClutterContext {
  const A = heartAnchors(heart);
  const dPhase = 0.004;
  const inv2Dt = 1 / (2 * dPhase * Math.max(0.1, tables.rrS));
  const sPlus = cycleStateAt(tables, (phase + dPhase) % 1, flowPeaks);
  const sMinus = cycleStateAt(tables, (phase - dPhase + 1) % 1, flowPeaks);
  const rvShortRate =
    (tvShortening(sPlus.contraction, A.rvRadialScale) -
      tvShortening(sMinus.contraction, A.rvRadialScale)) *
    inv2Dt;
  const vMvMps = clampLeafletSpeed((sPlus.mvOpen - sMinus.mvOpen) * inv2Dt);
  const vAvMps = clampLeafletSpeed((sPlus.avOpen - sMinus.avOpen) * inv2Dt);
  const vTvMps = clampLeafletSpeed((sPlus.tvOpen - sMinus.tvOpen) * inv2Dt);
  const vPvMps = clampLeafletSpeed((sPlus.pvOpen - sMinus.pvOpen) * inv2Dt);
  const mapse = heart.physiology.mapseCm;
  const tapse = heart.physiology.tapseCm;
  return {
    annulusZ: mapse * sampleTable(tables.longitudinal, phase),
    lvLongVelCmps: mapse * sampleTable(tables.longitudinalVelocity, phase),
    lvIsoCmps: lvIsoTransientCmps(heart, tables, phase),
    lvRadialRateCmps: lvRadialRateCmps(heart, tables, phase),
    wallRatioDelta: heart.physiology.ePrimeLateralCmps / heart.physiology.ePrimeSeptalCmps - 1,
    rvAnnulusZ: A.tvCenter.z + tapse * sampleTable(tables.rvLongitudinal, phase),
    rvApexZ: A.rvApexFrac * heart.lv.lengthCm,
    rvLongVelCmps: tapse * sampleTable(tables.rvLongitudinalVelocity, phase),
    rvShortRate,
    vMvMps,
    vAvMps,
    vTvMps,
    vPvMps,
    anyLeafletActive:
      Math.abs(vMvMps) > 0.005 ||
      Math.abs(vAvMps) > 0.005 ||
      Math.abs(vTvMps) > 0.005 ||
      Math.abs(vPvMps) > 0.005,
    mvCx: A.mvCenter.x,
    mvCy: A.mvCenter.y,
    mvCz: hp.zAnn,
    mvR: A.mvR,
    avCx: A.avCenter.x,
    avCy: A.avCenter.y,
    avCz: A.avCenter.z + hp.zAnn * ROOT_EXCURSION,
    avAx: A.avAxis.x,
    avAy: A.avAxis.y,
    avAz: A.avAxis.z,
    avR: A.avR,
    tvCx: A.tvCenter.x,
    tvCy: A.tvCenter.y,
    tvCz: A.tvCenter.z + hp.tvZ,
    tvR: A.tvR,
    pvCx: A.rvotB.x,
    pvCy: A.rvotB.y,
    pvCz: A.rvotB.z,
    pvAx: A.paDir.x,
    pvAy: A.paDir.y,
    pvAz: A.paDir.z,
    pvR: A.pvR,
    raRz: A.raR.z,
  };
}

function leafletVelocityFromCtx(
  ctx: ClutterContext,
  x: number,
  y: number,
  z: number,
  out: { vx: number; vy: number; vz: number },
): void {
  out.vx = 0;
  out.vy = 0;
  out.vz = 0;
  if (!ctx.anyLeafletActive) return;

  let bestD = LEAFLET_REACH_CM;
  let speed = 0;
  let ax = 0;
  let ay = 0;
  let az = 0;

  if (Math.abs(ctx.vMvMps) > 0.005) {
    const d = valveCylinderDistance(x, y, z, ctx.mvCx, ctx.mvCy, ctx.mvCz, 0, 0, 1, ctx.mvR);
    if (d < bestD) {
      bestD = d;
      speed = ctx.vMvMps;
      ax = 0;
      ay = 0;
      az = 1;
    }
  }
  if (Math.abs(ctx.vAvMps) > 0.005) {
    const d = valveCylinderDistance(
      x,
      y,
      z,
      ctx.avCx,
      ctx.avCy,
      ctx.avCz,
      ctx.avAx,
      ctx.avAy,
      ctx.avAz,
      ctx.avR,
    );
    if (d < bestD) {
      bestD = d;
      speed = ctx.vAvMps;
      ax = ctx.avAx;
      ay = ctx.avAy;
      az = ctx.avAz;
    }
  }
  if (Math.abs(ctx.vTvMps) > 0.005) {
    const d = valveCylinderDistance(x, y, z, ctx.tvCx, ctx.tvCy, ctx.tvCz, 0, 0, 1, ctx.tvR);
    if (d < bestD) {
      bestD = d;
      speed = ctx.vTvMps;
      ax = 0;
      ay = 0;
      az = 1;
    }
  }
  if (Math.abs(ctx.vPvMps) > 0.005) {
    const d = valveCylinderDistance(
      x,
      y,
      z,
      ctx.pvCx,
      ctx.pvCy,
      ctx.pvCz,
      ctx.pvAx,
      ctx.pvAy,
      ctx.pvAz,
      ctx.pvR,
    );
    if (d < bestD) {
      bestD = d;
      speed = ctx.vPvMps;
      ax = ctx.pvAx;
      ay = ctx.pvAy;
      az = ctx.pvAz;
    }
  }

  if (speed !== 0) {
    const w = Math.exp(-((bestD / (LEAFLET_REACH_CM * 0.65)) ** 2));
    const v = speed * w;
    out.vx = ax * v;
    out.vy = ay * v;
    out.vz = az * v;
  }
}

/**
 * Valve leaflet opening/closing snap velocity (m/s, heart frame) at point (x, y, z), driven by
 * the finite-difference time derivatives of `mvOpen`, `avOpen`, `tvOpen`, and `pvOpen`.
 */
export function sampleLeafletVelocity(
  heart: HeartModel,
  tables: BeatTables,
  hp: HeartPose,
  phase: number,
  x: number,
  y: number,
  z: number,
  flowPeaks?: FlowPeaks,
): { vx: number; vy: number; vz: number } {
  const ctx = prepareClutterContext(heart, tables, hp, phase, flowPeaks);
  const out = { vx: 0, vy: 0, vz: 0 };
  leafletVelocityFromCtx(ctx, x, y, z, out);
  return out;
}

const CLUTTER_VEC_TMP = { vx: 0, vy: 0, vz: 0 };

/**
 * Fast zero-allocation axial tissue/valve clutter velocity (+ toward transducer, m/s) for a polar
 * Color Doppler ROI sample, given a precomputed per-frame `ClutterContext`.
 */
export function sampleClutterAxialVelocity(
  ctx: ClutterContext,
  heart: HeartModel,
  x: number,
  y: number,
  z: number,
  tissue: number,
  structure: number,
  dhx: number,
  dhy: number,
  dhz: number,
): number {
  if (tissue === Tissue.Valve) {
    leafletVelocityFromCtx(ctx, x, y, z, CLUTTER_VEC_TMP);
    return -(
      CLUTTER_VEC_TMP.vx * dhx +
      CLUTTER_VEC_TMP.vy * dhy +
      CLUTTER_VEC_TMP.vz * dhz
    );
  }
  if (tissue !== Tissue.Myocardium) return 0;

  let vx = 0;
  let vy = 0;
  let vz = 0;

  if (structure === Structure.RaWall) {
    const free = Math.min(1, Math.max(0, (ctx.tvCx + 0.5 * ctx.tvR - x) / ctx.tvR));
    if (free > 0) {
      const above = Math.min(
        1,
        Math.max(0, (ctx.rvAnnulusZ - RA_VESTIBULE_CM - z) / (2 * ctx.raRz - RA_VESTIBULE_CM)),
      );
      const vRing = (ctx.rvLongVelCmps * (1 - (1 - RA_ROOF_DESCENT_SHARE) * above)) / 100;
      let vSeptal = 0;
      if (free < 1) {
        const level = Math.min(
          1,
          Math.max(
            0,
            (z - ctx.annulusZ - BASAL_ANNULAR_CUFF_CM) /
              Math.max(1, heart.lv.lengthCm - ctx.annulusZ - BASAL_ANNULAR_CUFF_CM),
          ),
        );
        const dxMv = x - ctx.mvCx;
        const dyMv = y - ctx.mvCy;
        const rho = Math.hypot(dxMv, dyMv);
        const lat = rho > 1e-6 ? 0.5 * (1 + dxMv / rho) : 0.5;
        vSeptal = ((ctx.lvLongVelCmps + ctx.lvIsoCmps) * (1 - level) * (1 + ctx.wallRatioDelta * lat)) / 100;
      }
      vz = free * vRing + (1 - free) * vSeptal;
      return -(vz * dhz);
    }
  }

  if (RV_TISSUE.has(structure)) {
    const rvLevel = Math.min(
      1,
      Math.max(
        0,
        (z - ctx.rvAnnulusZ - BASAL_ANNULAR_CUFF_CM) /
          Math.max(1, ctx.rvApexZ - ctx.rvAnnulusZ - BASAL_ANNULAR_CUFF_CM),
      ),
    );
    const lateral = Math.min(1, Math.max(0, (ctx.tvCx + ctx.tvR - x) / (2 * ctx.tvR)));
    vx = (2 * ctx.tvR * ctx.rvShortRate * lateral * (1 - rvLevel)) / 100;
    vz = (ctx.rvLongVelCmps * (1 - rvLevel)) / 100;
    return -(vx * dhx + vz * dhz);
  }

  const level = Math.min(
    1,
    Math.max(
      0,
      (z - ctx.annulusZ - BASAL_ANNULAR_CUFF_CM) /
        Math.max(1, heart.lv.lengthCm - ctx.annulusZ - BASAL_ANNULAR_CUFF_CM),
    ),
  );
  const dxMv = x - ctx.mvCx;
  const dyMv = y - ctx.mvCy;
  const rho = Math.hypot(dxMv, dyMv);
  const lat = rho > 1e-6 ? 0.5 * (1 + dxMv / rho) : 0.5;
  const wall = 1 + ctx.wallRatioDelta * lat;
  vz = ((ctx.lvLongVelCmps + ctx.lvIsoCmps) * (1 - level) * wall) / 100;

  const rhoLv = Math.hypot(x, y);
  if (
    z > ctx.annulusZ - 0.3 &&
    z < heart.lv.lengthCm + 1.2 &&
    rhoLv > 0.2 &&
    rhoLv < heart.lv.rMax * 2.5
  ) {
    const levelFrac = Math.min(
      1,
      Math.max(0, (z - ctx.annulusZ) / Math.max(1, heart.lv.lengthCm - ctx.annulusZ)),
    );
    const radialEnv = smoothstep(0, 0.18, levelFrac) * (1 - 0.55 * levelFrac * levelFrac);
    if (radialEnv > 1e-4) {
      const motion =
        heart.regionalMeanFrac > 0 ? lvRegionalMotion(heart.segAmp, Math.atan2(y, x), levelFrac) : 1;
      if (Math.abs(motion) > 1e-4) {
        const vRadMps = (ctx.lvRadialRateCmps * radialEnv * motion) / 100;
        vx = (x / rhoLv) * vRadMps;
        vy = (y / rhoLv) * vRadMps;
      }
    }
  }
  return -(vx * dhx + vy * dhy + vz * dhz);
}
