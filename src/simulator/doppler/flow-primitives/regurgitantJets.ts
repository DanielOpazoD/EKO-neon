import {
  heartAnchors,
  ROOT_EXCURSION,
  type HeartModel,
  type HeartPose,
} from '@/simulator/anatomy/heartModel';
import type { BeatTables } from '@/simulator/cardiac-cycle/cycleModel';
import { cycleStateAt, sampleTable } from '@/simulator/cardiac-cycle/cycleModel';
import { tvShortening } from '@/simulator/anatomy/valveSkirt';
import { smoothstep } from '@/core/vec3';
import { Structure } from '@/simulator/anatomy/tissue';
import { atrialScale, RA_ROOF_DESCENT_SHARE } from '@/simulator/anatomy/classify/atria';
import { PV_RADIUS, PV_REACH, pulmonaryVeinSegment } from '@/simulator/anatomy/pulmonaryVeins';
import type { FlowFieldParams } from './flowField';
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
  // S: during ventricular systole (annular descent pulls blood into the LA)
  const uS = (t - tm.ejectionStartS + 0.03) / (tm.ejectionEndS - tm.ejectionStartS + 0.08);
  const sWave = uS > 0 && uS < 1 ? Math.pow(Math.sin(Math.PI * uS), 0.9) : 0;
  // D: mirrors the mitral E wave (conduit), Ar: atrial contraction reversal
  const qmvMax = p.flowPeaks.mitral;
  const inE = t > tm.mitralOpenS && t < tm.mitralOpenS + tm.eAccelS + tm.eDecelS + 0.02;
  const dWave = inE ? sampleTable(tables.mitralFlowMlps, phase) / qmvMax : 0;
  const inA = tm.hasAWave && t > tm.aStartS && t < tm.aEndS + 0.03;
  const arWave = inA ? Math.sin((Math.PI * (t - tm.aStartS)) / (tm.aEndS + 0.03 - tm.aStartS)) : 0;
  const vAlong = pv.sMps * sWave + pv.dMps * dWave - pv.arMps * arWave; // + = into the LA
  if (Math.abs(vAlong) < 0.02) return;
  const rzL = (zBottom - zTop) / 2;
  const bo = atrialScale(hp.laBooster, pv.laReservoir, hp.state.contraction);
  for (let i = 0; i < 4; i++) {
    // the vein of the classifier (pulmonaryVeins.ts): the flow runs where the wall is drawn
    pulmonaryVeinSegment(i, la, lr, bo, czL, rzL, PV_SEG);
    const ox = PV_SEG[0]!,
      oy = PV_SEG[1]!,
      oz = PV_SEG[2]!;
    const ex = PV_SEG[3]!,
      ey = PV_SEG[4]!,
      ez = PV_SEG[5]!;
    // vein axis from the distal end toward the ostium, extended 0.8 cm into the LA
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
    if (along < 0 || along > L + 1.1) continue;
    const qx = dx - ux * along,
      qy = dy - uy * along,
      qz = dz - uz * along;
    const rho = Math.sqrt(qx * qx + qy * qy + qz * qz);
    const R = along <= L ? PV_RADIUS - 0.03 : PV_RADIUS - 0.03 + (along - L) * 0.6; // spreads into the atrium
    const sOut = Math.max(0, along - L);
    if (rho >= jetReach(0.84 * R, sOut)) continue;
    const prof = jetProfile(rho, 0.84 * R, sOut);
    const decay =
      (along <= L ? 1 : 1 / (1 + (along - L) / 0.6)) *
      smoothstep(0, 0.5, along) *
      fadeOut(along, L + 0.4, L + 1.1);
    const v = vAlong * prof * decay;
    out.vx += ux * v;
    out.vy += uy * v;
    out.vz += uz * v;
    out.dispersion = Math.max(out.dispersion, p.turbulence['pulmonary-vein'] ?? 0.06);
    out.present = 1;
  }
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
): void {
  const zAnn = hp.zAnn;
  // ---- MR jet ----
  if (p.enabled['mr-jet'] !== false && p.mrEroCm2 > 0) {
    const q = sampleTable(tables.mrFlowMlps, phase);
    if (q > 1) {
      const vj = Math.max(1e-3, sampleTable(tables.mrVelocityMps, phase));
      const r0 = Math.sqrt(Math.min(p.mrEroCm2, q / vj / 100) / Math.PI) * 1.1;
      const ox = p.mvCenter.x,
        oy = p.mvCenter.y,
        oz = zAnn - 0.1;
      const dirX = Math.sin(p.mrJetDirRad),
        dirZ = -Math.cos(p.mrJetDirRad);
      const dx = x - ox,
        dy = y - oy,
        dz = z - oz;
      const along = dx * dirX + dz * dirZ; // distance along the jet (into the LA)
      const px = dx - dirX * along,
        pz = dz - dirZ * along;
      const rho = Math.hypot(px, dy, pz);
      if (along >= 0 && along < 5.5) {
        const rHalf = 0.84 * (r0 + along * 0.3);
        if (rho < jetReach(rHalf, along, FAST_JET_SHEAR)) {
          const prof = jetProfile(rho, rHalf, along, FAST_JET_SHEAR);
          const decay = freeJetCentreline(along, (2 * r0) / 1.1) * fadeOut(along, 4, 5.5);
          const v = vj * prof * decay;
          out.vx += dirX * v;
          out.vz += dirZ * v;
          out.dispersion = Math.max(
            out.dispersion,
            (p.turbulence['mr-jet'] ?? 0.3) + 0.2 * Math.min(1, rho / (rHalf / 0.84)),
          );
          out.present = 1;
        }
      } else if (along < 0) {
        // PISA: hemispheric convergence on the LV side, v = Q / (2π r²), fading out where it used to stop
        const d = Math.hypot(dx, dy, dz);
        if (d < 1.8 && d > 0.05) {
          const v =
            Math.min(vj * 0.6, q / (2 * Math.PI * d * d) / 100) *
            fadeOut(d, 1.2, 1.8) *
            convergenceWeight(rho, -along, r0);
          if (v > 0.03) {
            out.vx += (-dx / d) * v;
            out.vy += (-dy / d) * v;
            out.vz += (-dz / d) * v;
            out.dispersion = Math.max(out.dispersion, 0.08);
            out.present = 1;
          }
        }
      }
    }
  }
  // ---- AR jet ----
  if (p.enabled['ar-jet'] !== false && p.arEroCm2 > 0) {
    const q = sampleTable(tables.arFlowMlps, phase);
    if (q > 1) {
      const vj = q / p.arEroCm2 / 100;
      const r0 = Math.sqrt(p.arEroCm2 / Math.PI) * 1.1;
      const ax = p.avAxis;
      const cz = p.avCenter.z + zAnn * ROOT_EXCURSION;
      const dx = x - p.avCenter.x,
        dy = y - p.avCenter.y,
        dz = z - cz;
      const t = dx * ax.x + dy * ax.y + dz * ax.z; // along the root axis (negative = into the LV)
      const qx = dx - ax.x * t,
        qy = dy - ax.y * t,
        qz = dz - ax.z * t;
      const rho = Math.sqrt(qx * qx + qy * qy + qz * qz);
      const along = -t;
      if (along >= 0 && along < 6) {
        const rHalf = 0.84 * (r0 + along * 0.35);
        if (rho < jetReach(rHalf, along, FAST_JET_SHEAR)) {
          const prof = jetProfile(rho, rHalf, along, FAST_JET_SHEAR);
          const decay = freeJetCentreline(along, (2 * r0) / 1.1) * fadeOut(along, 4.5, 6);
          const v = vj * prof * decay;
          out.vx -= ax.x * v;
          out.vy -= ax.y * v;
          out.vz -= ax.z * v;
          out.dispersion = Math.max(
            out.dispersion,
            (p.turbulence['ar-jet'] ?? 0.3) + 0.2 * Math.min(1, rho / (rHalf / 0.84)),
          );
          out.present = 1;
        }
      } else if (along < 0) {
        // convergence in the root, fading out where it used to stop
        const d = Math.hypot(dx, dy, dz);
        if (d > 0.05 && d < 1.5) {
          const v =
            Math.min(vj * 0.6, q / (2 * Math.PI * d * d) / 100) *
            fadeOut(d, 1.0, 1.5) *
            convergenceWeight(rho, -along, r0);
          if (v > 0.03) {
            out.vx += (-dx / d) * v;
            out.vy += (-dy / d) * v;
            out.vz += (-dz / d) * v;
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

/**
 * Tissue (myocardial) velocity for TDI (m/s, heart frame) at heart-frame point (x, y, z): the longitudinal annular motion of
 * the shared displacement table, scaled by level (base moves, apex fixed) and by wall (decision 98).
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
  if (structure === Structure.RaWall) {
    const A = heartAnchors(heart);
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
    const A = heartAnchors(heart);
    const apexZ = A.rvApexFrac * heart.lv.lengthCm;
    const annulusZ =
      A.tvCenter.z + heart.physiology.tapseCm * sampleTable(tables.rvLongitudinal, phase);
    const rvLevel = Math.min(1, Math.max(0, (z - annulusZ) / Math.max(1, apexZ - annulusZ)));
    const dPhase = 0.004;
    const shortening = (ph: number) =>
      tvShortening(cycleStateAt(tables, ph).contraction, A.rvRadialScale);
    const rate =
      (shortening((phase + dPhase) % 1) - shortening((phase - dPhase + 1) % 1)) /
      (2 * dPhase * tables.rrS);
    const lateral = Math.min(1, Math.max(0, (A.tvCenter.x + A.tvR - x) / (2 * A.tvR)));
    return {
      vx: (2 * A.tvR * rate * lateral * (1 - rvLevel)) / 100,
      vy: 0,
      vz:
        (heart.physiology.tapseCm *
          sampleTable(tables.rvLongitudinalVelocity, phase) *
          (1 - rvLevel)) /
        100,
    };
  }
  const longVel = sampleTable(tables.longitudinalVelocity, phase); // fraction of MAPSE per s
  const mapse = heart.physiology.mapseCm;
  const annulusZ = mapse * sampleTable(tables.longitudinal, phase);
  const level = Math.min(
    1,
    Math.max(0, (z - annulusZ) / Math.max(1, heart.lv.lengthCm - annulusZ)),
  );
  const rho = Math.hypot(x, y);
  const lateralness = rho > 1e-6 ? 0.5 * (1 + x / rho) : 0.5; // 1 at the lateral wall (+x), 0 at the septum (−x)
  const wall =
    1 + (heart.physiology.ePrimeLateralCmps / heart.physiology.ePrimeSeptalCmps - 1) * lateralness;
  const vz = (mapse * longVel * (1 - level) * wall) / 100; // cm/s → m/s, +z = toward the apex (systole)
  return { vx: 0, vy: 0, vz };
}
