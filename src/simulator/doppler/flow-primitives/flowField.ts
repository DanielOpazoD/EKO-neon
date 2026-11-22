import { sampleAorticNetwork } from './aorticNetwork';
import type { ThoraxModel } from '@/simulator/anatomy/thoraxModel';
import {
  caseLvotObstruction,
  caseOutflow,
  lvotNarrowing,
  solveLvotObstruction,
} from '@/simulator/cardiac-cycle/outflow';
import {
  heartAnchors,
  ROOT_EXCURSION,
  type HeartModel,
  type HeartPose,
} from '@/simulator/anatomy/heartModel';
import { measureFlowPeaks, type FlowPeaks } from '@/simulator/cardiac-cycle/flowPeaks';
import type { BeatTables } from '@/simulator/cardiac-cycle/cycleModel';
import { sampleTable } from '@/simulator/cardiac-cycle/cycleModel';
import type { CaseDefinition } from '@/cases/schema';
import { smoothstep } from '@/core/vec3';
import { lvotRadiusAt } from '@/simulator/anatomy/aorticValve';
import {
  convergenceWeight,
  conventionalPropagation,
  fadeOut,
  FAST_JET_SHEAR,
  flowPropagationCmps,
  freeJetCentreline,
  FUNNEL_FADE_TO_CM,
  inflowAxial,
  inflowJet,
  jetProfile,
  jetReach,
  LVOT_ENTRANCE_T,
  LVOT_OBSTRUCTION_SIGMA_CM,
  LVOT_OBSTRUCTION_T,
  LVOT_SINK_COS_FROM,
  LVOT_SINK_COS_TO,
  LVOT_SINK_FADE_FROM_CM,
  LVOT_SINK_FADE_TO_CM,
  pulmonaryVeinPeaks,
  SLOW_SHEAR,
  solveInflowWave,
  type FlowSample,
} from './jetProfiles';
import {
  RA_VESTIBULE_CM,
  samplePulmonaryVeins,
  sampleRegurgitantJets,
  sampleTissueVelocity,
} from './regurgitantJets';

export type { FlowSample };
export {
  conventionalPropagation,
  flowPropagationCmps,
  lvotNarrowing,
  pulmonaryVeinPeaks,
  RA_VESTIBULE_CM,
  samplePulmonaryVeins,
  sampleRegurgitantJets,
  sampleTissueVelocity,
  solveInflowWave,
  solveLvotObstruction,
};

/**
 * Parametric hemodynamic flow field (spec 0.5, 10.5, 63). Velocities in m/s, heart frame.
 * Every primitive is driven by the SAME beat tables that define volumes and ground truth, so
 * Color, PW, CW, TDI and measurements agree by construction. No CFD: jets are cores with
 * radial profiles, spreading and turbulence; regurgitant jets exist only when the case defines them.
 */
export interface FlowFieldParams {
  /** Snapshot belonging to the same immutable beat as the other derived flow parameters. */
  flowPeaks: FlowPeaks;
  heart: HeartModel;
  thorax: ThoraxModel;
  lvotAreaCm2: number;
  /** Annulus radius (cm): the outflow tract widens to it over its last LVOT_TAPER_CM (decision 161). */
  avR: number;
  avAreaCm2: number;
  mvAreaCm2: number;
  tvAreaCm2: number;
  rvotAreaCm2: number;
  trVmaxMps: number | null;
  turbulence: Record<string, number>;
  enabled: Record<string, boolean>;
  avAxis: { x: number; y: number; z: number };
  avCenter: { x: number; y: number; z: number };
  mvCenter: { x: number; y: number; z: number };
  mvR: number;
  tvCenter: { x: number; y: number; z: number };
  tvR: number;
  rvotA: { x: number; y: number; z: number };
  rvotB: { x: number; y: number; z: number };
  /** Regurgitant orifices (cm²) and the MR jet direction (rad, in the x–z plane; 0 = along the LV axis into the LA). */
  mrEroCm2: number;
  mrJetDirRad: number;
  arEroCm2: number;
  trEroCm2: number | null;
  /** Dynamic/subaortic LVOT obstruction: maximal area reduction fraction and its onset (fraction of ejection). */
  lvotObstruction: { fMax: number; dynamic: boolean } | null;
  /** Pulmonary venous flow: the four ostia (heart frame at ED) and the S/D/Ar peak velocities (m/s). */
  pulmonaryVeins: {
    laCenter: { x: number; y: number; z: number };
    laR: { x: number; y: number; z: number };
    /** Atrial reservoir scale (the veins ride on the atrial surface as it breathes). */
    laReservoir: number;
    sMps: number;
    dMps: number;
    arMps: number;
  };
  /** Colour M-mode flow propagation velocity (cm/s) the conventional slope reads, and the filling-wave speed that gives it. */
  inflowPropagationCmps: number;
  inflowWaveCmps: number;
}

export function buildFlowParams(
  c: CaseDefinition,
  heart: HeartModel,
  tables: BeatTables,
  thorax: ThoraxModel,
): FlowFieldParams {
  const A = heartAnchors(heart);
  const turbulence: Record<string, number> = {};
  const enabled: Record<string, boolean> = {};
  for (const fp of c.flowPrimitives) {
    turbulence[fp.site] = fp.turbulence;
    enabled[fp.site] = fp.enabled;
  }
  const tr = c.hemodynamics.trPresent ? tables.regurgitation.trVmaxMps : null;
  return {
    flowPeaks: measureFlowPeaks(tables),
    heart,
    thorax,
    lvotAreaCm2: caseOutflow(c).lvotAreaCm2,
    avR: A.avR,
    avAreaCm2: c.hemodynamics.avEffectiveAreaCm2,
    mvAreaCm2: tables.mvEffectiveAreaCm2,
    tvAreaCm2: tables.mvEffectiveAreaCm2 * 1.35,
    rvotAreaCm2: Math.PI * 1.1 * 1.1,
    trVmaxMps: tr,
    turbulence,
    enabled,
    avAxis: A.avAxis,
    avCenter: A.avCenter,
    mvCenter: A.mvCenter,
    mvR: A.mvR,
    tvCenter: A.tvCenter,
    tvR: A.tvR,
    rvotA: A.rvotA,
    rvotB: A.rvotB,
    mrEroCm2: c.hemodynamics.regurgitation.mr?.eroaCm2 ?? 0,
    mrJetDirRad: ((c.hemodynamics.regurgitation.mr?.jetDirectionDeg ?? 0) * Math.PI) / 180,
    arEroCm2: c.hemodynamics.regurgitation.ar?.eroaCm2 ?? 0,
    trEroCm2: c.hemodynamics.regurgitation.tr?.eroaCm2 ?? null,
    lvotObstruction: caseLvotObstruction(c, tables),
    pulmonaryVeins: {
      laCenter: A.laCenter,
      laR: A.laR,
      laReservoir: A.laReservoir,
      ...pulmonaryVeinPeaks(c),
    },
    inflowPropagationCmps: flowPropagationCmps(c),
    inflowWaveCmps: solveInflowWave(
      tables,
      c.physiology.mapseCm,
      heart.lv.lengthCm,
      tables.mvEffectiveAreaCm2,
      flowPropagationCmps(c),
    ),
  };
}

/** Sample the flow field at heart-frame point (x,y,z) for the given phase. */
export function sampleFlow(
  p: FlowFieldParams,
  tables: BeatTables,
  hp: HeartPose,
  phase: number,
  x: number,
  y: number,
  z: number,
  out: FlowSample,
): void {
  out.vx = 0;
  out.vy = 0;
  out.vz = 0;
  out.dispersion = 0;
  out.present = 0;
  if (sampleAorticNetwork(p.heart, p.thorax, tables, hp, phase, x, y, z, out)) {
    if (p.enabled['lvot'] === false && p.enabled['aortic-valve'] === false) {
      out.vx = out.vy = out.vz = out.dispersion = out.present = 0;
    }
    return;
  }
  const qao = sampleTable(tables.aorticFlowMlps, phase);
  const zAnn = hp.zAnn;

  // ---- Mitral inflow: from the LA through the annulus into the LV toward the apex ----
  const zrMv = z - zAnn; // distance beyond the annulus into the LV
  if (
    p.enabled['mitral-inflow'] !== false &&
    zrMv > -FUNNEL_FADE_TO_CM &&
    zrMv < hp.lengthNow - 0.2
  )
    inflowJet(
      tables,
      tables.mitralFlowMlps,
      phase,
      p.inflowWaveCmps,
      x - p.mvCenter.x,
      y - p.mvCenter.y,
      zrMv,
      p.mvR * 1.05,
      p.mvAreaCm2,
      (zr) => inflowAxial(zr, hp.lengthNow),
      p.turbulence['mitral-inflow'] ?? 0.04,
      out,
    );

  // ---- LVOT → aortic valve → ascending aorta ----
  if ((p.enabled['lvot'] !== false || p.enabled['aortic-valve'] !== false) && qao > 1) {
    const ax = p.avAxis;
    const cz = p.avCenter.z + zAnn * ROOT_EXCURSION;
    const dx = x - p.avCenter.x,
      dy = y - p.avCenter.y,
      dz = z - cz;
    const t = dx * ax.x + dy * ax.y + dz * ax.z;
    const qx = dx - ax.x * t,
      qy = dy - ax.y * t,
      qz = dz - ax.z * t;
    const rho = Math.sqrt(qx * qx + qy * qy + qz * qz);
    const rLvot = Math.sqrt(p.lvotAreaCm2 / Math.PI);
    const tmE = tables.timings;
    const uEject =
      (phase * tables.rrS - tmE.ejectionStartS) / (tmE.ejectionEndS - tmE.ejectionStartS);
    const narrowPeak = p.lvotObstruction
      ? lvotNarrowing(p.lvotObstruction.fMax, p.lvotObstruction.dynamic, uEject)
      : 0;
    const narrowAt = (tt: number): number =>
      narrowPeak *
      Math.exp(-((tt - LVOT_OBSTRUCTION_T) ** 2) / (2 * LVOT_OBSTRUCTION_SIGMA_CM ** 2));
    if (t < LVOT_ENTRANCE_T) {
      const up = LVOT_ENTRANCE_T - t;
      const d = Math.hypot(rho, up);
      if (d < LVOT_SINK_FADE_TO_CM) {
        const vEntrance = qao / (p.lvotAreaCm2 * (1 - narrowAt(LVOT_ENTRANCE_T))) / 100;
        const v =
          Math.min(vEntrance, qao / (2 * Math.PI * d * d) / 100) *
          fadeOut(d, LVOT_SINK_FADE_FROM_CM, LVOT_SINK_FADE_TO_CM) *
          convergenceWeight(rho, up, rLvot, LVOT_SINK_COS_FROM, LVOT_SINK_COS_TO);
        if (v > 0) {
          out.vx += ((-qx + ax.x * up) / d) * v;
          out.vy += ((-qy + ax.y * up) / d) * v;
          out.vz += ((-qz + ax.z * up) / d) * v;
          out.dispersion = Math.max(out.dispersion, p.turbulence['lvot'] ?? 0.04);
          out.present = 1;
        }
      }
    } else if (t < 5.5) {
      let area: number;
      let R: number;
      if (t < 0) {
        const widen = lvotRadiusAt(p.avR, rLvot, t) / rLvot;
        const narrow = narrowAt(t);
        R = rLvot * widen * Math.sqrt(1 - narrow);
        area = p.lvotAreaCm2 * widen * widen * (1 - narrow);
      } else if (t < 1.0) {
        R = Math.sqrt(p.avAreaCm2 / Math.PI) * 1.15;
        area = p.avAreaCm2;
      } else {
        const spread = 1 + (t - 1.0) * 0.45;
        R = Math.sqrt(p.avAreaCm2 / Math.PI) * 1.15 * spread;
        area = p.avAreaCm2 * spread * spread;
      }
      const rHalf = 0.917 * R;
      const sShear = Math.max(0, t - 1);
      const shear = p.avAreaCm2 < 2.0 ? FAST_JET_SHEAR : SLOW_SHEAR;
      let vJet = 0;
      if (narrowPeak > 0 && t > LVOT_OBSTRUCTION_T) {
        const aEff = p.lvotAreaCm2 * (1 - narrowPeak);
        const rEff = Math.sqrt(aEff / Math.PI);
        const along = t - LVOT_OBSTRUCTION_T;
        const rHalfJet = 0.917 * rEff * (1 + along * 0.45);
        if (rho < jetReach(rHalfJet, along, FAST_JET_SHEAR))
          vJet =
            (qao / aEff / 100) *
            freeJetCentreline(along, 2 * rEff) *
            jetProfile(rho, rHalfJet, along, FAST_JET_SHEAR) *
            fadeOut(t, 4, 5.5);
      }
      const reach = rho < jetReach(rHalf, sShear, shear);
      if (reach || vJet > 0) {
        const v = qao / area / 100;
        const vTract = reach ? v * jetProfile(rho, rHalf, sShear, shear) * fadeOut(t, 4, 5.5) : 0;
        const vOut = Math.max(vTract, vJet);
        out.vx += ax.x * vOut;
        out.vy += ax.y * vOut;
        out.vz += ax.z * vOut;
        const stenotic =
          p.avAreaCm2 < 2.0 && t > 0
            ? Math.min(0.6, (2.0 - p.avAreaCm2) * 0.5)
            : p.lvotObstruction && t < 0
              ? 0.4 * (1 - area / p.lvotAreaCm2)
              : 0;
        out.dispersion = Math.max(
          out.dispersion,
          (p.turbulence['lvot'] ?? 0.04) +
            stenotic * (t > 0 ? 1 : 0.3) +
            0.1 * Math.pow(Math.min(1, rho / R), 4),
          t > 0 ? (p.turbulence['aortic-valve'] ?? 0) : 0,
        );
        out.present = 1;
      }
    }
  }

  // ---- Tricuspid inflow ----
  const zrTv = z - (p.tvCenter.z + hp.tvZ);
  if (p.enabled['tricuspid-inflow'] !== false && zrTv > -FUNNEL_FADE_TO_CM && zrTv < 5)
    inflowJet(
      tables,
      tables.tricuspidFlowMlps,
      phase,
      p.inflowWaveCmps,
      x - p.tvCenter.x,
      y - p.tvCenter.y,
      zrTv,
      p.tvR * 1.05,
      p.tvAreaCm2,
      (zr) => (zr < 1.2 ? 1 : 1 / (1 + (zr - 1.2) / 2.2)) * fadeOut(zr, 4, 5),
      p.turbulence['tricuspid-inflow'] ?? 0.04,
      out,
    );

  // ---- RVOT / pulmonary ----
  if (p.enabled['rvot'] !== false) {
    const qpv = sampleTable(tables.pulmonaryFlowMlps, phase);
    if (qpv > 1) {
      const ax = p.rvotB.x - p.rvotA.x,
        ay = p.rvotB.y - p.rvotA.y,
        az = p.rvotB.z - p.rvotA.z;
      const L = Math.hypot(ax, ay, az);
      const ux = ax / L,
        uy = ay / L,
        uz = az / L;
      const dx = x - p.rvotA.x,
        dy = y - p.rvotA.y,
        dz = z - hp.pvZ - p.rvotA.z;
      const t = dx * ux + dy * uy + dz * uz;
      if (t > -1 && t < L + 1) {
        const qx = dx - ux * t,
          qy = dy - uy * t,
          qz = dz - uz * t;
        const rho = Math.sqrt(qx * qx + qy * qy + qz * qz);
        if (rho < jetReach(0.89 * 1.05, 0)) {
          const v = qpv / p.rvotAreaCm2 / 100;
          const prof =
            jetProfile(rho, 0.89 * 1.05, 0) * smoothstep(-1, 0, t) * fadeOut(t, L, L + 1);
          out.vx += ux * v * prof;
          out.vy += uy * v * prof;
          out.vz += uz * v * prof;
          out.dispersion = Math.max(out.dispersion, p.turbulence['rvot'] ?? 0.04);
          out.present = 1;
        }
      }
    }
  }

  // ---- TR jet (systole) ----
  if (p.enabled['tr-jet'] !== false && p.trVmaxMps) {
    const vmax = sampleTable(tables.trVelocityMps, phase);
    if (vmax > 0.02 * p.trVmaxMps) {
      const ox = p.tvCenter.x,
        oy = p.tvCenter.y,
        oz = p.tvCenter.z + hp.tvZ + 0.3;
      const dx = x - ox,
        dy = y - oy,
        dzj = oz - z;
      const rho = Math.hypot(dx, dy);
      if (dzj > -1.4 && dzj < 4.5) {
        const r0 = p.trEroCm2 ? Math.sqrt(p.trEroCm2 / Math.PI) * 1.1 : 0.25;
        const Rj = r0 + Math.max(0, dzj) * 0.28;
        const rHalf = 0.84 * Rj;
        if (dzj >= 0 && rho < jetReach(rHalf, dzj, FAST_JET_SHEAR)) {
          const prof = jetProfile(rho, rHalf, dzj, FAST_JET_SHEAR);
          const decay = freeJetCentreline(dzj, (2 * r0) / 1.1) * fadeOut(dzj, 3.5, 4.5);
          out.vz -= vmax * prof * decay;
          out.dispersion = Math.max(
            out.dispersion,
            (p.turbulence['tr-jet'] ?? 0.25) + 0.2 * Math.min(1, rho / Rj),
          );
          out.present = 1;
        } else if (dzj < 0) {
          const d = Math.hypot(rho, -dzj) + 0.2;
          if (d < 1.4) {
            const conv =
              Math.min(vmax * 0.5, (vmax * r0 * r0) / (2 * d * d)) *
              fadeOut(d, 0.8, 1.4) *
              convergenceWeight(rho, -dzj, r0);
            out.vz -= conv;
            out.present = 1;
            out.dispersion = Math.max(out.dispersion, 0.1);
          }
        }
      }
    }
  }
  sampleRegurgitantJets(p, tables, hp, phase, x, y, z, out);
  samplePulmonaryVeins(p, tables, hp, phase, x, y, z, out);
}
