import { sampleAorticNetwork } from './aorticNetwork';
import {
  pulmonaryRegurgitationVelocityMps,
  samplePulmonaryRegurgitation,
  samplePulmonarySystolicFlow,
} from './pulmonaryFlow';
import {
  cavalVenousVelocityMps,
  coronarySinusVelocityMps,
  sampleIvcHepaticFlow,
  sampleSystemicVenousFlow,
} from './systemicVeins';
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
import { Structure } from '@/simulator/anatomy/tissue';
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
  pulmonaryVeinPeaks,
  SLOW_SHEAR,
  solveInflowWave,
  type FlowSample,
} from './jetProfiles';
import {
  mrJetInitialDirection,
  prepareClutterContext,
  RA_VESTIBULE_CM,
  sampleClutterAxialVelocity,
  sampleLeafletVelocity,
  samplePulmonaryVeins,
  sampleRegurgitantJets,
  sampleTissueVelocity,
  type ClutterContext,
} from './regurgitantJets';

export type { ClutterContext, FlowSample };
export {
  cavalVenousVelocityMps,
  conventionalPropagation,
  coronarySinusVelocityMps,
  flowPropagationCmps,
  lvotNarrowing,
  mrJetInitialDirection,
  prepareClutterContext,
  pulmonaryRegurgitationVelocityMps,
  pulmonaryVeinPeaks,
  RA_VESTIBULE_CM,
  sampleClutterAxialVelocity,
  sampleLeafletVelocity,
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
  paspMmHg: number;
  rapMmHg: number;
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
    tvAreaCm2: tables.tvEffectiveAreaCm2,
    rvotAreaCm2: Math.PI * 1.1 * 1.1,
    trVmaxMps: tr,
    paspMmHg: c.hemodynamics.paspMmHg,
    rapMmHg: c.hemodynamics.rapMmHg,
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

/** Bitmask flags governing which hemodynamic flow primitives may occupy each anatomical blood structure. */
export const FLOW_MASK_AORTA_NET = 1 << 0;
export const FLOW_MASK_LVOT_AV = 1 << 1;
export const FLOW_MASK_MITRAL_IN = 1 << 2;
export const FLOW_MASK_MR_JET = 1 << 3;
export const FLOW_MASK_MR_PISA = 1 << 4;
export const FLOW_MASK_AR_JET = 1 << 5;
export const FLOW_MASK_AR_PISA = 1 << 6;
export const FLOW_MASK_PV = 1 << 7;
export const FLOW_MASK_TRICUSPID_IN = 1 << 8;
export const FLOW_MASK_RVOT_PA = 1 << 9;
export const FLOW_MASK_TR_JET = 1 << 10;
export const FLOW_MASK_TR_PISA = 1 << 11;
export const FLOW_MASK_SYS_VEINS = 1 << 12;
export const FLOW_MASK_ALL = 0xffff;

const STRUCTURE_FLOW_MASK = new Uint16Array(64).fill(FLOW_MASK_ALL);
// Left ventricle & outflow
STRUCTURE_FLOW_MASK[Structure.LvCavity] =
  FLOW_MASK_MITRAL_IN | FLOW_MASK_LVOT_AV | FLOW_MASK_MR_PISA | FLOW_MASK_AR_JET;
STRUCTURE_FLOW_MASK[Structure.Lvot] =
  FLOW_MASK_MITRAL_IN | FLOW_MASK_LVOT_AV | FLOW_MASK_MR_PISA | FLOW_MASK_AR_JET;
// Aortic root & arterial tree
STRUCTURE_FLOW_MASK[Structure.AorticRoot] =
  FLOW_MASK_AORTA_NET | FLOW_MASK_LVOT_AV | FLOW_MASK_AR_PISA;
STRUCTURE_FLOW_MASK[Structure.AscendingAorta] =
  FLOW_MASK_AORTA_NET | FLOW_MASK_LVOT_AV | FLOW_MASK_AR_PISA;
STRUCTURE_FLOW_MASK[Structure.AorticArch] = FLOW_MASK_AORTA_NET;
STRUCTURE_FLOW_MASK[Structure.BrachiocephalicArtery] = FLOW_MASK_AORTA_NET;
STRUCTURE_FLOW_MASK[Structure.LeftCommonCarotid] = FLOW_MASK_AORTA_NET;
STRUCTURE_FLOW_MASK[Structure.LeftSubclavian] = FLOW_MASK_AORTA_NET;
STRUCTURE_FLOW_MASK[Structure.DescendingAorta] = FLOW_MASK_AORTA_NET;
// Left atrium & pulmonary veins
STRUCTURE_FLOW_MASK[Structure.LaCavity] = FLOW_MASK_MITRAL_IN | FLOW_MASK_MR_JET | FLOW_MASK_PV;
STRUCTURE_FLOW_MASK[Structure.LaAppendage] = FLOW_MASK_MITRAL_IN | FLOW_MASK_MR_JET | FLOW_MASK_PV;
STRUCTURE_FLOW_MASK[Structure.PulmonaryVein] = FLOW_MASK_PV;
// Right ventricle, outflow & pulmonary artery
STRUCTURE_FLOW_MASK[Structure.RvCavity] =
  FLOW_MASK_TRICUSPID_IN | FLOW_MASK_RVOT_PA | FLOW_MASK_TR_PISA;
STRUCTURE_FLOW_MASK[Structure.Rvot] =
  FLOW_MASK_TRICUSPID_IN | FLOW_MASK_RVOT_PA | FLOW_MASK_TR_PISA;
STRUCTURE_FLOW_MASK[Structure.PulmonaryArtery] = FLOW_MASK_RVOT_PA;
// Right atrium & systemic veins
STRUCTURE_FLOW_MASK[Structure.RaCavity] =
  FLOW_MASK_TRICUSPID_IN | FLOW_MASK_TR_JET | FLOW_MASK_SYS_VEINS;
STRUCTURE_FLOW_MASK[Structure.RaAppendage] =
  FLOW_MASK_TRICUSPID_IN | FLOW_MASK_TR_JET | FLOW_MASK_SYS_VEINS;
STRUCTURE_FLOW_MASK[Structure.Svc] = FLOW_MASK_SYS_VEINS;
STRUCTURE_FLOW_MASK[Structure.Ivc] = FLOW_MASK_SYS_VEINS;
STRUCTURE_FLOW_MASK[Structure.HepaticVein] = FLOW_MASK_SYS_VEINS;
STRUCTURE_FLOW_MASK[Structure.CoronarySinus] = FLOW_MASK_SYS_VEINS;

/** Return the allowed flow-primitive mask for an anatomical structure (non-blood/unknown -> all). */
export function flowMaskForStructure(structure?: number): number {
  if (structure === undefined || structure < 0 || structure >= STRUCTURE_FLOW_MASK.length) {
    return FLOW_MASK_ALL;
  }
  return STRUCTURE_FLOW_MASK[structure]!;
}

/** Reusable scratch sample for aortic-network blending (zero per-sample allocation). */
const AORTA_TMP: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };

/** Sample the flow field at heart-frame point (x,y,z) for the given phase, gated to `structure` when known. */
export function sampleFlow(
  p: FlowFieldParams,
  tables: BeatTables,
  hp: HeartPose,
  phase: number,
  x: number,
  y: number,
  z: number,
  out: FlowSample,
  structure?: number,
): void {
  out.vx = 0;
  out.vy = 0;
  out.vz = 0;
  out.dispersion = 0;
  out.present = 0;
  const mask = flowMaskForStructure(structure);
  const aortaEnabled = p.enabled['lvot'] !== false || p.enabled['aortic-valve'] !== false;
  const allowAortaNet = (mask & FLOW_MASK_AORTA_NET) !== 0;
  const allowLvotAv = (mask & FLOW_MASK_LVOT_AV) !== 0;
  const inAorta =
    allowAortaNet && sampleAorticNetwork(p.heart, p.thorax, tables, hp, phase, x, y, z, AORTA_TMP);
  if (inAorta && !aortaEnabled) return;

  const qao = sampleTable(tables.aorticFlowMlps, phase);
  const zAnn = hp.zAnn;

  // ---- LVOT → aortic valve → ascending aorta (unified with aortic network across the STJ) ----
  const ax = p.avAxis;
  const cz = p.avCenter.z + zAnn * ROOT_EXCURSION;
  const dxAv = x - p.avCenter.x,
    dyAv = y - p.avCenter.y,
    dzAv = z - cz;
  const tAv = dxAv * ax.x + dyAv * ax.y + dzAv * ax.z;
  const hasStenoticJet = p.avAreaCm2 < 2.0 || p.lvotObstruction !== null;

  // Far downstream in the arch / descending aorta / branches: pure mass-conserving aortic network.
  if (inAorta && (!allowLvotAv || tAv >= (hasStenoticJet ? 5.5 : 3.2) || qao <= 1)) {
    out.vx = AORTA_TMP.vx;
    out.vy = AORTA_TMP.vy;
    out.vz = AORTA_TMP.vz;
    out.dispersion = AORTA_TMP.dispersion;
    out.present = AORTA_TMP.present;
    return;
  }

  // ---- Mitral inflow: from the LA through the annulus into the LV toward the apex ----
  const zrMv = z - zAnn; // distance beyond the annulus into the LV
  if (
    !inAorta &&
    (mask & FLOW_MASK_MITRAL_IN) !== 0 &&
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

  // ---- LVOT → aortic valve → proximal ascending aorta ----
  if (aortaEnabled && allowLvotAv && qao > 1) {
    const qx = dxAv - ax.x * tAv,
      qy = dyAv - ax.y * tAv,
      qz = dzAv - ax.z * tAv;
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

    let rootVx = 0,
      rootVy = 0,
      rootVz = 0,
      rootDisp = 0,
      rootSpeed = 0,
      rootPresent = 0;

    if (tAv < LVOT_ENTRANCE_T) {
      // Smooth anatomically tapered LV → LVOT laminar acceleration corridor: fades naturally into the
      // basal LV without a flat horizontal cutoff in A5C/A3C or a bright blue dome over the MV in A4C.
      const up = LVOT_ENTRANCE_T - tAv;
      if (up < 1.9) {
        const taper = 1 + 0.3 * up;
        const rUp = rLvot * taper;
        const rHalfUp = 0.78 * rUp;
        if (rho < jetReach(rHalfUp, up * 0.4, SLOW_SHEAR)) {
          const vEntrance = qao / (p.lvotAreaCm2 * (1 - narrowAt(LVOT_ENTRANCE_T))) / 100;
          const rNorm = Math.min(1.4, rho / Math.max(0.25, rUp));
          const radialCore = 1 / (1 + 0.72 * rNorm * rNorm);
          const v =
            (vEntrance / (taper * taper)) *
            radialCore *
            jetProfile(rho, rHalfUp, up * 0.4, SLOW_SHEAR) *
            fadeOut(up, 0.55, 1.9);
          if (v > 0.01) {
            // Streamline-following gentle inward tilt toward the LVOT axis (never reverses sign in PLAX)
            const inward = rho > 1e-4 ? (0.32 * Math.min(1, rho / rUp)) / taper : 0;
            const invRho = rho > 1e-4 ? 1 / rho : 0;
            const dirX = ax.x - qx * invRho * inward;
            const dirY = ax.y - qy * invRho * inward;
            const dirZ = ax.z - qz * invRho * inward;
            const dirL = Math.hypot(dirX, dirY, dirZ) || 1;
            rootVx = (dirX / dirL) * v;
            rootVy = (dirY / dirL) * v;
            rootVz = (dirZ / dirL) * v;
            rootSpeed = v;
            rootDisp = p.turbulence['lvot'] ?? 0.04;
            rootPresent = 1;
          }
        }
      }
    } else if (tAv < 5.5) {
      const isStenotic = p.avAreaCm2 < 2.0;
      const areaAsc = Math.PI * (p.avR * 1.28) * (p.avR * 1.28);
      let area: number;
      let R: number;
      if (tAv < 0.2) {
        const widen = lvotRadiusAt(p.avR, rLvot, Math.min(0, tAv)) / rLvot;
        const narrow = narrowAt(tAv);
        const lvotRNow = rLvot * widen * Math.sqrt(1 - narrow);
        const lvotAreaNow = p.lvotAreaCm2 * widen * widen * (1 - narrow);
        const avOrifR = Math.sqrt(p.avAreaCm2 / Math.PI) * 1.15;
        // Exact lvotAreaCm2 at t = -0.5 cm (PW LVOT VTI plane) and exact avAreaCm2 at t >= 0.2 cm (CW/PW AV peak)
        const uValve = isStenotic
          ? smoothstep(-0.18, 0.12, tAv)
          : smoothstep(-0.35, 0.2, tAv);
        area = lvotAreaNow + (p.avAreaCm2 - lvotAreaNow) * uValve;
        R = lvotRNow + (avOrifR - lvotRNow) * uValve;
      } else if (!isStenotic) {
        // Normal aortic valve & sinuses of Valsalva: exact avAreaCm2 through the leaflet tips (t = 0.2..0.70 cm),
        // then smooth deceleration into the wider aortic sinuses and ascending lumen.
        const avOrifR = Math.sqrt(p.avAreaCm2 / Math.PI) * 1.15;
        const rAsc = p.avR * 1.28;
        const uRec = smoothstep(0.7, 2.05, tAv);
        area = p.avAreaCm2 + (areaAsc - p.avAreaCm2) * uRec;
        R = avOrifR + (rAsc - avOrifR) * uRec;
      } else {
        // Stenotic post-valvular turbulent jet: keeps exact vena-contracta avAreaCm2 at t = 0.2..0.95 cm
        // (preserving CW Vmax) while its entraining turbulent plume spreads additively across the sinuses so
        // both PLAX and slightly oblique A5C slices show a rich post-stenotic mosaic jet.
        const avOrifR = Math.sqrt(p.avAreaCm2 / Math.PI) * 1.15;
        const alongSpread = Math.max(0, tAv - 0.35);
        R = Math.min(p.avR * 1.22, avOrifR + alongSpread * 0.44);
        const alongDecay = Math.max(0, tAv - 0.95);
        area = p.avAreaCm2 * (1 + alongDecay * 0.22) * (1 + alongDecay * 0.22);
      }
      const rHalf = (isStenotic && tAv > 0 ? 0.88 : !p.lvotObstruction ? 0.8 : 0.86) * R;
      const sShear = Math.max(0, tAv - (isStenotic ? 0.35 : 0.7));
      const shear = isStenotic ? FAST_JET_SHEAR : SLOW_SHEAR;
      let vJet = 0;
      if (narrowPeak > 0 && tAv > LVOT_OBSTRUCTION_T) {
        const aEff = p.lvotAreaCm2 * (1 - narrowPeak);
        const rEff = Math.sqrt(aEff / Math.PI);
        const along = tAv - LVOT_OBSTRUCTION_T;
        const rHalfJet = 0.86 * rEff * (1 + along * 0.32);
        if (rho < jetReach(rHalfJet, along, FAST_JET_SHEAR))
          vJet =
            (qao / aEff / 100) *
            freeJetCentreline(along, 2 * rEff) *
            jetProfile(rho, rHalfJet, along, FAST_JET_SHEAR) *
            fadeOut(tAv, 3.8, 5.4);
      }
      const reach = rho < jetReach(rHalf, sShear, shear);
      if (reach || vJet > 0) {
        const v = qao / area / 100;
        // Physiologic radial shear for laminar LVOT/AV (= 1.00 on centerline rho = 0, so PW/CW Vmax is exact)
        const rNorm = Math.min(1.4, rho / Math.max(0.25, R));
        const radialCore =
          isStenotic && tAv > 0
            ? 1
            : 1 / (1 + (p.lvotObstruction ? 0.45 : 0.78) * rNorm * rNorm);
        const vTract = reach
          ? v * radialCore * jetProfile(rho, rHalf, sShear, shear) * fadeOut(tAv, 3.8, 5.4)
          : 0;
        const vOut = Math.max(vTract, vJet);
        rootVx = ax.x * vOut;
        rootVy = ax.y * vOut;
        rootVz = ax.z * vOut;
        rootSpeed = vOut;
        const stenotic =
          isStenotic && tAv > 0
            ? Math.min(0.65, (2.0 - p.avAreaCm2) * 0.52)
            : p.lvotObstruction && tAv > LVOT_OBSTRUCTION_T - 0.4
              ? Math.max(
                  0.4 * Math.max(0, 1 - area / p.lvotAreaCm2),
                  vJet > 1.5 ? Math.min(0.55, (vJet - 1.2) * 0.16) : 0,
                )
              : 0;
        rootDisp = Math.max(
          (p.turbulence['lvot'] ?? 0.04) +
            stenotic * (tAv > 0 ? 1 : 0.35) +
            (isStenotic ? 0.12 : 0.04) * Math.pow(Math.min(1, rho / Math.max(0.25, R)), 4),
          tAv > 0 ? (p.turbulence['aortic-valve'] ?? 0) : 0,
        );
        rootPresent = 1;
      }
    }

    if (inAorta) {
      if (hasStenoticJet && rootPresent && (rootSpeed > 1.1 || rootDisp > 0.32)) {
        // Post-stenotic / SAM turbulent jet rides over the laminar ascending-aorta network instead of getting clipped at STJ
        const jetWeight =
          Math.max(smoothstep(0.9, 1.8, rootSpeed), smoothstep(0.28, 0.5, rootDisp)) *
          fadeOut(tAv, 3.6, 5.2);
        if (jetWeight > 0.01) {
          out.vx = AORTA_TMP.vx * (1 - jetWeight) + rootVx * jetWeight;
          out.vy = AORTA_TMP.vy * (1 - jetWeight) + rootVy * jetWeight;
          out.vz = AORTA_TMP.vz * (1 - jetWeight) + rootVz * jetWeight;
          out.dispersion = Math.max(AORTA_TMP.dispersion, rootDisp * jetWeight);
          out.present = 1;
          return;
        }
      }
      // Smooth cross-fade across the sinotubular junction (t = 1.5..2.8 cm): eliminates both the A5C velocity cliff
      // and the PLAX straight diagonal knife-edge cutoff.
      const wArch = rootPresent ? smoothstep(1.5, 2.8, tAv) : 1;
      out.vx = rootVx * (1 - wArch) + AORTA_TMP.vx * wArch;
      out.vy = rootVy * (1 - wArch) + AORTA_TMP.vy * wArch;
      out.vz = rootVz * (1 - wArch) + AORTA_TMP.vz * wArch;
      out.dispersion = Math.max(rootDisp * (1 - wArch), AORTA_TMP.dispersion);
      out.present = 1;
      return;
    }

    if (rootPresent) {
      out.vx += rootVx;
      out.vy += rootVy;
      out.vz += rootVz;
      out.dispersion = Math.max(out.dispersion, rootDisp);
      out.present = 1;
    }
  } else if (inAorta) {
    out.vx = AORTA_TMP.vx;
    out.vy = AORTA_TMP.vy;
    out.vz = AORTA_TMP.vz;
    out.dispersion = AORTA_TMP.dispersion;
    out.present = AORTA_TMP.present;
    return;
  }

  // ---- Tricuspid inflow ----
  const zrTv = z - (p.tvCenter.z + hp.tvZ);
  if (
    (mask & FLOW_MASK_TRICUSPID_IN) !== 0 &&
    p.enabled['tricuspid-inflow'] !== false &&
    zrTv > -FUNNEL_FADE_TO_CM &&
    zrTv < 5
  )
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

  // ---- RVOT / pulmonary trunk & branches (systolic ejection + diastolic PR) ----
  if ((mask & FLOW_MASK_RVOT_PA) !== 0 && p.enabled['rvot'] !== false) {
    samplePulmonarySystolicFlow(p, tables, hp, phase, x, y, z, out);
    samplePulmonaryRegurgitation(p, tables, hp, phase, x, y, z, out, structure);
  }

  // ---- TR jet (systole: vena contracta neck + septal-curving RA jet + true 3D hemispheric RV PISA) ----
  const allowTrJet = (mask & FLOW_MASK_TR_JET) !== 0;
  const allowTrPisa = (mask & FLOW_MASK_TR_PISA) !== 0;
  if ((allowTrJet || allowTrPisa) && p.enabled['tr-jet'] !== false && p.trVmaxMps) {
    const vmax = sampleTable(tables.trVelocityMps, phase);
    if (vmax > 0.02 * p.trVmaxMps) {
      const ox = p.tvCenter.x,
        oy = p.tvCenter.y,
        oz = p.tvCenter.z + hp.tvZ + 0.3;
      const dx = x - ox,
        dy = y - oy,
        dz = z - oz;
      const dzj = -dz; // + into RA, − into RV
      // Physiologic trivial TR (normal hearts, no regurgitant EROA and low PASP/Vmax) stays compact right behind
      // coaptation so it never paints false mid-RA pathology while keeping full CW peak velocity at the orifice.
      const isTraceTr = p.trEroCm2 === null && p.trVmaxMps < 2.6;
      const rVc = p.trEroCm2
        ? Math.sqrt(p.trEroCm2 / Math.PI)
        : isTraceTr
          ? 0.09
          : 0.22;
      const r0 = rVc * 1.1;
      const maxReach = isTraceTr ? 1.6 : 4.6;
      const fadeStart = isTraceTr ? 0.7 : 3.4;

      if (dzj >= 0 && dzj < maxReach && allowTrJet) {
        // Significant functional TR curves subtly medially (+x toward the IAS) past the vena contracta neck
        // while keeping the orifice neck straight along −z so CW Vmax stays exact.
        const septalBend = isTraceTr ? 0 : p.trEroCm2 && p.trEroCm2 >= 0.2 ? 0.15 : 0.06;
        const bendProgress = smoothstep(0.35, 2.4, dzj);
        const centerX = ox + septalBend * bendProgress * Math.max(0, dzj - 0.35);
        const rhoJet = Math.hypot(x - centerX, dy);
        const expandAlong = isTraceTr ? dzj : Math.max(0, dzj - 0.35);
        const spreadSlope = isTraceTr
          ? 0.16
          : 0.26 * (1 - 0.055 * Math.min(4, expandAlong));
        const Rj = (isTraceTr ? r0 : rVc) + expandAlong * spreadSlope;
        const rHalf = 0.84 * Rj;
        if (rhoJet < jetReach(rHalf, expandAlong, FAST_JET_SHEAR)) {
          const prof = jetProfile(rhoJet, rHalf, expandAlong, FAST_JET_SHEAR);
          const decay =
            freeJetCentreline(dzj, 2 * rVc) * fadeOut(dzj, fadeStart, maxReach);
          const v = vmax * prof * decay;
          const tx = septalBend * bendProgress;
          const tz = -Math.sqrt(Math.max(0, 1 - tx * tx));
          out.vx += tx * v;
          out.vz += tz * v;
          const neckBlend = isTraceTr ? 1 : smoothstep(0.15, 0.7, dzj);
          const baseTurb =
            (isTraceTr ? 0.16 : (p.turbulence['tr-jet'] ?? 0.25)) * (0.4 + 0.6 * neckBlend);
          out.dispersion = Math.max(
            out.dispersion,
            baseTurb +
              (isTraceTr ? 0.08 : 0.22 * neckBlend) *
                Math.min(1, rhoJet / Math.max(0.08, Rj)),
          );
          out.present = 1;
        }
      } else if (dz > 0 && allowTrPisa && !isTraceTr) {
        // True 3D hemispheric TR PISA on the RV side: v(d) = Q_TR / (2·π·d²·100), radial vectors toward orifice
        const d = Math.hypot(dx, dy, dz);
        const qTr = vmax * 100 * (p.trEroCm2 ?? Math.PI * rVc * rVc);
        const rShell30 = Math.sqrt(qTr / (2 * Math.PI * 30));
        const rMax = Math.min(2.4, Math.max(1.6, rShell30 * 1.9));
        if (d > 0.05 && d < rMax) {
          const vHemisphere = qTr / (2 * Math.PI * d * d * 100);
          const rhoAnn = Math.hypot(dx, dy);
          const hemiWeight = convergenceWeight(rhoAnn, dz, r0, 0.12, 0.38);
          const v =
            Math.min(vmax * 0.85, vHemisphere) * fadeOut(d, rMax * 0.78, rMax) * hemiWeight;
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
  if (
    (mask & (FLOW_MASK_MR_JET | FLOW_MASK_MR_PISA | FLOW_MASK_AR_JET | FLOW_MASK_AR_PISA)) !== 0
  ) {
    sampleRegurgitantJets(p, tables, hp, phase, x, y, z, out, mask);
  }
  if ((mask & FLOW_MASK_PV) !== 0) {
    samplePulmonaryVeins(p, tables, hp, phase, x, y, z, out);
  }
  if ((mask & FLOW_MASK_SYS_VEINS) !== 0) {
    sampleSystemicVenousFlow(p, tables, hp, phase, x, y, z, out, structure);
  }
}
