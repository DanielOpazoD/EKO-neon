import { heartAnchors, type HeartPose } from '@/simulator/anatomy/heartModel';
import { IVC_HEPATIC_SEGMENT_CM } from '@/simulator/anatomy/anchors';
import {
  CS_ABOVE_ANNULUS_CM,
  CS_CRUX_FROM_IAS_CM,
  CS_GROOVE_OFFSET_CM,
  CS_LATERAL_X_CM,
  CS_OSTIUM_ABOVE_ANNULUS_CM,
  CS_OSTIUM_FROM_IAS_CM,
  CS_OSTIUM_RADIUS_CM,
  CS_OSTIUM_Y_CM,
  CS_RADIUS_CM,
  CS_WALL_CM,
} from '@/simulator/anatomy/classify/atriaVessels';
import { LA_POSTERIOR_FLOOR_DROP_CM, LA_POSTERIOR_RATIO } from '@/simulator/anatomy/laGeometry';
import { Structure } from '@/simulator/anatomy/tissue';
import { sampleTable, type BeatTables } from '@/simulator/cardiac-cycle/cycleModel';
import { smoothstep } from '@/core/vec3';
import type { FlowFieldParams } from './flowField';
import { fadeOut, type FlowSample } from './jetProfiles';

/** Lumen radius (cm) of the middle hepatic vein (matches `sdCapsule(..., 0.4)` in `atriaVessels.ts`). */
const HV_RADIUS_CM = 0.4;
/** Reach (cm) of the gentle caval inflow confluence plumes inside the right atrium. */
const IVC_RA_CONFLUENCE_CM = 1.25;
const SVC_RA_CONFLUENCE_CM = 1.2;

/** Zero-allocation scratch accumulator for smooth junction-blended caval/hepatic flow. */
const VEIN_ACC = {
  wSum: 0,
  vxSum: 0,
  vySum: 0,
  vzSum: 0,
  dispMax: 0,
};

/**
 * Triphasic systemic caval / hepatic venous axial velocity (m/s; + = antegrade toward RA, − = retrograde reversal):
 * - S wave: ventricular systolic forward return driven by TAPSE / RA reservoir relaxation; blunts in moderate TR
 *   and reverses holosystolically in severe TR (`trEroCm2 >= 0.25`).
 * - D wave: early diastolic conduit filling mirroring tricuspid E opening; blunted in cardiac tamponade.
 * - Ar wave: late diastolic retrograde atrial kick, augmented with elevated RAP and absent in AF.
 * - Inspiratory augmentation scales antegrade return with `hp.ivcCollapse`.
 */
export function cavalVenousVelocityMps(
  p: FlowFieldParams,
  tables: BeatTables,
  hp: HeartPose,
  phase: number,
): number {
  const tm = tables.timings;
  const rr = tables.rrS;
  const t = (((phase % 1) + 1) % 1) * rr;

  // S wave: ventricular systole (annular descent / RA reservoir filling)
  const sStart = Math.max(0, tm.pulmonaryOpenS - 0.025);
  const sEnd = Math.min(rr, tm.pulmonaryCloseS + 0.06);
  const sDur = Math.max(0.05, sEnd - sStart);
  const uS = (t - sStart) / sDur;
  const sWave = uS > 0 && uS < 1 ? Math.pow(Math.sin(Math.PI * uS), 0.85) : 0;

  // D wave: early diastolic conduit flow mirroring tricuspid early inflow
  const tvEnd = Math.min(rr, tm.tricuspidOpenS + tm.eAccelS + tm.eDecelS + 0.03);
  const inD = t > tm.tricuspidOpenS && t < tvEnd;
  const dWave = inD ? sampleTable(tables.tricuspidFlowMlps, phase) / Math.max(1, p.flowPeaks.tricuspid) : 0;

  // Ar wave: atrial contraction retrograde kick (absent in AF)
  const inAr = tm.hasAWave && t > tm.aStartS && t < tm.aEndS + 0.03;
  const arWave = inAr
    ? Math.sin((Math.PI * (t - tm.aStartS)) / Math.max(0.02, tm.aEndS + 0.03 - tm.aStartS))
    : 0;

  const tapse = p.heart.physiology.tapseCm;
  const trEro = p.trEroCm2 ?? 0;
  const tamponade = p.heart.anatomy.pericardium.tamponade;

  // Normal S > D (~0.46 m/s vs ~0.32 m/s); moderate TR blunts S; severe TR (EROA >= 0.25 cm²) reverses S
  const sBase = 0.46 * Math.min(1.25, Math.max(0.55, tapse / 2.0));
  const sMod = 1 - 1.95 * Math.min(1, trEro / 0.32);
  let sComp = sBase * sMod * sWave;
  if (trEro >= 0.22 && p.trVmaxMps && p.trVmaxMps > 0.5) {
    // Follow the true TR systolic pressure curve when systolic hepatic reversal is present
    const trFrac = sampleTable(tables.trVelocityMps, phase) / p.trVmaxMps;
    const revAmp = 0.48 * smoothstep(0.2, 0.32, trEro);
    sComp = Math.min(sComp, -revAmp * trFrac);
  }

  // Tamponade blunts early diastolic RV filling (dominant systolic S, blunted D); severe TR augments D
  const dPeak =
    0.32 * (1 - 0.72 * Math.min(1, tamponade)) * (1 + 0.35 * Math.min(1, trEro / 0.3));
  const arPeak = tm.hasAWave ? 0.18 + 0.14 * smoothstep(5, 16, p.rapMmHg) : 0;

  const rawVel = sComp + dPeak * dWave - arPeak * arWave;
  // Inspiration (negative intrathoracic pressure) augments antegrade caval return into the RA
  const inspGain = 1 + 0.45 * Math.max(0, hp.ivcCollapse);
  return rawVel > 0 ? rawVel * inspGain : rawVel;
}

function accumulateVeinSegment(
  x: number,
  y: number,
  z: number,
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  radiusCm: number,
  axialVelMps: number,
  disp: number,
  outletReachCm: number,
): void {
  const dx = bx - ax;
  const dy = by - ay;
  const dz = bz - az;
  const L2 = dx * dx + dy * dy + dz * dz;
  if (L2 < 1e-8) return;
  const L = Math.sqrt(L2);
  const ux = dx / L;
  const uy = dy / L;
  const uz = dz / L;

  const px = x - ax;
  const py = y - ay;
  const pz = z - az;
  const t = px * ux + py * uy + pz * uz;

  if (t < -radiusCm * 1.1 || t > L + Math.max(radiusCm * 1.1, outletReachCm)) return;

  let rLocal = radiusCm;
  let axialFade = 1;
  let rho = 0;

  if (outletReachCm > 0 && t > L) {
    const past = t - L;
    rLocal = radiusCm + past * 0.42;
    axialFade = (1 / (1 + past / 0.85)) * fadeOut(past, outletReachCm * 0.35, outletReachCm);
    const qx = px - ux * t;
    const qy = py - uy * t;
    const qz = pz - uz * t;
    rho = Math.sqrt(qx * qx + qy * qy + qz * qz);
  } else {
    const tc = t < 0 ? 0 : t > L ? L : t;
    const qx = px - ux * tc;
    const qy = py - uy * tc;
    const qz = pz - uz * tc;
    rho = Math.sqrt(qx * qx + qy * qy + qz * qz);
  }

  const dNorm = rho / Math.max(0.15, rLocal);
  if (dNorm >= 1.08) return;

  const rNorm = Math.min(1, dNorm / 1.05);
  // Blunt venous velocity profile (zero slip at intima, rich core filling across the subcostal IVC/HV lumen)
  const r2 = rNorm * rNorm;
  const prof = (1 - r2 * r2) * axialFade;
  const v = axialVelMps * prof;
  if (Math.abs(v) < 0.005) return;

  const clearance = 1.08 - dNorm;
  const w = clearance * clearance * clearance;
  VEIN_ACC.wSum += w;
  VEIN_ACC.vxSum += w * ux * v;
  VEIN_ACC.vySum += w * uy * v;
  VEIN_ACC.vzSum += w * uz * v;
  VEIN_ACC.dispMax = Math.max(VEIN_ACC.dispMax, disp);
}

/**
 * Samples physiological triphasic venous flow in the inferior vena cava (suprahepatic + intrahepatic segments),
 * middle hepatic vein (`hvB → hvA`), and the gentle caval inflow confluence inside the right atrial floor.
 */
export function sampleIvcHepaticFlow(
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
  const inIvc = structure === undefined || structure === Structure.Ivc;
  const inHv = structure === undefined || structure === Structure.HepaticVein;
  const inRa = structure === undefined || structure === Structure.RaCavity;
  if (!inIvc && !inHv && !inRa) return;

  const vCaval = cavalVenousVelocityMps(p, tables, hp, phase);
  if (Math.abs(vCaval) < 0.015) return;

  const A = heartAnchors(p.heart);
  const rI = Math.max(0.25, A.ivcR * (1 - hp.ivcCollapse));
  // Slight turbulence during systolic retrograde TR pulse into IVC/hepatic vein
  const baseDisp = vCaval < -0.22 ? 0.14 : 0.05;

  VEIN_ACC.wSum = 0;
  VEIN_ACC.vxSum = 0;
  VEIN_ACC.vySum = 0;
  VEIN_ACC.vzSum = 0;
  VEIN_ACC.dispMax = 0;

  if (inIvc || inRa) {
    // 1. Intrahepatic IVC segment: distal ivcC → ivcB (antegrade toward RA)
    if (inIvc) {
      accumulateVeinSegment(
        x,
        y,
        z,
        A.ivcB.x + IVC_HEPATIC_SEGMENT_CM[0],
        A.ivcB.y + IVC_HEPATIC_SEGMENT_CM[1],
        A.ivcB.z + IVC_HEPATIC_SEGMENT_CM[2],
        A.ivcB.x,
        A.ivcB.y,
        A.ivcB.z,
        rI,
        vCaval,
        baseDisp,
        0,
      );
    }
    // 2. Suprahepatic IVC: ivcB → ivcA (+ gentle entry confluence into RaCavity)
    accumulateVeinSegment(
      x,
      y,
      z,
      A.ivcB.x,
      A.ivcB.y,
      A.ivcB.z,
      A.ivcA.x,
      A.ivcA.y,
      A.ivcA.z,
      rI,
      vCaval,
      baseDisp,
      inRa ? IVC_RA_CONFLUENCE_CM : 0,
    );
  }

  if (inHv || inIvc) {
    // 3. Middle hepatic vein: distal liver parenchyma (hvB) → IVC confluence (hvA)
    accumulateVeinSegment(
      x,
      y,
      z,
      A.hvB.x,
      A.hvB.y,
      A.hvB.z,
      A.hvA.x,
      A.hvA.y,
      A.hvA.z,
      HV_RADIUS_CM,
      vCaval * 1.12,
      baseDisp,
      0,
    );
  }

  if (VEIN_ACC.wSum > 0) {
    const invW = 1 / VEIN_ACC.wSum;
    out.vx += VEIN_ACC.vxSum * invW;
    out.vy += VEIN_ACC.vySum * invW;
    out.vz += VEIN_ACC.vzSum * invW;
    out.dispersion = Math.max(out.dispersion, VEIN_ACC.dispMax);
    out.present = 1;
  }
}

/**
 * Biphasic antegrade coronary sinus venous drainage velocity (m/s; + = lateral AV groove → RA ostium):
 * - Systolic squeezing wave from intramyocardial venous compression during ventricular ejection (~0.32 m/s).
 * - Diastolic runoff wave during passive coronary perfusion (~0.25 m/s).
 * - Brief late-diastolic slowing during atrial contraction (absent in AF).
 */
export function coronarySinusVelocityMps(tables: BeatTables, phase: number): number {
  const tm = tables.timings;
  const rr = tables.rrS;
  const t = (((phase % 1) + 1) % 1) * rr;

  // Systolic myocardial squeezing wave
  const sStart = Math.max(0, tm.ejectionStartS - 0.01);
  const sEnd = Math.min(rr, tm.ejectionEndS + 0.05);
  const uS = (t - sStart) / Math.max(0.05, sEnd - sStart);
  const sWave = uS > 0 && uS < 1 ? Math.pow(Math.sin(Math.PI * uS), 0.9) : 0;

  // Diastolic passive coronary venous drainage (early-to-mid diastole)
  const dStart = tm.mitralOpenS;
  const dEnd = tm.hasAWave ? Math.max(dStart + 0.08, tm.aStartS) : rr;
  const uD = (t - dStart) / Math.max(0.06, dEnd - dStart);
  const dWave = uD > 0 && uD < 1 ? Math.pow(Math.sin(Math.PI * uD), 0.75) : 0;

  // Late-diastolic atrial contraction dip (absent in AF)
  const inA = tm.hasAWave && t > tm.aStartS && t < tm.aEndS + 0.025;
  const aDip = inA
    ? Math.sin((Math.PI * (t - tm.aStartS)) / Math.max(0.02, tm.aEndS + 0.025 - tm.aStartS))
    : 0;

  return 0.32 * sWave + 0.25 * dWave - 0.08 * aDip;
}

/**
 * Samples physiological superior vena cava (`svcB → svcM → svcA` + RA roof confluence) and
 * coronary sinus (posterior AV groove → RA ostium) venous drainage, strictly contained to
 * `Structure.Svc`, `Structure.CoronarySinus`, and `Structure.RaCavity`.
 */
export function sampleSvcAndCoronarySinusFlow(
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
  const inSvc = structure === undefined || structure === Structure.Svc;
  const inRa = structure === undefined || structure === Structure.RaCavity;
  const inCs = structure === undefined || structure === Structure.CoronarySinus;
  if (!inSvc && !inRa && !inCs) return;

  const A = heartAnchors(p.heart);

  // ---- Superior vena cava (svcB → svcM → svcA + RA roof confluence) ----
  if (inSvc || inRa) {
    const vSvc = cavalVenousVelocityMps(p, tables, hp, phase) * 1.04;
    if (Math.abs(vSvc) >= 0.015) {
      const baseDisp = vSvc < -0.22 ? 0.14 : 0.05;
      VEIN_ACC.wSum = 0;
      VEIN_ACC.vxSum = 0;
      VEIN_ACC.vySum = 0;
      VEIN_ACC.vzSum = 0;
      VEIN_ACC.dispMax = 0;

      if (inSvc) {
        accumulateVeinSegment(
          x,
          y,
          z,
          A.svcB.x,
          A.svcB.y,
          A.svcB.z,
          A.svcM.x,
          A.svcM.y,
          A.svcM.z,
          A.svcR,
          vSvc,
          baseDisp,
          0,
        );
      }
      accumulateVeinSegment(
        x,
        y,
        z,
        A.svcM.x,
        A.svcM.y,
        A.svcM.z,
        A.svcA.x,
        A.svcA.y,
        A.svcA.z,
        A.svcR,
        vSvc,
        baseDisp,
        inRa ? SVC_RA_CONFLUENCE_CM : 0,
      );

      if (VEIN_ACC.wSum > 0) {
        const invW = 1 / VEIN_ACC.wSum;
        out.vx += VEIN_ACC.vxSum * invW;
        out.vy += VEIN_ACC.vySum * invW;
        out.vz += VEIN_ACC.vzSum * invW;
        out.dispersion = Math.max(out.dispersion, VEIN_ACC.dispMax);
        out.present = 1;
      }
    }
  }

  // ---- Coronary sinus (posterior AV groove lateral → crux → RA ostium) ----
  if (inCs) {
    const vCs = coronarySinusVelocityMps(tables, phase);
    if (Math.abs(vCs) < 0.01) return;

    const zs = hp.zAnn - CS_ABOVE_ANNULUS_CM;
    const x0 = A.iasX + CS_CRUX_FROM_IAS_CM;
    const la = A.laCenter;
    const lr = A.laR;
    const x1 = Math.min(CS_LATERAL_X_CM, la.x + 0.85 * lr.x);
    const ox = A.iasX - CS_OSTIUM_FROM_IAS_CM;
    const oy = CS_OSTIUM_Y_CM;
    const oz = hp.zAnn - CS_OSTIUM_ABOVE_ANNULUS_CM;

    // Quick bounding reject for unclassified callers
    if (
      structure === undefined &&
      (Math.abs(z - zs - 0.05) > 0.85 || x < ox - CS_OSTIUM_RADIUS_CM - 0.2 || x > x1 + CS_RADIUS_CM + 0.2)
    ) {
      return;
    }

    const zTop = la.z - lr.z;
    const zBottom = hp.zAnn + 0.25;
    const czL = 0.5 * (zTop + zBottom);
    const rzL = Math.max(0.5, 0.5 * (zBottom - zTop));
    const csGrooveY = (xq: number): number => {
      const u = (Math.max(xq, la.x) - la.x) / lr.x;
      const w = (zs - czL) / rzL;
      const q = Math.max(0, 1 - u * u - w * w);
      const yLa = Math.max(la.y - lr.y * Math.sqrt(q), la.y - LA_POSTERIOR_RATIO * lr.y);
      const n = A.laPostN;
      const yWall = (A.laPostD - CS_RADIUS_CM - CS_WALL_CM - xq * n.x - zs * n.z) / n.y;
      return Math.max(yLa - LA_POSTERIOR_FLOOR_DROP_CM - CS_GROOVE_OFFSET_CM, yWall);
    };

    const yCrux = csGrooveY(x0);
    let tx = -1;
    let ty = 0;
    let tz = 0;
    let rNorm = 0;

    if (x >= x0) {
      // Posterior AV groove body: tangent follows (xc, csGrooveY(xc), zs) toward the crux (−x)
      const xc = Math.min(x1, x);
      const yc = csGrooveY(xc);
      const xStep = Math.max(x0, xc - 0.25);
      const yStep = csGrooveY(xStep);
      const dxT = xStep - xc;
      const dyT = yStep - yc;
      const len = Math.hypot(dxT, dyT);
      if (len > 1e-4) {
        tx = dxT / len;
        ty = dyT / len;
      }
      const dist = Math.hypot(x - xc, y - yc, z - zs);
      rNorm = dist / CS_RADIUS_CM;
    } else {
      // Ostial funnel entering inferomedial RA: (x0, yCrux, zs) → (ox, oy, oz)
      const dxO = ox - x0;
      const dyO = oy - yCrux;
      const dzO = oz - zs;
      const L = Math.hypot(dxO, dyO, dzO) || 1;
      tx = dxO / L;
      ty = dyO / L;
      tz = dzO / L;
      const px = x - x0;
      const py = y - yCrux;
      const pz = z - zs;
      const u = Math.max(0, Math.min(1, (px * tx + py * ty + pz * tz) / L));
      const rLocal = CS_RADIUS_CM + (CS_OSTIUM_RADIUS_CM - CS_RADIUS_CM) * u;
      const dist = Math.hypot(px - tx * u * L, py - ty * u * L, pz - tz * u * L);
      rNorm = dist / rLocal;
    }

    if (structure === undefined && rNorm >= 1.05) return;
    const rn = Math.min(structure === Structure.CoronarySinus ? 0.85 : 1.0, rNorm);
    const prof = 1 - rn * rn * rn * rn;
    const v = vCs * prof;
    if (Math.abs(v) > 0.005) {
      out.vx += tx * v;
      out.vy += ty * v;
      out.vz += tz * v;
      out.dispersion = Math.max(out.dispersion, 0.05);
      out.present = 1;
    }
  }
}

/** Unified systemic venous return sampler (IVC, middle hepatic vein, SVC, coronary sinus, and RA caval confluences). */
export function sampleSystemicVenousFlow(
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
  sampleIvcHepaticFlow(p, tables, hp, phase, x, y, z, out, structure);
  sampleSvcAndCoronarySinusFlow(p, tables, hp, phase, x, y, z, out, structure);
}
