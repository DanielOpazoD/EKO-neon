import { heartAnchors, type HeartPose } from '@/simulator/anatomy/heartModel';
import { rvOutflowScale } from '@/simulator/anatomy/classify/rightVentricle';
import { sampleTable, type BeatTables } from '@/simulator/cardiac-cycle/cycleModel';
import { smoothstep } from '@/core/vec3';
import { aorticVelocityProfile } from './aorticNetwork';
import type { FlowFieldParams } from './flowField';
import type { FlowSample } from './jetProfiles';

/** Upstream convergence distance (cm) before the RVOT infundibular inlet (`rvotA`). */
const RVOT_INLET_FADE_CM = 1.2;

/**
 * Accumulator for smooth junction-blended evaluation across the 6-segment RVOT → PA tree
 * (zero per-sample heap allocation).
 */
const ACC = {
  wSum: 0,
  vxSum: 0,
  vySum: 0,
  vzSum: 0,
  dispMax: 0,
};

function accumulateSegment(
  x: number,
  y: number,
  z: number,
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  rLumA: number,
  rLumB: number,
  rEffA: number,
  rEffB: number,
  flowMlps: number,
  baseDisp: number,
  dilatedDisp: number,
  allowInletFade: boolean,
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

  let axialFade = 1;
  let u = 0;
  let rho = 0;
  let rLum = rLumA;
  let rEff = rEffA;

  if (allowInletFade && t < 0) {
    if (t <= -RVOT_INLET_FADE_CM) return;
    const up = -t;
    axialFade = smoothstep(-RVOT_INLET_FADE_CM, 0, t);
    const qx = px - ux * t;
    const qy = py - uy * t;
    const qz = pz - uz * t;
    rho = Math.sqrt(qx * qx + qy * qy + qz * qz);
    const widen = 1 + 0.25 * up;
    rLum = rLumA * widen;
    rEff = rEffA * widen;
  } else {
    const rMax = Math.max(rLumA, rLumB);
    if (t < -rMax * 1.15 || t > L + rMax * 1.15) return;
    const tc = t < 0 ? 0 : t > L ? L : t;
    u = tc / L;
    const qx = px - ux * tc;
    const qy = py - uy * tc;
    const qz = pz - uz * tc;
    rho = Math.sqrt(qx * qx + qy * qy + qz * qz);
    rLum = rLumA + (rLumB - rLumA) * u;
    rEff = rEffA + (rEffB - rEffA) * u;
  }

  const dNorm = rho / Math.max(0.15, rLum);
  if (dNorm >= 1.08) return;

  const rNorm = Math.min(1, dNorm / 1.04);
  const prof = aorticVelocityProfile(rNorm) * axialFade;
  if (prof <= 1e-4) return;

  const areaCm2 = Math.PI * rEff * rEff;
  const speed = (flowMlps / areaCm2 / 100) * prof;
  if (speed <= 0.005) return;

  // Smooth cubic clearance weight blends direction and speed seamlessly across RVOT bends and the PA bifurcation
  const clearance = 1.08 - dNorm;
  const w = clearance * clearance * clearance;
  ACC.wSum += w;
  ACC.vxSum += w * ux * speed;
  ACC.vySum += w * uy * speed;
  ACC.vzSum += w * uz * speed;
  const wallShear = 0.06 * smoothstep(0.62, 1.0, rNorm);
  ACC.dispMax = Math.max(ACC.dispMax, baseDisp + wallShear + dilatedDisp);
}

/**
 * Mass-conserving systolic flow through the anatomical RVOT infundibulum (`rvotA → rvotM → rvotB`),
 * pulmonary root (`rvotB → paStj`), main pulmonary artery trunk (`paStj → paEnd`), and right/left
 * pulmonary artery branches (`paEnd → rpaEnd`, `paEnd → lpaEnd`), mirroring `classifyRightVentricle.ts`
 * and `greatVessels.ts` geometry with the same moving pulmonary base (`hp.pvZ`).
 */
export function samplePulmonarySystolicFlow(
  p: FlowFieldParams,
  tables: BeatTables,
  hp: HeartPose,
  phase: number,
  x: number,
  y: number,
  z: number,
  out: FlowSample,
): void {
  if (p.enabled['rvot'] === false) return;
  const qpv = sampleTable(tables.pulmonaryFlowMlps, phase);
  if (qpv <= 1) return;

  const A = heartAnchors(p.heart);
  const pvZ = hp.pvZ;
  const k = rvOutflowScale(hp.state.contraction);
  const baseDisp = p.turbulence['rvot'] ?? 0.04;
  // Mild systolic swirling dispersion in dilated pulmonary trunks (e.g. pulmonary hypertension)
  const dilatedDisp = 0.08 * smoothstep(1.25, 1.65, A.paR);

  // Area-proportional flow split at the PA bifurcation (~53% RPA / ~47% LPA in adult anatomy)
  const rpaA = A.rpaR * A.rpaR;
  const lpaA = A.lpaR * A.lpaR;
  const rpaShare = rpaA / Math.max(1e-6, rpaA + lpaA);

  ACC.wSum = 0;
  ACC.vxSum = 0;
  ACC.vySum = 0;
  ACC.vzSum = 0;
  ACC.dispMax = 0;

  // 1. Proximal RVOT infundibulum (rvotA → rvotM, moving with pvZ)
  accumulateSegment(
    x,
    y,
    z,
    A.rvotA.x,
    A.rvotA.y,
    A.rvotA.z + pvZ,
    A.rvotM.x,
    A.rvotM.y,
    A.rvotM.z + pvZ,
    A.rvotRa * k,
    A.rvotRm * k,
    A.rvotRa,
    A.rvotRm,
    qpv,
    baseDisp,
    0,
    true,
  );

  // 2. Distal RVOT to pulmonary valve plane (rvotM → rvotB, moving with pvZ)
  accumulateSegment(
    x,
    y,
    z,
    A.rvotM.x,
    A.rvotM.y,
    A.rvotM.z + pvZ,
    A.rvotB.x,
    A.rvotB.y,
    A.rvotB.z + pvZ,
    A.rvotRm * k,
    A.rvotR * k,
    A.rvotRm,
    A.rvotR,
    qpv,
    baseDisp,
    0,
    false,
  );

  // 3. Pulmonary root / sinuses (rvotB → paStj, moving with pvZ)
  accumulateSegment(
    x,
    y,
    z,
    A.rvotB.x,
    A.rvotB.y,
    A.rvotB.z + pvZ,
    A.paStj.x,
    A.paStj.y,
    A.paStj.z + pvZ,
    A.paRootR,
    A.paR,
    A.rvotR,
    A.paR,
    qpv,
    baseDisp,
    dilatedDisp,
    false,
  );

  // 4. Main pulmonary artery trunk (paStj + pvZ → fixed bifurcation paEnd)
  accumulateSegment(
    x,
    y,
    z,
    A.paStj.x,
    A.paStj.y,
    A.paStj.z + pvZ,
    A.paEnd.x,
    A.paEnd.y,
    A.paEnd.z,
    A.paR,
    A.paR,
    A.paR,
    A.paR,
    qpv,
    baseDisp,
    dilatedDisp,
    false,
  );

  // 5. Right pulmonary artery (paEnd → rpaEnd, passing beneath the aortic arch)
  accumulateSegment(
    x,
    y,
    z,
    A.paEnd.x,
    A.paEnd.y,
    A.paEnd.z,
    A.rpaEnd.x,
    A.rpaEnd.y,
    A.rpaEnd.z,
    A.rpaR,
    A.rpaR,
    A.rpaR,
    A.rpaR,
    qpv * rpaShare,
    baseDisp,
    dilatedDisp * 0.6,
    false,
  );

  // 6. Left pulmonary artery (paEnd → lpaEnd)
  accumulateSegment(
    x,
    y,
    z,
    A.paEnd.x,
    A.paEnd.y,
    A.paEnd.z,
    A.lpaEnd.x,
    A.lpaEnd.y,
    A.lpaEnd.z,
    A.lpaR,
    A.lpaR,
    A.lpaR,
    A.lpaR,
    qpv * (1 - rpaShare),
    baseDisp,
    dilatedDisp * 0.6,
    false,
  );

  if (ACC.wSum > 0) {
    const invW = 1 / ACC.wSum;
    out.vx += ACC.vxSum * invW;
    out.vy += ACC.vySum * invW;
    out.vz += ACC.vzSum * invW;
    out.dispersion = Math.max(out.dispersion, ACC.dispMax);
    out.present = 1;
  }
}
