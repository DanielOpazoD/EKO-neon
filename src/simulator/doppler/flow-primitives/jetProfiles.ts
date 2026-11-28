import type { BeatTables } from '@/simulator/cardiac-cycle/cycleModel';
import { sampleTable } from '@/simulator/cardiac-cycle/cycleModel';
import type { CaseDefinition } from '@/cases/schema';
import { smoothstep } from '@/core/vec3';

/**
 * Shared jet profile, convergence sink, and inflow-wave propagation math used across
 * all hemodynamic Doppler flow primitives (mitral, tricuspid, LVOT/AV, RVOT, regurgitant jets, pulmonary veins).
 */
export interface FlowSample {
  vx: number;
  vy: number;
  vz: number;
  /** Fractional velocity dispersion (0 laminar … 1 fully turbulent). */
  dispersion: number;
  /** 1 if inside a flow primitive, 0 otherwise. */
  present: number;
}

/**
 * Colour M-mode flow propagation velocity (cm/s) of a case. The guideline cut-offs for raised filling pressure are a septal
 * E/e′ above 15 and an E/Vp of 2.5 or more (Nagueh et al., J Am Soc Echocardiogr 2009; 22:107–133), so at that pressure
 * Vp = 6·e′ septal; the proportion is kept at every e′ as a declared assumption. It gives 66 cm/s for the normal e′ of
 * 11 cm/s (normal Vp above 50 cm/s) and 36 cm/s for an impaired-relaxation e′ of 6 cm/s.
 */
export function flowPropagationCmps(c: CaseDefinition): number {
  return Math.min(120, Math.max(15, 6 * c.physiology.ePrimeSeptalCmps));
}

/**
 * Length (cm) of the inflow core beyond the annulus through the open mitral leaflet tips so both
 * early (E) and late-diastolic (A, after annular recoil) inflow preserve full orifice velocity at
 * the standard ASE leaflet-tip PW sampling plane.
 */
export const INFLOW_CORE_CM = 1.8;

/**
 * Distance (cm) beyond the tips over which the inflow core falls to half its velocity. The conventional Vp follows the
 * contour at half the maximal inflow velocity for 4 cm into the ventricle (Garcia et al., J Am Coll Cardiol 1998; 32:865–875,
 * as described by Chakraborty et al. 2019), so the core keeps at least half its velocity there; 5 cm is a declared value
 * above that bound.
 */
const INFLOW_DECAY_CM = 5;

/** Fraction of the core velocity left `zr` cm beyond the annulus, fading over the last centimetre before the apex. */
export function inflowAxial(zr: number, lengthNow: number): number {
  return (
    smoothstep(0.2, 1.2, lengthNow - zr) / (1 + Math.max(0, zr - INFLOW_CORE_CM) / INFLOW_DECAY_CM)
  );
}

const SHEAR_ORIFICE_CM = 0.25;
const SHEAR_GROWTH = 0.25;

/**
 * Shear layer of a regurgitant or stenotic jet: ~1.1 mm wide at its orifice, feathering organically downstream.
 */
export const FAST_JET_SHEAR = { orifice: 0.11, growth: 0.085 } as const;
export const SLOW_SHEAR = { orifice: SHEAR_ORIFICE_CM, growth: SHEAR_GROWTH } as const;
export type Shear = { orifice: number; growth: number };

/** Logistic fall of a jet's velocity at `rho` from its axis, with ½ at `rHalf` over a layer `w` wide. */
function logisticFall(rho: number, rHalf: number, w: number): number {
  return 1 / (1 + Math.exp((3 * (rho - rHalf)) / w));
}

/**
 * Centreline speed of a jet, as a fraction of its orifice speed, at distance x from an orifice of diameter d (decision
 * 240): a round turbulent jet decays as U_c/U_J = B/(x/d) with B = 5.8 (Pope, Turbulent Flows, 2000, eq. 5.6, on the
 * measurements of Hussein, Capp & George 1994), so it keeps its speed over its first B·d.
 */
const JET_DECAY_B = 5.8;
export function freeJetCentreline(x: number, dCm: number): number {
  return x <= JET_DECAY_B * dCm ? 1 : (JET_DECAY_B * dCm) / x;
}

/** Where the SAM–septal contact narrows the outflow tract (cm along the aortic axis from the valve) and over what length. */
export const LVOT_OBSTRUCTION_T = -0.6;
export const LVOT_OBSTRUCTION_SIGMA_CM = 0.45;

export function jetProfile(
  rho: number,
  rHalf: number,
  s: number,
  shear: Shear = SLOW_SHEAR,
): number {
  const w = shear.orifice + shear.growth * Math.max(0, s);
  if (rho >= rHalf + 3 * w) return 0;
  return logisticFall(rho, rHalf, w) / logisticFall(0, rHalf, w);
}

/** Distance from the axis beyond which a jet carries no velocity (0.01 % of its core). */
export function jetReach(rHalf: number, s: number, shear: Shear = SLOW_SHEAR): number {
  return rHalf + 3 * (shear.orifice + shear.growth * Math.max(0, s));
}

/** 1 before `from`, 0 after `to`: the end of a jet or of a convergence zone. */
export function fadeOut(s: number, from: number, to: number): number {
  return 1 - smoothstep(from, to, s);
}

const FUNNEL_FADE_FROM_CM = 2;
export const FUNNEL_FADE_TO_CM = 3;

export const LVOT_ENTRANCE_T = -0.8;
export const LVOT_SINK_FADE_FROM_CM = 1.1;
export const LVOT_SINK_FADE_TO_CM = 2.3;

export const LVOT_SINK_COS_FROM = 0.65;
export const LVOT_SINK_COS_TO = 0.88;
const FUNNEL_COS_FROM = 0.3;
const FUNNEL_COS_TO = 0.6;

/**
 * Weight of a convergence zone at `rho` from the axis and `upstream` cm before an orifice of radius `r`: whole over the
 * orifice, fading beside it toward the plane of the orifice, where the jet it feeds does not reach (decision 166).
 */
export function convergenceWeight(
  rho: number,
  upstream: number,
  r: number,
  cosFrom = FUNNEL_COS_FROM,
  cosTo = FUNNEL_COS_TO,
): number {
  if (upstream <= 0) return 0;
  return smoothstep(cosFrom, cosTo, upstream / Math.hypot(Math.max(0, rho - r), upstream));
}

/**
 * Inflow (mL/s) that left the leaflet tips when the filling wave now `zr` cm beyond an atrioventricular annulus was there
 * (decision 102): the wave takes (zr − INFLOW_CORE_CM)/speed to arrive. Zero for a wave that would have left before the
 * beat began, when the ventricle already contracts.
 */
export function inflowFlow(
  tables: BeatTables,
  phase: number,
  zr: number,
  waveCmps: number,
  table: Float32Array = tables.mitralFlowMlps,
): number {
  const delayed = phase - Math.max(0, zr - INFLOW_CORE_CM) / (waveCmps * tables.rrS);
  return delayed < 0 ? 0 : sampleTable(table, delayed);
}

/**
 * Filling through an atrioventricular valve (decision 166): the atrial convergence toward the orifice and the jet beyond it.
 * `zr` is the distance beyond the annulus, `r` the orifice radius. The filling wave reaches a point of the jet after the
 * distance from the orifice centre, so its front is a hemisphere around the orifice rather than a plane across the jet.
 */
export function inflowJet(
  tables: BeatTables,
  table: Float32Array,
  phase: number,
  waveCmps: number,
  dx: number,
  dy: number,
  zr: number,
  r: number,
  areaCm2: number,
  axial: (zr: number) => number,
  turbulence: number,
  out: FlowSample,
): void {
  const rho = Math.hypot(dx, dy);
  const re = Math.sqrt(areaCm2 / Math.PI);
  if (zr < 0) {
    // Smooth atrial convergence funnel meeting the wide mitral/tricuspid annulus continuously at 0.76x tip speed
    // so there is zero horizontal seam or premature aliasing line at zr = 0.
    const up = -zr;
    const d = Math.hypot(rho, up);
    if (d >= FUNNEL_FADE_TO_CM) return;
    const q = inflowFlow(tables, phase, 0, waveCmps, table);
    if (q <= 1) return;
    const rFunnel = re * 0.96 * (1 + 0.36 * up);
    const rNorm = Math.min(1.5, rho / Math.max(0.25, rFunnel));
    const radialCore = 1 / (1 + 0.48 * rNorm * rNorm);
    const dEff = Math.hypot(rho * 0.72, up + 0.79 * re);
    const vOrifice = q / areaCm2 / 100;
    const baseMag =
      vOrifice *
      Math.min(0.76, (0.48 * re * re) / (dEff * dEff)) *
      radialCore *
      fadeOut(d, FUNNEL_FADE_FROM_CM, FUNNEL_FADE_TO_CM) *
      convergenceWeight(rho, up + 0.24, r * 0.98, 0.18, 0.52);
    // Stenotic AV inflow (e.g. rheumatic mitral stenosis, MVA < 2.2 cm²): laminar 3D hemispheric
    // proximal flow convergence (LA-side PISA aliasing dome) accelerating into the restricted tips.
    let mag = baseMag;
    const msWeight = areaCm2 < 2.2 ? smoothstep(2.2, 1.3, areaCm2) : 0;
    if (msWeight > 0) {
      const dPisa = Math.hypot(rho, up + 0.14);
      const vHemi = Math.min(vOrifice * 0.84, q / (2 * Math.PI * dPisa * dPisa * 100));
      const pisaMag =
        msWeight *
        vHemi *
        fadeOut(d, 1.45, FUNNEL_FADE_TO_CM) *
        convergenceWeight(rho, up + 0.18, r * 0.98, 0.15, 0.45);
      if (pisaMag > mag) mag = pisaMag;
    }
    if (mag <= 0.01) return;
    const radialGain = msWeight > 0 ? 0.65 + 0.25 * msWeight : 0.65;
    const sx = -dx * radialGain,
      sy = -dy * radialGain,
      sz = r * (1 - 0.45 * msWeight) + up;
    const n = Math.hypot(sx, sy, sz);
    out.vx += (sx / n) * mag;
    out.vy += (sy / n) * mag;
    out.vz += (sz / n) * mag;
    out.dispersion = Math.max(out.dispersion, Math.min(turbulence, 0.08));
    out.present = 1;
    return;
  }
  // Sculpted transmitral/tricuspid filling flame: accelerates physiologically from the wider annulus (0.76x at
  // zr = 0) to full orifice speed at the leaflet tips (zr >= 0.65 cm), preserving exact E/A/DT/VTI/Vp at the
  // leaflet-tip PW sampling plane while rounding off outer margins smoothly.
  const rHalf = re + Math.min(2.5, Math.max(0, zr - INFLOW_CORE_CM)) * 0.16;
  if (rho >= jetReach(rHalf, zr)) return;
  const q = inflowFlow(tables, phase, Math.hypot(zr, rho), waveCmps, table);
  if (q <= 1) return;
  const rNorm = Math.min(1.5, rho / Math.max(0.25, rHalf));
  const rNorm2 = rNorm * rNorm;
  const distalTaper = 0.18 + 0.6 * smoothstep(INFLOW_CORE_CM, INFLOW_CORE_CM + 2.2, zr);
  const radialCore = 1 / (1 + distalTaper * rNorm2 * rNorm2);
  const annulusAccel = 0.76 + 0.24 * smoothstep(0, 0.65, zr);
  out.vz +=
    (q / areaCm2 / 100) * annulusAccel * radialCore * jetProfile(rho, rHalf, zr) * axial(zr);
  // Subtle transmitral/tricuspid vortex-ring roll-up during E deceleration stays well inside the laminar
  // regime so healthy leaflets never trigger false green variance turbulence.
  let decelShear = 0;
  if (table === tables.mitralFlowMlps || table === tables.tricuspidFlowMlps) {
    const tm = tables.timings;
    const tBeat = (((phase % 1) + 1) % 1) * tables.rrS;
    const ePeakT = tm.mitralOpenS + tm.eAccelS;
    const eEndT = Math.min(tables.rrS, ePeakT + tm.eDecelS * 1.1);
    if (tBeat > ePeakT && tBeat < eEndT && tm.eDecelS > 0.02) {
      const u = Math.min(1, (tBeat - ePeakT) / tm.eDecelS);
      decelShear = 0.03 * Math.sin(Math.PI * u);
    }
  }
  // Post-stenotic turbulence blooms downstream of the leaflet tips while the proximal LA convergence
  // and annular acceleration neck remain laminar (unchanged when turbulence <= 0.08).
  const turbNow =
    Math.min(turbulence, 0.08) + Math.max(0, turbulence - 0.08) * smoothstep(0.35, 0.95, zr);
  out.dispersion = Math.max(out.dispersion, turbNow + decelShear + 0.05 * Math.min(1, rho / r));
  out.present = 1;
}

/** Time constant (s) over which the diastolic LV vortex ring loses its circulation in diastasis (Fase C). */
export const LV_VORTEX_DECAY_S = 0.22;
/** The same for the smaller ring the atrial contraction rolls up: a shorter, weaker jet. */
const LV_VORTEX_A_DECAY_S = 0.12;
/** Time constant (s) of the ring's travel from the leaflet tips toward the mid cavity. */
const LV_VORTEX_TRAVEL_S = 0.11;

/**
 * Circulation of the diastolic LV vortex ring (0–1 of its peak) at time `t` (s) of the beat, and how far its core has
 * travelled from the leaflet tips toward the mid cavity (0–1). The ring rolls up at the edge of the early filling jet
 * from the middle of E acceleration, is complete in mid deceleration, then loses its circulation through diastasis; the
 * atrial contraction rolls up a second, weaker one (`aShare` of the first); by the QRS what is left has been handed to
 * the outflow tract (Kilner et al., Nature 2000;404:759-761; Pedrizzetti et al., Nat Rev Cardiol 2014;11:545-553).
 */
export function lvVortexState(
  tm: BeatTables['timings'],
  rrS: number,
  t: number,
  aShare: number,
): { strength: number; travel: number } {
  const tt = t < tm.ejectionStartS ? t + rrS : t;
  const t0 = tm.mitralOpenS + 0.35 * tm.eAccelS;
  const t1 = tm.mitralOpenS + tm.eAccelS + 0.55 * tm.eDecelS;
  if (!(tt > t0) || !(t1 > t0)) return { strength: 0, travel: 0 };
  let s = smoothstep(t0, t1, tt) * Math.exp(-Math.max(0, tt - t1) / LV_VORTEX_DECAY_S);
  if (tm.hasAWave && aShare > 0 && Number.isFinite(tm.aStartS)) {
    const a0 = tm.aStartS < t0 ? tm.aStartS + rrS : tm.aStartS;
    const a1 = a0 + 0.6 * Math.max(0.02, tm.aEndS - tm.aStartS);
    s = Math.max(
      s,
      aShare * smoothstep(a0, a1, tt) * Math.exp(-Math.max(0, tt - a1) / LV_VORTEX_A_DECAY_S),
    );
  }
  // gone by the QRS: the isovolumic contraction and the ejection carry their own outflow, and a lingering low-velocity
  // outflow-tract stream would run into the ejection's CW/PW envelope and stretch its mean gradient
  s *= 1 - smoothstep(rrS - 0.06, rrS, tt);
  return {
    strength: Math.max(0, Math.min(1, s)),
    travel: 1 - Math.exp(-(tt - t0) / LV_VORTEX_TRAVEL_S),
  };
}

/**
 * Physiological LV diastolic vortex (Fase C; Kilner, Pedrizzetti, Sengupta): a ring rolled up at the shear layer of the
 * transmitral jet, with a Lamb–Oseen core, that travels from the leaflet tips toward the mid cavity and keeps turning
 * through diastasis — inward behind the core, apically along the jet's flank, outward ahead of it and back toward the base
 * along the walls. Asymmetric as the anterior leaflet makes it: the lobe under it, toward the outflow tract, is the larger
 * and stronger and carries the returning blood along the anteroseptal wall toward the aortic valve (`avAxis`), priming the
 * outflow tract for ejection. Inside the jet's core (the leaflet-tip PW plane, the colour M-mode line) the ring adds
 * nothing, so E, A, DT, VTI and Vp stay exact.
 */
export function sampleLvDiastolicVortex(
  tables: BeatTables,
  phase: number,
  x: number,
  y: number,
  z: number,
  zAnn: number,
  lengthNow: number,
  mvCenter: { x: number; y: number; z: number },
  avCenter: { x: number; y: number; z: number },
  avAxis: { x: number; y: number; z: number },
  ePeakMps: number,
  ePrimeSeptalCmps: number,
  out: FlowSample,
  mvAreaCm2 = 4,
  aPeakMps = 0,
): void {
  const zr = z - zAnn;
  if (zr < 0.1 || zr > lengthNow - 0.25) return;

  const rr = tables.rrS;
  const t = (((phase % 1) + 1) % 1) * rr;
  const aShare = ePeakMps > 0 ? Math.min(0.75, 0.6 * (aPeakMps / ePeakMps)) : 0;
  const { strength, travel } = lvVortexState(tables.timings, rr, t, aShare);
  if (strength < 0.02) return;

  // Unit horizontal axis from mitral inflow center toward the anterior/septal LVOT (PLAX / A3C / A5C plane)
  const ox = avCenter.x - mvCenter.x;
  const oy = avCenter.y - mvCenter.y;
  const oLen = Math.hypot(ox, oy) || 1;
  const ux = ox / oLen;
  const uy = oy / oLen;

  const dx = x - mvCenter.x;
  const dy = y - mvCenter.y;
  const sOut = dx * ux + dy * uy;
  const sPerp = -dx * uy + dy * ux;
  const zeta = zr / Math.max(4, lengthNow);
  // weaker and shorter-lived where relaxation is impaired (low e′)
  const vPeak =
    Math.min(0.44, 0.36 * (ePeakMps / 0.8)) *
    Math.min(1.15, Math.max(0.45, ePrimeSeptalCmps / 9));

  let added = false;
  let lvotShare = 0;

  // 1. Anteroseptal / LVOT basal return limb (strictly anterior to the open anterior mitral leaflet, sOut > 0.85 cm)
  const wPerp = Math.exp(-(sPerp * sPerp) / (2 * 1.35 * 1.35));
  if (sOut > 0.85 && sOut < 3.4 && wPerp >= 0.04 && zeta > 0.03 && zeta < 0.86) {
    const wSide = smoothstep(0.85, 1.45, sOut) * fadeOut(sOut, 2.4, 3.4);
    const wZ = smoothstep(0.03, 0.15, zeta) * fadeOut(zeta, 0.62, 0.86);
    lvotShare = wSide * wZ * wPerp;
    const revBlend = strength * lvotShare;
    const vRet = vPeak * revBlend;
    if (vRet > 0.015) {
      // Anterior mitral leaflet separates the LVOT corridor from transmitral inflow:
      // smoothly redirect anteroseptal outflow streamlines along avAxis toward the aortic valve.
      const keep = 1 - Math.min(0.92, revBlend * 1.15);
      out.vz *= keep;
      out.vx += avAxis.x * vRet;
      out.vy += avAxis.y * vRet;
      out.vz += avAxis.z * vRet;
      out.dispersion = Math.min(out.dispersion, 0.09);
      added = true;
    }
  }

  // 2. The ring: Lamb–Oseen core at the jet's shear layer, travelling apically, larger and stronger toward the LVOT
  const re = Math.sqrt(Math.max(0.5, mvAreaCm2) / Math.PI);
  const rho = Math.hypot(dx, dy);
  const mJet = smoothstep(0.8 * re, 1.15 * re, rho);
  if (mJet > 0) {
    const wAnt = 0.5 + 0.5 * (sOut / rho);
    const rhoC = re + 0.3;
    const zNear = 0.9;
    const zFar = Math.max(zNear + 0.5, Math.min(4.2, 0.48 * lengthNow));
    const zc = zNear + (zFar - zNear) * travel;
    const core = 0.45 + 0.4 * wAnt;
    const dRho = rho - rhoC;
    const dZ = zr - zc;
    const d = Math.hypot(dRho, dZ);
    if (d > 1e-4 && d < 3.4) {
      const sc = d / core;
      // Lamb–Oseen azimuthal velocity, 1 at its peak (sc = 1.121)
      const lamb = (1 - Math.exp(-sc * sc)) / sc / 0.6382;
      const v =
        vPeak *
        strength *
        (0.35 + 0.65 * wAnt) *
        lamb *
        mJet *
        fadeOut(d, 2.4, 3.4) *
        fadeOut(zr, 0.8 * lengthNow, lengthNow - 0.25);
      if (v > 0.01) {
        // inward behind the core, apical on its inner side, outward ahead, basal on its outer side;
        // partition cleanly with the LVOT redirection limb so basal return in the anteroseptal corridor is not double-counted
        const vRho = ((v * dZ) / d) * (1 - 0.55 * lvotShare);
        const vzScale = dRho > 0 ? 1 - lvotShare : 1;
        out.vx += (vRho * dx) / rho;
        out.vy += (vRho * dy) / rho;
        out.vz += ((-v * dRho) / d) * vzScale;
        added = true;
      }
    }
  }

  if (added) {
    out.dispersion = Math.max(out.dispersion, 0.06);
    out.present = 1;
  }
}

/**
 * Conventional colour M-mode slope (cm/s) of an inflow wave travelling at `waveCmps`, on the inflow axis at a fixed
 * position as an M-mode line sees it: from the leaflet tips at valve opening to 4 cm beyond them, the first time the
 * velocity reaches half the early-filling maximum at the tips, fitted against depth.
 */
export function conventionalPropagation(
  tables: BeatTables,
  mapseCm: number,
  lvLengthCm: number,
  mvAreaCm2: number,
  waveCmps: number,
): number {
  const tm = tables.timings;
  const velocity = (z: number, t: number): number => {
    const phase = t / tables.rrS;
    const zAnn = mapseCm * sampleTable(tables.longitudinal, phase);
    const zr = z - zAnn;
    if (zr < 0) return 0;
    return (
      (inflowFlow(tables, phase, zr, waveCmps) / mvAreaCm2 / 100) *
      inflowAxial(zr, lvLengthCm - zAnn)
    );
  };
  const t0 = tm.mitralOpenS;
  const eEnd = Math.min(
    tables.rrS,
    t0 + 0.5,
    tm.hasAWave ? Math.max(tm.aStartS, t0 + tm.eAccelS + 0.05) : Infinity,
  );
  const tEnd = tables.rrS;
  const tips = mapseCm * sampleTable(tables.longitudinal, t0 / tables.rrS) + INFLOW_CORE_CM;
  let vMax = 0;
  for (let t = t0; t < eEnd; t += 0.001) vMax = Math.max(vMax, velocity(tips, t));
  let n = 0,
    st = 0,
    sd = 0,
    stt = 0,
    std = 0;
  for (let d = 0; d <= 4.0001; d += 0.25) {
    let reached = -1;
    for (let t = t0; t < tEnd; t += 0.001)
      if (velocity(tips + d, t) >= 0.5 * vMax) {
        reached = t;
        break;
      }
    if (reached < 0) return Number.NaN;
    n++;
    st += reached;
    sd += d;
    stt += reached * reached;
    std += reached * d;
  }
  return (n * std - st * sd) / (n * stt - st * st);
}

/** Filling-wave speed (cm/s) at which the conventional slope reads `targetCmps`; the fastest tried when none is fast enough. */
export function solveInflowWave(
  tables: BeatTables,
  mapseCm: number,
  lvLengthCm: number,
  mvAreaCm2: number,
  targetCmps: number,
): number {
  let lo = Math.log(5),
    hi = Math.log(5000);
  const reads = (logSpeed: number): number =>
    conventionalPropagation(tables, mapseCm, lvLengthCm, mvAreaCm2, Math.exp(logSpeed));
  if (!(reads(hi) > targetCmps)) return Math.exp(hi);
  for (let it = 0; it < 24; it++) {
    const mid = 0.5 * (lo + hi);
    if (reads(mid) > targetCmps) hi = mid;
    else lo = mid;
  }
  return Math.exp(0.5 * (lo + hi));
}

/**
 * Pulmonary venous velocities (spec 63): S follows the annular descent (LA reservoir), blunts in
 * moderate MR (`0 < S < D`), and reverses holosystolically in severe MR (`mrEro >= 0.35 cm²`, ASE
 * severe MR criterion); D mirrors the mitral E wave (conduit) and Ar is the atrial reversal (absent in AF).
 */
export function pulmonaryVeinPeaks(c: CaseDefinition): {
  sMps: number;
  dMps: number;
  arMps: number;
} {
  const mrEro = c.hemodynamics.regurgitation.mr?.eroaCm2 ?? 0;
  const sBase = 0.55 * Math.min(1.3, c.physiology.mapseCm / 1.3);
  const s =
    mrEro <= 0.3
      ? sBase * (1 - 0.85 * (mrEro / 0.32))
      : sBase * (1 - 0.85 * (0.3 / 0.32)) - 0.52 * Math.min(1.2, (mrEro - 0.3) / 0.15);
  const d = 0.58 * (c.physiology.ePeakMps / 0.8);
  const ar =
    c.rhythm.type === 'atrial-fibrillation' || c.physiology.aPeakMps <= 0
      ? 0
      : 0.22 + 0.12 * Math.min(1, (c.physiology.aPeakMps - 0.4) / 0.5);
  return { sMps: s, dMps: d, arMps: ar };
}
