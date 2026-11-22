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
 * Shear layer of a regurgitant or stenotic jet: 1 mm wide at its orifice, growing 0.05 cm per cm.
 */
export const FAST_JET_SHEAR = { orifice: 0.1, growth: 0.05 } as const;
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
    const d = Math.hypot(rho, zr);
    if (d >= FUNNEL_FADE_TO_CM) return;
    const q = inflowFlow(tables, phase, 0, waveCmps, table);
    if (q <= 1) return;
    const mag =
      (q / areaCm2 / 100) *
      Math.min(1, (re * re) / (2 * d * d)) *
      fadeOut(d, FUNNEL_FADE_FROM_CM, FUNNEL_FADE_TO_CM) *
      convergenceWeight(rho, -zr, r);
    const sx = -dx,
      sy = -dy,
      sz = r - zr;
    const n = Math.hypot(sx, sy, sz);
    out.vx += (sx / n) * mag;
    out.vy += (sy / n) * mag;
    out.vz += (sz / n) * mag;
    out.dispersion = Math.max(out.dispersion, turbulence);
    out.present = 1;
    return;
  }
  const rHalf = re + Math.min(4, Math.max(0, zr - INFLOW_CORE_CM)) * 0.35;
  if (rho >= jetReach(rHalf, zr)) return;
  const q = inflowFlow(tables, phase, Math.hypot(zr, rho), waveCmps, table);
  if (q <= 1) return;
  out.vz += (q / areaCm2 / 100) * jetProfile(rho, rHalf, zr) * axial(zr);
  // Subtle transmitral/tricuspid vortex-ring roll-up during E deceleration mildly feathers the inner flank
  // (toward baseline only) while leaving E-wave acceleration and peak razor-focused.
  let decelShear = 0;
  if (table === tables.mitralFlowMlps || table === tables.tricuspidFlowMlps) {
    const tm = tables.timings;
    const tBeat = (((phase % 1) + 1) % 1) * tables.rrS;
    const ePeakT = tm.mitralOpenS + tm.eAccelS;
    const eEndT = Math.min(tables.rrS, ePeakT + tm.eDecelS * 1.1);
    if (tBeat > ePeakT && tBeat < eEndT && tm.eDecelS > 0.02) {
      const u = Math.min(1, (tBeat - ePeakT) / tm.eDecelS);
      decelShear = 0.065 * Math.sin(Math.PI * u);
    }
  }
  out.dispersion = Math.max(out.dispersion, turbulence + decelShear + 0.15 * Math.min(1, rho / r));
  out.present = 1;
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
 * Pulmonary venous velocities (spec 63): S follows the annular descent (LA reservoir) and is blunted by
 * mitral regurgitation, D mirrors the mitral E wave (conduit) and Ar is the atrial reversal (absent in AF).
 */
export function pulmonaryVeinPeaks(c: CaseDefinition): {
  sMps: number;
  dMps: number;
  arMps: number;
} {
  const mrEro = c.hemodynamics.regurgitation.mr?.eroaCm2 ?? 0;
  const s = 0.55 * Math.min(1.3, c.physiology.mapseCm / 1.3) * (1 - 0.9 * Math.min(1, mrEro / 0.4));
  const d = 0.58 * (c.physiology.ePeakMps / 0.8);
  const ar =
    c.rhythm.type === 'atrial-fibrillation' || c.physiology.aPeakMps <= 0
      ? 0
      : 0.22 + 0.12 * Math.min(1, (c.physiology.aPeakMps - 0.4) / 0.5);
  return { sMps: Math.max(0, s), dMps: d, arMps: ar };
}
