import { measureFlowPeaks, type FlowPeaks } from './flowPeaks';
import { aWaveShape, TRICUSPID_LAG_S, type CycleTimings } from './timing';

/**
 * Per-beat tabulated model of LV volume and transvalvular flows. Built from the case physiology so
 * that Doppler velocities, volumes, valve motion and measurements share ONE source of truth.
 * Volumes in mL, flows in mL/s, time in s, phase in [0,1).
 */
export interface BeatTables {
  n: number;
  timings: CycleTimings;
  rrS: number;
  edvMl: number;
  esvMl: number;
  strokeVolumeMl: number;
  lvVolumeMl: Float32Array; // V(φ)
  aorticFlowMlps: Float32Array; // Q_ao(φ) ≥ 0 during ejection
  /** Q_pv(φ) ≥ 0: right ventricular ejection, with the case's pulmonary acceleration time (decision 105). */
  pulmonaryFlowMlps: Float32Array;
  /** Acceleration time (s) of the right ventricular ejection. */
  pulmonaryAccelerationS: number;
  mitralFlowMlps: Float32Array; // Q_mv(φ) ≥ 0 during filling
  /** Q_tv(φ) ≥ 0: tricuspid inflow, with its own early wave and effective flow area. */
  tricuspidFlowMlps: Float32Array;
  /** Tricuspid filling over the beat (mL): subtract quantified TR before chaining pulmonary forward ejection. */
  tricuspidFillMl: number;
  /** Own effective inflow area: accounts for forward flow and quantified TR, not mitral regurgitation. */
  tvEffectiveAreaCm2: number;
  pulmonaryStrokeVolumeMl: number;
  /** Integrated inflow minus pulmonary and TR outflow, relative to the beat start; not an absolute geometric RV volume. */
  rvVolumeChangeMl: Float32Array;
  rvEndVolumeChangeMl: number;
  trVelocityMps: Float32Array;
  trFlowMlps: Float32Array;
  mvEffectiveAreaCm2: number; // solved so that ∫Q_mv = SV with the requested E and A peak velocities
  /** Regurgitant flows (mL/s) through the mitral (systole) and aortic (diastole) valves; zero when absent. */
  mrFlowMlps: Float32Array;
  /** Vena contracta velocity (m/s) of the MR jet where it flows, zero elsewhere: the flow over the orifice open at that time. */
  mrVelocityMps: Float32Array;
  arFlowMlps: Float32Array;
  regurgitation: {
    mrVolumeMl: number;
    mrVmaxMps: number;
    mrVtiCm: number;
    arVolumeMl: number;
    arVmaxMps: number;
    arVtiCm: number;
    arPhtMs: number;
    trVolumeMl: number;
    trVtiCm: number;
    trVmaxMps: number;
  };
  /**
   * Volume (mL) the closing correction removed so that V(RR) = V(0): inflow minus outflow over the beat. With every flow
   * normalised on this table it is discretisation only (decision 95); `validateCase` rejects a case that needs more.
   */
  volumeCorrectionMl: number;
  /** Volume (mL) and annular displacements when the beat ends: where the next beat of atrial fibrillation starts. */
  endVolumeMl: number;
  /**
   * End-systole (s): the least left ventricular volume, never before aortic closure (decision 245). Without mitral
   * regurgitation it is aortic closure; with it the ventricle keeps emptying into the atrium until the mitral valve opens.
   * The smallest cavity is the end-systolic frame of the chamber quantification guideline (ASE/EACVI 2015).
   */
  endSystoleS: number;
  endLongitudinal: number;
  endRvLongitudinal: number;
  /** Longitudinal (annular) displacement toward apex as a fraction of MAPSE, [0,1]. */
  longitudinal: Float32Array;
  /** Longitudinal annular velocity in units of MAPSE per second (s⁻¹); multiply by MAPSE(cm) → cm/s. */
  longitudinalVelocity: Float32Array;
  /** Tricuspid annular displacement toward the apex as a fraction of TAPSE, and its velocity (s⁻¹) (decision 106). */
  rvLongitudinal: Float32Array;
  rvLongitudinalVelocity: Float32Array;
  /** Inferior vena cava collapse (fraction of its diameter) when the beat starts and ends, while breathing (decision 113). */
  ivcCollapse: readonly [number, number] | null;
}

export interface BeatOptions {
  /** Preload scaling: scales SV and E peak. 1 = nominal. */
  preloadFactor?: number;
  aWave?: boolean;
  n?: number;
  /**
   * A beat chained to the one before it (decision 107, atrial fibrillation; decision 108, free breathing): the volume each
   * ventricle ejects (what filled it in the diastole before), the mitral flow area of the case, the RR before it, where the
   * annuli were when that beat ended, and the factors on the early inflow waves of this beat.
   */
  chain?: ChainedBeat;
  /**
   * The outflow tract (decision 245): its area and whether its obstruction is dynamic (SAM). With it the left ventricular
   * pressure that drives the mitral regurgitation adds a subaortic gradient; without it, only a valvular one.
   */
  outflow?: { lvotAreaCm2: number; dynamicObstruction: boolean };
}

export interface ChainedBeat {
  ejectMl: number;
  /** Pulmonary forward ejection (mL): prior tricuspid filling minus quantified TR. */
  rvEjectMl?: number;
  mvAreaCm2: number;
  tvAreaCm2: number;
  previousRrS: number;
  startLongitudinal: number;
  startRvLongitudinal: number;
  /** Factors on the mitral and tricuspid E waves (respiration, decision 108); 1 when omitted. */
  mitralEFactor?: number;
  tricuspidEFactor?: number;
  /** Inferior vena cava collapse when the beat starts and when it ends (respiration, decision 113). */
  ivcCollapse?: readonly [number, number];
}

/**
 * Pulmonary acceleration time (s) for a mean pulmonary artery pressure (mmHg), inverting the Doppler regressions: Dabestani
 * et al. (Am J Cardiol 1987; 59:662–668) mPAP = 79 − 0.45·AcT for AcT ≥ 120 ms and Mahan's mPAP = 90 − 0.62·AcT for
 * AcT < 90 ms, joined linearly between them. A normal mean pressure of 17 mmHg gives 137 ms (normal 136–153 ms).
 */
export function pulmonaryAccelerationTimeS(mpapMmHg: number): number {
  const at120 = 79 - 0.45 * 120;
  const at90 = 90 - 0.62 * 90;
  if (mpapMmHg <= at120) return (79 - mpapMmHg) / 0.45 / 1000;
  if (mpapMmHg >= at90) return Math.max(40, (90 - mpapMmHg) / 0.62) / 1000;
  return (120 - (30 * (mpapMmHg - at120)) / (at90 - at120)) / 1000;
}

/** Mean pulmonary artery pressure (mmHg) from the systolic one: Chemla et al. (Chest 2004; 126:1313–1317), 0.61·sPAP + 2. */
export function meanPulmonaryPressureMmHg(systolicMmHg: number): number {
  return 0.61 * systolicMmHg + 2;
}

export function sampleTable(table: Float32Array, phase: number): number {
  const n = table.length;
  const x = ((phase % 1) + 1) % 1;
  const f = x * n - 0.5;
  const i0 = Math.floor(f);
  const t = f - i0;
  const a = table[((i0 % n) + n) % n] ?? 0;
  const b = table[(((i0 + 1) % n) + n) % n] ?? 0;
  return a + (b - a) * t;
}

/** Kinematic state of the heart at a given phase, derived from the beat tables. */
export interface CycleState {
  phase: number;
  timeInBeatS: number;
  rrS: number;
  lvVolumeMl: number;
  /** 0 at end-diastole, 1 at end-systole (volume based). */
  contraction: number;
  /** Mitral valve opening 0..1 (driven by inflow). */
  mvOpen: number;
  /** Aortic valve opening 0..1 (driven by ejection flow). */
  avOpen: number;
  /**
   * Time since the aortic valve opened, in ejection times: 0 when it opens, 1 when it closes, negative before and above 1
   * after, to the end of the beat. The outflow's drag on the anterior mitral leaflet (SAM, `samDrag`) follows it, along
   * the same timeline as the Doppler's dynamic obstruction. Absent from hand-built states, which take no SAM.
   */
  ejectionProgress?: number;
  /** Tricuspid / pulmonic openings mirror the left side with a small delay. */
  tvOpen: number;
  pvOpen: number;
  /** Longitudinal annular displacement fraction of MAPSE (0 = end diastole position). */
  longitudinal: number;
  /** Tricuspid annular displacement fraction of TAPSE (decision 106). */
  rvLongitudinal: number;
  /** Inferior vena cava collapse this frame while breathing freely (decision 113); the patient state's when absent. */
  ivcCollapse?: number;
  /** Atrial contraction 0..1 (0 in AF). */
  atrialContraction: number;
  /** 1 from the end of the A wave until ejection starts (the atria stay at their minimal volume), 0 in AF. */
  atrialHold: number;
  mitralFlowMlps: number;
  aorticFlowMlps: number;
  edvMl: number;
  esvMl: number;
  /**
   * Arterial pressure in the descending aorta behind the left atrium, as a fraction of the pulse pressure above the
   * diastolic one (decision 272): 0 at the foot of the pulse, 1 at the systolic peak.
   */
  aorticPressure: number;
}

/**
 * Transit time (s) of the pulse from the aortic valve to the descending aorta behind the left atrium: some 20 cm of
 * ascending aorta and arch at the aortic pulse wave velocity of a young adult (carotid–femoral 6.2 m/s under 30 years,
 * Reference Values for Arterial Stiffness' Collaboration, Eur Heart J 2010;31:2338-2350). The path length is an assumption
 * and the delay does not grow shorter with age, as the stiffer aorta of the old would make it.
 */
const DESC_AORTA_PULSE_DELAY_S = 0.032;
/**
 * Time constant (s) of the diastolic fall of arterial pressure (the windkessel's resistance × compliance), an assumed
 * typical adult value: the pressure decays from the dicrotic notch toward the next upstroke, which starts from the
 * diastolic pressure.
 */
const ARTERIAL_DECAY_TAU_S = 1.5;

/**
 * Opening of the atrioventricular leaflets between the filling waves, as a fraction of their opening at peak inflow. After
 * the early wave the leaflets float back to a semi-closed position and stay there until atrial contraction reopens them
 * (the F point of the mitral M-mode): at the end of diastasis the orifice is about half its size at peak E (Govindarajan
 * et al., Sci Rep 2018; 8:6187), and the anterior leaflet takes 144 ± 19 ms from the E point to that nadir (Park et al.,
 * Diagnostics 2023; 13:2412). An opening that followed the inflow alone closed the valve for 95 ms at 65 bpm (decision 100).
 */
export const DIASTASIS_OPENING = 0.5;

/** Time (s) the leaflets take to close from diastasis at the onset of systole when no atrial contraction closes them first. */
export const SYSTOLIC_CLOSURE_S = 0.03;

/**
 * Aortic pressure over ejection as a fraction of the pulse pressure above the diastolic one (decision 245): up to the
 * systolic peak at 0.4 of ejection, then down to 0.55 at the dicrotic notch — a shape assumed from the textbook waveform,
 * without a published table behind its numbers.
 */
export function aorticPulse(u: number): number {
  if (u <= 0) return 0;
  if (u < 0.4) {
    const s = Math.sin((Math.PI / 2) * (u / 0.4));
    return s * s;
  }
  return 1 - 0.45 * Math.pow(Math.min(1, (u - 0.4) / 0.6), 1.5);
}

/**
 * Pressure in the descending aorta at time `t` (s) of the beat, as a fraction of the pulse pressure (decision 272): the
 * aortic pulse over ejection, then an exponential diastolic decay from the dicrotic notch that reaches the diastolic
 * pressure as the next ejection starts, delayed by the transit of the pulse along the arch.
 */
export function aorticPressureFraction(tables: BeatTables, t: number): number {
  const tm = tables.timings;
  const rr = tables.rrS;
  const et = Math.max(1e-3, tm.ejectionEndS - tm.ejectionStartS);
  const s = (((t - DESC_AORTA_PULSE_DELAY_S - tm.ejectionStartS) % rr) + rr) % rr;
  if (s < et) return aorticPulse(s / et);
  const diastole = Math.max(1e-3, rr - et);
  const tail = Math.exp(-diastole / ARTERIAL_DECAY_TAU_S);
  return (aorticPulse(1) * (Math.exp(-(s - et) / ARTERIAL_DECAY_TAU_S) - tail)) / (1 - tail);
}

/**
 * Leaflet opening of an atrioventricular valve at phase p: it follows the inflow while that opens it wider than
 * DIASTASIS_OPENING, floats at DIASTASIS_OPENING from peak early inflow until atrial contraction peaks, then closes with
 * the end of the A wave — or, without one, in the first SYSTOLIC_CLOSURE_S of systole.
 */
function inflowOpening(
  tables: BeatTables,
  p: number,
  qMax: number,
  table: Float32Array = tables.mitralFlowMlps,
  side: 'mitral' | 'tricuspid' = 'mitral',
): number {
  const x = ((p % 1) + 1) % 1;
  const flow = Math.min(1, Math.pow(Math.max(0, sampleTable(table, x)) / qMax, 0.6));
  const tm = tables.timings;
  const t = x * tables.rrS;
  // the tricuspid valve opens at its own time and closes TRICUSPID_LAG_S after the mitral one (T1 after M1, decision 162)
  const openS = side === 'tricuspid' ? tm.tricuspidOpenS : tm.mitralOpenS;
  // closing runs on a clock TRICUSPID_LAG_S late: before it reaches zero the valve is still in the previous beat's end
  const tc = t - (side === 'tricuspid' ? TRICUSPID_LAG_S : 0);
  const afterE = t >= openS + tm.eAccelS;
  let floor = 0;
  if (tm.hasAWave) {
    const tw = ((tc % tables.rrS) + tables.rrS) % tables.rrS;
    const u = (tw - tm.aStartS) / (tm.aEndS - tm.aStartS);
    if (afterE && u < 0.5) floor = DIASTASIS_OPENING;
    else if (u >= 0.5 && u < 1) floor = DIASTASIS_OPENING * aWaveShape(u);
  } else if (afterE || tc < 0) floor = DIASTASIS_OPENING;
  else if (tc < SYSTOLIC_CLOSURE_S) floor = DIASTASIS_OPENING * (1 - tc / SYSTOLIC_CLOSURE_S);
  return Math.max(flow, floor);
}

/**
 * Instants (s from the start of the beat) at which each valve opens and closes, [open, close]: the aortic valve with the
 * start and end of left ventricular ejection, the pulmonary valve with the right ventricle's (before and after it), the
 * mitral valve at the end of isovolumic relaxation and with the end of atrial contraction (or once closed early in systole
 * without one), the tricuspid valve before the mitral one and closing after it (decision 162).
 */
export function valveEventTimes(tables: BeatTables): {
  mitral: [number, number];
  aortic: [number, number];
  tricuspid: [number, number];
  pulmonary: [number, number];
} {
  const tm = tables.timings;
  const mitral: [number, number] = [tm.mitralOpenS, tm.hasAWave ? tm.aEndS : SYSTOLIC_CLOSURE_S];
  return {
    mitral,
    aortic: [tm.ejectionStartS, tm.ejectionEndS],
    tricuspid: [tm.tricuspidOpenS, mitral[1] + TRICUSPID_LAG_S],
    pulmonary: [tm.pulmonaryOpenS, tm.pulmonaryCloseS],
  };
}

export function cycleStateAt(
  tables: BeatTables,
  phase: number,
  peaks: FlowPeaks = measureFlowPeaks(tables),
): CycleState {
  const p = ((phase % 1) + 1) % 1;
  const vol = sampleTable(tables.lvVolumeMl, p);
  const qmv = sampleTable(tables.mitralFlowMlps, p);
  const qao = sampleTable(tables.aorticFlowMlps, p);
  const { mitral: qmvMax, tricuspid: qtvMax, aortic: qaoMax, pulmonary: qpvMax } = peaks;
  const t = p * tables.rrS;
  const tm = tables.timings;
  let atrial = 0;
  if (tm.hasAWave && t > tm.aStartS && t < tm.aEndS)
    atrial = Math.sin((Math.PI * (t - tm.aStartS)) / (tm.aEndS - tm.aStartS));
  const atrialHold = tm.hasAWave && (t >= tm.aEndS || t < tm.ejectionStartS) ? 1 : 0;
  const mvOpen = inflowOpening(tables, p, qmvMax);
  const avOpen = Math.min(1, Math.pow(qao / qaoMax, 0.5));
  return {
    phase: p,
    timeInBeatS: t,
    rrS: tables.rrS,
    lvVolumeMl: vol,
    contraction: (tables.edvMl - vol) / Math.max(tables.edvMl - tables.esvMl, 1e-6),
    mvOpen,
    avOpen,
    ejectionProgress: (t - tm.ejectionStartS) / Math.max(1e-3, tm.ejectionEndS - tm.ejectionStartS),
    tvOpen: inflowOpening(tables, p, qtvMax, tables.tricuspidFlowMlps, 'tricuspid'),
    pvOpen: Math.min(1, Math.pow(sampleTable(tables.pulmonaryFlowMlps, p) / qpvMax, 0.5)),
    longitudinal: sampleTable(tables.longitudinal, p),
    rvLongitudinal: sampleTable(tables.rvLongitudinal, p),
    // linear across the beat, from where the breath had it when the beat started to where it has it when the beat ends
    ivcCollapse: tables.ivcCollapse
      ? tables.ivcCollapse[0] + (tables.ivcCollapse[1] - tables.ivcCollapse[0]) * p
      : undefined,
    atrialContraction: atrial,
    atrialHold,
    mitralFlowMlps: qmv,
    aorticFlowMlps: qao,
    edvMl: tables.edvMl,
    esvMl: tables.esvMl,
    aorticPressure: aorticPressureFraction(tables, t),
  };
}
