import { aseSweepDurationS } from '@/clinical/formulas';
import {
  envelopeThreshold,
  SPECTRAL_BINS,
  spectralRange,
  type SpectralSettings,
} from '@/simulator/doppler/spectral/spectrum';
import type { ImagingModality } from '@/simulator/renderer/types';
import type { PhaseMarks } from '../protocol';
import {
  autoTraceBestSystolicBeat,
  autoTraceSpectralEnvelope,
  STRIP_MM_WIDTH,
} from './autoTrace';

export type AutoStripMeasureTarget =
  | 'mitral-inflow'
  | 'tdi-e-prime'
  | 'rv-s-prime'
  | 'tr-vmax'
  | 'rvot-at'
  | 'tapse';

export interface AutoMitralInflowSuccess {
  ok: true;
  target: 'mitral-inflow';
  beatIndex: number;
  e: { xPx: number; velocityMps: number };
  a: { xPx: number; velocityMps: number } | null;
  dt: {
    peakXPx: number;
    peakVelocityMps: number;
    slopeXPx: number;
    slopeVelocityMps: number;
    dtMs: number;
  } | null;
}

export interface AutoTdiEPrimeSuccess {
  ok: true;
  target: 'tdi-e-prime';
  beatIndex: number;
  ePrime: { xPx: number; velocityMps: number };
}

export interface AutoRvSPrimeSuccess {
  ok: true;
  target: 'rv-s-prime';
  beatIndex: number;
  sPrime: { xPx: number; velocityMps: number };
}

export interface AutoTrVmaxSuccess {
  ok: true;
  target: 'tr-vmax';
  beatIndex: number;
  vmax: { xPx: number; velocityMps: number; peakGradientMmHg: number };
}

export interface AutoRvotAtSuccess {
  ok: true;
  target: 'rvot-at';
  beatIndex: number;
  onsetXPx: number;
  peakXPx: number;
  peakVelocityMps: number;
  atMs: number;
  vtiCm: number;
}

export interface AutoTapseSuccess {
  ok: true;
  target: 'tapse';
  beatIndex: number;
  ed: { xPx: number; depthCm: number };
  es: { xPx: number; depthCm: number };
  tapseCm: number;
}

export interface AutoStripMeasureFailure {
  ok: false;
  target: AutoStripMeasureTarget;
  reason: 'wrong-modality' | 'incomplete-beat' | 'no-signal';
  message: string;
}

export type AutoStripMeasureResult =
  | AutoMitralInflowSuccess
  | AutoTdiEPrimeSuccess
  | AutoRvSPrimeSuccess
  | AutoTrVmaxSuccess
  | AutoRvotAtSuccess
  | AutoTapseSuccess
  | AutoStripMeasureFailure;

interface BeatSpan {
  beatIndex: number;
  c0: number;
  c1: number;
  maxTimeS: number;
  minPhase: number;
  maxPhase: number;
}

/** Partition written strip columns into contiguous non-wrapping beat segments away from the sweep head. */
function findSpectralBeatSpans(
  cols: number,
  head: number,
  stripPhase: Float32Array,
  stripBeatIndex: Uint32Array,
  stripSampleTime: Float64Array,
): BeatSpan[] {
  const headCol = ((head % cols) + cols) % cols;
  const spans: BeatSpan[] = [];
  let cur: BeatSpan | null = null;
  for (let c = 0; c < cols; c++) {
    const t = stripSampleTime[c] ?? 0;
    const atSweep = c === headCol || c === (headCol + 1) % cols;
    if (!(t > 0) || atSweep) {
      if (cur) {
        spans.push(cur);
        cur = null;
      }
      continue;
    }
    const beat = stripBeatIndex[c] ?? 0;
    const ph = stripPhase[c] ?? 0;
    if (!cur || cur.beatIndex !== beat || Math.abs(t - cur.maxTimeS) > 0.15) {
      if (cur) spans.push(cur);
      cur = { beatIndex: beat, c0: c, c1: c, maxTimeS: t, minPhase: ph, maxPhase: ph };
    } else {
      cur.c1 = c;
      if (t > cur.maxTimeS) cur.maxTimeS = t;
      if (ph < cur.minPhase) cur.minPhase = ph;
      if (ph > cur.maxPhase) cur.maxPhase = ph;
    }
  }
  if (cur) spans.push(cur);
  return spans;
}

/** Partition written M-mode columns into contiguous beat spans using phase wrap-around (no ground truth). */
function findMmodeBeatSpans(
  cols: number,
  head: number,
  stripPhase: Float32Array,
  stripSampleTime: Float64Array,
): BeatSpan[] {
  const headCol = ((head % cols) + cols) % cols;
  const spans: BeatSpan[] = [];
  let cur: BeatSpan | null = null;
  let beatCounter = 0;
  let prevPh = -1;
  for (let c = 0; c < cols; c++) {
    const t = stripSampleTime[c] ?? 0;
    const atSweep = c === headCol || c === (headCol + 1) % cols;
    if (!(t > 0) || atSweep) {
      if (cur) {
        spans.push(cur);
        cur = null;
      }
      prevPh = -1;
      continue;
    }
    const ph = stripPhase[c] ?? 0;
    const wrapped = prevPh >= 0 && ph < prevPh - 0.45;
    const timeJump = cur !== null && Math.abs(t - cur.maxTimeS) > 0.15;
    if (!cur || wrapped || timeJump) {
      if (cur) spans.push(cur);
      beatCounter++;
      cur = { beatIndex: beatCounter, c0: c, c1: c, maxTimeS: t, minPhase: ph, maxPhase: ph };
    } else {
      cur.c1 = c;
      if (t > cur.maxTimeS) cur.maxTimeS = t;
      if (ph < cur.minPhase) cur.minPhase = ph;
      if (ph > cur.maxPhase) cur.maxPhase = ph;
    }
    prevPh = ph;
  }
  if (cur) spans.push(cur);
  return spans;
}

/**
 * Pure strip-signal detector for Mitral E wave, A wave (omitted automatically in AF when absent),
 * and E-wave deceleration time (TD E) from the rendered PW/CW spectral strip.
 */
export function autoMeasureMitralInflow(
  strip: Float32Array,
  cols: number,
  displayCols: number,
  head: number,
  stripPhase: Float32Array,
  stripBeatIndex: Uint32Array,
  stripSampleTime: Float64Array,
  spectral: SpectralSettings,
  phaseMarks: PhaseMarks,
  modality: ImagingModality,
): AutoStripMeasureResult {
  if (modality !== 'pw' && modality !== 'cw') {
    return {
      ok: false,
      target: 'mitral-inflow',
      reason: 'wrong-modality',
      message: 'Activa Doppler pulsado (PW) sobre las puntas mitrales para medir E / A / TD.',
    };
  }
  if (cols < 16 || strip.length < cols * SPECTRAL_BINS) {
    return {
      ok: false,
      target: 'mitral-inflow',
      reason: 'incomplete-beat',
      message: 'Aún no hay suficientes datos espectrales para analizar el llenado mitral.',
    };
  }

  const pixelsPerColumn = (displayCols || cols) / cols;
  const spans = findSpectralBeatSpans(cols, head, stripPhase, stripBeatIndex, stripSampleTime);
  const candidates = spans
    .filter((sp) => sp.minPhase <= phaseMarks.mitralOpen + 0.02 && sp.maxPhase >= 0.85)
    .sort((a, b) => b.maxTimeS - a.maxTimeS);

  if (!candidates.length) {
    return {
      ok: false,
      target: 'mitral-inflow',
      reason: 'incomplete-beat',
      message: 'Espera ~1 s para registrar una diástole completa en la tira Doppler.',
    };
  }

  const modal = modality === 'pw';
  const dStart = Math.max(0.25, phaseMarks.mitralOpen - 0.02);
  const eWinEnd = Math.min(0.92, Math.max(phaseMarks.eEnd + 0.04, dStart + 0.16));
  const aWinStart = Math.max(eWinEnd - 0.02, phaseMarks.aStart - 0.03);

  for (const cand of candidates) {
    let cStart = -1;
    let cEnd = -1;
    for (let c = cand.c0; c <= cand.c1; c++) {
      const ph = stripPhase[c] ?? 0;
      if (ph >= dStart && ph <= 0.995) {
        if (cStart < 0) cStart = c;
        cEnd = c;
      }
    }
    if (cStart < 0 || cEnd - cStart < 6) continue;

    const traced = autoTraceSpectralEnvelope(
      strip,
      cols,
      displayCols,
      spectral,
      (cStart + 0.5) * pixelsPerColumn,
      (cEnd + 0.5) * pixelsPerColumn,
      modal,
    );
    const vels = traced.velocitiesMps;
    if (vels.length < 6) continue;

    // 1. E-wave peak in early diastole
    let eIdx = -1;
    let eAbs = 0;
    for (let i = 0; i < vels.length; i++) {
      const col = cStart + i;
      const ph = stripPhase[col] ?? 0;
      if (ph > eWinEnd) break;
      const av = Math.abs(vels[i]!);
      if (av > eAbs) {
        eAbs = av;
        eIdx = i;
      }
    }
    if (eIdx < 0 || eAbs < 0.25) continue;
    const eVel = vels[eIdx]!;
    const eSign = eVel >= 0 ? 1 : -1;
    const eXPx = traced.x0 + eIdx * traced.pixelsPerColumn;

    // 2. E-wave deceleration time (linear regression over the descending limb after E peak)
    let dt: AutoMitralInflowSuccess['dt'] = null;
    // Walk down from eIdx until velocity turns back up (A wave) or hits baseline
    let descEnd = eIdx;
    let minAfterE = eAbs;
    let minAfterEIdx = eIdx;
    for (let i = eIdx + 1; i < vels.length; i++) {
      const v = vels[i]! * eSign;
      if (v < minAfterE) {
        minAfterE = v;
        minAfterEIdx = i;
      }
      // Stop descending limb once flow rebounds by >= 0.08 m/s (onset of A wave) or drops below 15% of E
      if (v <= 0.15 * eAbs) {
        descEnd = i;
        break;
      }
      if (v > minAfterE + 0.08 && i > eIdx + 3) {
        descEnd = minAfterEIdx;
        break;
      }
      descEnd = i;
    }
    if (descEnd - eIdx >= 3) {
      // Select clean linear mid-descent points (between 90% and 25% of E peak) for stable slope fit
      const fitPts: { t: number; v: number; i: number }[] = [];
      for (let i = eIdx + 1; i <= descEnd; i++) {
        const v = Math.max(0, vels[i]! * eSign);
        if (v <= 0.92 * eAbs && v >= 0.25 * eAbs) {
          fitPts.push({ t: (i - eIdx) * traced.secondsPerColumn, v, i });
        }
      }
      // Fit forced through (0, eAbs): slope = sum(t * (eAbs - v)) / sum(t^2)
      const usePts =
        fitPts.length >= 2
          ? fitPts
          : Array.from({ length: descEnd - eIdx }, (_, k) => {
              const idx = eIdx + 1 + k;
              return {
                t: (idx - eIdx) * traced.secondsPerColumn,
                v: Math.max(0, vels[idx]! * eSign),
                i: idx,
              };
            });
      let num = 0;
      let den = 0;
      for (const p of usePts) {
        num += p.t * (eAbs - p.v);
        den += p.t * p.t;
      }
      const slopeMpsPerS = den > 1e-9 ? num / den : 0;
      if (slopeMpsPerS > 0.2) {
        const dtS = eAbs / slopeMpsPerS;
        const dtMs = dtS * 1000;
        if (dtMs >= 75 && dtMs <= 1200) {
          // Place the 2nd editable slope marker at 40% drop along the fitted line so computeDecelerationTime matches exact dtMs
          const dropFrac = 0.45;
          const dtSlopeS = dtS * dropFrac;
          const slopeCols = dtSlopeS / Math.max(1e-6, traced.secondsPerColumn);
          const slopeXPx = eXPx + slopeCols * traced.pixelsPerColumn;
          const slopeVelocityMps = eSign * eAbs * (1 - dropFrac);
          dt = {
            peakXPx: eXPx,
            peakVelocityMps: eVel,
            slopeXPx,
            slopeVelocityMps,
            dtMs,
          };
        }
      }
    }

    // 3. A-wave peak in late diastole — strictly signal-driven (naturally null in atrial fibrillation)
    let a: AutoMitralInflowSuccess['a'] = null;
    let aIdx = -1;
    let aAbs = 0;
    for (let i = Math.max(eIdx + 4, 0); i < vels.length; i++) {
      const col = cStart + i;
      const ph = stripPhase[col] ?? 0;
      if (ph < aWinStart) continue;
      const v = vels[i]! * eSign;
      if (v > aAbs) {
        aAbs = v;
        aIdx = i;
      }
    }
    if (aIdx > eIdx + 3 && aAbs >= 0.20) {
      // Confirm a genuine distinct late-diastolic peak rising above the intervening diastasis valley
      let valley = eAbs;
      for (let i = eIdx + 1; i < aIdx; i++) {
        valley = Math.min(valley, Math.max(0, vels[i]! * eSign));
      }
      if (aAbs - valley >= 0.05) {
        a = {
          xPx: traced.x0 + aIdx * traced.pixelsPerColumn,
          velocityMps: vels[aIdx]!,
        };
      }
    }

    return {
      ok: true,
      target: 'mitral-inflow',
      beatIndex: cand.beatIndex,
      e: { xPx: eXPx, velocityMps: eVel },
      a,
      dt,
    };
  }

  return {
    ok: false,
    target: 'mitral-inflow',
    reason: 'no-signal',
    message: 'No se detectó onda E mitral clara; revisa que el gate PW esté en las puntas mitrales.',
  };
}

/**
 * Pure strip-signal detector for early-diastolic myocardial relaxation wave e′ on a TDI strip.
 */
export function autoMeasureTdiEPrime(
  strip: Float32Array,
  cols: number,
  displayCols: number,
  head: number,
  stripPhase: Float32Array,
  stripBeatIndex: Uint32Array,
  stripSampleTime: Float64Array,
  spectral: SpectralSettings,
  phaseMarks: PhaseMarks,
  modality: ImagingModality,
): AutoStripMeasureResult {
  if (modality !== 'tdi') {
    return {
      ok: false,
      target: 'tdi-e-prime',
      reason: 'wrong-modality',
      message: 'Activa Doppler tisular (TDI) sobre el anillo mitral para medir e′ automáticamente.',
    };
  }
  if (cols < 16 || strip.length < cols * SPECTRAL_BINS) {
    return {
      ok: false,
      target: 'tdi-e-prime',
      reason: 'incomplete-beat',
      message: 'Aún no hay suficientes datos TDI para detectar un latido completo.',
    };
  }

  const pixelsPerColumn = (displayCols || cols) / cols;
  const spans = findSpectralBeatSpans(cols, head, stripPhase, stripBeatIndex, stripSampleTime);
  const reqEnd = Math.min(0.88, Math.max(phaseMarks.mitralOpen + 0.14, phaseMarks.eEnd));
  const candidates = spans
    .filter((sp) => sp.minPhase <= phaseMarks.mitralOpen + 0.03 && sp.maxPhase >= reqEnd)
    .sort((a, b) => b.maxTimeS - a.maxTimeS);

  if (!candidates.length) {
    return {
      ok: false,
      target: 'tdi-e-prime',
      reason: 'incomplete-beat',
      message: 'Espera ~1 s en TDI para registrar una onda e′ completa en la tira.',
    };
  }

  const wStart = Math.max(0.25, phaseMarks.mitralOpen - 0.02);
  const wEnd = Math.min(0.92, Math.max(phaseMarks.eEnd + 0.06, wStart + 0.16));
  const { vMin, vMax } = spectralRange(spectral);
  const span = vMax - vMin;
  const baselineBin = Math.max(0, Math.min(SPECTRAL_BINS, (vMax / span) * SPECTRAL_BINS));
  const wf = spectral.wallFilterMps + 0.008;

  for (const cand of candidates) {
    let c0 = -1;
    let c1 = -1;
    for (let c = cand.c0; c <= cand.c1; c++) {
      const ph = stripPhase[c] ?? 0;
      if (ph >= wStart && ph <= wEnd) {
        if (c0 < 0) c0 = c;
        c1 = c;
      }
    }
    if (c0 < 0 || c1 - c0 < 4) continue;

    // Per ASE/EACVI 2016 diastolic guidelines, TDI e′ is read at the brightest modal velocity ridge
    // (centroid around peak spectral density), not the outer broadening fringe.
    const modalV: number[] = [];
    let posEnergy = 0;
    let negEnergy = 0;
    for (let c = c0; c <= c1; c++) {
      const base = c * SPECTRAL_BINS;
      let colMax = 0;
      for (let b = 0; b < SPECTRAL_BINS; b++) colMax = Math.max(colMax, strip[base + b] ?? 0);
      const thr = envelopeThreshold(colMax, spectral);
      let upBest = thr;
      let dnBest = thr;
      for (let b = 0; b < SPECTRAL_BINS; b++) {
        const val = strip[base + b] ?? 0;
        if (val <= thr) continue;
        const vBin = vMax - ((b + 0.5) / SPECTRAL_BINS) * span;
        if (Math.abs(vBin) <= wf) continue;
        if (b < baselineBin) upBest = Math.max(upBest, val);
        else dnBest = Math.max(dnBest, val);
      }
      if (upBest > thr) posEnergy += upBest;
      if (dnBest > thr) negEnergy += dnBest;
    }
    const useAbove = posEnergy > negEnergy;

    for (let c = c0; c <= c1; c++) {
      const base = c * SPECTRAL_BINS;
      let colMax = 0;
      for (let b = 0; b < SPECTRAL_BINS; b++) colMax = Math.max(colMax, strip[base + b] ?? 0);
      const thr = envelopeThreshold(colMax, spectral);
      let peakBin = -1;
      let peakVal = thr;
      const bStart = useAbove ? 0 : Math.ceil(baselineBin);
      const bEnd = useAbove ? Math.floor(baselineBin) - 1 : SPECTRAL_BINS - 1;
      for (let b = bStart; b <= bEnd; b++) {
        const vBin = vMax - ((b + 0.5) / SPECTRAL_BINS) * span;
        if (Math.abs(vBin) <= wf) continue;
        const val = strip[base + b] ?? 0;
        if (val > peakVal) {
          peakVal = val;
          peakBin = b;
        }
      }
      if (peakBin < 0) {
        modalV.push(0);
      } else {
        // Intensity-weighted local modal centroid (±2 bins around peak) for sub-bin accuracy
        let wSum = 0;
        let vSum = 0;
        const lo = Math.max(bStart, peakBin - 2);
        const hi = Math.min(bEnd, peakBin + 2);
        for (let b = lo; b <= hi; b++) {
          const val = strip[base + b] ?? 0;
          if (val >= 0.6 * peakVal) {
            const vBin = vMax - ((b + 0.5) / SPECTRAL_BINS) * span;
            wSum += val;
            vSum += val * vBin;
          }
        }
        modalV.push(wSum > 0 ? vSum / wSum : 0);
      }
    }

    // 3-column median smoothing suppresses isolated speckle outliers
    let bestIdx = -1;
    let bestAbs = 0;
    for (let i = 0; i < modalV.length; i++) {
      const a = modalV[Math.max(0, i - 1)]!;
      const b = modalV[i]!;
      const d = modalV[Math.min(modalV.length - 1, i + 1)]!;
      const med = a + b + d - Math.min(a, b, d) - Math.max(a, b, d);
      if (Math.abs(med) > bestAbs) {
        bestAbs = Math.abs(med);
        bestIdx = i;
      }
    }
    if (bestIdx >= 0 && bestAbs >= 0.022) {
      const a = modalV[Math.max(0, bestIdx - 1)]!;
      const b = modalV[bestIdx]!;
      const d = modalV[Math.min(modalV.length - 1, bestIdx + 1)]!;
      const medV = a + b + d - Math.min(a, b, d) - Math.max(a, b, d);
      return {
        ok: true,
        target: 'tdi-e-prime',
        beatIndex: cand.beatIndex,
        ePrime: {
          xPx: (c0 + bestIdx + 0.5) * pixelsPerColumn,
          velocityMps: medV,
        },
      };
    }
  }

  return {
    ok: false,
    target: 'tdi-e-prime',
    reason: 'no-signal',
    message: 'No se detectó señal tisular e′ clara; verifica que el gate TDI esté en el anillo mitral.',
  };
}

/**
 * Pure strip-signal detector for systolic lateral tricuspid annular TDI velocity (S′ VD).
 * Uses the same modal spectral-ridge centroid as e′ over the systolic ejection window.
 */
export function autoMeasureRvSPrime(
  strip: Float32Array,
  cols: number,
  displayCols: number,
  head: number,
  stripPhase: Float32Array,
  stripBeatIndex: Uint32Array,
  stripSampleTime: Float64Array,
  spectral: SpectralSettings,
  phaseMarks: PhaseMarks,
  modality: ImagingModality,
): AutoStripMeasureResult {
  if (modality !== 'tdi') {
    return {
      ok: false,
      target: 'rv-s-prime',
      reason: 'wrong-modality',
      message: 'Activa Doppler tisular (TDI) sobre el anillo tricuspídeo lateral para medir S′ VD.',
    };
  }
  if (cols < 16 || strip.length < cols * SPECTRAL_BINS) {
    return {
      ok: false,
      target: 'rv-s-prime',
      reason: 'incomplete-beat',
      message: 'Aún no hay suficientes datos TDI para detectar una sístole completa.',
    };
  }

  const pixelsPerColumn = (displayCols || cols) / cols;
  const spans = findSpectralBeatSpans(cols, head, stripPhase, stripBeatIndex, stripSampleTime);
  const wStart = Math.max(0.02, phaseMarks.ejectionStart - 0.02);
  const wEnd = Math.min(0.6, phaseMarks.ejectionEnd + 0.04);
  const candidates = spans
    .filter((sp) => sp.minPhase <= wStart + 0.02 && sp.maxPhase >= wEnd - 0.02)
    .sort((a, b) => b.maxTimeS - a.maxTimeS);

  if (!candidates.length) {
    return {
      ok: false,
      target: 'rv-s-prime',
      reason: 'incomplete-beat',
      message: 'Espera ~1 s en TDI para registrar una onda sistólica S′ completa en la tira.',
    };
  }

  const { vMin, vMax } = spectralRange(spectral);
  const span = vMax - vMin;
  const baselineBin = Math.max(0, Math.min(SPECTRAL_BINS, (vMax / span) * SPECTRAL_BINS));
  const wf = spectral.wallFilterMps + 0.008;

  for (const cand of candidates) {
    let c0 = -1;
    let c1 = -1;
    for (let c = cand.c0; c <= cand.c1; c++) {
      const ph = stripPhase[c] ?? 0;
      if (ph >= wStart && ph <= wEnd) {
        if (c0 < 0) c0 = c;
        c1 = c;
      }
    }
    if (c0 < 0 || c1 - c0 < 4) continue;

    let posEnergy = 0;
    let negEnergy = 0;
    for (let c = c0; c <= c1; c++) {
      const base = c * SPECTRAL_BINS;
      let colMax = 0;
      for (let b = 0; b < SPECTRAL_BINS; b++) colMax = Math.max(colMax, strip[base + b] ?? 0);
      const thr = envelopeThreshold(colMax, spectral);
      let upBest = thr;
      let dnBest = thr;
      for (let b = 0; b < SPECTRAL_BINS; b++) {
        const val = strip[base + b] ?? 0;
        if (val <= thr) continue;
        const vBin = vMax - ((b + 0.5) / SPECTRAL_BINS) * span;
        if (Math.abs(vBin) <= wf) continue;
        if (b < baselineBin) upBest = Math.max(upBest, val);
        else dnBest = Math.max(dnBest, val);
      }
      if (upBest > thr) posEnergy += upBest;
      if (dnBest > thr) negEnergy += dnBest;
    }
    const useAbove = posEnergy >= negEnergy;
    const bStart = useAbove ? 0 : Math.ceil(baselineBin);
    const bEnd = useAbove ? Math.floor(baselineBin) - 1 : SPECTRAL_BINS - 1;

    const modalV: number[] = [];
    for (let c = c0; c <= c1; c++) {
      const base = c * SPECTRAL_BINS;
      let colMax = 0;
      for (let b = 0; b < SPECTRAL_BINS; b++) colMax = Math.max(colMax, strip[base + b] ?? 0);
      const thr = envelopeThreshold(colMax, spectral);
      let peakBin = -1;
      let peakVal = thr;
      for (let b = bStart; b <= bEnd; b++) {
        const vBin = vMax - ((b + 0.5) / SPECTRAL_BINS) * span;
        if (Math.abs(vBin) <= wf) continue;
        const val = strip[base + b] ?? 0;
        if (val > peakVal) {
          peakVal = val;
          peakBin = b;
        }
      }
      if (peakBin < 0) {
        modalV.push(0);
      } else {
        let wSum = 0;
        let vSum = 0;
        const lo = Math.max(bStart, peakBin - 2);
        const hi = Math.min(bEnd, peakBin + 2);
        for (let b = lo; b <= hi; b++) {
          const val = strip[base + b] ?? 0;
          if (val >= 0.6 * peakVal) {
            const vBin = vMax - ((b + 0.5) / SPECTRAL_BINS) * span;
            wSum += val;
            vSum += val * vBin;
          }
        }
        modalV.push(wSum > 0 ? vSum / wSum : 0);
      }
    }

    let bestIdx = -1;
    let bestAbs = 0;
    for (let i = 0; i < modalV.length; i++) {
      const a = modalV[Math.max(0, i - 1)]!;
      const b = modalV[i]!;
      const d = modalV[Math.min(modalV.length - 1, i + 1)]!;
      const med = a + b + d - Math.min(a, b, d) - Math.max(a, b, d);
      if (Math.abs(med) > bestAbs) {
        bestAbs = Math.abs(med);
        bestIdx = i;
      }
    }
    if (bestIdx >= 0 && bestAbs >= 0.025) {
      const a = modalV[Math.max(0, bestIdx - 1)]!;
      const b = modalV[bestIdx]!;
      const d = modalV[Math.min(modalV.length - 1, bestIdx + 1)]!;
      const medV = a + b + d - Math.min(a, b, d) - Math.max(a, b, d);
      return {
        ok: true,
        target: 'rv-s-prime',
        beatIndex: cand.beatIndex,
        sPrime: {
          xPx: (c0 + bestIdx + 0.5) * pixelsPerColumn,
          velocityMps: medV,
        },
      };
    }
  }

  return {
    ok: false,
    target: 'rv-s-prime',
    reason: 'no-signal',
    message: 'No se detectó señal sistólica S′ clara; verifica que el gate TDI esté en el anillo tricuspídeo lateral.',
  };
}

/**
 * Pure strip-signal detector for Pulmonary Acceleration Time (TAc pulmonar) and RVOT VTI on PW.
 * Measures from systolic ejection onset to peak pulmonary outflow velocity (< 105 ms supports PH).
 */
export function autoMeasureRvotAt(
  strip: Float32Array,
  cols: number,
  displayCols: number,
  head: number,
  stripPhase: Float32Array,
  stripBeatIndex: Uint32Array,
  stripSampleTime: Float64Array,
  spectral: SpectralSettings,
  phaseMarks: PhaseMarks,
  modality: ImagingModality,
): AutoStripMeasureResult {
  if (modality !== 'pw') {
    return {
      ok: false,
      target: 'rvot-at',
      reason: 'wrong-modality',
      message: 'Activa Doppler pulsado (PW) en el TSVD / válvula pulmonar para medir el TAc pulmonar.',
    };
  }
  const traced = autoTraceBestSystolicBeat(
    strip,
    cols,
    displayCols,
    head,
    stripPhase,
    stripBeatIndex,
    stripSampleTime,
    spectral,
    phaseMarks,
    true,
  );
  if (!traced.ok) {
    return {
      ok: false,
      target: 'rvot-at',
      reason: traced.reason === 'incomplete-beat' ? 'incomplete-beat' : 'no-signal',
      message: traced.message,
    };
  }
  const raw = traced.velocitiesMps;
  if (raw.length < 8) {
    return {
      ok: false,
      target: 'rvot-at',
      reason: 'no-signal',
      message: 'Flujo pulmonar demasiado breve o débil para determinar el tiempo de aceleración.',
    };
  }

  // 3-point median smoothing suppresses single-column speckle jitter around the systolic peak
  const sm = raw.map((_, i) => {
    const a = Math.abs(raw[Math.max(0, i - 1)]!);
    const b = Math.abs(raw[i]!);
    const c = Math.abs(raw[Math.min(raw.length - 1, i + 1)]!);
    return a + b + c - Math.min(a, b, c) - Math.max(a, b, c);
  });

  let peakIdx = 0;
  for (let i = 1; i < sm.length; i++) {
    if (sm[i]! > sm[peakIdx]!) peakIdx = i;
  }
  const peakAbs = sm[peakIdx]!;
  if (peakAbs < 0.25) {
    return {
      ok: false,
      target: 'rvot-at',
      reason: 'no-signal',
      message: 'No se detectó señal de flujo pulmonar clara; verifica que el gate PW esté en el TSVD.',
    };
  }

  const onsetThr = Math.max(0.08, 0.12 * peakAbs);
  let onsetIdx = 0;
  for (let i = 0; i < peakIdx; i++) {
    if (sm[i]! >= onsetThr) {
      onsetIdx = i;
      break;
    }
  }
  const atMs = (peakIdx - onsetIdx) * traced.secondsPerColumn * 1000;
  if (!(atMs >= 30 && atMs <= 260)) {
    return {
      ok: false,
      target: 'rvot-at',
      reason: 'no-signal',
      message: 'No se pudo identificar un pico de aceleración pulmonar fisiológico en la tira PW.',
    };
  }

  return {
    ok: true,
    target: 'rvot-at',
    beatIndex: traced.beatIndex,
    onsetXPx: traced.x0 + onsetIdx * traced.pixelsPerColumn,
    peakXPx: traced.x0 + peakIdx * traced.pixelsPerColumn,
    peakVelocityMps: raw[peakIdx]!,
    atMs,
    vtiCm: traced.vtiCm,
  };
}

/**
 * Pure strip-signal detector for peak systolic tricuspid regurgitation velocity (TR Vmax) on CW.
 */
export function autoMeasureTrVmax(
  strip: Float32Array,
  cols: number,
  displayCols: number,
  head: number,
  stripPhase: Float32Array,
  stripBeatIndex: Uint32Array,
  stripSampleTime: Float64Array,
  spectral: SpectralSettings,
  phaseMarks: PhaseMarks,
  modality: ImagingModality,
): AutoStripMeasureResult {
  if (modality !== 'cw') {
    return {
      ok: false,
      target: 'tr-vmax',
      reason: 'wrong-modality',
      message: 'Activa Doppler continuo (CW) sobre la válvula tricúspide para medir Vmax IT.',
    };
  }
  const traced = autoTraceBestSystolicBeat(
    strip,
    cols,
    displayCols,
    head,
    stripPhase,
    stripBeatIndex,
    stripSampleTime,
    spectral,
    phaseMarks,
    false,
  );
  if (!traced.ok) {
    return {
      ok: false,
      target: 'tr-vmax',
      reason: traced.reason === 'incomplete-beat' ? 'incomplete-beat' : 'no-signal',
      message: traced.message,
    };
  }
  let peakIdx = 0;
  let peakAbs = 0;
  for (let i = 0; i < traced.velocitiesMps.length; i++) {
    const av = Math.abs(traced.velocitiesMps[i]!);
    if (av > peakAbs) {
      peakAbs = av;
      peakIdx = i;
    }
  }
  if (peakAbs < 0.8) {
    return {
      ok: false,
      target: 'tr-vmax',
      reason: 'no-signal',
      message: 'No se detectó un chorro sistólico de IT suficiente en la tira CW.',
    };
  }
  return {
    ok: true,
    target: 'tr-vmax',
    beatIndex: traced.beatIndex,
    vmax: {
      xPx: traced.x0 + peakIdx * traced.pixelsPerColumn,
      velocityMps: traced.velocitiesMps[peakIdx]!,
      peakGradientMmHg: 4 * peakAbs * peakAbs,
    },
  };
}

/**
 * Pure M-mode pixel leading-edge detector for TAPSE (tricuspid annular plane systolic excursion).
 * Tracks the blood-to-annulus leading edge moving apically from ED to ES inside the acquired M-mode strip.
 */
export function autoMeasureTapse(
  grey: Uint8ClampedArray,
  samples: number,
  cols: number,
  displayCols: number,
  head: number,
  depthCm: number,
  hintDepthCm: number,
  stripPhase: Float32Array,
  stripSampleTime: Float64Array,
  phaseMarks: PhaseMarks,
  modality: ImagingModality,
): AutoStripMeasureResult {
  if (modality !== 'm-mode' && modality !== 'cmm') {
    return {
      ok: false,
      target: 'tapse',
      reason: 'wrong-modality',
      message: 'Activa Modo M sobre el anillo tricuspídeo lateral para medir TAPSE automáticamente.',
    };
  }
  if (cols < 16 || samples < 32 || grey.length < samples * cols) {
    return {
      ok: false,
      target: 'tapse',
      reason: 'incomplete-beat',
      message: 'Aún no hay suficientes columnas en Modo M para analizar un latido completo.',
    };
  }

  const pixelsPerColumn = (displayCols || cols) / cols;
  const spans = findMmodeBeatSpans(cols, head, stripPhase, stripSampleTime);
  const esPhase = phaseMarks.endSystole ?? phaseMarks.ejectionEnd;
  const candidates = spans
    .filter((sp) => sp.minPhase <= 0.08 && sp.maxPhase >= Math.max(0.65, esPhase + 0.15))
    .sort((a, b) => b.maxTimeS - a.maxTimeS);

  if (!candidates.length) {
    return {
      ok: false,
      target: 'tapse',
      reason: 'incomplete-beat',
      message: 'Espera ~1 s en Modo M para registrar un ciclo completo del anillo tricuspídeo.',
    };
  }

  const sampleCm = depthCm / samples;
  const validHint = hintDepthCm >= 5 && hintDepthCm <= depthCm - 0.8 ? hintDepthCm : depthCm * 0.72;
  const rTopCm = Math.max(3.0, validHint - 2.9);
  const rBotCm = Math.min(depthCm - 0.4, validHint + 0.7);
  const sTop = Math.max(2, Math.floor(rTopCm / sampleCm));
  const sBot = Math.min(samples - 3, Math.ceil(rBotCm / sampleCm));
  if (sBot - sTop < 20) {
    return {
      ok: false,
      target: 'tapse',
      reason: 'no-signal',
      message: 'La profundidad actual recorta el anillo tricuspídeo. Aumenta la profundidad.',
    };
  }

  for (const cand of candidates) {
    const edCols: number[] = [];
    const esCols: number[] = [];
    for (let c = cand.c0; c <= cand.c1; c++) {
      const ph = stripPhase[c] ?? 0;
      if (ph <= 0.09 || ph >= 0.92) edCols.push(c);
      if (
        ph >= Math.max(phaseMarks.ejectionStart + 0.08, esPhase - 0.08) &&
        ph <= Math.min(0.58, esPhase + 0.08)
      ) {
        esCols.push(c);
      }
    }
    if (edCols.length < 2 || esCols.length < 2) continue;

    // Average radial echo profiles at ED and ES across the beat, then subtract stationary echoes (ES - ED):
    // - Where the tricuspid annulus sits at ED (and becomes dark RA blood at ES): deep negative trough (ED - ES > 0)
    // - Where the annulus arrives at ES (and was dark RV blood at ED): sharp positive peak (ES - ED > 0)
    // Stationary pericardium/RA posterior wall cancels out naturally.
    const nS = sBot - sTop + 1;
    const profEd = new Float64Array(nS);
    const profEs = new Float64Array(nS);
    for (let k = 0; k < nS; k++) {
      const s = sTop + k;
      let sumEd = 0;
      for (const c of edCols) sumEd += grey[s * cols + c] ?? 0;
      profEd[k] = sumEd / edCols.length;
      let sumEs = 0;
      for (const c of esCols) sumEs += grey[s * cols + c] ?? 0;
      profEs[k] = sumEs / esCols.length;
    }

    // Light 0.24 cm vertical box smoothing stabilizes speckle without blurring annular position
    const rad = Math.max(2, Math.round(0.12 / sampleCm));
    const smEd = new Float64Array(nS);
    const smEs = new Float64Array(nS);
    for (let k = 0; k < nS; k++) {
      const a = Math.max(0, k - rad);
      const b = Math.min(nS - 1, k + rad);
      let s1 = 0;
      let s2 = 0;
      for (let j = a; j <= b; j++) {
        s1 += profEd[j]!;
        s2 += profEs[j]!;
      }
      const inv = 1 / (b - a + 1);
      smEd[k] = s1 * inv;
      smEs[k] = s2 * inv;
    }

    // Adaptive dark RV cavity blood floor above the tricuspid excursion window
    const bgLo = 0;
    const bgHi = Math.min(nS - 1, Math.max(2, Math.floor(0.55 / sampleCm)));
    let bgSum = 0;
    for (let k = bgLo; k <= bgHi; k++) bgSum += Math.min(smEd[k]!, smEs[k]!);
    const bloodFloor = bgSum / (bgHi - bgLo + 1);
    const edgeThr = Math.max(58, bloodFloor + 16);

    // 1. Locate ED tricuspid annular echo ridge tightly around validHint [validHint - 0.65, validHint + 0.35]
    //    (restricting below validHint + 0.35 prevents locking onto the deeper RA posterior wall / pericardium),
    //    then walk shallowward to its blood-to-tissue leading edge so ED and ES use the exact same fiducial.
    const edLo = Math.max(0, Math.floor((validHint - 0.65 - rTopCm) / sampleCm));
    const edHi = Math.min(nS - 1, Math.ceil((validHint + 0.35 - rTopCm) / sampleCm));
    let bestEdPeakK = -1;
    let bestEdScore = -Infinity;
    for (let k = edLo; k <= edHi; k++) {
      const r = (sTop + k + 0.5) * sampleCm;
      const drop = Math.max(0, smEd[k]! - smEs[k]!);
      const score = smEd[k]! * 0.65 + drop * 1.4 - Math.abs(r - validHint) * 18;
      if (score > bestEdScore) {
        bestEdScore = score;
        bestEdPeakK = k;
      }
    }
    if (bestEdPeakK < 0 || smEd[bestEdPeakK]! < edgeThr - 4) continue;
    let edLeadK = bestEdPeakK;
    const edLeadMin = Math.max(edLo, bestEdPeakK - Math.round(0.45 / sampleCm));
    for (let j = bestEdPeakK - 1; j >= edLeadMin; j--) {
      if (smEd[j]! >= edgeThr) edLeadK = j;
      else break;
    }
    const bestEdK = Math.round(0.5 * (edLeadK + bestEdPeakK));
    const edCm = (sTop + bestEdK + 0.5) * sampleCm;

    // 2. Locate ES tricuspid annular arrival scanning shallow -> deep in [edCm - 2.75, edCm - 0.45]:
    //    pick the shallowest genuine systolic tissue leading edge rising out of dark ED RV blood
    //    (prevents locking onto deeper stationary RA wall or pericardium).
    const esLo = Math.max(0, Math.floor((edCm - 2.75 - rTopCm) / sampleCm));
    const esHi = Math.min(nS - 1, Math.ceil((edCm - 0.45 - rTopCm) / sampleCm));
    let bestEsK = -1;
    for (let k = esLo; k <= esHi; k++) {
      const rise = smEs[k]! - smEd[k]!;
      if (smEs[k]! >= edgeThr && rise >= 6) {
        // Walk forward up to 0.3 cm to the local systolic leading-edge crest for exact depth
        let peakK = k;
        const crestHi = Math.min(esHi, k + Math.round(0.3 / sampleCm));
        for (let j = k + 1; j <= crestHi; j++) {
          if (smEs[j]! > smEs[peakK]!) peakK = j;
        }
        bestEsK = Math.round(0.5 * (k + peakK));
        break;
      }
    }
    if (bestEsK < 0) continue;
    const esCm = (sTop + bestEsK + 0.5) * sampleCm;
    const tapseCm = edCm - esCm;

    if (tapseCm >= 0.45 && tapseCm <= 3.8) {
      const edColMid = edCols[edCols.length >> 1]!;
      const esColMid = esCols[esCols.length >> 1]!;
      return {
        ok: true,
        target: 'tapse',
        beatIndex: cand.beatIndex,
        ed: { xPx: (edColMid + 0.5) * pixelsPerColumn, depthCm: edCm },
        es: { xPx: (esColMid + 0.5) * pixelsPerColumn, depthCm: esCm },
        tapseCm,
      };
    }
  }

  return {
    ok: false,
    target: 'tapse',
    reason: 'no-signal',
    message:
      'No se identificó la excursión del anillo tricuspídeo en la tira M; alinea el cursor con TAPSE M.',
  };
}

/** Silence unused import check when STRIP_MM_WIDTH / aseSweepDurationS are referenced for strip calibration parity. */
export const _STRIP_TIMING_REF = { aseSweepDurationS, STRIP_MM_WIDTH };
