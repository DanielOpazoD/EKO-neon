import { Tissue } from '@/simulator/anatomy/tissue';
import type { AcquisitionSettings, PolarFrameSpec } from '../types';

/**
 * Clinical one-touch Auto-Optimization (iScan / Auto-Opt):
 * Analyzes the displayed polar frame across the 8 TGC depth bands, excluding outside-body
 * and dead-lung zones, then estimates optimal overall gain and per-depth TGC corrections
 * to bring myocardial walls to a balanced clinical brightness (~145/255 p85) without washing
 * out blood-filled chambers (guarding p25 cavity floor).
 */
export interface AutoOptResult {
  gainDb: number;
  tgcDb: number[];
}

const TARGET_WALL_P85 = 148;
const MAX_CAVITY_P25 = 64;
const BAND_COUNT = 8;

function percentileFromHist(hist: Uint32Array, total: number, q: number): number {
  if (total <= 0) return 0;
  const target = total * q;
  let acc = 0;
  for (let i = 0; i < 256; i++) {
    acc += hist[i]!;
    if (acc >= target) return i;
  }
  return 255;
}

export function computeAutoOpt(
  display: Uint8ClampedArray,
  tissue: Uint8Array,
  spec: Pick<PolarFrameSpec, 'lines' | 'samples'>,
  current: Pick<AcquisitionSettings, 'gainDb' | 'tgcDb' | 'dynamicRangeDb'>,
): AutoOptResult {
  const { lines, samples } = spec;
  const n = lines * samples;
  if (display.length < n || tissue.length < n || lines < 2 || samples < BAND_COUNT) {
    return { gainDb: current.gainDb, tgcDb: [...current.tgcDb] };
  }

  const dynRange = Math.max(30, Math.min(90, current.dynamicRangeDb || 70));
  const rawDelta = new Float64Array(BAND_COUNT);
  const validBand = new Uint8Array(BAND_COUNT);
  const hist = new Uint32Array(256);

  // Exclude outer 8% of lines on each side where rib/lung edges dominate
  const lStart = Math.floor(lines * 0.08);
  const lEnd = Math.max(lStart + 1, Math.ceil(lines * 0.92));

  for (let b = 0; b < BAND_COUNT; b++) {
    hist.fill(0);
    const sStart = Math.floor((b * samples) / BAND_COUNT);
    const sEnd = Math.max(sStart + 1, Math.floor(((b + 1) * samples) / BAND_COUNT));
    let count = 0;
    for (let li = lStart; li < lEnd; li++) {
      const row = li * samples;
      for (let si = sStart; si < sEnd; si++) {
        const idx = row + si;
        const t = tissue[idx]!;
        // Ignore outside body and lung reverberation dead-zone
        if (t === Tissue.None || t === Tissue.Lung) continue;
        hist[display[idx]!]!++;
        count++;
      }
    }
    if (count < 24) {
      validBand[b] = 0;
      rawDelta[b] = 0;
      continue;
    }
    validBand[b] = 1;
    const p85 = percentileFromHist(hist, count, 0.85);
    const p25 = percentileFromHist(hist, count, 0.25);

    // Primary drive: bring tissue/endocardial p85 toward TARGET_WALL_P85
    let deltaGray = TARGET_WALL_P85 - p85;
    // Cavity guard: if blood pool floor is already washed out, pull down instead of boosting noise
    if (p25 > MAX_CAVITY_P25 && deltaGray > -((p25 - MAX_CAVITY_P25) * 1.1)) {
      deltaGray = Math.min(deltaGray, -(p25 - MAX_CAVITY_P25) * 1.1);
    }
    // Convert gray delta to dB (clinical gray curve has higher slope in highlights, so damp by 0.58)
    rawDelta[b] = Math.max(-14, Math.min(14, (deltaGray / 255) * dynRange * 0.58));
  }

  // Fill any empty bands (e.g. deep lung shadow) from nearest valid neighbor
  for (let b = 0; b < BAND_COUNT; b++) {
    if (validBand[b]) continue;
    let best = 0;
    let bestDist = 99;
    for (let k = 0; k < BAND_COUNT; k++) {
      if (validBand[k] && Math.abs(k - b) < bestDist) {
        bestDist = Math.abs(k - b);
        best = rawDelta[k]!;
      }
    }
    rawDelta[b] = best * 0.5;
  }

  // Smooth the 8-band dB profile with [0.25, 0.5, 0.25] to avoid horizontal TGC banding
  const smooth = new Float64Array(BAND_COUNT);
  for (let b = 0; b < BAND_COUNT; b++) {
    const prev = rawDelta[Math.max(0, b - 1)]!;
    const curr = rawDelta[b]!;
    const next = rawDelta[Math.min(BAND_COUNT - 1, b + 1)]!;
    smooth[b] = 0.25 * prev + 0.5 * curr + 0.25 * next;
  }

  // Combine with existing gain + TGC so repeated presses converge smoothly instead of drifting
  const totalPerBand = new Float64Array(BAND_COUNT);
  let meanTotal = 0;
  for (let b = 0; b < BAND_COUNT; b++) {
    const curTgc = current.tgcDb[b] ?? 0;
    totalPerBand[b] = current.gainDb + curTgc + smooth[b]!;
    meanTotal += totalPerBand[b]!;
  }
  meanTotal /= BAND_COUNT;

  const gainDb = Math.max(-24, Math.min(24, Math.round(meanTotal)));
  const tgcDb: number[] = [];
  for (let b = 0; b < BAND_COUNT; b++) {
    const residual = totalPerBand[b]! - gainDb;
    tgcDb.push(Math.max(-12, Math.min(12, Math.round(residual))));
  }

  return { gainDb, tgcDb };
}
