import type { CaseDefinition } from '@/cases/schema';
import { ecgSample } from '@/simulator/cardiac-cycle/ecg';
import type { EcgPoint } from './protocol';

export const ECG_HZ = 200;
/** ECG kept and sent before the oldest cine frame, so its beat reads at the left edge of a frozen strip (s). */
export const CINE_ECG_MARGIN_S = 0.3;
const LIVE_TAIL_SAMPLES = 1200;
const LIVE_WINDOW_S = 6;
const COMPACT_HEAD_THRESHOLD = 256;

/**
 * Rolling 200 Hz ECG buffer with O(1) amortized head advancement (no per-sample Array.shift).
 * Keeps at least 6 seconds live and extends back to cover the full cine buffer when frozen.
 */
export class EcgRing {
  private samples: EcgPoint[] = [];
  private headIdx = 0;
  private accumS = 0;

  /** Number of active retained samples. */
  get length(): number {
    return this.samples.length - this.headIdx;
  }

  /** Advance the ECG clock by `dt` seconds and prune samples older than both the 6 s window and the cine buffer. */
  advance(
    timeInBeatStart: number,
    rrS: number,
    dt: number,
    timeS: number,
    oldestCineTimeS: number | undefined,
    rhythm: CaseDefinition['rhythm'],
    beatIndex: number,
    seed: number,
  ): void {
    this.accumS += dt;
    const step = 1 / ECG_HZ;
    let t = timeInBeatStart;
    while (this.accumS >= step) {
      this.accumS -= step;
      t += step;
      const tb = t % rrS;
      const v = ecgSample(tb, rrS, rhythm, beatIndex, seed);
      this.samples.push({ t: timeS - this.accumS, v });
    }
    // six seconds, and at least back to the oldest cine frame, so a frozen strip can show the whole cine (decision 197)
    const cutoff = Math.min(timeS - LIVE_WINDOW_S, (oldestCineTimeS ?? timeS) - CINE_ECG_MARGIN_S);
    while (this.headIdx < this.samples.length && (this.samples[this.headIdx]?.t ?? 0) < cutoff) {
      this.headIdx++;
    }
    if (this.headIdx >= COMPACT_HEAD_THRESHOLD) {
      this.samples = this.samples.slice(this.headIdx);
      this.headIdx = 0;
    }
  }

  /**
   * Pack the visible ECG window as interleaved (time s, amplitude) Float64 pairs ready for worker transfer.
   * Live: the last 1200 samples (6 s). Frozen: extends back to the oldest cine frame minus margin.
   */
  extractTail(frozen: boolean, timeS: number, oldestCineTimeS: number | undefined): Float64Array {
    let from = Math.max(this.headIdx, this.samples.length - LIVE_TAIL_SAMPLES);
    if (frozen) {
      const t0 = (oldestCineTimeS ?? timeS) - CINE_ECG_MARGIN_S;
      while (from > this.headIdx && (this.samples[from - 1]?.t ?? -Infinity) >= t0) from--;
    }
    const out = new Float64Array(2 * (this.samples.length - from));
    for (let i = from, k = 0; i < this.samples.length; i++, k += 2) {
      const pt = this.samples[i]!;
      out[k] = pt.t;
      out[k + 1] = pt.v;
    }
    return out;
  }
}
