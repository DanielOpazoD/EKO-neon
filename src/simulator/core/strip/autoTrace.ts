import { aseSweepDurationS } from '@/clinical/formulas';
import {
  CLICK_SIGMA_S,
  cwEnvelopeThreshold,
  cwJetEnvelopeThreshold,
  envelopeThreshold,
  isClickColumn,
  PW_MODAL_EDGE_FRAC,
  SPECTRAL_BINS,
  spectralRange,
  type SpectralSettings,
} from '@/simulator/doppler/spectral/spectrum';
import type { SimResponse } from '../protocol';

export const STRIP_MM_WIDTH = 100; // physical width represented by the strip (mm)

/**
 * How far one column of an auto-traced envelope may stand above both of its neighbours before it is
 * taken for a line and not a flow (decision 232). A flow rises steeply — up to 0.47 m/s per column
 * (6.3 ms) in the jet of the severe stenosis and 0.75 at the onset of the severe regurgitation —
 * but keeps rising or falls back over several columns; it does not gain and lose half a metre per
 * second within one.
 */
const SPIKE_MPS = 0.5;

/**
 * Auto-trace the spectral velocity envelope between two display x coordinates, bridging valve
 * clicks and suppressing single-column click-tail spikes.
 *
 * Per ASE 2019 / Quiñones et al. (JASE 2002):
 * - PW/TDI (`modal = true`): traces the outer edge of the dense modal velocity ridge (-3 dB half-power
 *   threshold `PW_MODAL_EDGE_FRAC * peak`), excluding low-intensity spectral broadening fringes.
 * - CW (`modal = false`): traces the outer edge of the filled spectral jet (`0.35 * columnMax`).
 */
export function autoTraceSpectralEnvelope(
  strip: Float32Array,
  cols: number,
  displayCols: number,
  spectral: SpectralSettings,
  reqX0: number,
  reqX1: number,
  modal = false,
): Extract<SimResponse, { kind: 'autoTrace' }> {
  const { vMin, vMax } = spectralRange(spectral);
  const span = vMax - vMin;
  const binWidthMps = span / SPECTRAL_BINS;
  const pixelsPerColumn = (displayCols || cols) / cols;
  const secondsShown = aseSweepDurationS(spectral.sweepSpeedMmPerS, STRIP_MM_WIDTH);
  const spc = cols ? secondsShown / cols : 0;
  const x0 = Math.max(0, Math.min(cols - 1, Math.floor(Math.min(reqX0, reqX1) / pixelsPerColumn)));
  const x1 = Math.max(0, Math.min(cols - 1, Math.floor(Math.max(reqX0, reqX1) / pixelsPerColumn)));
  const baselineBin = Math.max(0, Math.min(SPECTRAL_BINS, (vMax / span) * SPECTRAL_BINS));
  const maxGap = modal ? 1 : 2;
  const wallMargin = modal ? 0.5 * binWidthMps : binWidthMps;

  /** Envelope velocity (m/s) of column x on one side of the baseline; 0 without signal there. */
  const edgeOn = (x: number, above: boolean): number => {
    const base = x * SPECTRAL_BINS;
    let max = 0;
    for (let b = 0; b < SPECTRAL_BINS; b++) max = Math.max(max, strip[base + b] ?? 0);
    const detectThr = modal ? envelopeThreshold(max, spectral) : cwEnvelopeThreshold(max, spectral);
    // Bin b is centred at (b + 0.5); linearly interpolate the exact sub-bin threshold crossing.
    let edgeBin = baselineBin;
    if (!modal) {
      // CW ASE outer-envelope detector: scan inward from the highest-velocity edge for the
      // outermost contiguous spectral jet body (>= 2 consecutive bins above threshold). High-velocity
      // jet bands (|v| >= 1.1 m/s) use cwJetEnvelopeThreshold so eccentric MR/TR jets are not buried
      // beneath a much brighter slow intracavitary flow near baseline.
      const minRun = 2;
      const highVelCutoff = Math.max(1.1, spectral.wallFilterMps + 0.5);
      const binThr = (b: number, jetThr: number): number => {
        const vBin = vMax - ((b + 0.5) / SPECTRAL_BINS) * span;
        return Math.abs(vBin) >= highVelCutoff ? jetThr : detectThr;
      };
      if (above) {
        const limit = Math.floor(baselineBin) - 1;
        let highVelMax = 0;
        for (let b = 0; b <= limit; b++) {
          const vBin = vMax - ((b + 0.5) / SPECTRAL_BINS) * span;
          if (vBin >= highVelCutoff) highVelMax = Math.max(highVelMax, strip[base + b] ?? 0);
        }
        const jetThr = cwJetEnvelopeThreshold(max, highVelMax, spectral);
        let run = 0;
        let outerBin = -1;
        for (let b = 0; b <= limit; b++) {
          if ((strip[base + b] ?? 0) > binThr(b, jetThr)) {
            if (++run >= minRun) {
              outerBin = b - minRun + 1;
              break;
            }
          } else {
            run = 0;
          }
        }
        if (outerBin >= 0) {
          const thrEdge = binThr(outerBin, jetThr);
          if (outerBin > 0) {
            const vIn = strip[base + outerBin] ?? thrEdge;
            const vOut = strip[base + outerBin - 1] ?? 0;
            const frac = Math.max(0, Math.min(1, (vIn - thrEdge) / Math.max(1e-6, vIn - vOut)));
            edgeBin = outerBin + 0.5 - frac;
          } else {
            edgeBin = 0.5;
          }
        }
      } else {
        const limit = Math.ceil(baselineBin);
        let highVelMax = 0;
        for (let b = SPECTRAL_BINS - 1; b >= limit; b--) {
          const vBin = vMax - ((b + 0.5) / SPECTRAL_BINS) * span;
          if (-vBin >= highVelCutoff) highVelMax = Math.max(highVelMax, strip[base + b] ?? 0);
        }
        const jetThr = cwJetEnvelopeThreshold(max, highVelMax, spectral);
        let run = 0;
        let outerBin = -1;
        for (let b = SPECTRAL_BINS - 1; b >= limit; b--) {
          if ((strip[base + b] ?? 0) > binThr(b, jetThr)) {
            if (++run >= minRun) {
              outerBin = b + minRun - 1;
              break;
            }
          } else {
            run = 0;
          }
        }
        if (outerBin >= 0) {
          const thrEdge = binThr(outerBin, jetThr);
          if (outerBin < SPECTRAL_BINS - 1) {
            const vIn = strip[base + outerBin] ?? thrEdge;
            const vOut = strip[base + outerBin + 1] ?? 0;
            const frac = Math.max(0, Math.min(1, (vIn - thrEdge) / Math.max(1e-6, vIn - vOut)));
            edgeBin = outerBin + 0.5 + frac;
          } else {
            edgeBin = SPECTRAL_BINS - 0.5;
          }
        }
      }
    } else if (above) {
      // PW/TDI modal ridge: walk outward from the brightest bin on that side through contiguous signal.
      let peak = -1;
      let peakVal = detectThr;
      for (let b = Math.floor(baselineBin) - 1; b >= 0; b--) {
        const val = strip[base + b] ?? 0;
        if (val > peakVal) {
          peakVal = val;
          peak = b;
        }
      }
      const traceThr = Math.max(detectThr, PW_MODAL_EDGE_FRAC * peakVal);
      let lastBin = -1;
      let gap = 0;
      for (let b = peak; peak >= 0 && b >= 0; b--) {
        if ((strip[base + b] ?? 0) > traceThr) {
          lastBin = b;
          gap = 0;
        } else if (++gap > maxGap) break;
      }
      if (lastBin >= 0) {
        if (lastBin > 0) {
          const vIn = strip[base + lastBin] ?? traceThr;
          const vOut = strip[base + lastBin - 1] ?? 0;
          const frac = Math.max(0, Math.min(1, (vIn - traceThr) / Math.max(1e-6, vIn - vOut)));
          edgeBin = lastBin + 0.5 - frac;
        } else {
          edgeBin = 0.5;
        }
      }
    } else {
      let peak = -1;
      let peakVal = detectThr;
      for (let b = Math.ceil(baselineBin); b < SPECTRAL_BINS; b++) {
        const val = strip[base + b] ?? 0;
        if (val > peakVal) {
          peakVal = val;
          peak = b;
        }
      }
      const traceThr = Math.max(detectThr, PW_MODAL_EDGE_FRAC * peakVal);
      let lastBin = -1;
      let gap = 0;
      for (let b = peak; peak >= 0 && b < SPECTRAL_BINS; b++) {
        if ((strip[base + b] ?? 0) > traceThr) {
          lastBin = b;
          gap = 0;
        } else if (++gap > maxGap) break;
      }
      if (lastBin >= 0) {
        if (lastBin < SPECTRAL_BINS - 1) {
          const vIn = strip[base + lastBin] ?? traceThr;
          const vOut = strip[base + lastBin + 1] ?? 0;
          const frac = Math.max(0, Math.min(1, (vIn - traceThr) / Math.max(1e-6, vIn - vOut)));
          edgeBin = lastBin + 0.5 + frac;
        } else {
          edgeBin = SPECTRAL_BINS - 0.5;
        }
      }
    }
    const v = vMax - (edgeBin / SPECTRAL_BINS) * span;
    // An edge inside the band the wall filter removes is not a flow (decision 232)
    return Math.abs(v) <= spectral.wallFilterMps + wallMargin ? 0 : v;
  };

  // Valve clicks have no envelope: across a click the trace joins the columns on either side (decision 103).
  const core = Array.from({ length: x1 - x0 + 1 }, (_, i) =>
    isClickColumn(
      strip.subarray((x0 + i) * SPECTRAL_BINS, (x0 + i + 1) * SPECTRAL_BINS),
      spectral,
      !modal,
    ),
  );
  // Bridge 2.5 click widths on each side for faint click tails
  const reach = Math.ceil((2.5 * CLICK_SIGMA_S) / Math.max(1e-6, spc));
  const click = core.map((_, i) =>
    core.slice(Math.max(0, i - reach), i + reach + 1).some(Boolean),
  );

  // Pick one side for the whole selection: the side holding more flow outside valve clicks (decision 232)
  const up: number[] = [];
  const down: number[] = [];
  for (let x = x0; x <= x1; x++) {
    up.push(edgeOn(x, true));
    down.push(edgeOn(x, false));
  }

  // Suppress isolated single-column spikes (> SPIKE_MPS above both neighbours), single-column noise
  // islands (flanked by 0 on both sides), and single-column mid-flow dropouts.
  for (const side of [up, down]) {
    for (let i = 0; i < side.length; i++) {
      const prev = i > 0 ? side[i - 1]! : side[i + 1];
      const next = i < side.length - 1 ? side[i + 1]! : prev;
      if (prev === undefined || next === undefined) continue;
      if (Math.abs(side[i]!) - Math.max(Math.abs(prev), Math.abs(next)) > SPIKE_MPS) {
        side[i] = (prev + next) / 2;
      } else if (side[i] !== 0 && i > 0 && i < side.length - 1 && prev === 0 && next === 0) {
        side[i] = 0;
      } else if (
        side[i] === 0 &&
        i > 0 &&
        i < side.length - 1 &&
        prev * next > 0 &&
        Math.min(Math.abs(prev), Math.abs(next)) > spectral.wallFilterMps
      ) {
        side[i] = (prev + next) / 2;
      }
    }
  }

  let flowUp = 0;
  let flowDown = 0;
  for (let i = 0; i < up.length; i++) {
    if (click[i]) continue;
    flowUp += up[i]!;
    flowDown -= down[i]!;
  }
  const velocitiesMps = flowUp >= flowDown ? up : down;

  // Interpolate across bridged valve click columns:
  // - Opening click (0 -> flow): flow begins at the click centre (no phantom pre-opening ramp).
  // - Closing click (flow -> 0): flow terminates at the click centre (no phantom post-closing ramp).
  // - Mid-flow click (flow -> flow): linear bridge across the masked span.
  for (let i = 0; i < velocitiesMps.length; i++) {
    if (!click[i]) continue;
    let a = i - 1;
    let b = i + 1;
    while (a >= 0 && click[a]) a--;
    while (b < velocitiesMps.length && click[b]) b++;
    const va = a >= 0 ? velocitiesMps[a]! : b < velocitiesMps.length ? velocitiesMps[b]! : 0;
    const vb = b < velocitiesMps.length ? velocitiesMps[b]! : va;
    let coreSum = 0;
    let coreCount = 0;
    for (let k = a + 1; k < b; k++) {
      if (core[k]) {
        coreSum += k;
        coreCount++;
      }
    }
    const mid = coreCount > 0 ? coreSum / coreCount : (a + b) / 2;
    if (va === 0 && vb !== 0 && b < velocitiesMps.length) {
      const denom = Math.max(1e-6, b - mid);
      for (let k = a + 1; k < b; k++) {
        velocitiesMps[k] = k <= mid ? 0 : (vb * (k - mid)) / denom;
      }
    } else if (va !== 0 && vb === 0 && a >= 0) {
      const denom = Math.max(1e-6, mid - a);
      for (let k = a + 1; k < b; k++) {
        velocitiesMps[k] = k >= mid ? 0 : (va * (mid - k)) / denom;
      }
    } else {
      for (let k = a + 1; k < b; k++) velocitiesMps[k] = va + ((vb - va) * (k - a)) / (b - a);
    }
    i = b - 1;
  }

  return {
    kind: 'autoTrace',
    velocitiesMps,
    secondsPerColumn: spc,
    // Align trace sample i with the painted centre of column (x0 + i)
    x0: (x0 + 0.5) * pixelsPerColumn,
    pixelsPerColumn,
  };
}
