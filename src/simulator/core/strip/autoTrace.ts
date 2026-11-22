import {
  CLICK_SIGMA_S,
  envelopeThreshold,
  isClickColumn,
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
 * Auto-trace the outer spectral velocity envelope between two display x coordinates, bridging valve
 * clicks and suppressing single-column click-tail spikes.
 */
export function autoTraceSpectralEnvelope(
  strip: Float32Array,
  cols: number,
  displayCols: number,
  spectral: SpectralSettings,
  reqX0: number,
  reqX1: number,
): Extract<SimResponse, { kind: 'autoTrace' }> {
  const { vMin, vMax } = spectralRange(spectral);
  const pixelsPerColumn = (displayCols || cols) / cols;
  const secondsShown = STRIP_MM_WIDTH / spectral.sweepSpeedMmPerS;
  const spc = cols ? secondsShown / cols : 0;
  const x0 = Math.max(0, Math.min(cols - 1, Math.floor(Math.min(reqX0, reqX1) / pixelsPerColumn)));
  const x1 = Math.max(0, Math.min(cols - 1, Math.floor(Math.max(reqX0, reqX1) / pixelsPerColumn)));
  const baselineBin = (vMax / (vMax - vMin)) * SPECTRAL_BINS;

  /** Outer edge (m/s) of the envelope of column x on one side of the baseline; 0 without signal there. */
  const edgeOn = (x: number, above: boolean): number => {
    const base = x * SPECTRAL_BINS;
    let max = 0;
    for (let b = 0; b < SPECTRAL_BINS; b++)
      max = Math.max(max, strip[base + b] ?? 0);
    const thr = envelopeThreshold(max, spectral);
    // Walk outward from the brightest bin of that side through the contiguous signal (gaps <= 2 bins),
    // so isolated noise far from it does not pull the envelope to the top of the scale (decision 96).
    // Bin b is centred at (b + 0.5); linearly interpolate the exact sub-bin threshold crossing.
    let edgeBin = baselineBin;
    if (above) {
      let peak = -1;
      for (let b = Math.floor(baselineBin) - 1, best = thr; b >= 0; b--)
        if ((strip[base + b] ?? 0) > best)
          best = strip[base + (peak = b)]!;
      let lastBin = -1;
      let gap = 0;
      for (let b = peak; peak >= 0 && b >= 0; b--) {
        if ((strip[base + b] ?? 0) > thr) {
          lastBin = b;
          gap = 0;
        } else if (++gap > 2) break;
      }
      if (lastBin >= 0) {
        if (lastBin > 0) {
          const vIn = strip[base + lastBin] ?? thr;
          const vOut = strip[base + lastBin - 1] ?? 0;
          const frac = Math.max(0, Math.min(1, (vIn - thr) / Math.max(1e-6, vIn - vOut)));
          edgeBin = lastBin + 0.5 - frac;
        } else {
          edgeBin = 0.5;
        }
      }
    } else {
      let peak = -1;
      for (let b = Math.ceil(baselineBin), best = thr; b < SPECTRAL_BINS; b++)
        if ((strip[base + b] ?? 0) > best)
          best = strip[base + (peak = b)]!;
      let lastBin = -1;
      let gap = 0;
      for (let b = peak; peak >= 0 && b < SPECTRAL_BINS; b++) {
        if ((strip[base + b] ?? 0) > thr) {
          lastBin = b;
          gap = 0;
        } else if (++gap > 2) break;
      }
      if (lastBin >= 0) {
        if (lastBin < SPECTRAL_BINS - 1) {
          const vIn = strip[base + lastBin] ?? thr;
          const vOut = strip[base + lastBin + 1] ?? 0;
          const frac = Math.max(0, Math.min(1, (vIn - thr) / Math.max(1e-6, vIn - vOut)));
          edgeBin = lastBin + 0.5 + frac;
        } else {
          edgeBin = SPECTRAL_BINS - 0.5;
        }
      }
    }
    const v = vMax - (edgeBin / SPECTRAL_BINS) * (vMax - vMin);
    // An edge inside the band the wall filter removes, one bin beyond it, is not a flow (decision 232)
    return Math.abs(v) <= spectral.wallFilterMps + (vMax - vMin) / SPECTRAL_BINS ? 0 : v;
  };

  // Valve clicks have no envelope: across a click the trace joins the columns on either side (decision 103).
  const core = Array.from({ length: x1 - x0 + 1 }, (_, i) =>
    isClickColumn(
      strip.subarray((x0 + i) * SPECTRAL_BINS, (x0 + i + 1) * SPECTRAL_BINS),
      spectral,
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

  // Suppress isolated single-column spikes (> SPIKE_MPS above both neighbours)
  for (const side of [up, down]) {
    for (let i = 0; i < side.length; i++) {
      const prev = i > 0 ? side[i - 1]! : side[i + 1];
      const next = i < side.length - 1 ? side[i + 1]! : prev;
      if (prev === undefined || next === undefined) continue;
      if (Math.abs(side[i]!) - Math.max(Math.abs(prev), Math.abs(next)) > SPIKE_MPS)
        side[i] = (prev + next) / 2;
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

  // Linearly interpolate across bridged valve click columns
  for (let i = 0; i < velocitiesMps.length; i++) {
    if (!click[i]) continue;
    let a = i - 1;
    let b = i + 1;
    while (a >= 0 && click[a]) a--;
    while (b < velocitiesMps.length && click[b]) b++;
    const va = a >= 0 ? velocitiesMps[a]! : b < velocitiesMps.length ? velocitiesMps[b]! : 0;
    const vb = b < velocitiesMps.length ? velocitiesMps[b]! : va;
    for (let k = a + 1; k < b; k++) velocitiesMps[k] = va + ((vb - va) * (k - a)) / (b - a);
    i = b - 1;
  }

  return {
    kind: 'autoTrace',
    velocitiesMps,
    secondsPerColumn: spc,
    x0: x0 * pixelsPerColumn,
    pixelsPerColumn,
  };
}
