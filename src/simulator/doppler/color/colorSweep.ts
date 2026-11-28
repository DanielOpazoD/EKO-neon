/**
 * The colour sweep (Fase C). A colour frame is not a snapshot: its transmit lines fire one after the other across the box,
 * each with its whole packet, so the frame spans `colorFrameTimeS` (frameRate.ts) — 73 ms with the default box at
 * ±0.62 m/s, a tenth of a beat at 80 bpm — and the blood its first line reads is that much older than its last line's.
 * At valve opening the inflow is coloured on one side of the box a frame before the other; a regurgitant jet that starts
 * or ends inside the sweep is cut by a straight edge along a beam line, the temporal "tear" of real colour frames.
 *
 * Only the haemodynamic instant of each colour line moves: the geometry the flow is drawn in (annuli, the chambers'
 * blood that gates it) stays the B-mode frame's, so the colour never leaves the walls and leaflets shown. The frame's
 * phase is the middle of the sweep, which keeps every timing measured on the frame unbiased.
 */

/**
 * Phase offset (fraction of the beat) of colour line `li` of the box [liMin, liMax] against the frame's phase, for a
 * sweep of `sweepS` seconds over a beat of `rrS`: the lines fire from `liMin` to `liMax`, the middle line at the frame's
 * phase. Zero without a sweep; lines outside the box take the edge's offset.
 */
export function colorLinePhaseOffset(
  li: number,
  liMin: number,
  liMax: number,
  sweepS: number,
  rrS: number,
): number {
  if (!(sweepS > 0) || !(rrS > 0) || liMax < liMin) return 0;
  const u = (li - liMin + 0.5) / (liMax - liMin + 1) - 0.5;
  return (Math.max(-0.5, Math.min(0.5, u)) * sweepS) / rrS;
}

/** Phase in [0, 1). */
export function wrapPhase(phase: number): number {
  return ((phase % 1) + 1) % 1;
}
