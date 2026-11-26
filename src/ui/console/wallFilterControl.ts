import {
  colorWallFilterCutoffMps,
  colorWallFilterRangeMps,
} from '@/simulator/doppler/color/kasaiEstimator';

/** A wall filter cutoff as the console shows it: to 0.1 cm/s, or off. */
export function wallFilterText(mps: number): string {
  return mps > 0 ? `${(mps * 100).toFixed(1)} cm/s` : 'apagado';
}

/**
 * The colour wall filter's control at an acquired Nyquist scale (2D colour and colour M-mode). The store keeps what the
 * learner asks for; the slider shows and sets the cutoff that filters, the request bounded to what 8 pulses allow there
 * (0.11 to 0.32 of it), between those exact bounds in steps of about 1 mm/s, and says when the scale limits the request.
 * Rounded to the centimetre, the bounds left the lowest cutoff out of reach and, under 0.62 m/s, the default outside
 * the slider (at 0.35 m/s: 4–11 cm/s for a value of 11.2). The limit is told as the console shows both values: compared
 * to 1e-9 m/s, a PRF-limited scale of 0.6231 m/s read «Corte pedido 20.0 cm/s: la escala lo limita a 20.0 cm/s.».
 */
export function wallFilterControl(
  requestMps: number,
  scaleMps: number,
): { min: number; max: number; step: number; cutMps: number; limited: boolean } {
  const { min, max } = colorWallFilterRangeMps(scaleMps);
  const cutMps = colorWallFilterCutoffMps(requestMps, scaleMps);
  return {
    min,
    max,
    step: (max - min) / Math.max(1, Math.round((max - min) / 0.001)),
    cutMps,
    limited: requestMps > 0 && wallFilterText(cutMps) !== wallFilterText(requestMps),
  };
}
