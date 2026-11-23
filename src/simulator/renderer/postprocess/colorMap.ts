/**
 * Colour Doppler map and overlay blend, shared by the CPU overlay (doppler/color/colorDoppler.ts
 * `overlayColorField`) and the GPU present pass (gpu/glslImage.ts): the constants below reach the shader as
 * `#define`s from `colorMapDefinesGlsl`, so neither side carries its own copy of the numbers.
 */
export const COLOR_MAP = {
  /** Base level and range of the dominant channel (deep crimson/midnight-navy near zero -> saturated red/royal-blue). */
  DOMINANT_BASE: 24,
  DOMINANT_RANGE: 231,
  DOMINANT_GAIN: 1.18,
  /** Green channel: pure red/blue up to 64% Nyquist, brightening cleanly to gold-yellow/cyan approaching Nyquist. */
  GREEN_BASE_TOWARD: 6,
  GREEN_BASE_AWAY: 16,
  GREEN_RANGE_TOWARD: 234,
  GREEN_RANGE_AWAY: 224,
  GREEN_KNEE: 0.64,
  GREEN_GAIN: 2.75,
  /** Level of the quiet channel. */
  QUIET: 10,
  /** Variance display: keeps normal laminar valves pure BART while giving turbulent jets authentic multi-color mosaics. */
  VARIANCE_MIN: 0.28,
  VARIANCE_GAIN: 1.25,
  VARIANCE_GREEN: 226,
  VARIANCE_FADE: 0.28,
} as const;

/** Colour over grey inside the colour box: lets subtle chamber speckle texture breathe through. */
export const COLOR_BLEND = 0.78;

/**
 * Velocity → RGB (clinical BART: deep maroon→red→orange→yellow toward; midnight→blue→cyan away;
 * emerald/chartreuse mosaic on turbulence variance). When `baselineShift !== 0`, positive velocities
 * saturate at `scale + baselineShift` and negative velocities saturate at `baselineShift - scale`
 * while zero velocity remains the dark baseline reference (matching clinical PISA baseline shift).
 */
export function colorMap(
  v: number,
  scale: number,
  variance: number,
  showVariance: boolean,
  out: [number, number, number],
  baselineShift = 0,
): void {
  const M = COLOR_MAP;
  const posSpan = Math.max(1e-3, scale + baselineShift);
  const negSpan = Math.max(1e-3, scale - baselineShift);
  const t = v >= 0 ? Math.min(1, v / posSpan) : -Math.min(1, -v / negSpan);
  const a = Math.abs(t);
  const dominant = M.DOMINANT_BASE + M.DOMINANT_RANGE * Math.min(1, a * M.DOMINANT_GAIN);
  if (t >= 0) {
    out[0] = dominant;
    out[1] = Math.min(
      255,
      M.GREEN_BASE_TOWARD + M.GREEN_RANGE_TOWARD * Math.max(0, a - M.GREEN_KNEE) * M.GREEN_GAIN,
    );
    out[2] = M.QUIET;
  } else {
    out[0] = M.QUIET;
    out[1] = Math.min(
      255,
      M.GREEN_BASE_AWAY + M.GREEN_RANGE_AWAY * Math.max(0, a - M.GREEN_KNEE) * M.GREEN_GAIN,
    );
    out[2] = dominant;
  }
  if (showVariance && variance > M.VARIANCE_MIN) {
    const g = Math.min(1, (variance - M.VARIANCE_MIN) * M.VARIANCE_GAIN);
    out[1] = out[1] * (1 - g) + M.VARIANCE_GREEN * g;
    out[0] = out[0] * (1 - g * M.VARIANCE_FADE);
    out[2] = out[2] * (1 - g * M.VARIANCE_FADE);
  }
}

const glslFloat = (v: number): string => (Number.isInteger(v) ? `${v}.0` : `${v}`);

/** `#define CM_<NAME>` for every colour-map constant plus `COLOR_BLEND`. */
export function colorMapDefinesGlsl(): string {
  return [
    ...Object.entries(COLOR_MAP).map(([k, v]) => `#define CM_${k} ${glslFloat(v)}`),
    `#define COLOR_BLEND ${glslFloat(COLOR_BLEND)}`,
  ].join('\n');
}
