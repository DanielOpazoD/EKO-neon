/**
 * Clinical Color Doppler LUT palettes, shared between the CPU overlay (`overlayColorField`),
 * CMM strip painter (`paintMmodeStripColumn`), color bar legend (`drawColorBar`), and the
 * WebGL2 present pass (`GLSL_PRESENT_FRAG`).
 *
 * All palette constants live here as a single source of truth and are emitted into GLSL via
 * `colorMapDefinesGlsl()` / `colorMapGlsl()`, guaranteeing bit-consistent CPU↔GPU parity.
 */
export type ColorPaletteId = 'vivid' | 'classic' | 'variance' | 'ice-fire';

export const COLOR_PALETTE_DEFAULT: ColorPaletteId = 'vivid';

export const COLOR_PALETTE_IDS = ['vivid', 'classic', 'variance', 'ice-fire'] as const;

export const COLOR_PALETTE_OPTIONS: readonly { id: ColorPaletteId; label: string }[] = [
  { id: 'vivid', label: 'Vivid' },
  { id: 'classic', label: 'Clásica' },
  { id: 'variance', label: 'Varianza' },
  { id: 'ice-fire', label: 'Fuego' },
];

export function colorPaletteIndex(palette: ColorPaletteId | undefined): number {
  switch (palette) {
    case 'classic':
      return 1;
    case 'variance':
      return 2;
    case 'ice-fire':
      return 3;
    default:
      return 0;
  }
}

export interface ColorPaletteParams {
  /** Base level and range of the dominant channel (toward=red, away=blue). */
  readonly DOMINANT_BASE: number;
  readonly DOMINANT_RANGE: number;
  readonly DOMINANT_GAIN: number;
  /** Green channel curve brightening toward Nyquist (yellow/cyan). */
  readonly GREEN_BASE_TOWARD: number;
  readonly GREEN_BASE_AWAY: number;
  readonly GREEN_RANGE_TOWARD: number;
  readonly GREEN_RANGE_AWAY: number;
  readonly GREEN_KNEE: number;
  readonly GREEN_GAIN: number;
  /** Baseline quiet channel plus optional high-velocity hot-tip boost (for ice-fire white-gold / ice-white). */
  readonly QUIET: number;
  readonly QUIET_BOOST_TOWARD: number;
  readonly QUIET_BOOST_AWAY: number;
  /** Variance display parameters: laminar purity below VARIANCE_MIN, turbulence mosaic above. */
  readonly VARIANCE_MIN: number;
  readonly VARIANCE_GAIN: number;
  readonly VARIANCE_GREEN: number;
  readonly VARIANCE_FADE: number;
}

/** Default clinical BART constants (Vivid palette) — preserved verbatim for backward compatibility. */
export const COLOR_MAP = {
  /** Base level and range of the dominant channel (deep crimson/midnight-navy near zero -> saturated red/royal-blue). */
  DOMINANT_BASE: 28,
  DOMINANT_RANGE: 227,
  DOMINANT_GAIN: 1.25,
  /** Green channel: pure red/blue up to 66% Nyquist, brightening smoothly to warm gold-yellow/ice-cyan at Nyquist. */
  GREEN_BASE_TOWARD: 6,
  GREEN_BASE_AWAY: 16,
  GREEN_RANGE_TOWARD: 230,
  GREEN_RANGE_AWAY: 220,
  GREEN_KNEE: 0.66,
  GREEN_GAIN: 2.8,
  /** Level of the quiet channel. */
  QUIET: 10,
  /** Variance display: keeps normal laminar valves pure BART while giving turbulent jets authentic multi-color mosaics. */
  VARIANCE_MIN: 0.28,
  VARIANCE_GAIN: 1.25,
  VARIANCE_GREEN: 226,
  VARIANCE_FADE: 0.28,
} as const;

export const COLOR_PALETTES: Record<ColorPaletteId, ColorPaletteParams> = {
  vivid: {
    ...COLOR_MAP,
    QUIET_BOOST_TOWARD: 0,
    QUIET_BOOST_AWAY: 0,
  },
  /** Traditional GE/Philips high-contrast dark-baseline BART: deeper black center, ruby-gold / cobalt-cyan. */
  classic: {
    DOMINANT_BASE: 10,
    DOMINANT_RANGE: 245,
    DOMINANT_GAIN: 1.08,
    GREEN_BASE_TOWARD: 2,
    GREEN_BASE_AWAY: 8,
    GREEN_RANGE_TOWARD: 245,
    GREEN_RANGE_AWAY: 238,
    GREEN_KNEE: 0.56,
    GREEN_GAIN: 2.3,
    QUIET: 4,
    QUIET_BOOST_TOWARD: 0,
    QUIET_BOOST_AWAY: 0,
    VARIANCE_MIN: 0.3,
    VARIANCE_GAIN: 1.2,
    VARIANCE_GREEN: 214,
    VARIANCE_FADE: 0.22,
  },
  /** Enhanced Vel-Var diagnostic palette: earlier, stronger emerald-lime turbulence mosaic on regurgitant jets. */
  variance: {
    DOMINANT_BASE: 22,
    DOMINANT_RANGE: 224,
    DOMINANT_GAIN: 1.18,
    GREEN_BASE_TOWARD: 4,
    GREEN_BASE_AWAY: 12,
    GREEN_RANGE_TOWARD: 185,
    GREEN_RANGE_AWAY: 190,
    GREEN_KNEE: 0.72,
    GREEN_GAIN: 2.4,
    QUIET: 14,
    QUIET_BOOST_TOWARD: 0,
    QUIET_BOOST_AWAY: 0,
    VARIANCE_MIN: 0.16,
    VARIANCE_GAIN: 1.65,
    VARIANCE_GREEN: 250,
    VARIANCE_FADE: 0.42,
  },
  /** High-luminance Fire/Ice perceptual map: burnt amber→white-gold toward, indigo→ice-white away. */
  'ice-fire': {
    DOMINANT_BASE: 36,
    DOMINANT_RANGE: 219,
    DOMINANT_GAIN: 1.38,
    GREEN_BASE_TOWARD: 14,
    GREEN_BASE_AWAY: 26,
    GREEN_RANGE_TOWARD: 236,
    GREEN_RANGE_AWAY: 224,
    GREEN_KNEE: 0.34,
    GREEN_GAIN: 1.55,
    QUIET: 12,
    QUIET_BOOST_TOWARD: 185,
    QUIET_BOOST_AWAY: 195,
    VARIANCE_MIN: 0.26,
    VARIANCE_GAIN: 1.3,
    VARIANCE_GREEN: 238,
    VARIANCE_FADE: 0.26,
  },
};

/** Colour over grey inside the colour box: lets subtle chamber speckle texture breathe through. */
export const COLOR_BLEND = 0.78;

/**
 * Velocity → RGB using the selected clinical palette (`vivid` default).
 * When `baselineShift !== 0`, positive velocities saturate at `scale + baselineShift` and negative
 * velocities saturate at `baselineShift - scale` while zero velocity remains the dark baseline reference.
 */
export function colorMap(
  v: number,
  scale: number,
  variance: number,
  showVariance: boolean,
  out: [number, number, number],
  baselineShift = 0,
  palette: ColorPaletteId = COLOR_PALETTE_DEFAULT,
): void {
  const M = COLOR_PALETTES[palette] ?? COLOR_PALETTES.vivid;
  const posSpan = Math.max(1e-3, scale + baselineShift);
  const negSpan = Math.max(1e-3, scale - baselineShift);
  const t = v >= 0 ? Math.min(1, v / posSpan) : -Math.min(1, -v / negSpan);
  const a = Math.abs(t);
  const dominant = M.DOMINANT_BASE + M.DOMINANT_RANGE * Math.min(1, a * M.DOMINANT_GAIN);
  const hotTip = Math.min(1, Math.max(0, (a - 0.72) / 0.28));
  if (t >= 0) {
    out[0] = dominant;
    out[1] = Math.min(
      255,
      M.GREEN_BASE_TOWARD + M.GREEN_RANGE_TOWARD * Math.max(0, a - M.GREEN_KNEE) * M.GREEN_GAIN,
    );
    out[2] = Math.min(255, M.QUIET + M.QUIET_BOOST_TOWARD * hotTip);
  } else {
    out[0] = Math.min(255, M.QUIET + M.QUIET_BOOST_AWAY * hotTip);
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

/** `#define CM_<NAME>` for default colour-map constants plus `COLOR_BLEND`. */
export function colorMapDefinesGlsl(): string {
  return [
    ...Object.entries(COLOR_MAP).map(([k, v]) => `#define CM_${k} ${glslFloat(v)}`),
    `#define COLOR_BLEND ${glslFloat(COLOR_BLEND)}`,
  ].join('\n');
}

function glslPaletteBranch(idx: number, p: ColorPaletteParams): string {
  const f = glslFloat;
  return `  ${idx === 0 ? 'if' : 'else if'} (paletteId == ${idx}) {
    domBase = ${f(p.DOMINANT_BASE)}; domRange = ${f(p.DOMINANT_RANGE)}; domGain = ${f(p.DOMINANT_GAIN)};
    gBaseT = ${f(p.GREEN_BASE_TOWARD)}; gBaseA = ${f(p.GREEN_BASE_AWAY)};
    gRangeT = ${f(p.GREEN_RANGE_TOWARD)}; gRangeA = ${f(p.GREEN_RANGE_AWAY)};
    gKnee = ${f(p.GREEN_KNEE)}; gGain = ${f(p.GREEN_GAIN)};
    quiet = ${f(p.QUIET)}; qBoostT = ${f(p.QUIET_BOOST_TOWARD)}; qBoostA = ${f(p.QUIET_BOOST_AWAY)};
    varMin = ${f(p.VARIANCE_MIN)}; varGain = ${f(p.VARIANCE_GAIN)};
    varGreen = ${f(p.VARIANCE_GREEN)}; varFade = ${f(p.VARIANCE_FADE)};
  }`;
}

/** Matching multi-palette GLSL `colorMap()` generated directly from `COLOR_PALETTES`. */
export function colorMapGlsl(): string {
  const branches = COLOR_PALETTE_IDS.map((id, i) => glslPaletteBranch(i, COLOR_PALETTES[id])).join(
    '\n',
  );
  return /* glsl */ `
// postprocess/colorMap.ts colorMap
vec3 colorMap(float v, float scale, float variance, bool showVariance, float baselineShift, int paletteId) {
  float domBase = CM_DOMINANT_BASE;
  float domRange = CM_DOMINANT_RANGE;
  float domGain = CM_DOMINANT_GAIN;
  float gBaseT = CM_GREEN_BASE_TOWARD;
  float gBaseA = CM_GREEN_BASE_AWAY;
  float gRangeT = CM_GREEN_RANGE_TOWARD;
  float gRangeA = CM_GREEN_RANGE_AWAY;
  float gKnee = CM_GREEN_KNEE;
  float gGain = CM_GREEN_GAIN;
  float quiet = CM_QUIET;
  float qBoostT = 0.0;
  float qBoostA = 0.0;
  float varMin = CM_VARIANCE_MIN;
  float varGain = CM_VARIANCE_GAIN;
  float varGreen = CM_VARIANCE_GREEN;
  float varFade = CM_VARIANCE_FADE;
${branches}
  float posSpan = max(1e-3, scale + baselineShift);
  float negSpan = max(1e-3, scale - baselineShift);
  float t = v >= 0.0 ? min(1.0, v / posSpan) : -min(1.0, -v / negSpan);
  float a = abs(t);
  float dominant = domBase + domRange * min(1.0, a * domGain);
  float hotTip = clamp((a - 0.72) / 0.28, 0.0, 1.0);
  vec3 c = t >= 0.0
    ? vec3(dominant, min(255.0, gBaseT + gRangeT * max(0.0, a - gKnee) * gGain), min(255.0, quiet + qBoostT * hotTip))
    : vec3(min(255.0, quiet + qBoostA * hotTip), min(255.0, gBaseA + gRangeA * max(0.0, a - gKnee) * gGain), dominant);
  if (showVariance && variance > varMin) {
    float g = min(1.0, (variance - varMin) * varGain);
    c.g = c.g * (1.0 - g) + varGreen * g;
    c.r = c.r * (1.0 - g * varFade);
    c.b = c.b * (1.0 - g * varFade);
  }
  return c;
}
`;
}
