/**
 * Clinical B-mode / M-mode / Doppler chroma tint maps (GE Vivid / Philips EPIQ style).
 * Every curve maps 0 → (0,0,0) and 1 → (1,1,1) monotonically so black stays pure black,
 * peak highlights stay crisp white, and midtones gain subtle perceptual contrast.
 * Exported as both fast 256-entry CPU LUTs and matching GLSL for WebGL2 parity.
 */
export type TintMapId = 'gray' | 'cool-blue' | 'sepia' | 'amber';

/** Independent chroma selection for spectral & M-mode strips ('inherit' follows 2D B-mode tint). */
export type StripTintMapChoice = 'inherit' | TintMapId;

/** Resolved internal strip LUT id ('neutral-gray' renders pure achromatic grayscale on strips). */
export type ResolvedStripTint = TintMapId | 'neutral-gray';

export const TINT_DEFAULT: TintMapId = 'gray';

export const STRIP_TINT_DEFAULT: StripTintMapChoice = 'inherit';

export const TINT_MAP_OPTIONS: readonly { id: TintMapId; label: string }[] = [
  { id: 'gray', label: 'Gris' },
  { id: 'cool-blue', label: 'Azul' },
  { id: 'sepia', label: 'Sepia' },
  { id: 'amber', label: 'Ámbar' },
];

export const STRIP_TINT_OPTIONS: readonly { id: StripTintMapChoice; label: string }[] = [
  { id: 'inherit', label: '2D' },
  { id: 'gray', label: 'Gris' },
  { id: 'amber', label: 'Ámbar' },
  { id: 'sepia', label: 'Sepia' },
  { id: 'cool-blue', label: 'Azul' },
];

/** Resolve effective strip chroma LUT from independent strip preference and active 2D B-mode tint. */
export function resolveStripTint(
  stripTint: StripTintMapChoice | undefined,
  bmodeTint: TintMapId | undefined,
): ResolvedStripTint | undefined {
  if (!stripTint || stripTint === 'inherit') return bmodeTint;
  if (stripTint === 'gray') return 'neutral-gray';
  return stripTint;
}

export function tintMapIndex(tint: TintMapId | undefined): number {
  switch (tint) {
    case 'cool-blue':
      return 1;
    case 'sepia':
      return 2;
    case 'amber':
      return 3;
    default:
      return 0;
  }
}

/** Map a continuous scalar luminance (0..255 float) to tinted RGB bytes (0..255). */
export function tintGrayFloat(
  gFloat: number,
  tint: TintMapId | undefined,
  out: [number, number, number],
): void {
  const g = Math.max(0, Math.min(255, gFloat)) / 255;
  let r = g;
  let gn = g;
  let b = g;
  if (tint === 'cool-blue') {
    r = g * (0.84 + 0.16 * g);
    gn = g * (0.94 + 0.06 * g);
    b = Math.min(1, g * (1.14 - 0.14 * g));
  } else if (tint === 'sepia') {
    r = Math.min(1, g * (1.11 - 0.11 * g));
    gn = g * (0.95 + 0.05 * g);
    b = g * (0.76 + 0.24 * g);
  } else if (tint === 'amber') {
    r = Math.min(1, g * (1.18 - 0.18 * g));
    gn = g * (0.87 + 0.13 * g);
    b = g * (0.46 + 0.54 * g * g);
  }
  out[0] = Math.floor(r * 255 + 0.5);
  out[1] = Math.floor(gn * 255 + 0.5);
  out[2] = Math.floor(b * 255 + 0.5);
}

/** Map a scalar luminance byte (0..255) to tinted RGB bytes (0..255). */
export function tintGrayByte(
  gByte: number,
  tint: TintMapId | undefined,
  out: [number, number, number],
): void {
  tintGrayFloat(gByte, tint, out);
}

const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

function packRgb(r: number, g: number, b: number): number {
  return LITTLE_ENDIAN
    ? ((0xff000000 | (b << 16) | (g << 8) | r) >>> 0)
    : (((r << 24) | (g << 16) | (b << 8) | 0xff) >>> 0);
}

function buildPackedLut(tint: TintMapId): Uint32Array {
  const lut = new Uint32Array(256);
  const rgb: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < 256; i++) {
    tintGrayFloat(i, tint, rgb);
    lut[i] = packRgb(rgb[0], rgb[1], rgb[2]);
  }
  return lut;
}

/** 10-bit sub-gray steps per byte (0.25 gray resolution → 1021 entries covering 0..255.0). */
export const TINT_SUB_STEPS = 4;
export const TINT_SUB_MAX_INDEX = 255 * TINT_SUB_STEPS;

function buildPackedSubLut(tint: TintMapId): Uint32Array {
  const lut = new Uint32Array(TINT_SUB_MAX_INDEX + 1);
  const rgb: [number, number, number] = [0, 0, 0];
  for (let i = 0; i <= TINT_SUB_MAX_INDEX; i++) {
    tintGrayFloat(i / TINT_SUB_STEPS, tint, rgb);
    lut[i] = packRgb(rgb[0], rgb[1], rgb[2]);
  }
  return lut;
}

const PACKED_LUTS: Record<TintMapId, Uint32Array> = {
  gray: buildPackedLut('gray'),
  'cool-blue': buildPackedLut('cool-blue'),
  sepia: buildPackedLut('sepia'),
  amber: buildPackedLut('amber'),
};

const PACKED_SUB_LUTS: Record<TintMapId, Uint32Array> = {
  gray: buildPackedSubLut('gray'),
  'cool-blue': buildPackedSubLut('cool-blue'),
  sepia: buildPackedSubLut('sepia'),
  amber: buildPackedSubLut('amber'),
};

/** 256-entry packed RGBA32 LUT for O(1) strip painting and byte-indexed callers. */
export function tintPackedLut(tint: TintMapId | undefined): Uint32Array {
  return PACKED_LUTS[tint ?? 'gray'] ?? PACKED_LUTS.gray;
}

/** 1021-entry (10-bit, 0.25-gray step) packed RGBA32 LUT for continuous bicubic scan conversion. */
export function tintPackedSubLut(tint: TintMapId | undefined): Uint32Array {
  return PACKED_SUB_LUTS[tint ?? 'gray'] ?? PACKED_SUB_LUTS.gray;
}

/** Matching GLSL helper returning RGB in 0..255 byte scale (integer-rounded like the CPU LUT). */
export function tintMapGlsl(): string {
  return /* glsl */ `
vec3 applyTintMap(float gByte, int tintId) {
  float g = clamp(gByte, 0.0, 255.0) * (1.0 / 255.0);
  vec3 c = vec3(g);
  if (tintId == 1) {
    c = vec3(g * (0.84 + 0.16 * g), g * (0.94 + 0.06 * g), min(1.0, g * (1.14 - 0.14 * g)));
  } else if (tintId == 2) {
    c = vec3(min(1.0, g * (1.11 - 0.11 * g)), g * (0.95 + 0.05 * g), g * (0.76 + 0.24 * g));
  } else if (tintId == 3) {
    c = vec3(min(1.0, g * (1.18 - 0.18 * g)), g * (0.87 + 0.13 * g), g * (0.46 + 0.54 * g * g));
  }
  return floor(c * 255.0 + 0.5);
}
`;
}
