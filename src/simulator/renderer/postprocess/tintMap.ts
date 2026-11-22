/**
 * Clinical B-mode / M-mode / Doppler chroma tint maps (GE Vivid / Philips EPIQ style).
 * Every curve maps 0 → (0,0,0) and 1 → (1,1,1) monotonically so black stays pure black,
 * peak highlights stay crisp white, and midtones gain subtle perceptual contrast.
 * Exported as both fast 256-entry CPU LUTs and matching GLSL for WebGL2 parity.
 */
export type TintMapId = 'gray' | 'cool-blue' | 'sepia' | 'amber';

export const TINT_DEFAULT: TintMapId = 'gray';

export const TINT_MAP_OPTIONS: readonly { id: TintMapId; label: string }[] = [
  { id: 'gray', label: 'Gris' },
  { id: 'cool-blue', label: 'Azul' },
  { id: 'sepia', label: 'Sepia' },
  { id: 'amber', label: 'Ámbar' },
];

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

/** Map a scalar luminance byte (0..255) to tinted RGB bytes (0..255). */
export function tintGrayByte(
  gByte: number,
  tint: TintMapId | undefined,
  out: [number, number, number],
): void {
  const g = Math.max(0, Math.min(255, gByte)) / 255;
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
  out[0] = Math.round(r * 255);
  out[1] = Math.round(gn * 255);
  out[2] = Math.round(b * 255);
}

const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

function buildPackedLut(tint: TintMapId): Uint32Array {
  const lut = new Uint32Array(256);
  const rgb: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < 256; i++) {
    tintGrayByte(i, tint, rgb);
    const r = rgb[0];
    const g = rgb[1];
    const b = rgb[2];
    lut[i] = LITTLE_ENDIAN
      ? ((0xff000000 | (b << 16) | (g << 8) | r) >>> 0)
      : (((r << 24) | (g << 16) | (b << 8) | 0xff) >>> 0);
  }
  return lut;
}

const PACKED_LUTS: Record<TintMapId, Uint32Array> = {
  gray: buildPackedLut('gray'),
  'cool-blue': buildPackedLut('cool-blue'),
  sepia: buildPackedLut('sepia'),
  amber: buildPackedLut('amber'),
};

/** 256-entry packed RGBA32 LUT for O(1) scan-conversion and strip painting. */
export function tintPackedLut(tint: TintMapId | undefined): Uint32Array {
  return PACKED_LUTS[tint ?? 'gray'] ?? PACKED_LUTS.gray;
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
