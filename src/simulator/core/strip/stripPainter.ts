import { colorMap, type ColorSettings } from '@/simulator/doppler/color/colorDoppler';
import { SPECTRAL_BINS, spectralRange, type SpectralSettings } from '@/simulator/doppler/spectral/spectrum';
import { tintGrayByte } from '@/simulator/renderer/postprocess/tintMap';
import type { TintMapId } from '@/simulator/renderer/types';
import type { RowMap } from '../mmodeStrip';

/** Paint one column of the displayed M-mode (and optional Colour M-mode) strip RGBA buffer. */
export function paintMmodeStripColumn(
  col: number,
  img: Uint8ClampedArray,
  map: RowMap,
  strip: Uint8ClampedArray,
  cols: number,
  samples: number,
  tint: TintMapId | undefined,
  cmm: Float32Array | null,
  color: ColorSettings | undefined,
): void {
  if (map.samples !== samples) return;
  const { rows, taps, first, weights } = map;
  const rgbGray: [number, number, number] = [0, 0, 0];
  for (let y = 0; y < rows; y++) {
    let g = 0;
    const i0 = first[y]!;
    for (let k = 0; k < taps; k++) {
      const w = weights[y * taps + k]!;
      if (w > 0 && i0 + k < samples) g += w * strip[(i0 + k) * cols + col]!;
    }
    const o = (y * cols + col) * 4;
    if (tint && tint !== 'gray') {
      tintGrayByte(Math.round(g), tint, rgbGray);
      img[o] = rgbGray[0];
      img[o + 1] = rgbGray[1];
      img[o + 2] = rgbGray[2];
    } else {
      img[o] = g;
      img[o + 1] = g;
      img[o + 2] = g;
    }
    img[o + 3] = 255;
  }
  if (cmm && color) {
    const F = cmm.length / cols;
    const rgb: [number, number, number] = [0, 0, 0];
    for (let y = 0; y < rows; y++) {
      const v = cmm[Math.min(F - 1, Math.floor((y / rows) * F)) * cols + col]!;
      if (Number.isNaN(v)) continue;
      colorMap(color.invert ? -v : v, color.scaleMps, 0, false, rgb);
      const o = (y * cols + col) * 4;
      img[o] = rgb[0];
      img[o + 1] = rgb[1];
      img[o + 2] = rgb[2];
    }
  }
}

/** Rasterize the displayed spectral Doppler strip with baseline and tint map into the frame RGBA buffer. */
export function renderSpectralStripRgba(
  rgba: Uint8ClampedArray,
  W: number,
  H: number,
  y0: number,
  h: number,
  stripDisplay: Float32Array,
  cols: number,
  spectral: SpectralSettings,
  tintMap?: TintMapId,
): { vMin: number; vMax: number } {
  const { vMin, vMax } = spectralRange(spectral);
  const customTint = tintMap && tintMap !== 'gray';
  const rgb: [number, number, number] = [0, 0, 0];
  for (let y = 0; y < h; y++) {
    const b = Math.min(SPECTRAL_BINS - 1, Math.floor((y / h) * SPECTRAL_BINS));
    for (let x = 0; x < W; x++) {
      const col = Math.min(cols - 1, Math.floor((x * cols) / W));
      const v = stripDisplay[col * SPECTRAL_BINS + b] ?? 0;
      const g = Math.round(Math.min(1, v) * 255);
      const o = ((y0 + y) * W + x) * 4;
      if (customTint) {
        tintGrayByte(g, tintMap, rgb);
        rgba[o] = rgb[0];
        rgba[o + 1] = rgb[1];
        rgba[o + 2] = rgb[2];
      } else {
        rgba[o] = g;
        rgba[o + 1] = Math.round(g * 0.93);
        rgba[o + 2] = Math.round(g * 0.8);
      }
      rgba[o + 3] = 255;
    }
  }
  const yb = y0 + Math.round((vMax / (vMax - vMin)) * h);
  for (let x = 0; x < W; x++) {
    const o = (Math.min(H - 1, yb) * W + x) * 4;
    rgba[o] = 90;
    rgba[o + 1] = 160;
    rgba[o + 2] = 90;
  }
  return { vMin, vMax };
}

/** Draw the vertical green sweep-head bar over the active strip region. */
export function drawSweepMarker(
  rgba: Uint8ClampedArray,
  W: number,
  y0: number,
  h: number,
  head: number,
): void {
  const x = Math.min(W - 1, head);
  for (let y = 0; y < h; y++) {
    const o = ((y0 + y) * W + x) * 4;
    rgba[o] = 120;
    rgba[o + 1] = 200;
    rgba[o + 2] = 120;
  }
}
