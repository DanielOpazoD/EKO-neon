import {
  COLOR_THRESHOLD,
  colorMap,
  type ColorSettings,
} from '@/simulator/doppler/color/colorDoppler';
import {
  colorPowerConfidence,
  ST_POW_ALPHA_MIN,
} from '@/simulator/doppler/color/spatiotemporalFilter';
import { SPECTRAL_BINS, spectralRange, type SpectralSettings } from '@/simulator/doppler/spectral/spectrum';
import { COLOR_BLEND } from '@/simulator/renderer/postprocess/colorMap';
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
  cmmVar?: Float32Array | null,
  cmmPow?: Float32Array | null,
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
    const scale = Math.max(1e-3, color.scaleMps);
    const baseline = color.baselineShiftMps;
    const kPhi = Math.PI / scale;
    const invPiScale = scale / Math.PI;
    const rgb: [number, number, number] = [0, 0, 0];
    const powAlphaAt = (idx: number): number => {
      if (!cmmPow) return 1;
      const conf = colorPowerConfidence(cmmPow[idx] ?? 1, COLOR_THRESHOLD);
      return ST_POW_ALPHA_MIN + (1 - ST_POW_ALPHA_MIN) * conf;
    };
    for (let y = 0; y < rows; y++) {
      const sf = ((y + 0.5) / rows) * F - 0.5;
      const s0 = Math.max(0, Math.min(F - 1, Math.floor(sf)));
      const s1 = Math.min(F - 1, s0 + 1);
      const f1 = Math.max(0, Math.min(1, sf - s0));
      const f0 = 1 - f1;
      const i0 = s0 * cols + col;
      const i1 = s1 * cols + col;
      const v0 = cmm[i0]!;
      const v1 = cmm[i1]!;
      if (v0 !== v0 && v1 !== v1) continue;

      let re = 0;
      let im = 0;
      let varAcc = 0;
      let powAcc = 0;
      let wSum = 0;
      if (v0 === v0) {
        const ph = (v0 - baseline) * kPhi;
        re += f0 * Math.cos(ph);
        im += f0 * Math.sin(ph);
        varAcc += f0 * (cmmVar ? (cmmVar[i0] ?? 0) : 0);
        powAcc += f0 * powAlphaAt(i0);
        wSum += f0;
      }
      if (v1 === v1) {
        const ph = (v1 - baseline) * kPhi;
        re += f1 * Math.cos(ph);
        im += f1 * Math.sin(ph);
        varAcc += f1 * (cmmVar ? (cmmVar[i1] ?? 0) : 0);
        powAcc += f1 * powAlphaAt(i1);
        wSum += f1;
      }
      if (wSum <= 0.18) continue;

      // cmm values are already inverted and aliased into [baseline - scale, baseline + scale]
      const v = baseline + invPiScale * Math.atan2(im, re);
      const vari = varAcc / wSum;
      const powAlpha = powAcc / wSum;
      colorMap(v, scale, vari, color.showVariance, rgb, baseline);

      const edgeT = Math.min(1, Math.max(0, (wSum - 0.18) / 0.54));
      const edgeAlpha = edgeT * edgeT * (3 - 2 * edgeT);
      const o = (y * cols + col) * 4;
      const r0 = img[o] ?? 0;
      const g0 = img[o + 1] ?? 0;
      const b0 = img[o + 2] ?? 0;
      const lum = 0.299 * r0 + 0.587 * g0 + 0.114 * b0;
      // Clinical tissue write priority: preserves bright specular leaflet excursion lines through color plumes
      const tpU = Math.min(1, Math.max(0, (lum - 98) / 74));
      const tissuePrio = 1 - tpU * tpU * (3 - 2 * tpU);
      const velU = Math.min(1, Math.max(0, (Math.abs(v) / scale - 0.05) / 0.17));
      const velAlpha = 0.62 + 0.38 * velU * velU * (3 - 2 * velU);
      const blend = COLOR_BLEND * edgeAlpha * powAlpha * velAlpha * tissuePrio;
      const inv = 1 - blend;
      img[o] = Math.round(rgb[0] * blend + r0 * inv);
      img[o + 1] = Math.round(rgb[1] * blend + g0 * inv);
      img[o + 2] = Math.round(rgb[2] * blend + b0 * inv);
    }
  }
}

/** Rasterize the displayed spectral Doppler strip with bilinear resampling, baseline, and tint map. */
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
  headCol?: number,
): { vMin: number; vMax: number } {
  const { vMin, vMax } = spectralRange(spectral);
  const customTint = tintMap && tintMap !== 'gray';
  const rgb: [number, number, number] = [0, 0, 0];
  const seamCol = headCol !== undefined && cols > 1 ? ((headCol % cols) + cols) % cols : -1;
  const maxBin = SPECTRAL_BINS - 1;
  const maxCol = Math.max(0, cols - 1);

  for (let y = 0; y < h; y++) {
    const bf = ((y + 0.5) / h) * SPECTRAL_BINS - 0.5;
    const b0 = Math.max(0, Math.min(maxBin, Math.floor(bf)));
    const b1 = Math.min(maxBin, b0 + 1);
    const fy = Math.max(0, Math.min(1, bf - b0));
    const fy0 = 1 - fy;
    const rowOff = (y0 + y) * W;

    for (let x = 0; x < W; x++) {
      const cf = ((x + 0.5) / W) * cols - 0.5;
      let c0 = Math.max(0, Math.min(maxCol, Math.floor(cf)));
      let c1 = Math.min(maxCol, c0 + 1);
      let fx = Math.max(0, Math.min(1, cf - c0));
      // Do not blend across the sweep-head discontinuity between the newest column and stale data.
      if (seamCol >= 0 && c0 !== c1 && c1 === seamCol) {
        if (fx < 0.5) {
          c1 = c0;
          fx = 0;
        } else {
          c0 = c1;
          fx = 0;
        }
      }
      const fx0 = 1 - fx;
      const base0 = c0 * SPECTRAL_BINS;
      const base1 = c1 * SPECTRAL_BINS;
      const v0 = (stripDisplay[base0 + b0] ?? 0) * fx0 + (stripDisplay[base1 + b0] ?? 0) * fx;
      const v1 = (stripDisplay[base0 + b1] ?? 0) * fx0 + (stripDisplay[base1 + b1] ?? 0) * fx;
      const v = v0 * fy0 + v1 * fy;
      const g = Math.round(Math.max(0, Math.min(1, v)) * 255);
      const o = (rowOff + x) * 4;
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
  if (yb >= y0 && yb < Math.min(H, y0 + h)) {
    for (let x = 0; x < W; x++) {
      const o = (yb * W + x) * 4;
      rgba[o] = 90;
      rgba[o + 1] = 160;
      rgba[o + 2] = 90;
    }
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
