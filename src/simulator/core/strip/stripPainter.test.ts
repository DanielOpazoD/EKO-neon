import assert from 'node:assert/strict';
import test from 'node:test';
import {
  DEFAULT_SPECTRAL,
  SPECTRAL_BINS,
  spectralRange,
  type SpectralSettings,
} from '@/simulator/doppler/spectral/spectrum';
import type { TintMapId } from '@/simulator/renderer/types';
import { resolveStripTint } from '@/simulator/renderer/postprocess/tintMap';
import { baseInput } from '../baseInput';
import { simInputKey, stripAcquisitionKey } from '../acquisitionKeys';
import {
  buildSpectralRowMap,
  paintSpectralStripColumn,
  renderSpectralStripRgba,
} from './stripPainter';

function makeStripDisplay(cols: number, seed = 1): Float32Array {
  const buf = new Float32Array(cols * SPECTRAL_BINS);
  let s = seed | 0;
  for (let i = 0; i < buf.length; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) | 0;
    const u = ((s >>> 8) & 0xffffff) / 0x1000000;
    buf[i] = u;
  }
  return buf;
}

test('paintSpectralStripColumn matches renderSpectralStripRgba bit-for-bit across tint maps, reject, and baseline shifts', () => {
  const W = 96;
  const H = 220;
  const y0 = 60;
  const h = 156;
  const cols = W;
  const stripDisplay = makeStripDisplay(cols, 42);
  const map = buildSpectralRowMap(h);

  const configs: Array<{
    spectral: SpectralSettings;
    tint?: TintMapId;
    reject: number;
    headCol: number;
  }> = [
    { spectral: { ...DEFAULT_SPECTRAL, scaleMps: 1.2, baselineShiftMps: 0 }, tint: 'gray', reject: 0, headCol: 17 },
    { spectral: { ...DEFAULT_SPECTRAL, scaleMps: 4.0, baselineShiftMps: -1.2 }, tint: 'sepia', reject: 0.24, headCol: 0 },
    { spectral: { ...DEFAULT_SPECTRAL, scaleMps: 0.2, baselineShiftMps: 0.1 }, tint: 'cool-blue', reject: 0.15, headCol: W - 1 },
  ];

  for (const cfg of configs) {
    const refRgba = new Uint8ClampedArray(W * H * 4);
    const { vMin, vMax } = renderSpectralStripRgba(
      refRgba,
      W,
      H,
      y0,
      h,
      stripDisplay,
      cols,
      cfg.spectral,
      cfg.tint,
      cfg.headCol,
      cfg.reject,
    );

    const incBuf = new Uint8ClampedArray(cols * h * 4);
    const ybAbs = y0 + Math.round((vMax / (vMax - vMin)) * h);
    const ybRel = ybAbs >= y0 && ybAbs < Math.min(H, y0 + h) ? ybAbs - y0 : -1;
    for (let c = 0; c < cols; c++) {
      paintSpectralStripColumn(c, incBuf, map, stripDisplay, cols, cfg.reject, ybRel, cfg.tint);
    }

    const incRgba = new Uint8ClampedArray(W * H * 4);
    incRgba.set(incBuf, y0 * W * 4);

    let maxDiff = 0;
    for (let i = 0; i < refRgba.length; i++) {
      const d = Math.abs(refRgba[i]! - incRgba[i]!);
      if (d > maxDiff) maxDiff = d;
    }
    assert.equal(
      maxDiff,
      0,
      `Expected 0 RGBA diff for tint=${cfg.tint ?? 'gray'} reject=${cfg.reject}, got ${maxDiff}`,
    );
  }
});

test('incremental column updates on wrap-around match full renderSpectralStripRgba bit-for-bit', () => {
  const W = 64;
  const H = 180;
  const y0 = 40;
  const h = 136;
  const spectral: SpectralSettings = { ...DEFAULT_SPECTRAL, scaleMps: 1.5, baselineShiftMps: 0.3 };
  const { vMin, vMax } = spectralRange(spectral);
  const reject = 0.18;
  const tint: TintMapId = 'sepia';
  const ybAbs = y0 + Math.round((vMax / (vMax - vMin)) * h);
  const ybRel = ybAbs >= y0 && ybAbs < Math.min(H, y0 + h) ? ybAbs - y0 : -1;

  const stripDisplay = makeStripDisplay(W, 99);
  const map = buildSpectralRowMap(h);
  const incBuf = new Uint8ClampedArray(W * h * 4);
  for (let c = 0; c < W; c++) {
    paintSpectralStripColumn(c, incBuf, map, stripDisplay, W, reject, ybRel, tint);
  }

  // Overwrite 5 columns wrapping across the right edge -> col 0, 1, 2
  const updatedCols = [W - 2, W - 1, 0, 1, 2];
  for (const col of updatedCols) {
    for (let b = 0; b < SPECTRAL_BINS; b++) {
      stripDisplay[col * SPECTRAL_BINS + b] = ((b * 17 + col * 31) % 100) / 100;
    }
    paintSpectralStripColumn(col, incBuf, map, stripDisplay, W, reject, ybRel, tint);
  }

  const refRgba = new Uint8ClampedArray(W * H * 4);
  renderSpectralStripRgba(refRgba, W, H, y0, h, stripDisplay, W, spectral, tint, 3, reject);

  const incRgba = new Uint8ClampedArray(W * H * 4);
  incRgba.set(incBuf, y0 * W * 4);

  for (let i = 0; i < refRgba.length; i++) {
    assert.equal(incRgba[i], refRgba[i], `Byte mismatch at index ${i}`);
  }
});

test('independent stripTintMap resolves independently of 2D tint and never resets strip acquisition history', () => {
  assert.equal(resolveStripTint('inherit', 'gray'), 'gray');
  assert.equal(resolveStripTint('inherit', 'cool-blue'), 'cool-blue');
  assert.equal(resolveStripTint('amber', 'gray'), 'amber');
  assert.equal(resolveStripTint('gray', 'amber'), 'neutral-gray');

  // Neutral-gray explicit override renders pure achromatic pixels (R === G === B)
  const W = 16;
  const H = 64;
  const stripDisplay = new Float32Array(W * SPECTRAL_BINS).fill(0.75);
  const rgbaNeutral = new Uint8ClampedArray(W * H * 4);
  const rgbaAmber = new Uint8ClampedArray(W * H * 4);
  const sp = { ...DEFAULT_SPECTRAL, scaleMps: 1.0, baselineShiftMps: 0 };
  renderSpectralStripRgba(rgbaNeutral, W, H, 0, H, stripDisplay, W, sp, 'neutral-gray');
  renderSpectralStripRgba(rgbaAmber, W, H, 0, H, stripDisplay, W, sp, 'amber');
  // Sample non-baseline row y = 10
  const o = (10 * W + 4) * 4;
  assert.equal(rgbaNeutral[o], rgbaNeutral[o + 1], 'neutral-gray strip must keep R === G');
  assert.equal(rgbaNeutral[o + 1], rgbaNeutral[o + 2], 'neutral-gray strip must keep G === B');
  assert.ok(
    rgbaAmber[o]! > rgbaAmber[o + 2]! + 40,
    `amber strip should tint warm gold (R=${rgbaAmber[o]} > B=${rgbaAmber[o + 2]})`,
  );

  // Changing stripTintMap changes simInputKey (re-presents frozen/live frame) without resetting stripAcquisitionKey
  const a = baseInput();
  a.modality = 'pw';
  a.spectral = { ...a.spectral, stripTintMap: 'inherit' };
  const b = { ...a, spectral: { ...a.spectral, stripTintMap: 'amber' as const } };
  assert.equal(stripAcquisitionKey(a), stripAcquisitionKey(b));
  assert.notEqual(simInputKey(a), simInputKey(b));
});
