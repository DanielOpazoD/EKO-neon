import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildScanLut,
  computeSectorMapping,
  lutKey,
  packScanLutTexels,
  pixelToPolar,
  polarToPixel,
  scanConvertLut,
  scanConvertLutReference,
  LUT_TEXEL_INSIDE,
} from './scanConvert';
import { DEFAULT_ACQUISITION, polarSpecFor, type PolarFrameSpec } from './types';

const SPEC: PolarFrameSpec = {
  lines: 64,
  samples: 96,
  sectorRad: Math.PI / 2,
  depthCm: 16,
  focusCm: 8,
  elevationSamples: 1,
};

test('polarToPixel and pixelToPolar are exact inverses for normal, invertLR, and zoom mappings', () => {
  for (const invertLR of [false, true]) {
    for (const zoom of [1, 1.5]) {
      const m = computeSectorMapping(SPEC, 320, 240, invertLR, zoom);
      for (const rCm of [1.5, 6.0, 12.5]) {
        for (const thetaRad of [-0.6, 0, 0.45]) {
          const pt = polarToPixel(m, rCm, thetaRad);
          const back = pixelToPolar(m, pt.x, pt.y);
          assert.ok(Math.abs(back.rCm - rCm) < 1e-9, `r mismatch (${back.rCm} vs ${rCm})`);
          assert.ok(
            Math.abs(back.thetaRad - thetaRad) < 1e-9,
            `theta mismatch (${back.thetaRad} vs ${thetaRad})`,
          );
        }
      }
    }
  }
});

test('invertLR mirrors horizontal pixel coordinates symmetrically around apexX', () => {
  const mNorm = computeSectorMapping(SPEC, 300, 240, false, 1);
  const mInv = computeSectorMapping(SPEC, 300, 240, true, 1);
  const pNorm = polarToPixel(mNorm, 8, 0.35);
  const pInv = polarToPixel(mInv, 8, 0.35);
  assert.ok(Math.abs(pNorm.y - pInv.y) < 1e-9);
  assert.ok(Math.abs((pNorm.x - mNorm.apexX) + (pInv.x - mInv.apexX)) < 1e-9);
  assert.notEqual(lutKey(SPEC, mNorm), lutKey(SPEC, mInv));
});

test('buildScanLut masks outside-sector pixels as -1 and packs matching GPU LUT texels', () => {
  const m = computeSectorMapping(SPEC, 160, 120, false, 1);
  const lut = buildScanLut(SPEC, m);

  // Corners above/beside apex are outside sector
  assert.equal(lut.idx[0], -1);
  assert.equal(lut.idx[159], -1);
  // Point along central axis at mid depth must lie inside sector
  const mid = polarToPixel(m, 8, 0);
  const midIdx = Math.round(mid.y) * m.width + Math.round(mid.x);
  assert.ok(lut.idx[midIdx]! >= 0, 'mid-sector pixel must be inside');

  // Compact gather list count matches total inside pixels
  let insideCount = 0;
  for (let i = 0; i < lut.idx.length; i++) if (lut.idx[i]! >= 0) insideCount++;
  assert.equal(lut.inPix.length, insideCount);
  assert.ok(insideCount > 0 && insideCount < m.width * m.height);

  // GPU packed texels mark exact same inside pixels with LUT_TEXEL_INSIDE bit
  const texels = packScanLutTexels(lut);
  let packedInside = 0;
  for (let i = 0; i < m.width * m.height; i++) {
    const inside = (texels[i * 4 + 2]! & LUT_TEXEL_INSIDE) !== 0;
    assert.equal(inside, lut.idx[i]! >= 0);
    if (inside) packedInside++;
  }
  assert.equal(packedInside, insideCount);
});

test('scanConvertLut fast path preserves uniform fields and agrees with float reference within 1 gray level', () => {
  const m = computeSectorMapping(SPEC, 160, 120, false, 1);
  const lut = buildScanLut(SPEC, m);
  const polar = new Uint8ClampedArray(SPEC.lines * SPEC.samples);
  const rgbaFast = new Uint8ClampedArray(m.width * m.height * 4);
  const rgbaRef = new Uint8ClampedArray(m.width * m.height * 4);

  // 1) Uniform field → every inside pixel equals constant, outside pixels are opaque black
  polar.fill(142);
  scanConvertLut(polar, lut, rgbaFast);
  for (let i = 0; i < m.width * m.height; i++) {
    const o = i * 4;
    if (lut.idx[i]! < 0) {
      assert.equal(rgbaFast[o], 0);
      assert.equal(rgbaFast[o + 1], 0);
      assert.equal(rgbaFast[o + 2], 0);
      assert.equal(rgbaFast[o + 3], 255);
    } else {
      assert.equal(rgbaFast[o], 142);
      assert.equal(rgbaFast[o + 1], 142);
      assert.equal(rgbaFast[o + 2], 142);
      assert.equal(rgbaFast[o + 3], 255);
    }
  }

  // 2) Non-uniform ramp + high-contrast edge → fast Catmull-Rom matches float reference to ≤ 1 gray level
  for (let li = 0; li < SPEC.lines; li++) {
    for (let si = 0; si < SPEC.samples; si++) {
      polar[li * SPEC.samples + si] = si > 40 && si < 55 ? 230 : Math.min(255, li * 3 + si);
    }
  }
  scanConvertLut(polar, lut, rgbaFast);
  scanConvertLutReference(polar, lut, rgbaRef);
  let maxDiff = 0;
  for (let i = 0; i < rgbaFast.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(rgbaFast[i]! - rgbaRef[i]!));
  }
  assert.ok(maxDiff <= 1, `fast vs reference maxDiff=${maxDiff} exceeded 1 gray level`);
});

test('focalized HD Write-Zoom centers focusCm, preserves exact caliper calibration, invalidates lutKey, and boosts polar resolution', () => {
  const deepSpec: PolarFrameSpec = { ...SPEC, focusCm: 9 };
  const shallowSpec: PolarFrameSpec = { ...SPEC, focusCm: 5 };
  const m1x = computeSectorMapping(deepSpec, 640, 480, false, 1);
  const m2x = computeSectorMapping(deepSpec, 640, 480, false, 2);
  const m2xShallow = computeSectorMapping(shallowSpec, 640, 480, false, 2);

  // At 2× zoom, pxPerCm doubles and focusCm sits comfortably near vertical canvas center (not pushed off bottom!)
  assert.ok(Math.abs(m2x.pxPerCm - m1x.pxPerCm * 2) < 1e-6);
  assert.ok(m2x.apexY < m1x.apexY, 'zooming on mid/deep focus shifts apex upward to keep focus centered');
  const focusPt = polarToPixel(m2x, deepSpec.focusCm, 0);
  assert.ok(
    focusPt.y >= 480 * 0.3 && focusPt.y <= 480 * 0.7,
    `expected focusCm centered in [144, 336], got y=${focusPt.y.toFixed(1)}`,
  );

  // Exact polarToPixel <-> pixelToPolar inversion at 2× zoom (caliper accuracy)
  for (const rCm of [5, 9, 12.5]) {
    for (const th of [-0.35, 0, 0.35]) {
      const p = polarToPixel(m2x, rCm, th);
      const back = pixelToPolar(m2x, p.x, p.y);
      assert.ok(Math.abs(back.rCm - rCm) < 1e-9);
      assert.ok(Math.abs(back.thetaRad - th) < 1e-9);
    }
  }

  // Shifting focus while zoomed updates apexY and invalidates lutKey
  assert.notEqual(m2x.apexY, m2xShallow.apexY);
  assert.notEqual(lutKey(deepSpec, m2x), lutKey(shallowSpec, m2xShallow));

  // Real Write-Zoom boosts polar scan-line & sample density cleanly within safe caps
  const p1 = polarSpecFor({ ...DEFAULT_ACQUISITION, zoom: 1 }, 'high');
  const p2 = polarSpecFor({ ...DEFAULT_ACQUISITION, zoom: 2 }, 'high');
  assert.ok(p2.lines > p1.lines, `expected zoomed lines ${p2.lines} > ${p1.lines}`);
  assert.ok(p2.samples > p1.samples, `expected zoomed samples ${p2.samples} > ${p1.samples}`);
  assert.ok(p2.lines <= 320 && p2.samples <= 512, 'HD zoom grid must stay within bounded limits');
});
