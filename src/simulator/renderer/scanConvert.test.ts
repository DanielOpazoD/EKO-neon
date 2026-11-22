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
import type { PolarFrameSpec } from './types';

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
