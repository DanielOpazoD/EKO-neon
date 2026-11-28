import type { PolarFrameSpec, TintMapId } from './types';
import {
  tintGrayFloat,
  tintPackedSubLut,
  TINT_SUB_MAX_INDEX,
  TINT_SUB_STEPS,
} from './postprocess/tintMap';

/**
 * Sector scan conversion: polar (line, sample) → cartesian RGBA. Apex at top-centre, radius = depth.
 * A lookup table (bilinear indices + weights + polar coordinates per pixel) is built once per
 * mapping and reused, so per-frame work is a gather. The mapping is also the single source of the
 * pixel↔cm geometry used by calipers and overlays.
 */
export interface SectorMapping {
  apexX: number;
  apexY: number;
  pxPerCm: number;
  width: number;
  height: number;
  sectorRad: number;
  depthCm: number;
  invertLR: boolean;
}

export function computeSectorMapping(
  spec: PolarFrameSpec,
  width: number,
  height: number,
  invertLR: boolean,
  zoom = 1,
): SectorMapping {
  const margin = 14;
  const usableH = height - margin * 2;
  const halfAngle = spec.sectorRad / 2;
  const usableW = width - margin * 2;
  const pxByH = usableH / spec.depthCm;
  const pxByW = usableW / (2 * spec.depthCm * Math.sin(halfAngle));
  const pxPerCm = Math.min(pxByH, pxByW) * zoom;
  let apexY = margin;
  if (zoom > 1.001) {
    // Focalized HD zoom (Write-Zoom): keep the focal depth centered in the viewport instead of
    // pinning the transducer apex at the top edge and pushing mid/deep valves off-screen.
    const rawFocus = spec.focusCm > 0 ? spec.focusCm : spec.depthCm * 0.5;
    const centerR = Math.max(spec.depthCm * 0.2, Math.min(spec.depthCm * 0.88, rawFocus));
    const targetApexY = height * 0.5 - centerR * pxPerCm;
    const minApexY = Math.min(margin, height - margin - spec.depthCm * pxPerCm);
    apexY = Math.max(minApexY, Math.min(margin, targetApexY));
  }
  return {
    apexX: width / 2,
    apexY,
    pxPerCm,
    width,
    height,
    sectorRad: spec.sectorRad,
    depthCm: spec.depthCm,
    invertLR,
  };
}

export function pixelToPolar(
  m: SectorMapping,
  px: number,
  py: number,
): { rCm: number; thetaRad: number } {
  const dx = (px - m.apexX) * (m.invertLR ? -1 : 1);
  const dy = py - m.apexY;
  return { rCm: Math.hypot(dx, dy) / m.pxPerCm, thetaRad: Math.atan2(dx, dy) };
}

export function polarToPixel(
  m: SectorMapping,
  rCm: number,
  thetaRad: number,
): { x: number; y: number } {
  const dx = rCm * m.pxPerCm * Math.sin(thetaRad) * (m.invertLR ? -1 : 1);
  const dy = rCm * m.pxPerCm * Math.cos(thetaRad);
  return { x: m.apexX + dx, y: m.apexY + dy };
}

export interface ScanLut {
  key: string;
  idx: Int32Array; // polar index of the (l0,s0) corner, −1 outside the sector
  /** Compact gather lists over the pixels inside the sector (per-frame fast path). */
  inPix: Int32Array; // pixel index
  inIdx: Int32Array; // polar index of the (l0,s0) corner
  inWl: Uint16Array; // line weight × 1024
  inWs: Uint16Array; // sample weight × 1024
  inDl: Uint8Array; // 1 when a next line exists and carries weight
  inDs: Uint8Array; // 1 when a next sample exists and carries weight
  /** Precomputed line and sample indices for inside pixels (avoids per-pixel division in scanConvertLut). */
  inL0?: Int16Array;
  inS0?: Int16Array;
  wl: Float32Array; // line interpolation weight
  ws: Float32Array; // sample interpolation weight
  li: Int16Array; // nearest line index (for colour/masks)
  si: Int16Array; // nearest sample index
  rCm: Float32Array;
  theta: Float32Array;
  samples: number;
  lines: number;
}

export function lutKey(spec: PolarFrameSpec, m: SectorMapping): string {
  return `${spec.lines}x${spec.samples}|${spec.sectorRad.toFixed(4)}|${spec.depthCm}|${m.width}x${m.height}|${m.apexX.toFixed(2)},${m.apexY.toFixed(2)}|${m.pxPerCm.toFixed(3)}|${m.invertLR ? 1 : 0}`;
}

export function buildScanLut(spec: PolarFrameSpec, m: SectorMapping): ScanLut {
  const { lines, samples, sectorRad, depthCm } = spec;
  const half = sectorRad / 2;
  const w = m.width,
    h = m.height;
  const n = w * h;
  const idx = new Int32Array(n).fill(-1);
  const wl = new Float32Array(n);
  const ws = new Float32Array(n);
  const li = new Int16Array(n);
  const si = new Int16Array(n);
  const rArr = new Float32Array(n);
  const thArr = new Float32Array(n);
  const inPix = new Int32Array(n);
  const inIdx = new Int32Array(n);
  const inWl = new Uint16Array(n);
  const inWs = new Uint16Array(n);
  const inDl = new Uint8Array(n);
  const inDs = new Uint8Array(n);
  const inL0 = new Int16Array(n);
  const inS0 = new Int16Array(n);
  let count = 0;
  const invPx = 1 / m.pxPerCm;
  const maxR = depthCm;
  const tanHalf = Math.tan(half);
  for (let y = 0; y < h; y++) {
    const dy = y - m.apexY;
    if (dy < 0) continue;
    const xLimit = dy * tanHalf;
    for (let x = 0; x < w; x++) {
      let dx = x - m.apexX;
      if (m.invertLR) dx = -dx;
      if (dx > xLimit || dx < -xLimit) continue;
      const r = Math.sqrt(dx * dx + dy * dy) * invPx;
      if (r > maxR) continue;
      const th = Math.atan2(dx, dy);
      if (th < -half || th > half) continue;
      const p = y * w + x;
      const lf = ((th + half) / sectorRad) * lines - 0.5;
      const sf = (r / depthCm) * samples - 0.5;
      let l0 = Math.floor(lf),
        s0 = Math.floor(sf);
      let lt = lf - l0,
        st = sf - s0;
      if (l0 < 0) {
        l0 = 0;
        lt = 0;
      }
      if (s0 < 0) {
        s0 = 0;
        st = 0;
      }
      if (l0 >= lines - 1) {
        l0 = lines - 1;
        lt = 0;
      }
      if (s0 >= samples - 1) {
        s0 = samples - 1;
        st = 0;
      }
      idx[p] = l0 * samples + s0;
      wl[p] = lt;
      ws[p] = st;
      // 10-bit fixed-point weights keep the whole bilinear sum inside int32 (255·1024² < 2³¹)
      const wlQ = Math.floor(lt * 1024);
      const wsQ = Math.floor(st * 1024);
      inPix[count] = p;
      inIdx[count] = l0 * samples + s0;
      inWl[count] = wlQ;
      inWs[count] = wsQ;
      inDl[count] = wlQ > 0 && l0 < lines - 1 ? 1 : 0;
      inDs[count] = wsQ > 0 && s0 < samples - 1 ? 1 : 0;
      inL0[count] = l0;
      inS0[count] = s0;
      count++;
      li[p] = Math.min(lines - 1, Math.round(lf));
      si[p] = Math.min(samples - 1, Math.round(sf));
      rArr[p] = r;
      thArr[p] = th;
    }
  }
  return {
    key: lutKey(spec, m),
    idx,
    wl,
    ws,
    li,
    si,
    rCm: rArr,
    theta: thArr,
    samples,
    lines,
    inPix: inPix.slice(0, count),
    inIdx: inIdx.slice(0, count),
    inWl: inWl.slice(0, count),
    inWs: inWs.slice(0, count),
    inDl: inDl.slice(0, count),
    inDs: inDs.slice(0, count),
    inL0: inL0.slice(0, count),
    inS0: inS0.slice(0, count),
  };
}

const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

/** Precomputed 1024×4 Catmull-Rom basis weights (16 KB, L1-cache resident) for quantized 10-bit fractional coords. */
const CR_WEIGHTS = (() => {
  const lut = new Float32Array(1024 * 4);
  for (let i = 0; i < 1024; i++) {
    const t = i * (1 / 1024);
    const t2 = t * t;
    const t3 = t2 * t;
    const o = i << 2;
    lut[o] = -0.5 * t + t2 - 0.5 * t3;
    lut[o + 1] = 1 - 2.5 * t2 + 1.5 * t3;
    lut[o + 2] = 0.5 * t + 2 * t2 - 1.5 * t3;
    lut[o + 3] = -0.5 * t2 + 0.5 * t3;
  }
  return lut;
})();

/**
 * Gather polar intensities into RGBA using a prebuilt LUT. Pixels outside the sector are black.
 * Uses clamped Catmull-Rom 4×4 bicubic interpolation (anti-ringing clamped to the local 2×2
 * range) for crisp valve/endocardial borders in the far field, followed by an O(1) 256-entry
 * clinical chroma/tint LUT lookup. Matches GPU `GLSL_PRESENT_FRAG` to ≤ 1 gray level.
 */
export function scanConvertLut(
  polar: Uint8ClampedArray,
  lut: ScanLut,
  rgba: Uint8ClampedArray,
  tintMap?: TintMapId,
): void {
  const n = lut.idx.length;
  if (LITTLE_ENDIAN && rgba.byteOffset % 4 === 0 && rgba.length >= n * 4) {
    const px = new Uint32Array(rgba.buffer, rgba.byteOffset, n);
    px.fill(0xff000000);
    const tintLut = tintPackedSubLut(tintMap);
    const cr = CR_WEIGHTS;
    const S = lut.samples;
    const L = lut.lines;
    const maxS = S - 1;
    const maxL = L - 1;
    const maxS2 = S - 2;
    const maxL2 = L - 2;
    const { inPix, inIdx, inWl, inWs, inL0, inS0 } = lut;
    const m = inPix.length;
    for (let k = 0; k < m; k++) {
      let l0: number;
      let s0: number;
      if (inL0 && inS0) {
        l0 = inL0[k]!;
        s0 = inS0[k]!;
      } else {
        const i0 = inIdx[k]!;
        l0 = (i0 / S) | 0;
        s0 = i0 - l0 * S;
      }

      const os = inWs[k]! << 2;
      const ws0 = cr[os]!;
      const ws1 = cr[os + 1]!;
      const ws2 = cr[os + 2]!;
      const ws3 = cr[os + 3]!;

      const ol = inWl[k]! << 2;
      const wl0 = cr[ol]!;
      const wl1 = cr[ol + 1]!;
      const wl2 = cr[ol + 2]!;
      const wl3 = cr[ol + 3]!;

      let c0: number;
      let c1: number;
      let c2: number;
      let c3: number;
      let v00: number;
      let v01: number;
      let v10: number;
      let v11: number;

      if (l0 > 0 && l0 < maxL2 && s0 > 0 && s0 < maxS2) {
        // Interior fast path (~97% of sector pixels): zero boundary clamps
        const r1 = l0 * S + s0;
        const r0 = r1 - S;
        const r2 = r1 + S;
        const r3 = r2 + S;

        c0 = polar[r0 - 1]! * ws0 + polar[r0]! * ws1 + polar[r0 + 1]! * ws2 + polar[r0 + 2]! * ws3;
        v00 = polar[r1]!;
        v01 = polar[r1 + 1]!;
        c1 = polar[r1 - 1]! * ws0 + v00 * ws1 + v01 * ws2 + polar[r1 + 2]! * ws3;
        v10 = polar[r2]!;
        v11 = polar[r2 + 1]!;
        c2 = polar[r2 - 1]! * ws0 + v10 * ws1 + v11 * ws2 + polar[r2 + 2]! * ws3;
        c3 = polar[r3 - 1]! * ws0 + polar[r3]! * ws1 + polar[r3 + 1]! * ws2 + polar[r3 + 2]! * ws3;
      } else {
        const sm1 = s0 > 0 ? s0 - 1 : 0;
        const s1 = s0 < maxS ? s0 + 1 : maxS;
        const s2 = s0 + 2 <= maxS ? s0 + 2 : maxS;

        const lm1 = l0 > 0 ? l0 - 1 : 0;
        const l1 = l0 < maxL ? l0 + 1 : maxL;
        const l2 = l0 + 2 <= maxL ? l0 + 2 : maxL;

        const r0 = lm1 * S;
        const r1 = l0 * S;
        const r2 = l1 * S;
        const r3 = l2 * S;

        c0 = polar[r0 + sm1]! * ws0 + polar[r0 + s0]! * ws1 + polar[r0 + s1]! * ws2 + polar[r0 + s2]! * ws3;
        v00 = polar[r1 + s0]!;
        v01 = polar[r1 + s1]!;
        c1 = polar[r1 + sm1]! * ws0 + v00 * ws1 + v01 * ws2 + polar[r1 + s2]! * ws3;
        v10 = polar[r2 + s0]!;
        v11 = polar[r2 + s1]!;
        c2 = polar[r2 + sm1]! * ws0 + v10 * ws1 + v11 * ws2 + polar[r2 + s2]! * ws3;
        c3 = polar[r3 + sm1]! * ws0 + polar[r3 + s0]! * ws1 + polar[r3 + s1]! * ws2 + polar[r3 + s2]! * ws3;
      }

      let val = c0 * wl0 + c1 * wl1 + c2 * wl2 + c3 * wl3;
      // Local 2×2 min/max anti-ringing clamp without variadic Math.min/max overhead
      const loA = v00 < v10 ? v00 : v10;
      const hiA = v00 > v10 ? v00 : v10;
      const loB = v01 < v11 ? v01 : v11;
      const hiB = v01 > v11 ? v01 : v11;
      const lo = loA < loB ? loA : loB;
      const hi = hiA > hiB ? hiA : hiB;
      if (val < lo) val = lo;
      else if (val > hi) val = hi;
      const subIdx = (val * TINT_SUB_STEPS) | 0;
      px[inPix[k]!] =
        tintLut[subIdx < 0 ? 0 : subIdx > TINT_SUB_MAX_INDEX ? TINT_SUB_MAX_INDEX : subIdx]!;
    }
    return;
  }
  scanConvertLutReference(polar, lut, rgba, tintMap);
}

/** Float clamped Catmull-Rom gather (reference for the fast path and fallback on big-endian hosts). */
export function scanConvertLutReference(
  polar: Uint8ClampedArray,
  lut: ScanLut,
  rgba: Uint8ClampedArray,
  tintMap?: TintMapId,
): void {
  const n = lut.idx.length;
  const S = lut.samples;
  const L = lut.lines;
  const maxS = S - 1;
  const maxL = L - 1;
  const rgb: [number, number, number] = [0, 0, 0];
  for (let p = 0, o = 0; p < n; p++, o += 4) {
    const i0 = lut.idx[p] ?? -1;
    if (i0 < 0) {
      rgba[o] = 0;
      rgba[o + 1] = 0;
      rgba[o + 2] = 0;
      rgba[o + 3] = 255;
      continue;
    }
    const l0 = (i0 / S) | 0;
    const s0 = i0 - l0 * S;
    const fl = lut.wl[p] ?? 0;
    const fs = lut.ws[p] ?? 0;
    const sm1 = s0 > 0 ? s0 - 1 : 0;
    const s1 = s0 < maxS ? s0 + 1 : maxS;
    const s2 = s0 + 2 <= maxS ? s0 + 2 : maxS;
    const lm1 = l0 > 0 ? l0 - 1 : 0;
    const l1 = l0 < maxL ? l0 + 1 : maxL;
    const l2 = l0 + 2 <= maxL ? l0 + 2 : maxL;
    const r0 = lm1 * S,
      r1 = l0 * S,
      r2 = l1 * S,
      r3 = l2 * S;
    const fs2 = fs * fs,
      fs3 = fs2 * fs;
    const ws0 = -0.5 * fs + fs2 - 0.5 * fs3;
    const ws1 = 1 - 2.5 * fs2 + 1.5 * fs3;
    const ws2 = 0.5 * fs + 2 * fs2 - 1.5 * fs3;
    const ws3 = -0.5 * fs2 + 0.5 * fs3;
    const c0 =
      (polar[r0 + sm1] ?? 0) * ws0 +
      (polar[r0 + s0] ?? 0) * ws1 +
      (polar[r0 + s1] ?? 0) * ws2 +
      (polar[r0 + s2] ?? 0) * ws3;
    const v00 = polar[r1 + s0] ?? 0;
    const v01 = polar[r1 + s1] ?? 0;
    const c1 =
      (polar[r1 + sm1] ?? 0) * ws0 + v00 * ws1 + v01 * ws2 + (polar[r1 + s2] ?? 0) * ws3;
    const v10 = polar[r2 + s0] ?? 0;
    const v11 = polar[r2 + s1] ?? 0;
    const c2 =
      (polar[r2 + sm1] ?? 0) * ws0 + v10 * ws1 + v11 * ws2 + (polar[r2 + s2] ?? 0) * ws3;
    const c3 =
      (polar[r3 + sm1] ?? 0) * ws0 +
      (polar[r3 + s0] ?? 0) * ws1 +
      (polar[r3 + s1] ?? 0) * ws2 +
      (polar[r3 + s2] ?? 0) * ws3;
    const fl2 = fl * fl,
      fl3 = fl2 * fl;
    let val =
      c0 * (-0.5 * fl + fl2 - 0.5 * fl3) +
      c1 * (1 - 2.5 * fl2 + 1.5 * fl3) +
      c2 * (0.5 * fl + 2 * fl2 - 1.5 * fl3) +
      c3 * (-0.5 * fl2 + 0.5 * fl3);
    const lo = Math.min(v00, v10, v01, v11);
    const hi = Math.max(v00, v10, v01, v11);
    if (val < lo) val = lo;
    else if (val > hi) val = hi;
    const q = Math.floor(Math.max(0, Math.min(255, val)) * TINT_SUB_STEPS) / TINT_SUB_STEPS;
    tintGrayFloat(q, tintMap, rgb);
    rgba[o] = rgb[0];
    rgba[o + 1] = rgb[1];
    rgba[o + 2] = rgb[2];
    rgba[o + 3] = 255;
  }
}

/** Reference (LUT-free) implementation, kept for tests and offline tools. */
export function scanConvert(
  polar: Uint8ClampedArray,
  spec: PolarFrameSpec,
  m: SectorMapping,
  rgba: Uint8ClampedArray,
  tintMap?: TintMapId,
): void {
  const lut = buildScanLut(spec, m);
  scanConvertLut(polar, lut, rgba, tintMap);
}

/** Bit 11 of the third texel channel: the pixel lies inside the sector. */
export const LUT_TEXEL_INSIDE = 2048;

/**
 * The LUT as RGBA16UI texels (row-major, top row first) for the GPU present pass (decision 54): line and
 * sample of the (l0, s0) corner, line weight | next-line flag << 10 | inside << 11, sample weight |
 * next-sample flag << 10. Outside the sector every channel is 0. The GPU gather with these integers is the
 * same arithmetic as `scanConvertLut`, so both produce identical grey levels.
 */
export function packScanLutTexels(
  lut: ScanLut,
  out?: Uint16Array<ArrayBuffer>,
): Uint16Array<ArrayBuffer> {
  const n = lut.idx.length;
  const t = out && out.length === n * 4 ? out.fill(0) : new Uint16Array(n * 4);
  const S = lut.samples;
  const { inPix, inIdx, inWl, inWs, inDl, inDs } = lut;
  for (let k = 0; k < inPix.length; k++) {
    const o = inPix[k]! * 4;
    const i = inIdx[k]!;
    t[o] = (i / S) | 0;
    t[o + 1] = i % S;
    t[o + 2] = inWl[k]! | (inDl[k]! << 10) | LUT_TEXEL_INSIDE;
    t[o + 3] = inWs[k]! | (inDs[k]! << 10);
  }
  return t;
}
