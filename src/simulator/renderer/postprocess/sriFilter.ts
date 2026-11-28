/**
 * Two-scale structure-oriented anisotropic Speckle Reduction Imaging (SRI-HD / XRES Pro style) in polar coordinates.
 * Combines a fine 3×3 neighborhood and a dilated 5×5 (stride-2, speckle-cell scale) neighborhood to estimate robust
 * anatomical tangent/normal directions without chasing individual Rayleigh grains, smoothing along the wall/leaflet
 * tangent with bilateral range protection while applying multi-scale local contrast enhancement across the normal.
 * Exact identity at strength = 0 and DC-preserving on uniform regions.
 * Constants are exported to GLSL via `sriDefinesGlsl()` so CPU and WebGL2 match closely.
 */
export const SRI_DEFAULT = 0.35;
export const SRI_RANGE_SIGMA = 0.15;
export const SRI_COARSE_SIGMA = 0.11;
export const SRI_COARSE_SHARE = 0.42;
export const SRI_EDGE_REF = 0.65;
export const SRI_SMOOTH_BASE = 0.25;
export const SRI_SMOOTH_COH = 0.44;
export const SRI_SHARP_GAIN = 0.26;
export const SRI_CENTER_WEIGHT = 0.52;

const INV_SQRT2 = Math.SQRT1_2;
const INV_2_SIGMA2 = 1 / (2 * SRI_RANGE_SIGMA * SRI_RANGE_SIGMA);
const INV_2_COARSE_SIGMA2 = 1 / (2 * SRI_COARSE_SIGMA * SRI_COARSE_SIGMA);

export function applySriPolar(
  src: Float32Array,
  dst: Float32Array,
  lines: number,
  samples: number,
  strength: number,
): void {
  const n = lines * samples;
  if (!(strength > 0) || lines < 3 || samples < 3) {
    dst.set(src.subarray(0, n));
    return;
  }
  const s = Math.min(1, strength);
  const S = samples;
  // Copy outer 1-pixel border unchanged
  for (let si = 0; si < S; si++) {
    dst[si] = src[si]!;
    dst[(lines - 1) * S + si] = src[(lines - 1) * S + si]!;
  }
  for (let li = 1; li < lines - 1; li++) {
    const row = li * S;
    const prev = row - S;
    const next = row + S;
    const hasCoarseL = li >= 2 && li < lines - 2;
    const prev2 = row - 2 * S;
    const next2 = row + 2 * S;
    dst[row] = src[row]!;
    dst[row + S - 1] = src[row + S - 1]!;
    for (let si = 1; si < S - 1; si++) {
      const nw = src[prev + si - 1]!;
      const nn = src[prev + si]!;
      const ne = src[prev + si + 1]!;
      const ww = src[row + si - 1]!;
      const cc = src[row + si]!;
      const ee = src[row + si + 1]!;
      const sw = src[next + si - 1]!;
      const ss = src[next + si]!;
      const se = src[next + si + 1]!;

      // Fine 3×3 Sobel gradients: gs along sample (axial), gl along line (lateral)
      let gs = ne + 2 * ee + se - (nw + 2 * ww + sw);
      let gl = sw + 2 * ss + se - (nw + 2 * nn + ne);

      const hasCoarse = hasCoarseL && si >= 2 && si < S - 2;
      let nw2 = 0,
        nn2 = 0,
        ne2 = 0,
        ww2 = 0,
        ee2 = 0,
        sw2 = 0,
        ss2 = 0,
        se2 = 0;
      if (hasCoarse) {
        nw2 = src[prev2 + si - 2]!;
        nn2 = src[prev2 + si]!;
        ne2 = src[prev2 + si + 2]!;
        ww2 = src[row + si - 2]!;
        ee2 = src[row + si + 2]!;
        sw2 = src[next2 + si - 2]!;
        ss2 = src[next2 + si]!;
        se2 = src[next2 + si + 2]!;
        // Multi-scale gradient locks onto true wall/leaflet geometry across the speckle cell
        const gs2 = ne2 + 2 * ee2 + se2 - (nw2 + 2 * ww2 + sw2);
        const gl2 = sw2 + 2 * ss2 + se2 - (nw2 + 2 * nn2 + ne2);
        gs = 0.65 * gs + 0.35 * gs2;
        gl = 0.65 * gl + 0.35 * gl2;
      }

      const mag = Math.hypot(gs, gl);
      const edgeConf = mag < SRI_EDGE_REF ? mag / SRI_EDGE_REF : 1;

      let ns = 1;
      let nl = 0;
      if (mag > 1e-5) {
        ns = gs / mag;
        nl = gl / mag;
      }
      // Tangent perpendicular to gradient normal
      const ts = -nl;
      const tl = ns;

      // Fourth-power directional alignment weights for the 4 symmetric neighbor pairs
      const wt0 = ts * ts * ts * ts;
      const wt1 = tl * tl * tl * tl;
      const dT2 = (ts + tl) * INV_SQRT2;
      const dT3 = (ts - tl) * INV_SQRT2;
      const wt2 = dT2 * dT2 * dT2 * dT2;
      const wt3 = dT3 * dT3 * dT3 * dT3;

      const wn0 = ns * ns * ns * ns;
      const wn1 = nl * nl * nl * nl;
      const dN2 = (ns + nl) * INV_SQRT2;
      const dN3 = (ns - nl) * INV_SQRT2;
      const wn2 = dN2 * dN2 * dN2 * dN2;
      const wn3 = dN3 * dN3 * dN3 * dN3;

      const dirScale = 0.2 + 0.8 * edgeConf;
      const isoFloor = 0.18 * (1 - edgeConf);

      const g0 = wt0 * dirScale + isoFloor;
      const g1 = wt1 * dirScale + isoFloor;
      const g2 = wt2 * dirScale + isoFloor;
      const g3 = wt3 * dirScale + isoFloor;

      let de = ee - cc;
      const we = g0 * Math.exp(-de * de * INV_2_SIGMA2);
      de = ww - cc;
      const w_w = g0 * Math.exp(-de * de * INV_2_SIGMA2);
      de = ss - cc;
      const ws = g1 * Math.exp(-de * de * INV_2_SIGMA2);
      de = nn - cc;
      const wn = g1 * Math.exp(-de * de * INV_2_SIGMA2);
      de = se - cc;
      const wse = g2 * Math.exp(-de * de * INV_2_SIGMA2);
      de = nw - cc;
      const wnw = g2 * Math.exp(-de * de * INV_2_SIGMA2);
      de = ne - cc;
      const wne = g3 * Math.exp(-de * de * INV_2_SIGMA2);
      de = sw - cc;
      const wsw = g3 * Math.exp(-de * de * INV_2_SIGMA2);

      const tangW1 = we + w_w + ws + wn + wse + wnw + wne + wsw + SRI_CENTER_WEIGHT;
      let tangMean =
        (we * ee +
          w_w * ww +
          ws * ss +
          wn * nn +
          wse * se +
          wnw * nw +
          wne * ne +
          wsw * sw +
          SRI_CENTER_WEIGHT * cc) /
        tangW1;

      const normW = 2 * (wn0 + wn1 + wn2 + wn3);
      let normMean =
        normW > 1e-6
          ? (wn0 * (ee + ww) + wn1 * (ss + nn) + wn2 * (se + nw) + wn3 * (ne + sw)) / normW
          : cc;

      if (hasCoarse) {
        de = ee2 - cc;
        const cwe = g0 * Math.exp(-de * de * INV_2_COARSE_SIGMA2);
        de = ww2 - cc;
        const cw_w = g0 * Math.exp(-de * de * INV_2_COARSE_SIGMA2);
        de = ss2 - cc;
        const cws = g1 * Math.exp(-de * de * INV_2_COARSE_SIGMA2);
        de = nn2 - cc;
        const cwn = g1 * Math.exp(-de * de * INV_2_COARSE_SIGMA2);
        de = se2 - cc;
        const cwse = g2 * Math.exp(-de * de * INV_2_COARSE_SIGMA2);
        de = nw2 - cc;
        const cwnw = g2 * Math.exp(-de * de * INV_2_COARSE_SIGMA2);
        de = ne2 - cc;
        const cwne = g3 * Math.exp(-de * de * INV_2_COARSE_SIGMA2);
        de = sw2 - cc;
        const cwsw = g3 * Math.exp(-de * de * INV_2_COARSE_SIGMA2);

        const tangW2 = cwe + cw_w + cws + cwn + cwse + cwnw + cwne + cwsw + SRI_CENTER_WEIGHT;
        const tangMean2 =
          (cwe * ee2 +
            cw_w * ww2 +
            cws * ss2 +
            cwn * nn2 +
            cwse * se2 +
            cwnw * nw2 +
            cwne * ne2 +
            cwsw * sw2 +
            SRI_CENTER_WEIGHT * cc) /
          tangW2;
        tangMean = (1 - SRI_COARSE_SHARE) * tangMean + SRI_COARSE_SHARE * tangMean2;

        // Bilateral-clamped coarse normal contrast so step edges sharpen cleanly without wide-halo overshoot
        const clampRange = (v: number): number => {
          const d = v - cc;
          return Math.abs(d) <= 0.28 ? v : cc + (d > 0 ? 0.28 : -0.28);
        };
        const normMean2 =
          normW > 1e-6
            ? (wn0 * (clampRange(ee2) + clampRange(ww2)) +
                wn1 * (clampRange(ss2) + clampRange(nn2)) +
                wn2 * (clampRange(se2) + clampRange(nw2)) +
                wn3 * (clampRange(ne2) + clampRange(sw2))) /
              normW
            : cc;
        normMean = 0.75 * normMean + 0.25 * normMean2;
      }

      const smoothMix = s * (SRI_SMOOTH_BASE + SRI_SMOOTH_COH * edgeConf);
      const smoothed = cc + (tangMean - cc) * smoothMix;
      const sharp = (cc - normMean) * (s * SRI_SHARP_GAIN * edgeConf);
      const out = smoothed + sharp;
      dst[row + si] = out < 0 ? 0 : out > 1 ? 1 : out;
    }
  }
}

const glslFloat = (v: number): string => (Number.isInteger(v) ? `${v}.0` : `${v}`);

export function sriDefinesGlsl(): string {
  return [
    `#define SRI_INV_2_SIGMA2 ${glslFloat(INV_2_SIGMA2)}`,
    `#define SRI_INV_2_COARSE_SIGMA2 ${glslFloat(INV_2_COARSE_SIGMA2)}`,
    `#define SRI_COARSE_SHARE ${glslFloat(SRI_COARSE_SHARE)}`,
    `#define SRI_EDGE_REF ${glslFloat(SRI_EDGE_REF)}`,
    `#define SRI_SMOOTH_BASE ${glslFloat(SRI_SMOOTH_BASE)}`,
    `#define SRI_SMOOTH_COH ${glslFloat(SRI_SMOOTH_COH)}`,
    `#define SRI_SHARP_GAIN ${glslFloat(SRI_SHARP_GAIN)}`,
    `#define SRI_CENTER_WEIGHT ${glslFloat(SRI_CENTER_WEIGHT)}`,
  ].join('\n');
}
