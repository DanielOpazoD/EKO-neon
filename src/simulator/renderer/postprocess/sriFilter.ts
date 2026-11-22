/**
 * Structure-oriented anisotropic Speckle Reduction Imaging (SRI / XRES-style) in polar coordinates.
 * Smooths speckle along the local structure tangent (endocardium, myocardium, valves) with a
 * bilateral range gate so boundaries are never smeared, while subtly sharpening across the normal
 * direction. Exact identity at strength = 0 and DC-preserving on uniform regions.
 * Constants are exported to GLSL via `sriDefinesGlsl()` so CPU and WebGL2 match closely.
 */
export const SRI_DEFAULT = 0.35;
export const SRI_RANGE_SIGMA = 0.15;
export const SRI_EDGE_REF = 0.65;
export const SRI_SMOOTH_BASE = 0.25;
export const SRI_SMOOTH_COH = 0.42;
export const SRI_SHARP_GAIN = 0.24;
export const SRI_CENTER_WEIGHT = 0.55;

const INV_SQRT2 = Math.SQRT1_2;
const INV_2_SIGMA2 = 1 / (2 * SRI_RANGE_SIGMA * SRI_RANGE_SIGMA);

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

      // Sobel 3x3 gradients: gs along sample (axial), gl along line (lateral)
      const gs = ne + 2 * ee + se - (nw + 2 * ww + sw);
      const gl = sw + 2 * ss + se - (nw + 2 * nn + ne);
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

      const tangW = we + w_w + ws + wn + wse + wnw + wne + wsw + SRI_CENTER_WEIGHT;
      const tangMean =
        (we * ee +
          w_w * ww +
          ws * ss +
          wn * nn +
          wse * se +
          wnw * nw +
          wne * ne +
          wsw * sw +
          SRI_CENTER_WEIGHT * cc) /
        tangW;

      const normW = 2 * (wn0 + wn1 + wn2 + wn3);
      const normMean =
        normW > 1e-6
          ? (wn0 * (ee + ww) + wn1 * (ss + nn) + wn2 * (se + nw) + wn3 * (ne + sw)) / normW
          : cc;

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
    `#define SRI_EDGE_REF ${glslFloat(SRI_EDGE_REF)}`,
    `#define SRI_SMOOTH_BASE ${glslFloat(SRI_SMOOTH_BASE)}`,
    `#define SRI_SMOOTH_COH ${glslFloat(SRI_SMOOTH_COH)}`,
    `#define SRI_SHARP_GAIN ${glslFloat(SRI_SHARP_GAIN)}`,
    `#define SRI_CENTER_WEIGHT ${glslFloat(SRI_CENTER_WEIGHT)}`,
  ].join('\n');
}
