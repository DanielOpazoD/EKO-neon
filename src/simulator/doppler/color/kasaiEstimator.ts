import { hash3 } from '@/core/random';
import { APERTURE_MM } from '@/simulator/probe/transducer';

/**
 * The colour field as a scanner estimates it (decision 116). Each update the velocity comes from the autocorrelation of a short
 * packet of echoes and the power from the same echoes of moving blood, so both scatter: the power of the blood's speckle
 * (exponential, averaged over the packet and the sample volume) decides whether a sample passes the display threshold,
 * and the velocity estimate scatters around the velocity, more with a broad spectrum and with a weak echo. Both are
 * correlated over about two samples and two lines (the resolution cell) and new at every update, as the blood moves on.
 * Without an estimate the field is the expected one. Declared values: no measured colour Doppler texture was found to
 * calibrate against.
 */
export interface ColorEstimate {
  /** Index of the colour update: every update draws a new realization. */
  realization: number;
  seed: number;
}

export interface KasaiResult {
  velocity: Float32Array;
  power: Float32Array;
  variance: Float32Array;
  coherence: Float32Array;
}

/** Pulses per colour line: the packet of the frame-rate model (decision 178). */
export const COLOR_PACKET = 8;

/**
 * Receiver noise power, relative to the echo of blood at the transmission soft tissue would leave, and the pulse-to-pulse
 * phase jitter of a laminar flow (rad): the transit of the scatterers through the sample volume and the beam's
 * geometric broadening. Declared values, set so a strong laminar flow scatters by a few per cent of Nyquist; no measured
 * colour texture was found to calibrate against. The jitter is white from pulse to pulse: it keeps e^(−0.16) of the echo
 * coherent and spreads the other 15 % flat over the PRF, where no wall filter removes it, so filtered slow blood keeps a
 * floor near −9 dB and reads fast just above the cutoff (16×16 cells, 6 updates: +4 to +8 % at twice the cutoff, 0 to
 * +4 % without jitter). A drift correlated over the packet with the same lag-one coherence does worse (a random walk:
 * floor −4 to −5 dB, +5 to +17 % at twice the cutoff): 0.4 rad per pulse is far more decorrelation than transit through
 * a real beam gives slow laminar flow. The physical model, a transit decorrelation that grows with speed, would move the
 * variance map and the noise gate with it, so the white jitter stays.
 */
export const NOISE_POWER = 0.02;
export const TRANSIT_JITTER_RAD = 0.4;
/** Below this |R1|/R0 a weak echo is rejected as noise, unless it is NOISE_POWER_GATE times over the display threshold. */
export const NOISE_COHERENCE = 0.5;
export const NOISE_POWER_GATE = 2.5;
/** 1 − |R1|/R0 of a strong laminar flow: the transit jitter and the receiver noise alone. */
export const LAMINAR_DECORRELATION =
  1 - Math.exp(-(TRANSIT_JITTER_RAD ** 2)) / (1 + NOISE_POWER);

/** Active transducer aperture in cm (shared with spectral Doppler geometric broadening). */
const APERTURE_CM = APERTURE_MM / 10;
const INV_SQRT12 = 1 / Math.sqrt(12);
/** Turbulent spectral-width factor (same 0.9 * disp * |v| calibration as `spectrum.ts`). */
export const COLOR_TURBULENT_WIDTH_K = 0.9;

/**
 * Absolute Doppler spectral velocity width σ_v (m/s) combining turbulent dispersion and
 * transverse aperture geometric broadening (Newhouse et al. 1980: v_perp · D / (r · √12)),
 * matching `spectrum.ts` so near-90° transverse flow retains geometric broadening even as v_axial → 0.
 */
export function colorSpectralWidthMps(
  vAxialMps: number,
  disp: number,
  vPerpMps: number,
  depthCm: number,
): number {
  const turb = COLOR_TURBULENT_WIDTH_K * disp * Math.abs(vAxialMps);
  const geom =
    vPerpMps > 0 && depthCm > 0.25
      ? (Math.abs(vPerpMps) * APERTURE_CM * INV_SQRT12) / Math.max(1.0, depthCm)
      : 0;
  return geom > 0 ? Math.hypot(turb, geom) : turb;
}

/** Effective post-RF-compression clutter-to-blood power ratio (+8 dB ≈ 6.31×). */
export const CLUTTER_BLOOD_RATIO_POW = 6.31;
/** Narrowband coherent spectral width (m/s) of solid myocardial/valve tissue clutter. */
export const CLUTTER_SIGMA_V_MPS = 0.02;
/** Low pulse-to-pulse phase jitter (rad) of rigid myocardial/valve scatterers over a ~1.5 ms packet. */
export const CLUTTER_JITTER_RAD = 0.04;

/** Precomputed orthonormal linear Legendre basis (P1) over the fixed packet length COLOR_PACKET. */
export const KASAI_LIN_BASIS = (() => {
  const lin = new Float64Array(COLOR_PACKET);
  let normSq = 0;
  for (let p = 0; p < COLOR_PACKET; p++) {
    lin[p] = p - (COLOR_PACKET - 1) / 2;
    normSq += lin[p]! * lin[p]!;
  }
  const invNorm = 1 / Math.sqrt(normSq);
  for (let p = 0; p < COLOR_PACKET; p++) lin[p]! *= invNorm;
  return lin;
})();
const LIN_BASIS = KASAI_LIN_BASIS;

/** Precomputed orthonormal quadratic Legendre basis (P2) over COLOR_PACKET, orthogonal to P0 and P1. */
export const KASAI_QUAD_BASIS = (() => {
  const quad = new Float64Array(COLOR_PACKET);
  const mid = (COLOR_PACKET - 1) / 2;
  let meanU2 = 0;
  for (let p = 0; p < COLOR_PACKET; p++) {
    const u = p - mid;
    meanU2 += u * u;
  }
  meanU2 /= COLOR_PACKET;
  let normSq = 0;
  for (let p = 0; p < COLOR_PACKET; p++) {
    const u = p - mid;
    const q = u * u - meanU2;
    quad[p] = q;
    normSq += q * q;
  }
  const invNorm = 1 / Math.sqrt(normSq);
  for (let p = 0; p < COLOR_PACKET; p++) quad[p]! *= invNorm;
  return quad;
})();
const QUAD_BASIS = KASAI_QUAD_BASIS;

/**
 * Share of a unit tone's packet power that lies on the Legendre components P0 (mean), P1 and P2, for a tone turning by
 * `w` rad per pulse (π·v/Nyquist). A regression filter that removes the share s_k of each component leaves the tone
 * 1 − Σ (2·s_k − s_k²)·a_k of its power: the filter's response, computed instead of guessed.
 */
function legendreTonePower(w: number, out: Float64Array): Float64Array {
  let r0 = 0;
  let i0 = 0;
  let r1 = 0;
  let i1 = 0;
  let r2 = 0;
  let i2 = 0;
  for (let p = 0; p < COLOR_PACKET; p++) {
    const c = Math.cos(w * p);
    const s = Math.sin(w * p);
    r0 += c;
    i0 += s;
    r1 += c * LIN_BASIS[p]!;
    i1 += s * LIN_BASIS[p]!;
    r2 += c * QUAD_BASIS[p]!;
    i2 += s * QUAD_BASIS[p]!;
  }
  out[0] = (r0 * r0 + i0 * i0) / (COLOR_PACKET * COLOR_PACKET);
  out[1] = (r1 * r1 + i1 * i1) / COLOR_PACKET;
  out[2] = (r2 * r2 + i2 * i2) / COLOR_PACKET;
  return out;
}

const TONE_POWER = new Float64Array(3);

/** Shares of the packet's mean (P0), linear trend (P1) and quadratic (P2) a regression wall filter removes. */
export interface WallFilterShares {
  mean: number;
  trend: number;
  quad: number;
}

/** What a tone with Legendre powers `a` keeps through a filter removing these shares: 1 − Σ (2·s_k − s_k²)·a_k. */
function keptPower(a: Float64Array, s: WallFilterShares): number {
  return Math.max(
    0,
    1 -
      (2 * s.mean - s.mean * s.mean) * a[0]! -
      (2 * s.trend - s.trend * s.trend) * a[1]! -
      (2 * s.quad - s.quad * s.quad) * a[2]!,
  );
}

/**
 * Expected share of its power a laminar flow at `vMps` keeps through the regression wall filter with these shares: the
 * response of the 8-pulse packet to a tone turning by π·v/Nyquist per pulse. It acts on the sampled phase, so it is
 * periodic in the PRF (a flow aliasing to near zero is rejected as a slow one) and even in v. The estimator's power
 * follows it above the cutoff; below, the transit jitter keeps a floor near −9 dB (`TRANSIT_JITTER_RAD`). The colour
 * M-mode paints with it, tabulated (`WallFilterResponseTable`), so the strip and the 2D colour cut at the same −3 dB
 * point.
 */
export function wallFilterPowerResponse(
  vMps: number,
  scaleMps: number,
  shares: WallFilterShares,
): number {
  return keptPower(legendreTonePower((Math.PI * vMps) / Math.max(1e-3, scaleMps), TONE_POWER), shares);
}

/**
 * −3 dB cutoff, as a fraction of Nyquist, of the regression wall filter with these shares: the first speed at which a
 * tone keeps half its power (scanned by 0.01 Nyquist, then bisected). 0 when a tone at rest keeps half already, a
 * filter that by this measure cuts nothing (a partial mean removal under 1 − √½ = 29 %).
 */
export function wallFilterCutoffNyquist(shares: WallFilterShares): number {
  const kept = (c: number) => keptPower(legendreTonePower(Math.PI * c, TONE_POWER), shares);
  if (kept(0) >= 0.5) return 0;
  let lo = 0;
  let hi = 0.01;
  while (hi < 1 && kept(hi) < 0.5) {
    lo = hi;
    hi += 0.01;
  }
  for (let it = 0; it < 60; it++) {
    const mid = 0.5 * (lo + hi);
    if (kept(mid) < 0.5) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/**
 * Range of the colour wall filter's −3 dB cutoff, as fractions of Nyquist, that an 8-pulse packet allows. The lower bound
 * is physics: a filter that cancels stationary echo and amplifies nothing passes at most what the mean removal passes
 * (|F·x|² = |F·(x − x̄)|² ≤ |x − x̄|²), so no such filter cuts below the mean removal's −3 dB, 0.1115 Nyquist. The upper
 * bound is the steepest filter here, mean + trend + quadratic removed, 0.321 Nyquist; a higher cutoff would need a longer
 * packet or a higher-order regression. What filters is bounded to this range (`colorWallFilterCutoffMps`).
 */
export const WF_MIN_NYQUIST = wallFilterCutoffNyquist({ mean: 1, trend: 0, quad: 0 });
export const WF_MAX_NYQUIST = wallFilterCutoffNyquist({ mean: 1, trend: 1, quad: 1 });

/** The −3 dB cutoffs (m/s) the colour wall filter can have at a Nyquist scale: 7.6 to 21.8 cm/s at 0.68 m/s. */
export function colorWallFilterRangeMps(scaleMps: number): { min: number; max: number } {
  const scale = Math.max(1e-3, scaleMps);
  return { min: WF_MIN_NYQUIST * scale, max: WF_MAX_NYQUIST * scale };
}

/**
 * The −3 dB cutoff (m/s) the colour wall filter has for a requested one at the Nyquist it acquires with: bounded to what
 * the packet allows, 0 = off. The request is what the store and the review reports keep (`ColorSettings.wallFilterMps`);
 * each consumer bounds it here: the estimator's shares (CPU and GPU, `wallFilterShares`), the colour M-mode, the
 * expected field and the console's readout. Bounded where it was kept, a lowered scale lowered it for good: back at
 * 0.68 m/s after 0.35 the 20 cm/s default stayed 11.2 cm/s, and the moving myocardium of an A4C (CPU, 44 frames) went
 * from 112 coloured samples to 7073.
 */
export function colorWallFilterCutoffMps(wallFilterMps: number, scaleMps: number): number {
  if (!(wallFilterMps > 0)) return 0;
  const scale = Math.max(1e-3, scaleMps);
  return Math.min(WF_MAX_NYQUIST * scale, Math.max(WF_MIN_NYQUIST * scale, wallFilterMps));
}

/** Share s of a component holding power `ak` at the cutoff whose removal takes `taken` of it: (2s − s²)·ak = taken. */
function shareTaking(taken: number, ak: number): number {
  return 1 - Math.sqrt(1 - Math.min(1, Math.max(0, taken / Math.max(1e-12, ak))));
}

/**
 * Shares of the packet's mean, linear trend and quadratic the regression wall filter removes so that its −3 dB point
 * falls on the requested cutoff, bounded to what the packet allows (`colorWallFilterCutoffMps`): the mean always whole,
 * then the trend, then the quadratic, each solved in closed form from the tone powers at the cutoff. One function for the
 * CPU estimator, the GPU pass (a uniform) and the colour M-mode, so all three filter alike. Through the estimator
 * (jitter and noise included; uniform laminar flow, 16×16 cells, 6 updates, scales 0.3–1 m/s) the power is −2.4 to
 * −2.9 dB at the cutoff and the velocity reads +4 to +8 % fast at twice it: the edge of any filter reads fast, the more
 * so over the jitter's floor (`TRANSIT_JITTER_RAD`). The shares used to be ramps in WF/Nyquist never checked against
 * the response: the default 6 cm/s at 0.68 m/s removed mean + trend + 55 % of the quadratic and cut at 0.205 m/s, 3.4
 * times the label, read 0.15 m/s as 0.32, and every setting from 8 cm/s up gave the same filter.
 */
export function wallFilterShares(
  wallFilterMps: number,
  scaleMps: number,
  out: WallFilterShares = { mean: 0, trend: 0, quad: 0 },
): WallFilterShares {
  out.mean = 0;
  out.trend = 0;
  out.quad = 0;
  const cutMps = colorWallFilterCutoffMps(wallFilterMps, scaleMps);
  if (cutMps <= 0) return out;
  const a = legendreTonePower((Math.PI * cutMps) / Math.max(1e-3, scaleMps), TONE_POWER);
  // a tone at the cutoff must lose half its power: the mean takes a0, the trend and the quadratic what is left
  out.mean = 1;
  if (a[0]! + a[1]! >= 0.5) {
    out.trend = shareTaking(0.5 - a[0]!, a[1]!);
  } else {
    out.trend = 1;
    out.quad = shareTaking(0.5 - a[0]! - a[1]!, a[2]!);
  }
  return out;
}

/** Intervals of |v|/Nyquist over [0, 1] that `WallFilterResponseTable` samples the response on. */
const WF_TABLE_STEPS = 1024;

/**
 * `wallFilterPowerResponse` for one request at one Nyquist, tabulated for a loop that filters many velocities alike:
 * the colour M-mode reads it for every blood sample of its columns. Even and periodic in 2·Nyquist, the response is a
 * function of |v|/Nyquist folded into [0, 1], sampled here at 1025 points and read by linear interpolation. That errs
 * by at most Δ²/8·max|h''| with Δ = 1/1024: the response is a trigonometric polynomial of degree 7 in π·v/Nyquist
 * with values in [0, 1], so |h''| ≤ 7²·π²/2 (Bernstein), at most 2.9e-5 of the power; over every cutoff the packet
 * allows |h''| peaks at (8² − 1)·π²/6 ≈ 104, next to zero with the mean removal, and the error at 1.24e-5. The table
 * is refilled only when the request or the Nyquist changes (1025 responses, 0.1 ms). Per sample, the response's 16
 * sin/cos took ~0.1 µs: with A4C through the crux at high quality, 1600 px and 100 mm/s (27 columns of 118 blood
 * samples a frame) the strip's colour loop took 0.30 ms a frame, 11 µs from the table (6 µs with the old hard cut).
 */
export class WallFilterResponseTable {
  private readonly power = new Float64Array(WF_TABLE_STEPS + 1);
  private readonly shares: WallFilterShares = { mean: 0, trend: 0, quad: 0 };
  private wallFilterMps = NaN;
  private scaleMps = NaN;
  /** Table steps per m/s of |v|. */
  private stepsPerMps = 0;

  /** The response for this request at this acquired Nyquist; refilled only when either changes. */
  set(wallFilterMps: number, scaleMps: number): this {
    if (wallFilterMps === this.wallFilterMps && scaleMps === this.scaleMps) return this;
    this.wallFilterMps = wallFilterMps;
    this.scaleMps = scaleMps;
    const scale = Math.max(1e-3, scaleMps);
    wallFilterShares(wallFilterMps, scale, this.shares);
    for (let i = 0; i <= WF_TABLE_STEPS; i++)
      this.power[i] = wallFilterPowerResponse((i / WF_TABLE_STEPS) * scale, scale, this.shares);
    this.stepsPerMps = WF_TABLE_STEPS / scale;
    return this;
  }

  /** Share of its power a laminar flow at `vMps` keeps (`wallFilterPowerResponse`), from the table. */
  at(vMps: number): number {
    // |v| in table steps, folded into [0, 1] Nyquist: the response is even and periodic in 2·Nyquist
    let x = Math.abs(vMps) * this.stepsPerMps;
    x -= 2 * WF_TABLE_STEPS * Math.floor(x / (2 * WF_TABLE_STEPS));
    if (x > WF_TABLE_STEPS) x = 2 * WF_TABLE_STEPS - x;
    const i = Math.min(WF_TABLE_STEPS - 1, Math.floor(x));
    const lo = this.power[i]!;
    return lo + (this.power[i + 1]! - lo) * (x - i);
  }
}

const WF_SHARES: WallFilterShares = { mean: 0, trend: 0, quad: 0 };

/** Module-level scratch buffers reused across every packet evaluation (zero per-update heap churn). */
const XR_BUF = new Float64Array(COLOR_PACKET);
const XI_BUF = new Float64Array(COLOR_PACKET);
const PAIR_TMP = { a: 0, b: 0 };
const JIT_SLOTS: Float32Array[] = new Array<Float32Array>(COLOR_PACKET);

/**
 * Scratch buffers reused across colour updates (decision 304): an update allocated ~18 typed arrays the size of the colour
 * box. Each name keeps one buffer, grown when a box needs more; the views returned are only valid until the next update.
 */
const scratchPool = new Map<string, Float32Array>();

export function colorScratch(name: string, n: number): Float32Array {
  let b = scratchPool.get(name);
  if (!b || b.length < n) {
    b = new Float32Array(Math.max(n, Math.ceil((b?.length ?? 0) * 1.5)));
    scratchPool.set(name, b);
  }
  return b.subarray(0, n);
}

/** Unit Gaussian from two hashes (Box–Muller). */
function hashGaussian(li: number, si: number, stream: number, seed: number): number {
  const u1 = hash3(li, si, stream, seed);
  const u2 = hash3(li, si, stream + 1, seed);
  return Math.sqrt(-2 * Math.log(Math.max(1e-12, u1))) * Math.cos(2 * Math.PI * u2);
}

/** A pair of independent unit Gaussians from the same two hashes (both outputs of Box–Muller), into `out`. */
function hashGaussianPair(
  li: number,
  si: number,
  stream: number,
  seed: number,
  out: { a: number; b: number },
): void {
  const u1 = hash3(li, si, stream, seed);
  const u2 = hash3(li, si, stream + 1, seed);
  const r = Math.sqrt(-2 * Math.log(Math.max(1e-12, u1)));
  out.a = r * Math.cos(2 * Math.PI * u2);
  out.b = r * Math.sin(2 * Math.PI * u2);
}

/** Binomial [1 2 1]/4 smoothing of a lines × samples grid, along lines and along samples, in place. */
function smoothGrid(g: Float32Array, nL: number, nS: number, tmp: Float32Array): void {
  for (let l = 0; l < nL; l++)
    for (let k = 0; k < nS; k++) {
      const i = l * nS + k;
      tmp[i] = 0.25 * (g[k > 0 ? i - 1 : i]! + 2 * g[i]! + g[k < nS - 1 ? i + 1 : i]!);
    }
  for (let l = 0; l < nL; l++)
    for (let k = 0; k < nS; k++) {
      const i = l * nS + k;
      g[i] = 0.25 * (tmp[l > 0 ? i - nS : i]! + 2 * tmp[i]! + tmp[l < nL - 1 ? i + nS : i]!);
    }
}

/**
 * The colour estimate of one update (decision 300), as a scanner forms it. Each sample of the box receives a packet of
 * `COLOR_PACKET` echoes: the blood's speckle amplitude (a complex Gaussian correlated over the resolution cell, new at every
 * update), turning by π·v/Nyquist per pulse, with a pulse-to-pulse jitter from transit and turbulence (the spectral width
 * 0.9 × dispersion × speed of spectrum.ts), plus receiver noise. A regression wall filter removes the packet's mean, and
 * its linear trend and quadratic as the cutoff rises, −3 dB at the cutoff (`wallFilterShares`): it acts on the sampled
 * phase, so it is periodic in the PRF and rejects a jet that aliases to near zero as well as slow flow. The lag-zero
 * and lag-one autocorrelations R0 and R1, averaged over the resolution cell as a scanner's range and lateral averaging
 * does, give the power, the velocity Nyquist·arg(R1)/π (Kasai) and the variance 1 − |R1|/R0. Until then the field was the true velocity plus a Gaussian scatter and the wall filter a cut on
 * the true velocity, which kept a jet at twice Nyquist that a real filter rejects.
 */
export function kasaiEstimate(
  cellV: Float32Array,
  cellSigmaV: Float32Array,
  cellRel: Float32Array,
  liMin: number,
  siMin: number,
  nL: number,
  nS: number,
  scaleMps: number,
  wallFilterMps: number,
  estimate: ColorEstimate,
  cellClass?: Float32Array,
): KasaiResult {
  const n = nL * nS;
  const stream = estimate.realization * 64;
  // the blood's speckle amplitude: 2-look incoherent range-cell envelope (multi-cycle Color Doppler
  // transmit burst spans ~2 axial scatterer sub-volumes), unit mean power, correlated over the resolution cell
  const zr = colorScratch('zr', n);
  const zi = colorScratch('zi', n);
  const zr2 = colorScratch('zr2', n);
  const zi2 = colorScratch('zi2', n);
  const tmp = colorScratch('tmp', n);
  for (let l = 0; l < nL; l++)
    for (let k = 0; k < nS; k++) {
      const idx = l * nS + k;
      zr[idx] = hashGaussian(liMin + l, siMin + k, stream, estimate.seed);
      zi[idx] = hashGaussian(liMin + l, siMin + k, stream + 2, estimate.seed);
      zr2[idx] = hashGaussian(liMin + l, siMin + k, stream + 40, estimate.seed);
      zi2[idx] = hashGaussian(liMin + l, siMin + k, stream + 42, estimate.seed);
    }
  smoothGrid(zr, nL, nS, tmp);
  smoothGrid(zi, nL, nS, tmp);
  smoothGrid(zr2, nL, nS, tmp);
  smoothGrid(zi2, nL, nS, tmp);
  // the pulse-to-pulse jitter of each pulse, correlated over the same cell: neighbouring samples share their scatterers
  const N = COLOR_PACKET;
  for (let p = 0; p < N; p++) {
    const j = colorScratch(`jit${p}`, n);
    for (let l = 0; l < nL; l++)
      for (let k = 0; k < nS; k++)
        j[l * nS + k] = hashGaussian(liMin + l, siMin + k, stream + 4 + 2 * p, estimate.seed);
    smoothGrid(j, nL, nS, tmp);
    for (let g = 0; g < n; g++) j[g]! /= 0.375;
    JIT_SLOTS[p] = j;
  }
  // [1 2 1]/4 in two directions keeps (6/16)² of the variance; each part then carries half the unit power
  const unit = 1 / (0.375 * Math.SQRT2);
  // 3-basis orthonormal Legendre regression wall filter (P0 mean + P1 linear + P2 quadratic), its −3 dB point on the
  // requested cutoff (`wallFilterShares`)
  const { mean: meanShare, trend: trendShare, quad: quadShare } = wallFilterShares(
    wallFilterMps,
    scaleMps,
    WF_SHARES,
  );
  const r0 = colorScratch('r0', n);
  const r1r = colorScratch('r1r', n);
  const r1i = colorScratch('r1i', n);
  const valid = colorScratch('valid', n);
  valid.fill(0);

  for (let g = 0; g < n; g++) {
    const v = cellV[g]!;
    if (Number.isNaN(v)) continue;
    const cls = cellClass ? cellClass[g]! : 1;
    if (cls < 0.25) continue;
    valid[g] = cls;
    const isClutter = cls < 0.75;
    const l = Math.floor(g / nS);
    const k = g - l * nS;
    const li = liMin + l;
    const si = siMin + k;
    const amp = Math.sqrt(cellRel[g]!);
    const z1r = zr[g]!;
    const z1i = zi[g]!;
    const z2r = zr2[g]!;
    const z2i = zi2[g]!;
    const p1 = z1r * z1r + z1i * z1i;
    const p2 = z2r * z2r + z2i * z2i;
    const lookBoost = Math.sqrt(0.5 * (1 + p2 / Math.max(1e-8, p1)));
    const ar = amp * z1r * unit * lookBoost;
    const ai = amp * z1i * unit * lookBoost;
    const phi = (Math.PI * v) / scaleMps;
    const turbulent = (Math.PI * cellSigmaV[g]!) / scaleMps / Math.SQRT2;
    const baseJitter = isClutter ? CLUTTER_JITTER_RAD : TRANSIT_JITTER_RAD;
    const jitter = Math.hypot(baseJitter, turbulent);
    const noise = Math.sqrt(NOISE_POWER / 2);
    for (let p = 0; p < N; p++) {
      const ph = p * phi + jitter * JIT_SLOTS[p]![g]!;
      const c = Math.cos(ph);
      const sn = Math.sin(ph);
      hashGaussianPair(li, si, stream + 24 + 2 * p, estimate.seed, PAIR_TMP);
      XR_BUF[p] = ar * c - ai * sn + noise * PAIR_TMP.a;
      XI_BUF[p] = ar * sn + ai * c + noise * PAIR_TMP.b;
    }
    // 3-basis orthonormal Legendre regression wall filter (P0, P1, P2)
    if (meanShare > 0) {
      let mr = 0;
      let mi = 0;
      for (let p = 0; p < N; p++) {
        mr += XR_BUF[p]!;
        mi += XI_BUF[p]!;
      }
      mr = (mr / N) * meanShare;
      mi = (mi / N) * meanShare;
      for (let p = 0; p < N; p++) {
        XR_BUF[p]! -= mr;
        XI_BUF[p]! -= mi;
      }
      if (trendShare > 0) {
        let pr = 0;
        let pi = 0;
        for (let p = 0; p < N; p++) {
          pr += XR_BUF[p]! * LIN_BASIS[p]!;
          pi += XI_BUF[p]! * LIN_BASIS[p]!;
        }
        pr *= trendShare;
        pi *= trendShare;
        for (let p = 0; p < N; p++) {
          XR_BUF[p]! -= pr * LIN_BASIS[p]!;
          XI_BUF[p]! -= pi * LIN_BASIS[p]!;
        }
        if (quadShare > 0) {
          let qr = 0;
          let qi = 0;
          for (let p = 0; p < N; p++) {
            qr += XR_BUF[p]! * QUAD_BASIS[p]!;
            qi += XI_BUF[p]! * QUAD_BASIS[p]!;
          }
          qr *= quadShare;
          qi *= quadShare;
          for (let p = 0; p < N; p++) {
            XR_BUF[p]! -= qr * QUAD_BASIS[p]!;
            XI_BUF[p]! -= qi * QUAD_BASIS[p]!;
          }
        }
      }
    }
    let a0 = 0;
    let a1r = 0;
    let a1i = 0;
    for (let p = 0; p < N; p++) a0 += XR_BUF[p]! * XR_BUF[p]! + XI_BUF[p]! * XI_BUF[p]!;
    for (let p = 0; p + 1 < N; p++) {
      // x[p+1] · conj(x[p])
      a1r += XR_BUF[p + 1]! * XR_BUF[p]! + XI_BUF[p + 1]! * XI_BUF[p]!;
      a1i += XI_BUF[p + 1]! * XR_BUF[p]! - XR_BUF[p + 1]! * XI_BUF[p]!;
    }
    r0[g] = a0 / N;
    r1r[g] = a1r / (N - 1);
    r1i[g] = a1i / (N - 1);
  }

  // averaged over the resolution cell ([1 2 1] in range and across lines), strictly within the same class
  // (blood 1.0 vs tissue clutter 0.5) so wall flash never biases adjacent laminar blood flow
  const avg = (src: Float32Array, name: string): Float32Array => {
    const outA = colorScratch(name, n);
    outA.fill(0);
    for (let l = 0; l < nL; l++)
      for (let k = 0; k < nS; k++) {
        const g = l * nS + k;
        const cls = valid[g]!;
        if (cls < 0.25) continue;
        let sum = 0;
        let w = 0;
        for (let dl = -1; dl <= 1; dl++)
          for (let dk = -1; dk <= 1; dk++) {
            const ll = l + dl;
            const kk = k + dk;
            if (ll < 0 || ll >= nL || kk < 0 || kk >= nS) continue;
            const h = ll * nS + kk;
            const nCls = valid[h]!;
            if (nCls < 0.25 || Math.abs(nCls - cls) >= 0.25) continue;
            const wt = (dl === 0 ? 2 : 1) * (dk === 0 ? 2 : 1);
            sum += wt * src[h]!;
            w += wt;
          }
        outA[g] = sum / w;
      }
    return outA;
  };

  const R0 = avg(r0, 'R0');
  const R1r = avg(r1r, 'R1r');
  const R1i = avg(r1i, 'R1i');
  const velocity = colorScratch('velocity', n).fill(NaN);
  const power = colorScratch('power', n).fill(0);
  const variance = colorScratch('variance', n).fill(0);
  const coherence = colorScratch('coherence', n).fill(0);
  for (let g = 0; g < n; g++) {
    if (!valid[g]) continue;
    const p0 = R0[g]!;
    if (p0 <= 0) continue;
    velocity[g] = (scaleMps * Math.atan2(R1i[g]!, R1r[g]!)) / Math.PI;
    // the noise is part of what the receiver measures: subtract its expected share, as a scanner's power estimate does
    power[g] = Math.max(0, p0 - NOISE_POWER);
    // the variance over what a laminar flow already shows through transit and noise: laminar flow reads 0, as on a
    // scanner whose variance map starts at the system's own decorrelation
    coherence[g] = Math.hypot(R1r[g]!, R1i[g]!) / p0;
    const raw = 1 - coherence[g]!;
    variance[g] = Math.min(
      1,
      Math.max(0, (raw - LAMINAR_DECORRELATION) / (1 - LAMINAR_DECORRELATION)),
    );
  }
  return { velocity, power, variance, coherence };
}
