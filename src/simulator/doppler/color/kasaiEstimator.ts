import { hash3 } from '@/core/random';

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
 * colour texture was found to calibrate against.
 */
export const NOISE_POWER = 0.02;
export const TRANSIT_JITTER_RAD = 0.4;
/** Below this |R1|/R0 a weak echo is rejected as noise, unless it is NOISE_POWER_GATE times over the display threshold. */
export const NOISE_COHERENCE = 0.5;
export const NOISE_POWER_GATE = 2.5;
/** 1 − |R1|/R0 of a strong laminar flow: the transit jitter and the receiver noise alone. */
export const LAMINAR_DECORRELATION =
  1 - Math.exp(-(TRANSIT_JITTER_RAD ** 2)) / (1 + NOISE_POWER);

/** Precomputed orthonormal linear regression basis over the fixed packet length COLOR_PACKET. */
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
 * 0.9 × dispersion × speed of spectrum.ts), plus receiver noise. A regression wall filter removes the packet's mean (and
 * its linear trend for a high cutoff): it acts on the sampled phase, so it is periodic in the PRF and rejects a jet that
 * aliases to near zero as well as slow flow. The lag-zero and lag-one autocorrelations R0 and R1, averaged over the
 * resolution cell as a scanner's range and lateral averaging does, give the power, the velocity Nyquist·arg(R1)/π (Kasai) and the
 * variance 1 − |R1|/R0. Until then the field was the true velocity plus a Gaussian scatter and the wall filter a cut on
 * the true velocity, which kept a jet at twice Nyquist that a real filter rejects.
 */
export function kasaiEstimate(
  cellV: Float32Array,
  cellDisp: Float32Array,
  cellRel: Float32Array,
  liMin: number,
  siMin: number,
  nL: number,
  nS: number,
  scaleMps: number,
  wallFilterMps: number,
  estimate: ColorEstimate,
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
  // the wall filter follows its cutoff continuously (decision 304): the share of the packet's mean it removes grows to the
  // whole of it at 10 % of Nyquist, then the share of its linear trend up to 35 % (removing part of the trend below that
  // biased a laminar 0.30 m/s to 0.31). With an order chosen by thresholds the
  // slider had three states: 0.01-0.09 m/s drew the same image, and so did 0.10-0.30
  const cut = Math.max(0, wallFilterMps / scaleMps);
  const meanShare = Math.min(1, cut / 0.1);
  const trendShare = Math.min(1, Math.max(0, (cut - 0.16) / 0.24));
  const r0 = colorScratch('r0', n);
  const r1r = colorScratch('r1r', n);
  const r1i = colorScratch('r1i', n);
  const valid = colorScratch('valid', n);
  valid.fill(0);

  for (let g = 0; g < n; g++) {
    const v = cellV[g]!;
    if (Number.isNaN(v)) continue;
    valid[g] = 1;
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
    const turbulent = (Math.PI * 0.9 * cellDisp[g]! * Math.abs(v)) / scaleMps / Math.SQRT2;
    const jitter = Math.hypot(TRANSIT_JITTER_RAD, turbulent);
    const noise = Math.sqrt(NOISE_POWER / 2);
    for (let p = 0; p < N; p++) {
      const ph = p * phi + jitter * JIT_SLOTS[p]![g]!;
      const c = Math.cos(ph);
      const sn = Math.sin(ph);
      hashGaussianPair(li, si, stream + 24 + 2 * p, estimate.seed, PAIR_TMP);
      XR_BUF[p] = ar * c - ai * sn + noise * PAIR_TMP.a;
      XI_BUF[p] = ar * sn + ai * c + noise * PAIR_TMP.b;
    }
    // regression wall filter: the packet less its projection on the low-order polynomials, in the shares above
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

  // averaged over the resolution cell ([1 2 1] in range and across lines), over the samples that carry colour only
  const avg = (src: Float32Array, name: string): Float32Array => {
    const outA = colorScratch(name, n);
    outA.fill(0);
    for (let l = 0; l < nL; l++)
      for (let k = 0; k < nS; k++) {
        const g = l * nS + k;
        if (!valid[g]) continue;
        let sum = 0;
        let w = 0;
        for (let dl = -1; dl <= 1; dl++)
          for (let dk = -1; dk <= 1; dk++) {
            const ll = l + dl;
            const kk = k + dk;
            if (ll < 0 || ll >= nL || kk < 0 || kk >= nS) continue;
            const h = ll * nS + kk;
            if (!valid[h]) continue;
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
