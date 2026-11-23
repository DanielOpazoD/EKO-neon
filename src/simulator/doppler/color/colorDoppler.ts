import { softTissueTransmission } from '@/simulator/renderer/acoustic/acoustics';
import { aliasVelocity } from '@/clinical/formulas';
import { Tissue } from '@/simulator/anatomy/tissue';
import type { PolarFrame } from '@/simulator/renderer/types';
import type { ScanLut } from '@/simulator/renderer/scanConvert';
import { COLOR_BLEND, colorMap } from '@/simulator/renderer/postprocess/colorMap';
import {
  COLOR_PACKET,
  colorScratch,
  kasaiEstimate,
  NOISE_COHERENCE,
  NOISE_POWER_GATE,
  type ColorEstimate,
} from './kasaiEstimator';
import {
  adaptiveTemporalStep,
  applyColorSpatialFilter,
  colorPowerConfidence,
  DEFAULT_COLOR_SMOOTHING,
  ST_POW_ALPHA_MIN,
  type TemporalStepInput,
  type TemporalStepOutput,
} from './spatiotemporalFilter';

export { COLOR_PACKET, DEFAULT_COLOR_SMOOTHING, type ColorEstimate };

export interface ColorSettings {
  boxThetaMinRad: number;
  boxThetaMaxRad: number;
  boxRMinCm: number;
  boxRMaxCm: number;
  scaleMps: number; // Nyquist
  baselineShiftMps: number;
  wallFilterMps: number;
  gainDb: number;
  persistence: number;
  /** Spatiotemporal adaptive smoothing strength in [0, 1] (0 = off, 0.6 = clinical default). */
  smoothing: number;
  showVariance: boolean;
  invert: boolean;
}

export const DEFAULT_COLOR: ColorSettings = {
  boxThetaMinRad: -0.32,
  boxThetaMaxRad: 0.32,
  boxRMinCm: 5,
  boxRMaxCm: 13,
  scaleMps: 0.66,
  baselineShiftMps: 0,
  wallFilterMps: 0.06,
  gainDb: 0,
  persistence: 0.35,
  smoothing: DEFAULT_COLOR_SMOOTHING,
  showVariance: true,
  invert: false,
};

/** Acquisition the colour field was formed with: the expected tissue attenuation depends on it. */
export interface ColorAcquisition {
  frequencyMHz: number;
  harmonics: boolean;
}

/** Below this fraction of the expected transmission a sample lies in an acoustic shadow: no Doppler signal at any gain. */
export const DOPPLER_SHADOW_TRANSMISSION = 0.02;

/**
 * Two-way transmission of a frame sample relative to what soft tissue (0.5 dB/cm/MHz) would leave at its depth for the
 * acquisition that formed the frame (decisions 87 and 96): depth alone does not make a shadow.
 */
export function relativeTransmission(
  transmission: number,
  depthCm: number,
  acquisition: ColorAcquisition,
): number {
  return (
    transmission / softTissueTransmission(depthCm, acquisition.frequencyMHz, acquisition.harmonics)
  );
}

/** Second-order wall-filter magnitude at |v| over the cutoff: −3 dB at the cutoff, flat from about twice it. */
export function wallFilterResponse(vAbs: number, cutoffMps: number): number {
  if (cutoffMps <= 0) return 1;
  const x = (vAbs / cutoffMps) ** 2;
  return x / Math.sqrt(1 + x * x);
}

/** Per-sample colour result buffers (polar). vel in m/s after aliasing, NaN where no colour. */
export interface ColorField {
  vel: Float32Array;
  variance: Float32Array;
  power: Float32Array;
  /** Pre-spatial temporally filtered history buffers (prevents recursive spatial blur drift across frames). */
  rawVel?: Float32Array;
  rawVariance?: Float32Array;
  rawPower?: Float32Array;
  /** Velocity scale, baseline and inversion the field was formed with: its phases mean nothing under other ones. */
  scaleMps: number;
  baselineShiftMps: number;
  invert: boolean;
}

/** Display threshold of the colour power (after gain). */
export const COLOR_THRESHOLD = 0.25;

export function allocColorField(n: number): ColorField {
  return {
    vel: new Float32Array(n).fill(NaN),
    variance: new Float32Array(n),
    power: new Float32Array(n),
    rawVel: new Float32Array(n).fill(NaN),
    rawVariance: new Float32Array(n),
    rawPower: new Float32Array(n),
    scaleMps: 0,
    baselineShiftMps: 0,
    invert: false,
  };
}

/**
 * Compute colour for samples inside the box. `axialVelocity(idx)` returns the projected velocity
 * (m/s, +toward) and dispersion for a polar sample, computed by the caller from the flow field.
 * Blooming: gain above 0 extends colour into adjacent non-blood samples (dilation).
 * Signal power (decision 87) is the colour gain × the sample's two-way transmission relative to what soft tissue would
 * leave at that depth for this acquisition × the wall-filter response, so low gain loses attenuated, partly shadowed and
 * slow flow first; under a real shadow (< 2 % of the expected transmission) there is no signal at any gain. It used to
 * depend on velocity only, with the shadow threshold computed at a fixed 2.5 MHz.
 * Persistence blends each coloured sample with `prev`, the field of the previous update, as the lag-one autocorrelation
 * the velocity comes from: power-weighted phasors at the Doppler phase of each velocity, whose angle is the blended
 * velocity. Averaging aliased velocities linearly turned two flows on either side of Nyquist into a false slow flow.
 * `prev` must be a different buffer from `out`, which is cleared first.
 */
export function computeColorField(
  frame: PolarFrame,
  s: ColorSettings,
  prev: ColorField | null,
  axialVelocity: (
    idx: number,
    li: number,
    si: number,
    out: { v: number; disp: number; present: number },
  ) => void,
  out: ColorField,
  acquisition: ColorAcquisition,
  estimate?: ColorEstimate,
): { colorLines: number } {
  const { lines, samples, sectorRad, depthCm } = frame.spec;
  const tmp = { v: 0, disp: 0, present: 0 };
  const half = sectorRad / 2;
  const liMin = Math.max(0, Math.floor(((s.boxThetaMinRad + half) / sectorRad) * lines));
  const liMax = Math.min(lines - 1, Math.ceil(((s.boxThetaMaxRad + half) / sectorRad) * lines));
  const siMin = Math.max(0, Math.floor((s.boxRMinCm / depthCm) * samples));
  const siMax = Math.min(samples - 1, Math.ceil((s.boxRMaxCm / depthCm) * samples));
  const totalSamples = lines * samples;
  if (!out.rawVel || out.rawVel.length !== totalSamples) {
    out.rawVel = new Float32Array(totalSamples);
    out.rawVariance = new Float32Array(totalSamples);
    out.rawPower = new Float32Array(totalSamples);
  }
  out.vel.fill(NaN);
  out.power.fill(0);
  out.variance.fill(0);
  out.rawVel.fill(NaN);
  out.rawPower!.fill(0);
  out.rawVariance!.fill(0);
  out.scaleMps = s.scaleMps;
  out.baselineShiftMps = s.baselineShiftMps;
  out.invert = s.invert;
  // a history formed under another scale, baseline or inversion is not blended: its velocities are other phases
  const history =
    prev &&
    prev.scaleMps === s.scaleMps &&
    prev.baselineShiftMps === s.baselineShiftMps &&
    prev.invert === s.invert
      ? prev
      : null;
  const histVel = history ? (history.rawVel ?? history.vel) : null;
  const histVar = history ? (history.rawVariance ?? history.variance) : null;
  const histPow = history ? (history.rawPower ?? history.power) : null;
  const smoothing = s.smoothing ?? DEFAULT_COLOR_SMOOTHING;
  const gainLin = Math.pow(10, s.gainDb / 20);
  const bloomSamples = Math.max(0, Math.round((s.gainDb - 2) * 0.6)); // dilation radius grows with gain
  const nL = Math.max(0, liMax - liMin + 1);
  const nS = Math.max(0, siMax - siMin + 1);
  // pass 1: what reaches the receiver at each sample of the box — moving blood (or, with blooming, a sample beside it)
  // over the shadow threshold: its axial velocity, the dispersion of its spectrum and its transmission
  const cellV = colorScratch('cellV', nL * nS).fill(NaN);
  const cellDisp = colorScratch('cellDisp', nL * nS).fill(0);
  const cellRel = colorScratch('cellRel', nL * nS).fill(0);
  const bloodMask = colorScratch('bloodMask', nL * nS).fill(0);
  for (let li = liMin; li <= liMax; li++) {
    for (let si = siMin; si <= siMax; si++) {
      const idx = li * samples + si;
      const tissue = frame.tissue[idx];
      const trans = frame.transmission[idx] ?? 0;
      const r = ((si + 0.5) / samples) * depthCm;
      const relTrans = relativeTransmission(trans, r, acquisition);
      if (relTrans < DOPPLER_SHADOW_TRANSMISSION) continue; // acoustic shadow: no Doppler signal at any gain
      let isBlood = tissue === Tissue.Blood;
      if (!isBlood && bloomSamples > 0) {
        // blooming: colour bleeds onto adjacent tissue when gain is high
        for (let d = 1; d <= bloomSamples && !isBlood; d++) {
          if (frame.tissue[idx - d] === Tissue.Blood || frame.tissue[idx + d] === Tissue.Blood)
            isBlood = true;
          if (li > 0 && frame.tissue[idx - samples] === Tissue.Blood) isBlood = true;
          if (li < lines - 1 && frame.tissue[idx + samples] === Tissue.Blood) isBlood = true;
        }
      }
      if (!isBlood) continue;
      axialVelocity(idx, li, si, tmp);
      if (!tmp.present) continue;
      const g = (li - liMin) * nS + (si - siMin);
      cellV[g] = s.invert ? -tmp.v : tmp.v;
      cellDisp[g] = tmp.disp;
      cellRel[g] = Math.min(1, relTrans);
      bloodMask[g] = 1;
    }
  }
  // the estimate of this update (decision 300): a packet of echoes per sample, wall-filtered and autocorrelated
  const kasai =
    estimate && nL > 0 && nS > 0
      ? kasaiEstimate(
          cellV,
          cellDisp,
          cellRel,
          liMin,
          siMin,
          nL,
          nS,
          s.scaleMps,
          s.wallFilterMps,
          estimate,
        )
      : null;

  const tIn: TemporalStepInput = {
    isBloodValid: false,
    hasCurr: false,
    vCurr: 0,
    varCurr: 0,
    powCurr: 0,
    hasPrev: false,
    vPrev: 0,
    varPrev: 0,
    powPrev: 0,
    scaleMps: s.scaleMps,
    baselineShiftMps: s.baselineShiftMps,
    persistence: s.persistence,
    smoothing,
    colorThreshold: COLOR_THRESHOLD,
  };
  const tOut: TemporalStepOutput = { valid: false, vel: NaN, variance: 0, power: 0 };

  for (let li = liMin; li <= liMax; li++) {
    for (let si = siMin; si <= siMax; si++) {
      const g = (li - liMin) * nS + (si - siMin);
      const vTrue = cellV[g]!;
      const isBloodValid = !Number.isNaN(vTrue);
      if (!isBloodValid) continue;

      const idx = li * samples + si;
      let hasCurr = false;
      let power = 0;
      let vEstimate = 0;
      let variance = 0;

      if (kasai) {
        const kv = kasai.velocity[g]!;
        if (!Number.isNaN(kv)) {
          power = Math.min(1, gainLin * kasai.power[g]!);
          vEstimate = kv;
          variance = kasai.variance[g]!;
          // noise rejection, as a scanner's: a weak echo whose packet has lost its coherence is noise and leaks of the
          // wall filter, not flow (they drew green and red confetti over slow intracavitary blood). A strong incoherent
          // echo is turbulence and keeps its mosaic
          const rejectedNoise =
            kasai.coherence[g]! < NOISE_COHERENCE && power < NOISE_POWER_GATE * COLOR_THRESHOLD;
          hasCurr = !rejectedNoise && power >= COLOR_THRESHOLD;
        }
      } else {
        // the expected field: the wall filter's response on the true velocity, its spectral spread as variance
        if (Math.abs(vTrue) >= s.wallFilterMps) {
          power = Math.min(
            1,
            gainLin * cellRel[g]! * wallFilterResponse(Math.abs(vTrue), s.wallFilterMps),
          );
          vEstimate = vTrue;
          variance = Math.min(1, cellDisp[g]! * 1.6);
          hasCurr = power >= COLOR_THRESHOLD;
        }
      }

      const pv = histVel ? (histVel[idx] ?? NaN) : NaN;
      const pp = histPow ? (histPow[idx] ?? 0) : 0;
      const hasPrev = !Number.isNaN(pv) && pp > 0;
      if (!hasCurr && !hasPrev) continue;

      tIn.isBloodValid = true;
      tIn.hasCurr = hasCurr;
      tIn.vCurr = hasCurr ? aliasVelocity(vEstimate, s.scaleMps, s.baselineShiftMps) : 0;
      tIn.varCurr = variance;
      tIn.powCurr = power;
      tIn.hasPrev = hasPrev;
      tIn.vPrev = hasPrev ? pv : 0;
      tIn.varPrev = hasPrev ? (histVar![idx] ?? 0) : 0;
      tIn.powPrev = pp;

      adaptiveTemporalStep(tIn, tOut);
      if (tOut.valid) {
        out.rawVel[idx] = tOut.vel;
        out.rawVariance![idx] = tOut.variance;
        out.rawPower![idx] = tOut.power;
      }
    }
  }

  // Pass 3 (CPU mirror of GPU Pass D3): coherence- & variance-adaptive circular spatial filter
  applyColorSpatialFilter({
    samples,
    liMin,
    liMax,
    siMin,
    siMax,
    scaleMps: s.scaleMps,
    baselineShiftMps: s.baselineShiftMps,
    smoothing,
    bloodMask,
    srcVel: out.rawVel,
    srcVar: out.rawVariance!,
    srcPow: out.rawPower!,
    dstVel: out.vel,
    dstVar: out.variance,
    dstPow: out.power,
  });

  return { colorLines: Math.max(0, liMax - liMin + 1) };
}

export { colorMap } from '@/simulator/renderer/postprocess/colorMap';

/**
 * Blend a polar colour field over a scan-converted grey image inside the colour box using
 * bilinear circular interpolation in the baseline-centered Nyquist complex plane
 * (`e^{i·π·(v − baseline)/scale}`) so aliased transitions across `baseline ± scale` (including
 * shifted PISA aliasing shells) never produce false zero-velocity bands or spurious wraps,
 * with smooth power-modulated jet boundaries and clinical tissue write priority over leaflets/walls.
 * The GPU present pass (gpu/glslImage.ts) evaluates the same math.
 */
export function overlayColorField(
  rgba: Uint8ClampedArray,
  lut: ScanLut,
  vel: Float32Array,
  variance: Float32Array,
  c: ColorSettings,
  power?: Float32Array,
): void {
  const rgb: [number, number, number] = [0, 0, 0];
  const S = lut.samples;
  const maxL = lut.lines - 1;
  const maxS = S - 1;
  const n = lut.idx.length;
  const scale = Math.max(1e-3, c.scaleMps);
  const baseline = c.baselineShiftMps;
  const kPhi = Math.PI / scale;
  const invPiScale = scale / Math.PI;
  const powAlphaAt = (idx: number): number => {
    if (!power) return 1;
    const conf = colorPowerConfidence(power[idx] ?? 1, COLOR_THRESHOLD);
    return ST_POW_ALPHA_MIN + (1 - ST_POW_ALPHA_MIN) * conf;
  };
  for (let p = 0, o = 0; p < n; p++, o += 4) {
    const i0 = lut.idx[p] ?? -1;
    if (i0 < 0) continue;
    const r = lut.rCm[p] ?? 0;
    if (r < c.boxRMinCm || r > c.boxRMaxCm) continue;
    const th = lut.theta[p] ?? 0;
    if (th < c.boxThetaMinRad || th > c.boxThetaMaxRad) continue;

    const l0 = (i0 / S) | 0;
    const s0 = i0 - l0 * S;
    const dl = l0 < maxL ? S : 0;
    const ds = s0 < maxS ? 1 : 0;
    const v00 = vel[i0]!;
    const v10 = vel[i0 + dl]!;
    const v01 = vel[i0 + ds]!;
    const v11 = vel[i0 + dl + ds]!;
    // Fast skip when all four corners carry no flow
    if (v00 !== v00 && v10 !== v10 && v01 !== v01 && v11 !== v11) continue;

    const fl = lut.wl[p] ?? 0;
    const fs = lut.ws[p] ?? 0;
    const w00 = (1 - fl) * (1 - fs);
    const w10 = fl * (1 - fs);
    const w01 = (1 - fl) * fs;
    const w11 = fl * fs;

    let re = 0;
    let im = 0;
    let varAcc = 0;
    let powAcc = 0;
    let wSum = 0;
    if (v00 === v00) {
      const ph = (v00 - baseline) * kPhi;
      re += w00 * Math.cos(ph);
      im += w00 * Math.sin(ph);
      varAcc += w00 * (variance[i0] ?? 0);
      powAcc += w00 * powAlphaAt(i0);
      wSum += w00;
    }
    if (v10 === v10) {
      const ph = (v10 - baseline) * kPhi;
      re += w10 * Math.cos(ph);
      im += w10 * Math.sin(ph);
      varAcc += w10 * (variance[i0 + dl] ?? 0);
      powAcc += w10 * powAlphaAt(i0 + dl);
      wSum += w10;
    }
    if (v01 === v01) {
      const ph = (v01 - baseline) * kPhi;
      re += w01 * Math.cos(ph);
      im += w01 * Math.sin(ph);
      varAcc += w01 * (variance[i0 + ds] ?? 0);
      powAcc += w01 * powAlphaAt(i0 + ds);
      wSum += w01;
    }
    if (v11 === v11) {
      const ph = (v11 - baseline) * kPhi;
      re += w11 * Math.cos(ph);
      im += w11 * Math.sin(ph);
      varAcc += w11 * (variance[i0 + dl + ds] ?? 0);
      powAcc += w11 * powAlphaAt(i0 + dl + ds);
      wSum += w11;
    }
    if (wSum <= 0.18) continue;

    const v = baseline + invPiScale * Math.atan2(im, re);
    const vari = varAcc / wSum;
    const powAlpha = powAcc / wSum;
    colorMap(v, scale, vari, c.showVariance, rgb, baseline);

    const edgeT = Math.min(1, Math.max(0, (wSum - 0.18) / 0.54));
    const edgeAlpha = edgeT * edgeT * (3 - 2 * edgeT);
    const r0 = rgba[o] ?? 0;
    const g0 = rgba[o + 1] ?? 0;
    const b0 = rgba[o + 2] ?? 0;
    const lum = 0.299 * r0 + 0.587 * g0 + 0.114 * b0;
    // Clinical tissue write priority: full color over dark blood pool, smoothly dropping to 0 on bright endocardium/valves
    const tpU = Math.min(1, Math.max(0, (lum - 98) / 74));
    const tissuePrio = 1 - tpU * tpU * (3 - 2 * tpU);
    // Gentle low-velocity opacity feathering so near-baseline flow dissolves organically into dark chamber blood
    const velU = Math.min(1, Math.max(0, (Math.abs(v) / scale - 0.05) / 0.17));
    const velAlpha = 0.62 + 0.38 * velU * velU * (3 - 2 * velU);
    const blend = COLOR_BLEND * edgeAlpha * powAlpha * velAlpha * tissuePrio;
    const inv = 1 - blend;
    rgba[o] = Math.round(rgb[0] * blend + r0 * inv);
    rgba[o + 1] = Math.round(rgb[1] * blend + g0 * inv);
    rgba[o + 2] = Math.round(rgb[2] * blend + b0 * inv);
  }
}
