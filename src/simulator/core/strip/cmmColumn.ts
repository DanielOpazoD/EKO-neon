import { aliasVelocity } from '@/clinical/formulas';
import { Tissue } from '@/simulator/anatomy/tissue';
import {
  COLOR_THRESHOLD,
  DOPPLER_SHADOW_TRANSMISSION,
  type ColorSettings,
} from '@/simulator/doppler/color/colorDoppler';
import {
  CLUTTER_BLOOD_RATIO_POW,
  CLUTTER_SIGMA_V_MPS,
  colorScratch,
  kasaiEstimate,
  NOISE_COHERENCE,
  NOISE_POWER_GATE,
  type ColorEstimate,
} from '@/simulator/doppler/color/kasaiEstimator';
import { encodeColorBreakthroughPower } from '@/simulator/doppler/color/spatiotemporalFilter';
import type { MmodeLine } from '../mmodeStrip';

/**
 * Writes column `col` of the colour M-mode strip ([sample × cols + col]) from the traced line using the
 * full 8-pulse Kasai estimator (`kasaiEstimate`), identical in physics to 2D Color Doppler: acoustic shadow
 * gating, gain-driven blood blooming, blood vs myocardial/valve tissue clutter classes, 3-basis orthonormal
 * Legendre regression wall filtering (periodic in PRF), axial [1 2 1] resolution-cell autocorrelation averaging,
 * noise coherence gating, Nyquist aliasing, and tissue-write-priority breakthrough encoding.
 */
export function writeCmmColumn(
  line: MmodeLine,
  samples: number,
  color: ColorSettings,
  columnSerial: number,
  estimate: ColorEstimate,
  col: number,
  cols: number,
  cmm: Float32Array,
  cmmVar: Float32Array | null,
  cmmPow: Float32Array | null,
): void {
  const v = line.velocity;
  const sig = line.sigmaV;
  const tr = line.relTrans;
  const vw = line.wallVelocity;
  const ti = line.tissue;
  const gainLin = Math.pow(10, color.gainDb / 20);
  const bloomSamples = Math.max(0, Math.round((color.gainDb - 2) * 0.6));

  const cellV = colorScratch('cmmCellV', samples).fill(NaN);
  const cellSigmaV = colorScratch('cmmCellSigmaV', samples).fill(0);
  const cellRel = colorScratch('cmmCellRel', samples).fill(0);
  const cellClass = colorScratch('cmmCellClass', samples).fill(0);

  for (let si = 0; si < samples; si++) {
    const rel = tr ? tr[si]! : 1;
    if (rel < DOPPLER_SHADOW_TRANSMISSION) continue;

    let isBlood: boolean;
    let isClutterTissue: boolean;
    if (ti) {
      const tId = ti[si]!;
      isBlood = tId === Tissue.Blood;
      if (!isBlood && bloomSamples > 0) {
        for (let d = 1; d <= bloomSamples && !isBlood; d++) {
          if (si - d >= 0 && ti[si - d] === Tissue.Blood) isBlood = true;
          if (!isBlood && si + d < samples && ti[si + d] === Tissue.Blood) isBlood = true;
        }
      }
      isClutterTissue = tId === Tissue.Myocardium || tId === Tissue.Valve;
    } else {
      const vRaw = v ? v[si]! : NaN;
      isBlood = vRaw === vRaw;
      isClutterTissue = !isBlood && vw != null && Math.abs(vw[si]!) > 1e-4;
    }

    const vRaw = v ? v[si]! : NaN;
    if (isBlood && vRaw === vRaw) {
      cellV[si] = color.invert ? -vRaw : vRaw;
      cellSigmaV[si] = sig ? sig[si]! : 0;
      cellRel[si] = Math.min(1, rel);
      cellClass[si] = 1;
    } else if (isClutterTissue && vw && Math.abs(vw[si]!) > 1e-4) {
      const wv = vw[si]!;
      cellV[si] = color.invert ? -wv : wv;
      cellSigmaV[si] = CLUTTER_SIGMA_V_MPS;
      cellRel[si] = Math.min(1, rel) * CLUTTER_BLOOD_RATIO_POW;
      cellClass[si] = 0.5;
    }
  }

  const kasai = kasaiEstimate(
    cellV,
    cellSigmaV,
    cellRel,
    columnSerial,
    0,
    1,
    samples,
    color.scaleMps,
    color.wallFilterMps,
    estimate,
    cellClass,
  );

  for (let si = 0; si < samples; si++) {
    const idx = si * cols + col;
    const cls = cellClass[si]!;
    const kv = kasai.velocity[si]!;
    if (cls >= 0.25 && kv === kv) {
      const pow = Math.min(1, gainLin * kasai.power[si]!);
      const rejectedNoise =
        kasai.coherence[si]! < NOISE_COHERENCE && pow < NOISE_POWER_GATE * COLOR_THRESHOLD;
      if (!rejectedNoise && pow >= COLOR_THRESHOLD) {
        cmm[idx] = aliasVelocity(kv, color.scaleMps, color.baselineShiftMps);
        if (cmmVar) cmmVar[idx] = kasai.variance[si]!;
        if (cmmPow) cmmPow[idx] = encodeColorBreakthroughPower(pow, cls < 0.75, color.gainDb);
        continue;
      }
    }
    cmm[idx] = NaN;
    if (cmmVar) cmmVar[idx] = 0;
    if (cmmPow) cmmPow[idx] = 0;
  }
}
