import { aliasVelocity } from '@/clinical/formulas';
import { COLOR_THRESHOLD, type ColorSettings } from '@/simulator/doppler/color/colorDoppler';
import type { WallFilterResponseTable } from '@/simulator/doppler/color/kasaiEstimator';
import type { MmodeLine } from '../mmodeStrip';

/**
 * Writes column `col` of the colour M-mode strip ([sample × cols + col]) from the line it is formed from. Each blood
 * sample keeps the power the 2D colour's regression wall filter leaves its velocity, on the request bounded at this
 * Nyquist (`wallFilter`, set here), times its relative transmission and the gain; above the display threshold it is
 * painted with its aliased velocity and its dispersion. No packet is formed: the strip has the filter's response, not
 * the estimate's scatter. The response acts on the sampled phase, so a flow aliasing to near zero is rejected too. It
 * was a hard cut at the request on the true velocity, which with the 20 cm/s default left nothing under 20 cm/s while
 * the 2D colours half the 15–20 cm/s flow on the same line (A4C through the crux: strip colour 17.7 % with the old
 * 6 cm/s cut, 9.5 % with the hard 20 cm/s one, 11.2 % with this response).
 */
export function writeCmmColumn(
  line: MmodeLine,
  samples: number,
  color: ColorSettings,
  wallFilter: WallFilterResponseTable,
  col: number,
  cols: number,
  cmm: Float32Array,
  cmmVar: Float32Array | null,
  cmmPow: Float32Array | null,
): void {
  const v = line.velocity;
  const d = line.dispersion;
  const tr = line.relTrans;
  const gainLin = Math.pow(10, color.gainDb / 20);
  wallFilter.set(color.wallFilterMps, color.scaleMps);
  for (let si = 0; si < samples; si++) {
    const idx = si * cols + col;
    const x = v ? v[si]! : NaN;
    const vx = color.invert ? -x : x;
    const rel = tr ? tr[si]! : 1;
    const pow = Number.isNaN(vx) ? 0 : Math.min(1, gainLin * rel * wallFilter.at(vx));
    if (pow < COLOR_THRESHOLD) {
      cmm[idx] = NaN;
      if (cmmVar) cmmVar[idx] = 0;
      if (cmmPow) cmmPow[idx] = 0;
      continue;
    }
    cmm[idx] = aliasVelocity(vx, color.scaleMps, color.baselineShiftMps);
    if (cmmVar) cmmVar[idx] = d ? d[si]! : 0;
    if (cmmPow) cmmPow[idx] = pow;
  }
}
