import type { HemodynamicConfig } from '@/cases/schema';
import { aorticPulse, SYSTOLIC_CLOSURE_S } from './cycleState';
import { TRICUSPID_LAG_S, type CycleTimings } from './timing';

/**
 * Time constant (s) of RV isovolumic relaxation: shorter than the LV's 45 ms because the right
 * ventricle relaxes against pulmonary arterial pressures.
 */
const RV_RELAXATION_TAU_S = 0.03;

/**
 * Tricuspid regurgitation waveform driven by an RV pressure model over the right ventricle's own
 * valve timings (tricuspid closure → RV IVCT → RV ejection [pulmonary opening..closure] → RV IVRT →
 * tricuspid opening), mirroring the LV pressure curve used for MR in `cycleModel.ts`.
 * Constant declared EROA: Q(t) = EROA·v(t), RVol = EROA·VTI (ASE regurgitation 2017).
 */
export function tricuspidRegurgitation(
  hemo: HemodynamicConfig,
  timings: CycleTimings,
  rrS: number,
  n: number,
) {
  const velocityMps = new Float32Array(n);
  const flowMlps = new Float32Array(n);
  const rap = hemo.rapMmHg;
  const pasp = hemo.paspMmHg;
  const vmaxMps = hemo.trPresent ? Math.sqrt(Math.max(0, (pasp - rap) / 4)) : 0;
  const eroaCm2 = hemo.regurgitation.tr?.eroaCm2 ?? 0;
  const dt = rrS / n;

  // Tricuspid closes TRICUSPID_LAG_S after mitral closure (T1 after M1), leaving a genuine RV IVCT
  // before pulmonary opening; TR ends when the tricuspid valve opens after RV IVRT.
  const tClose = Math.min(timings.pulmonaryOpenS - 0.01, SYSTOLIC_CLOSURE_S + TRICUSPID_LAG_S);
  const tPvOpen = timings.pulmonaryOpenS;
  const tPvClose = timings.pulmonaryCloseS;
  const tTvOpen = timings.tricuspidOpenS;
  const ivctRv = Math.max(1e-3, tPvOpen - tClose);
  const etRv = Math.max(1e-3, tPvClose - tPvOpen);
  const ivrtRv = Math.max(0, tTvOpen - tPvClose);
  // Pulmonary arterial diastolic pressure (~40% of the systolic rise above RAP, matching adult PA pulse)
  const padp = rap + Math.max(0, pasp - rap) * 0.4;
  const pEs = padp + (pasp - padp) * aorticPulse(1);
  const tail = ivrtRv > 1e-4 ? Math.exp(-ivrtRv / RV_RELAXATION_TAU_S) : 0;

  let vtiCm = 0;
  let volumeMl = 0;
  if (hemo.trPresent && pasp > rap) {
    for (let i = 0; i < n; i++) {
      const t = (i + 0.5) * dt;
      let pRv = rap;
      if (t >= tClose && t < tPvOpen) {
        // RV isovolumic contraction: smooth rise from RAP to pulmonary diastolic pressure
        const x = (t - tClose) / ivctRv;
        pRv = rap + (padp - rap) * 0.5 * (1 - Math.cos(Math.PI * x));
      } else if (t >= tPvOpen && t < tPvClose) {
        // RV ejection: pressure dome peaking at PASP (exact vmaxMps = sqrt((PASP - RAP) / 4))
        const u = (t - tPvOpen) / etRv;
        pRv = padp + (pasp - padp) * aorticPulse(u);
      } else if (t >= tPvClose && t < tTvOpen && ivrtRv > 1e-4) {
        // RV isovolumic relaxation: exponential fall from end-systolic PA pressure to RAP
        const s = t - tPvClose;
        pRv = rap + ((pEs - rap) * (Math.exp(-s / RV_RELAXATION_TAU_S) - tail)) / (1 - tail);
      }
      const velocity = pRv > rap ? Math.sqrt((pRv - rap) / 4) : 0;
      velocityMps[i] = velocity;
      flowMlps[i] = velocity * 100 * eroaCm2;
      vtiCm += velocity * 100 * dt;
      volumeMl += flowMlps[i]! * dt;
    }
  }
  return { velocityMps, flowMlps, vmaxMps, vtiCm, volumeMl };
}
