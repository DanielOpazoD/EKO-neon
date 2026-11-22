import type { RhythmConfig } from '@/cases/schema';
import { PR_INTERVAL_S } from './timing';
import { hash3 } from '@/core/random';

/**
 * Synthetic educational ECG (spec 24): sum of Gaussians anchored at QRS onset (phase 0).
 * Not diagnostic. In AF: no P wave, fibrillatory baseline, irregular RR from the clock.
 */
function gauss(t: number, mu: number, sigma: number, amp: number): number {
  const d = (t - mu) / sigma;
  return amp * Math.exp(-0.5 * d * d);
}

export function ecgSample(
  timeInBeatS: number,
  rrS: number,
  rhythm: RhythmConfig,
  beatIndex = 0,
  seed = 1,
): number {
  const t = timeInBeatS;
  const qt = 0.39 * Math.sqrt(Math.min(rrS, 1.3)); // Bazett-like QT scaling
  let v = 0;
  // QRS anchored at onset t = 0 so mitral A-wave closure meets QRS onset/upstroke
  v += gauss(t, 0.014, 0.005, -0.12);
  v += gauss(t, 0.026, 0.008, 1.0);
  v += gauss(t, 0.042, 0.007, -0.25);
  // T wave
  v += gauss(t, qt - 0.072, 0.044, 0.28);
  if (rhythm.type === 'atrial-fibrillation') {
    // fibrillatory baseline: deterministic pseudo-random small waves ~6–8 Hz
    const k = Math.floor(t * 40);
    const n = hash3(k, beatIndex, 0, seed) - 0.5;
    v += 0.05 * Math.sin(t * 2 * Math.PI * 7 + n * 6) + 0.03 * n;
  } else {
    // P wave of the NEXT beat sits at rr − PR, of this beat at −PR (visible at the end of the beat)
    v += gauss(t, rrS - PR_INTERVAL_S + 0.042, 0.021, 0.16);
    if (t < 0.02) v += gauss(t, -PR_INTERVAL_S + 0.042, 0.021, 0.16);
  }
  return v;
}
