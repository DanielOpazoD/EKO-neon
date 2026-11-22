import type { PhysiologyConfig } from '@/cases/schema';
import type { ChainedBeat } from './cycleState';
import type { CycleTimings } from './timing';

/** Time constant (s) of the annuli following the contraction through ejection. */
const SYSTOLIC_TAU_S = 0.03;

export interface AnnularMotionInput {
  n: number;
  dt: number;
  rrS: number;
  timings: CycleTimings;
  physiology: PhysiologyConfig;
  vol: Float32Array;
  mrFlow: Float32Array;
  edvRef: number;
  esvRef: number;
  minV: number;
  af?: ChainedBeat;
}

export interface AnnularMotionResult {
  endSystoleS: number;
  longitudinal: Float32Array;
  longitudinalVelocity: Float32Array;
  rvLongitudinal: Float32Array;
  rvLongitudinalVelocity: Float32Array;
}

/**
 * Solves mitral and tricuspid longitudinal annular displacements and velocities across one beat
 * (decisions 106, 114, 162, 245) so tissue Doppler reproduces the case's septal e′, a′ = A·e′/E,
 * and tricuspid S′ while stopping systolic shortening at aortic closure.
 */
export function solveAnnularMotion(input: AnnularMotionInput): AnnularMotionResult {
  const { n, dt, rrS, timings, physiology, vol, mrFlow, edvRef, esvRef, minV, af } = input;
  // Longitudinal annular displacement: follows the contraction fraction with a first-order lag — close in systole and
  // during atrial contraction, limited by relaxation in early diastole. The early-diastolic time constant is solved so
  // that the annulus recoils at the case's e′ (peak MAPSE·d(long)/dt), because tissue Doppler reads this very curve.
  // It used to be 0.09·(10/e′) s, which in the normal heart (e′ 11 cm/s) recoiled at 4.6 cm/s and kept 86% of the
  // systolic descent at mid-E: every diastolic frame showed the base too apical and tissue Doppler measured e′ 4.7.
  // the annuli shorten with the ejection and stop at aortic closure: a regurgitant mitral valve keeps emptying the ventricle
  // through isovolumic relaxation, but the myocardium is no longer shortening, and an annulus that followed that volume
  // drew post-systolic shortening on tissue Doppler (decision 245). Their course reaches its full excursion at aortic closure.
  const atClosure = vol[Math.min(n - 1, Math.floor(timings.ejectionEndS / dt))] ?? esvRef;
  const esvAnnulus = esvRef + Math.max(0, atClosure - minV);
  const contraction = new Float32Array(n);
  for (let i = 0; i < n; i++)
    contraction[i] = Math.min(
      1,
      (edvRef - (vol[i] ?? edvRef)) / Math.max(edvRef - esvAnnulus, 1e-6),
    );
  const longitudinal = new Float32Array(n);
  const longVel = new Float32Array(n);
  const eWaveEnd = timings.mitralOpenS + timings.eAccelS + timings.eDecelS;
  // without atrial contraction (AF) the recoil runs with the filling until the next beat
  const earlyEnd = timings.hasAWave ? timings.aStartS : rrS;
  const contractionAt = (time: number): number => {
    const f = (((time / dt - 0.5) % n) + n) % n;
    const i0 = Math.floor(f);
    const w = f - i0;
    return (contraction[i0] ?? 0) * (1 - w) + (contraction[(i0 + 1) % n] ?? 0) * w;
  };
  // atrialSpeed > 1 compresses the course during atrial contraction the same way (decision 162), solved below for a′
  let atrialSpeed = 1;
  // speed > 1 compresses the early-diastolic course in time: a healthy annulus recoils ahead of the filling it drives
  // (e′ precedes E), so it can move faster than the volume curve alone allows
  // sysSpeed compresses the ejection course the same way (decision 106): the tricuspid annulus reaches its excursion
  // earlier in systole than the volume curve when its case S′ asks for it
  // the ventricle shortens until its volume is least: at aortic closure, or with a regurgitant mitral valve at its opening,
  // the ventricle still emptying into the atrium through isovolumic relaxation (decision 245)
  let shortenEndS = timings.ejectionEndS;
  {
    // only a regurgitant mitral valve empties the ventricle past aortic closure: a volume that drifts by a few thousandths
    // of a mL through isovolumic relaxation (the periodicity correction) is not a later end-systole
    let regurgitates = false;
    for (let i = 0; i < n; i++) {
      const t = (i + 0.5) * dt;
      if (t > timings.ejectionEndS && t < timings.mitralOpenS && (mrFlow[i] ?? 0) > 0.5)
        regurgitates = true;
    }
    let least = Infinity;
    if (regurgitates)
      for (let i = 0; i < n; i++) {
        const t = (i + 0.5) * dt;
        if (t < timings.ejectionStartS || t > timings.mitralOpenS) continue;
        if ((vol[i] ?? 0) < least - 1e-6) {
          least = vol[i] ?? 0;
          shortenEndS = Math.max(timings.ejectionEndS, t);
        }
      }
  }
  const simulate = (
    tauE: number,
    speed: number,
    sysSpeed = 1,
    into: Float32Array = longitudinal,
  ): void => {
    const start = af ? (into === longitudinal ? af.startLongitudinal : af.startRvLongitudinal) : 0;
    let l = start;
    // two passes for periodic steady state; a beat of atrial fibrillation runs once from where the previous one ended
    for (let pass = 0; pass < (af ? 1 : 2); pass++) {
      for (let i = 0; i < n; i++) {
        const t = (i + 0.5) * dt;
        const early = t > timings.mitralOpenS && t <= earlyEnd;
        const ejecting = t >= timings.ejectionStartS && t < timings.ejectionEndS;
        const atrialKick = timings.hasAWave && t > timings.aStartS;
        // compressed over the E wave, holding the diastasis value once it is reached (never running into atrial filling)
        const target = early
          ? contractionAt(
              Math.min(eWaveEnd, timings.mitralOpenS + (t - timings.mitralOpenS) * speed),
            )
          : ejecting
            ? contractionAt(
                Math.min(
                  timings.ejectionEndS,
                  timings.ejectionStartS + (t - timings.ejectionStartS) * sysSpeed,
                ),
              )
            : atrialKick
              ? contractionAt(
                  Math.min(timings.aEndS, timings.aStartS + (t - timings.aStartS) * atrialSpeed),
                )
              : (contraction[i] ?? 0);
        const tau = t < timings.ejectionEndS ? SYSTOLIC_TAU_S : t > earlyEnd ? 0.035 : tauE;
        // past aortic closure the annulus only recoils: the myocardium has stopped shortening, and a follower still short of
        // its target kept shortening through isovolumic relaxation and into early diastole (7–14 % of s′ in every case,
        // 22 % with a regurgitant mitral valve, decision 245)
        const step = ((target - l) * dt) / tau;
        if (t < timings.ejectionEndS || step < 0) l += step;
        into[i] = l;
      }
    }
  };
  const peakRecoilCmps = (): number => {
    let peak = 0;
    for (let i = 1; i < n - 1; i++) {
      const t = (i + 0.5) * dt;
      // both neighbours of the central difference inside the window: at its edge it reads the next phase (decision 162)
      if (t - dt < timings.mitralOpenS || t + dt > earlyEnd) continue;
      peak = Math.max(
        peak,
        (-((longitudinal[i + 1] ?? 0) - (longitudinal[i - 1] ?? 0)) / (2 * dt)) *
          physiology.mapseCm,
      );
    }
    return peak;
  };
  // recoil speed falls monotonically as τ grows and rises with the time compression
  const ePrime = physiology.ePrimeSeptalCmps;
  let tauEarly = 0.03,
    earlySpeed = 1;
  simulate(0.03, 1);
  if (peakRecoilCmps() < ePrime) {
    let lo = 1,
      hi = 3;
    for (let it = 0; it < 20; it++) {
      const mid = (lo + hi) / 2;
      simulate(0.03, mid);
      if (peakRecoilCmps() < ePrime) lo = mid;
      else hi = mid;
    }
    earlySpeed = hi;
  } else {
    let lo = 0.03,
      hi = 1.5;
    for (let it = 0; it < 24; it++) {
      const mid = Math.sqrt(lo * hi);
      simulate(mid, 1);
      if (peakRecoilCmps() > ePrime) lo = mid;
      else hi = mid;
    }
    tauEarly = hi;
  }
  simulate(tauEarly, earlySpeed);
  // a′ (decision 162): atrial contraction moves the annulus back towards the atrium by what the atrial filling adds to
  // the ventricle, but over its volume course the tissue Doppler read 3.9 cm/s in the normal heart, where adults show
  // 8–10. The annulus is solved to move ahead of that filling, as it recoils ahead of the early one for e′, until it
  // reaches a′ = A·e′/E: the assumption that tissue and blood keep one ratio in both filling waves (7.6 cm/s here).
  if (timings.hasAWave && physiology.ePeakMps > 0) {
    const aPrimeTarget = (physiology.aPeakMps * physiology.ePrimeSeptalCmps) / physiology.ePeakMps;
    const peakAtrialCmps = (): number => {
      let peak = 0;
      for (let i = 1; i < n - 1; i++) {
        const t = (i + 0.5) * dt;
        if (t - dt < timings.aStartS || t + dt > timings.aEndS) continue;
        peak = Math.max(
          peak,
          (-((longitudinal[i + 1] ?? 0) - (longitudinal[i - 1] ?? 0)) / (2 * dt)) *
            physiology.mapseCm,
        );
      }
      return peak;
    };
    if (peakAtrialCmps() < aPrimeTarget) {
      let lo = 1,
        hi = 6;
      for (let it = 0; it < 20; it++) {
        atrialSpeed = (lo + hi) / 2;
        simulate(tauEarly, earlySpeed);
        if (peakAtrialCmps() < aPrimeTarget) lo = atrialSpeed;
        else hi = atrialSpeed;
      }
      atrialSpeed = hi;
      simulate(tauEarly, earlySpeed);
    }
  }
  // A beat that repeats is differentiated periodically. A chained beat starts where the previous one ended and ends elsewhere
  // (decision 114): before its first sample comes the displacement it started from, and its last sample has no next one.
  // Differentiated across that seam, the first sample read up to -625 MAPSE per second, a tissue Doppler spike at every QRS.
  const derivative = (from: Float32Array, to: Float32Array, start: number | undefined): void => {
    for (let i = 0; i < n; i++) {
      if (start !== undefined && i === 0) to[i] = ((from[1] ?? 0) - start) / (2 * dt);
      else if (start !== undefined && i === n - 1)
        to[i] = ((from[n - 1] ?? 0) - (from[n - 2] ?? 0)) / dt;
      else to[i] = ((from[(i + 1) % n] ?? 0) - (from[(i - 1 + n) % n] ?? 0)) / (2 * dt);
    }
  };
  derivative(longitudinal, longVel, af?.startLongitudinal);

  // Tricuspid annulus (decision 106): the right ventricle shortens along the same course, relaxes like the left one and
  // reaches its systolic peak velocity at the case S′ for its TAPSE, by compressing its ejection course in time. The tissue
  // Doppler of the right ventricle used to read the left ventricular curve scaled by MAPSE (5.3 cm/s at the free wall of
  // the normal heart for an S′ of 13), while the tricuspid annulus of the image moved with TAPSE.
  const rvLongitudinal = new Float32Array(n);
  const rvLongVel = new Float32Array(n);
  const peakSystolicCmps = (): number => {
    derivative(rvLongitudinal, rvLongVel, af?.startRvLongitudinal);
    let peak = 0;
    for (let i = 0; i < n; i++) {
      const t = (i + 0.5) * dt;
      if (t >= timings.ejectionStartS && t <= timings.ejectionEndS)
        peak = Math.max(peak, (rvLongVel[i] ?? 0) * physiology.tapseCm);
    }
    return peak;
  };
  {
    let lo = 0.5,
      hi = 3;
    for (let it = 0; it < 24; it++) {
      const mid = Math.sqrt(lo * hi);
      simulate(tauEarly, earlySpeed, mid, rvLongitudinal);
      if (peakSystolicCmps() < physiology.sPrimeTricuspidCmps) lo = mid;
      else hi = mid;
    }
    simulate(tauEarly, earlySpeed, Math.sqrt(lo * hi), rvLongitudinal);
    derivative(rvLongitudinal, rvLongVel, af?.startRvLongitudinal);
  }

  return {
    endSystoleS: shortenEndS,
    longitudinal,
    longitudinalVelocity: longVel,
    rvLongitudinal,
    rvLongitudinalVelocity: rvLongVel,
  };
}
