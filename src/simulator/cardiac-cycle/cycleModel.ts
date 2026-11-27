import type { PhysiologyConfig, RhythmConfig, HemodynamicConfig } from '@/cases/schema';
import { solveAnnularMotion } from './annularMotion';
import {
  aorticPulse,
  meanPulmonaryPressureMmHg,
  pulmonaryAccelerationTimeS,
  SYSTOLIC_CLOSURE_S,
  type BeatOptions,
  type BeatTables,
} from './cycleState';
import { lvotNarrowing, solveLvotObstruction } from './outflow';
import { tricuspidRegurgitation } from './rightHeartFlow';
import {
  aorticStenosisSeverity,
  aWaveShape,
  computeCycleTimings,
  ejectionShape,
  eWaveShape,
} from './timing';

export {
  aorticPressureFraction,
  cycleStateAt,
  DIASTASIS_OPENING,
  meanPulmonaryPressureMmHg,
  pulmonaryAccelerationTimeS,
  sampleTable,
  valveEventTimes,
  type BeatOptions,
  type BeatTables,
  type ChainedBeat,
  type CycleState,
} from './cycleState';

/** Left atrial pressure (mmHg) under a regurgitant mitral valve: the 15 the model has always assumed, not a case value. */
const MR_LA_PRESSURE_MMHG = 15;
/** The least a beat ejects forward (mL), the floor the forward stroke volume always had. */
const MIN_FORWARD_ML = 5;
/**
 * Time constant (s) of isovolumic relaxation: a typical normal value of the monoexponential τ (some 30–50 ms), taken as an
 * assumption for every case (decision 245). It shapes the fall of the regurgitant jet's speed; where it ends, at mitral
 * opening, is the case's own.
 */
const LV_RELAXATION_TAU_S = 0.045;

export function buildBeatTables(
  rrS: number,
  physiology: PhysiologyConfig,
  rhythm: RhythmConfig,
  hemo: HemodynamicConfig,
  opts: BeatOptions = {},
): BeatTables {
  const n = opts.n ?? 512;
  const preload = opts.preloadFactor ?? 1;
  const af = opts.chain;
  const timings = computeCycleTimings(
    rrS,
    physiology,
    { ...rhythm, type: opts.aWave === false ? 'atrial-fibrillation' : rhythm.type },
    af?.previousRrS ?? rrS,
  );
  const svNominal = physiology.edvMl - physiology.esvMl;
  // in atrial fibrillation a beat ejects what the diastole before it filled, from the case's end-systolic volume
  const edv = af ? physiology.esvMl + af.ejectMl : physiology.edvMl * (0.85 + 0.15 * preload);
  const svTotal = af ? af.ejectMl : svNominal * preload; // EDV − ESV: everything that leaves the LV in systole (forward + regurgitant)
  const dt = rrS / n;

  // Regurgitant jets (spec 63): velocity from the simplified Bernoulli pressure difference, volume = ERO × VTI.
  // AR decays through diastole with the case's pressure half-time. MR runs from mitral closure to mitral opening, driven
  // by the left ventricle's pressure over the atrium's (decision 245): it used to follow the ejection ±20 ms at a speed
  // set by the arterial pressure alone, so it lasted 314–339 ms instead of the 400–475 ms from closure to opening, and in
  // the obstructive cardiomyopathy it read 5.24 m/s where the ventricle pushes against a 64 mmHg outflow gradient too.
  const mrFlow = new Float32Array(n);
  const mrVelocity = new Float32Array(n);
  const arFlow = new Float32Array(n);
  const mr = hemo.regurgitation.mr;
  const ar = hemo.regurgitation.ar;
  // Peak AR velocity occurs immediately after aortic valve closure (dicrotic-notch aortic pressure
  // minus early-diastolic LV pressure), then decays exponentially at the case's pressure half-time.
  const arDrivingMmHg =
    hemo.diastolicBpMmHg + 0.35 * (hemo.systolicBpMmHg - hemo.diastolicBpMmHg) - 12;
  const arVmax = ar && ar.eroaCm2 > 0 ? Math.sqrt(Math.max(1, arDrivingMmHg) / 4) : 0;
  const arPht = ar?.phtMs ?? 450;
  const et = timings.ejectionEndS - timings.ejectionStartS;
  // left ventricular pressure (mmHg) at time t of the beat, with an outflow gradient during ejection
  const lap = MR_LA_PRESSURE_MMHG;
  const dbp = hemo.diastolicBpMmHg,
    sbp = hemo.systolicBpMmHg;
  const ivr = Math.max(0.01, timings.mitralOpenS - timings.ejectionEndS);
  const lvPressure = (t: number, gradient: (t: number) => number): number => {
    if (t < SYSTOLIC_CLOSURE_S || t >= timings.mitralOpenS) return lap;
    if (t < timings.ejectionStartS) {
      // isovolumic contraction: from the atrium's pressure to the aorta's diastolic one
      const x =
        (t - SYSTOLIC_CLOSURE_S) / Math.max(1e-3, timings.ejectionStartS - SYSTOLIC_CLOSURE_S);
      return lap + (dbp - lap) * 0.5 * (1 - Math.cos(Math.PI * x));
    }
    if (t < timings.ejectionEndS) {
      const u = (t - timings.ejectionStartS) / et;
      return dbp + (sbp - dbp) * aorticPulse(u) + gradient(t);
    }
    // isovolumic relaxation: an exponential fall from the end-systolic pressure, reaching the atrium's at mitral opening
    const pEs = dbp + (sbp - dbp) * aorticPulse(1);
    const s = t - timings.ejectionEndS;
    const tail = Math.exp(-ivr / LV_RELAXATION_TAU_S);
    return lap + ((pEs - lap) * (Math.exp(-s / LV_RELAXATION_TAU_S) - tail)) / (1 - tail);
  };
  // Fraction of the case's orifice open at time t (decision 297). A central or prolapse jet leaks from mitral closure to
  // opening through the whole orifice. A jet made by systolic anterior motion leaks only while the leaflet is drawn into
  // the outflow tract, with the late-peaking narrowing of the dynamic obstruction: through all of systole it carried 30 mL
  // (RF 42 %) for an orifice of 0.15 cm² the case calls mild, and left a forward stroke volume index of 20 mL/m².
  // A functional jet (tethered leaflets) is widest early and late in systole and narrowest at the pressure peak, when the
  // ventricle's own force closes it best (Schwammenthal 1994; Hung 1999): about half the orifice at mid-systole.
  const samJet = mr?.timing === 'sam';
  const functionalJet = mr?.timing === 'functional';
  const mrSpanS = Math.max(1e-3, timings.mitralOpenS - SYSTOLIC_CLOSURE_S);
  const orificeOpen = (t: number): number => {
    if (functionalJet) {
      const x = Math.min(1, Math.max(0, (t - SYSTOLIC_CLOSURE_S) / mrSpanS));
      return 1 - 0.5 * Math.sin(Math.PI * x);
    }
    if (!samJet) return 1;
    if (t <= timings.ejectionStartS || t >= timings.ejectionEndS) return 0;
    return opts.outflow?.dynamicObstruction
      ? lvotNarrowing(1, true, (t - timings.ejectionStartS) / et)
      : 1;
  };
  let mrVmax = 0,
    mrVti = 0,
    mrEffVti = 0; // ∫ open fraction · v dt (cm): regurgitant volume = ERO × this
  const fillMr = (gradient: (t: number) => number): void => {
    mrVmax = 0;
    mrVti = 0;
    mrEffVti = 0;
    if (!mr || mr.eroaCm2 <= 0) return;
    for (let i = 0; i < n; i++) {
      const t = (i + 0.5) * dt;
      const f = orificeOpen(t);
      const v = f > 0 ? Math.sqrt(Math.max(0, lvPressure(t, gradient) - lap) / 4) : 0;
      mrFlow[i] = v * 100 * mr.eroaCm2 * f;
      mrVelocity[i] = v;
      mrVti += v * 100 * dt;
      mrEffVti += f * v * 100 * dt;
      mrVmax = Math.max(mrVmax, v);
    }
  };
  // first without the outflow gradient, which needs the forward flow the regurgitation leaves; then with it
  fillMr(() => 0);
  let arVti = 0;
  for (let i = 0; i < n; i++) {
    const t = (i + 0.5) * dt;
    if (arVmax > 0) {
      const tDia =
        t >= timings.ejectionEndS ? t - timings.ejectionEndS : t + rrS - timings.ejectionEndS; // time since AV closure
      const inDiastole = t >= timings.ejectionEndS || t < timings.ejectionStartS;
      const v = inDiastole ? arVmax * Math.pow(2, -tDia / ((2 * arPht) / 1000)) : 0;
      arFlow[i] = v * 100 * ar!.eroaCm2;
      arVti += v * 100 * dt;
    }
  }
  // a stenotic aortic valve rounds the ejection and delays its peak (decision 298)
  const stenosis = aorticStenosisSeverity(hemo.avEffectiveAreaCm2);
  let shapeInt = 0;
  for (let i = 0; i < 400; i++) shapeInt += ejectionShape((i + 0.5) / 400, stenosis) * (et / 400);
  if (mr && mr.eroaCm2 > 0) {
    const kFirst = Math.max(5, svTotal - mr.eroaCm2 * mrEffVti) / shapeInt;
    const aorticFirst = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const t = (i + 0.5) * dt;
      if (t > timings.ejectionStartS && t < timings.ejectionEndS)
        aorticFirst[i] = kFirst * ejectionShape((t - timings.ejectionStartS) / et, stenosis);
    }
    const obstruction = opts.outflow
      ? solveLvotObstruction(
          { n, rrS, timings, aorticFlowMlps: aorticFirst },
          opts.outflow.lvotAreaCm2,
          hemo.lvotPeakGradientMmHg,
          opts.outflow.dynamicObstruction,
        )
      : null;
    // the valvular and the subaortic gradients are in series and add
    fillMr((t) => {
      const q = aorticFirst[Math.min(n - 1, Math.floor(t / dt))] ?? 0;
      if (q <= 0) return 0;
      const vAv = q / hemo.avEffectiveAreaCm2 / 100;
      let g = 4 * vAv * vAv;
      if (obstruction && opts.outflow) {
        const u = (t - timings.ejectionStartS) / et;
        const area =
          opts.outflow.lvotAreaCm2 * (1 - lvotNarrowing(obstruction.fMax, obstruction.dynamic, u));
        const vL = q / Math.max(0.05, area) / 100;
        g += 4 * vL * vL;
      }
      return g;
    });
  }
  // a beat cannot regurgitate more than it ejects: a short beat of atrial fibrillation that filled little lost to the
  // regurgitation computed from pressure alone more than it held, and its volume no longer closed
  if (mrVmax > 0 && mr!.eroaCm2 * mrEffVti > svTotal - MIN_FORWARD_ML) {
    const k = Math.max(0, svTotal - MIN_FORWARD_ML) / (mr!.eroaCm2 * mrEffVti);
    for (let i = 0; i < n; i++) {
      mrFlow[i] = (mrFlow[i] ?? 0) * k;
      mrVelocity[i] = (mrVelocity[i] ?? 0) * k;
    }
    mrVti *= k;
    mrEffVti *= k;
    // a beat that holds little drives the jet with a lower pressure: its speed falls with its volume
    mrVmax *= k;
  }
  const rvolMr = mrVmax > 0 ? mr!.eroaCm2 * mrEffVti : 0;
  const rvolAr = arVmax > 0 ? ar!.eroaCm2 * arVti : 0;
  // forward (aortic) ejection = total − MR; mitral inflow = total − AR (the AR volume enters through the aorta)
  const sv = Math.max(MIN_FORWARD_ML, svTotal - rvolMr);
  const svMitral = Math.max(5, svTotal - rvolAr);

  // Ejection: Q_ao = k·shape(u), ∫ = SV
  const kAo = sv / shapeInt;

  // Filling: E and A shapes with peak velocities; solve mitral flow area A_mv so ∫Q_mv = SV. The velocity integral is
  // taken on this table, where the next beat cuts an E wave that has not ended (decision 95): integrating the whole wave
  // left the tamponade inflow 3.2 mL (6.4%) short of its stroke volume, and the closing correction hid it.
  const eCm = physiology.ePeakMps * 100 * Math.sqrt(preload);
  const aCm = timings.hasAWave ? physiology.aPeakMps * 100 : 0;
  const aDur = timings.hasAWave ? timings.aEndS - timings.aStartS : 0;
  const aShapeAt = (t: number): number =>
    timings.hasAWave && t > timings.aStartS && t < timings.aEndS
      ? aWaveShape((t - timings.aStartS) / aDur)
      : 0;
  // The end of atrial contraction closes the valve: an E wave still running then decays with the second half of the A
  // wave and nothing enters after it (decision 101). Cut only by the next beat, it kept entering through the last 10 ms of
  // the beat (0.33 m/s in tamponade) and the flow and the leaflets stopped at once with the R wave.
  const eAt = (t: number): number => {
    if (t <= timings.mitralOpenS) return 0;
    const e = eCm * eWaveShape(t - timings.mitralOpenS, timings.eAccelS, timings.eDecelS);
    if (!timings.hasAWave || t <= timings.aStartS + aDur / 2) return e;
    return e * aShapeAt(t);
  };
  // The case E and A are the peaks a Doppler trace shows, and the A wave is measured from the baseline over whatever E
  // flow is still running (decision 97). Atrial contraction adds the increment that brings the inflow up to A: added in
  // full on top of an unfinished E wave, the peak read 0.99 m/s for an A of 0.7 (pulmonary hypertension), 1.04 for 0.85
  // (artifact case) and 1.09 in tamponade, where the waves fuse at 108 bpm and no separate E of 0.75 could be measured.
  // When the E flow alone already exceeds A during atrial contraction, the fused wave is that of E.
  const aWindowPeak = (scale: number): number => {
    let peak = 0;
    for (let i = 0; i < n; i++) {
      const t = (i + 0.5) * dt;
      if (aShapeAt(t) > 0) peak = Math.max(peak, eAt(t) + scale * aCm * aShapeAt(t));
    }
    return peak;
  };
  let aScale = 1;
  if (aCm > 0 && aWindowPeak(1) > aCm * 1.001) {
    let lo = 0,
      hi = 1;
    if (aWindowPeak(0) >= aCm) hi = 0;
    else
      for (let it = 0; it < 30; it++) {
        const mid = 0.5 * (lo + hi);
        if (aWindowPeak(mid) > aCm) hi = mid;
        else lo = mid;
      }
    aScale = hi;
  }
  // respiration scales the early waves of each inflow and leaves atrial contraction as the case measured it (decision 108)
  const mitralEFactor = af?.mitralEFactor ?? 1;
  const tricuspidEFactor = af?.tricuspidEFactor ?? 1;
  const tr = tricuspidRegurgitation(hemo, timings, rrS, n);
  const mvVelocity = new Float32Array(n); // cm/s
  const tvVelocity = new Float32Array(n); // cm/s, with its own inflow area
  let velIntegralCm = 0; // cm (VTI of mitral inflow)
  let tvIntegralCm = 0;
  // When a forced mitral orifice area elevates transmitral E (mitral stenosis), keep tricuspid
  // inflow velocity physiological (~0.65 m/s) rather than copying the stenotic mitral velocity.
  const tvVelocityScale = hemo.mvEffectiveAreaCm2
    ? Math.min(1, 0.65 / Math.max(0.1, physiology.ePeakMps))
    : 1;
  // the tricuspid valve opens TRICUSPID_LEAD_S before the mitral one (decision 162): its early wave runs that much earlier
  const tvLead = timings.mitralOpenS - timings.tricuspidOpenS;
  for (let i = 0; i < n; i++) {
    const t = (i + 0.5) * dt;
    const atrial = aScale * aCm * aShapeAt(t);
    const v = eAt(t) * mitralEFactor + atrial;
    mvVelocity[i] = v;
    tvVelocity[i] = (eAt(t + tvLead) * tricuspidEFactor + atrial) * tvVelocityScale;
    velIntegralCm += v * dt;
    tvIntegralCm += tvVelocity[i]! * dt;
  }
  // a beat of atrial fibrillation fills through the case's orifice for as long as its diastole lasts (decision 107)
  const mvArea = af
    ? af.mvAreaCm2
    : (hemo.mvEffectiveAreaCm2 ?? svMitral / Math.max(velIntegralCm, 1e-6));
  const scaleMv =
    !af && hemo.mvEffectiveAreaCm2 ? svMitral / Math.max(mvArea * velIntegralCm, 1e-6) : 1; // enforce ∫=SV if area forced

  // Right ventricular ejection (decision 105): the forward stroke volume with a skewed shape peaking at the acceleration
  // time of the case's mean pulmonary pressure (it used to copy the aortic flow, whose peak at 0.4 of ejection gave 121 ms
  // in the normal heart and 108 ms at 72 mmHg), over the right ventricle's own ejection period (decision 162): from the
  // pulmonary opening, before the aortic one, to the pulmonary closure, after it.
  const etRv = timings.pulmonaryCloseS - timings.pulmonaryOpenS;
  const pulmonaryAccelerationS = Math.min(
    0.6 * etRv,
    Math.max(0.12 * etRv, pulmonaryAccelerationTimeS(meanPulmonaryPressureMmHg(hemo.paspMmHg))),
  );
  const peakU = pulmonaryAccelerationS / etRv;
  const pvShape = (u: number): number =>
    u <= 0 || u >= 1 ? 0 : Math.pow(u, 2.25 * peakU) * Math.pow(1 - u, 2.25 * (1 - peakU));
  let pvInt = 0;
  for (let i = 0; i < 400; i++) pvInt += pvShape((i + 0.5) / 400) * (etRv / 400);
  const forwardMl = Math.max(5, af?.rvEjectMl ?? sv - rvolAr);
  const kPv = forwardMl / pvInt;
  // Nominal periodic balance: TV inflow = pulmonary forward volume + TR volume.
  // Chained beats reuse their own nominal orifice so respiratory/AF filling is not normalized away.
  const tvArea = af?.tvAreaCm2 ?? (forwardMl + tr.volumeMl) / Math.max(tvIntegralCm, 1e-6);

  const aorticFlow = new Float32Array(n);
  const pulmonaryFlow = new Float32Array(n);
  const mitralFlow = new Float32Array(n);
  const tricuspidFlow = new Float32Array(n);
  let tricuspidFillMl = 0;
  for (let i = 0; i < n; i++) {
    const t = (i + 0.5) * dt;
    if (t > timings.ejectionStartS && t < timings.ejectionEndS) {
      aorticFlow[i] = kAo * ejectionShape((t - timings.ejectionStartS) / et, stenosis);
    }
    const tp = (((t - timings.pulmonaryOpenS) % rrS) + rrS) % rrS;
    if (tp < etRv) pulmonaryFlow[i] = kPv * pvShape(tp / etRv);
    mitralFlow[i] = (mvVelocity[i] ?? 0) * mvArea * scaleMv;
    tricuspidFlow[i] = (tvVelocity[i] ?? 0) * tvArea;
    tricuspidFillMl += tricuspidFlow[i]! * dt;
  }
  // Normalize pulmonary discretization to the requested forward volume; no hidden RV closing correction.
  const pvSampledMl = pulmonaryFlow.reduce((sum, q) => sum + q * dt, 0);
  const rvVolumeChangeMl = new Float32Array(n);
  let pulmonaryStrokeVolumeMl = 0,
    rvEndVolumeChangeMl = 0;
  for (let i = 0; i < n; i++) {
    pulmonaryFlow[i] = (pulmonaryFlow[i]! * forwardMl) / Math.max(pvSampledMl, 1e-6);
    pulmonaryStrokeVolumeMl += pulmonaryFlow[i]! * dt;
    rvEndVolumeChangeMl += (tricuspidFlow[i]! - pulmonaryFlow[i]! - tr.flowMlps[i]!) * dt;
    rvVolumeChangeMl[i] = rvEndVolumeChangeMl;
  }
  // Integrate volume; then remove the residual drift so V(0)=V(RR)=EDV exactly (ensures periodicity).
  const vol = new Float32Array(n);
  let v = edv;
  for (let i = 0; i < n; i++) {
    v += ((mitralFlow[i] ?? 0) + (arFlow[i] ?? 0) - (aorticFlow[i] ?? 0) - (mrFlow[i] ?? 0)) * dt;
    vol[i] = v;
  }
  const drift = v - edv;
  // a beat of atrial fibrillation does not close on itself: what it filled beyond what it ejected starts the next one
  if (!af) for (let i = 0; i < n; i++) vol[i] = (vol[i] ?? 0) - (drift * (i + 1)) / n;
  let minV = Infinity,
    maxV = -Infinity;
  for (let i = 0; i < n; i++) {
    minV = Math.min(minV, vol[i] ?? 0);
    maxV = Math.max(maxV, vol[i] ?? 0); // with AR the LV keeps filling until the aortic valve opens
  }
  // contraction, wall thickening and annular motion of fibrillating beats share the case's volumes, so they run on
  // across beats that eject and fill different volumes
  const edvRef = af ? physiology.edvMl : edv;
  const esvRef = af ? physiology.esvMl : minV;

  const annular = solveAnnularMotion({
    n,
    dt,
    rrS,
    timings,
    physiology,
    vol,
    mrFlow,
    edvRef,
    esvRef,
    minV,
    af,
  });

  return {
    n,
    timings,
    rrS,
    edvMl: af ? edvRef : maxV,
    esvMl: af ? esvRef : minV,
    strokeVolumeMl: maxV - minV,
    lvVolumeMl: vol,
    volumeCorrectionMl: af ? 0 : drift,
    endVolumeMl: v,
    endSystoleS: annular.endSystoleS,
    endLongitudinal: annular.longitudinal[n - 1] ?? 0,
    endRvLongitudinal: annular.rvLongitudinal[n - 1] ?? 0,
    aorticFlowMlps: aorticFlow,
    pulmonaryFlowMlps: pulmonaryFlow,
    pulmonaryAccelerationS,
    mitralFlowMlps: mitralFlow,
    tricuspidFlowMlps: tricuspidFlow,
    tricuspidFillMl,
    tvEffectiveAreaCm2: tvArea,
    pulmonaryStrokeVolumeMl,
    rvVolumeChangeMl,
    rvEndVolumeChangeMl,
    trVelocityMps: tr.velocityMps,
    trFlowMlps: tr.flowMlps,
    // Preserve forced effective mitral orifice area (e.g. mitral stenosis); mass conservation
    // ∫Q_mv dt = svMitral is already carried by scaleMv inside mitralFlowMlps.
    mvEffectiveAreaCm2: mvArea,
    mrFlowMlps: mrFlow,
    mrVelocityMps: mrVelocity,
    arFlowMlps: arFlow,
    regurgitation: {
      mrVolumeMl: rvolMr,
      mrVmaxMps: mrVmax,
      mrVtiCm: mrVti,
      arVolumeMl: rvolAr,
      arVmaxMps: arVmax,
      arVtiCm: arVti,
      arPhtMs: arPht,
      trVolumeMl: tr.volumeMl,
      trVtiCm: tr.vtiCm,
      trVmaxMps: tr.vmaxMps,
    },
    longitudinal: annular.longitudinal,
    longitudinalVelocity: annular.longitudinalVelocity,
    rvLongitudinal: annular.rvLongitudinal,
    rvLongitudinalVelocity: annular.rvLongitudinalVelocity,
    ivcCollapse: af?.ivcCollapse ?? null,
  };
}
