import type { CaseDefinition } from '@/cases/schema';
import {
  computeWmsi,
  type SegmentAssessment,
  type WallMotionScore,
} from '@/clinical/segmentation/wallMotion';
import type { WallMotionId } from '@/clinical/segmentation/catalog';
import { buildBeatTables, cycleStateAt, type BeatTables } from '@/simulator/cardiac-cycle/cycleModel';
import { CardiacClock } from '@/simulator/cardiac-cycle/clock';
import { BREATH_PERIOD_S } from '@/simulator/cardiac-cycle/respiration';
import { buildChainedBeatTables } from '@/simulator/core/beatChaining';
import {
  classifyHeart,
  computeHeartPose,
  createHeartModel,
  type HeartModel,
  type HeartPose,
} from '@/simulator/anatomy/heartModel';
import { makeSample, Structure } from '@/simulator/anatomy/tissue';
import { caseLvotObstruction, caseOutflow } from '@/simulator/cardiac-cycle/outflow';
import { lvotNarrowing, pulmonaryVeinPeaks } from '@/simulator/doppler/flow-primitives/flowField';
import {
  bsaMosteller,
  cardiacOutput,
  continuityAva,
  ejectionFraction,
  meanGradientFromEnvelope,
  pisaRadiusCm,
  rvspFromTr,
  simplifiedBernoulli,
  velocityRatio,
  venaContractaFromEroaCm,
} from '@/clinical/formulas';

/**
 * StructuredEchoTruth: the ground truth that scoring/report compare against. Every value derives
 * from the same beat tables + anatomy that drive rendering and Doppler (spec 15, 32, 67).
 */
export interface StructuredEchoTruth {
  /** The sex the reference ranges are read for (decision 175). */
  sex: 'male' | 'female';
  /** Optional for older saved reports; current synthetic cases always provide it. */
  ageYears?: number;
  bmiKgM2?: number;
  bsaM2: number;
  heartRateBpm: number;
  rhythm: string;
  lv: {
    eddCm: number;
    esdCm: number;
    ivsdCm: number;
    lvpwdCm: number;
    edvMl: number;
    esvMl: number;
    efPct: number;
    strokeVolumeMl: number;
    cardiacOutputLpm: number;
    mapseCm: number;
  };
  lvot: {
    diameterCm: number;
    areaCm2: number;
    vtiCm: number;
    vmaxMps: number;
    strokeVolumeMl: number;
    peakGradientMmHg: number;
    dynamicObstruction: boolean;
  };
  regurgitation: {
    mr: {
      eroaCm2: number;
      regurgitantVolumeMl: number;
      regurgitantFractionPct: number;
      vmaxMps: number;
      vtiCm: number;
      jetDirectionDeg: number;
      /**
       * Expected vena contracta width (cm): the circular-equivalent width of the model orifice, 2·√(EROA/π), which its
       * jet neck is built from. A value to read a caliper against, never a grading criterion: it is the EROA again.
       */
      expectedVenaContractaCm?: number;
      /** Expected hemispheric PISA radius (cm) at standard aliasing velocity v_a = 0.40 m/s. */
      pisaRadius40Cm?: number;
    } | null;
    ar: {
      eroaCm2: number;
      regurgitantVolumeMl: number;
      regurgitantFractionPct: number;
      vmaxMps: number;
      phtMs: number;
    } | null;
    tr: {
      eroaCm2: number;
      /** Optional for compatibility with archived reports predating the right-heart ledger. */
      regurgitantVolumeMl?: number;
      vtiCm?: number;
      vmaxMps?: number;
    } | null;
  };
  wallMotion: {
    /** 16-segment ids the case scores 2 or worse (its declared scores, never the simulation amplitude). */
    abnormalSegments: number[];
    /** Declared score of segments 1–16 (index 0 = segment 1); a segment the case leaves alone is 1, or 2 under global hypokinesis. */
    scores: WallMotionScore[];
    /** Wall-motion score index of the synthetic ground truth, where every segment is scored. */
    wmsi: number;
    description: string;
  };
  pulmonaryVein: { sMps: number; dMps: number; arMps: number; sdRatio: number };
  rv: { basalDiameterCm: number; freeWallThicknessCm: number; septalFlattening: number };
  aorticValve: {
    effectiveAreaCm2: number;
    vmaxMps: number;
    vtiCm: number;
    peakGradientMmHg: number;
    meanGradientMmHg: number;
    continuityAvaCm2: number;
    velocityRatio: number;
  };
  mitral: {
    ePeakMps: number;
    aPeakMps: number;
    eOverA: number | null;
    decelerationTimeMs: number;
    ivrtMs: number;
    ePrimeSeptalCmps: number;
    ePrimeLateralCmps: number;
    eOverEPrimeAvg: number;
    effectiveAreaCm2: number;
  };
  rightHeart: {
    tapseCm: number;
    sPrimeCmps: number;
    trVmaxMps: number | null;
    rvspMmHg: number | null;
    rapMmHg: number;
    ivcCm: number;
    ivcCollapsePct: number;
    rvotAccelerationTimeMs: number;
  };
  la: { volumeMl: number; volumeIndexMlM2: number; apDiameterCm: number };
  aorta: { annulusCm: number; sinusCm: number; ascendingCm: number };
  pericardium: { effusionCm: number; tamponade: number };
}

export interface GroundTruthOptions {
  /** True when the patient is breathing freely (`respiration === 'free-breathing'`). */
  breathing?: boolean;
  /** Optional already-built heart model (avoids rebuilding it when the caller has one). */
  heart?: HeartModel;
}

/** PLAX anteroseptal–inferolateral caliper direction (azimuth 120° / −60°, perpendicular to the LV long axis). */
const PLAX_DIR_X = -0.5;
const PLAX_DIR_Y = Math.sqrt(3) / 2;
/** Protocol level of LVIDd (cm below the mitral annulus at end diastole, matching `measureModel.ts`). */
const PLAX_EDD_Z_CM = 2.0;

/**
 * Measure the LV internal diameter (cm) along the PLAX anteroseptal–inferolateral axis at height `z`
 * on the actual 3D heart SDF (`classifyHeart`), matching the inner-edge-to-inner-edge PLAX caliper
 * (including elliptical cross-section, annular neck, septal flattening and regional wall motion).
 */
function measurePlaxLvDiameter(heart: HeartModel, pose: HeartPose, z: number): number {
  const s = makeSample();
  const inside = (u: number): boolean =>
    classifyHeart(heart, pose, PLAX_DIR_X * u, PLAX_DIR_Y * u, z, s) &&
    (s.structure === Structure.LvCavity ||
      s.structure === Structure.PapillaryMuscle ||
      s.structure === Structure.Chordae);
  if (!inside(0)) return 0;
  const step = 0.02;
  const span = 6;
  let pos = 0;
  while (pos < span && inside(pos + step)) pos += step;
  let loP = pos;
  let hiP = Math.min(span, pos + step);
  for (let i = 0; i < 6; i++) {
    const mid = 0.5 * (loP + hiP);
    if (inside(mid)) loP = mid;
    else hiP = mid;
  }
  let neg = 0;
  while (neg > -span && inside(neg - step)) neg -= step;
  let hiN = neg;
  let loN = Math.max(-span, neg - step);
  for (let i = 0; i < 6; i++) {
    const mid = 0.5 * (loN + hiN);
    if (inside(mid)) hiN = mid;
    else loN = mid;
  }
  return loP - hiN;
}

/**
 * LV internal diameter (cm) of the geometric model at end diastole along the PLAX anteroseptal–inferolateral
 * axis, 2 cm below the mitral annulus (what the PLAX image and M-mode caliper show).
 */
export function geometricEdd(
  c: CaseDefinition,
  edvMl: number,
  tables?: BeatTables,
  heart?: HeartModel,
): number {
  const h = heart ?? createHeartModel(c.anatomy, c.physiology, undefined, c.seed);
  const t =
    tables ??
    buildBeatTables(60 / c.rhythm.heartRateBpm, c.physiology, c.rhythm, c.hemodynamics, {
      outflow: caseOutflow(c),
    });
  const edState = { ...cycleStateAt(t, 0), lvVolumeMl: edvMl };
  const edPose = computeHeartPose(h, edState);
  return measurePlaxLvDiameter(h, edPose, PLAX_EDD_Z_CM);
}

interface BeatMetrics {
  rrS: number;
  edvMl: number;
  esvMl: number;
  svTotalMl: number;
  forwardSvMl: number;
  lvotVtiCm: number;
  lvotPeakV: number;
  avVmaxMps: number;
  avVtiCm: number;
  avMeanGradientMmHg: number;
  mrVolumeMl: number;
  mrVmaxMps: number;
  mrVtiCm: number;
  arVolumeMl: number;
  arVmaxMps: number;
  arPhtMs: number;
  trVolumeMl: number;
  trVtiCm: number;
  trVmaxMps: number;
  ePeakMps: number;
  aPeakMps: number;
  rvotAtMs: number;
  dynamicObstruction: boolean;
}

function summarizeBeat(c: CaseDefinition, t: BeatTables, lvotArea: number): BeatMetrics {
  const dt = t.rrS / t.n;
  const avEnv: number[] = [];
  let lvotVmax = 0;
  let avVmax = 0;
  let lvotVti = 0;
  let avVti = 0;
  for (let i = 0; i < t.n; i++) {
    const q = t.aorticFlowMlps[i] ?? 0;
    if (q <= 0) continue;
    const vL = q / lvotArea / 100; // m/s
    const vA = q / c.hemodynamics.avEffectiveAreaCm2 / 100;
    avEnv.push(vA);
    lvotVmax = Math.max(lvotVmax, vL);
    avVmax = Math.max(avVmax, vA);
    lvotVti += vL * 100 * dt;
    avVti += vA * 100 * dt;
  }
  const obs = caseLvotObstruction(c, t);
  let lvotPeakV = lvotVmax;
  if (obs) {
    lvotPeakV = 0;
    const et = Math.max(1e-3, t.timings.ejectionEndS - t.timings.ejectionStartS);
    for (let i = 0; i < t.n; i++) {
      const q = t.aorticFlowMlps[i] ?? 0;
      if (q <= 0) continue;
      const u = (((i + 0.5) / t.n) * t.rrS - t.timings.ejectionStartS) / et;
      lvotPeakV = Math.max(
        lvotPeakV,
        q / Math.max(0.05, lvotArea * (1 - lvotNarrowing(obs.fMax, obs.dynamic, u))) / 100,
      );
    }
  }
  // End-diastolic volume is the pre-ejection maximum (start of beat plus any pre-ejection AR filling);
  // taking the max over the whole beat would add an upward bias whenever endVolumeMl > startVolumeMl
  // in chained AF or free-breathing beats.
  let edv = t.lvVolumeMl[0] ?? t.edvMl;
  let esv = Infinity;
  for (let i = 0; i < t.n; i++) {
    const v = t.lvVolumeMl[i] ?? edv;
    const time = (i + 0.5) * dt;
    if (time <= t.timings.ejectionStartS && v > edv) edv = v;
    if (v < esv) esv = v;
  }
  if (!Number.isFinite(esv)) esv = t.esvMl;
  const svTotal = Math.max(0, edv - esv);
  const forwardSv = lvotArea * lvotVti;
  const rg = t.regurgitation;
  // Early diastolic E-wave peak velocity from the beat's actual transmitral velocity table,
  // sampled at the E-wave peak time so A-wave fusion or dominant A waves (E < A) do not contaminate E.
  let ePeakMps = c.physiology.ePeakMps;
  if (t.timings.mitralOpenS >= t.rrS) {
    ePeakMps = 0;
  } else if (t.mvEffectiveAreaCm2 > 0) {
    const tPeak = Math.min(t.rrS - 0.5 * dt, t.timings.mitralOpenS + t.timings.eAccelS);
    const idx = Math.min(t.n - 1, Math.max(0, Math.floor(tPeak / dt)));
    ePeakMps = (t.mitralFlowMlps[idx] ?? 0) / t.mvEffectiveAreaCm2 / 100;
  }
  const aPeakMps = t.timings.hasAWave ? c.physiology.aPeakMps : 0;
  return {
    rrS: t.rrS,
    edvMl: edv,
    esvMl: esv,
    svTotalMl: svTotal,
    forwardSvMl: forwardSv,
    lvotVtiCm: lvotVti,
    lvotPeakV,
    avVmaxMps: avVmax,
    avVtiCm: avVti,
    avMeanGradientMmHg: meanGradientFromEnvelope(avEnv),
    mrVolumeMl: rg.mrVolumeMl,
    mrVmaxMps: rg.mrVmaxMps,
    mrVtiCm: rg.mrVtiCm,
    arVolumeMl: rg.arVolumeMl,
    arVmaxMps: rg.arVmaxMps,
    arPhtMs: rg.arPhtMs,
    trVolumeMl: rg.trVolumeMl,
    trVtiCm: rg.trVtiCm,
    trVmaxMps: rg.trVmaxMps,
    ePeakMps,
    aPeakMps,
    rvotAtMs: t.pulmonaryAccelerationS * 1000,
    dynamicObstruction: Boolean(obs?.dynamic),
  };
}

/**
 * Evaluate the stationary beat-averaged hemodynamic observables for chained beats (atrial fibrillation
 * and/or free breathing) using the exact same CardiacClock + buildChainedBeatTables pipeline as
 * SimulatorCore. For regular sinus rhythm at fixed respiration, returns the nominal beat directly.
 */
function ensembleMetrics(
  c: CaseDefinition,
  nominal: BeatTables,
  lvotArea: number,
  breathing: boolean,
): BeatMetrics {
  const isAf = c.rhythm.type === 'atrial-fibrillation';
  if (!isAf && !breathing) return summarizeBeat(c, nominal, lvotArea);

  const clock = new CardiacClock(c.rhythm, c.seed);
  let prev = nominal;
  let timeS = 0;
  const warmupBeats = 6;
  // Span enough beats for stable AF variance reduction and an integer number of full respiratory cycles.
  const minMeasuredBeats = isAf ? 80 : 24;
  const minMeasuredDurationS = breathing ? BREATH_PERIOD_S * (isAf ? 12 : 8) : 0;
  let count = 0;
  let measuredStartS = 0;
  let totalRr = 0;
  let edv = 0,
    esv = 0,
    svTotal = 0,
    forwardSv = 0,
    lvotVti = 0,
    lvotPeakV = 0,
    avVmax = 0,
    avVti = 0,
    avMeanGrad = 0,
    mrVol = 0,
    mrVmax = 0,
    mrVti = 0,
    arVol = 0,
    arVmax = 0,
    arPht = 0,
    trVol = 0,
    trVti = 0,
    trVmax = 0,
    ePeak = 0,
    aPeak = 0,
    rvotAt = 0;
  let dynamicObstruction = false;

  for (let bi = 0; bi < 400; bi++) {
    const b = clock.current;
    const t = buildChainedBeatTables(c, b, timeS, breathing, prev, nominal);
    prev = t;
    if (bi === warmupBeats) measuredStartS = timeS;
    if (bi >= warmupBeats) {
      const m = summarizeBeat(c, t, lvotArea);
      count++;
      totalRr += m.rrS;
      edv += m.edvMl;
      esv += m.esvMl;
      svTotal += m.svTotalMl;
      forwardSv += m.forwardSvMl;
      lvotVti += m.lvotVtiCm;
      lvotPeakV += m.lvotPeakV;
      avVmax += m.avVmaxMps;
      avVti += m.avVtiCm;
      avMeanGrad += m.avMeanGradientMmHg;
      mrVol += m.mrVolumeMl;
      mrVmax += m.mrVmaxMps;
      mrVti += m.mrVtiCm;
      arVol += m.arVolumeMl;
      arVmax += m.arVmaxMps;
      arPht += m.arPhtMs;
      trVol += m.trVolumeMl;
      trVti += m.trVtiCm;
      trVmax += m.trVmaxMps;
      ePeak += m.ePeakMps;
      aPeak += m.aPeakMps;
      rvotAt += m.rvotAtMs;
      if (m.dynamicObstruction) dynamicObstruction = true;
    }
    const stepRr = b.rrS;
    clock.advance(stepRr);
    timeS += stepRr;
    if (count >= minMeasuredBeats && timeS - measuredStartS >= minMeasuredDurationS) break;
  }

  const inv = 1 / Math.max(1, count);
  return {
    rrS: totalRr * inv,
    edvMl: edv * inv,
    esvMl: esv * inv,
    svTotalMl: svTotal * inv,
    forwardSvMl: forwardSv * inv,
    lvotVtiCm: lvotVti * inv,
    lvotPeakV: lvotPeakV * inv,
    avVmaxMps: avVmax * inv,
    avVtiCm: avVti * inv,
    avMeanGradientMmHg: avMeanGrad * inv,
    mrVolumeMl: mrVol * inv,
    mrVmaxMps: mrVmax * inv,
    mrVtiCm: mrVti * inv,
    arVolumeMl: arVol * inv,
    arVmaxMps: arVmax * inv,
    arPhtMs: arPht * inv,
    trVolumeMl: trVol * inv,
    trVtiCm: trVti * inv,
    trVmaxMps: trVmax * inv,
    ePeakMps: ePeak * inv,
    aPeakMps: aPeak * inv,
    rvotAtMs: rvotAt * inv,
    dynamicObstruction,
  };
}

export function computeGroundTruth(
  c: CaseDefinition,
  tables?: BeatTables,
  options: GroundTruthOptions = {},
): StructuredEchoTruth {
  const rr = 60 / c.rhythm.heartRateBpm;
  const t =
    tables ??
    buildBeatTables(rr, c.physiology, c.rhythm, c.hemodynamics, { outflow: caseOutflow(c) });
  const breathing = options.breathing === true;
  const bsa = bsaMosteller(c.demographics.heightCm, c.demographics.weightKg);
  const lvotArea = caseOutflow(c).lvotAreaCm2;
  const m = ensembleMetrics(c, t, lvotArea, breathing);
  const sv = m.svTotalMl; // total (EDV − ESV)
  const ef = ejectionFraction(m.edvMl, m.esvMl);
  const mrCfg = c.hemodynamics.regurgitation.mr;
  const arCfg = c.hemodynamics.regurgitation.ar;
  // the case's declared scores (decision 152): the amplitude is the mechanics, the score the reading it intends. A ventricle
  // with no regional abnormality and an EF under 40 % contracts uniformly weakly: every segment reads hypokinetic (global
  // hypokinesis, WMSI 2.0), not normal as the HFrEF case did at EF 24 %, and none is a regional abnormality
  const globalHypokinesis = c.anatomy.wallMotion.length === 0 && ef < 40;
  const scores = Array.from({ length: 16 }, (_, i): WallMotionScore => {
    const w = c.anatomy.wallMotion.find((x) => x.segment === i + 1);
    return w?.score ?? (globalHypokinesis ? 2 : 1);
  });
  const abnormal = globalHypokinesis ? [] : scores.flatMap((s, i) => (s >= 2 ? [i + 1] : []));
  // every segment is scored in the synthetic ground truth: the index of the tested consolidation, never a sum of our own
  const assessments = scores.map((s, i): SegmentAssessment => ({
    model: 'LV_16',
    segmentId: (i + 1) as WallMotionId,
    source: 'synthetic_ground_truth',
    score: s,
    views: [],
    scores: [s],
    discrepant: false,
  }));
  const wmsi = computeWmsi(assessments).value ?? 1;
  const pvPeaks = pulmonaryVeinPeaks(c);
  // EDD and ESD measured directly on the 3D heart SDF along the PLAX anteroseptal–inferolateral axis
  // (azimuth 120° / −60°, perpendicular to the LV long axis), matching the PLAX 2D / M-mode caliper.
  const heart = options.heart ?? createHeartModel(c.anatomy, c.physiology, undefined, c.seed);
  const edPose = computeHeartPose(heart, { ...cycleStateAt(t, 0), lvVolumeMl: m.edvMl });
  const esPose = computeHeartPose(heart, {
    ...cycleStateAt(t, t.endSystoleS / t.rrS),
    lvVolumeMl: m.esvMl,
  });
  const edd = measurePlaxLvDiameter(heart, edPose, PLAX_EDD_Z_CM);
  const zEs = esPose.zAnn + (PLAX_EDD_Z_CM / heart.lv.lengthCm) * esPose.lengthNow;
  const esd = measurePlaxLvDiameter(heart, esPose, zEs);
  const tr = c.hemodynamics.trPresent
    ? Math.sqrt(Math.max(0, (c.hemodynamics.paspMmHg - c.hemodynamics.rapMmHg) / 4))
    : null;
  const eOverA = t.timings.hasAWave && m.aPeakMps > 0 ? m.ePeakMps / m.aPeakMps : null;
  const ePrimeAvg = (c.physiology.ePrimeSeptalCmps + c.physiology.ePrimeLateralCmps) / 2;
  const effectiveHrBpm = 60 / Math.max(0.2, m.rrS);
  return {
    sex: c.demographics.sexForReference,
    ageYears: c.demographics.ageYears,
    bmiKgM2: c.demographics.weightKg / (c.demographics.heightCm / 100) ** 2,
    bsaM2: bsa,
    heartRateBpm: c.rhythm.heartRateBpm,
    rhythm: c.rhythm.type,
    lv: {
      eddCm: edd,
      esdCm: esd,
      ivsdCm: c.anatomy.lv.ivsdCm,
      lvpwdCm: c.anatomy.lv.lvpwdCm,
      edvMl: m.edvMl,
      esvMl: m.esvMl,
      efPct: ef,
      strokeVolumeMl: sv,
      // what reaches the aorta, not what leaves the ventricle: with mitral regurgitation the total stroke volume overstated
      // the output by the regurgitant volume and hid a forward flow that had collapsed (decision 297)
      cardiacOutputLpm: cardiacOutput(m.forwardSvMl, effectiveHrBpm),
      mapseCm: c.physiology.mapseCm,
    },
    lvot: {
      diameterCm: c.anatomy.aorta.lvotDiameterCm,
      areaCm2: lvotArea,
      vtiCm: m.lvotVtiCm,
      vmaxMps: m.lvotPeakV,
      strokeVolumeMl: lvotArea * m.lvotVtiCm,
      peakGradientMmHg: simplifiedBernoulli(m.lvotPeakV),
      dynamicObstruction: m.dynamicObstruction,
    },
    regurgitation: {
      mr:
        mrCfg && mrCfg.eroaCm2 > 0
          ? {
              eroaCm2: mrCfg.eroaCm2,
              regurgitantVolumeMl: m.mrVolumeMl,
              regurgitantFractionPct: (m.mrVolumeMl / Math.max(sv, 1)) * 100,
              vmaxMps: m.mrVmaxMps,
              vtiCm: m.mrVtiCm,
              jetDirectionDeg: mrCfg.jetDirectionDeg,
              expectedVenaContractaCm: venaContractaFromEroaCm(mrCfg.eroaCm2),
              pisaRadius40Cm:
                m.mrVmaxMps > 0 ? pisaRadiusCm(mrCfg.eroaCm2, m.mrVmaxMps, 0.4) : undefined,
            }
          : null,
      ar:
        arCfg && arCfg.eroaCm2 > 0
          ? {
              eroaCm2: arCfg.eroaCm2,
              regurgitantVolumeMl: m.arVolumeMl,
              regurgitantFractionPct: (m.arVolumeMl / Math.max(sv, 1)) * 100,
              vmaxMps: m.arVmaxMps,
              phtMs: m.arPhtMs,
            }
          : null,
      tr: c.hemodynamics.regurgitation.tr
        ? {
            eroaCm2: c.hemodynamics.regurgitation.tr.eroaCm2,
            regurgitantVolumeMl: m.trVolumeMl,
            vtiCm: m.trVtiCm,
            vmaxMps: m.trVmaxMps,
          }
        : null,
    },
    pulmonaryVein: { ...pvPeaks, sdRatio: pvPeaks.dMps > 0 ? pvPeaks.sMps / pvPeaks.dMps : 0 },
    wallMotion: {
      abnormalSegments: abnormal,
      scores,
      wmsi,
      description: globalHypokinesis
        ? `Hipocinesia global; índice de motilidad ${wmsi.toFixed(2).replace('.', ',')}`
        : abnormal.length
          ? `Alteración segmentaria en ${abnormal.length} segmento(s) del modelo de 16: ${abnormal.join(', ')}; índice de motilidad ${wmsi.toFixed(2).replace('.', ',')}`
          : 'Motilidad segmentaria normal',
    },
    rv: {
      basalDiameterCm: c.anatomy.rv.basalDiameterCm,
      freeWallThicknessCm: c.anatomy.rv.freeWallThicknessCm,
      septalFlattening: c.anatomy.rv.septalFlattening,
    },
    aorticValve: {
      effectiveAreaCm2: c.hemodynamics.avEffectiveAreaCm2,
      vmaxMps: m.avVmaxMps,
      vtiCm: m.avVtiCm,
      peakGradientMmHg: simplifiedBernoulli(m.avVmaxMps),
      meanGradientMmHg: m.avMeanGradientMmHg,
      continuityAvaCm2: continuityAva(c.anatomy.aorta.lvotDiameterCm, m.lvotVtiCm, m.avVtiCm),
      velocityRatio: velocityRatio(m.lvotVtiCm, m.avVtiCm),
    },
    mitral: {
      ePeakMps: m.ePeakMps,
      aPeakMps: m.aPeakMps,
      eOverA,
      decelerationTimeMs: c.physiology.decelerationTimeMs,
      ivrtMs: c.physiology.ivrtMs,
      ePrimeSeptalCmps: c.physiology.ePrimeSeptalCmps,
      ePrimeLateralCmps: c.physiology.ePrimeLateralCmps,
      eOverEPrimeAvg: (m.ePeakMps * 100) / ePrimeAvg,
      effectiveAreaCm2: t.mvEffectiveAreaCm2,
    },
    rightHeart: {
      tapseCm: c.physiology.tapseCm,
      sPrimeCmps: c.physiology.sPrimeTricuspidCmps,
      trVmaxMps: tr,
      rvspMmHg: tr !== null ? rvspFromTr(tr, c.hemodynamics.rapMmHg) : null,
      rapMmHg: c.hemodynamics.rapMmHg,
      ivcCm: c.anatomy.ivc.diameterCm,
      ivcCollapsePct: c.anatomy.ivc.collapsePct,
      rvotAccelerationTimeMs: m.rvotAtMs,
    },
    la: {
      volumeMl: c.anatomy.la.volumeMl,
      volumeIndexMlM2: c.anatomy.la.volumeMl / bsa,
      apDiameterCm: c.anatomy.la.apDiameterCm,
    },
    aorta: {
      annulusCm: c.anatomy.aorta.annulusCm,
      sinusCm: c.anatomy.aorta.sinusCm,
      ascendingCm: c.anatomy.aorta.ascendingCm,
    },
    pericardium: {
      effusionCm: c.anatomy.pericardium.effusionCm,
      tamponade: c.anatomy.pericardium.tamponade,
    },
  };
}
