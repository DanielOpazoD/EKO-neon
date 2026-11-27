import { modelDiastolicAssessment, modelPulmonaryAssessment } from './modelInterpretation';
import { rangeFlag, rangeText } from '@/clinical/guidelines/normalRanges';
import {
  efCategory,
  gradeAorticStenosis,
  gradeMitralRegurgitation,
} from '@/clinical/guidelines/valveGrading';
import { aorticStenosisInput } from './impression';
import type { Measurement } from '@/simulator/measurements/types';
import type { StructuredEchoTruth } from '@/simulator/hemodynamics/groundTruth';
import {
  AORTIC_STENOSIS_RULES,
  formatClinical,
  RIGHT_HEART_RULES,
} from '@/clinical/reference-values';
import { discProfileFromContour } from '@/simulator/measurements/simpson';
import {
  aggregateProtocolMeasurement,
  formatAggregateValue,
  multiBeatAverages,
  type MeasurementAggregate,
} from './multiBeat';
import { truthFor } from './scoring/scoring';
export { aggregateProtocolMeasurement, multiBeatAverages, type MeasurementAggregate };
import {
  fractionalAreaChange,
  fractionalShorteningPct,
  lvMassDevereuxG,
  mvaFromPht,
  pisaEroaCm2,
  relativeWallThickness,
  rvspFromTr,
  simpsonBiplaneVolume,
} from '@/clinical/formulas';

/**
 * Educational report assembly (spec 26). Deterministic; compares user measurements with the model
 * truth only when allowed (never in exam mode). Style differences are not penalised.
 */
export interface ReportRow {
  id: string;
  label: string;
  value: string;
  modality: string;
  view: string | null;
  viewScore: number | null;
  truth: string | null;
  deviation: string | null;
  /** Normal range for the patient's sex and where the value sits (decision 175); null without a range. */
  range: string | null;
  rangeFlag: 'low' | 'normal' | 'high' | null;
  /** technique grade (0..100) and its non-ok findings; null for free measurements */
  technique: { score: number; level: 'ok' | 'warn' | 'invalid'; notes: string[] } | null;
}

/** Calculations derived from the user's own protocol measurements (never from the truth). */
export interface DerivedRow {
  id: string;
  label: string;
  value: string;
  formula: string;
  inputs: string;
}

export interface EducationalReport {
  studyQuality: string;
  rows: ReportRow[];
  derived: DerivedRow[];
  impression: string[];
}

export function buildEducationalReport(
  measurements: Measurement[],
  truth: StructuredEchoTruth | null,
  hideTruth: boolean,
): EducationalReport {
  const sex = truth?.sex ?? null;
  const rows: ReportRow[] = measurements.map((m) => {
    let truthVal: string | null = null;
    let dev: string | null = null;
    if (truth && !hideTruth) {
      const ref = matchTruth(m, truth);
      if (ref !== null) {
        truthVal = ref.text;
        const d = ((m.value - ref.value) / ref.value) * 100;
        dev = `${d >= 0 ? '+' : ''}${d.toFixed(0)} %`;
      }
    }
    // until the exam is finished the report names no view and grades no technique: both say which view the image
    // shows, which is part of what the exam asks (decision 154)
    const tq =
      m.technique && !hideTruth
        ? {
            score: Math.round(m.technique.score * 100),
            level: m.technique.findings.some((f) => f.level === 'invalid')
              ? ('invalid' as const)
              : m.technique.findings.some((f) => f.level === 'warn')
                ? ('warn' as const)
                : ('ok' as const),
            notes: m.technique.findings.filter((f) => f.level !== 'ok').map((f) => f.message),
          }
        : null;
    return {
      id: m.id,
      label: m.label,
      value: `${m.value.toFixed(m.kind === 'time' || m.kind === 'volume' ? 0 : 2)} ${m.units}`,
      modality: m.modality,
      view: hideTruth ? null : m.sourceViewId,
      viewScore: hideTruth ? null : m.viewScore,
      truth: truthVal,
      deviation: dev,
      // the patient's sex is a demographic, not the truth: the range is shown in the exam too (decision 175)
      range: sex && m.measurementId ? rangeText(m.measurementId, sex, truth?.ageYears) : null,
      rangeFlag:
        sex && m.measurementId ? rangeFlag(m.measurementId, sex, m.value, truth?.ageYears) : null,
      technique: tq,
    };
  });
  const lowQuality = measurements.filter((m) =>
    m.technique
      ? m.technique.findings.some((f) => f.level === 'invalid')
      : (m.viewScore ?? 100) < 50,
  ).length;
  const studyQuality =
    measurements.length === 0
      ? 'Sin mediciones registradas.'
      : hideTruth
        ? `${measurements.length} ${measurements.length === 1 ? 'medición registrada' : 'mediciones registradas'}; la técnica se evalúa al finalizar el examen.`
        : lowQuality
          ? `${lowQuality} de ${measurements.length} mediciones tienen problemas de técnica que invalidan el valor (vista, fase, posición o alineación); su validez es limitada.`
          : 'Mediciones obtenidas con técnica adecuada.';
  const derived = deriveCalculations(measurements, truth?.bsaM2 ?? null);
  const impression: string[] = [];
  if (hideTruth)
    impression.push(
      'Redacta tu impresión diagnóstica a partir de tus propias mediciones; la comparación se entrega al finalizar el examen.',
    );
  else if (truth) {
    impression.push(
      `FEVI del modelo ${formatClinical(truth.lv.efPct, 'percent')}; ${efCategory(truth.lv.efPct, truth.sex) === 'normal' ? 'función sistólica conservada' : 'función sistólica reducida'}.`,
    );
    // integrative grading (ASE/EACVI 2017, valveGrading.ts): one criterion alone no longer decides a severe stenosis
    const as = gradeAorticStenosis(aorticStenosisInput(truth));
    impression.push(
      as.grade === 'none' || as.grade === 'mild'
        ? as.rationale
        : `${as.rationale.replace(/\.$/, '')} (${AORTIC_STENOSIS_RULES.severeVmax.referenceId}).`,
    );
    impression.push(...pathologyImpressions(truth));
  }
  return { studyQuality, rows, derived, impression };
}

/** Latest protocol measurement value by id. */
function latest(ms: Measurement[], id: string): Measurement | null {
  for (let i = ms.length - 1; i >= 0; i--) if (ms[i]!.measurementId === id) return ms[i]!;
  return null;
}

/**
 * The biplane method of discs (decision 180): the latest trace of a measurement in the four-chamber view and the latest
 * in the two-chamber view, each cut into 20 discs from its contour at the scale it was captured with, combined as
 * elliptical discs over the longer of the two long axes (ASE/EACVI 2015). Null without a trace in both views.
 */
export function biplaneVolume(
  ms: Measurement[],
  id: string,
): { volumeMl: number; a4cLongAxisCm: number; a2cLongAxisCm: number } | null {
  const trace = (view: string) => {
    for (let i = ms.length - 1; i >= 0; i--) {
      const m = ms[i]!;
      if (
        m.measurementId === id &&
        m.sourceViewId === view &&
        m.captureSector &&
        m.geometry.length >= 5
      )
        return discProfileFromContour(m.geometry, m.captureSector.pxPerCm);
    }
    return null;
  };
  const a4c = trace('a4c'),
    a2c = trace('a2c');
  if (!a4c || !a2c) return null;
  return {
    volumeMl: simpsonBiplaneVolume(
      a4c.diametersCm,
      a2c.diametersCm,
      Math.max(a4c.longAxisCm, a2c.longAxisCm),
    ),
    a4cLongAxisCm: a4c.longAxisCm,
    a2cLongAxisCm: a2c.longAxisCm,
  };
}

/** Upper limit of the normal LA volume index, both sexes (mL/m², ASE/EACVI 2015). */
const LA_VOLUME_INDEX_MAX = 34;

/**
 * Composite calculations from the user's measurements (spec 26.3): each row states its formula and inputs. The body
 * surface area is a demographic of the patient, like the sex, and indexes the atrial volume.
 */
export function deriveCalculations(ms: Measurement[], bsaM2: number | null = null): DerivedRow[] {
  const rows: DerivedRow[] = [];
  const agg = (id: string) => aggregateProtocolMeasurement(ms, id);
  const v = (id: string) => agg(id)?.value ?? null;
  const beatNote = (a: MeasurementAggregate | null) =>
    a && a.isAveraged ? ` (prom. ${a.beatCount} lat.)` : '';
  const edv = v('lv-edv-simpson'),
    esv = v('lv-esv-simpson');
  if (edv !== null && esv !== null && edv > 0) {
    rows.push({
      id: 'ef-simpson',
      label: 'FEVI (Simpson)',
      value: `${(((edv - esv) / edv) * 100).toFixed(0)} %`,
      formula: '(VTD − VTS) / VTD',
      inputs: `VTD ${edv.toFixed(0)} mL, VTS ${esv.toFixed(0)} mL`,
    });
    rows.push({
      id: 'sv-simpson',
      label: 'Volumen sistólico (Simpson)',
      value: `${(edv - esv).toFixed(0)} mL`,
      formula: 'VTD − VTS',
      inputs: `VTD ${edv.toFixed(0)} mL, VTS ${esv.toFixed(0)} mL`,
    });
  }
  // decision 180: the two apical planes combined, and the lengths they were traced with
  const lengths = (b: { a4cLongAxisCm: number; a2cLongAxisCm: number }, maxGapCm: number) =>
    `L A4C ${b.a4cLongAxisCm.toFixed(1)} cm, A2C ${b.a2cLongAxisCm.toFixed(1)} cm` +
    (Math.abs(b.a4cLongAxisCm - b.a2cLongAxisCm) > maxGapCm
      ? ' — longitudes dispares: revisa el acortamiento'
      : '');
  const lvGap = (b: { a4cLongAxisCm: number; a2cLongAxisCm: number }) =>
    0.1 * Math.max(b.a4cLongAxisCm, b.a2cLongAxisCm);
  const edvBi = biplaneVolume(ms, 'lv-edv-simpson'),
    esvBi = biplaneVolume(ms, 'lv-esv-simpson');
  for (const [id, label, b] of [
    ['lv-edv-biplane', 'VTD biplano', edvBi],
    ['lv-esv-biplane', 'VTS biplano', esvBi],
  ] as const)
    if (b)
      rows.push({
        id,
        label,
        value: `${b.volumeMl.toFixed(0)} mL`,
        formula: 'Σ π/4 · a·b · L/20 (discos elípticos A4C × A2C, L la mayor)',
        inputs: lengths(b, lvGap(b)),
      });
  if (edvBi && esvBi && edvBi.volumeMl > 0)
    rows.push({
      id: 'ef-biplane',
      label: 'FEVI (Simpson biplano)',
      value: `${(((edvBi.volumeMl - esvBi.volumeMl) / edvBi.volumeMl) * 100).toFixed(0)} %`,
      formula: '(VTD − VTS) / VTD, biplanos',
      inputs: `VTD ${edvBi.volumeMl.toFixed(0)} mL, VTS ${esvBi.volumeMl.toFixed(0)} mL`,
    });
  const laBi = biplaneVolume(ms, 'la-volume');
  if (laBi)
    rows.push({
      id: 'la-volume-biplane',
      label: 'Volumen de la AI (biplano)',
      value: `${laBi.volumeMl.toFixed(0)} mL`,
      formula: 'Σ π/4 · a·b · L/20 (discos elípticos A4C × A2C, L la mayor)',
      // the two atrial lengths should agree within 5 mm (ASE/EACVI 2015)
      inputs: lengths(laBi, 0.5),
    });
  const laVolume = laBi?.volumeMl ?? v('la-volume');
  if (laVolume !== null && bsaM2 !== null && bsaM2 > 0) {
    const index = laVolume / bsaM2;
    rows.push({
      id: 'la-volume-index',
      label: 'Índice de volumen de la AI',
      value: `${index.toFixed(0)} mL/m²${index > LA_VOLUME_INDEX_MAX ? ' (dilatada)' : ''}`,
      formula: `V / SC (normal ≤ ${LA_VOLUME_INDEX_MAX} mL/m²)`,
      inputs: `${laBi ? 'biplano' : 'monoplano'} ${laVolume.toFixed(0)} mL / ${bsaM2.toFixed(2)} m²`,
    });
  }
  const lvEddAgg = agg('lv-edd'),
    lvEsdAgg = agg('lv-esd'),
    ivsdAgg = agg('ivsd'),
    lvpwdAgg = agg('lvpwd');
  const lvEdd = lvEddAgg?.value ?? null,
    lvEsd = lvEsdAgg?.value ?? null,
    ivsd = ivsdAgg?.value ?? null,
    lvpwd = lvpwdAgg?.value ?? null;
  if (lvEdd !== null && lvEsd !== null && lvEdd > 0) {
    rows.push({
      id: 'lv-fs',
      label: 'Fracción de acortamiento (VI)',
      value: `${fractionalShorteningPct(lvEdd, lvEsd).toFixed(0)} %`,
      formula: '(DTDVI − DTSVI) / DTDVI',
      inputs: `DTDVI ${lvEdd.toFixed(2)} cm${beatNote(lvEddAgg)}, DTSVI ${lvEsd.toFixed(2)} cm${beatNote(lvEsdAgg)}`,
    });
  }
  if (lvEdd !== null && lvpwd !== null && lvEdd > 0 && lvpwd > 0) {
    const rwt = relativeWallThickness(lvEdd, lvpwd);
    rows.push({
      id: 'lv-rwt',
      label: 'Grosor parietal relativo (GPR)',
      value: `${rwt.toFixed(2)}${rwt > 0.42 ? ' (concéntrico)' : ''}`,
      formula: '2·PPd / DTDVI (normal ≤ 0,42)',
      inputs: `PPd ${lvpwd.toFixed(2)} cm${beatNote(lvpwdAgg)}, DTDVI ${lvEdd.toFixed(2)} cm${beatNote(lvEddAgg)}`,
    });
  }
  if (lvEdd !== null && ivsd !== null && lvpwd !== null && lvEdd > 0 && ivsd > 0 && lvpwd > 0) {
    const massG = lvMassDevereuxG(lvEdd, ivsd, lvpwd);
    rows.push({
      id: 'lv-mass',
      label: 'Masa del VI (Devereux)',
      value: `${massG.toFixed(0)} g`,
      formula: '0,8·1,04·[(DTD + SIV + PP)³ − DTD³] + 0,6',
      inputs: `DTD ${lvEdd.toFixed(2)}, SIV ${ivsd.toFixed(2)}, PP ${lvpwd.toFixed(2)} cm`,
    });
    if (bsaM2 !== null && bsaM2 > 0) {
      rows.push({
        id: 'lv-mass-index',
        label: 'Índice de masa del VI (IMVI)',
        value: `${(massG / bsaM2).toFixed(0)} g/m²`,
        formula: 'Masa VI / SC',
        inputs: `${massG.toFixed(0)} g / ${bsaM2.toFixed(2)} m²`,
      });
    }
  }
  const lvotDAgg = agg('lvot-diameter'),
    lvotVtiAgg = agg('lvot-vti');
  const lvotD = lvotDAgg?.value ?? null,
    lvotVti = lvotVtiAgg?.value ?? null;
  let lvotArea: number | null = null;
  if (lvotD !== null) {
    lvotArea = Math.PI * (lvotD / 2) * (lvotD / 2);
    rows.push({
      id: 'lvot-area',
      label: 'Área del TSVI',
      value: `${lvotArea.toFixed(2)} cm²`,
      formula: 'π·(D/2)²',
      inputs: `D ${lvotD.toFixed(2)} cm${beatNote(lvotDAgg)}`,
    });
  }
  if (lvotArea !== null && lvotVti !== null) {
    rows.push({
      id: 'sv-doppler',
      label: 'Volumen sistólico (Doppler)',
      value: `${(lvotArea * lvotVti).toFixed(0)} mL`,
      formula: 'Área TSVI × VTI TSVI',
      inputs: `${lvotArea.toFixed(2)} cm² × ${lvotVti.toFixed(1)} cm${beatNote(lvotVtiAgg)}`,
    });
    const avVtiAgg = agg('av-vti');
    const avVti = avVtiAgg?.value ?? null;
    if (avVti !== null && avVti > 0) {
      rows.push({
        id: 'ava-continuity',
        label: 'AVA por continuidad',
        value: `${((lvotArea * lvotVti) / avVti).toFixed(2)} cm²`,
        formula: 'Área TSVI × VTI TSVI / VTI VAo',
        inputs: `${lvotArea.toFixed(2)} cm² × ${lvotVti.toFixed(1)} cm${beatNote(lvotVtiAgg)} / ${avVti.toFixed(1)} cm${beatNote(avVtiAgg)}`,
      });
      rows.push({
        id: 'velocity-ratio',
        label: 'Índice de velocidades (VTI)',
        value: (lvotVti / avVti).toFixed(2),
        formula: 'VTI TSVI / VTI VAo',
        inputs: `${lvotVti.toFixed(1)}${beatNote(lvotVtiAgg)} / ${avVti.toFixed(1)}${beatNote(avVtiAgg)}`,
      });
    }
  }
  const avVmaxAgg = agg('av-vmax');
  const avVmax = avVmaxAgg?.value ?? null;
  if (avVmax !== null)
    rows.push({
      id: 'av-peak-gradient',
      label: 'Gradiente pico aórtico',
      value: `${(4 * avVmax * avVmax).toFixed(0)} mmHg`,
      formula: '4·Vmax²',
      inputs: `Vmax ${avVmax.toFixed(2)} m/s${beatNote(avVmaxAgg)}`,
    });
  const mrVmaxAgg = agg('mr-vmax');
  const mrVmax = mrVmaxAgg?.value ?? null;
  if (mrVmax !== null)
    rows.push({
      id: 'mr-peak-gradient',
      label: 'Gradiente pico IM (VI–AI)',
      value: `${(4 * mrVmax * mrVmax).toFixed(0)} mmHg`,
      formula: '4·Vmax²',
      inputs: `Vmax IM ${mrVmax.toFixed(2)} m/s${beatNote(mrVmaxAgg)}`,
    });
  const mrPisaAgg = agg('mr-pisa-radius');
  if (mrPisaAgg && mrPisaAgg.value > 0 && mrVmax !== null && mrVmax > 0) {
    const va = mrPisaAgg.latest.derived?.aliasingVelocityMps ?? 0.4;
    const eroa = pisaEroaCm2(mrPisaAgg.value, va, mrVmax);
    rows.push({
      id: 'mr-eroa-pisa',
      label: 'ORE de la IM (PISA)',
      value: `${eroa.toFixed(2)} cm²`,
      formula: '2π·r²·v_a / Vmax IM',
      inputs: `r ${mrPisaAgg.value.toFixed(2)} cm${beatNote(mrPisaAgg)}, v_a ${va.toFixed(2)} m/s, Vmax ${mrVmax.toFixed(2)} m/s${beatNote(mrVmaxAgg)}`,
    });
    const mrVtiAgg = agg('mr-vti');
    const mrVti = mrVtiAgg?.value ?? null;
    if (mrVti !== null && mrVti > 0) {
      rows.push({
        id: 'mr-rvol-pisa',
        label: 'Volumen regurgitante IM (PISA)',
        value: `${(eroa * mrVti).toFixed(0)} mL`,
        formula: 'ORE × VTI IM',
        inputs: `ORE ${eroa.toFixed(2)} cm² × VTI ${mrVti.toFixed(0)} cm${beatNote(mrVtiAgg)}`,
      });
    }
  }
  const mitralPhtAgg = agg('mitral-pht');
  const mitralPht = mitralPhtAgg?.value ?? null;
  if (mitralPht !== null && mitralPht > 0)
    rows.push({
      id: 'mva-pht',
      label: 'Área valvular mitral (PHT)',
      value: `${mvaFromPht(mitralPht).toFixed(2)} cm²`,
      formula: '220 / PHT',
      inputs: `PHT ${mitralPht.toFixed(0)} ms${beatNote(mitralPhtAgg)}`,
    });
  const pvSAgg = agg('pv-s'),
    pvDAgg = agg('pv-d');
  const pvS = pvSAgg?.value ?? null,
    pvD = pvDAgg?.value ?? null;
  if (pvS !== null && pvD !== null && pvD > 0)
    rows.push({
      id: 'pv-sd-ratio',
      label: 'Relación S/D venosa pulmonar',
      value: (pvS / pvD).toFixed(2),
      formula: 'S / D',
      inputs: `S ${pvS.toFixed(2)} m/s${beatNote(pvSAgg)}, D ${pvD.toFixed(2)} m/s${beatNote(pvDAgg)}`,
    });
  const eAgg = agg('mitral-e'),
    aAgg = agg('mitral-a');
  const e = eAgg?.value ?? null,
    a = aAgg?.value ?? null;
  if (e !== null && a !== null && a > 0)
    rows.push({
      id: 'e-a',
      label: 'Relación E/A',
      value: (e / a).toFixed(2),
      formula: 'E / A',
      inputs: `E ${e.toFixed(2)} m/s${beatNote(eAgg)}, A ${a.toFixed(2)} m/s${beatNote(aAgg)}`,
    });
  const esAgg = agg('e-prime-septal'),
    elAgg = agg('e-prime-lateral');
  const es = esAgg?.value ?? null,
    el = elAgg?.value ?? null;
  if (e !== null && (es !== null || el !== null)) {
    const ep = es !== null && el !== null ? (es + el) / 2 : (es ?? el)!;
    const epNote =
      esAgg?.isAveraged || elAgg?.isAveraged
        ? ` (prom. ${Math.max(esAgg?.beatCount ?? 0, elAgg?.beatCount ?? 0)} lat.)`
        : '';
    rows.push({
      id: 'e-eprime',
      label: `E/e′ ${es !== null && el !== null ? 'promedio' : es !== null ? 'septal' : 'lateral'}`,
      value: ((e * 100) / ep).toFixed(1),
      formula: 'E / e′',
      inputs: `E ${(e * 100).toFixed(0)} cm/s${beatNote(eAgg)}, e′ ${ep.toFixed(1)} cm/s${epNote}`,
    });
  }
  const trAgg = agg('tr-vmax');
  const tr = trAgg?.value ?? null;
  if (tr !== null) {
    const rap = rightAtrialPressure(v('ivc-diameter'), v('ivc-diameter-inspiration'));
    rows.push({
      id: 'rvsp',
      label: `PSVD estimada (PAD ${rap.rapMmHg} mmHg)`,
      value: `${rvspFromTr(tr, rap.rapMmHg).toFixed(0)} mmHg`,
      formula: '4·Vmax IT² + PAD',
      inputs: `Vmax IT ${tr.toFixed(2)} m/s${beatNote(trAgg)}, PAD ${rap.rapMmHg} mmHg (${rap.basis})`,
    });
  }
  const rvEdvM = latest(ms, 'rv-edv'),
    rvEsvM = latest(ms, 'rv-esv');
  if (rvEdvM && rvEsvM && rvEdvM.value > 0) {
    const rvSv = rvEdvM.value - rvEsvM.value;
    const rvEf = (rvSv / rvEdvM.value) * 100;
    rows.push({
      id: 'ef-rv',
      label: 'FEVD estimada (2D)',
      value: `${rvEf.toFixed(0)} %`,
      formula: '(VTD VD − VTS VD) / VTD VD',
      inputs: `VTD VD ${rvEdvM.value.toFixed(0)} mL, VTS VD ${rvEsvM.value.toFixed(0)} mL`,
    });
    rows.push({
      id: 'sv-rv',
      label: 'Volumen sistólico del VD (2D)',
      value: `${rvSv.toFixed(0)} mL`,
      formula: 'VTD VD − VTS VD',
      inputs: `VTD VD ${rvEdvM.value.toFixed(0)} mL, VTS VD ${rvEsvM.value.toFixed(0)} mL`,
    });
    const edA = rvEdvM.derived?.areaCm2,
      esA = rvEsvM.derived?.areaCm2;
    if (edA !== undefined && esA !== undefined && edA > 0) {
      const fac = fractionalAreaChange(edA, esA);
      rows.push({
        id: 'rv-fac',
        label: 'Cambio de área fraccional del VD (CAF)',
        value: `${fac.toFixed(0)} %`,
        formula: '(ATD − ATS) / ATD (normal ≥ 35 %)',
        inputs: `ATD ${edA.toFixed(1)} cm², ATS ${esA.toFixed(1)} cm²`,
      });
    }
  }
  // Surface explicit multi-beat ASE average rows whenever >= 2 distinct beats were measured for any protocol ID
  for (const mb of multiBeatAverages(ms)) {
    rows.push({
      id: `avg-${mb.measurementId}`,
      label: `${mb.label} (prom. ${mb.beatCount} lat.)`,
      value: `${formatAggregateValue(mb.kind, mb.units, mb.value)} ${mb.units}`,
      formula: `Promedio ASE (${mb.beatCount} latidos)`,
      inputs: `rango ${formatAggregateValue(mb.kind, mb.units, mb.min)}–${formatAggregateValue(mb.kind, mb.units, mb.max)} ${mb.units}`,
    });
  }
  return rows;
}

/**
 * Right atrial pressure from the inferior vena cava (decision 233), as the ASE estimates it: a cava of at most 2.1 cm that
 * collapses more than 50 % with a sniff gives 3 mmHg, a wider one that collapses less gives 15, anything else 8. The report
 * used to assume 3 whatever the cava: a pulmonary hypertension with a dilated, barely collapsing cava read 53 mmHg for 72.
 * Without the inspiratory diameter the collapse is unknown and the intermediate value applies.
 */
export function rightAtrialPressure(
  expiratoryCm: number | null,
  inspiratoryCm: number | null,
): { rapMmHg: number; basis: string } {
  const r = RIGHT_HEART_RULES.rapFromIvc.value;
  if (expiratoryCm === null || expiratoryCm <= 0)
    return { rapMmHg: r.intermediate.rapMmHg, basis: 'sin VCI medida: valor intermedio' };
  if (inspiratoryCm === null)
    return {
      rapMmHg: r.intermediate.rapMmHg,
      basis: `VCI ${expiratoryCm.toFixed(1)} cm sin colapso medido: valor intermedio`,
    };
  const collapsePct = (1 - inspiratoryCm / expiratoryCm) * 100;
  const small = expiratoryCm <= r.normal.ivcMaxCm;
  // more than half collapses and less than half does not: a cava that collapses exactly half is in neither branch
  const rapMmHg =
    small && collapsePct > r.normal.collapsePct
      ? r.normal.rapMmHg
      : !small && collapsePct < r.normal.collapsePct
        ? r.high.rapMmHg
        : r.intermediate.rapMmHg;
  // the diameter to the tenth of a millimetre the rule was decided on (2.14 cm is above 2.1), and a cava wider in
  // inspiration than in expiration does not collapse
  return {
    rapMmHg,
    basis: `VCI ${expiratoryCm.toFixed(2)} cm, colapso ${Math.max(0, collapsePct).toFixed(0)} %`,
  };
}

/** Educational impression lines for the modelled pathologies (thresholds: ASE/EACVI, see docs/REFERENCES.md). */
export function pathologyImpressions(t: StructuredEchoTruth): string[] {
  const out: string[] = [];
  if (t.lv.efPct < 41)
    out.push(
      `Disfunción sistólica ${t.lv.efPct < 30 ? 'severa' : 'moderada'} del VI (FEVI ${t.lv.efPct.toFixed(0)} %).`,
    );
  if (t.wallMotion.abnormalSegments.length) out.push(`${t.wallMotion.description}.`);
  const mr = t.regurgitation.mr;
  if (mr) {
    // graded on the EROA, regurgitant volume and fraction the line cites (valveGrading.ts, ASE 2017). It cites neither a
    // vena contracta nor a PISA radius: both are the EROA restated (as a width, and at a 40 cm/s aliasing velocity for
    // the jet's own peak), and the radius of a mild orifice (0.47–0.62 cm in the AF, artifact and HCM cases) sits in the
    // guideline's intermediate band, so the line read «leve (… radio PISA@40 0.62 cm)» against its own grade, as it had
    // with «VC 0.44 cm»
    const grade = { none: 'trivial', mild: 'leve', moderate: 'moderada', severe: 'severa' }[
      gradeMitralRegurgitation(mr)
    ];
    out.push(
      `Insuficiencia mitral ${grade} (ORE ${mr.eroaCm2.toFixed(2)} cm², volumen regurgitante ${mr.regurgitantVolumeMl.toFixed(0)} mL, fracción ${mr.regurgitantFractionPct.toFixed(0)} %)${Math.abs(mr.jetDirectionDeg) > 15 ? ', chorro excéntrico' : ''}.`,
    );
  }
  const ar = t.regurgitation.ar;
  if (ar) {
    const grade = ar.eroaCm2 >= 0.3 ? 'severa' : ar.eroaCm2 >= 0.1 ? 'moderada' : 'leve';
    out.push(
      `Insuficiencia aórtica ${grade} (ORE ${ar.eroaCm2.toFixed(2)} cm², PHT ${ar.phtMs.toFixed(0)} ms, volumen regurgitante ${ar.regurgitantVolumeMl.toFixed(0)} mL).`,
    );
  }
  if (t.mitral.effectiveAreaCm2 <= 1.5 && t.mitral.decelerationTimeMs >= 350) {
    const phtMs = t.mitral.decelerationTimeMs * 0.29;
    const grade = t.mitral.effectiveAreaCm2 <= 1.0 ? 'muy severa' : 'severa';
    out.push(
      `Estenosis mitral ${grade} (AVM ${t.mitral.effectiveAreaCm2.toFixed(2)} cm², PHT ${phtMs.toFixed(0)} ms, onda E ${t.mitral.ePeakMps.toFixed(2)} m/s).`,
    );
  }
  if (t.lvot.peakGradientMmHg >= 30)
    out.push(
      `Obstrucción ${t.lvot.dynamicObstruction ? 'dinámica' : 'fija'} del TSVI con gradiente pico ${t.lvot.peakGradientMmHg.toFixed(0)} mmHg${t.lvot.peakGradientMmHg >= 50 ? ' (significativa)' : ''}.`,
    );
  if (t.lv.ivsdCm >= 1.5 && t.lv.ivsdCm / Math.max(0.5, t.lv.lvpwdCm) >= 1.3)
    out.push(
      `Hipertrofia septal asimétrica (SIV ${t.lv.ivsdCm.toFixed(1)} cm, SIV/PP ${(t.lv.ivsdCm / t.lv.lvpwdCm).toFixed(1)}).`,
    );
  if (t.rightHeart.rvspMmHg !== null && t.rightHeart.rvspMmHg >= 36)
    out.push(`Presión sistólica del VD estimada ${t.rightHeart.rvspMmHg.toFixed(0)} mmHg.`);
  if (modelPulmonaryAssessment(t).suggestive)
    out.push(
      'Hallazgos ecográficos sugestivos de hipertensión pulmonar; la confirmación es hemodinámica invasiva.',
    );
  if (t.rv.basalDiameterCm > 4.1)
    out.push(
      `Ventrículo derecho dilatado (diámetro basal ${t.rv.basalDiameterCm.toFixed(1)} cm)${t.rv.septalFlattening > 0.3 ? ' con septo aplanado (sobrecarga de presión)' : ''}${t.rightHeart.tapseCm < 1.7 ? `; TAPSE reducido (${t.rightHeart.tapseCm.toFixed(1)} cm)` : ''}.`,
    );
  if (t.pericardium.effusionCm > 0) {
    const size =
      t.pericardium.effusionCm >= 2
        ? 'severo'
        : t.pericardium.effusionCm >= 1
          ? 'moderado'
          : 'leve';
    out.push(
      `Derrame pericárdico ${size} (${t.pericardium.effusionCm.toFixed(1)} cm)${t.pericardium.tamponade > 0.3 ? ' con signos de taponamiento (colapso del VD y de la AD, corazón oscilante)' : ''}.`,
    );
  }
  if (t.rhythm === 'atrial-fibrillation')
    out.push(
      'Fibrilación auricular: sin onda A; integrar E, E/e′ septal, IT/PASP y desaceleración, promediando ciclos representativos; el tamaño de la AI no basta para estimar presión.',
    );
  const diastolic = modelDiastolicAssessment(t);
  if (diastolic.dysfunctionPresent)
    out.push('Criterios del modelo compatibles con disfunción diastólica.');
  // a normal study stays silent: the pressure is named when it is abnormal or uncertain, or normal with dysfunction
  const fp = diastolic.fillingPressure;
  if (
    fp === 'elevated' ||
    fp === 'indeterminate' ||
    (fp === 'normal' && diastolic.dysfunctionPresent)
  ) {
    const label = { elevated: 'elevada', normal: 'normal', indeterminate: 'indeterminada' }[fp];
    const algorithm = t.rhythm === 'atrial-fibrillation' ? 'de FA' : 'de ritmo sinusal';
    out.push(
      `Presión de llenado izquierda ${label} según el algoritmo ${algorithm} (ASE 2025), sobre valores nominales del modelo.`,
    );
  }
  if (t.la.volumeIndexMlM2 > 34)
    out.push(`Aurícula izquierda dilatada (${t.la.volumeIndexMlM2.toFixed(0)} mL/m²).`);
  return out;
}

/**
 * The model's value a measurement is compared with: that of its protocol measurement, the one its id says it measured,
 * when the case has it (`truthFor`, the scorer's rule). Two comparisons the report made are gone. A truth of zero, the A
 * wave of a case in AF, divided the deviation by zero: «0.00 m/s · +Infinity %», or «NaN %» for 0.00 measured. And a free
 * measurement, which records no structure, was given the closest of a few true values, so it always looked right: a
 * 3.4 cm caliper across the RV of the normal case was compared with the LA diameter (3.5 cm) and read −3 %. The protocol
 * compares by id, never by the closest number.
 */
function matchTruth(
  m: Measurement,
  t: StructuredEchoTruth,
): { value: number; text: string } | null {
  const truth = m.measurementId ? truthFor(m.measurementId, t) : null;
  if (!truth) return null;
  const val = truth.value;
  const fmt =
    m.kind === 'velocity' && m.units === 'm/s'
      ? formatClinical(val, 'velocityMps')
      : m.kind === 'vti'
        ? formatClinical(val, 'vtiCm')
        : m.kind === 'linear'
          ? formatClinical(val, 'linearCm')
          : `${val.toFixed(m.kind === 'time' || m.kind === 'volume' ? 0 : 1)} ${m.units}`;
  return { value: val, text: fmt };
}
