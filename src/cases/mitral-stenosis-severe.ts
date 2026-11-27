import type { CaseDefinitionInput } from './schema';
import { normalExcellentCase } from './normal-excellent';

/**
 * Case 14 — Rheumatic severe mitral stenosis: thickened, calcified, restricted mitral leaflets
 * (MVA 1.10 cm²), high-velocity turbulent diastolic transmitral inflow ("candle-flame" jet,
 * E ≈ 2.05 m/s, prolonged DT 690 ms → PHT 200 ms), severe LA dilation, and secondary post-capillary
 * pulmonary hypertension with preserved LV systolic function.
 */
export const mitralStenosisSevereCase: CaseDefinitionInput = {
  ...normalExcellentCase,
  id: 'mitral-stenosis-severe',
  title: 'Estenosis mitral reumática severa con dilatación de la AI',
  seed: 1414,
  history:
    'Paciente sintética de 48 años con disnea de esfuerzo progresiva, antecedente de fiebre reumática juvenil y retumbo diastólico apical. Sin datos reales de paciente.',
  demographics: { ageYears: 48, sexForReference: 'female', heightCm: 162, weightKg: 60 },
  bodyHabitus: {
    chestWallThicknessCm: 1.9,
    chestWidthCm: 30,
    chestDepthCm: 19.5,
    ribSpacingCm: 2.5,
    intercostalWidthCm: 1.6,
  },
  rhythm: { type: 'sinus', heartRateBpm: 72, rrVariabilityPct: 2, pvcProbability: 0 },
  anatomy: {
    ...normalExcellentCase.anatomy,
    lv: {
      eddCm: 4.5,
      lengthEdCm: 8.1,
      ivsdCm: 0.85,
      lvpwdCm: 0.85,
      sphericity: 0.5,
      apexWallThicknessCm: 0.65,
    },
    la: { apDiameterCm: 4.8, volumeMl: 98 },
    rv: { basalDiameterCm: 3.7, lengthCm: 7.4, freeWallThicknessCm: 0.45, septalFlattening: 0.15 },
    ra: { volumeMl: 54 },
    aorta: { lvotDiameterCm: 2.0, annulusCm: 2.2, sinusCm: 3.0, ascendingCm: 2.9 },
    mitral: {
      annulusDiameterCm: 2.9,
      anteriorLeafletLengthCm: 2.4,
      posteriorLeafletLengthCm: 1.3,
      maxOpeningDeg: 32,
      calcification: 0.7,
      thickeningCm: 0.24,
      samSeverity: 0,
      prolapse: 0,
    },
    tricuspid: { annulusDiameterCm: 3.4 },
    pulmonaryArtery: { trunkDiameterCm: 2.7 },
    ivc: { diameterCm: 1.9, collapsePct: 55 },
  },
  physiology: {
    ...normalExcellentCase.physiology,
    edvMl: 96,
    esvMl: 34,
    mapseCm: 1.3,
    tapseCm: 2.0,
    ePeakMps: 2.05,
    aPeakMps: 1.6,
    decelerationTimeMs: 690,
    ivrtMs: 55,
    ePrimeSeptalCmps: 9.0,
    ePrimeLateralCmps: 12.0,
    sPrimeTricuspidCmps: 11.5,
  },
  hemodynamics: {
    systolicBpMmHg: 114,
    diastolicBpMmHg: 72,
    rapMmHg: 5,
    paspMmHg: 52,
    avEffectiveAreaCm2: 2.8,
    mvEffectiveAreaCm2: 1.1,
    trPresent: true,
    lvotPeakGradientMmHg: 0,
    regurgitation: { tr: { eroaCm2: 0.15 } },
  },
  flowPrimitives: [
    { id: 'mv', site: 'mitral-inflow', enabled: true, turbulence: 0.48 },
    { id: 'lvot', site: 'lvot', enabled: true, turbulence: 0.04 },
    { id: 'av', site: 'aortic-valve', enabled: true, turbulence: 0.05 },
    { id: 'tv', site: 'tricuspid-inflow', enabled: true, turbulence: 0.04 },
    { id: 'rvot', site: 'rvot', enabled: true, turbulence: 0.04 },
    { id: 'tr', site: 'tr-jet', enabled: true, turbulence: 0.25 },
  ],
  artifacts: [
    { type: 'rib-shadow', intensity: 1, enabled: true },
    { type: 'lung-reverberation', intensity: 1, enabled: true },
    { type: 'calcium-shadow', intensity: 0.85, enabled: true },
    { type: 'near-field-clutter', intensity: 0.3, enabled: true },
  ],
  learningObjectives: [
    'Reconocer las valvas mitrales engrosadas, calcificadas y con apertura diastólica restringida en PLAX y PSAX nivel mitral.',
    'Identificar el chorro diastólico estrecho y turbulento en «llama de vela» en Doppler color con convergencia laminar proximal en la AI.',
    'Medir la onda E transmitral y el tiempo de hemipresión (PHT ≈ 200 ms) con Doppler continuo en A4C para estimar el AVM (220 / PHT ≈ 1,1 cm²).',
    'Evaluar las repercusiones retrógradas: dilatación severa de la aurícula izquierda e hipertensión pulmonar poscapilar secundaria por velocidad de IT.',
  ],
  requiredViews: [
    { viewId: 'plax', minScore: 65 },
    { viewId: 'psax-mv', minScore: 60 },
    { viewId: 'a4c', minScore: 65 },
  ],
  requiredMeasurements: [
    { measurementId: 'mitral-e', tolerancePct: 12 },
    { measurementId: 'mitral-pht', tolerancePct: 15 },
    { measurementId: 'la-volume', tolerancePct: 15 },
    { measurementId: 'tr-vmax', tolerancePct: 12 },
  ],
  difficulty: 4,
  references: [
    {
      referenceId: 'ase-eacvi-chamber-2015',
      usage: 'cuantificación de la dilatación auricular izquierda y dimensiones ventriculares',
    },
    {
      referenceId: 'ase-right-heart-2025',
      usage: 'evaluación de hipertensión pulmonar secundaria mediante velocidad de IT y PSVD',
    },
    {
      referenceId: 'ase-diastolic-2025',
      usage: 'exclusión clínica del algoritmo diastólico convencional en estenosis mitral significativa',
    },
  ],
  impressionTruth: [
    'Estenosis mitral reumática severa (AVM ≈ 1,10 cm², PHT ≈ 200 ms, velocidad transmitral E ≈ 2,0 m/s) con valvas engrosadas y apertura restringida.',
    'Aurícula izquierda severamente dilatada e hipertensión pulmonar poscapilar secundaria (IT ≈ 3,4 m/s, PSVD estimada ≈ 52 mmHg).',
    'Ventrículo izquierdo de dimensiones y función sistólica conservadas; el algoritmo convencional de función diastólica no aplica por la estenosis mitral.',
  ],
  expectedDeviations: ['la-ap', 'lavi', 'la-ao', 'pa-trunk'],
};
