import type { CaseDefinitionInput } from './schema';
import { normalExcellentCase } from './normal-excellent';

/**
 * Case 13 — Chronic severe aortic regurgitation with aortic root dilation and compensated eccentric LV
 * volume overload: wide pulse pressure (152/48 mmHg), EROA 0.45 cm², steep PHT 185 ms, RVol > 60 mL,
 * RF > 50 %, holodiastolic flow reversal in the descending thoracic aorta, and preserved LVEF.
 */
export const aorticRegurgitationSevereCase: CaseDefinitionInput = {
  ...normalExcellentCase,
  id: 'aortic-regurgitation-severe',
  title: 'Insuficiencia aórtica crónica severa con dilatación del VI',
  seed: 1313,
  history:
    'Paciente sintético de 58 años con soplo diastólico decreciente paraesternal izquierdo, pulso amplio e intolerancia progresiva al esfuerzo. Sin datos reales de paciente.',
  demographics: { ageYears: 58, sexForReference: 'male', heightCm: 176, weightKg: 76 },
  bodyHabitus: {
    chestWallThicknessCm: 2.1,
    chestWidthCm: 33,
    chestDepthCm: 21,
    ribSpacingCm: 2.6,
    intercostalWidthCm: 1.7,
  },
  rhythm: { type: 'sinus', heartRateBpm: 72, rrVariabilityPct: 2, pvcProbability: 0 },
  anatomy: {
    ...normalExcellentCase.anatomy,
    lv: {
      eddCm: 6.3,
      lengthEdCm: 9.2,
      ivsdCm: 1.0,
      lvpwdCm: 1.0,
      sphericity: 0.62,
      apexWallThicknessCm: 0.72,
    },
    la: { apDiameterCm: 3.9, volumeMl: 62 },
    rv: { basalDiameterCm: 3.4, lengthCm: 7.4, freeWallThicknessCm: 0.4, septalFlattening: 0 },
    ra: { volumeMl: 45 },
    aorta: { lvotDiameterCm: 2.3, annulusCm: 2.6, sinusCm: 4.1, ascendingCm: 4.2 },
    aorticValve: {
      calcification: 0.12,
      cuspThicknessCm: 0.11,
      bicuspid: false,
    },
  },
  physiology: {
    ...normalExcellentCase.physiology,
    edvMl: 226,
    esvMl: 94,
    mapseCm: 1.5,
    tapseCm: 2.2,
    ePeakMps: 0.85,
    aPeakMps: 0.65,
    decelerationTimeMs: 175,
    ivrtMs: 70,
    ePrimeSeptalCmps: 9.5,
    ePrimeLateralCmps: 12.5,
  },
  hemodynamics: {
    systolicBpMmHg: 152,
    diastolicBpMmHg: 48,
    rapMmHg: 3,
    paspMmHg: 30,
    avEffectiveAreaCm2: 3.9,
    trPresent: true,
    lvotPeakGradientMmHg: 0,
    regurgitation: { ar: { eroaCm2: 0.47, phtMs: 185 } },
  },
  flowPrimitives: [
    ...normalExcellentCase.flowPrimitives,
    { id: 'ar', site: 'ar-jet', enabled: true, turbulence: 0.42 },
  ],
  learningObjectives: [
    'Identificar el chorro diastólico de insuficiencia aórtica en color en PLAX, A5C y A3C dirigido hacia el TSVI y la valva mitral anterior.',
    'Medir el tiempo de hemipresión (PHT < 200 ms) y la Vmax del chorro regurgitante con Doppler continuo alineado en A5C o A3C.',
    'Buscar la inversión holodiastólica del flujo en la aorta torácica descendente desde la ventana supraesternal como signo de severidad.',
    'Reconocer la dilatación excéntrica compensada del ventrículo izquierdo con fracción de eyección conservada y la dilatación de la raíz aórtica.',
  ],
  requiredViews: [
    { viewId: 'plax', minScore: 65 },
    { viewId: 'a5c', minScore: 55 },
    { viewId: 'a3c', minScore: 55 },
    { viewId: 'suprasternal-arch', minScore: 50 },
  ],
  requiredMeasurements: [
    { measurementId: 'lv-edd', tolerancePct: 10 },
    { measurementId: 'lv-edv-simpson', tolerancePct: 15 },
    { measurementId: 'ar-vmax', tolerancePct: 10 },
    { measurementId: 'ar-pht', tolerancePct: 15 },
  ],
  difficulty: 4,
  references: [
    {
      referenceId: 'ase-scmr-regurgitation-2017',
      usage: 'cuantificación de severidad de insuficiencia aórtica (ORE, PHT, inversión holodiastólica)',
    },
    {
      referenceId: 'ase-eacvi-chamber-2015',
      usage: 'remodelado excéntrico del VI y dimensiones de la raíz aórtica',
    },
  ],
  impressionTruth: [
    'Insuficiencia aórtica crónica severa (ORE ≈ 0,47 cm², PHT ≈ 185 ms, volumen regurgitante > 60 mL, fracción regurgitante > 50 %) con inversión holodiastólica en la aorta descendente.',
    'Ventrículo izquierdo dilatado por sobrecarga crónica de volumen con fracción de eyección conservada (FEVI ≈ 58 %).',
    'Dilatación de la raíz aórtica y aorta ascendente sin estenosis aórtica significativa.',
  ],
  expectedDeviations: [
    'lv-edv',
    'lv-esv',
    'lv-edvi',
    'lv-idd',
    'lv-ids',
    'ao-sinus',
    'ao-ascending',
  ],
};
