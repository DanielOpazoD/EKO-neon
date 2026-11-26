import type { CaseDefinition, CaseDefinitionInput } from './schema';

/**
 * Zero-dependency normalizer for statically typed built-in CaseDefinitionInput objects.
 * Applies the exact same defaults and physiological invariants as `validateCase` in `./schema.ts`
 * without pulling Zod into the runtime entry or worker bundles. Full Zod schema verification
 * stays enforced in `verifyCore.test.ts` (including deep equality against `normalizeCase`).
 */
export function normalizeCase(input: CaseDefinitionInput): CaseDefinition {
  const reg = input.hemodynamics.regurgitation;
  const normalized: CaseDefinition = {
    schemaVersion: input.schemaVersion,
    id: input.id,
    title: input.title,
    seed: input.seed,
    history: input.history ?? '',
    demographics: { ...input.demographics },
    bodyHabitus: { ...input.bodyHabitus },
    rhythm: {
      type: input.rhythm.type,
      heartRateBpm: input.rhythm.heartRateBpm,
      rrVariabilityPct: input.rhythm.rrVariabilityPct ?? 1,
      pvcProbability: input.rhythm.pvcProbability ?? 0,
    },
    anatomy: {
      lv: {
        eddCm: input.anatomy.lv.eddCm,
        lengthEdCm: input.anatomy.lv.lengthEdCm,
        ivsdCm: input.anatomy.lv.ivsdCm,
        lvpwdCm: input.anatomy.lv.lvpwdCm,
        sphericity: input.anatomy.lv.sphericity ?? 0.55,
        apexWallThicknessCm: input.anatomy.lv.apexWallThicknessCm ?? 0.7,
        myocardialBackscatterDb: input.anatomy.lv.myocardialBackscatterDb ?? 0,
      },
      la: { ...input.anatomy.la },
      rv: {
        basalDiameterCm: input.anatomy.rv.basalDiameterCm,
        lengthCm: input.anatomy.rv.lengthCm,
        freeWallThicknessCm: input.anatomy.rv.freeWallThicknessCm,
        septalFlattening: input.anatomy.rv.septalFlattening ?? 0,
      },
      ra: { ...input.anatomy.ra },
      aorta: { ...input.anatomy.aorta },
      mitral: {
        annulusDiameterCm: input.anatomy.mitral.annulusDiameterCm,
        anteriorLeafletLengthCm: input.anatomy.mitral.anteriorLeafletLengthCm,
        posteriorLeafletLengthCm: input.anatomy.mitral.posteriorLeafletLengthCm,
        maxOpeningDeg: input.anatomy.mitral.maxOpeningDeg,
        calcification: input.anatomy.mitral.calcification ?? 0,
        thickeningCm: input.anatomy.mitral.thickeningCm ?? 0.1,
        samSeverity: input.anatomy.mitral.samSeverity ?? 0,
        prolapse: input.anatomy.mitral.prolapse ?? 0,
      },
      aorticValve: {
        calcification: input.anatomy.aorticValve.calcification,
        cuspThicknessCm: input.anatomy.aorticValve.cuspThicknessCm,
        bicuspid: input.anatomy.aorticValve.bicuspid ?? false,
      },
      tricuspid: { ...input.anatomy.tricuspid },
      pulmonaryArtery: {
        trunkDiameterCm: input.anatomy.pulmonaryArtery?.trunkDiameterCm ?? 2.3,
      },
      ivc: { ...input.anatomy.ivc },
      pericardium: {
        effusionCm: input.anatomy.pericardium.effusionCm ?? 0,
        tamponade: input.anatomy.pericardium.tamponade ?? 0,
      },
      heartPosition: {
        baseCm: { ...input.anatomy.heartPosition.baseCm },
        longAxis: { ...input.anatomy.heartPosition.longAxis },
        anterior: { ...input.anatomy.heartPosition.anterior },
      },
      wallMotion: (input.anatomy.wallMotion ?? []).map((w) => ({
        segment: w.segment,
        amplitude: w.amplitude,
        delayPhase: w.delayPhase ?? 0,
        ...(w.score !== undefined ? { score: w.score } : {}),
      })),
    },
    physiology: {
      edvMl: input.physiology.edvMl,
      esvMl: input.physiology.esvMl,
      mapseCm: input.physiology.mapseCm,
      tapseCm: input.physiology.tapseCm,
      ePeakMps: input.physiology.ePeakMps,
      aPeakMps: input.physiology.aPeakMps,
      decelerationTimeMs: input.physiology.decelerationTimeMs,
      ivrtMs: input.physiology.ivrtMs,
      ePrimeSeptalCmps: input.physiology.ePrimeSeptalCmps,
      ePrimeLateralCmps: input.physiology.ePrimeLateralCmps,
      sPrimeTricuspidCmps: input.physiology.sPrimeTricuspidCmps,
      contractility: input.physiology.contractility ?? 1,
    },
    hemodynamics: {
      systolicBpMmHg: input.hemodynamics.systolicBpMmHg,
      diastolicBpMmHg: input.hemodynamics.diastolicBpMmHg,
      rapMmHg: input.hemodynamics.rapMmHg,
      paspMmHg: input.hemodynamics.paspMmHg,
      avEffectiveAreaCm2: input.hemodynamics.avEffectiveAreaCm2,
      ...(input.hemodynamics.mvEffectiveAreaCm2 !== undefined
        ? { mvEffectiveAreaCm2: input.hemodynamics.mvEffectiveAreaCm2 }
        : {}),
      trPresent: input.hemodynamics.trPresent ?? true,
      lvotPeakGradientMmHg: input.hemodynamics.lvotPeakGradientMmHg ?? 0,
      regurgitation: {
        ...(reg?.mr
          ? {
              mr: {
                eroaCm2: reg.mr.eroaCm2,
                jetDirectionDeg: reg.mr.jetDirectionDeg,
                ...(reg.mr.timing !== undefined ? { timing: reg.mr.timing } : {}),
              },
            }
          : {}),
        ...(reg?.ar
          ? {
              ar: {
                eroaCm2: reg.ar.eroaCm2,
                phtMs: reg.ar.phtMs ?? 450,
              },
            }
          : {}),
        ...(reg?.tr ? { tr: { eroaCm2: reg.tr.eroaCm2 } } : {}),
      },
    },
    acousticWindow: {
      chestWallAttenuation: input.acousticWindow.chestWallAttenuation,
      lungOverlapCm: input.acousticWindow.lungOverlapCm,
      clutterLevel: input.acousticWindow.clutterLevel,
      obesityAttenuation: input.acousticWindow.obesityAttenuation,
      emphysemaScatter: input.acousticWindow.emphysemaScatter,
      cardiacRotationDeg: input.acousticWindow.cardiacRotationDeg ?? 0,
    },
    flowPrimitives: input.flowPrimitives.map((fp) => ({
      id: fp.id,
      site: fp.site,
      enabled: fp.enabled ?? true,
      turbulence: fp.turbulence ?? 0.05,
    })),
    artifacts: (input.artifacts ?? []).map((a) => ({
      type: a.type,
      intensity: a.intensity,
      enabled: a.enabled ?? true,
    })),
    learningObjectives: [...input.learningObjectives],
    requiredViews: input.requiredViews.map((v) => ({
      viewId: v.viewId,
      minScore: v.minScore ?? 60,
    })),
    requiredMeasurements: input.requiredMeasurements.map((m) => ({
      measurementId: m.measurementId,
      tolerancePct: m.tolerancePct ?? 15,
    })),
    references: input.references.map((r) => ({ ...r })),
    difficulty: input.difficulty,
    impressionTruth: [...input.impressionTruth],
    expectedDeviations: [...(input.expectedDeviations ?? [])],
  };

  const errors: string[] = [];
  if (normalized.physiology.esvMl >= normalized.physiology.edvMl)
    errors.push('physiology.esvMl must be < edvMl');
  if (normalized.hemodynamics.diastolicBpMmHg >= normalized.hemodynamics.systolicBpMmHg)
    errors.push('diastolic BP must be < systolic BP');
  if (normalized.rhythm.type === 'atrial-fibrillation' && normalized.physiology.aPeakMps > 0)
    errors.push(
      'atrial fibrillation cannot have an organized A wave (physiology.aPeakMps must be 0)',
    );
  if (normalized.anatomy.aorta.lvotDiameterCm > normalized.anatomy.aorta.annulusCm + 0.3)
    errors.push('LVOT diameter should not exceed the aortic annulus by more than 3 mm');
  for (const w of normalized.anatomy.wallMotion) {
    if (w.segment === 17 && w.score !== undefined)
      errors.push('anatomy.wallMotion: the apex cap (17) is not scored');
    if (w.segment <= 16 && w.amplitude !== 1 && w.score === undefined)
      errors.push(
        `anatomy.wallMotion: segment ${w.segment} changes its amplitude and needs a declared score`,
      );
  }
  if (
    new Set(normalized.anatomy.wallMotion.map((w) => w.segment)).size !==
    normalized.anatomy.wallMotion.length
  )
    errors.push('anatomy.wallMotion: a segment appears twice');
  if (errors.length) throw new Error(`Case ${input.id} invalid: ${errors.join('; ')}`);

  return normalized;
}
