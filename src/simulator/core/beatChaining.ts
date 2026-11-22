import type { CaseDefinition } from '@/cases/schema';
import type { ClockState } from '@/simulator/cardiac-cycle/clock';
import { buildBeatTables, type BeatTables } from '@/simulator/cardiac-cycle/cycleModel';
import { caseOutflow } from '@/simulator/cardiac-cycle/outflow';
import {
  inflowRespiratoryVariation,
  respiratoryDepth,
} from '@/simulator/cardiac-cycle/respiration';
import { ejectionTimeS, preEjectionPeriodS } from '@/simulator/cardiac-cycle/timing';
import type { PhaseMarks } from './protocol';

/**
 * Build chained beat tables for atrial fibrillation and/or free-breathing respiration
 * (decisions 107, 108, 113): each beat carries forward stroke volume and annular displacement
 * from the preceding beat and modulates IVC collapse and mitral/tricuspid inflow with breath depth.
 */
export function buildChainedBeatTables(
  caseDef: CaseDefinition,
  beat: ClockState,
  timeS: number,
  breathing: boolean,
  prev: BeatTables,
  nominal: BeatTables,
): BeatTables {
  const first = prev === nominal;
  const phys = caseDef.physiology;
  const svNominal = phys.edvMl - phys.esvMl;
  const clampSv = (ml: number): number =>
    Math.min(1.2 * svNominal, Math.max(0.2 * svNominal, ml));
  const ejectMl = first ? svNominal : clampSv(prev.endVolumeMl - phys.esvMl);

  let mitralEFactor = 1;
  let tricuspidEFactor = 1;
  let ivcCollapse: [number, number] | undefined;
  if (breathing) {
    const start = timeS - beat.timeInBeatS;
    const k = caseDef.anatomy.ivc.collapsePct / 100;
    ivcCollapse = [k * respiratoryDepth(start), k * respiratoryDepth(start + beat.rrS)];
    const opening =
      timeS -
      beat.timeInBeatS +
      preEjectionPeriodS(60 / beat.rrS) +
      ejectionTimeS(60 / beat.previousRrS, phys.contractility) +
      phys.ivrtMs / 1000;
    const depth = respiratoryDepth(opening);
    const variation = inflowRespiratoryVariation(caseDef.anatomy.pericardium.tamponade);
    mitralEFactor = 1 - variation.mitral * depth;
    tricuspidEFactor = 1 + variation.tricuspid * depth;
  }

  return buildBeatTables(beat.rrS, phys, caseDef.rhythm, caseDef.hemodynamics, {
    outflow: caseOutflow(caseDef),
    chain: {
      ejectMl,
      rvEjectMl: first
        ? nominal.pulmonaryStrokeVolumeMl
        : clampSv(prev.tricuspidFillMl - prev.regurgitation.trVolumeMl),
      mvAreaCm2: nominal.mvEffectiveAreaCm2,
      tvAreaCm2: nominal.tvEffectiveAreaCm2,
      previousRrS: beat.previousRrS,
      startLongitudinal: first ? 0 : prev.endLongitudinal,
      startRvLongitudinal: first ? 0 : prev.endRvLongitudinal,
      mitralEFactor,
      tricuspidEFactor,
      ivcCollapse,
    },
  });
}

/** Cardiac phase landmarks as fractions of RR (for phase checks in the technique engine). */
export function computePhaseMarks(tables: BeatTables): PhaseMarks {
  const t = tables.timings;
  const rr = tables.rrS;
  return {
    ejectionStart: t.ejectionStartS / rr,
    ejectionEnd: t.ejectionEndS / rr,
    endSystole: tables.endSystoleS / rr,
    mitralOpen: t.mitralOpenS / rr,
    eEnd: (t.mitralOpenS + t.eAccelS + t.eDecelS) / rr,
    aStart: t.hasAWave ? t.aStartS / rr : 1,
    aEnd: t.hasAWave ? t.aEndS / rr : 1,
    hasAWave: t.hasAWave,
  };
}
