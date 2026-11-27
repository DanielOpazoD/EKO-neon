import type { Measurement } from '@/simulator/measurements/types';

/**
 * Multi-beat ASE averaging (particularly essential in atrial fibrillation with variable R–R).
 * Biplane/chamber volume IDs are excluded because A4C and A2C captures represent orthogonal planes
 * combined by `biplaneVolume()`, not repeated single-plane beats.
 */
export const NON_BEAT_AVERAGED_IDS = new Set([
  'lv-edv-simpson',
  'lv-esv-simpson',
  'la-volume',
  'rv-edv',
  'rv-esv',
]);

/** ASE/EACVI guideline cap: average the most recent 3–5 representative consecutive beats. */
export const MAX_ASE_AVG_BEATS = 5;

export interface MeasurementAggregate {
  measurementId: string;
  label: string;
  units: string;
  kind: Measurement['kind'];
  value: number;
  beatCount: number;
  isAveraged: boolean;
  min: number;
  max: number;
  latest: Measurement;
  beats: Measurement[];
}

export function formatAggregateValue(
  kind: Measurement['kind'],
  units: string,
  value: number,
): string {
  const decimals =
    kind === 'time' || kind === 'volume' ? 0 : kind === 'vti' || units === 'cm/s' ? 1 : 2;
  return value.toFixed(decimals);
}

/**
 * Resolve the representative value for a protocol measurement ID.
 * - Keeps latest capture per distinct `beatIndex` (so re-measuring the same beat replaces it).
 * - Prefers technically valid captures whenever at least one exists.
 * - Averages up to the last 5 distinct beats (`isAveraged = beatCount >= 2`).
 * - Falls back to latest-wins when `beatIndex` is absent or for biplane volume IDs.
 */
export function aggregateProtocolMeasurement(
  ms: Measurement[],
  id: string,
): MeasurementAggregate | null {
  const all = ms.filter((m) => m.measurementId === id);
  if (all.length === 0) return null;
  const lastAll = all[all.length - 1]!;
  if (NON_BEAT_AVERAGED_IDS.has(id)) {
    return {
      measurementId: id,
      label: lastAll.label,
      units: lastAll.units,
      kind: lastAll.kind,
      value: lastAll.value,
      beatCount: 1,
      isAveraged: false,
      min: lastAll.value,
      max: lastAll.value,
      latest: lastAll,
      beats: [lastAll],
    };
  }
  const valid = all.filter((m) => !m.technique?.findings.some((f) => f.level === 'invalid'));
  const pool = valid.length > 0 ? valid : all;
  const byBeat = new Map<number | 'single', Measurement>();
  for (const m of pool) {
    const key =
      typeof m.beatIndex === 'number' && Number.isFinite(m.beatIndex) ? m.beatIndex : 'single';
    if (byBeat.has(key)) byBeat.delete(key);
    byBeat.set(key, m);
  }
  const beats = Array.from(byBeat.values()).slice(-MAX_ASE_AVG_BEATS);
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const b of beats) {
    sum += b.value;
    if (b.value < min) min = b.value;
    if (b.value > max) max = b.value;
  }
  const newest = beats[beats.length - 1]!;
  return {
    measurementId: id,
    label: newest.label,
    units: newest.units,
    kind: newest.kind,
    value: sum / beats.length,
    beatCount: beats.length,
    isAveraged: beats.length >= 2,
    min,
    max,
    latest: newest,
    beats,
  };
}

/** All protocol measurements that currently span >= 2 distinct cardiac beats. */
export function multiBeatAverages(ms: Measurement[]): MeasurementAggregate[] {
  const seen = new Set<string>();
  const out: MeasurementAggregate[] = [];
  for (const m of ms) {
    if (!m.measurementId || seen.has(m.measurementId)) continue;
    seen.add(m.measurementId);
    const agg = aggregateProtocolMeasurement(ms, m.measurementId);
    if (agg?.isAveraged) out.push(agg);
  }
  return out;
}
