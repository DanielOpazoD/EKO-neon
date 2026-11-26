import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { computeGroundTruth } from '@/simulator/hemodynamics/groundTruth';
import type { Measurement } from '@/simulator/measurements/types';
import { buildEducationalReport } from './report';
import { scoreMeasurements } from './scoring/scoring';

/** A measurement as the tools commit it: free (no protocol id) unless `measurementId` is given. */
function measurement(
  m: Pick<Measurement, 'kind' | 'value' | 'units' | 'modality'> & Partial<Measurement>,
): Measurement {
  return {
    id: `m-${m.kind}-${m.value}`,
    label: m.measurementId ?? 'Libre',
    measurementId: null,
    technique: null,
    sourceViewId: null,
    viewScore: 90,
    frameId: 1,
    phase: 0,
    timeS: 0,
    geometry: [],
    imageQualityScore: null,
    userAssisted: false,
    referenceGuidelineIds: [],
    createdAt: '2026-01-01T00:00:00Z',
    ...m,
  };
}

describe('matching measurements with the model truth', () => {
  const afCase = loadCaseById('af-diastolic');
  const af = computeGroundTruth(afCase);

  it('lets a free measurement stand only for a required one it can be: kind, units and modality', () => {
    // a free CW velocity, on the case's MR jet say: by kind alone it scored 100 as the septal e′ (5.5 cm/s)
    const cw = measurement({ kind: 'velocity', value: 5.0, units: 'm/s', modality: 'cw' });
    const rows = scoreMeasurements(afCase, af, [cw]).rows;
    for (const id of ['e-prime-septal', 'e-prime-lateral', 'mitral-e']) {
      const row = rows.find((r) => r.measurementId === id);
      assert.ok(row, id);
      assert.equal(row.measured, null, `a CW velocity in m/s is not the ${row.label}`);
      assert.equal(row.points, 0);
    }
    // the septal e′ taken where it is measured, TDI in cm/s, still stands for it in practice
    const tdi = measurement({
      kind: 'velocity',
      value: af.mitral.ePrimeSeptalCmps,
      units: 'cm/s',
      modality: 'tdi',
    });
    const septal = scoreMeasurements(afCase, af, [tdi]).rows.find(
      (r) => r.measurementId === 'e-prime-septal',
    );
    assert.equal(septal?.measured, af.mitral.ePrimeSeptalCmps);
    assert.equal(septal?.points, 100);
  });

  it('compares a protocol measurement only with a truth the case has: no A wave in AF', () => {
    assert.equal(af.mitral.aPeakMps, 0);
    for (const value of [0.3, 0]) {
      const a = measurement({
        kind: 'velocity',
        value,
        units: 'm/s',
        modality: 'pw',
        measurementId: 'mitral-a',
      });
      const row = buildEducationalReport([a], af, false).rows[0]!;
      assert.equal(row.truth, null, `A ${value} m/s`);
      assert.equal(row.deviation, null, `A ${value} m/s: no «+Infinity %» or «NaN %»`);
    }
    const e = measurement({
      kind: 'velocity',
      value: af.mitral.ePeakMps * 1.1,
      units: 'm/s',
      modality: 'pw',
      measurementId: 'mitral-e',
    });
    const row = buildEducationalReport([e], af, false).rows[0]!;
    assert.ok(row.truth?.endsWith('m/s'));
    assert.equal(row.deviation, '+10 %');
  });

  it('gives a free measurement no truth to compare with: it records no structure', () => {
    const t = computeGroundTruth(loadCaseById('normal-excellent-window'));
    const free = [
      measurement({ kind: 'linear', value: 3.4, units: 'cm', modality: '2d' }),
      measurement({ kind: 'velocity', value: 1.1, units: 'm/s', modality: 'cw' }),
      measurement({ kind: 'vti', value: 21, units: 'cm', modality: 'pw' }),
    ];
    for (const row of buildEducationalReport(free, t, false).rows) {
      assert.equal(row.truth, null, row.value);
      assert.equal(row.deviation, null, row.value);
    }
  });
});
