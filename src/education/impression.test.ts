import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { listCases, loadCaseById } from '@/cases';
import { computeGroundTruth } from '@/simulator/hemodynamics/groundTruth';
import { expectedFindings, FINDINGS, getFinding } from './impression';

/** A label that names a rhythm, or the algorithm of one (ASE 2025 has one for sinus rhythm and one for AF). */
const NAMES_AF = /\bFA\b|fibrilaci/i;
const NAMES_SINUS = /sinusal/i;

describe('structured impression and the rhythm of the case', () => {
  it('names a rhythm only in the rhythm finding: the sheet is the same for every case', () => {
    for (const f of FINDINGS) {
      if (f.id === 'af') continue;
      assert.doesNotMatch(f.label, NAMES_AF, `${f.id}: «${f.label}»`);
      assert.doesNotMatch(f.label, NAMES_SINUS, `${f.id}: «${f.label}»`);
    }
  });

  it('expects no finding whose label contradicts the case rhythm, across the 12 cases', () => {
    let sinusFillingPressure = 0;
    for (const { id } of listCases()) {
      const t = computeGroundTruth(loadCaseById(id));
      const af = t.rhythm === 'atrial-fibrillation';
      for (const f of expectedFindings(t)) {
        const label = getFinding(f)?.label ?? '';
        assert.doesNotMatch(label, af ? NAMES_SINUS : NAMES_AF, `${id} (${t.rhythm}): «${label}»`);
        if (!af && f.startsWith('filling-pressure-')) sinusFillingPressure++;
      }
    }
    // the filling pressure is concluded in sinus rhythm too (assessDiastolic): the check above is not empty
    assert.ok(sinusFillingPressure > 0, 'a sinus-rhythm case expects a filling-pressure finding');
  });
});
