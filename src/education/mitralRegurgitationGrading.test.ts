import test from 'node:test';
import assert from 'node:assert/strict';
import { CASE_INPUTS, loadCaseById } from '@/cases';
import { venaContractaFromEroaCm } from '@/clinical/formulas';
import { gradeMitralRegurgitation, type ValveGrade } from '@/clinical/guidelines/valveGrading';
import { buildCaseModels, REST_PATIENT } from '@/simulator/anatomy/caseModels';
import { computeGroundTruth, type StructuredEchoTruth } from '@/simulator/hemodynamics/groundTruth';
import { expectedFindings } from './impression';
import { pathologyImpressions } from './report';

const truths = new Map<string, StructuredEchoTruth>();

/** The truth the simulator core computes for a case at rest. */
function truthFor(id: string): StructuredEchoTruth {
  let t = truths.get(id);
  if (!t) {
    const c = loadCaseById(id);
    t = computeGroundTruth(c, buildCaseModels(c, REST_PATIENT).tables);
    truths.set(id, t);
  }
  return t;
}

/** The MR sentence of the report's impression, if the case has one. */
function mrLine(t: StructuredEchoTruth): string | undefined {
  return pathologyImpressions(t).find((l) => l.startsWith('Insuficiencia mitral'));
}

// EROA (cm²), regurgitant volume (mL) and fraction (%) that disagree, and the grade two of the three give (ASE 2017,
// Table 8). The first four read otherwise while the vena contracta derived from the same EROA voted beside them.
const DISCORDANT: [number, number, number, ValveGrade][] = [
  [0.3, 62, 52, 'severe'], // volume and fraction severe: read moderate with its VC of 0.62 cm
  [0.25, 70, 64, 'severe'], // the 64 % fraction of valveGrading.ts's header: read moderate with 0.56 cm
  [0.42, 45, 40, 'moderate'], // only the orifice severe: read severe with its own 0.73 cm
  [0.22, 25, 28, 'mild'], // volume and fraction mild: read moderate with 0.53 cm
  [0.45, 62, 45, 'severe'],
  [0.38, 58, 49, 'moderate'],
  [0.15, 35, 25, 'mild'],
  [0.12, 28, 45, 'mild'],
  [0.18, 32, 33, 'moderate'],
  [0.08, 12, 15, 'none'],
];

test('MR is graded by two of EROA, regurgitant volume and fraction; the vena contracta derived from the EROA does not vote', () => {
  for (const [eroaCm2, regurgitantVolumeMl, regurgitantFractionPct, grade] of DISCORDANT) {
    const criteria = { eroaCm2, regurgitantVolumeMl, regurgitantFractionPct };
    const label = `${eroaCm2} cm², ${regurgitantVolumeMl} mL, ${regurgitantFractionPct} %`;
    assert.equal(gradeMitralRegurgitation(criteria), grade, label);
    // the width of that same orifice riding along, as the ground truth used to pass it: the orifice must not count twice
    const withVc = { ...criteria, venaContractaCm: venaContractaFromEroaCm(eroaCm2) };
    assert.equal(
      gradeMitralRegurgitation(withVc),
      grade,
      `${label}, VC ${withVc.venaContractaCm.toFixed(2)} cm`,
    );
  }
});

// Each case's MR grade, frozen (null: no mitral jet). The HFrEF case's functional MR is moderate by design (0.25 cm²).
const CASE_MR_GRADE: Record<string, ValveGrade | null> = {
  'normal-excellent-window': null,
  'normal-difficult-window': null,
  'hfref-severe-mr': 'moderate',
  'inferior-rwma': null,
  'aortic-stenosis-moderate': null,
  'aortic-stenosis-severe': null,
  'hocm-sam': 'mild',
  'mvp-primary-mr': 'severe',
  'pulmonary-hypertension-rv': null,
  'pericardial-effusion-tamponade': null,
  'af-diastolic': 'mild',
  'artifact-challenge': 'mild',
  'aortic-regurgitation-severe': null,
};

const SPANISH: Record<ValveGrade, string> = {
  none: 'trivial',
  mild: 'leve',
  moderate: 'moderada',
  severe: 'severa',
};

test('the 12 cases keep their MR grade in the truth, the expected findings and the report', () => {
  assert.deepEqual(Object.keys(CASE_MR_GRADE).sort(), CASE_INPUTS.map((c) => c.id).sort());
  for (const [id, grade] of Object.entries(CASE_MR_GRADE)) {
    const t = truthFor(id);
    const mr = t.regurgitation.mr;
    assert.equal(mr ? gradeMitralRegurgitation(mr) : null, grade, id);
    assert.ok(expectedFindings(t).includes(`mr-${grade ?? 'none'}`), id);
    if (!mr || grade === null) {
      assert.equal(mrLine(t), undefined, id);
      continue;
    }
    // the truth record, expected vena contracta and all, grades as its three criteria alone
    const { eroaCm2, regurgitantVolumeMl, regurgitantFractionPct } = mr;
    assert.equal(
      gradeMitralRegurgitation({ eroaCm2, regurgitantVolumeMl, regurgitantFractionPct }),
      grade,
      id,
    );
    const line = mrLine(t);
    assert.ok(line?.startsWith(`Insuficiencia mitral ${SPANISH[grade]} (`), `${id}: ${line}`);
  }
});

test('the MR impression cites no restatement of the EROA that could read against its grade', () => {
  let lines = 0;
  for (const { id } of CASE_INPUTS) {
    const line = mrLine(truthFor(id));
    if (!line) continue;
    lines++;
    // the vena contracta and the PISA radius are the EROA as a width and as a radius: a mild orifice's fell in the
    // guideline's intermediate band of both (VC 0.36–0.50 cm, PISA@40 0.47–0.62 cm). The line grades on, and cites, the
    // EROA, the regurgitant volume and the fraction alone
    assert.doesNotMatch(line, /\bVC \d|PISA/i, `${id}: «${line}»`);
  }
  assert.ok(lines >= 5, 'the cases with MR write their line');
});
