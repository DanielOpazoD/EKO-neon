import assert from 'node:assert/strict';
import test from 'node:test';
import { CASE_INPUTS } from '.';
import { validateCase, type CaseDefinitionInput } from './schema';

/**
 * validateCase reads untrusted case JSON, and the schema drops the keys it does not know. A key the model has stopped
 * reading must not pass in silence: a case that typed the aortic opening by hand loaded, and its valve opened as far as
 * the effective orifice area says (0.99 for 3.0 cm²) instead of the 0.3 it declared.
 */
const RETIRED_OPENING =
  'anatomy.aorticValve.maxOpeningFraction: ya no se usa: la apertura de las cúspides sale de hemodynamics.avEffectiveAreaCm2';

test('validateCase rejects the aortic opening typed by hand, which the effective orifice area now sets', () => {
  const severe = CASE_INPUTS.find((c) => c.id === 'aortic-stenosis-severe')!;
  const legacy = JSON.parse(JSON.stringify(severe));
  legacy.anatomy.aorticValve.maxOpeningFraction = 0.3;
  legacy.hemodynamics.avEffectiveAreaCm2 = 3.0;
  const r = validateCase(legacy);
  assert.equal(r.ok, false);
  assert.equal(r.case, undefined);
  assert.deepEqual(r.errors, [RETIRED_OPENING]);
  // reported with the schema's own errors too
  legacy.anatomy.aorticValve.calcification = 2;
  const both = validateCase(legacy);
  assert.equal(both.errors[0], RETIRED_OPENING);
  assert.ok(
    both.errors.some((e) => e.startsWith('anatomy.aorticValve.calcification')),
    both.errors.join('; '),
  );
  // without the key the same case validates
  delete legacy.anatomy.aorticValve.maxOpeningFraction;
  legacy.anatomy.aorticValve.calcification = 0.5;
  assert.deepEqual(validateCase(legacy).errors, []);
  // the built-in cases are typed literals, which normalizeCase reads: TypeScript refuses the key in them
  const typed: CaseDefinitionInput = {
    ...severe,
    anatomy: {
      ...severe.anatomy,
      // @ts-expect-error the aortic valve has no maxOpeningFraction any more
      aorticValve: { ...severe.anatomy.aorticValve, maxOpeningFraction: 0.3 },
    },
  };
  assert.deepEqual(validateCase(typed).errors, [RETIRED_OPENING]);
});
