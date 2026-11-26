import assert from 'node:assert/strict';
import test from 'node:test';
import { DEFAULT_COLOR } from '@/simulator/doppler/color/colorDoppler';
import { colorWallFilterRangeMps } from '@/simulator/doppler/color/kasaiEstimator';
import { wallFilterControl, wallFilterRequestFor, wallFilterText } from './wallFilterControl';

test('the wall filter slider spans exactly what 8 pulses allow, in ~1 mm/s steps, and holds the cutoff that filters', () => {
  for (const scale of [0.15, 0.2, 0.35, 0.5, 0.62, 0.68, 1.0, 1.2]) {
    const { min, max } = colorWallFilterRangeMps(scale);
    const c = wallFilterControl(DEFAULT_COLOR.wallFilterMps, scale);
    // the exact bounds, both reachable on the step grid: the packet's lowest cutoff can be chosen
    assert.equal(c.min, min);
    assert.equal(c.max, max);
    const steps = (c.max - c.min) / c.step;
    assert.ok(Math.abs(steps - Math.round(steps)) < 1e-9, `scale ${scale}: ${steps} steps`);
    assert.ok(c.step > 0.0009 && c.step < 0.0011, `scale ${scale}: step ${c.step}`);
    // the value is inside the slider at every scale (the default sat outside the rounded bounds under 0.62 m/s)
    assert.ok(c.cutMps >= c.min && c.cutMps <= c.max, `scale ${scale}: ${c.cutMps} outside ${c.min}–${c.max}`);
    // and the request is said to be limited when the scale bounds it (at these scales, by more than the 0.1 cm/s shown)
    assert.equal(c.limited, DEFAULT_COLOR.wallFilterMps > max, `scale ${scale}`);
  }
  // a request under the packet's lowest cutoff is limited too; off is off, not limited
  const low = wallFilterControl(0.04, 0.68);
  assert.equal(low.cutMps, colorWallFilterRangeMps(0.68).min);
  assert.equal(low.limited, true);
  const off = wallFilterControl(0, 0.68);
  assert.equal(off.cutMps, 0);
  assert.equal(off.limited, false);
});

test('the wall filter notice tells a limit the console shows, to its 0.1 cm/s', () => {
  // from 0.6232 m/s down the scale bounds the 20 cm/s default: at a PRF-limited 0.6231 m/s by 0.002 cm/s, which both
  // values read alike. Compared to 1e-9 m/s it said «Corte pedido 20.0 cm/s: la escala lo limita a 20.0 cm/s.»
  const unseen = wallFilterControl(DEFAULT_COLOR.wallFilterMps, 0.6231);
  assert.ok(unseen.cutMps < DEFAULT_COLOR.wallFilterMps, `bounded to ${unseen.cutMps}`);
  assert.equal(wallFilterText(unseen.cutMps), '20.0 cm/s');
  assert.equal(unseen.limited, false);
  // at 0.621 m/s it reads 19.9 against 20.0: told
  const seen = wallFilterControl(DEFAULT_COLOR.wallFilterMps, 0.621);
  assert.equal(wallFilterText(seen.cutMps), '19.9 cm/s');
  assert.equal(seen.limited, true);
});

test('the slider left at an end the scale imposes keeps a request past it: back at 0.68 m/s the filter is 20 cm/s again', () => {
  const at035 = wallFilterControl(0.2, 0.35);
  // the learner drags the slider to its top at 0.35 m/s, where it already shows the 11.2 cm/s bound
  assert.equal(wallFilterRequestFor(at035.max, 0.2, 0.35), 0.2);
  assert.ok(Math.abs(wallFilterControl(wallFilterRequestFor(at035.max, 0.2, 0.35), 0.68).cutMps - 0.2) < 1e-12);
  // a position inside the range sets itself, and so does the end when the request was not past it
  const inside = at035.min + 10 * at035.step;
  assert.equal(wallFilterRequestFor(inside, 0.2, 0.35), inside);
  assert.equal(wallFilterRequestFor(at035.max, 0.08, 0.35), at035.max);
  // the same at the bottom: a request under the lowest cutoff survives the slider left there
  const at1 = wallFilterControl(0.05, 1);
  assert.equal(wallFilterRequestFor(at1.min, 0.05, 1), 0.05);
  assert.equal(wallFilterRequestFor(at1.min, 0.2, 1), at1.min);
});
