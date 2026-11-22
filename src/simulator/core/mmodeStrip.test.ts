import assert from 'node:assert/strict';
import test from 'node:test';
import {
  binsToTrace,
  buildRowMap,
  columnSource,
  MmodeLineCache,
  mmodeLineSamples,
  mmodePhaseBins,
  mmodePulsesPerColumn,
} from './mmodeStrip';
import { StripTimeGrid } from './stripTimeGrid';

test('StripTimeGrid emits contiguous cell-centre timestamps across variable frame steps', () => {
  const grid = new StripTimeGrid();
  const cps = 100; // 10 ms per column

  const t1 = grid.advance(1.035, 0.035, cps);
  assert.equal(t1.length, 3);
  assert.ok(Math.abs(t1[0]! - 1.005) < 1e-9);
  assert.ok(Math.abs(t1[1]! - 1.015) < 1e-9);
  assert.ok(Math.abs(t1[2]! - 1.025) < 1e-9);

  // Next step picks up seamlessly from column 3 without dropping or duplicating columns
  const t2 = grid.advance(1.06, 0.025, cps);
  assert.equal(t2.length, 3);
  assert.ok(Math.abs(t2[0]! - 1.035) < 1e-9);
  assert.ok(Math.abs(t2[2]! - 1.055) < 1e-9);

  grid.reset();
  const t3 = grid.advance(5.02, 0.02, cps);
  assert.equal(t3.length, 2);
  assert.ok(Math.abs(t3[0]! - 5.005) < 1e-9);
});

test('MmodeLineCache, columnSource, and binsToTrace handle circular wraparound and trace budgets', () => {
  assert.equal(mmodeLineSamples(16), 800);
  assert.equal(mmodeLineSamples(0.5), 64);
  assert.equal(mmodePhaseBins(200, 1.0), 256);
  assert.equal(mmodePhaseBins(10, 1.0), 128);
  assert.equal(mmodePhaseBins(5000, 1.0), 2048);
  assert.equal(mmodePulsesPerColumn(200), 5);

  const cache = new MmodeLineCache();
  cache.reset('k1', 128);
  assert.equal(columnSource(cache, 0.25, 8), null);

  const lineA = { amp: new Float32Array([1]), velocity: null };
  const lineB = { amp: new Float32Array([2]), velocity: null };
  cache.set(127, lineA);
  cache.set(0, lineB);
  assert.equal(cache.filled, 2);

  // Phase 127.5 / 128 wraps between bin 127 and bin 0 with span = 1 and t = 0.5
  const src = columnSource(cache, 127.5 / 128, 4);
  assert.ok(src);
  assert.equal(src.lo, lineA);
  assert.equal(src.hi, lineB);
  assert.equal(src.span, 1);
  assert.ok(Math.abs(src.t - 0.5) < 1e-6);

  // binsToTrace skips already cached bins (127, 0) and spreads evenly when capped by maxTraces
  const missing = binsToTrace(cache, [0 / 128, 10 / 128, 20 / 128, 30 / 128, 40 / 128], 3);
  assert.equal(missing.length, 3);
  assert.ok(!missing.includes(0) && !missing.includes(127));
});

test('buildRowMap weights form a partition of unity for both downsampling and upsampling', () => {
  for (const [rows, samples] of [
    [120, 800], // downsampling (ratio > 1)
    [300, 120], // upsampling (ratio < 1)
    [200, 200], // 1:1
  ] as const) {
    const map = buildRowMap(rows, samples);
    assert.equal(map.rows, rows);
    assert.equal(map.samples, samples);
    for (let y = 0; y < rows; y++) {
      let sum = 0;
      for (let k = 0; k < map.taps; k++) {
        const w = map.weights[y * map.taps + k]!;
        assert.ok(w >= -1e-7, `negative weight ${w}`);
        sum += w;
      }
      assert.ok(
        Math.abs(sum - 1) < 1e-5,
        `row ${y} weights for (${rows},${samples}) summed to ${sum}`,
      );
    }
  }
});
