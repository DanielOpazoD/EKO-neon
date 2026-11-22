import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { cineOffsetAtX, ecgTimeAtX, ecgTracePoints, ecgX, type EcgLayout } from './ecgTrace';

describe('ECG trace geometry & cine scrubbing', () => {
  it('inverts ecgX and ecgTimeAtX cleanly in sliding-window mode and maps cine offsets', () => {
    const layout: EcgLayout = {
      x0: 10,
      width: 600,
      y: 440,
      height: 30,
      spanS: 3.0,
      headS: 10.0,
    };
    for (const t of [7.0, 8.25, 9.5, 10.0]) {
      const x = ecgX(t, layout);
      assert.ok(Math.abs(ecgTimeAtX(x, layout) - t) < 1e-6);
    }
    // 61 cine frames spanning 7.0..10.0 s: start -> -60, midpoint -> -30, end -> 0
    const win = { startS: 7.0, endS: 10.0 };
    assert.equal(cineOffsetAtX(ecgX(7.0, layout), layout, win, 61), -60);
    assert.equal(cineOffsetAtX(ecgX(8.5, layout), layout, win, 61), -30);
    assert.equal(cineOffsetAtX(ecgX(10.0, layout), layout, win, 61), 0);
    // Out-of-bounds clicks clamp safely to [-(length-1), 0]
    assert.equal(cineOffsetAtX(-50, layout, win, 61), -60);
    assert.equal(cineOffsetAtX(999, layout, win, 61), 0);
  });

  it('splits sweep-mode ECG polylines at the wrap-around head without drawing a horizontal flyback', () => {
    const layout: EcgLayout = {
      x0: 0,
      width: 400,
      y: 450,
      height: 30,
      spanS: 2.0,
      headS: 5.0,
      sweep: { headColumn: 100, columns: 400 },
    };
    // 2 seconds of samples from t=3.05 to t=4.95 every 0.05 s
    const pairs: number[] = [];
    for (let i = 0; i < 39; i++) {
      pairs.push(3.05 + i * 0.05, 0.25);
    }
    const lines = ecgTracePoints(new Float64Array(pairs), layout);
    assert.equal(lines.length, 2, 'Sweep wrapping across right edge should split into 2 polylines');
    // Each polyline must be strictly increasing in x (no flyback segment across the screen)
    for (const line of lines) {
      for (let i = 1; i < line.length; i++) {
        assert.ok(line[i]!.x >= line[i - 1]!.x, 'Polyline segment moved backward in x');
      }
    }
    // Sweep ecgX <-> ecgTimeAtX roundtrip inside the active sweep
    const xSample = ecgX(4.2, layout);
    assert.ok(Math.abs(ecgTimeAtX(xSample, layout) - 4.2) < 1e-6);
  });
});
