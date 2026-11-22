import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { CINE_ECG_MARGIN_S, ECG_HZ, EcgRing } from './ecgRing';

const SINUS_RHYTHM = {
  type: 'sinus' as const,
  heartRateBpm: 65,
  rrVariabilityPct: 0,
  pvcProbability: 0,
};

describe('EcgRing rolling buffer', () => {
  it('accumulates 200 Hz samples, compacts cleanly over 12 seconds, and caps live tail at 1200 samples', () => {
    const ring = new EcgRing();
    const rrS = 60 / 65;
    const dt = 1 / 30;
    let timeS = 0;
    for (let step = 0; step < 360; step++) {
      const startBeatT = timeS % rrS;
      timeS += dt;
      ring.advance(startBeatT, rrS, dt, timeS, timeS - 2.0, SINUS_RHYTHM, Math.floor(timeS / rrS), 42);
    }

    // Active window stays bounded near 6 s * 200 Hz (~1200 samples), not unbounded growth (2400)
    assert.ok(ring.length >= 1190 && ring.length <= 1220, `Unexpected retained length: ${ring.length}`);

    const live = ring.extractTail(false, timeS, timeS - 2.0);
    assert.equal(live.length % 2, 0);
    const livePairs = live.length / 2;
    assert.ok(livePairs <= 1200 && livePairs >= 1190);
    // Monotonically increasing timestamps at ~1/ECG_HZ spacing
    for (let i = 2; i < live.length; i += 2) {
      const stepS = live[i]! - live[i - 2]!;
      assert.ok(Math.abs(stepS - 1 / ECG_HZ) < 1e-6, `Expected 5ms ECG spacing, got ${stepS}`);
    }
  });

  it('extends frozen ECG tail back to cover a long slow-cadence cine window', () => {
    const ring = new EcgRing();
    const rrS = 0.9;
    const dt = 1 / 20;
    let timeS = 0;
    // Simulate 10 seconds with oldest cine frame held at t = 1.0 s (so cutoff keeps samples back to 1.0 - margin)
    for (let step = 0; step < 200; step++) {
      const startBeatT = timeS % rrS;
      timeS += dt;
      ring.advance(startBeatT, rrS, dt, timeS, 1.0, SINUS_RHYTHM, Math.floor(timeS / rrS), 7);
    }

    const live = ring.extractTail(false, timeS, 1.0);
    const frozen = ring.extractTail(true, timeS, 1.0);
    assert.equal(live.length / 2, 1200);
    assert.ok(frozen.length > live.length, 'Frozen tail should include older cine history beyond 6 s');
    const oldestFrozenT = frozen[0]!;
    assert.ok(
      oldestFrozenT <= 1.0 - CINE_ECG_MARGIN_S + 1 / ECG_HZ + 1e-6,
      `Expected frozen ECG start <= ${(1.0 - CINE_ECG_MARGIN_S).toFixed(2)}s, got ${oldestFrozenT}`,
    );
  });
});
