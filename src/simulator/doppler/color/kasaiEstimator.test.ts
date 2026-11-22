import assert from 'node:assert/strict';
import test from 'node:test';
import { kasaiEstimate } from './kasaiEstimator';

function fillGrid(n: number, v: number, disp: number, rel = 1) {
  return {
    cellV: new Float32Array(n).fill(v),
    cellDisp: new Float32Array(n).fill(disp),
    cellRel: new Float32Array(n).fill(rel),
  };
}

function meanOf(arr: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < arr.length; i++) sum += arr[i]!;
  return sum / arr.length;
}

test('kasaiEstimate recovers laminar velocity and is deterministic per (realization, seed)', () => {
  const nL = 10;
  const nS = 10;
  const n = nL * nS;
  const { cellV, cellDisp, cellRel } = fillGrid(n, 0.35, 0.02, 1.0);

  const r1 = kasaiEstimate(cellV, cellDisp, cellRel, 12, 30, nL, nS, 0.62, 0.06, {
    realization: 3,
    seed: 1337,
  });
  const vel1 = r1.velocity.slice();
  const pow1 = r1.power.slice();

  const r2 = kasaiEstimate(cellV, cellDisp, cellRel, 12, 30, nL, nS, 0.62, 0.06, {
    realization: 3,
    seed: 1337,
  });

  assert.deepEqual(r2.velocity, vel1);
  assert.deepEqual(r2.power, pow1);

  const avgVel = meanOf(vel1);
  assert.ok(
    Math.abs(avgVel - 0.35) < 0.06,
    `expected mean estimated velocity near 0.35 m/s, got ${avgVel.toFixed(4)}`,
  );
  assert.ok(meanOf(pow1) > 0.3, 'laminar flow should have strong post-filter power');
});

test('kasaiEstimate regression wall filter suppresses both stationary clutter and 2x Nyquist aliased clutter', () => {
  const nL = 10;
  const nS = 10;
  const n = nL * nS;
  const scale = 0.62;
  const wf = 0.08;

  const flow = fillGrid(n, 0.35, 0.02, 1.0);
  const flowPow = meanOf(
    kasaiEstimate(flow.cellV, flow.cellDisp, flow.cellRel, 5, 20, nL, nS, scale, wf, {
      realization: 1,
      seed: 42,
    }).power,
  );

  const dc = fillGrid(n, 0.0, 0.0, 1.0);
  const dcPow = meanOf(
    kasaiEstimate(dc.cellV, dc.cellDisp, dc.cellRel, 5, 20, nL, nS, scale, wf, {
      realization: 1,
      seed: 42,
    }).power,
  );

  // 2 * scale aliases to phase 2π ≡ 0 per pulse, so a real regression wall filter suppresses it like DC
  const twoNyq = fillGrid(n, 2 * scale, 0.0, 1.0);
  const twoNyqPow = meanOf(
    kasaiEstimate(twoNyq.cellV, twoNyq.cellDisp, twoNyq.cellRel, 5, 20, nL, nS, scale, wf, {
      realization: 1,
      seed: 42,
    }).power,
  );

  assert.ok(dcPow < flowPow * 0.25, `stationary clutter not suppressed: dc=${dcPow}, flow=${flowPow}`);
  assert.ok(
    twoNyqPow < flowPow * 0.25,
    `2x Nyquist aliased clutter not suppressed: twoNyq=${twoNyqPow}, flow=${flowPow}`,
  );
});

test('kasaiEstimate raises variance and lowers coherence for turbulent jets vs laminar flow', () => {
  const nL = 10;
  const nS = 10;
  const n = nL * nS;

  const lam = fillGrid(n, 0.4, 0.02, 1.0);
  const lamRes = kasaiEstimate(lam.cellV, lam.cellDisp, lam.cellRel, 8, 25, nL, nS, 0.62, 0.06, {
    realization: 2,
    seed: 77,
  });
  const lamVar = meanOf(lamRes.variance);
  const lamCoh = meanOf(lamRes.coherence);

  const turb = fillGrid(n, 0.4, 0.85, 1.0);
  const turbRes = kasaiEstimate(
    turb.cellV,
    turb.cellDisp,
    turb.cellRel,
    8,
    25,
    nL,
    nS,
    0.62,
    0.06,
    { realization: 2, seed: 77 },
  );
  const turbVar = meanOf(turbRes.variance);
  const turbCoh = meanOf(turbRes.coherence);

  assert.ok(turbVar > lamVar + 0.15, `expected turbulent variance (${turbVar}) > laminar (${lamVar})`);
  assert.ok(turbCoh < lamCoh - 0.1, `expected turbulent coherence (${turbCoh}) < laminar (${lamCoh})`);
});
