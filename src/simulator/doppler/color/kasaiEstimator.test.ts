import assert from 'node:assert/strict';
import test from 'node:test';
import {
  CLUTTER_BLOOD_RATIO_POW,
  CLUTTER_SIGMA_V_MPS,
  colorSpectralWidthMps,
  kasaiEstimate,
} from './kasaiEstimator';

function fillGrid(n: number, v: number, disp: number, rel = 1, vPerp = 0, depthCm = 8) {
  const sigmaV = colorSpectralWidthMps(v, disp, vPerp, depthCm);
  return {
    cellV: new Float32Array(n).fill(v),
    cellSigmaV: new Float32Array(n).fill(sigmaV),
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
  const { cellV, cellSigmaV, cellRel } = fillGrid(n, 0.35, 0.02, 1.0);

  const r1 = kasaiEstimate(cellV, cellSigmaV, cellRel, 12, 30, nL, nS, 0.62, 0.06, {
    realization: 3,
    seed: 1337,
  });
  const vel1 = r1.velocity.slice();
  const pow1 = r1.power.slice();

  const r2 = kasaiEstimate(cellV, cellSigmaV, cellRel, 12, 30, nL, nS, 0.62, 0.06, {
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
    kasaiEstimate(flow.cellV, flow.cellSigmaV, flow.cellRel, 5, 20, nL, nS, scale, wf, {
      realization: 1,
      seed: 42,
    }).power,
  );

  const dc = fillGrid(n, 0.0, 0.0, 1.0);
  const dcPow = meanOf(
    kasaiEstimate(dc.cellV, dc.cellSigmaV, dc.cellRel, 5, 20, nL, nS, scale, wf, {
      realization: 1,
      seed: 42,
    }).power,
  );

  // 2 * scale aliases to phase 2π ≡ 0 per pulse, so a real regression wall filter suppresses it like DC
  const twoNyq = fillGrid(n, 2 * scale, 0.0, 1.0);
  const twoNyqPow = meanOf(
    kasaiEstimate(twoNyq.cellV, twoNyq.cellSigmaV, twoNyq.cellRel, 5, 20, nL, nS, scale, wf, {
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
  const lamRes = kasaiEstimate(lam.cellV, lam.cellSigmaV, lam.cellRel, 8, 25, nL, nS, 0.62, 0.06, {
    realization: 2,
    seed: 77,
  });
  const lamVar = meanOf(lamRes.variance);
  const lamCoh = meanOf(lamRes.coherence);

  const turb = fillGrid(n, 0.4, 0.85, 1.0);
  const turbRes = kasaiEstimate(
    turb.cellV,
    turb.cellSigmaV,
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

test('kasaiEstimate responds to absolute sigmaVMps from transverse geometric broadening even at low v_axial', () => {
  const nL = 12;
  const nS = 12;
  const n = nL * nS;

  // θ ≈ 80° laminar flow: small v_axial = 0.14 m/s, large transverse v_perp
  const sigmaAxialOnly = colorSpectralWidthMps(0.14, 0.04, 0, 8);
  const sigmaTransverse = colorSpectralWidthMps(0.14, 0.04, 0.79, 8);
  assert.ok(
    sigmaTransverse > sigmaAxialOnly * 3,
    `expected transverse sigmaV (${sigmaTransverse.toFixed(4)}) > 3× axial-only (${sigmaAxialOnly.toFixed(4)})`,
  );

  // Fast transverse jet crossing at near-90° (v_axial = 0.14, v_perp = 1.8 m/s at r = 6 cm)
  const axialOnly = fillGrid(n, 0.14, 0.04, 1.0, 0, 6);
  const transverse = fillGrid(n, 0.14, 0.04, 1.0, 1.8, 6);
  const resAxial = kasaiEstimate(
    axialOnly.cellV,
    axialOnly.cellSigmaV,
    axialOnly.cellRel,
    4,
    20,
    nL,
    nS,
    0.68,
    0.06,
    { realization: 5, seed: 91 },
  );
  const varAxial = meanOf(resAxial.variance);
  const resTrans = kasaiEstimate(
    transverse.cellV,
    transverse.cellSigmaV,
    transverse.cellRel,
    4,
    20,
    nL,
    nS,
    0.68,
    0.06,
    { realization: 5, seed: 91 },
  );
  const varTrans = meanOf(resTrans.variance);
  assert.ok(
    varTrans >= varAxial + 0.08,
    `expected transverse broadening to elevate Kasai variance by >= 0.08, got axial=${varAxial.toFixed(3)} trans=${varTrans.toFixed(3)}`,
  );
});

test('kasaiEstimate 3-basis Legendre filter rejects slow myocardial clutter at WF=6 cm/s while passing wall flash at WF=2 cm/s and valve snaps', () => {
  const nL = 10;
  const nS = 10;
  const n = nL * nS;
  const scale = 0.68;
  const clutterCls = new Float32Array(n).fill(0.5);
  const bloodCls = new Float32Array(n).fill(1.0);

  const runClutter = (vWall: number, wfMps: number) => {
    const cellV = new Float32Array(n).fill(vWall);
    const cellSigmaV = new Float32Array(n).fill(CLUTTER_SIGMA_V_MPS);
    const cellRel = new Float32Array(n).fill(CLUTTER_BLOOD_RATIO_POW);
    return meanOf(
      kasaiEstimate(
        cellV,
        cellSigmaV,
        cellRel,
        6,
        24,
        nL,
        nS,
        scale,
        wfMps,
        { realization: 4, seed: 314 },
        clutterCls,
      ).power,
    );
  };

  // Stationary myocardium (0 cm/s), cutoff-speed wall (6 cm/s), and peak systolic wall (8 cm/s) at default WF=6 cm/s stay below COLOR_THRESHOLD (0.25)
  const pStatic = runClutter(0.0, 0.06);
  const pWall6Wf6 = runClutter(0.06, 0.06);
  const pWall8Wf6 = runClutter(0.08, 0.06);
  assert.ok(pStatic < 0.05, `stationary tissue clutter should stay suppressed (< 0.05), got ${pStatic.toFixed(4)}`);
  assert.ok(pWall6Wf6 < 0.1, `6 cm/s wall motion at WF=6 cm/s should be strongly rejected (< 0.10), got ${pWall6Wf6.toFixed(4)}`);
  assert.ok(
    pWall8Wf6 < 0.22,
    `8 cm/s wall motion at WF=6 cm/s should stay below COLOR_THRESHOLD (< 0.22), got ${pWall8Wf6.toFixed(4)}`,
  );

  // Lowering wall filter to 2 cm/s lets 6 cm/s moving myocardium break through strongly
  const pWall6Wf2 = runClutter(0.06, 0.02);
  assert.ok(
    pWall6Wf2 > 1.0,
    `6 cm/s wall motion at WF=2 cm/s should produce strong clutter flash (> 1.0), got ${pWall6Wf2.toFixed(4)}`,
  );

  // Fast valve leaflet snap (20 cm/s) breaks through even at default WF=6 cm/s
  const pSnap20Wf6 = runClutter(0.2, 0.06);
  assert.ok(
    pSnap20Wf6 > 1.5,
    `20 cm/s valve snap at WF=6 cm/s should break through (> 1.5), got ${pSnap20Wf6.toFixed(4)}`,
  );

  // Laminar blood at 0.30 m/s stays accurate under the 3-basis Legendre filter
  const lam = fillGrid(n, 0.3, 0.02, 1.0);
  const lamRes = kasaiEstimate(
    lam.cellV,
    lam.cellSigmaV,
    lam.cellRel,
    6,
    24,
    nL,
    nS,
    scale,
    0.06,
    { realization: 4, seed: 314 },
    bloodCls,
  );
  const lamMeanV = meanOf(lamRes.velocity);
  assert.ok(
    Math.abs(lamMeanV - 0.3) <= 0.02,
    `laminar blood 0.30 m/s should stay unbiased under 3-basis filter, got ${lamMeanV.toFixed(4)}`,
  );
});

test('kasaiEstimate isolates 3x3 resolution-cell averaging strictly by Doppler class (blood vs clutter)', () => {
  const nL = 8;
  const nS = 8;
  const n = nL * nS;
  const scale = 0.68;
  const wf = 0.02;

  // Left half (k < 4): laminar blood (+0.40 m/s); right half (k >= 4): strong opposite-sign clutter (-0.12 m/s)
  const cellV = new Float32Array(n);
  const cellSigmaV = new Float32Array(n);
  const cellRel = new Float32Array(n);
  const cellClass = new Float32Array(n);
  const bloodOnlyRel = new Float32Array(n);
  const bloodOnlyClass = new Float32Array(n);

  const bloodSigma = colorSpectralWidthMps(0.4, 0.02, 0, 8);
  for (let l = 0; l < nL; l++) {
    for (let k = 0; k < nS; k++) {
      const i = l * nS + k;
      if (k < 4) {
        cellV[i] = 0.4;
        cellSigmaV[i] = bloodSigma;
        cellRel[i] = 1.0;
        cellClass[i] = 1.0;
        bloodOnlyRel[i] = 1.0;
        bloodOnlyClass[i] = 1.0;
      } else {
        cellV[i] = -0.12;
        cellSigmaV[i] = CLUTTER_SIGMA_V_MPS;
        cellRel[i] = CLUTTER_BLOOD_RATIO_POW;
        cellClass[i] = 0.5;
        bloodOnlyRel[i] = 0;
        bloodOnlyClass[i] = 0;
      }
    }
  }

  const withClutter = kasaiEstimate(
    cellV,
    cellSigmaV,
    cellRel,
    10,
    20,
    nL,
    nS,
    scale,
    wf,
    { realization: 2, seed: 55 },
    cellClass,
  ).velocity.slice();
  const withoutClutter = kasaiEstimate(
    cellV,
    cellSigmaV,
    bloodOnlyRel,
    10,
    20,
    nL,
    nS,
    scale,
    wf,
    { realization: 2, seed: 55 },
    bloodOnlyClass,
  ).velocity.slice();

  // Check boundary blood column (k = 3) immediately adjacent to clutter (k = 4)
  let maxDiff = 0;
  for (let l = 0; l < nL; l++) {
    const idx = l * nS + 3;
    maxDiff = Math.max(maxDiff, Math.abs(withClutter[idx]! - withoutClutter[idx]!));
  }
  assert.ok(
    maxDiff < 1e-5,
    `adjacent tissue clutter must not bias blood 3x3 resolution-cell velocity, maxDiff=${maxDiff}`,
  );
});
