import assert from 'node:assert/strict';
import test from 'node:test';
import { DEFAULT_COLOR } from './colorDoppler';
import {
  CLUTTER_BLOOD_RATIO_POW,
  CLUTTER_SIGMA_V_MPS,
  colorSpectralWidthMps,
  colorWallFilterCutoffMps,
  colorWallFilterRangeMps,
  COLOR_PACKET,
  KASAI_LIN_BASIS,
  KASAI_QUAD_BASIS,
  kasaiEstimate,
  wallFilterCutoffNyquist,
  wallFilterPowerResponse,
  WallFilterResponseTable,
  wallFilterShares,
  WF_MAX_NYQUIST,
  WF_MIN_NYQUIST,
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
  // the default filter (bounded to 0.32 Nyquist at 0.62 m/s): the old 0.06 meant it, and now asks for the lowest one
  const wf = DEFAULT_COLOR.wallFilterMps;

  const r1 = kasaiEstimate(cellV, cellSigmaV, cellRel, 12, 30, nL, nS, 0.62, wf, {
    realization: 3,
    seed: 1337,
  });
  const vel1 = r1.velocity.slice();
  const pow1 = r1.power.slice();

  const r2 = kasaiEstimate(cellV, cellSigmaV, cellRel, 12, 30, nL, nS, 0.62, wf, {
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

  // the default filter, which the old 0.06 meant
  const wf = DEFAULT_COLOR.wallFilterMps;
  const lam = fillGrid(n, 0.4, 0.02, 1.0);
  const lamRes = kasaiEstimate(lam.cellV, lam.cellSigmaV, lam.cellRel, 8, 25, nL, nS, 0.62, wf, {
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
    wf,
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

  // Fast transverse jet crossing at near-90° (v_axial = 0.14, v_perp = 1.8 m/s at r = 6 cm), through the packet's
  // lowest cutoff (7.6 cm/s): under the default 20 cm/s the 14 cm/s flow is filtered to the jitter's floor, whose own
  // variance leaves the transverse broadening a thin margin (+0.098 against +0.148)
  const wf = colorWallFilterRangeMps(0.68).min;
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
    wf,
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
    wf,
    { realization: 5, seed: 91 },
  );
  const varTrans = meanOf(resTrans.variance);
  assert.ok(
    varTrans >= varAxial + 0.08,
    `expected transverse broadening to elevate Kasai variance by >= 0.08, got axial=${varAxial.toFixed(3)} trans=${varTrans.toFixed(3)}`,
  );
});

test('kasaiEstimate 3-basis Legendre filter rejects slow myocardial clutter at the default WF while passing wall flash at a low WF and valve snaps', () => {
  const nL = 10;
  const nS = 10;
  const n = nL * nS;
  const scale = 0.68;
  // the default cutoff, 20 cm/s at 0.68 m/s: the same filter as the uncalibrated "6 cm/s" these checks were written for
  const wfDefault = DEFAULT_COLOR.wallFilterMps;
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

  // Stationary myocardium (0 cm/s), slow wall (6 cm/s), and peak systolic wall (8 cm/s) at the default WF stay below COLOR_THRESHOLD (0.25)
  const pStatic = runClutter(0.0, wfDefault);
  const pWall6Wf6 = runClutter(0.06, wfDefault);
  const pWall8Wf6 = runClutter(0.08, wfDefault);
  assert.ok(pStatic < 0.05, `stationary tissue clutter should stay suppressed (< 0.05), got ${pStatic.toFixed(4)}`);
  assert.ok(pWall6Wf6 < 0.1, `6 cm/s wall motion at the default WF should be strongly rejected (< 0.10), got ${pWall6Wf6.toFixed(4)}`);
  assert.ok(
    pWall8Wf6 < 0.22,
    `8 cm/s wall motion at the default WF should stay below COLOR_THRESHOLD (< 0.22), got ${pWall8Wf6.toFixed(4)}`,
  );

  // Lowering the wall filter to the packet's lowest cutoff (the mean removal, 7.6 cm/s) lets 6 cm/s moving myocardium
  // break through strongly
  const pWall6WfMin = runClutter(0.06, colorWallFilterRangeMps(scale).min);
  assert.ok(
    pWall6WfMin > 1.0,
    `6 cm/s wall motion at the lowest WF should produce strong clutter flash (> 1.0), got ${pWall6WfMin.toFixed(4)}`,
  );

  // Fast valve leaflet snap (20 cm/s) breaks through even at the default WF
  const pSnap20Wf6 = runClutter(0.2, wfDefault);
  assert.ok(
    pSnap20Wf6 > 1.5,
    `20 cm/s valve snap at the default WF should break through (> 1.5), got ${pSnap20Wf6.toFixed(4)}`,
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
    wfDefault,
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
  // the packet's lowest cutoff, so the −0.12 m/s clutter passes and flashes beside the blood
  const wf = colorWallFilterRangeMps(scale).min;

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

/** Power a unit tone at `cutFrac`·Nyquist keeps through the packet regression with these shares (projections, not formulas). */
function toneThroughFilter(cutFrac: number, shares: { mean: number; trend: number; quad: number }): number {
  const N = COLOR_PACKET;
  const w = Math.PI * cutFrac;
  const xr = Array.from({ length: N }, (_, p) => Math.cos(w * p));
  const xi = Array.from({ length: N }, (_, p) => Math.sin(w * p));
  const bases: Array<[ArrayLike<number>, number]> = [
    [Array.from({ length: N }, () => 1 / Math.sqrt(N)), shares.mean],
    [KASAI_LIN_BASIS, shares.trend],
    [KASAI_QUAD_BASIS, shares.quad],
  ];
  for (const [b, s] of bases) {
    let cr = 0;
    let ci = 0;
    for (let p = 0; p < N; p++) {
      cr += xr[p]! * b[p]!;
      ci += xi[p]! * b[p]!;
    }
    for (let p = 0; p < N; p++) {
      xr[p]! -= s * cr * b[p]!;
      xi[p]! -= s * ci * b[p]!;
    }
  }
  let pow = 0;
  for (let p = 0; p < N; p++) pow += xr[p]! ** 2 + xi[p]! ** 2;
  return pow / N;
}

test('wallFilterShares puts the regression filter at −3 dB exactly on the cutoff, which 8 pulses bound to 0.11–0.32 Nyquist', () => {
  // the bounds: the mean removal and the mean + trend + quadratic removal, each at −3 dB on its own cutoff
  assert.ok(Math.abs(toneThroughFilter(WF_MIN_NYQUIST, { mean: 1, trend: 0, quad: 0 }) - 0.5) < 1e-9);
  assert.ok(Math.abs(toneThroughFilter(WF_MAX_NYQUIST, { mean: 1, trend: 1, quad: 1 }) - 0.5) < 1e-9);
  assert.ok(Math.abs(WF_MIN_NYQUIST - 0.1115) < 5e-4, `mean-removal cutoff ${WF_MIN_NYQUIST}`);
  assert.ok(Math.abs(WF_MAX_NYQUIST - 0.3209) < 5e-4, `quadratic-regression cutoff ${WF_MAX_NYQUIST}`);

  for (const scale of [0.3, 0.68, 1.0]) {
    const { min, max } = colorWallFilterRangeMps(scale);
    for (let i = 0; i <= 10; i++) {
      const wf = min + ((max - min) * i) / 10;
      const shares = wallFilterShares(wf, scale);
      const atCut = toneThroughFilter(wf / scale, shares);
      assert.ok(
        Math.abs(atCut - 0.5) < 1e-6,
        `scale ${scale} WF ${wf.toFixed(4)}: tone at the cutoff keeps ${atCut.toFixed(6)} of its power, not one half`,
      );
      // a high-pass around the cutoff: less below it, more above it
      assert.ok(toneThroughFilter((0.8 * wf) / scale, shares) < 0.5);
      assert.ok(toneThroughFilter((1.25 * wf) / scale, shares) > 0.5);
    }
  }

  // what the packet cannot do is bounded, not faked: 0 is off, below 0.11 Nyquist the mean removal, above 0.32 all three
  assert.equal(colorWallFilterCutoffMps(0, 0.68), 0);
  assert.deepEqual(wallFilterShares(0, 0.68), { mean: 0, trend: 0, quad: 0 });
  assert.ok(Math.abs(colorWallFilterCutoffMps(0.02, 0.68) - WF_MIN_NYQUIST * 0.68) < 1e-12);
  assert.deepEqual(wallFilterShares(0.02, 0.68), { mean: 1, trend: 0, quad: 0 });
  assert.ok(Math.abs(colorWallFilterCutoffMps(0.3, 0.4) - WF_MAX_NYQUIST * 0.4) < 1e-12);
  assert.deepEqual(wallFilterShares(0.3, 0.4), { mean: 1, trend: 1, quad: 1 });
  assert.equal(colorWallFilterCutoffMps(DEFAULT_COLOR.wallFilterMps, DEFAULT_COLOR.scaleMps), 0.2);
});

/** Kasai power and mean velocity of uniform laminar flow at `v` over a 16×16 box and four updates. */
function laminarKasai(v: number, scale: number, wf: number): { power: number; velocity: number } {
  const nL = 16;
  const nS = 16;
  const n = nL * nS;
  const { cellV, cellSigmaV, cellRel } = fillGrid(n, v, 0.02, 1.0);
  let power = 0;
  let velocity = 0;
  for (let r = 0; r < 4; r++) {
    const res = kasaiEstimate(cellV, cellSigmaV, cellRel, 10, 30, nL, nS, scale, wf, {
      realization: r + 1,
      seed: 2718,
    });
    power += meanOf(res.power);
    velocity += meanOf(res.velocity);
  }
  return { power: power / 4, velocity: velocity / 4 };
}

// cutoffs inside what the packet allows at each scale, low and high in that range, and the default. [1.0, 0.1] was not:
// under the 0.1115 m/s floor it filtered at 0.1115 and was measured at 0.9 times the cutoff it ran with
const WF_CASES: Array<[number, number]> = [
  [0.3, 0.05],
  [0.4, 0.06],
  [0.68, 0.1],
  [0.68, DEFAULT_COLOR.wallFilterMps],
  [1.0, 0.12],
  [1.0, 0.2],
];

test('the wall filter cases are cutoffs the packet allows at their scale', () => {
  for (const [scale, wf] of WF_CASES) {
    const { min, max } = colorWallFilterRangeMps(scale);
    assert.ok(wf >= min && wf <= max, `WF ${wf} m/s at ${scale} m/s lies outside ${min.toFixed(4)}–${max.toFixed(4)}`);
    assert.equal(colorWallFilterCutoffMps(wf, scale), wf);
  }
});

test('Kasai colour power is −3 dB ± 2 dB at the wall filter cutoff it is set to, at several scales', () => {
  // it was −8.6 dB at the default "6 cm/s", whose −3 dB fell at 0.19–0.20 m/s
  for (const [scale, wf] of WF_CASES) {
    const dB = (v: number) =>
      10 * Math.log10(laminarKasai(v, scale, wf).power / laminarKasai(v, scale, 0).power);
    const atCut = dB(wf);
    assert.ok(
      atCut > -5 && atCut < -1,
      `scale ${scale} m/s, WF ${wf} m/s: power at v = WF is ${atCut.toFixed(1)} dB, expected −3 ± 2 dB`,
    );
    assert.ok(dB(0.5 * wf) < -4.5, `scale ${scale} WF ${wf}: half the cutoff must be stopped`);
    assert.ok(dB(2 * wf) > -1.5, `scale ${scale} WF ${wf}: twice the cutoff must pass`);
  }
});

test('Kasai colour reads laminar flow within 20 % from twice the wall filter cutoff up, at several scales', () => {
  // it read 0.20 m/s as 0.335 at WF 0.10 and 0.68 m/s, and 0.12 as 0.194 at WF 0.06 and 0.40 m/s
  for (const [scale, wf] of WF_CASES) {
    for (const m of [2, 3]) {
      const v = m * wf;
      if (v > 0.9 * scale) continue;
      const read = laminarKasai(v, scale, wf).velocity;
      assert.ok(
        Math.abs(read - v) < 0.2 * v,
        `scale ${scale} m/s, WF ${wf} m/s: ${v.toFixed(3)} m/s read as ${read.toFixed(3)} (${(((read - v) / v) * 100).toFixed(0)} %)`,
      );
    }
  }
});

// requests inside the packet's range at each scale (its bounds, the middle, the default) and outside it (2 and 50 cm/s)
const RESPONSE_CASES: Array<[number, number]> = [0.3, 0.68, 1.0].flatMap((scale) => {
  const { min, max } = colorWallFilterRangeMps(scale);
  return [min, 0.5 * (min + max), max, DEFAULT_COLOR.wallFilterMps, 0.02, 0.5].map(
    (wf) => [scale, wf] as [number, number],
  );
});

test('wallFilterPowerResponse is the packet regression’s response: −3 dB on the bounded cutoff, periodic in the PRF', () => {
  for (const [scale, wf] of RESPONSE_CASES) {
    const shares = wallFilterShares(wf, scale);
    const cut = colorWallFilterCutoffMps(wf, scale);
    const label = `scale ${scale} m/s, WF ${wf.toFixed(4)} m/s`;
    // −3 dB on the cutoff that filters: the request where the packet allows it, its bound where it does not
    assert.ok(Math.abs(wallFilterPowerResponse(cut, scale, shares) - 0.5) < 1e-9, `${label}: not −3 dB at ${cut}`);
    for (let i = -60; i <= 60; i++) {
      const v = (i / 40) * scale;
      const h = wallFilterPowerResponse(v, scale, shares);
      // the projections of a tone on the packet's Legendre basis, not the closed form
      assert.ok(Math.abs(h - toneThroughFilter(v / scale, shares)) < 1e-9, `${label}: response at ${v.toFixed(3)} m/s`);
      // it acts on the sampled phase: even, and periodic in the PRF (2 × Nyquist)
      assert.ok(Math.abs(h - wallFilterPowerResponse(-v, scale, shares)) < 1e-12);
      assert.ok(Math.abs(h - wallFilterPowerResponse(v + 2 * scale, scale, shares)) < 1e-9);
    }
    // stationary echo and a flow aliasing to zero are both removed
    assert.ok(wallFilterPowerResponse(0, scale, shares) < 1e-12);
    assert.ok(wallFilterPowerResponse(2 * scale, scale, shares) < 1e-9);
  }
  // off passes everything
  for (const v of [0, 0.05, 0.3, 1.2]) assert.equal(wallFilterPowerResponse(v, 0.68, wallFilterShares(0, 0.68)), 1);
});

test('WallFilterResponseTable reads the response within 1.3e-5 at any speed, for the request and Nyquist last set', () => {
  // a trigonometric polynomial of degree 7 in π·v/Nyquist, linearly interpolated on 1024 steps of [0, 1] Nyquist: at
  // most Δ²/8·max|h''|, 1.24e-5 with the mean removal next to zero (Bernstein bounds it by 2.9e-5). Speeds off the
  // table's points over three PRFs either side: negative and aliased ones fold onto it, the response being even and
  // periodic in 2·Nyquist
  const table = new WallFilterResponseTable();
  const check = (wf: number, scale: number) => {
    const shares = wallFilterShares(wf, scale);
    table.set(wf, scale);
    for (let i = -3000; i <= 3000; i++) {
      const v = ((i + 0.37) / 500) * scale;
      const err = Math.abs(table.at(v) - wallFilterPowerResponse(v, scale, shares));
      assert.ok(err < 1.3e-5, `scale ${scale} m/s, WF ${wf.toFixed(4)} m/s: ${v.toFixed(4)} m/s off by ${err}`);
    }
  };
  // one table through every case, as the strip keeps one: each new request or Nyquist refills it
  for (const [scale, wf] of RESPONSE_CASES) check(wf, scale);
  // the same request at another Nyquist is another filter (bounded to 0.32 Nyquist at 0.35 m/s), and off passes all
  check(DEFAULT_COLOR.wallFilterMps, 0.68);
  check(DEFAULT_COLOR.wallFilterMps, 0.35);
  table.set(0, 0.68);
  for (const v of [0, 0.05, 0.3, 1.2]) assert.equal(table.at(v), 1);
});

test('wallFilterCutoffNyquist is the −3 dB point of any shares: the packet’s bounds, the shares solved for a cutoff', () => {
  assert.equal(WF_MIN_NYQUIST, wallFilterCutoffNyquist({ mean: 1, trend: 0, quad: 0 }));
  assert.equal(WF_MAX_NYQUIST, wallFilterCutoffNyquist({ mean: 1, trend: 1, quad: 1 }));
  for (const [scale, wf] of RESPONSE_CASES) {
    const back = wallFilterCutoffNyquist(wallFilterShares(wf, scale)) * scale;
    assert.ok(Math.abs(back - colorWallFilterCutoffMps(wf, scale)) < 1e-9, `scale ${scale} WF ${wf}: ${back}`);
  }
  // a partial mean removal under 1 − √½ leaves a tone at rest more than half its power: no −3 dB point
  assert.equal(wallFilterCutoffNyquist({ mean: 0.29, trend: 0, quad: 0 }), 0);
  const partial = wallFilterCutoffNyquist({ mean: 0.5, trend: 0, quad: 0 });
  assert.ok(partial > 0 && partial < WF_MIN_NYQUIST, `partial mean removal cuts at ${partial}`);
  assert.ok(Math.abs(toneThroughFilter(partial, { mean: 0.5, trend: 0, quad: 0 }) - 0.5) < 1e-9);
  assert.equal(wallFilterCutoffNyquist({ mean: 0, trend: 0, quad: 0 }), 0);
});

test('the colour power the estimator measures follows wallFilterPowerResponse from the cutoff up, within 1 dB', () => {
  // the 2D colour and the colour M-mode (which paints with the response) cut alike. Measured: within 0.56 dB from the
  // cutoff up at every case; below it the estimator keeps the transit jitter's white floor (−5.5 to −9 dB at half the
  // cutoff where the response is −8 to −16 dB). In colour the 2D's noise gate rejects that weak incoherent echo, so just
  // under the cutoff the strip colours more than the 2D, not less (stripEngine.test.ts compares them on one line)
  for (const [scale, wf] of WF_CASES) {
    const shares = wallFilterShares(wf, scale);
    for (const m of [0.5, 1, 1.5, 2, 3]) {
      const v = m * wf;
      if (v > 0.9 * scale) continue;
      const kasai = 10 * Math.log10(laminarKasai(v, scale, wf).power / laminarKasai(v, scale, 0).power);
      const response = 10 * Math.log10(wallFilterPowerResponse(v, scale, shares));
      const label = `scale ${scale} m/s, WF ${wf} m/s, v ${v.toFixed(3)}: estimator ${kasai.toFixed(2)} dB, response ${response.toFixed(2)} dB`;
      if (m >= 1) assert.ok(Math.abs(kasai - response) < 1, label);
      else assert.ok(kasai > response, label);
    }
  }
});
