import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { discProfileFromContour, volumeFromProfileMl, cutBySectorDepth } from './simpson';
import { summarizeEnvelope } from './vti';
import { reprojectGeometry } from './geometry';
import { computeSectorMapping } from '../renderer/scanConvert';
import {
  aliasVelocity,
  bernoulliWithProximal,
  bsaDuBois,
  bsaMosteller,
  cardiacOutput,
  circularArea,
  continuityAva,
  ejectionFraction,
  fractionalAreaChange,
  maxPrfForDepth,
  mvaFromPht,
  nyquistVelocity,
  rvspFromTr,
  simplifiedBernoulli,
  simpsonBiplaneVolume,
  strokeVolume,
  velocityRatio,
} from '@/clinical/formulas';

/** Generate an elliptical LV-like contour from one side of the mitral annulus through the apex to the other. */
function makeEllipsoidContour(
  lengthCm: number,
  radiusCm: number,
  pxPerCm: number,
  rotRad = 0,
  nSteps = 80,
): { x: number; y: number }[] {
  const cx = 200;
  const cy = 200;
  const cos = Math.cos(rotRad);
  const sin = Math.sin(rotRad);
  const pts: { x: number; y: number }[] = [];
  // Semi-ellipse along long axis [0, L]: base at z = 0 (radius = 0 -> slightly open annulus), apex at z = L
  // Actually full prolate spheroid of length L and max radius R: center at z = L/2, half-axes a = L/2, b = R
  // Parameterized from theta = -pi/2 + eps (base-left) through 0 (apex) to +pi/2 - eps (base-right)
  const a = lengthCm / 2;
  const b = radiusCm;
  for (let i = 0; i <= nSteps; i++) {
    const phi = -Math.PI + (2 * Math.PI * i) / nSteps;
    // phi = -pi -> base, phi = 0 -> apex, phi = +pi -> base
    const z = a * (1 + Math.cos(phi));
    const r = b * Math.sin(phi);
    const lx = r * pxPerCm;
    const ly = z * pxPerCm;
    pts.push({
      x: cx + lx * cos - ly * sin,
      y: cy + lx * sin + ly * cos,
    });
  }
  return pts;
}

describe('Clinical formulas (single source of truth)', () => {
  it('computes BSA, Bernoulli gradients, continuity AVA, EF and cardiac output accurately', () => {
    assert.ok(Math.abs(bsaMosteller(175, 70) - Math.sqrt((175 * 70) / 3600)) < 1e-12);
    assert.ok(bsaDuBois(175, 70) > 1.75 && bsaDuBois(175, 70) < 1.95);

    assert.equal(simplifiedBernoulli(4), 64);
    assert.equal(bernoulliWithProximal(4, 1), 60);
    assert.equal(rvspFromTr(3, 10), 46);

    // Continuity equation: LVOT d = 2.0 cm (A = pi cm²), LVOT VTI = 20 cm (SV ≈ 62.83 mL), AV VTI = 80 cm -> AVA = pi/4 ≈ 0.785 cm²
    assert.ok(Math.abs(circularArea(2.0) - Math.PI) < 1e-12);
    assert.ok(Math.abs(strokeVolume(2.0, 20) - 20 * Math.PI) < 1e-12);
    assert.ok(Math.abs(continuityAva(2.0, 20, 80) - Math.PI / 4) < 1e-12);
    assert.equal(velocityRatio(1.0, 4.0), 0.25);

    assert.ok(Math.abs(ejectionFraction(120, 48) - 60) < 1e-12);
    assert.ok(Math.abs(cardiacOutput(70, 70) - 4.9) < 1e-12);
    assert.ok(Math.abs(mvaFromPht(220) - 1.0) < 1e-12);
    assert.ok(Math.abs(fractionalAreaChange(20, 11) - 45) < 1e-12);
  });

  it('throws RangeError on non-positive or invalid physiological inputs', () => {
    assert.throws(() => bsaMosteller(0, 70), RangeError);
    assert.throws(() => circularArea(-1), RangeError);
    assert.throws(() => continuityAva(2, 20, 0), RangeError);
    assert.throws(() => ejectionFraction(100, 120), RangeError);
    assert.throws(() => maxPrfForDepth(0), RangeError);
  });

  it('computes Doppler Nyquist limit and wraps aliased velocities with baseline shift', () => {
    // At 15 cm (0.15 m), max PRF = 1540 / 0.30 ≈ 5133.33 Hz
    const prf = maxPrfForDepth(0.15);
    assert.ok(Math.abs(prf - 1540 / 0.3) < 1e-9);
    const vN = nyquistVelocity(prf, 2.5e6);
    assert.ok(Math.abs(vN - (prf * 1540) / 10e6) < 1e-9);

    // Velocity within [-1, +1] stays unchanged; +1.3 wraps to -0.7
    assert.ok(Math.abs(aliasVelocity(0.6, 1.0, 0) - 0.6) < 1e-12);
    assert.ok(Math.abs(aliasVelocity(1.3, 1.0, 0) - -0.7) < 1e-12);
    assert.ok(Math.abs(aliasVelocity(-1.4, 1.0, 0) - 0.6) < 1e-12);
    // Shifted baseline [+0.5] -> interval [-0.5, +1.5], so +1.3 does not alias
    assert.ok(Math.abs(aliasVelocity(1.3, 1.0, 0.5) - 1.3) < 1e-12);
  });
});

describe('Simpson method of discs & sector cutoff', () => {
  it('recovers prolate ellipsoid volume within 1% and is rotation-invariant', () => {
    const L = 8.4; // cm
    const R = 2.4; // cm
    const exactVolMl = (4 / 3) * Math.PI * (L / 2) * R * R;

    const c0 = makeEllipsoidContour(L, R, 32, 0);
    const p0 = discProfileFromContour(c0, 32, 20);
    assert.ok(p0);
    assert.equal(p0.diametersCm.length, 20);
    assert.ok(Math.abs(p0.longAxisCm - L) < 0.05);
    const v0 = volumeFromProfileMl(p0);
    assert.ok(
      Math.abs(v0 - exactVolMl) / exactVolMl < 0.015,
      `Expected ~${exactVolMl.toFixed(2)} mL, got ${v0.toFixed(2)} mL`,
    );

    // Rotated by 37 degrees: identical volume
    const cRot = makeEllipsoidContour(L, R, 32, (37 * Math.PI) / 180);
    const pRot = discProfileFromContour(cRot, 32, 20);
    assert.ok(pRot);
    assert.ok(Math.abs(volumeFromProfileMl(pRot) - v0) < 0.25);

    // Biplane Simpson with orthogonal axes matches 4/3 * pi * (L/2) * Rx * Ry
    const Rx = 2.6;
    const Ry = 2.1;
    const pA4 = discProfileFromContour(makeEllipsoidContour(L, Rx, 32, 0), 32, 20)!;
    const pA2 = discProfileFromContour(makeEllipsoidContour(L, Ry, 32, 0.3), 32, 20)!;
    const vBi = simpsonBiplaneVolume(pA4.diametersCm, pA2.diametersCm, L);
    const exactBi = (4 / 3) * Math.PI * (L / 2) * Rx * Ry;
    assert.ok(Math.abs(vBi - exactBi) / exactBi < 0.015);
  });

  it('rejects degenerate contours with fewer than 5 points', () => {
    assert.equal(
      discProfileFromContour(
        [
          { x: 0, y: 0 },
          { x: 10, y: 20 },
          { x: 20, y: 0 },
        ],
        30,
      ),
      null,
    );
  });

  it('detects when a target structure is clipped at the deepest sector rows', () => {
    const lines = 16;
    const samples = 40;
    const st = new Uint8Array(lines * samples);
    assert.equal(cutBySectorDepth({ structure: st, polar: { lines, samples } }, [5]), false);
    // Mid-depth structure is not cut
    st[8 * samples + 20] = 5;
    assert.equal(cutBySectorDepth({ structure: st, polar: { lines, samples } }, [5]), false);
    // Touching the last sample flags cutoff
    st[8 * samples + (samples - 1)] = 5;
    assert.equal(cutBySectorDepth({ structure: st, polar: { lines, samples } }, [5]), true);
  });
});

describe('Spectral VTI envelope summary & geometry reprojection', () => {
  it('computes VTI, peak/mean gradient on signed CW envelopes and ignores trailing zero columns', () => {
    // Triangular negative ejection jet: 0 -> -4 m/s -> 0 over 0.30 s, padded with 0.40 s of diastole zeros
    const dt = 0.01;
    const jet = [0, -1, -2, -3, -4, -3, -2, -1, 0];
    const padded = [...jet, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const sJet = summarizeEnvelope(jet, dt);
    const sPad = summarizeEnvelope(padded, dt);

    assert.equal(sJet.vmaxMps, 4);
    assert.equal(sJet.peakGradientMmHg, 64);
    // Trapezoidal integral of [0,1,2,3,4,3,2,1,0] * 0.01 s * 100 cm/m = 16 cm
    assert.ok(Math.abs(sJet.vtiCm - 16) < 1e-9);
    // Trailing zeros must NOT dilute mean gradient or alter VTI
    assert.ok(Math.abs(sPad.vtiCm - sJet.vtiCm) < 1e-9);
    assert.ok(Math.abs(sPad.meanGradientMmHg - sJet.meanGradientMmHg) < 1e-9);
    assert.ok(sJet.meanGradientMmHg > 20 && sJet.meanGradientMmHg < 64);
  });

  it('reprojects caliper points accurately across canvas resize, zoom, and L/R inversion', () => {
    const spec = {
      lines: 96,
      samples: 200,
      sectorRad: (80 * Math.PI) / 180,
      depthCm: 16,
      elevationSamples: 1,
      focusCm: 9,
    };
    const m1 = computeSectorMapping(spec, 640, 480, false, 1.0);
    const m2 = computeSectorMapping(spec, 960, 720, true, 1.5);
    const pts = [
      { x: 320, y: 240 },
      { x: 380, y: 310 },
    ];
    const toM2 = reprojectGeometry(pts, m1, m2);
    const back = reprojectGeometry(toM2, m2, m1);
    for (let i = 0; i < pts.length; i++) {
      assert.ok(Math.abs(back[i]!.x - pts[i]!.x) < 1e-6);
      assert.ok(Math.abs(back[i]!.y - pts[i]!.y) < 1e-6);
    }
    // Undefined source mapping returns input unchanged
    assert.deepEqual(reprojectGeometry(pts, undefined, m2), pts);
  });
});
