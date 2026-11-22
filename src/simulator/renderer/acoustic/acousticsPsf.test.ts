import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MYO_ANISO_FLOOR,
  myoHelixGain,
  papillaryFibreGain,
  pleuralCoherence,
  pleuralIncidenceCos,
  softTissueTransmission,
} from './acoustics';
import {
  axialFwhmMm,
  buildNoiseKernels,
  buildPsfKernels,
  lateralFwhmMm,
  LATERAL_TAPS,
  MAX_LATERAL_RADIUS,
  psfKey,
  SIDE_LOBE_DB,
  sideLobeLevelDb,
} from './psf';
import type { PolarFrameSpec } from '../types';

const SPEC: PolarFrameSpec = {
  lines: 96,
  samples: 128,
  sectorRad: Math.PI / 2,
  depthCm: 16,
  focusCm: 8,
  elevationSamples: 1,
};

test('myoHelixGain reflects transmural fibre helix orientation (+60° endo, 0° mid, -60° epi)', () => {
  // Mid-wall (u = 0.5) has 0° circumferential fibres:
  // beam along circumferential (dphi = 1, dz = 0) hits the anisotropy floor;
  // beam along long axis (dphi = 0, dz = 1) is perpendicular to fibres → gain = 1.
  assert.ok(Math.abs(myoHelixGain(1, 0, 0.5) - MYO_ANISO_FLOOR) < 1e-6);
  assert.ok(Math.abs(myoHelixGain(0, 1, 0.5) - 1) < 1e-6);

  // From apex (dphi = 0, dz = 1), mid-wall is brighter than subendocardium & subepicardium (Streeter helix)
  const gMid = myoHelixGain(0, 1, 0.5);
  const gEndo = myoHelixGain(0, 1, 0);
  const gEpi = myoHelixGain(0, 1, 1);
  assert.ok(gMid > gEndo && gMid > gEpi);
  assert.ok(Math.abs(gEndo - gEpi) < 1e-6);
});

test('papillaryFibreGain peaks perpendicular to long axis and hits floor parallel to it', () => {
  assert.equal(papillaryFibreGain(0), 1);
  assert.ok(Math.abs(papillaryFibreGain(1) - MYO_ANISO_FLOOR) < 1e-6);
  assert.ok(Math.abs(papillaryFibreGain(-1) - MYO_ANISO_FLOOR) < 1e-6);
  assert.ok(papillaryFibreGain(0.5) > MYO_ANISO_FLOOR && papillaryFibreGain(0.5) < 1);
});

test('pleural incidence & specular coherence follow |n·d|^4 law', () => {
  // Flat pleura perpendicular to beam (dEntry = 0) → cos = 1, coherence = 1
  assert.equal(pleuralIncidenceCos(0, 0.5), 1);
  assert.equal(pleuralCoherence(1), 1);
  // 45° slope (dEntry = arc) → cos = 1/√2, coherence = 0.25
  const c45 = pleuralIncidenceCos(0.5, 0.5);
  assert.ok(Math.abs(c45 - Math.SQRT1_2) < 1e-9);
  assert.ok(Math.abs(pleuralCoherence(c45) - 0.25) < 1e-9);
  // Grazing incidence reduces coherence steeply
  assert.ok(pleuralCoherence(pleuralIncidenceCos(2.0, 0.5)) < 0.01);
});

test('softTissueTransmission decreases monotonically with depth, frequency, and harmonic mode', () => {
  assert.equal(softTissueTransmission(0, 2.5, false), 1);
  const shallow = softTissueTransmission(4, 2.5, false);
  const deep = softTissueTransmission(12, 2.5, false);
  const highFreq = softTissueTransmission(12, 4.0, false);
  const harmonic = softTissueTransmission(12, 2.5, true);

  assert.ok(shallow < 1 && shallow > deep);
  assert.ok(highFreq < deep, 'higher frequency attenuates more strongly');
  assert.ok(harmonic < deep, 'harmonic mode increases effective attenuation');
});

test('PSF resolution formulas obey diffraction physics (frequency, focus, harmonics)', () => {
  // Higher frequency sharpens both axial and lateral resolution
  assert.ok(axialFwhmMm(4.0, false) < axialFwhmMm(2.0, false));
  assert.ok(lateralFwhmMm(8, 8, 4.0, false) < lateralFwhmMm(8, 8, 2.0, false));
  // Harmonics narrow lateral beam while slightly trading axial resolution
  assert.ok(lateralFwhmMm(8, 8, 2.5, true) < lateralFwhmMm(8, 8, 2.5, false));
  assert.ok(axialFwhmMm(2.5, true) > axialFwhmMm(2.5, false));
  // Beam width boost widens beam away from focus more than at focus
  const atFocusBoost = lateralFwhmMm(8, 8, 2.5, false, 1) / lateralFwhmMm(8, 8, 2.5, false, 0);
  const farFocusBoost = lateralFwhmMm(14, 8, 2.5, false, 1) / lateralFwhmMm(14, 8, 2.5, false, 0);
  assert.ok(Math.abs(atFocusBoost - 1) < 1e-9);
  assert.ok(farFocusBoost > 2);
});

test('buildPsfKernels produces symmetric, unit-energy axial and lateral kernels', () => {
  const k = buildPsfKernels(SPEC, 2.5, true, 0, SIDE_LOBE_DB);

  // Axial kernel: symmetric and unit energy (Σw² = 1)
  let axialEnergy = 0;
  for (let i = 0; i < k.axial.length; i++) {
    axialEnergy += k.axial[i]! * k.axial[i]!;
    assert.ok(
      Math.abs(k.axial[i]! - k.axial[k.axial.length - 1 - i]!) < 1e-6,
      'axial kernel must be symmetric',
    );
  }
  assert.ok(Math.abs(axialEnergy - 1) < 1e-5, `axial energy ${axialEnergy} should be 1`);

  // Lateral kernel at every sampled depth: symmetric and unit energy (Σw² = 1)
  for (const si of [0, 32, 64, 127]) {
    const R = k.lateralRadius[si]!;
    const centre = si * LATERAL_TAPS + MAX_LATERAL_RADIUS;
    let energy = k.lateral[centre]! ** 2;
    for (let j = 1; j <= R; j++) {
      const wPos = k.lateral[centre + j]!;
      const wNeg = k.lateral[centre - j]!;
      assert.ok(Math.abs(wPos - wNeg) < 1e-6, `lateral kernel symmetric at si=${si}, j=${j}`);
      energy += wPos * wPos + wNeg * wNeg;
    }
    assert.ok(Math.abs(energy - 1) < 1e-5, `lateral energy at si=${si} (${energy}) should be 1`);
  }

  // Side-lobe artifact raises side-lobe level monotonically and changes cache key
  assert.ok(sideLobeLevelDb(1) > sideLobeLevelDb(0));
  const keyBase = psfKey(SPEC, 2.5, true, 0, sideLobeLevelDb(0));
  const keyArtifact = psfKey(SPEC, 2.5, true, 0, sideLobeLevelDb(1));
  assert.notEqual(keyBase, keyArtifact);

  // Noise kernel excludes side lobes so its lateral radius is strictly smaller than full PSF
  const kNoise = buildNoiseKernels(SPEC, 2.5, true);
  assert.ok(kNoise.lateralRadius[64]! < k.lateralRadius[64]!);
});
