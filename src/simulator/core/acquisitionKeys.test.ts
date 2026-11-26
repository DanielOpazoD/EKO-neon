import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_COLOR } from '@/simulator/doppler/color/colorDoppler';
import { DEFAULT_SPECTRAL } from '@/simulator/doppler/spectral/spectrum';
import { DEFAULT_ACQUISITION, type PolarFrameSpec } from '@/simulator/renderer/types';
import type { SimInput } from './protocol';
import {
  classifyDopplerAngle,
  DOPPLER_ANGLE_OK_DEG,
  DOPPLER_ANGLE_WARN_DEG,
} from '@/education/technique';
import {
  artifactOverridesKey,
  colorAcquisitionKey,
  colorCpuAcquisitionKey,
  colorGeometryKey,
  patientKey,
  resolveFrozenColorSettings,
  simInputKey,
  stripAcquisitionKey,
} from './acquisitionKeys';

const SPEC: PolarFrameSpec = {
  lines: 112,
  samples: 256,
  sectorRad: Math.PI / 2,
  depthCm: 16,
  focusCm: 8,
  elevationSamples: 1,
};

function makeInput(overrides: Partial<SimInput> = {}): SimInput {
  return {
    probe: { u: 3, v: 0.5, rotationDeg: 20, tiltDeg: 0, rockDeg: 0, pressure: 0.7 },
    patient: { position: 'left-lateral', respiration: 'expiration', headElevationDeg: 0 },
    settings: { ...DEFAULT_ACQUISITION, tgcDb: [...DEFAULT_ACQUISITION.tgcDb] },
    modality: 'color',
    frozen: false,
    cineOffset: 0,
    color: { ...DEFAULT_COLOR },
    spectral: { ...DEFAULT_SPECTRAL },
    cursorThetaRad: 0,
    gateDepthCm: 8,
    quality: 'auto',
    display: { width: 640, height: 480 },
    rendererBackend: 'atlas',
    artifactOverrides: null,
    ...overrides,
  };
}

test('colorAcquisitionKey excludes presentation-only showVariance and palette while tracking acquisition changes', () => {
  const base = { ...DEFAULT_COLOR, showVariance: true, palette: 'vivid' as const };
  const noVar = { ...DEFAULT_COLOR, showVariance: false, palette: 'vivid' as const };
  assert.equal(
    colorAcquisitionKey(base),
    colorAcquisitionKey(noVar),
    'toggling showVariance must not invalidate Kasai acquisition shading',
  );
  for (const palette of ['vivid', 'classic', 'variance', 'ice-fire'] as const) {
    assert.equal(
      colorAcquisitionKey(base),
      colorAcquisitionKey({ ...base, palette }),
      `switching palette to ${palette} must not invalidate Kasai acquisition shading`,
    );
  }

  for (const patch of [
    { scaleMps: 0.9 },
    { baselineShiftMps: 0.2 },
    { wallFilterMps: 0.1 },
    { gainDb: 4 },
    { persistence: 0.6 },
    { smoothing: 0.2 },
    { invert: true },
    { boxRMinCm: 3 },
  ]) {
    assert.notEqual(
      colorAcquisitionKey(base),
      colorAcquisitionKey({ ...base, ...patch }),
      `acquisition change ${JSON.stringify(patch)} must invalidate colorAcquisitionKey`,
    );
  }
});

test('resolveFrozenColorSettings keeps acquisition controls locked while applying live palette, showVariance, and clamped baselineShift', () => {
  const acquired = {
    ...DEFAULT_COLOR,
    scaleMps: 0.68,
    baselineShiftMps: 0,
    wallFilterMps: 0.06,
    gainDb: 0,
    invert: false,
    showVariance: true,
    palette: 'vivid' as const,
  };
  const live = {
    ...acquired,
    scaleMps: 1.2,
    baselineShiftMps: 0.3,
    wallFilterMps: 0.14,
    gainDb: 10,
    invert: true,
    showVariance: false,
    palette: 'ice-fire' as const,
  };
  const resolved = resolveFrozenColorSettings(acquired, live);
  assert.equal(resolved.scaleMps, 0.68, 'scale stays locked to acquired frame');
  assert.equal(resolved.wallFilterMps, 0.06, 'wall filter stays locked to acquired frame');
  assert.equal(resolved.gainDb, 0, 'gain stays locked to acquired frame');
  assert.equal(resolved.invert, false, 'invert stays locked to acquired frame');
  assert.equal(resolved.showVariance, false, 'postprocess showVariance updates immediately when frozen');
  assert.equal(resolved.palette, 'ice-fire', 'postprocess palette updates immediately when frozen');
  assert.equal(resolved.baselineShiftMps, 0.3, 'postprocess baselineShift updates immediately when frozen');

  // Extreme live baseline shift (from a higher live scale) is safely clamped to ±0.85 * acquired.scaleMps
  const clamped = resolveFrozenColorSettings(acquired, { ...live, baselineShiftMps: 0.95 });
  assert.ok(
    Math.abs(clamped.baselineShiftMps - 0.85 * 0.68) < 1e-6,
    `frozen baseline shift must clamp to ±0.85*acqScale, got ${clamped.baselineShiftMps}`,
  );
});

test('stripAcquisitionKey isolates modality-relevant controls and CMM presentation controls so strip history is not wiped', () => {
  const pw = makeInput({ modality: 'pw' });
  const pwColorChanged = makeInput({
    modality: 'pw',
    color: { ...DEFAULT_COLOR, gainDb: 8, scaleMps: 1.1 },
  });
  assert.equal(
    stripAcquisitionKey(pw),
    stripAcquisitionKey(pwColorChanged),
    'changing Color Doppler controls must not reset PW spectral strip history',
  );

  const pwScaleChanged = makeInput({
    modality: 'pw',
    spectral: { ...DEFAULT_SPECTRAL, scaleMps: 1.6 },
  });
  assert.notEqual(stripAcquisitionKey(pw), stripAcquisitionKey(pwScaleChanged));

  const cmm = makeInput({ modality: 'cmm' });
  const cmmSpectralChanged = makeInput({
    modality: 'cmm',
    spectral: { ...DEFAULT_SPECTRAL, gateLengthCm: 0.8, gainDb: 6 },
  });
  assert.equal(
    stripAcquisitionKey(cmm),
    stripAcquisitionKey(cmmSpectralChanged),
    'changing PW spectral controls must not reset CMM strip history',
  );

  const cmmPresentationChanged = makeInput({
    modality: 'cmm',
    color: { ...DEFAULT_COLOR, palette: 'ice-fire', showVariance: false },
  });
  assert.equal(
    stripAcquisitionKey(cmm),
    stripAcquisitionKey(cmmPresentationChanged),
    'changing CMM palette or showVariance must repaint without resetting CMM strip history',
  );
});

test('simInputKey detects every field change deterministically without JSON.stringify', () => {
  const a = makeInput();
  const b = makeInput();
  assert.equal(simInputKey(a), simInputKey(b));

  assert.notEqual(simInputKey(a), simInputKey(makeInput({ frozen: true })));
  assert.notEqual(simInputKey(a), simInputKey(makeInput({ cineOffset: -3 })));
  assert.notEqual(simInputKey(a), simInputKey(makeInput({ needSegments: true })));
  assert.notEqual(simInputKey(a), simInputKey(makeInput({ needStructure: false })));
  assert.notEqual(
    simInputKey(a),
    simInputKey(makeInput({ color: { ...DEFAULT_COLOR, showVariance: false } })),
    'simInputKey must still transmit presentation toggles to worker',
  );
  assert.notEqual(
    simInputKey(a),
    simInputKey(makeInput({ color: { ...DEFAULT_COLOR, palette: 'ice-fire' } })),
    'simInputKey must transmit palette changes to worker',
  );
  assert.notEqual(
    simInputKey(a),
    simInputKey(
      makeInput({
        artifactOverrides: { sideLobe: 0.5, mirror: 0, beamWidth: 0, clutter: 0 },
      }),
    ),
  );

  const g1 = colorGeometryKey(a.probe, a.color, 2.5, true, 1, SPEC);
  const g2 = colorGeometryKey(a.probe, { ...a.color, gainDb: 8 }, 2.5, true, 1, SPEC);
  assert.equal(g1, g2, 'gain change must not invalidate colorGeometryKey');
  assert.notEqual(
    colorCpuAcquisitionKey(a.probe, a.color, 2.5, true, 1, SPEC),
    colorCpuAcquisitionKey(a.probe, { ...a.color, gainDb: 8 }, 2.5, true, 1, SPEC),
  );
  assert.notEqual(
    simInputKey(a),
    simInputKey(makeInput({ spectral: { ...DEFAULT_SPECTRAL, reject: 0.24 } })),
    'simInputKey must transmit spectral.reject postprocess updates to worker',
  );
  assert.equal(
    stripAcquisitionKey(makeInput({ modality: 'pw', spectral: { ...DEFAULT_SPECTRAL, reject: 0 } })),
    stripAcquisitionKey(makeInput({ modality: 'pw', spectral: { ...DEFAULT_SPECTRAL, reject: 0.3 } })),
    'spectral.reject is presentation-only and must not wipe PW/CW/TDI strip history',
  );
  assert.notEqual(patientKey(a.patient), patientKey({ ...a.patient, respiration: 'inspiration' }));
  assert.equal(artifactOverridesKey(null), 'none');
});

test('classifyDopplerAngle enforces canonical 20° / 32° clinical thresholds', () => {
  assert.equal(DOPPLER_ANGLE_OK_DEG, 20);
  assert.equal(DOPPLER_ANGLE_WARN_DEG, 32);
  assert.equal(classifyDopplerAngle(0), 'ok');
  assert.equal(classifyDopplerAngle(20), 'ok');
  assert.equal(classifyDopplerAngle(21), 'warn');
  assert.equal(classifyDopplerAngle(32), 'warn');
  assert.equal(classifyDopplerAngle(33), 'invalid');
  assert.equal(classifyDopplerAngle(28, 30), 'ok');
  assert.equal(classifyDopplerAngle(42, 30), 'warn');
  assert.equal(classifyDopplerAngle(43, 30), 'invalid');
});
