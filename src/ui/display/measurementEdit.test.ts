import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import type { SimOutput } from '@/simulator/core/protocol';
import type { Measurement } from '@/simulator/measurements/types';
import {
  editableVertexCount,
  hitTestMeasurementVertex,
  recomputeEditedMeasurement,
} from './measurementEdit';

function makeHud(overrides: Partial<SimOutput> = {}): SimOutput {
  return {
    width: 480,
    height: 380,
    rgba: new ArrayBuffer(0),
    frameId: 42,
    phase: 0,
    timeS: 2.0,
    rrS: 0.85,
    beatIndex: 2,
    frozen: true,
    cineOffset: 0,
    cineLength: 60,
    cineWindow: { startS: 0, endS: 2.0, frameS: 2.0 },
    polar: {
      lines: 64,
      samples: 64,
      depthCm: 16,
      sectorRad: 1.4,
    },
    sector: {
      x: 0,
      y: 0,
      apexX: 240,
      apexY: 16,
      pxPerCm: 20,
      depthCm: 16,
      sectorRad: 1.4,
      width: 480,
      height: 360,
      invertLR: false,
    },
    strip: {
      kind: null,
      x: 0,
      y: 200,
      width: 480,
      height: 180,
      topValue: 1.0,
      bottomValue: -1.0,
      secondsPerColumn: 0.005,
      columns: 480,
      headColumn: 200,
      headTimeS: 2.0,
      acquisitionId: 7,
    },
    ecg: new Float64Array(0),
    ecgHead: 0,
    heartRateBpm: 70,
    simulatedFps: 50,
    cadenceHz: 30,
    spectrumColumn: null,
    spectralRange: { vMin: -1, vMax: 1 },
    cineFramePhase: 0,
    colorFps: 0,
    probeBeam: {
      origin: [0, 0, 0],
      forward: [0, 0, 1],
      lateral: [1, 0, 0],
      normal: [0, 1, 0],
    },
    structure: new Uint8Array(64 * 64),
    segment: new Uint8Array(0),
    gate: null,
    view: null,
    stats: {},
    ...overrides,
  };
}

function makeMeasurement(partial: Partial<Measurement>): Measurement {
  return {
    id: 'm1',
    kind: 'linear',
    measurementId: null,
    technique: null,
    label: 'Distancia',
    value: 4.0,
    units: 'cm',
    modality: '2d',
    sourceViewId: 'plax',
    viewScore: 95,
    frameId: 42,
    phase: 0,
    timeS: 2.0,
    geometry: [
      { x: 200, y: 120 },
      { x: 280, y: 120 },
    ],
    captureSector: makeHud().sector,
    imageQualityScore: 90,
    userAssisted: false,
    referenceGuidelineIds: [],
    createdAt: '2026-01-01T00:00:00Z',
    ...partial,
  };
}

const OPTS = {
  phaseMarks: null,
  lvLengthCm: 8.5,
  color: { scaleMps: 0.68, baselineShiftMps: 0 },
};

describe('direct drag-to-edit on frozen measurements', () => {
  it('hit-tests editable caliper handles on frozen frames and skips live frames or dense auto-contours', () => {
    const hud = makeHud();
    const caliper = makeMeasurement({});
    const autoContour = makeMeasurement({
      id: 'auto-vol',
      kind: 'volume',
      geometry: Array.from({ length: 40 }, (_, i) => ({ x: 200 + i, y: 120 })),
    });

    assert.equal(editableVertexCount(caliper), 2);
    assert.equal(editableVertexCount(autoContour), 0);

    // Hit endpoint 1 within radius
    const hit = hitTestMeasurementVertex({ x: 284, y: 123 }, [caliper, autoContour], hud, '2d');
    assert.ok(hit);
    assert.equal(hit.measurementId, 'm1');
    assert.equal(hit.vertexIndex, 1);
    assert.equal(hit.space, 'sector');

    // Miss outside radius
    assert.equal(hitTestMeasurementVertex({ x: 240, y: 120 }, [caliper], hud, '2d'), null);

    // Disabled when image is live (not frozen)
    assert.equal(
      hitTestMeasurementVertex({ x: 280, y: 120 }, [caliper], makeHud({ frozen: false }), '2d'),
      null,
    );
  });

  it('recomputes 2D caliper distance live when dragging an endpoint', () => {
    const hud = makeHud();
    const caliper = makeMeasurement({
      geometry: [
        { x: 200, y: 120 },
        { x: 280, y: 120 },
      ],
      value: 4.0,
    });

    // Drag point 1 by +20 px -> 100 px / 20 pxPerCm = 5.0 cm
    const edited = recomputeEditedMeasurement(caliper, 1, { x: 300, y: 120 }, hud, OPTS);
    assert.ok(edited);
    assert.equal(edited.vertexIndex, 1);
    assert.ok(Math.abs((edited.patch.value ?? 0) - 5.0) < 1e-6);
    assert.deepEqual(edited.patch.geometry, [
      { x: 200, y: 120 },
      { x: 300, y: 120 },
    ]);
  });

  it('recomputes spectral velocity & Bernoulli gradient, M-mode TAPSE, and deceleration slope live', () => {
    // Spectral velocity peak
    const specHud = makeHud({
      strip: {
        kind: 'spectral',
        x: 0,
        y: 200,
        width: 480,
        height: 180,
        topValue: 2.0,
        bottomValue: -2.0,
        secondsPerColumn: 0.005,
        columns: 480,
        headColumn: 200,
        headTimeS: 2.0,
        acquisitionId: 7,
      },
    });
    const velMs = makeMeasurement({
      id: 'v1',
      kind: 'velocity',
      modality: 'cw',
      units: 'm/s',
      value: 1.0,
      captureSector: undefined,
      captureStrip: specHud.strip,
      geometry: [{ x: 150, y: 245 }], // y=245 -> v = +1.0 m/s
    });
    // Drag to y=200 (topValue = +2.0 m/s -> ΔP = 16 mmHg)
    const velEdit = recomputeEditedMeasurement(velMs, 0, { x: 160, y: 200 }, specHud, OPTS);
    assert.ok(velEdit);
    assert.ok(Math.abs((velEdit.patch.value ?? 0) - 2.0) < 1e-6);
    assert.ok(Math.abs((velEdit.patch.derived?.gradientMmHg ?? 0) - 16.0) < 1e-6);

    // Spectral deceleration slope (3-point: only vertices 0 & 1 editable, intercept recomputed)
    const slopeMs = makeMeasurement({
      id: 's1',
      kind: 'time',
      modality: 'pw',
      units: 'ms',
      value: 200,
      captureSector: undefined,
      captureStrip: specHud.strip,
      geometry: [
        { x: 100, y: 245 }, // v = 1.0 m/s
        { x: 120, y: 267.5 }, // v = 0.5 m/s -> 20 cols * 2 = 40 cols * 5ms = 200 ms
        { x: 140, y: 290 }, // baseline intercept
      ],
    });
    assert.equal(editableVertexCount(slopeMs), 2);
    // Drag slope point right to x=130 -> 30 cols half-drop -> 60 cols * 5 ms = 300 ms
    const slopeEdit = recomputeEditedMeasurement(slopeMs, 1, { x: 130, y: 267.5 }, specHud, OPTS);
    assert.ok(slopeEdit);
    assert.ok(Math.abs((slopeEdit.patch.value ?? 0) - 300) < 1e-4);
    assert.equal(slopeEdit.patch.geometry?.length, 3);
    assert.ok(Math.abs(slopeEdit.patch.geometry![2]!.x - 160) < 1e-4);

    // Invalid drag (slope point above peak) returns null cleanly
    assert.equal(recomputeEditedMeasurement(slopeMs, 1, { x: 130, y: 210 }, specHud, OPTS), null);

    // M-mode TAPSE excursion
    const mmHud = makeHud({
      strip: {
        ...specHud.strip,
        kind: 'm-mode',
        topValue: 0,
        bottomValue: 18,
      },
    });
    const tapseMs = makeMeasurement({
      id: 't1',
      kind: 'linear',
      modality: 'm-mode',
      units: 'cm',
      value: 1.8,
      captureSector: undefined,
      captureStrip: mmHud.strip,
      geometry: [
        { x: 120, y: 280 },
        { x: 140, y: 298 },
      ],
    });
    const tapseEdit = recomputeEditedMeasurement(tapseMs, 1, { x: 140, y: 304 }, mmHud, OPTS);
    assert.ok(tapseEdit);
    // |304 - 280| / 180 * 18 = 2.4 cm
    assert.ok(Math.abs((tapseEdit.patch.value ?? 0) - 2.4) < 1e-6);
  });
});
