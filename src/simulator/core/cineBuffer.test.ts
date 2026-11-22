import assert from 'node:assert/strict';
import test from 'node:test';
import { v3 } from '@/core/vec3';
import { DEFAULT_COLOR } from '@/simulator/doppler/color/colorDoppler';
import { DEFAULT_SPECTRAL } from '@/simulator/doppler/spectral/spectrum';
import { DEFAULT_ACQUISITION, type Scene } from '@/simulator/renderer/types';
import { CineBuffer, type CineFrame } from './cineBuffer';

function makeCineFrame(frameId: number, phase: number, timeS: number, fillByte: number): CineFrame {
  return {
    acquisition: {
      modality: '2d',
      probe: { u: 3, v: 0, rotationDeg: 25, tiltDeg: 5, rockDeg: 0, pressure: 0.6 },
      patient: { position: 'left-lateral', respiration: 'expiration', headElevationDeg: 15 },
      settings: { ...DEFAULT_ACQUISITION, tgcDb: [...DEFAULT_ACQUISITION.tgcDb] },
      color: { ...DEFAULT_COLOR },
      spectral: { ...DEFAULT_SPECTRAL },
      cursorThetaRad: 0,
      gateDepthCm: 8,
      modelVersion: 1,
      tablesVersion: 1,
      phaseMarks: {
        ejectionStart: 0.06,
        ejectionEnd: 0.35,
        endSystole: 0.35,
        mitralOpen: 0.43,
        eEnd: 0.65,
        aStart: 0.82,
        aEnd: 0.96,
        hasAWave: true,
      },
      stripAcquisitionId: null,
      colorTimeS: null,
    },
    scene: {} as Scene,
    gate: null,
    spectrumColumn: null,
    spectralRange: { vMin: -1, vMax: 1 },
    display: new Uint8ClampedArray([fillByte, fillByte + 1, fillByte + 2, fillByte + 3]),
    structure: new Uint8Array([1, 2, 3, 4]),
    segment: new Uint8Array([0, 1, 0, 2]),
    spec: { lines: 2, samples: 2, sectorRad: 1.4, depthCm: 16, focusCm: 8, elevationSamples: 1 },
    frameId,
    rrS: 1.0,
    phase,
    timeS,
    beatIndex: Math.floor(timeS),
    colorVel: null,
    colorVar: null,
    beam: {
      origin: v3(0, 0, 0),
      forward: v3(0, 0, -1),
      lateral: v3(1, 0, 0),
      normal: v3(0, 1, 0),
      contact: 1,
    },
    analysis: null,
  };
}

test('CineBuffer enforces capacity ring eviction, clamps offsets, and isolates captured arrays', () => {
  assert.throws(() => new CineBuffer(0), RangeError);

  const buf = new CineBuffer(4);
  assert.equal(buf.length, 0);
  assert.equal(buf.oldest === undefined, true);
  assert.equal(buf.newest === undefined, true);
  assert.equal(buf.findNearestPhaseOffset(0), null);

  const src = makeCineFrame(1, 0.1, 0.1, 50);
  buf.capture(src);
  // Mutating caller buffers after capture must not corrupt the stored cine frame
  src.display[0] = 255;
  src.acquisition.probe.u = 99;
  assert.equal(buf.newest?.display[0], 50);
  assert.equal(buf.newest?.acquisition.probe.u, 3);

  for (let i = 2; i <= 6; i++) {
    buf.capture(makeCineFrame(i, (i * 0.15) % 1, i * 0.15, i * 10));
  }

  assert.equal(buf.length, 4);
  assert.equal(buf.oldest?.frameId, 3);
  assert.equal(buf.newest?.frameId, 6);

  assert.equal(buf.select(0)?.frameId, 6);
  assert.equal(buf.select(-1)?.frameId, 5);
  assert.equal(buf.select(-3)?.frameId, 3);
  assert.equal(buf.select(-99)?.frameId, 3);
  assert.equal(buf.select(10)?.frameId, 6);
  assert.equal(buf.select(Number.NaN)?.frameId, 6);
});

test('CineBuffer.findNearestPhaseOffset handles circular wraparound and prefers the latest beat', () => {
  const buf = new CineBuffer(16);
  // Beat 0: phases 0.02, 0.35, 0.70, 0.98
  buf.capture(makeCineFrame(1, 0.02, 0.02, 1));
  buf.capture(makeCineFrame(2, 0.35, 0.35, 2));
  buf.capture(makeCineFrame(3, 0.7, 0.7, 3));
  buf.capture(makeCineFrame(4, 0.98, 0.98, 4));
  // Beat 1: phases 0.02, 0.35, 0.70
  buf.capture(makeCineFrame(5, 0.02, 1.02, 5));
  buf.capture(makeCineFrame(6, 0.35, 1.35, 6));
  buf.capture(makeCineFrame(7, 0.7, 1.7, 7));

  // Target ES = 0.35 appears in both beat 0 (offset -5) and beat 1 (offset -1); latest beat wins
  const es = buf.findNearestPhaseOffset(0.35);
  assert.deepEqual(es, { offset: -1, phase: 0.35 });

  // Target ED = 0.00 wraps circularly across 0/1: beat 1's 0.02 (offset -2) beats beat 0's 0.98 (offset -3)
  const ed = buf.findNearestPhaseOffset(0.0);
  assert.deepEqual(ed, { offset: -2, phase: 0.02 });
});
