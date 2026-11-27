import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import type { SimOutput } from '@/simulator/core/protocol';
import { captureStepPrompt, computeLivePreview, minPointsToFinish } from './measurementPreview';

const MOCK_2D_HUD: Pick<SimOutput, 'frozen' | 'sector' | 'strip' | 'acquisition'> = {
  frozen: true,
  sector: {
    x: 0,
    y: 0,
    apexX: 250,
    apexY: 20,
    pxPerCm: 25,
    depthCm: 16,
    sectorRad: 1.4,
    width: 500,
    height: 440,
    invertLR: false,
  },
  strip: {
    kind: null,
    x: 0,
    y: 440,
    width: 500,
    height: 0,
    topValue: 0,
    bottomValue: 0,
    secondsPerColumn: 0.005,
    headColumn: 0,
    columns: 0,
  },
  acquisition: undefined,
};

const MOCK_SPECTRAL_HUD: Pick<SimOutput, 'frozen' | 'sector' | 'strip' | 'acquisition'> = {
  frozen: true,
  sector: { ...MOCK_2D_HUD.sector, height: 220 },
  strip: {
    kind: 'spectral',
    x: 0,
    y: 220,
    width: 500,
    height: 200,
    topValue: 2,
    bottomValue: -2,
    secondsPerColumn: 0.005,
    headColumn: 250,
    columns: 500,
  },
  acquisition: undefined,
};

const MOCK_MMODE_HUD: Pick<SimOutput, 'frozen' | 'sector' | 'strip' | 'acquisition'> = {
  frozen: true,
  sector: { ...MOCK_2D_HUD.sector, height: 220 },
  strip: {
    kind: 'm-mode',
    x: 0,
    y: 220,
    width: 500,
    height: 200,
    topValue: 0,
    bottomValue: 16,
    secondsPerColumn: 0.005,
    headColumn: 250,
    columns: 500,
  },
  acquisition: undefined,
};

describe('live measurement rubber-band preview & step prompts', () => {
  it('returns null when live (unfrozen) or when pointer is outside the tool zone', () => {
    assert.equal(
      computeLivePreview(
        'caliper',
        null,
        [{ x: 200, y: 150 }],
        { x: 300, y: 150 },
        { ...MOCK_2D_HUD, frozen: false },
        '2d',
      ),
      null,
    );
    assert.equal(
      computeLivePreview('velocity', null, [], { x: 250, y: 100 }, MOCK_SPECTRAL_HUD, 'pw'),
      null,
    );
  });

  it('computes exact Caliper distance, TAPSE excursion, and Time interval previews', () => {
    // 100 px / 25 px/cm = 4.00 cm
    const cal = computeLivePreview(
      'caliper',
      'lv-edd',
      [{ x: 200, y: 180 }],
      { x: 300, y: 180 },
      MOCK_2D_HUD,
      '2d',
    );
    assert.equal(cal?.shape, 'line');
    assert.equal(cal?.label, '4.00 cm');

    // 25 px over 200 px height × 16 cm = 2.00 cm
    const tapse = computeLivePreview(
      'tapse',
      'tapse',
      [{ x: 180, y: 320 }],
      { x: 220, y: 295 },
      MOCK_MMODE_HUD,
      'm-mode',
    );
    assert.equal(tapse?.shape, 'line');
    assert.equal(tapse?.label, '2.00 cm');

    // 40 cols × 0.005 s = 200 ms
    const time = computeLivePreview(
      'time',
      null,
      [{ x: 100, y: 300 }],
      { x: 140, y: 310 },
      MOCK_SPECTRAL_HUD,
      'pw',
    );
    assert.equal(time?.label, '200 ms');
  });

  it('computes spectral velocity (m/s & TDI cm/s), deceleration time / PHT, and auto-VTI span', () => {
    // y = 270 on [220..420] spanning [2..-2] -> v = 1.00 m/s, ΔP = 4 mmHg
    const velPw = computeLivePreview(
      'velocity',
      'av-vmax',
      [],
      { x: 200, y: 270 },
      MOCK_SPECTRAL_HUD,
      'cw',
    );
    assert.equal(velPw?.shape, 'guide');
    assert.equal(velPw?.label, '1.00 m/s · ΔP 4 mmHg');

    const velTdi = computeLivePreview(
      'velocity',
      'e-prime-septal',
      [],
      { x: 200, y: 325 },
      MOCK_SPECTRAL_HUD,
      'tdi',
    );
    assert.equal(velTdi?.label, '10 cm/s');

    // Slope: peak at y=270 (1.0 m/s, x=100), slope at y=295 (0.5 m/s, x=120 -> 100 ms later) -> DT = 200 ms, PHT = 58 ms
    const dt = computeLivePreview(
      'slope',
      'mitral-dt',
      [{ x: 100, y: 270 }],
      { x: 120, y: 295 },
      MOCK_SPECTRAL_HUD,
      'pw',
    );
    assert.equal(dt?.shape, 'ray');
    assert.equal(dt?.label, 'TD 200 ms');

    const pht = computeLivePreview(
      'slope',
      'ar-pht',
      [{ x: 100, y: 270 }],
      { x: 120, y: 295 },
      MOCK_SPECTRAL_HUD,
      'cw',
    );
    assert.equal(pht?.label, 'PHT 58 ms');

    const span = computeLivePreview(
      'auto-vti',
      'lvot-vti',
      [{ x: 100, y: 300 }],
      { x: 160, y: 310 },
      MOCK_SPECTRAL_HUD,
      'pw',
    );
    assert.equal(span?.shape, 'span');
    assert.equal(span?.label, 'Ventana 300 ms');
  });

  it('computes live Simpson LV and RV contour volume readouts once 5 vertices exist', () => {
    const pts = [
      { x: 200, y: 260 },
      { x: 195, y: 190 },
      { x: 240, y: 100 },
      { x: 285, y: 190 },
    ];
    const lv = computeLivePreview('simpson', 'lv-edv-simpson', pts, { x: 280, y: 260 }, MOCK_2D_HUD, '2d');
    assert.equal(lv?.shape, 'contour');
    assert.match(lv?.label ?? '', /\d+ mL$/);

    const rv = computeLivePreview('simpson', 'rv-edv', pts, { x: 280, y: 260 }, MOCK_2D_HUD, '2d');
    assert.match(rv?.label ?? '', /\d+ mL · \d+\.\d cm²$/);
  });

  it('produces actionable step prompts for modality mismatch, unfrozen image, and multi-point traces', () => {
    assert.match(captureStepPrompt('auto-vti', 0, false, false, 'PW'), /Cambia a PW/);
    assert.match(captureStepPrompt('caliper', 0, false, true, null), /Congela/);
    assert.match(captureStepPrompt('caliper', 0, true, true, null), /Paso 1\/2/);
    assert.match(captureStepPrompt('caliper', 1, true, true, null), /Paso 2\/2/);
    assert.match(captureStepPrompt('simpson', 5, true, true, null), /Enter/);
    assert.equal(minPointsToFinish('simpson'), 5);
    assert.equal(minPointsToFinish('vti'), 2);
  });
});
