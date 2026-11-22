import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { StripTimeGrid } from './stripTimeGrid';
import {
  MmodeLineCache,
  binsToTrace,
  buildRowMap,
  columnSource,
  mmodeLineSamples,
  mmodePhaseBins,
  mmodePulsesPerColumn,
} from './mmodeStrip';
import { SimulatorCore } from './simulatorCore';
import { loadCaseById } from '@/cases';
import { DEFAULT_ACQUISITION } from '@/simulator/renderer/types';
import { DEFAULT_COLOR } from '@/simulator/doppler/color/colorDoppler';
import { DEFAULT_SPECTRAL } from '@/simulator/doppler/spectral/spectrum';
import type { SimInput } from './protocol';

function baseInput(overrides: Partial<SimInput> = {}): SimInput {
  return {
    probe: { u: 3.0, v: 0.5, rotationDeg: 20, tiltDeg: 0, rockDeg: 0, pressure: 0.7 },
    patient: { position: 'left-lateral', respiration: 'expiration', headElevationDeg: 0 },
    settings: { ...DEFAULT_ACQUISITION },
    modality: 'm-mode',
    frozen: false,
    cineOffset: 0,
    color: { ...DEFAULT_COLOR },
    spectral: { ...DEFAULT_SPECTRAL },
    cursorThetaRad: 0,
    gateDepthCm: 8,
    quality: 'low',
    display: { width: 256, height: 256 },
    rendererBackend: 'procedural',
    artifactOverrides: null,
    ...overrides,
  };
}

describe('StripTimeGrid', () => {
  it('emits monotonic contiguous cell-centre timestamps independent of step jitter', () => {
    const g = new StripTimeGrid();
    const cps = 100; // 10 ms per column
    const t1 = g.advance(0.035, 0.035, cps); // 3 columns due: 0.005, 0.015, 0.025
    assert.equal(t1.length, 3);
    assert.ok(Math.abs(t1[0]! - 0.005) < 1e-9);
    assert.ok(Math.abs(t1[1]! - 0.015) < 1e-9);
    assert.ok(Math.abs(t1[2]! - 0.025) < 1e-9);

    const t2 = g.advance(0.06, 0.025, cps); // 3 more columns due: 0.035, 0.045, 0.055
    assert.equal(t2.length, 3);
    assert.ok(Math.abs(t2[0]! - 0.035) < 1e-9);
    assert.ok(Math.abs(t2[2]! - 0.055) < 1e-9);

    g.reset();
    const t3 = g.advance(1.02, 0.02, cps);
    assert.equal(t3.length, 2);
    assert.ok(Math.abs(t3[0]! - 1.005) < 1e-9);
  });
});

describe('M-mode line cache, phase bin scheduler and row resampling', () => {
  it('computes valid M-mode sampling parameters', () => {
    assert.equal(mmodeLineSamples(16), 800);
    assert.ok(mmodeLineSamples(0.5) >= 64);

    // Phase bins are powers of two within [128, 2048]
    const bins = mmodePhaseBins(200, 0.9);
    assert.equal(bins & (bins - 1), 0);
    assert.ok(bins >= 128 && bins <= 2048);
    assert.equal(mmodePhaseBins(1, 0.5), 128);
    assert.equal(mmodePhaseBins(5000, 2), 2048);

    assert.equal(mmodePulsesPerColumn(100), 10);
    assert.equal(mmodePulsesPerColumn(5000), 1);
  });

  it('interpolates M-mode columns across wrapped phase bins and budgets missing traces evenly', () => {
    const cache = new MmodeLineCache();
    cache.reset('k', 128);
    const l0 = { amp: new Float32Array([1, 2]), velocity: null };
    const l4 = { amp: new Float32Array([5, 6]), velocity: null };
    cache.set(126, l0);
    cache.set(2, l4);
    assert.equal(cache.filled, 2);

    // Phase 0 maps to bin 0, halfway between bin 126 (-2) and bin 2 (+2) -> span = 4, t = 0.5
    const src = columnSource(cache, 0, 16);
    assert.ok(src);
    assert.equal(src.lo, l0);
    assert.equal(src.hi, l4);
    assert.equal(src.span, 4);
    assert.ok(Math.abs(src.t - 0.5) < 1e-9);

    // binsToTrace spreads limited budget over missing bins
    const phases = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    const picked = binsToTrace(cache, phases, 3);
    assert.equal(picked.length, 3);
    assert.equal(new Set(picked).size, 3);
  });

  it('buildRowMap preserves unit weight sum for both downsampling and upsampling', () => {
    for (const [rows, samples] of [
      [120, 800], // downsampling (typical M-mode display)
      [300, 128], // upsampling
      [160, 160], // 1:1
    ] as const) {
      const map = buildRowMap(rows, samples);
      assert.equal(map.rows, rows);
      assert.equal(map.samples, samples);
      for (let y = 0; y < rows; y++) {
        let sum = 0;
        for (let k = 0; k < map.taps; k++) {
          const w = map.weights[y * map.taps + k]!;
          assert.ok(w >= -1e-6 && w <= 1 + 1e-6);
          sum += w;
        }
        assert.ok(
          Math.abs(sum - 1) < 1e-4,
          `Row ${y} (${rows}<-${samples}) weights sum to ${sum}, expected 1`,
        );
      }
    }
  });
});

describe('StripEngine integration via SimulatorCore (M-mode, PW spectral & chroma tint)', () => {
  it('produces valid M-mode and PW spectral strips and applies chroma tints cleanly', () => {
    const caseDef = loadCaseById('normal-excellent-window');
    const core = new SimulatorCore(caseDef, baseInput({ modality: 'm-mode' }));
    try {
      let out = null;
      for (let i = 0; i < 6; i++) out = core.step(1 / 30);
      assert.ok(out);
      assert.equal(out.strip.kind, 'm-mode');
      assert.equal(out.strip.topValue, 0);
      assert.equal(out.strip.bottomValue, DEFAULT_ACQUISITION.depthCm);
      assert.ok(out.strip.headColumn > 0);

      // Switch to sepia tint and verify non-neutral RGB pixels appear in the M-mode strip
      core.setInput(
        baseInput({
          modality: 'm-mode',
          settings: { ...DEFAULT_ACQUISITION, tintMap: 'sepia' },
        }),
      );
      const tinted = core.step(1 / 30);
      assert.ok(tinted);
      const rgba = new Uint8ClampedArray(tinted.rgba);
      let warmCount = 0;
      const stripStart = tinted.strip.y * tinted.width * 4;
      for (let i = stripStart; i < rgba.length; i += 4) {
        const r = rgba[i]!;
        const b = rgba[i + 2]!;
        if (r > b + 6) warmCount++;
      }
      assert.ok(warmCount > 20, `Expected warm tinted strip pixels, found ${warmCount}`);

      // Switch to PW spectral Doppler
      core.setInput(baseInput({ modality: 'pw' }));
      let pwOut = null;
      for (let i = 0; i < 4; i++) pwOut = core.step(1 / 30);
      assert.ok(pwOut);
      assert.equal(pwOut.strip.kind, 'spectral');
      assert.ok(pwOut.strip.topValue > pwOut.strip.bottomValue);
      assert.ok(pwOut.gate !== null);
    } finally {
      core.dispose();
    }
  });
});
