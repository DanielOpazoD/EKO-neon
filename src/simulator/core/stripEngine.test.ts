import { before, describe, it } from 'node:test';
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
import type { StripCtx } from './stripEngine';
import { loadCaseById } from '@/cases';
import { Tissue } from '@/simulator/anatomy/tissue';
import { allocPolarFrame, DEFAULT_ACQUISITION, type Scene } from '@/simulator/renderer/types';
import { softTissueTransmission } from '@/simulator/renderer/acoustic/acoustics';
import {
  allocColorField,
  colorSpectralWidthMps,
  computeColorField,
  DEFAULT_COLOR,
  type ColorAcquisition,
  type ColorSettings,
} from '@/simulator/doppler/color/colorDoppler';
import {
  colorWallFilterCutoffMps,
  colorWallFilterRangeMps,
  WF_MAX_NYQUIST,
  WF_MIN_NYQUIST,
} from '@/simulator/doppler/color/kasaiEstimator';
import { writeCmmColumn } from './strip/cmmColumn';
import {
  DEFAULT_SPECTRAL,
  envelopeThreshold,
  SPECTRAL_BINS,
  spectralDefaultsFor,
} from '@/simulator/doppler/spectral/spectrum';
import { aliasVelocity } from '@/clinical/formulas';
import { constrainAcquisition } from './acquisitionInput';
import {
  applySpectralRejectCurve,
  paintMmodeStripColumn,
  renderSpectralStripRgba,
} from './strip/stripPainter';
import { controlThroughCrux, getViewTarget } from '@/simulator/windows/viewTargets';
import { beamFrameFromPose, poseFromControl } from '@/simulator/probe/pose';
import { buildFlowParams } from '@/simulator/doppler/flow-primitives/flowField';
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

  it('matches CMM invert + baseline shift polarity to 2D Color Doppler and preserves bright leaflet tissue', () => {
    const cols = 4;
    const rows = 20;
    const samples = 20;
    const map = buildRowMap(rows, samples);
    const strip = new Uint8ClampedArray(samples * cols).fill(20);
    // Bright specular mitral anterior leaflet at row 10 (lum = 220)
    for (let c = 0; c < cols; c++) strip[10 * cols + c] = 220;

    const color = {
      ...DEFAULT_COLOR,
      scaleMps: 0.6,
      baselineShiftMps: -0.25,
      invert: true,
      showVariance: true,
    };
    // True flow +0.45 m/s (toward probe): inverted -> -0.45 m/s, inside [-0.85, +0.35], must paint BLUE (B > R)
    const vInvertedAliased = aliasVelocity(-0.45, color.scaleMps, color.baselineShiftMps);
    const cmm = new Float32Array(samples * cols).fill(vInvertedAliased);
    const cmmVar = new Float32Array(samples * cols).fill(0);
    const cmmPow = new Float32Array(samples * cols).fill(1);
    const img = new Uint8ClampedArray(cols * rows * 4);
    paintMmodeStripColumn(1, img, map, strip, cols, samples, undefined, cmm, color, cmmVar, cmmPow);

    const bloodO = (5 * cols + 1) * 4;
    assert.ok(
      img[bloodO + 2]! > img[bloodO]! + 60,
      `Expected inverted positive inflow (+0.45 m/s) to paint BLUE in CMM, got rgb=(${img[bloodO]},${img[bloodO + 1]},${img[bloodO + 2]})`,
    );
    // Bright specular leaflet pixel must keep high grayscale integrity via tissue write priority
    const leafO = (10 * cols + 1) * 4;
    assert.equal(img[leafO], 220, 'Bright mitral leaflet line in CMM should not be overwritten by color');
    assert.equal(img[leafO + 1], 220);
    assert.equal(img[leafO + 2], 220);
  });

  it('keeps the colour wall filter asked for; what filters is it bounded to what 8 pulses allow at the acquired Nyquist', () => {
    const base = {
      modality: 'color' as const,
      gateDepthCm: 8,
      settings: { ...DEFAULT_ACQUISITION },
      spectral: { ...DEFAULT_SPECTRAL },
    };
    const color = { ...DEFAULT_COLOR };
    const kept = constrainAcquisition({ ...base, color });
    assert.equal(kept.color, color, 'a request within the PRF is kept as it is');
    const effective = (c: { color: { wallFilterMps: number; scaleMps: number } }) =>
      colorWallFilterCutoffMps(c.color.wallFilterMps, c.color.scaleMps);
    // below 0.11 Nyquist the mean removal is the lowest cutoff that cancels stationary tissue; above 0.32 nothing here
    // cuts. The request stays as it was: written back bounded, it stayed lowered after any drop of Nyquist
    const low = constrainAcquisition({ ...base, color: { ...DEFAULT_COLOR, wallFilterMps: 0.02 } });
    assert.equal(low.color.wallFilterMps, 0.02);
    assert.ok(Math.abs(effective(low) - WF_MIN_NYQUIST * low.color.scaleMps) < 1e-12);
    const high = constrainAcquisition({
      ...base,
      color: { ...DEFAULT_COLOR, scaleMps: 0.4, wallFilterMps: 0.3 },
    });
    assert.equal(high.color.wallFilterMps, 0.3);
    assert.ok(Math.abs(effective(high) - WF_MAX_NYQUIST * 0.4) < 1e-12);
    // a depth-limited PRF lowers Nyquist, and the cutoff that filters comes down with it
    const deep = constrainAcquisition({
      ...base,
      settings: { ...DEFAULT_ACQUISITION, depthCm: 24 },
      color: { ...DEFAULT_COLOR, scaleMps: 1.2, boxRMaxCm: 22, wallFilterMps: 0.3 },
    });
    assert.ok(deep.color.scaleMps < 1.2, `Nyquist ${deep.color.scaleMps} should be PRF-limited at 22 cm`);
    assert.equal(deep.color.wallFilterMps, 0.3);
    assert.ok(Math.abs(effective(deep) - WF_MAX_NYQUIST * deep.color.scaleMps) < 1e-12);
    // off stays off
    const off = constrainAcquisition({ ...base, color: { ...DEFAULT_COLOR, wallFilterMps: 0 } });
    assert.equal(effective(off), 0);
  });

  describe('colour M-mode wall filter, on the mitral inflow of an A4C through the crux', () => {
    // requests at 0.68 m/s (0.676 acquired at this depth): under the packet's lowest cutoff (bounded up to 7.5 cm/s),
    // inside the range far from both bounds, and the default near its top (21.7 cm/s); at 0.35 m/s, where the inflow
    // aliases: inside the range, and the default bounded down to 0.32 Nyquist (11.2 cm/s). Checked at the default alone,
    // a strip that always filtered with the steepest regression passed
    const CASES: Array<[number, number]> = [
      [0.68, 0.02],
      [0.68, 0.12],
      [0.68, DEFAULT_COLOR.wallFilterMps],
      [0.35, 0.06],
      [0.35, DEFAULT_COLOR.wallFilterMps],
    ];
    /** A traced line as a colour M-mode column reads it: blood axial velocity, σ_v, transmission, clutter and tissue. */
    type CmmLine = {
      velocity: Float32Array | null;
      sigmaV?: Float32Array | null;
      relTrans?: Float32Array | null;
      wallVelocity?: Float32Array | null;
      tissue?: Uint8Array | null;
    };
    type CmmInternals = {
      input: SimInput;
      strips: {
        stripCmm: Float32Array | null;
        stripCmmVar: Float32Array | null;
        stripCmmPow: Float32Array | null;
        stripCols: number;
        writeMmodeColumn(
          col: number,
          src: { lo: CmmLine; hi: CmmLine; t: number },
          phase: number,
          pulses: number,
          spec: { depthCm: number },
          ctx: unknown,
        ): void;
      };
    };
    interface CmmRun {
      request: number;
      /** The Nyquist acquired and the wall filter the core holds. */
      scaleMps: number;
      wallFilterMps: number;
      /** Each column written, with the line its colour was formed from (its last write). */
      lines: Map<number, CmmLine>;
      cmm: Float32Array;
      cmmVar: Float32Array;
      pow: Float32Array;
      cols: number;
      depthCm: number;
      acquisition: ColorAcquisition;
    }
    const runs: CmmRun[] = [];

    before(() => {
      const inp = baseInput({ modality: 'cmm', display: { width: 128, height: 256 } });
      const core = new SimulatorCore(loadCaseById('normal-excellent-window'), inp);
      try {
        const probe = controlThroughCrux(getViewTarget('a4c'), core.models.heart, core.models.thorax, null);
        const { strips } = core as unknown as CmmInternals;
        let lines = new Map<number, CmmLine>();
        let depthCm = 0;
        const write = strips.writeMmodeColumn;
        strips.writeMmodeColumn = function (col, src, phase, pulses, spec, ctx) {
          write.call(this, col, src, phase, pulses, spec, ctx);
          // the strip reads the flow of the nearer of the two traced lines (writeMmodeColumn: t < 0.5 ? lo : hi)
          lines.set(col, src.t < 0.5 ? src.lo : src.hi);
          depthCm = spec.depthCm;
        };
        // a beat first: its traced lines stay cached (the colour is not in their key), so the lines the cases' columns
        // are formed from no longer depend on how many a step's time budget traced
        core.setInput({ ...inp, probe });
        for (let i = 0; i < 30; i++) core.step(1 / 30);
        for (const [scaleMps, request] of CASES) {
          // a new colour acquisition restarts the strip: only the columns written under it are kept
          lines = new Map();
          core.setInput({ ...inp, probe, color: { ...DEFAULT_COLOR, scaleMps, wallFilterMps: request } });
          for (let i = 0; i < 30; i++) core.step(1 / 30);
          const { input } = core as unknown as CmmInternals;
          runs.push({
            request,
            scaleMps: input.color.scaleMps,
            wallFilterMps: input.color.wallFilterMps,
            lines,
            cmm: strips.stripCmm!.slice(),
            cmmVar: strips.stripCmmVar!.slice(),
            pow: strips.stripCmmPow!.slice(),
            cols: strips.stripCols,
            depthCm,
            acquisition: {
              frequencyMHz: input.settings.frequencyMHz,
              harmonics: input.settings.harmonics,
            },
          });
        }
      } finally {
        core.dispose();
      }
    });

    it('runs the full Kasai estimator: stochastic scatter, spectral-width variance, PRF-periodic wall filter, and myocardial clutter flash', () => {
      // 1. Real A4C runs keep the requested wall filter, paint blood flow, and reject non-blood/non-clutter samples
      for (const run of runs) {
        const label = `WF ${run.request} m/s at ${run.scaleMps.toFixed(3)} m/s`;
        assert.equal(run.wallFilterMps, run.request, `${label}: the core keeps the request`);
        let painted = 0;
        let velErrSum = 0;
        let velErrCount = 0;
        for (const [col, line] of run.lines) {
          const v = line.velocity!;
          const ti = line.tissue!;
          for (let si = 0; si < v.length; si++) {
            const i = si * run.cols + col;
            const on = Number.isFinite(run.cmm[i]!);
            const isBlood = ti[si] === Tissue.Blood && Number.isFinite(v[si]!);
            const isClutter = ti[si] === Tissue.Myocardium || ti[si] === Tissue.Valve;
            if (!isBlood && !isClutter) {
              assert.ok(!on, `${label}: colour outside blood/clutter at sample ${si}`);
              continue;
            }
            if (isBlood && on) {
              painted++;
              // Stochastic Kasai scatter: estimated velocity scatters around the true aliased velocity
              const trueAliased = aliasVelocity(v[si]!, run.scaleMps, 0);
              const dv = Math.abs(run.cmm[i]! - trueAliased);
              const wrapDv = Math.min(dv, Math.abs(dv - 2 * run.scaleMps));
              velErrSum += wrapDv;
              velErrCount++;
            }
          }
        }
        assert.ok(painted > 500, `${label}: ${painted} blood samples painted`);
        const meanScatter = velErrSum / Math.max(1, velErrCount);
        assert.ok(
          meanScatter > 0.005 && meanScatter < 0.15,
          `${label}: expected realistic Kasai velocity scatter (0.005..0.15 m/s), got ${meanScatter.toFixed(4)}`,
        );
      }

      // 2. Direct writeCmmColumn checks: turbulent/transverse sigmaV raises Kasai variance, 2x Nyquist is rejected,
      //    and myocardial wall clutter is suppressed at default WF (20 cm/s) while flashing with breakthrough at min WF
      const S = 48;
      const cols = 16;
      const cfg: ColorSettings = { ...DEFAULT_COLOR, scaleMps: 0.68, gainDb: 0, smoothing: 0, persistence: 0 };
      const runSynthetic = (
        line: {
          velocity: Float32Array;
          sigmaV: Float32Array;
          relTrans: Float32Array;
          wallVelocity?: Float32Array;
          tissue?: Uint8Array;
        },
        wfMps: number,
      ) => {
        const cmm = new Float32Array(S * cols).fill(NaN);
        const cmmVar = new Float32Array(S * cols);
        const cmmPow = new Float32Array(S * cols);
        const c = { ...cfg, wallFilterMps: wfMps };
        for (let col = 0; col < cols; col++) {
          writeCmmColumn(
            { amp: new Float32Array(0), ...line },
            S,
            c,
            col + 10,
            { realization: 2, seed: 314 },
            col,
            cols,
            cmm,
            cmmVar,
            cmmPow,
          );
        }
        return { cmm, cmmVar, cmmPow };
      };

      // Laminar vs turbulent/transverse broadening (sigmaV)
      const lam = runSynthetic(
        {
          velocity: new Float32Array(S).fill(0.4),
          sigmaV: new Float32Array(S).fill(colorSpectralWidthMps(0.4, 0.03, 0, 8)),
          relTrans: new Float32Array(S).fill(1),
        },
        DEFAULT_COLOR.wallFilterMps,
      );
      const turb = runSynthetic(
        {
          velocity: new Float32Array(S).fill(0.4),
          sigmaV: new Float32Array(S).fill(colorSpectralWidthMps(0.4, 0.75, 0, 8)),
          relTrans: new Float32Array(S).fill(1),
        },
        DEFAULT_COLOR.wallFilterMps,
      );
      const statsOf = (res: ReturnType<typeof runSynthetic>, s0 = 4, s1 = S - 4) => {
        let on = 0;
        let tot = 0;
        let vSum = 0;
        let maxPow = 0;
        for (let si = s0; si < s1; si++) {
          for (let col = 0; col < cols; col++) {
            tot++;
            const i = si * cols + col;
            if (Number.isFinite(res.cmm[i])) {
              on++;
              vSum += res.cmmVar[i]!;
              maxPow = Math.max(maxPow, res.cmmPow[i]!);
            }
          }
        }
        return { fill: on / tot, meanVar: on ? vSum / on : 0, maxPow };
      };
      const lamS = statsOf(lam);
      const turbS = statsOf(turb);
      assert.ok(lamS.fill > 0.9, `expected laminar CMM fill > 90%, got ${(lamS.fill * 100).toFixed(1)}%`);
      assert.ok(
        turbS.meanVar > lamS.meanVar + 0.15,
        `expected turbulent CMM Kasai variance (${turbS.meanVar.toFixed(3)}) > laminar (${lamS.meanVar.toFixed(3)}) + 0.15`,
      );

      // 2x Nyquist aliased flow rejected by PRF-periodic regression wall filter
      const twoNyq = runSynthetic(
        {
          velocity: new Float32Array(S).fill(2 * cfg.scaleMps),
          sigmaV: new Float32Array(S).fill(0),
          relTrans: new Float32Array(S).fill(1),
        },
        DEFAULT_COLOR.wallFilterMps,
      );
      assert.ok(
        statsOf(twoNyq).fill < 0.05,
        `expected 2x Nyquist aliased flow to be rejected in CMM, got fill=${(statsOf(twoNyq).fill * 100).toFixed(1)}%`,
      );

      // Myocardial clutter (-0.08 m/s): suppressed at default WF (20 cm/s), flashes with breakthrough power > 1 at min WF
      const tissue = new Uint8Array(S).fill(Tissue.Blood);
      const vel = new Float32Array(S).fill(NaN);
      const sig = new Float32Array(S);
      const wall = new Float32Array(S);
      for (let si = 0; si < S; si++) {
        if (si < S / 2) {
          vel[si] = 0.4;
          sig[si] = colorSpectralWidthMps(0.4, 0.03, 0, 8);
        } else {
          tissue[si] = Tissue.Myocardium;
          wall[si] = -0.08;
        }
      }
      const clutterLine = { velocity: vel, sigmaV: sig, relTrans: new Float32Array(S).fill(1), wallVelocity: wall, tissue };
      const wfDefRes = runSynthetic(clutterLine, DEFAULT_COLOR.wallFilterMps);
      const wfMinRes = runSynthetic(clutterLine, colorWallFilterRangeMps(cfg.scaleMps).min);
      const myoDef = statsOf(wfDefRes, S / 2 + 2, S - 2);
      const myoMin = statsOf(wfMinRes, S / 2 + 2, S - 2);
      assert.ok(
        myoDef.fill < 0.05,
        `expected < 5% myocardial clutter flash in CMM at default WF, got ${(myoDef.fill * 100).toFixed(1)}%`,
      );
      assert.ok(
        myoMin.fill > 0.6 && myoMin.maxPow > 1.05,
        `expected myocardial clutter flash (> 60%, breakthrough > 1.05) in CMM at min WF, got fill=${(myoMin.fill * 100).toFixed(1)}% maxPow=${myoMin.maxPow.toFixed(2)}`,
      );
    });

    it('colours the same line as the 2D colour: as much of the blood, from about the same speed', () => {
      // each column's line, replicated over 8 lines, through the 2D colour itself (computeColorField and its Kasai
      // estimate, a new realization per column; persistence and spatial smoothing off, as the strip has neither), over
      // the default box's depths. Both the strip and the 2D colour now run the exact same Kasai estimator and noise
      // coherence gate, so their coloured blood share and velocity onset match directly.
      const NL = 8;
      for (const run of runs) {
        const settings: ColorSettings = {
          ...DEFAULT_COLOR,
          scaleMps: run.scaleMps,
          wallFilterMps: run.wallFilterMps,
          boxThetaMinRad: -0.02,
          boxThetaMaxRad: 0.02,
          persistence: 0,
          smoothing: 0,
        };
        const { frequencyMHz, harmonics } = run.acquisition;
        /** Blood samples in the box: speed, coloured in the strip (0/1), share of the 2D's inner lines coloured. */
        const blood: Array<[number, number, number]> = [];
        let realization = 0;
        for (const [col, line] of run.lines) {
          const v = line.velocity!;
          const sig = line.sigmaV!;
          const rel = line.relTrans!;
          const ti = line.tissue!;
          const F = v.length;
          const frame = allocPolarFrame({
            lines: NL,
            samples: F,
            sectorRad: 0.04,
            depthCm: run.depthCm,
            focusCm: 8,
            elevationSamples: 1,
          });
          for (let si = 0; si < F; si++) {
            if (ti[si] !== Tissue.Blood || !Number.isFinite(v[si]!)) continue;
            const r = ((si + 0.5) / F) * run.depthCm;
            for (let l = 0; l < NL; l++) {
              frame.tissue[l * F + si] = Tissue.Blood;
              frame.transmission[l * F + si] = rel[si]! * softTissueTransmission(r, frequencyMHz, harmonics);
            }
          }
          const field = allocColorField(NL * F);
          computeColorField(
            frame,
            settings,
            null,
            (_idx, _li, si, res) => {
              const vs = v[si]!;
              res.v = vs;
              res.disp = Math.abs(vs) > 1e-4 ? sig[si]! / (0.9 * Math.abs(vs)) : 0;
              res.vPerp = 0;
              res.present = 1;
            },
            field,
            run.acquisition,
            { realization: ++realization, seed: 4242 },
          );
          for (let si = 0; si < F; si++) {
            const r = ((si + 0.5) / F) * run.depthCm;
            if (ti[si] !== Tissue.Blood || !Number.isFinite(v[si]!) || r < settings.boxRMinCm || r > settings.boxRMaxCm)
              continue;
            // the inner lines, away from the edges of the estimator's 3 × 3 cells
            let coloured2d = 0;
            for (let l = 2; l < NL - 2; l++) if (Number.isFinite(field.vel[l * F + si]!)) coloured2d++;
            const strip = Number.isFinite(run.cmm[si * run.cols + col]!) ? 1 : 0;
            blood.push([Math.abs(v[si]!), strip, coloured2d / (NL - 4)]);
          }
        }
        const cut = colorWallFilterCutoffMps(run.wallFilterMps, run.scaleMps);
        const label = `WF ${run.request} m/s at ${run.scaleMps.toFixed(3)} m/s (cutoff ${(cut * 100).toFixed(1)} cm/s)`;
        assert.ok(blood.length > 1000, `${label}: ${blood.length} blood samples`);
        const share = (k: 1 | 2) => blood.reduce((s, b) => s + b[k], 0) / blood.length;
        assert.ok(
          Math.abs(share(1) - share(2)) < 0.04,
          `${label}: the strip colours ${(100 * share(1)).toFixed(1)} % of the blood, the 2D ${(100 * share(2)).toFixed(1)} %`,
        );
        const principal = blood.filter((b) => b[0] < 0.8 * run.scaleMps).sort((a, b) => a[0] - b[0]);
        const onset = (k: 1 | 2) => {
          // all called coloured, the dark ones are the errors; each sample moved below the onset swaps its error
          let errors = principal.reduce((s, b) => s + 1 - b[k], 0);
          let best = errors;
          let at = 0;
          for (let i = 0; i < principal.length; i++) {
            errors += 2 * principal[i]![k] - 1;
            if (errors < best) {
              best = errors;
              at = principal[i + 1]?.[0] ?? Infinity;
            }
          }
          return at;
        };
        const onsetDiff = Math.abs(onset(2) - onset(1)) / cut;
        assert.ok(
          onsetDiff < 0.15,
          `${label}: the strip's colour begins at ${(onset(1) * 100).toFixed(1)} cm/s, the 2D's at ${(onset(2) * 100).toFixed(1)} (diff ${(onsetDiff * 100).toFixed(1)}% of cutoff)`,
        );
      }
    });
  });

  it('clamps baselineShiftMps under depth-limited PRF and guards renderSpectralStripRgba against B-mode bleed', () => {
    const constrained = constrainAcquisition({
      modality: 'pw',
      gateDepthCm: 16,
      settings: { ...DEFAULT_ACQUISITION, depthCm: 18, frequencyMHz: 4.0 },
      color: { ...DEFAULT_COLOR, scaleMps: 0.9, baselineShiftMps: 0.75 },
      spectral: { ...DEFAULT_SPECTRAL, scaleMps: 1.5, baselineShiftMps: -1.4 },
    });
    assert.ok(
      Math.abs(constrained.color.baselineShiftMps) <= 0.85 * constrained.color.scaleMps + 1e-6,
      `Color baselineShiftMps (${constrained.color.baselineShiftMps}) exceeded 0.85 * scale (${constrained.color.scaleMps})`,
    );
    assert.ok(
      Math.abs(constrained.spectral.baselineShiftMps) <= 0.85 * constrained.spectral.scaleMps + 1e-6,
      `Spectral baselineShiftMps (${constrained.spectral.baselineShiftMps}) exceeded 0.85 * scale (${constrained.spectral.scaleMps})`,
    );

    // Extreme off-range baseline (vMax < 0) must never paint the green baseline into the 2D B-mode region above y0
    const W = 32;
    const H = 64;
    const y0 = 30;
    const h = 30;
    const rgba = new Uint8ClampedArray(W * H * 4);
    renderSpectralStripRgba(
      rgba,
      W,
      H,
      y0,
      h,
      new Float32Array(W * SPECTRAL_BINS),
      W,
      { ...DEFAULT_SPECTRAL, scaleMps: 0.5, baselineShiftMps: -0.9 },
    );
    for (let y = 0; y < y0; y++) {
      for (let x = 0; x < W; x++) {
        const o = (y * W + x) * 4;
        assert.equal(rgba[o + 1], 0, `Green baseline bled into B-mode sector at row ${y}`);
      }
    }
  });

  it('applySpectralRejectCurve zeroes low noise, stays monotonic, and preserves bright spectral peaks', () => {
    assert.equal(applySpectralRejectCurve(0.24, 0.25), 0);
    assert.equal(applySpectralRejectCurve(0.25, 0.25), 0);
    assert.equal(applySpectralRejectCurve(0, 0), 0);
    assert.equal(applySpectralRejectCurve(1, 0.25), 1);

    let prev = 0;
    for (let i = 0; i <= 100; i++) {
      const y = applySpectralRejectCurve(i / 100, 0.25);
      assert.ok(y >= prev - 1e-9, `reject curve lost monotonicity at x=${i / 100}`);
      prev = y;
    }
    const peakByte = applySpectralRejectCurve(0.92, 0.25) * 255;
    assert.ok(
      peakByte > 200,
      `Expected bright spectral peak (0.92) to stay > 200/255 at reject=0.25, got ${peakByte.toFixed(1)}`,
    );
  });

  it('starts a new strip acquisition when the heart changes (its ray caches know nothing of the anatomy)', () => {
    // the experimental volume never arrives here: the change of anatomy alone must start over
    const core = new SimulatorCore(loadCaseById('aortic-stenosis-severe'), baseInput({ modality: 'cw' }), {
      fetchAsset: () => new Promise(() => {}),
    });
    try {
      let out = null;
      for (let i = 0; i < 6; i++) out = core.step(1 / 30) ?? out;
      assert.ok(out && out.strip.kind === 'spectral');
      const analytic = out.strip.acquisitionId;
      core.setInput(baseInput({ modality: 'cw', anatomy: 'rodero' }));
      let rodero = null;
      for (let i = 0; i < 3; i++) rodero = core.step(1 / 30) ?? rodero;
      assert.ok(rodero);
      assert.notEqual(rodero.strip.acquisitionId, analytic, 'a CW strip of the other heart kept going');
    } finally {
      core.dispose();
    }
  });

  it('applies spectral.reject immediately live and frozen without resetting strip acquisitionId', () => {
    const caseDef = loadCaseById('normal-excellent-window');
    const core = new SimulatorCore(
      caseDef,
      baseInput({ modality: 'pw', spectral: { ...DEFAULT_SPECTRAL, gainDb: 14, reject: 0 } }),
    );
    try {
      const a4cPose = controlThroughCrux(
        getViewTarget('a4c'),
        core.models.heart,
        core.models.thorax,
        null,
      );
      const pwLive = baseInput({
        modality: 'pw',
        probe: a4cPose,
        gateDepthCm: 8.5,
        spectral: { ...DEFAULT_SPECTRAL, gainDb: 14, reject: 0 },
      });
      core.setInput(pwLive);
      let out0 = null;
      for (let i = 0; i < 8; i++) out0 = core.step(1 / 30);
      assert.ok(out0 && out0.strip.kind === 'spectral');
      const acqIdBefore = out0.strip.acquisitionId;

      // Freeze and raise reject to 0.30: strip history must survive and dark pixels must increase
      core.setInput({ ...pwLive, frozen: true });
      const frozen0 = core.step(1 / 30);
      assert.ok(frozen0 && frozen0.strip.kind === 'spectral');
      const rgba0 = new Uint8ClampedArray(frozen0.rgba);

      core.setInput({
        ...pwLive,
        frozen: true,
        spectral: { ...pwLive.spectral, reject: 0.3 },
      });
      const frozenRej = core.step(1 / 30);
      assert.ok(frozenRej && frozenRej.strip.kind === 'spectral');
      assert.equal(
        frozenRej.strip.acquisitionId,
        acqIdBefore,
        'changing spectral.reject while frozen must NOT reset strip acquisitionId',
      );

      const rgbaRej = new Uint8ClampedArray(frozenRej.rgba);
      const stripStart = frozen0.strip.y * frozen0.width * 4;
      let darkBefore = 0;
      let darkAfter = 0;
      for (let i = stripStart; i < rgba0.length; i += 4) {
        if (rgba0[i] === 0 && rgba0[i + 1] === 0 && rgba0[i + 2] === 0) darkBefore++;
        if (rgbaRej[i] === 0 && rgbaRej[i + 1] === 0 && rgbaRej[i + 2] === 0) darkAfter++;
      }
      assert.ok(
        darkAfter > darkBefore + 100,
        `Raising spectral.reject to 0.30 while frozen should zero noisy background pixels (darkBefore=${darkBefore}, darkAfter=${darkAfter})`,
      );
    } finally {
      core.dispose();
    }
  });

  it('computes CW beam-flow angle along the ray on severe aortic stenosis (A5C)', () => {
    const caseDef = loadCaseById('aortic-stenosis-severe');
    const core = new SimulatorCore(caseDef, baseInput({ modality: 'cw' }));
    try {
      const a5cPose = controlThroughCrux(
        getViewTarget('a5c'),
        core.models.heart,
        core.models.thorax,
        null,
      );
      const beam = beamFrameFromPose(poseFromControl(core.models.thorax, a5cPose), 1);
      const hf = core.models.heart.frame;
      const av = buildFlowParams(
        caseDef,
        core.models.heart,
        core.models.tables,
        core.models.thorax,
      ).avAxis;
      const expectedAngleAt = (thetaRad: number) => {
        const ct = Math.cos(thetaRad);
        const sn = Math.sin(thetaRad);
        const dx = beam.forward.x * ct + beam.lateral.x * sn;
        const dy = beam.forward.y * ct + beam.lateral.y * sn;
        const dz = beam.forward.z * ct + beam.lateral.z * sn;
        const dhx = dx * hf.ex.x + dy * hf.ex.y + dz * hf.ex.z;
        const dhy = dx * hf.ey.x + dy * hf.ey.y + dz * hf.ey.z;
        const dhz = dx * hf.ez.x + dy * hf.ez.y + dz * hf.ez.z;
        const cos = Math.abs(av.x * dhx + av.y * dhy + av.z * dhz);
        return (Math.acos(Math.min(1, cos)) * 180) / Math.PI;
      };

      // Center A5C ray intersects the stenotic AV jet and reports its exact geometric angle vs avAxis
      core.setInput(
        baseInput({
          modality: 'cw',
          probe: a5cPose,
          cursorThetaRad: 0,
        }),
      );
      const outCenter = core.step(1 / 30);
      assert.ok(outCenter && outCenter.gate !== null, 'CW mode should emit GateInfo');
      assert.equal(outCenter.gate.flowPresent, true, 'CW ray in A5C should intersect blood flow');
      const expectedCenter = expectedAngleAt(0);
      assert.ok(
        outCenter.gate.flowAngleDeg !== null &&
          Number.isFinite(outCenter.gate.flowAngleDeg) &&
          Math.abs(outCenter.gate.flowAngleDeg - expectedCenter) < 2.5,
        `Expected CW A5C center ray angle near geometric avAxis (${expectedCenter.toFixed(1)}°), got ${outCenter.gate.flowAngleDeg}`,
      );

      // Steering the CW cursor -5° toward the aortic orifice aligns the ray tighter (< 25°)
      const steeredTheta = (-5 * Math.PI) / 180;
      core.setInput(
        baseInput({
          modality: 'cw',
          probe: a5cPose,
          cursorThetaRad: steeredTheta,
        }),
      );
      const outSteered = core.step(0.1);
      assert.ok(outSteered && outSteered.gate !== null && outSteered.gate.flowAngleDeg !== null);
      assert.ok(
        outSteered.gate.flowAngleDeg < 25,
        `Expected steered CW A5C jet angle < 25°, got ${outSteered.gate.flowAngleDeg}`,
      );
    } finally {
      core.dispose();
    }
  });
});

describe('Spectral strips in free breathing', () => {
  const breathing = {
    position: 'left-lateral',
    respiration: 'free-breathing',
    headElevationDeg: 0,
  } as const;

  it('acquires each column in the thorax the 2D images at that instant', () => {
    const caseDef = loadCaseById('normal-excellent-window');
    const inp = baseInput({
      modality: 'pw',
      patient: breathing,
      gateDepthCm: 9.42,
      cursorThetaRad: (3.7 * Math.PI) / 180,
    });
    const core = new SimulatorCore(caseDef, inp);
    try {
      const probe = controlThroughCrux(
        getViewTarget('a4c'),
        core.models.heart,
        core.models.thorax,
        null,
      );
      core.setInput({ ...inp, probe });
      const internals = core as unknown as {
        stripCtx(): StripCtx;
        scene(phase: number): Scene;
        timeS: number;
      };
      let lo = Infinity;
      let hi = -Infinity;
      // 0.9 s into the first inspiration (1.6 s long): the lung border slides 0.69 cm meanwhile
      for (let i = 0; i < 9; i++) {
        core.step(0.1);
        const t = internals.timeS;
        const spectral = internals.stripCtx().moment(t).scene.thorax;
        const image = internals.scene(core.currentPhase).thorax;
        assert.deepEqual(
          [spectral.lungShiftCm, spectral.diaphragmRiseCm],
          [image.lungShiftCm, image.diaphragmRiseCm],
          `lung border and diaphragm of the spectral column and of the 2D at ${t.toFixed(3)} s`,
        );
        lo = Math.min(lo, spectral.lungShiftCm);
        hi = Math.max(hi, spectral.lungShiftCm);
      }
      assert.ok(hi - lo > 0.5, `the lung border should slide with the breath, moved ${(hi - lo).toFixed(2)} cm`);
    } finally {
      core.dispose();
    }
  });

  it('loses the window to the lung in inspiration in the spectrum as in the 2D (PW and CW)', () => {
    const caseDef = loadCaseById('normal-excellent-window');
    const gateCm = 7;
    for (const modality of ['pw', 'cw'] as const) {
      const inp = baseInput({
        modality,
        patient: breathing,
        spectral: spectralDefaultsFor(modality),
        gateDepthCm: gateCm,
      });
      const core = new SimulatorCore(caseDef, inp);
      try {
        // The apical window slid 1.5 cm laterally: its ray clears the lingula in expiration and meets it under the chest
        // wall from about half-way into inspiration (0.85 s into the breath), in the 2D along the cursor as in Doppler.
        const a4c = controlThroughCrux(
          getViewTarget('a4c'),
          core.models.heart,
          core.models.thorax,
          null,
        );
        core.setInput({ ...inp, probe: { ...a4c, u: a4c.u + 1.5 } });
        // whether each 2D frame shows lung at the gate on both lines either side of the cursor (θ = 0); null if they differ
        const frames: Array<{ t: number; lung: boolean | null }> = [];
        let frameId = -1;
        for (let i = 0; i < 19; i++) {
          const out = core.step(0.1);
          const f = core.lastFrame;
          if (!out || !f || out.frameId === frameId) continue;
          frameId = out.frameId;
          const { lines, samples, depthCm } = f.spec;
          const si = Math.floor((gateCm / depthCm) * samples);
          const [a, b] = [Math.floor((lines - 1) / 2), Math.ceil((lines - 1) / 2)].map(
            (li) => f.tissue[li * samples + si] === Tissue.Lung,
          );
          frames.push({ t: out.timeS, lung: a === b ? a! : null });
        }
        const strip = core.spectralStrip;
        assert.ok(strip.data && strip.head < strip.cols, 'the 1.9 s strip should not have wrapped');
        // a column takes the 2D's verdict only inside a stretch where the two frames on each side of it agree
        const verdictAt = (t: number): boolean | null => {
          const k = frames.findIndex((f) => f.t > t);
          if (k < 2 || k + 2 > frames.length) return null;
          const near = frames.slice(k - 2, k + 2).map((f) => f.lung);
          return near.every((s) => s === near[0]) ? near[0]! : null;
        };
        // a column has a spectrum when a bin clears the envelope threshold, 1.5× the brightest bin the noise alone draws
        const noiseFloor = envelopeThreshold(0, inp.spectral);
        const behindLung: boolean[] = [];
        const open: boolean[] = [];
        for (let c = 0; c < strip.head; c++) {
          let peak = 0;
          for (let k = 0; k < SPECTRAL_BINS; k++)
            peak = Math.max(peak, strip.data[c * SPECTRAL_BINS + k]!);
          const lung = verdictAt(strip.sampleTimeS[c]!);
          if (lung === true) behindLung.push(peak > noiseFloor);
          else if (lung === false) open.push(peak > noiseFloor);
        }
        assert.ok(
          behindLung.length > 30 && open.length > 30,
          `${modality}: the 2D should keep the window early in the breath and lose it later (${open.length} open, ${behindLung.length} behind lung)`,
        );
        const heard = open.filter(Boolean).length;
        assert.ok(
          heard > 0.2 * open.length,
          `${modality}: only ${heard} of ${open.length} columns had a spectrum through the open window`,
        );
        const leaked = behindLung.filter(Boolean).length;
        assert.equal(
          leaked,
          0,
          `${modality}: ${leaked} of ${behindLung.length} columns kept a spectrum while the 2D showed lung over the gate`,
        );
      } finally {
        core.dispose();
      }
    }
  });
});
