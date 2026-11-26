import type { CaseDefinition } from '@/cases/schema';
import { aliasVelocity, aseSweepDurationS } from '@/clinical/formulas';
import type { HeartModel } from '@/simulator/anatomy/heartModel';
import { makeSample, Tissue } from '@/simulator/anatomy/tissue';
import type { BeatTables } from '@/simulator/cardiac-cycle/cycleModel';
import {
  COLOR_THRESHOLD,
  DOPPLER_SHADOW_TRANSMISSION,
  relativeTransmission,
  wallFilterResponse,
  type ColorSettings,
} from '@/simulator/doppler/color/colorDoppler';
import {
  sampleFlow,
  type FlowFieldParams,
  type FlowSample,
} from '@/simulator/doppler/flow-primitives/flowField';
import {
  presentSpectralColumn,
  SPECTRAL_BINS,
  spectralRange,
  type SpectralSettings,
} from '@/simulator/doppler/spectral/spectrum';
import { valveClickSignal } from '@/simulator/doppler/spectral/valveClicks';
import { DUPLEX_BMODE_SHARE } from '@/simulator/renderer/pulseTiming';
import {
  createSpectralFrameTiming,
  resetSpectralFrameTiming,
  type SpectralFrameTiming,
} from '@/simulator/doppler/dopplerPerf';
import { depthLimitedPulsedAcquisition } from '@/simulator/doppler/spectral/scatteringModel';
import { pulsedAcquisition } from '@/simulator/renderer/pulseTiming';
import type { BeamFrame } from '@/simulator/probe/pose';
import { MODALITIES } from '@/simulator/renderer/modality';
import {
  applyConsole,
  createConsoleState,
  NO_ARTIFACTS,
} from '@/simulator/renderer/postprocess/consolePipeline';
import type { ProceduralSliceRenderer } from '@/simulator/renderer/procedural/sliceRenderer';
import type { PolarFrame, PolarFrameSpec, Scene, ScenePhysics, TintMapId } from '@/simulator/renderer/types';
import {
  binsToTrace,
  buildRowMap,
  columnSource,
  MmodeLineCache,
  mmodeLineSamples,
  mmodePhaseBins,
  mmodePulsesPerColumn,
  type ColumnSource,
  type MmodeLine,
  type RowMap,
} from './mmodeStrip';
import type { GateInfo, SimInput, SimRequest, SimResponse, StripInfo } from './protocol';
import { autoTraceSpectralEnvelope, STRIP_MM_WIDTH } from './strip/autoTrace';
import { computeGateInfo, sampleSpectralColumn, type GateFlowCache } from './strip/spectralSampler';
import { PATH_STEP_CM, relativeEchoAmplitude } from '@/simulator/doppler/acousticAccess';
import {
  buildSpectralRowMap,
  drawSweepMarker,
  paintMmodeStripColumn,
  paintSpectralStripColumn,
  renderSpectralStripRgba,
  type SpectralRowMap,
} from './strip/stripPainter';
import { StripTimeGrid } from './stripTimeGrid';
import { resolveFrozenColorSettings, stripAcquisitionKey } from './acquisitionKeys';

/**
 * How finely the per-bin propagation caches tell breathing thoraxes apart (cm): half the step the central ray is marched
 * in, so a cached ray misplaces the lung border or the diaphragm by less than the march itself resolves. Keyed to
 * 0.01 mm, the breathing lung changed the key on 90% of the columns (320 columns/s) and the caches served 6% of the PW
 * rays and 4% of the CW spectra; at half a step 53% and 31%, as with the static thorax (56% and 32%: free breathing
 * renews the beat tables, and with them the caches, every beat).
 */
const BREATHING_KEY_CM = PATH_STEP_CM / 2;

/** Core services the strip engine reads per call (the core rebuilds it; mutable state lives on the engine). */
export interface StripCtx {
  input: SimInput;
  caseDef: CaseDefinition;
  heart: HeartModel;
  tables: BeatTables;
  flow: FlowFieldParams;
  timeS: number;
  modelVersion: number;
  tablesVersion: number;
  /** Cycle phase and RR at the moment of the call. */
  phaseNow: number;
  rrS: number;
  procedural: ProceduralSliceRenderer;
  moment(timeS: number): {
    phase: number;
    beatIndex: number;
    tablesVersion: number;
    tables: BeatTables;
    flow: FlowFieldParams;
    scene: Scene;
  };
  scene(phase: number): Scene;
  physics(): ScenePhysics;
}

/**
 * The strip machinery of the simulator: spectral (PW/CW/TDI) and M-mode/colour-M-mode columns, their displayed
 * image and the gate/auto-trace read-outs over them. Columns are written at their own instants (decision 115);
 * M-mode columns are formed from lines traced per phase bin (decision 84).
 */
export class StripEngine {
  private stripSpectral: Float32Array | null = null;
  /** What the screen shows of the spectral strip (decision 115): the estimate with its grain; the envelope reads `stripSpectral`. */
  private stripDisplay: Float32Array | null = null;
  /** colour M-mode: aliased axial velocity per (sample, column), NaN where no flow */
  private stripCmm: Float32Array | null = null;
  private stripCmmVar: Float32Array | null = null;
  private stripCmmPow: Float32Array | null = null;
  private stripMmode: Uint8ClampedArray | null = null;
  private stripCols = 0;
  private stripHead = 0;
  private acquisitionId = 0;
  private headTimeS: number | undefined;
  private timeline = new StripTimeGrid();
  private stripSampleTime = new Float64Array(0);
  private stripBeatIndex = new Uint32Array(0);
  private stripTablesVersion = new Uint32Array(0);
  private lastColumn: Float32Array | null = null;
  private stripKind: 'spectral' | 'm-mode' | null = null;
  private acquisitionKey = '';
  /** Calibration belongs to the stored signal, never to pending console input. */
  private acquired: {
    spectral: SpectralSettings;
    color: ColorSettings;
    depthCm: number;
    gateDepthCm: number;
    frequencyMHz: number;
    aliasing: boolean;
  } | null = null;
  private displayCols = 0;
  private lineAmp = new Float32Array(0);
  private lineSt = new Uint8Array(0);
  private lineTr = new Float32Array(0);
  private lineTi = new Uint8Array(0);
  /** M-mode (decision 84): traced lines per phase bin, line samples, and the strip columns' instants and bin spans. */
  private mmode = new MmodeLineCache();
  private mmodeSamples = 0;
  private stripPhase = new Float32Array(0);
  private stripSpan = new Uint16Array(0);
  private mmodeColumn = new Float32Array(0);
  private mmodeGrey = new Uint8ClampedArray(0);
  private mmodeFrame: PolarFrame | null = null;
  private mmodeConsole = createConsoleState(0);
  /** The strip as displayed (RGBA, strip rows × columns), repainted one column at a time; its layout key. */
  private stripRgba: Uint8ClampedArray | null = null;
  private stripRgbaKey = '';
  private spectralRgba: Uint8ClampedArray | null = null;
  private spectralRgbaKey = '';
  private spectralRowMap: SpectralRowMap | null = null;
  private lastSpectralReject = 0;
  private lastSpectralYbRel = -1;
  private spectralColScratch = new Float32Array(SPECTRAL_BINS);
  private spectralDispScratch = new Float32Array(SPECTRAL_BINS);
  /** Per-phase-bin central-ray attenuation cache (128 bins/cycle) keyed by pose, physics, model & tables. */
  private rayAccessKey = '';
  private rayAccessBins: Array<((depthCm: number) => number) | undefined> = new Array(128);
  /** Per-phase-bin raw CW spectrum cache (192 bins/cycle, ~4.4 ms/bin) keyed by pose, depth, CW scale/WF/invert, model & tables. */
  private cwRawKey = '';
  private cwRawBins: Array<
    | {
        rawColumn: Float32Array;
        clickPoints: number[];
        clickPowers: number[];
        rayHeartDir: { x: number; y: number; z: number };
      }
    | undefined
  > = new Array(192);
  private lastTintMap: TintMapId | undefined = undefined;
  private rowMap: RowMap | null = null;
  /** Peak flow vector over the cycle at the last gate position (heart frame), keyed by that position. */
  private gateFlow: GateFlowCache = { key: '', best: 0, bx: 0, by: 0, bz: 0 };
  private flowSample: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };
  private tissueSample = makeSample();
  private stepTiming: SpectralFrameTiming = createSpectralFrameTiming();
  private cmmColorMs = 0;

  /** Reset per-step Doppler timing accumulators at the beginning of a simulation step. */
  resetStepTiming(): void {
    resetSpectralFrameTiming(this.stepTiming);
    this.cmmColorMs = 0;
  }

  get lastStepTiming(): Readonly<SpectralFrameTiming> {
    return this.stepTiming;
  }

  get lastCmmColorMs(): number {
    return this.cmmColorMs;
  }

  /** Reset on a modality or acquisition-calibration change; old columns must not acquire new units. */
  reset(): void {
    this.acquisitionId++;
    this.headTimeS = undefined;
    this.stripHead = 0;
    this.timeline.reset();
    this.stripKind = null;
    this.lastColumn = null;
    this.acquired = null;
    this.spectralRgba = null;
    this.spectralRgbaKey = '';
    this.rayAccessKey = '';
    this.cwRawKey = '';
  }

  get historyId(): number {
    return this.acquisitionId;
  }

  private effectiveSpectral(acquired: NonNullable<StripEngine['acquired']>): SpectralSettings {
    const scaleMps = acquired.aliasing
      ? depthLimitedPulsedAcquisition(
          acquired.gateDepthCm,
          acquired.spectral.gateLengthCm,
          acquired.frequencyMHz,
          acquired.spectral.scaleMps,
        ).effectiveScaleMps
      : acquired.spectral.scaleMps;
    const maxShift = 0.85 * Math.max(0.01, scaleMps);
    const baselineShiftMps = Math.max(-maxShift, Math.min(maxShift, acquired.spectral.baselineShiftMps));
    return scaleMps !== acquired.spectral.scaleMps ||
      baselineShiftMps !== acquired.spectral.baselineShiftMps
      ? { ...acquired.spectral, scaleMps, baselineShiftMps }
      : acquired.spectral;
  }

  private effectiveColor(acquired: NonNullable<StripEngine['acquired']>): ColorSettings {
    const scaleMps = pulsedAcquisition(
      acquired.depthCm,
      acquired.frequencyMHz,
      acquired.color.scaleMps,
    ).scaleMps;
    const maxShift = 0.85 * Math.max(0.01, scaleMps);
    const baselineShiftMps = Math.max(-maxShift, Math.min(maxShift, acquired.color.baselineShiftMps));
    return scaleMps !== acquired.color.scaleMps ||
      baselineShiftMps !== acquired.color.baselineShiftMps
      ? { ...acquired.color, scaleMps, baselineShiftMps }
      : acquired.color;
  }

  get velocityRange(): { vMin: number; vMax: number } | null {
    return this.acquired && this.stripKind === 'spectral'
      ? spectralRange(this.effectiveSpectral(this.acquired))
      : null;
  }

  get spectrumColumn(): Float32Array | null {
    return this.lastColumn;
  }

  /**
   * The M-mode strip as stored (decision 84): grey levels indexed [sample × columns + column] over `samples` line samples,
   * the cycle phase of each column, the phase bins between the two traced lines it was formed from (1 at full time
   * resolution) and the head (next column to write). Null outside M-mode.
   */
  get mmodeStrip(): {
    samples: number;
    cols: number;
    head: number;
    grey: Uint8ClampedArray;
    phase: Float32Array;
    span: Uint16Array;
    bins: number;
    traced: number;
  } | null {
    if (this.stripKind !== 'm-mode' || !this.stripMmode) return null;
    return {
      samples: this.mmodeSamples,
      cols: this.stripCols,
      head: this.stripHead,
      grey: this.stripMmode,
      phase: this.stripPhase,
      span: this.stripSpan,
      bins: this.mmode.bins,
      traced: this.mmode.filled,
    };
  }

  /** The spectral strip (SPECTRAL_BINS values per column), the cycle phase each column was sampled at, and the head. */
  get spectralStrip(): {
    data: Float32Array | null;
    display: Float32Array | null;
    cols: number;
    head: number;
    phase: Float32Array;
    sampleTimeS: Float64Array;
    beatIndex: Uint32Array;
    tablesVersion: Uint32Array;
  } {
    return {
      data: this.stripSpectral,
      display: this.stripDisplay,
      cols: this.stripCols,
      head: this.stripHead,
      phase: this.stripPhase,
      sampleTimeS: this.stripSampleTime,
      beatIndex: this.stripBeatIndex,
      tablesVersion: this.stripTablesVersion,
    };
  }

  advance(
    dt: number,
    beam: BeamFrame,
    spec: PolarFrameSpec,
    traceBudgetMs: number,
    ctx: StripCtx,
  ): void {
    const inp = ctx.input;
    const key = stripAcquisitionKey(inp);
    if (key !== this.acquisitionKey || !this.acquired) {
      this.reset();
      this.acquisitionKey = key;
      this.acquired = {
        spectral: { ...inp.spectral },
        color: { ...inp.color },
        depthCm: spec.depthCm,
        gateDepthCm: inp.gateDepthCm,
        frequencyMHz: inp.settings.frequencyMHz,
        aliasing: MODALITIES[inp.modality].aliasing,
      };
    }
    const kind: 'spectral' | 'm-mode' = MODALITIES[inp.modality].strip ?? 'spectral';
    const stripWidth = Math.max(64, inp.display.width);
    const secondsShown = aseSweepDurationS(inp.spectral.sweepSpeedMmPerS, STRIP_MM_WIDTH);
    const cps = stripWidth / secondsShown;
    const lineSamples = mmodeLineSamples(spec.depthCm);
    const cmm = inp.modality === 'cmm';
    const needMmodeRealloc =
      kind === 'm-mode' &&
      (!this.stripMmode ||
        this.stripMmode.length !== lineSamples * stripWidth ||
        (this.stripCmm !== null) !== cmm ||
        (this.stripCmm !== null && this.stripCmm.length !== spec.samples * stripWidth));
    if (this.stripKind !== kind || this.stripCols !== stripWidth || needMmodeRealloc) {
      this.acquisitionId++;
      this.headTimeS = undefined;
      this.stripKind = kind;
      this.stripCols = stripWidth;
      this.stripHead = 0;
      this.stripSpectral =
        kind === 'spectral' ? new Float32Array(SPECTRAL_BINS * stripWidth) : null;
      this.stripDisplay = kind === 'spectral' ? new Float32Array(SPECTRAL_BINS * stripWidth) : null;
      this.stripMmode = kind === 'm-mode' ? new Uint8ClampedArray(lineSamples * stripWidth) : null;
      this.stripCmm = cmm ? new Float32Array(spec.samples * stripWidth).fill(NaN) : null;
      this.stripCmmVar = cmm ? new Float32Array(spec.samples * stripWidth) : null;
      this.stripCmmPow = cmm ? new Float32Array(spec.samples * stripWidth) : null;
      this.timeline.reset();
      this.stripSampleTime = new Float64Array(stripWidth);
      this.stripBeatIndex = new Uint32Array(stripWidth);
      this.stripTablesVersion = new Uint32Array(stripWidth);
      this.stripPhase = new Float32Array(stripWidth);
      this.stripSpan = new Uint16Array(stripWidth);
      this.stripRgba = null;
      this.stripRgbaKey = '';
      this.spectralRgba = null;
      this.spectralRgbaKey = '';
      this.mmodeSamples = lineSamples;
    }
    const times = this.timeline.advance(ctx.timeS, dt, cps);
    if (kind === 'm-mode') {
      const tMm = cmm ? performance.now() : 0;
      this.advanceMmode(times, cps, ctx.rrS, ctx.phaseNow, traceBudgetMs, beam, spec, ctx);
      if (cmm) {
        this.cmmColorMs += performance.now() - tMm;
        this.stepTiming.columns += times.length;
      }
      return;
    }
    const theta = Math.max(-spec.sectorRad / 2, Math.min(spec.sectorRad / 2, inp.cursorThetaRad));
    const ct = Math.cos(theta);
    const sn = Math.sin(theta);
    const rayDir = {
      x: beam.forward.x * ct + beam.lateral.x * sn,
      y: beam.forward.y * ct + beam.lateral.y * sn,
      z: beam.forward.z * ct + beam.lateral.z * sn,
    };
    for (const timeS of times) {
      const tMoment = performance.now();
      const moment = ctx.moment(timeS);
      this.stepTiming.momentSetupMs += performance.now() - tMoment;
      const phase = moment.phase;
      const col = this.stripHead % this.stripCols;
      const sc = moment.scene;
      const rayKey = [
        beam.origin.x.toFixed(4),
        beam.origin.y.toFixed(4),
        beam.origin.z.toFixed(4),
        rayDir.x.toFixed(5),
        rayDir.y.toFixed(5),
        rayDir.z.toFixed(5),
        beam.contact.toFixed(4),
        sc.physics.frequencyMHz,
        (sc.physics.windowAttenuation ?? 0).toFixed(4),
        Math.round(sc.thorax.lungShiftCm / BREATHING_KEY_CM),
        Math.round(sc.thorax.diaphragmRiseCm / BREATHING_KEY_CM),
        ctx.modelVersion,
        moment.tablesVersion,
      ].join('|');
      if (this.rayAccessKey !== rayKey) {
        this.rayAccessKey = rayKey;
        this.rayAccessBins.fill(undefined);
      }
      const rayBin = Math.floor((((phase % 1) + 1) % 1) * 128) & 127;
      let received = this.rayAccessBins[rayBin];
      if (!received) {
        received = relativeEchoAmplitude(sc, beam.origin, rayDir, beam.contact);
        this.rayAccessBins[rayBin] = received;
      }
      let signalColumn: Float32Array;
      let signalDisplay: Float32Array;
      const isCw = inp.modality === 'cw';
      if (isCw) {
        const cwKey = `${rayKey}|${spec.depthCm.toFixed(2)}|${spec.focusCm.toFixed(2)}|${inp.spectral.scaleMps}|${inp.spectral.baselineShiftMps}|${inp.spectral.wallFilterMps}|${inp.spectral.invert ? 1 : 0}`;
        if (this.cwRawKey !== cwKey) {
          this.cwRawKey = cwKey;
          this.cwRawBins.fill(undefined);
        }
        const cwBin = Math.floor((((phase % 1) + 1) % 1) * 192) % 192;
        const cached = this.cwRawBins[cwBin];
        if (cached) {
          this.spectralColScratch.set(cached.rawColumn);
          const tClicks = performance.now();
          const clickSig = cached.clickPoints.length
            ? valveClickSignal(
                sc.heart,
                sc.heartPose,
                moment.tables,
                phase * moment.tables.rrS,
                cached.clickPoints,
                cached.rayHeartDir,
                cached.clickPowers,
              )
            : { weight: 0, direction: 0 };
          this.stepTiming.valveClicksMs += performance.now() - tClicks;
          const tPres = performance.now();
          presentSpectralColumn(
            this.spectralColScratch,
            inp.spectral,
            this.stripHead,
            ctx.caseDef.seed,
            inp.settings.frequencyMHz,
            1 - DUPLEX_BMODE_SHARE,
            clickSig.weight,
            this.spectralDispScratch,
            timeS,
            true,
            clickSig.direction,
          );
          this.stepTiming.spectrumPresentMs += performance.now() - tPres;
          this.stepTiming.columns += 1;
          signalColumn = this.spectralColScratch;
          signalDisplay = this.spectralDispScratch;
        } else {
          const signal = sampleSpectralColumn(
            beam,
            spec,
            phase,
            col,
            this.stripHead,
            timeS,
            {
              ...ctx,
              heart: sc.heart,
              tables: moment.tables,
              flow: moment.flow,
              scene: () => sc,
            },
            this.flowSample,
            this.tissueSample,
            this.stepTiming,
            {
              received,
              outColumn: this.spectralColScratch,
              outDisplay: this.spectralDispScratch,
              onRawCw: (rawColumn, clickPoints, clickPowers, rayHeartDir) => {
                this.cwRawBins[cwBin] = {
                  rawColumn: rawColumn.slice(),
                  clickPoints: clickPoints.slice(),
                  clickPowers: clickPowers.slice(),
                  rayHeartDir,
                };
              },
            },
          );
          signalColumn = signal.column;
          signalDisplay = signal.display;
        }
      } else {
        const signal = sampleSpectralColumn(
          beam,
          spec,
          phase,
          col,
          this.stripHead,
          timeS,
          {
            ...ctx,
            heart: sc.heart,
            tables: moment.tables,
            flow: moment.flow,
            scene: () => sc,
          },
          this.flowSample,
          this.tissueSample,
          this.stepTiming,
          {
            received,
            outColumn: this.spectralColScratch,
            outDisplay: this.spectralDispScratch,
          },
        );
        signalColumn = signal.column;
        signalDisplay = signal.display;
      }
      this.stripSpectral!.set(signalColumn, col * SPECTRAL_BINS);
      this.stripDisplay!.set(signalDisplay, col * SPECTRAL_BINS);
      this.lastColumn = signalColumn;
      this.stripPhase[col] = phase;
      this.stripSampleTime[col] = timeS;
      this.stripBeatIndex[col] = moment.beatIndex;
      this.stripTablesVersion[col] = moment.tablesVersion;
      this.headTimeS = timeS;
      this.stripHead++;
      if (this.spectralRgba && this.spectralRowMap) {
        const tPaint = performance.now();
        paintSpectralStripColumn(
          col,
          this.spectralRgba,
          this.spectralRowMap,
          this.stripDisplay!,
          this.stripCols,
          this.lastSpectralReject,
          this.lastSpectralYbRel,
          this.lastTintMap,
        );
        this.stepTiming.stripRasterMs += performance.now() - tPaint;
      }
    }
  }

  /**
   * M-mode columns of this step (decision 84): every column that is due is drawn at its own instant.
   * Lines are traced per phase bin for this probe pose, cursor and imaging settings, as many as the
   * step's budget allows; a column is formed from the traced lines on each side of its instant.
   */
  private advanceMmode(
    times: readonly number[],
    cps: number,
    rr: number,
    phaseNow: number,
    traceBudgetMs: number,
    beam: BeamFrame,
    spec: PolarFrameSpec,
    ctx: StripCtx,
  ): void {
    const n = times.length;
    if (n <= 0) return;
    const inp = ctx.input;
    const theta = Math.max(-spec.sectorRad / 2, Math.min(spec.sectorRad / 2, inp.cursorThetaRad));
    const cmm = inp.modality === 'cmm';
    const bins = mmodePhaseBins(cps, ctx.tables.rrS);
    const ph = ctx.physics();
    const v3 = (v: { x: number; y: number; z: number }): string =>
      `${v.x.toFixed(6)},${v.y.toFixed(6)},${v.z.toFixed(6)}`;
    const key = [
      v3(beam.origin),
      v3(beam.forward),
      v3(beam.lateral),
      v3(beam.normal),
      beam.contact.toFixed(4),
      theta.toFixed(6),
      spec.lines,
      spec.samples,
      spec.depthCm,
      spec.sectorRad.toFixed(6),
      spec.focusCm,
      this.mmodeSamples,
      ph.frequencyMHz,
      ph.harmonics,
      ph.clutterLevel.toFixed(4),
      ph.windowAttenuation.toFixed(4),
      (ph.beamWidth ?? 0).toFixed(4),
      cmm,
      ctx.modelVersion,
      ctx.tablesVersion,
    ].join('|');
    if (this.mmode.key !== key || this.mmode.bins !== bins) this.mmode.reset(key, bins);
    const phases = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const tBack = ctx.timeS - times[i]!;
      phases[i] = (((phaseNow - tBack / rr) % 1) + 1) % 1;
    }
    const maxTraces =
      this.mmode.traceMs > 0 ? Math.max(1, Math.floor(traceBudgetMs / this.mmode.traceMs)) : 1;
    for (const b of binsToTrace(this.mmode, phases, maxTraces)) {
      const t0 = performance.now();
      this.mmode.set(b, this.traceMmodeLine(beam, spec, b, theta, cmm, ctx));
      this.mmode.measure(performance.now() - t0);
    }
    const pulses = mmodePulsesPerColumn(cps);
    for (let i = 0; i < n; i++) {
      const src =
        columnSource(this.mmode, phases[i]!, bins >> 2) ??
        columnSource(this.mmode, phases[i]!, bins >> 1);
      if (src)
        this.writeMmodeColumn(this.stripHead % this.stripCols, src, phases[i]!, pulses, spec, ctx);
      this.headTimeS = times[i]!;
      this.stripHead++;
    }
  }

  /** Trace the M-mode line at `phase`: the fine line envelope and, in colour M-mode, the axial flow velocity. */
  private traceMmodeLine(
    beam: BeamFrame,
    spec: PolarFrameSpec,
    bin: number,
    theta: number,
    cmm: boolean,
    ctx: StripCtx,
  ): MmodeLine {
    const phase = bin / this.mmode.bins;
    const S = this.mmodeSamples;
    if (this.lineAmp.length !== S) {
      this.lineAmp = new Float32Array(S);
      this.lineSt = new Uint8Array(S);
      this.lineTr = new Float32Array(S);
      this.lineTi = new Uint8Array(S);
    }
    const scene = ctx.scene(phase);
    ctx.procedural.renderMmodeLine(
      scene,
      beam,
      spec,
      S,
      theta,
      bin,
      this.lineAmp,
      this.lineSt,
      this.lineTr,
      this.lineTi,
    );
    const cmmData = cmm ? this.cmmVelocity(beam, spec, phase, theta, scene, ctx) : null;
    return {
      amp: this.lineAmp.slice(),
      velocity: cmmData ? cmmData.velocity : null,
      dispersion: cmmData ? cmmData.dispersion : null,
      relTrans: cmmData ? cmmData.relTrans : null,
    };
  }

  /** Colour M-mode: axial flow velocity (+ toward transducer), variance, and relative acoustic transmission. */
  private cmmVelocity(
    beam: BeamFrame,
    spec: PolarFrameSpec,
    phase: number,
    theta: number,
    scene: Scene,
    ctx: StripCtx,
  ): { velocity: Float32Array; dispersion: Float32Array; relTrans: Float32Array } {
    const hf = ctx.heart.frame;
    const ct = Math.cos(theta),
      sn = Math.sin(theta);
    const dx = beam.forward.x * ct + beam.lateral.x * sn;
    const dy = beam.forward.y * ct + beam.lateral.y * sn;
    const dz = beam.forward.z * ct + beam.lateral.z * sn;
    const dhx = dx * hf.ex.x + dy * hf.ex.y + dz * hf.ex.z;
    const dhy = dx * hf.ey.x + dy * hf.ey.y + dz * hf.ey.z;
    const dhz = dx * hf.ez.x + dy * hf.ez.y + dz * hf.ez.z;
    const fs = this.flowSample;
    const F = spec.samples;
    const S = this.mmodeSamples;
    const velocity = new Float32Array(F);
    const dispersion = new Float32Array(F);
    const relTrans = new Float32Array(F);
    const dr = spec.depthCm / F;
    const acq = {
      frequencyMHz: ctx.input.settings.frequencyMHz,
      harmonics: ctx.input.settings.harmonics,
    };
    for (let si = 0; si < F; si++) {
      const li = Math.min(S - 1, Math.floor(((si + 0.5) / F) * S));
      const r = (si + 0.5) * dr;
      const rel = relativeTransmission(this.lineTr[li] ?? 0, r, acq);
      if (this.lineTi[li] !== Tissue.Blood || rel < DOPPLER_SHADOW_TRANSMISSION) {
        velocity[si] = NaN;
        continue;
      }
      const px = beam.origin.x + dx * r,
        py = beam.origin.y + dy * r,
        pz = beam.origin.z + dz * r;
      const hx =
        (px - hf.origin.x) * hf.ex.x + (py - hf.origin.y) * hf.ex.y + (pz - hf.origin.z) * hf.ex.z;
      const hy =
        (px - hf.origin.x) * hf.ey.x + (py - hf.origin.y) * hf.ey.y + (pz - hf.origin.z) * hf.ey.z;
      const hz =
        (px - hf.origin.x) * hf.ez.x + (py - hf.origin.y) * hf.ez.y + (pz - hf.origin.z) * hf.ez.z;
      sampleFlow(ctx.flow, ctx.tables, scene.heartPose, phase, hx, hy, hz, fs);
      if (!fs.present) {
        velocity[si] = NaN;
        continue;
      }
      velocity[si] = -(fs.vx * dhx + fs.vy * dhy + fs.vz * dhz);
      dispersion[si] = Math.min(1, fs.dispersion * 1.6);
      relTrans[si] = Math.min(1, rel);
    }
    return { velocity, dispersion, relTrans };
  }

  /** Form an M-mode column from its traced lines, pass it through the console and store it. */
  private writeMmodeColumn(
    col: number,
    src: ColumnSource,
    phase: number,
    pulses: number,
    spec: PolarFrameSpec,
    ctx: StripCtx,
  ): void {
    const inp = ctx.input;
    const S = this.mmodeSamples;
    if (this.mmodeColumn.length !== S) {
      this.mmodeColumn = new Float32Array(S);
      this.mmodeGrey = new Uint8ClampedArray(S);
    }
    if (
      !this.mmodeFrame ||
      this.mmodeFrame.spec.samples !== S ||
      this.mmodeFrame.spec.depthCm !== spec.depthCm
    ) {
      this.mmodeFrame = {
        spec: { ...spec, lines: 1, samples: S },
        amplitude: this.mmodeColumn,
        structure: new Uint8Array(S),
        transmission: new Float32Array(S),
        tissue: new Uint8Array(S),
        segment: new Uint8Array(S),
      };
    }
    const a = src.lo.amp,
      b = src.hi.amp,
      t = src.t;
    const column = this.mmodeColumn;
    for (let s = 0; s < S; s++) column[s] = a[s]! + (b[s]! - a[s]!) * t;
    const st = this.mmodeConsole;
    st.seed = (ctx.caseDef.seed + this.stripHead) | 0;
    st.frameIndex = 0;
    st.prev = null;
    applyConsole(this.mmodeFrame, inp.settings, st, this.mmodeGrey, NO_ARTIFACTS, {
      noisePulses: pulses,
      edgeStep: S / spec.samples,
    });
    const cols = this.stripCols;
    const strip = this.stripMmode!;
    const grey = this.mmodeGrey;
    for (let s = 0; s < S; s++) strip[s * cols + col] = grey[s]!;
    const effColor = this.acquired ? this.effectiveColor(this.acquired) : inp.color;
    if (this.stripCmm) {
      const lineSrc = t < 0.5 ? src.lo : src.hi;
      const v = lineSrc.velocity;
      const d = lineSrc.dispersion;
      const tr = lineSrc.relTrans;
      const cmm = this.stripCmm;
      const cmmVar = this.stripCmmVar;
      const cmmPow = this.stripCmmPow;
      const gainLin = Math.pow(10, effColor.gainDb / 20);
      for (let si = 0; si < spec.samples; si++) {
        const idx = si * cols + col;
        const x = v ? v[si]! : NaN;
        const vx = effColor.invert ? -x : x;
        const absV = Math.abs(vx);
        if (Number.isNaN(vx) || absV < effColor.wallFilterMps) {
          cmm[idx] = NaN;
          if (cmmVar) cmmVar[idx] = 0;
          if (cmmPow) cmmPow[idx] = 0;
          continue;
        }
        const rel = tr ? tr[si]! : 1;
        const pow = Math.min(1, gainLin * rel * wallFilterResponse(absV, effColor.wallFilterMps));
        if (pow < COLOR_THRESHOLD) {
          cmm[idx] = NaN;
          if (cmmVar) cmmVar[idx] = 0;
          if (cmmPow) cmmPow[idx] = 0;
          continue;
        }
        cmm[idx] = aliasVelocity(vx, effColor.scaleMps, effColor.baselineShiftMps);
        if (cmmVar) cmmVar[idx] = d ? d[si]! : 0;
        if (cmmPow) cmmPow[idx] = pow;
      }
    }
    this.stripPhase[col] = phase;
    this.stripSpan[col] = Math.min(65535, src.span);
    if (this.stripRgba && this.rowMap) {
      const displayColor = this.stripCmm
        ? resolveFrozenColorSettings(effColor, inp.color)
        : effColor;
      paintMmodeStripColumn(
        col,
        this.stripRgba,
        this.rowMap,
        strip,
        cols,
        S,
        this.lastTintMap,
        this.stripCmm,
        displayColor,
        this.stripCmmVar,
        this.stripCmmPow,
      );
    }
  }

  /** Structures at the Doppler/M-mode cursor and the beam–flow angle at the PW/TDI gate. */
  gateInfo(
    beam: BeamFrame,
    spec: PolarFrameSpec,
    phase: number,
    structure: Uint8Array,
    ctx: StripCtx,
  ): GateInfo {
    const t0 = performance.now();
    const info = computeGateInfo(
      beam,
      spec,
      phase,
      structure,
      ctx,
      this.gateFlow,
      this.flowSample,
      this.tissueSample,
    );
    this.stepTiming.gateInfoMs += performance.now() - t0;
    return info;
  }

  /** Answer an on-demand request (auto-trace of the spectral envelope between two strip columns). */
  request(req: SimRequest): SimResponse | null {
    if (req.kind === 'autoTrace') {
      const strip = this.stripSpectral;
      if (!strip || this.stripKind !== 'spectral' || !this.acquired) return null;
      return autoTraceSpectralEnvelope(
        strip,
        this.stripCols,
        this.displayCols,
        this.effectiveSpectral(this.acquired),
        req.x0,
        req.x1,
        this.acquired.aliasing,
      );
    }
    return null;
  }

  drawStrip(
    rgba: Uint8ClampedArray,
    W: number,
    H: number,
    sectorH: number,
    tintMap?: TintMapId,
    liveColor?: ColorSettings,
    liveSpectral?: SpectralSettings,
  ): StripInfo {
    const acquired = this.acquired;
    this.displayCols = W;
    this.lastTintMap = tintMap;
    const y0 = sectorH + 2;
    const h = H - y0 - 4;
    const cols = this.stripCols;
    const secondsShown = aseSweepDurationS(
      this.acquired?.spectral.sweepSpeedMmPerS ?? 50,
      STRIP_MM_WIDTH,
    );
    const spc = cols ? secondsShown / W : 0;
    const kind = this.stripKind;
    const head = Math.floor(((this.stripHead % Math.max(1, cols)) / Math.max(1, cols)) * W);
    if (kind === 'm-mode' && this.stripMmode && acquired && h > 0) {
      const effColor = this.effectiveColor(acquired);
      const displayColor =
        this.stripCmm && liveColor ? resolveFrozenColorSettings(effColor, liveColor) : effColor;
      const cmmLayout = this.stripCmm
        ? `|${displayColor.palette ?? 'vivid'}|${displayColor.showVariance ? 1 : 0}|${displayColor.baselineShiftMps}`
        : '';
      const layout = `${cols}x${h}|${this.mmodeSamples}|${tintMap ?? 'gray'}${cmmLayout}`;
      if (this.stripRgbaKey !== layout || !this.stripRgba) {
        this.stripRgba = new Uint8ClampedArray(cols * h * 4);
        this.rowMap = buildRowMap(h, this.mmodeSamples);
        this.stripRgbaKey = layout;
        for (let c = 0; c < cols; c++) {
          paintMmodeStripColumn(
            c,
            this.stripRgba,
            this.rowMap,
            this.stripMmode,
            cols,
            this.mmodeSamples,
            tintMap,
            this.stripCmm,
            displayColor,
            this.stripCmmVar,
            this.stripCmmPow,
          );
        }
      }
      if (cols === W) rgba.set(this.stripRgba, y0 * W * 4);
      else {
        for (let y = 0; y < h; y++)
          for (let x = 0; x < W; x++) {
            const src = (y * cols + Math.floor((x * cols) / W)) * 4;
            const dst = ((y0 + y) * W + x) * 4;
            for (let c = 0; c < 4; c++) rgba[dst + c] = this.stripRgba[src + c]!;
          }
      }
      drawSweepMarker(rgba, W, y0, h, head);
      return {
        x: 0,
        y: y0,
        width: W,
        height: h,
        secondsPerColumn: spc,
        acquisitionId: this.acquisitionId,
        headTimeS: this.headTimeS,
        headColumn: head,
        columns: W,
        sweepSpeedMmPerS: acquired.spectral.sweepSpeedMmPerS,
        topValue: 0,
        bottomValue: acquired.depthCm,
        kind: 'm-mode',
      };
    }
    if (kind === 'spectral' && this.stripDisplay && acquired) {
      const tRaster = performance.now();
      const effectiveSpectral = this.effectiveSpectral(acquired);
      const reject = Math.max(
        0,
        Math.min(0.6, liveSpectral?.reject ?? effectiveSpectral.reject ?? 0),
      );
      const { vMin, vMax } = spectralRange(effectiveSpectral);
      if (cols === W && h > 0) {
        const ybAbs = y0 + Math.round((vMax / (vMax - vMin)) * h);
        const ybRel = ybAbs >= y0 && ybAbs < Math.min(H, y0 + h) ? ybAbs - y0 : -1;
        const layout = `${cols}x${h}|${tintMap ?? 'gray'}|${reject.toFixed(4)}|${ybRel}`;
        if (this.spectralRgbaKey !== layout || !this.spectralRgba) {
          this.spectralRgba = new Uint8ClampedArray(cols * h * 4);
          if (!this.spectralRowMap || this.spectralRowMap.rows !== h) {
            this.spectralRowMap = buildSpectralRowMap(h);
          }
          this.spectralRgbaKey = layout;
          this.lastSpectralReject = reject;
          this.lastSpectralYbRel = ybRel;
          for (let c = 0; c < cols; c++) {
            paintSpectralStripColumn(
              c,
              this.spectralRgba,
              this.spectralRowMap,
              this.stripDisplay,
              cols,
              reject,
              ybRel,
              tintMap,
            );
          }
        }
        rgba.set(this.spectralRgba, y0 * W * 4);
      } else {
        renderSpectralStripRgba(
          rgba,
          W,
          H,
          y0,
          h,
          this.stripDisplay,
          cols,
          effectiveSpectral,
          tintMap,
          this.stripHead % Math.max(1, cols),
          reject,
        );
      }
      drawSweepMarker(rgba, W, y0, h, head);
      this.stepTiming.stripRasterMs += performance.now() - tRaster;
      const pulsedTiming = acquired.aliasing
        ? depthLimitedPulsedAcquisition(
            acquired.gateDepthCm,
            acquired.spectral.gateLengthCm,
            acquired.frequencyMHz,
            acquired.spectral.scaleMps,
          )
        : null;
      return {
        x: 0,
        y: y0,
        width: W,
        height: h,
        secondsPerColumn: spc,
        acquisitionId: this.acquisitionId,
        headTimeS: this.headTimeS,
        headColumn: head,
        columns: W,
        sweepSpeedMmPerS: acquired.spectral.sweepSpeedMmPerS,
        wallFilterMps: acquired.spectral.wallFilterMps,
        frequencyMHz: acquired.frequencyMHz,
        prfHz: pulsedTiming?.prfHz,
        maxPrfHz: pulsedTiming?.maxPrfHz,
        nyquistMps: pulsedTiming?.effectiveScaleMps,
        gateDepthCm: acquired.aliasing ? acquired.gateDepthCm : undefined,
        aliasing: acquired.aliasing,
        estimatorIntervalS: secondsShown / cols,
        topValue: vMax,
        bottomValue: vMin,
        kind: 'spectral',
      };
    }
    return {
      x: 0,
      y: y0,
      width: W,
      height: h,
      secondsPerColumn: spc,
      headColumn: 0,
      columns: 0,
      topValue: 0,
      bottomValue: 0,
      kind: null,
    };
  }
}
