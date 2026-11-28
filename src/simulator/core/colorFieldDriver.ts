import type { HeartModel } from '@/simulator/anatomy/heartModel';
import { Tissue } from '@/simulator/anatomy/tissue';
import type { BeatTables } from '@/simulator/cardiac-cycle/cycleModel';
import type { BeamFrame, ProbeControl } from '@/simulator/probe/pose';
import {
  computeDopplerBoxBounds,
  type DeferredDopplerRequest,
} from '@/simulator/renderer/gpu/deferredDoppler';
import type { Webgl2Renderer } from '@/simulator/renderer/gpu/webgl2Renderer';
import { TRANS_DECODE } from '@/simulator/renderer/transmissionCode';
import type { PolarFrame, PolarFrameSpec, Scene } from '@/simulator/renderer/types';
import { persistenceOverTime } from '@/simulator/renderer/postprocess/consolePipeline';
import {
  prepareClutterContext,
  sampleClutterAxialVelocity,
  sampleFlow,
  type FlowFieldParams,
  type FlowSample,
} from '@/simulator/doppler/flow-primitives/flowField';
import {
  allocColorField,
  colorSpectralWidthMps,
  computeColorField,
  type ColorField,
  type ColorSettings,
} from '@/simulator/doppler/color/colorDoppler';
import type { CineFrame } from './cineBuffer';
import { colorLinePhaseOffset, wrapPhase } from '@/simulator/doppler/color/colorSweep';
import { colorFrameTimeS } from '@/simulator/renderer/frameRate';
import {
  colorAcquisitionKey,
  colorCpuAcquisitionKey,
  colorGeometryKey,
} from './acquisitionKeys';

export interface ColorUpdateContext {
  frame: PolarFrame;
  scene: Scene;
  heart: HeartModel;
  flow: FlowFieldParams;
  tables: BeatTables;
  beam: BeamFrame;
  spec: PolarFrameSpec;
  phase: number;
  probe: ProbeControl;
  color: ColorSettings;
  frequencyMHz: number;
  harmonics: boolean;
  modelVersion: number;
  timeS: number;
  frameIntervalS: number;
  seed: number;
  /**
   * Duration (s) of the colour frame's sweep across the box (`colorFrameTimeS`): each colour line reads the flow at its
   * own instant, centred on `phase` (colorSweep.ts). Zero or absent: every line at `phase`.
   */
  sweepS?: number;
}

/** Phase of each polar line of the box [liMin, liMax] for a sweep, the frame's phase without one. */
function linePhases(
  phase: number,
  liMin: number,
  liMax: number,
  sweepS: number | undefined,
  rrS: number,
  out: Float64Array,
): void {
  for (let li = liMin; li <= liMax; li++)
    out[li - liMin] = wrapPhase(phase + colorLinePhaseOffset(li, liMin, liMax, sweepS ?? 0, rrS));
}

/**
 * Owns the double-buffered Color Doppler polar field (CPU path) and the sparse ROI-box raw flow
 * sampler for the independent GPU deferred Doppler shading pipeline, including full acquired-frame
 * cadence (matching `cadenceHz`, which already accounts for Color packets), acquisition
 * invalidation keys, the freeze of the GPU field on screen, and lazy frozen-cine reconstruction
 * from each frame's own beat.
 */
export class ColorFieldDriver {
  /** The two colour field buffers: each update writes the one that does not hold `prev` (decision 56). */
  private buffers: [ColorField, ColorField] | null = null;
  /**
   * Field of the latest colour update: the composite, the cine and the GPU present pass show it, and the next update
   * blends it (persistence). Null until the first update after a polar spec or modality change.
   */
  private prev: ColorField | null = null;
  private acquisitionKey = '';
  private gpuGeometryKey = '';
  private gpuSettingsKey = '';
  private _version = 0;
  private _lastTimeS = NaN;
  private flowSample: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };
  private rawFlowBox = new Float32Array(0);
  private rawFlowHasNeighbors = false;
  private cineTransScratch = new Float32Array(0);
  private linePhase = new Float64Array(0);

  /** The per-line phases of the box for this update (the frame's phase on every line without a sweep). */
  private phasesFor(phase: number, liMin: number, liMax: number, sweepS: number | undefined, rrS: number): Float64Array {
    const n = Math.max(0, liMax - liMin + 1);
    if (this.linePhase.length < n) this.linePhase = new Float64Array(n);
    linePhases(phase, liMin, liMax, sweepS, rrS, this.linePhase);
    return this.linePhase;
  }

  get current(): ColorField | null {
    return this.prev;
  }

  get version(): number {
    return this._version;
  }

  get lastTimeS(): number {
    return this._lastTimeS;
  }

  resize(sampleCount: number): void {
    this.buffers = [allocColorField(sampleCount), allocColorField(sampleCount)];
    this.prev = null;
    this._lastTimeS = NaN;
    this.gpuGeometryKey = '';
    this.gpuSettingsKey = '';
    this.rawFlowHasNeighbors = false;
  }

  resetHistory(): void {
    this.prev = null;
    this.gpuGeometryKey = '';
    this.gpuSettingsKey = '';
    this.rawFlowHasNeighbors = false;
  }

  /** Build the per-geometry key (probe, ROI box, frequency/harmonics, polar grid) for GPU raw-flow reuse. */
  private geometryKey(ctx: ColorUpdateContext): string {
    return colorGeometryKey(
      ctx.probe,
      ctx.color,
      ctx.scene.physics.frequencyMHz,
      ctx.scene.physics.harmonics,
      ctx.modelVersion,
      ctx.spec,
    );
  }

  /** Build the Doppler acquisition-only settings key (excludes presentation-only toggles like `showVariance`). */
  static settingsKey(c: ColorSettings): string {
    return colorAcquisitionKey(c);
  }

  /**
   * Independent GPU Deferred Doppler step: samples only the sparse ROI box hemodynamic flow on CPU
   * and delegates shadow rejection, gain blooming, 8-pulse Kasai packet autocorrelation, regression
   * wall filtering, 3×3 resolution-cell averaging, Nyquist aliasing, and circular phasor persistence
   * to the GPU deferred Doppler passes against the resident Polar G-Buffer.
   */
  stepDeferredGpu(ctx: ColorUpdateContext, gpu: Webgl2Renderer): boolean {
    const gKey = this.geometryKey(ctx);
    const sKey = ColorFieldDriver.settingsKey(ctx.color);
    const geometryChanged = gKey !== this.gpuGeometryKey;
    if (geometryChanged) this.gpuGeometryKey = gKey;
    this.gpuSettingsKey = sKey;

    const colorElapsed = Number.isFinite(this._lastTimeS)
      ? ctx.timeS - this._lastTimeS
      : ctx.frameIntervalS;
    this._lastTimeS = ctx.timeS;
    const effPersist = persistenceOverTime(
      ctx.color.persistence,
      colorElapsed,
      ctx.frameIntervalS,
    );

    const box = computeDopplerBoxBounds(ctx.spec, ctx.color);
    const bloomSamples = Math.max(0, Math.round((ctx.color.gainDb - 2) * 0.6));
    let rawFlowBox: Float32Array | undefined;
    if (!box.empty) {
      const needed = box.boxW * box.boxH * 4;
      if (this.rawFlowBox.length < needed) this.rawFlowBox = new Float32Array(needed);
      rawFlowBox =
        this.rawFlowBox.length === needed ? this.rawFlowBox : this.rawFlowBox.subarray(0, needed);
      this.sampleRawFlowBox(ctx, box.liMin, box.liMax, box.siMin, box.siMax, bloomSamples, rawFlowBox);
      this.rawFlowHasNeighbors = bloomSamples > 0;
    }

    const req: DeferredDopplerRequest = {
      spec: ctx.spec,
      settings: { ...ctx.color, persistence: effPersist },
      frequencyMHz: ctx.frequencyMHz,
      harmonics: ctx.harmonics,
      rawFlowBox,
      estimate: { realization: this._version, seed: ctx.seed },
      resetHistory: geometryChanged,
    };
    const ok = gpu.shadeDoppler(req);
    if (ok) this._version++;
    return ok;
  }

  /**
   * Re-shade Doppler immediately against the GPU-resident Polar G-Buffer when Doppler-only console
   * controls change between B-mode frames — zero B-mode anatomical passes executed.
   */
  reshadeDeferredGpu(ctx: ColorUpdateContext, gpu: Webgl2Renderer): boolean {
    const sKey = ColorFieldDriver.settingsKey(ctx.color);
    if (sKey === this.gpuSettingsKey) return false;
    const gKey = this.geometryKey(ctx);
    const geometryChanged = gKey !== this.gpuGeometryKey;
    this.gpuGeometryKey = gKey;
    this.gpuSettingsKey = sKey;

    const box = computeDopplerBoxBounds(ctx.spec, ctx.color);
    const bloomSamples = Math.max(0, Math.round((ctx.color.gainDb - 2) * 0.6));
    const needBloomNeighbors = bloomSamples > 0 && !this.rawFlowHasNeighbors;
    let rawFlowBox: Float32Array | undefined;
    if (geometryChanged || needBloomNeighbors || !gpu.canReshadeDoppler) {
      if (!box.empty) {
        const needed = box.boxW * box.boxH * 4;
        if (this.rawFlowBox.length < needed) this.rawFlowBox = new Float32Array(needed);
        rawFlowBox =
          this.rawFlowBox.length === needed ? this.rawFlowBox : this.rawFlowBox.subarray(0, needed);
        this.sampleRawFlowBox(ctx, box.liMin, box.liMax, box.siMin, box.siMax, bloomSamples, rawFlowBox);
        this.rawFlowHasNeighbors = bloomSamples > 0;
      }
    }

    const ok = gpu.shadeDoppler({
      spec: ctx.spec,
      settings: { ...ctx.color, persistence: 0 },
      frequencyMHz: ctx.frequencyMHz,
      harmonics: ctx.harmonics,
      rawFlowBox,
      estimate: { realization: this._version, seed: ctx.seed },
      resetHistory: true,
    });
    if (ok) this._version++;
    return ok;
  }

  /** Sample only the polar ROI box into a compact RGBA float buffer `[vAxial, sigmaVMps, present, vWall]`. */
  private sampleRawFlowBox(
    ctx: ColorUpdateContext,
    liMin: number,
    liMax: number,
    siMin: number,
    siMax: number,
    bloomSamples: number,
    out: Float32Array,
  ): void {
    const { frame, scene, heart, flow, tables, beam, spec, phase } = ctx;
    const hf = heart.frame;
    const fs = this.flowSample;
    const { samples, sectorRad, lines, depthCm } = spec;
    const dr = depthCm / samples;
    const hp = scene.heartPose;
    const boxW = siMax - siMin + 1;
    const ox = beam.origin.x - hf.origin.x;
    const oy = beam.origin.y - hf.origin.y;
    const oz = beam.origin.z - hf.origin.z;
    const ohx = ox * hf.ex.x + oy * hf.ex.y + oz * hf.ex.z;
    const ohy = ox * hf.ey.x + oy * hf.ey.y + oz * hf.ey.z;
    const ohz = ox * hf.ez.x + oy * hf.ez.y + oz * hf.ez.z;
    const tissue = frame.tissue;
    const structure = frame.structure;
    const hasTissueMap = tissue.length === lines * samples;
    const filterByBlood = bloomSamples === 0 && hasTissueMap;
    const clutterCtx = prepareClutterContext(heart, tables, hp, phase, flow.flowPeaks);
    const phases = this.phasesFor(phase, liMin, liMax, ctx.sweepS, tables.rrS);

    for (let li = liMin; li <= liMax; li++) {
      const phaseLi = phases[li - liMin]!;
      const theta = -sectorRad / 2 + (sectorRad * (li + 0.5)) / lines;
      const ct = Math.cos(theta);
      const sn = Math.sin(theta);
      const dx = beam.forward.x * ct + beam.lateral.x * sn;
      const dy = beam.forward.y * ct + beam.lateral.y * sn;
      const dz = beam.forward.z * ct + beam.lateral.z * sn;
      const dhx = dx * hf.ex.x + dy * hf.ex.y + dz * hf.ex.z;
      const dhy = dx * hf.ey.x + dy * hf.ey.y + dz * hf.ey.z;
      const dhz = dx * hf.ez.x + dy * hf.ez.y + dz * hf.ez.z;

      const rowOff = (li - liMin) * boxW * 4;
      const polarRow = li * samples;
      for (let si = siMin; si <= siMax; si++) {
        const o = rowOff + (si - siMin) * 4;
        const pIdx = polarRow + si;
        const tId = hasTissueMap ? tissue[pIdx]! : Tissue.Blood;
        const isClutterTissue = tId === Tissue.Myocardium || tId === Tissue.Valve;

        if (filterByBlood && tId !== Tissue.Blood) {
          out[o] = 0;
          out[o + 1] = 0;
          out[o + 2] = 0;
          if (isClutterTissue) {
            const r = (si + 0.5) * dr;
            out[o + 3] = sampleClutterAxialVelocity(
              clutterCtx,
              heart,
              ohx + dhx * r,
              ohy + dhy * r,
              ohz + dhz * r,
              tId,
              structure[pIdx] ?? -1,
              dhx,
              dhy,
              dhz,
            );
          } else {
            out[o + 3] = 0;
          }
          continue;
        }
        const r = (si + 0.5) * dr;
        const hx = ohx + dhx * r;
        const hy = ohy + dhy * r;
        const hz = ohz + dhz * r;
        const bloodStructure = tId === Tissue.Blood ? structure[pIdx] : undefined;
        sampleFlow(flow, tables, hp, phaseLi, hx, hy, hz, fs, bloodStructure);
        if (fs.present) {
          const vAxial = -(fs.vx * dhx + fs.vy * dhy + fs.vz * dhz);
          const vSq = fs.vx * fs.vx + fs.vy * fs.vy + fs.vz * fs.vz;
          const vPerp = Math.sqrt(Math.max(0, vSq - vAxial * vAxial));
          out[o] = vAxial;
          out[o + 1] = colorSpectralWidthMps(vAxial, fs.dispersion, vPerp, r);
          out[o + 2] = 1;
        } else {
          out[o] = 0;
          out[o + 1] = 0;
          out[o + 2] = 0;
        }
        out[o + 3] = isClutterTissue
          ? sampleClutterAxialVelocity(
              clutterCtx,
              heart,
              hx,
              hy,
              hz,
              tId,
              structure[pIdx] ?? -1,
              dhx,
              dhy,
              dhz,
            )
          : 0;
      }
    }
  }

  /**
   * Freeze the newest GPU-deferred frame as it was seen: read back the shaded ROI box the present pass was blending over
   * it (one box-only readPixels per freeze, off the live path), instead of re-estimating it on the CPU with another noise
   * realization, no persistence and the CPU estimator. Re-estimated, the samples that gain or lose colour amount to two
   * thirds of the coloured samples shown (af-diastolic A4C, measured on the CPU path, where the cine keeps the field shown).
   * False, leaving the frame untouched, when the resident field was shaded for another acquisition than the frame's (a
   * settings re-shade after it) or cannot be read.
   */
  readbackDeferredGpu(cf: CineFrame, gpu: Webgl2Renderer): boolean {
    const a = cf.acquisition;
    const c = a.color;
    const physics = cf.scene.physics;
    const shadedForFrame =
      this.gpuSettingsKey === ColorFieldDriver.settingsKey(c) &&
      this.gpuGeometryKey ===
        colorGeometryKey(a.probe, c, physics.frequencyMHz, physics.harmonics, a.modelVersion, cf.spec);
    if (!shadedForFrame) return false;
    const n = cf.spec.lines * cf.spec.samples;
    const vel = new Float32Array(n);
    const variance = new Float32Array(n);
    const power = new Float32Array(n);
    if (!gpu.readbackDopplerBox(cf.spec, c.scaleMps, c.baselineShiftMps, vel, variance, power))
      return false;
    cf.colorVel = vel;
    cf.colorVar = variance;
    cf.colorPower = power;
    return true;
  }

  /**
   * Lazily reconstruct CPU Color Doppler buffers for a frozen cine frame that was captured on the
   * zero-readback GPU deferred path, caching the result directly on the CineFrame. The flow is sampled from the frame's
   * own beat (`cf.tables`, `cf.flow`, `cf.scene.heart`), never the live one.
   */
  reconstructCineColor(cf: CineFrame, seed: number): void {
    if (cf.colorVel || !cf.colorTissue || !cf.colorTransCode) return;
    const { spec, beam, scene, phase, flow, tables } = cf;
    const heart = scene.heart;
    const n = spec.lines * spec.samples;
    if (this.cineTransScratch.length !== n) this.cineTransScratch = new Float32Array(n);
    const trans = this.cineTransScratch;
    const codes = cf.colorTransCode;
    for (let i = 0; i < n; i++) trans[i] = TRANS_DECODE[codes[i]!]!;

    const syntheticFrame: PolarFrame = {
      spec,
      amplitude: trans,
      transmission: trans,
      structure: cf.structure,
      tissue: cf.colorTissue,
      segment: cf.segment,
    };
    const out = allocColorField(n);
    const hf = heart.frame;
    const fs = this.flowSample;
    const { samples, sectorRad, lines, depthCm } = spec;
    const dr = depthCm / samples;
    const hp = scene.heartPose;
    let cachedLi = -1;
    let dhx = 0,
      dhy = 0,
      dhz = 0;
    const ox = beam.origin.x - hf.origin.x;
    const oy = beam.origin.y - hf.origin.y;
    const oz = beam.origin.z - hf.origin.z;
    const ohx = ox * hf.ex.x + oy * hf.ex.y + oz * hf.ex.z;
    const ohy = ox * hf.ey.x + oy * hf.ey.y + oz * hf.ey.z;
    const ohz = ox * hf.ez.x + oy * hf.ez.y + oz * hf.ez.z;

    const clutterCtx = prepareClutterContext(heart, tables, hp, phase, flow.flowPeaks);
    const bloomSamples = Math.max(0, Math.round((cf.acquisition.color.gainDb - 2) * 0.6));
    // the frame's own sweep, from the console it was acquired with
    const cineBox = computeDopplerBoxBounds(spec, cf.acquisition.color);
    const cinePhases = this.phasesFor(
      phase,
      cineBox.liMin,
      cineBox.liMax,
      colorFrameTimeS(cf.acquisition.settings, cf.acquisition.color),
      tables.rrS,
    );
    const phaseAt = (li: number): number =>
      li >= cineBox.liMin && li <= cineBox.liMax ? cinePhases[li - cineBox.liMin]! : phase;

    computeColorField(
      syntheticFrame,
      { ...cf.acquisition.color, persistence: 0 },
      null,
      (idx, li, si, o) => {
        if (li !== cachedLi) {
          cachedLi = li;
          const theta = -sectorRad / 2 + (sectorRad * (li + 0.5)) / lines;
          const ct = Math.cos(theta),
            sn = Math.sin(theta);
          const dx = beam.forward.x * ct + beam.lateral.x * sn;
          const dy = beam.forward.y * ct + beam.lateral.y * sn;
          const dz = beam.forward.z * ct + beam.lateral.z * sn;
          dhx = dx * hf.ex.x + dy * hf.ex.y + dz * hf.ex.z;
          dhy = dx * hf.ey.x + dy * hf.ey.y + dz * hf.ey.z;
          dhz = dx * hf.ez.x + dy * hf.ez.y + dz * hf.ez.z;
        }
        const r = (si + 0.5) * dr;
        const hx = ohx + dhx * r;
        const hy = ohy + dhy * r;
        const hz = ohz + dhz * r;
        const tId = syntheticFrame.tissue[idx]!;
        const isClutterTissue = tId === Tissue.Myocardium || tId === Tissue.Valve;
        if (isClutterTissue) {
          o.vWall = sampleClutterAxialVelocity(
            clutterCtx,
            heart,
            hx,
            hy,
            hz,
            tId,
            syntheticFrame.structure[idx] ?? -1,
            dhx,
            dhy,
            dhz,
          );
          if (bloomSamples === 0) {
            o.present = 0;
            return;
          }
        }
        const bloodStructure = tId === Tissue.Blood ? syntheticFrame.structure[idx] : undefined;
        sampleFlow(flow, tables, hp, phaseAt(li), hx, hy, hz, fs, bloodStructure);
        const vAxial = -(fs.vx * dhx + fs.vy * dhy + fs.vz * dhz);
        const vSq = fs.vx * fs.vx + fs.vy * fs.vy + fs.vz * fs.vz;
        o.v = vAxial;
        o.disp = fs.dispersion;
        o.vPerp = Math.sqrt(Math.max(0, vSq - vAxial * vAxial));
        o.present = fs.present;
      },
      out,
      {
        frequencyMHz: cf.acquisition.settings.frequencyMHz,
        harmonics: cf.acquisition.settings.harmonics,
      },
      { realization: cf.frameId, seed },
    );
    cf.colorVel = out.vel;
    cf.colorVar = out.variance;
    cf.colorPower = out.power;
  }

  /** Step Color Doppler for one acquired Color frame (CPU fallback path). */
  step(ctx: ColorUpdateContext): {
    vel: Float32Array | null;
    variance: Float32Array | null;
    power: Float32Array | null;
  } {
    const colorKey = colorCpuAcquisitionKey(
      ctx.probe,
      ctx.color,
      ctx.scene.physics.frequencyMHz,
      ctx.scene.physics.harmonics,
      ctx.modelVersion,
      ctx.spec,
    );
    if (colorKey !== this.acquisitionKey) {
      this.prev = null;
      this.acquisitionKey = colorKey;
    }
    // Acquisition cadence already accounts for Color Doppler packets (`cadenceHz`): update every acquired frame.
    this.compute(ctx);
    return this.prev
      ? { vel: this.prev.vel, variance: this.prev.variance, power: this.prev.power }
      : { vel: null, variance: null, power: null };
  }

  private compute(ctx: ColorUpdateContext): void {
    const { frame, scene, heart, flow, tables, beam, spec, phase } = ctx;
    const hf = heart.frame;
    const fs = this.flowSample;
    const { samples, sectorRad, lines, depthCm } = spec;
    const dr = depthCm / samples;
    const hp = scene.heartPose;
    // write into the buffer that does not hold the previous field, which persistence blends
    const [a, b] = this.buffers!;
    const out = this.prev === a ? b : a;
    // persistence decays over the acquired Color frame interval (decision 94)
    const colorElapsed = Number.isFinite(this._lastTimeS)
      ? ctx.timeS - this._lastTimeS
      : ctx.frameIntervalS;
    this._lastTimeS = ctx.timeS;
    // Cache per-line beam direction & heart-frame basis projection (reused across all depth samples on line `li`)
    let cachedLi = -1;
    let dx = 0,
      dy = 0,
      dz = 0;
    let dhx = 0,
      dhy = 0,
      dhz = 0;
    const ox = beam.origin.x - hf.origin.x,
      oy = beam.origin.y - hf.origin.y,
      oz = beam.origin.z - hf.origin.z;
    const ohx = ox * hf.ex.x + oy * hf.ex.y + oz * hf.ex.z;
    const ohy = ox * hf.ey.x + oy * hf.ey.y + oz * hf.ey.z;
    const ohz = ox * hf.ez.x + oy * hf.ez.y + oz * hf.ez.z;
    const clutterCtx = prepareClutterContext(heart, tables, hp, phase, flow.flowPeaks);
    const bloomSamples = Math.max(0, Math.round((ctx.color.gainDb - 2) * 0.6));
    const box = computeDopplerBoxBounds(spec, ctx.color);
    const phases = this.phasesFor(phase, box.liMin, box.liMax, ctx.sweepS, tables.rrS);
    const phaseAt = (li: number): number =>
      li >= box.liMin && li <= box.liMax ? phases[li - box.liMin]! : phase;
    computeColorField(
      frame,
      {
        ...ctx.color,
        persistence: persistenceOverTime(
          ctx.color.persistence,
          colorElapsed,
          ctx.frameIntervalS,
        ),
      },
      this.prev,
      (idx, li, si, o) => {
        if (li !== cachedLi) {
          cachedLi = li;
          const theta = -sectorRad / 2 + (sectorRad * (li + 0.5)) / lines;
          const ct = Math.cos(theta),
            sn = Math.sin(theta);
          dx = beam.forward.x * ct + beam.lateral.x * sn;
          dy = beam.forward.y * ct + beam.lateral.y * sn;
          dz = beam.forward.z * ct + beam.lateral.z * sn;
          dhx = dx * hf.ex.x + dy * hf.ex.y + dz * hf.ex.z;
          dhy = dx * hf.ey.x + dy * hf.ey.y + dz * hf.ey.z;
          dhz = dx * hf.ez.x + dy * hf.ez.y + dz * hf.ez.z;
        }
        const r = (si + 0.5) * dr;
        const hx = ohx + dhx * r;
        const hy = ohy + dhy * r;
        const hz = ohz + dhz * r;
        const tId = frame.tissue[idx]!;
        const isClutterTissue = tId === Tissue.Myocardium || tId === Tissue.Valve;
        if (isClutterTissue) {
          o.vWall = sampleClutterAxialVelocity(
            clutterCtx,
            heart,
            hx,
            hy,
            hz,
            tId,
            frame.structure[idx] ?? -1,
            dhx,
            dhy,
            dhz,
          );
          if (bloomSamples === 0) {
            o.present = 0;
            return;
          }
        }
        const bloodStructure = tId === Tissue.Blood ? frame.structure[idx] : undefined;
        sampleFlow(flow, tables, hp, phaseAt(li), hx, hy, hz, fs, bloodStructure);
        const vAxial = -(fs.vx * dhx + fs.vy * dhy + fs.vz * dhz); // + = toward the transducer
        const vSq = fs.vx * fs.vx + fs.vy * fs.vy + fs.vz * fs.vz;
        o.v = vAxial;
        o.disp = fs.dispersion;
        o.vPerp = Math.sqrt(Math.max(0, vSq - vAxial * vAxial));
        o.present = fs.present;
      },
      out,
      { frequencyMHz: ctx.frequencyMHz, harmonics: ctx.harmonics },
      // every update is a new estimate from new echoes (decision 116)
      { realization: this._version, seed: ctx.seed },
    );
    this.prev = out;
    this._version++;
  }
}
