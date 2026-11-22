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
  sampleFlow,
  type FlowFieldParams,
  type FlowSample,
} from '@/simulator/doppler/flow-primitives/flowField';
import {
  allocColorField,
  computeColorField,
  type ColorField,
  type ColorSettings,
} from '@/simulator/doppler/color/colorDoppler';
import type { CineFrame } from './cineBuffer';

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
}

/**
 * Owns the double-buffered Color Doppler polar field (CPU path) and the sparse ROI-box raw flow
 * sampler for the independent GPU deferred Doppler shading pipeline, including packet cadence
 * (every other B-mode frame), acquisition invalidation keys, and lazy frozen-cine reconstruction.
 */
export class ColorFieldDriver {
  /** The two colour field buffers: each update writes the one that does not hold `prev` (decision 56). */
  private buffers: [ColorField, ColorField] | null = null;
  /**
   * Field of the latest colour update: the composite, the cine and the GPU present pass show it, and the next update
   * blends it (persistence). Null until the first update after a polar spec or modality change.
   */
  private prev: ColorField | null = null;
  private frameCounter = 0;
  private acquisitionKey = '';
  private gpuGeometryKey = '';
  private gpuSettingsKey = '';
  private _version = 0;
  private _lastTimeS = NaN;
  private flowSample: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };
  private rawFlowBox = new Float32Array(0);
  private rawFlowHasNeighbors = false;
  private cineTransScratch = new Float32Array(0);

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
    const c = ctx.color;
    return JSON.stringify([
      ctx.probe,
      c.boxThetaMinRad,
      c.boxThetaMaxRad,
      c.boxRMinCm,
      c.boxRMaxCm,
      ctx.scene.physics.frequencyMHz,
      ctx.scene.physics.harmonics,
      ctx.modelVersion,
      ctx.spec.lines,
      ctx.spec.samples,
      ctx.spec.depthCm,
      ctx.spec.sectorRad,
    ]);
  }

  /** Build the Doppler console-only settings key (scale, baseline, wall filter, gain, persistence, invert, variance). */
  static settingsKey(c: ColorSettings): string {
    return `${c.boxThetaMinRad}|${c.boxThetaMaxRad}|${c.boxRMinCm}|${c.boxRMaxCm}|${c.scaleMps}|${c.baselineShiftMps}|${c.wallFilterMps}|${c.gainDb}|${c.persistence}|${c.invert ? 1 : 0}|${c.showVariance ? 1 : 0}`;
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
    const settingsChanged = sKey !== this.gpuSettingsKey;
    if (geometryChanged) this.gpuGeometryKey = gKey;
    this.gpuSettingsKey = sKey;

    this.frameCounter++;
    const needUpdate =
      geometryChanged ||
      settingsChanged ||
      this.frameCounter % 2 === 0 ||
      !gpu.hasDeferredDoppler;
    if (!needUpdate) return true;

    const colorElapsed = Number.isFinite(this._lastTimeS)
      ? ctx.timeS - this._lastTimeS
      : 2 * ctx.frameIntervalS;
    this._lastTimeS = ctx.timeS;
    const effPersist = persistenceOverTime(
      ctx.color.persistence,
      colorElapsed,
      2 * ctx.frameIntervalS,
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

  /** Sample only the polar ROI box into a compact RGBA float buffer `[vAxial, dispersion, present, 0]`. */
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
    const filterByBlood = bloomSamples === 0 && tissue.length === lines * samples;

    for (let li = liMin; li <= liMax; li++) {
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
        if (filterByBlood && tissue[polarRow + si] !== Tissue.Blood) {
          out[o] = 0;
          out[o + 1] = 0;
          out[o + 2] = 0;
          out[o + 3] = 0;
          continue;
        }
        const r = (si + 0.5) * dr;
        sampleFlow(flow, tables, hp, phase, ohx + dhx * r, ohy + dhy * r, ohz + dhz * r, fs);
        if (fs.present) {
          out[o] = -(fs.vx * dhx + fs.vy * dhy + fs.vz * dhz);
          out[o + 1] = fs.dispersion;
          out[o + 2] = 1;
          out[o + 3] = 0;
        } else {
          out[o] = 0;
          out[o + 1] = 0;
          out[o + 2] = 0;
          out[o + 3] = 0;
        }
      }
    }
  }

  /**
   * Lazily reconstruct CPU Color Doppler buffers for a frozen cine frame that was captured on the
   * zero-readback GPU deferred path, caching the result directly on the CineFrame.
   */
  reconstructCineColor(
    cf: CineFrame,
    heart: HeartModel,
    flow: FlowFieldParams,
    tables: BeatTables,
    seed: number,
  ): void {
    if (cf.colorVel || !cf.colorTissue || !cf.colorTransCode) return;
    const { spec, beam, scene, phase } = cf;
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

    computeColorField(
      syntheticFrame,
      { ...cf.acquisition.color, persistence: 0 },
      null,
      (_idx, li, si, o) => {
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
        sampleFlow(flow, tables, hp, phase, ohx + dhx * r, ohy + dhy * r, ohz + dhz * r, fs);
        o.v = -(fs.vx * dhx + fs.vy * dhy + fs.vz * dhz);
        o.disp = fs.dispersion;
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
  }

  /** Step Color Doppler for one B-mode frame (CPU fallback path); recomputes every other frame or when settings/geometry changed. */
  step(ctx: ColorUpdateContext): { vel: Float32Array | null; variance: Float32Array | null } {
    const colorKey = JSON.stringify([
      ctx.probe,
      ctx.color,
      ctx.scene.physics.frequencyMHz,
      ctx.scene.physics.harmonics,
      ctx.modelVersion,
      ctx.spec.lines,
      ctx.spec.samples,
      ctx.spec.depthCm,
      ctx.spec.sectorRad,
    ]);
    if (colorKey !== this.acquisitionKey) {
      this.prev = null;
      this.acquisitionKey = colorKey;
    }
    this.frameCounter++;
    // colour packets cost frames: update the colour field every other B-mode frame
    if (this.frameCounter % 2 === 0 || !this.prev) this.compute(ctx);
    return this.prev
      ? { vel: this.prev.vel, variance: this.prev.variance }
      : { vel: null, variance: null };
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
    // the colour field updates every other B-mode frame: its persistence decays over that nominal interval (decision 94)
    const colorElapsed = Number.isFinite(this._lastTimeS)
      ? ctx.timeS - this._lastTimeS
      : 2 * ctx.frameIntervalS;
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
    computeColorField(
      frame,
      {
        ...ctx.color,
        persistence: persistenceOverTime(
          ctx.color.persistence,
          colorElapsed,
          2 * ctx.frameIntervalS,
        ),
      },
      this.prev,
      (_idx, li, si, o) => {
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
        sampleFlow(flow, tables, hp, phase, hx, hy, hz, fs);
        o.v = -(fs.vx * dhx + fs.vy * dhy + fs.vz * dhz); // + = toward the transducer
        o.disp = fs.dispersion;
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
