import type { HeartModel } from '@/simulator/anatomy/heartModel';
import type { BeatTables } from '@/simulator/cardiac-cycle/cycleModel';
import type { BeamFrame, ProbeControl } from '@/simulator/probe/pose';
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
 * Owns the double-buffered Color Doppler polar field, packet cadence (every other B-mode frame),
 * acquisition invalidation key and per-line cached ray/basis flow sampling.
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
  private _version = 0;
  private _lastTimeS = NaN;
  private flowSample: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };

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
  }

  resetHistory(): void {
    this.prev = null;
  }

  /** Step Color Doppler for one B-mode frame; recomputes every other frame or when settings/geometry changed. */
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
