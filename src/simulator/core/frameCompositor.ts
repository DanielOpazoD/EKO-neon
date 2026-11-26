import { overlayColorField, type ColorField } from '@/simulator/doppler/color/colorDoppler';
import { spectralRange } from '@/simulator/doppler/spectral/spectrum';
import type { BeamFrame } from '@/simulator/probe/pose';
import type { Webgl2Renderer } from '@/simulator/renderer/gpu/webgl2Renderer';
import { isSpectralModality, isStripModality } from '@/simulator/renderer/modality';
import {
  buildScanLut,
  computeSectorMapping,
  lutKey,
  scanConvertLut,
  type ScanLut,
  type SectorMapping,
} from '@/simulator/renderer/scanConvert';
import type { PolarFrame, PolarFrameSpec, TintMapId } from '@/simulator/renderer/types';
import type { ViewAnalysis } from '@/simulator/view-recognition/viewQuality';
import type { CineBuffer, CineFrame, FrameFormation } from './cineBuffer';
import type { EcgRing } from './ecgRing';
import type { SimInput, SimOutput, StripInfo } from './protocol';
import type { StripCtx, StripEngine } from './stripEngine';
import { resolveFrozenColorSettings } from './acquisitionKeys';

/** What composing and presenting an output cost (ms): the last composite's, and this one's present and colour overlay. */
export interface CompositeTiming {
  compositeMs: number;
  presentMs: number;
  overlayColorMs?: number;
}

export interface CompositeContext {
  input: SimInput;
  beam: BeamFrame;
  spec: PolarFrameSpec;
  frozen: boolean;
  cf?: CineFrame;
  display: Uint8ClampedArray | null;
  frame: PolarFrame | null;
  displayOnGpu: boolean;
  dopplerOnGpu: boolean;
  gpu: Webgl2Renderer | null;
  onGpuContextLost: () => void;
  /** How the frame shown was formed: a frozen one's own (`CineFrame.formation`), the pipeline's now when live. */
  formation: FrameFormation;
  /** The GPU port's state now, apart from the frame's: 'ok', lost (`GpuRecovery.status`), or why there is none. */
  gpuLive: string;
  colorCurrent: ColorField | null;
  colorVersion: number;
  colorFps: number;
  strips: StripEngine;
  stripCtx: () => StripCtx;
  lastAnalysis: ViewAnalysis | null;
  clockCurrent: { phase: number; beatIndex: number; rrS: number };
  timeS: number;
  frameId: number;
  frameIntervalS: number;
  acquisitionFps: number;
  ecgRing: EcgRing;
  cine: CineBuffer;
  /** This output's composition and presentation; what forming the frame cost is its `formation`'s. */
  timing: CompositeTiming;
}

export interface CompositeResult {
  output: SimOutput;
  sector: SectorMapping & { x: number; y: number };
  strip: StripInfo;
  displayedFrame: CineFrame | undefined;
}

/**
 * Owns scan-conversion LUT caching, transferable RGBA buffer pooling, GPU/CPU sector presentation,
 * color Doppler overlay, M-mode/spectral strip compositing, and `SimOutput` packaging.
 */
export class FrameCompositor {
  private rgbaPool: ArrayBuffer[] = [];
  /**
   * Byte size of the frames formed now: the pool keeps buffers of this size only. Buffers of an earlier display size
   * were kept as well, and three of them filled the pool for good; from then on every buffer of the current size that
   * came back was refused and each CPU frame allocated a new one, up to 3.4 MB at 30 Hz (700 MB in 12 s of PW across
   * six window sizes; 2 buffers per size now).
   */
  private rgbaBytes = 0;
  private lut: ScanLut | null = null;
  /**
   * The sector above a spectral strip, scan-converted once per 2D frame. The sweep moves at 30 Hz while PW, CW and TDI
   * keep a quarter of the 2D frame rate (17 Hz), and each tick converted the same frame again: 0.77 of the 1.11 ms of a
   * PW composite at 1024×820. A new frame, LUT (size, zoom, inversion) or tint map converts it again. M-mode keeps the
   * full 2D rate, a new frame nearly every tick, and converts directly.
   */
  private stripSector: {
    frameId: number;
    lut: ScanLut;
    tintMap: TintMapId | undefined;
    rgba: Uint8ClampedArray;
  } | null = null;

  recycle(buffer: ArrayBuffer): void {
    if (buffer.byteLength > 0 && buffer.byteLength === this.rgbaBytes && this.rgbaPool.length < 3)
      this.rgbaPool.push(buffer);
  }

  takeBuffer(size: number): ArrayBuffer {
    if (size !== this.rgbaBytes) {
      // the display was resized: the pooled buffers will never be asked for again
      this.rgbaPool.length = 0;
      this.rgbaBytes = size;
    }
    return this.rgbaPool.pop() ?? new ArrayBuffer(size);
  }

  get poolSize(): number {
    return this.rgbaPool.length;
  }

  clear(): void {
    this.rgbaPool.length = 0;
    this.rgbaBytes = 0;
    this.lut = null;
    this.stripSector = null;
  }

  /** The sector of frame `frameId` above a spectral strip, converted only when the frame, LUT or tint map changed. */
  private sectorAboveStrip(
    display: Uint8ClampedArray,
    frameId: number,
    lut: ScanLut,
    tintMap: TintMapId | undefined,
    bytes: number,
  ): Uint8ClampedArray {
    const s = this.stripSector;
    if (s && s.frameId === frameId && s.lut === lut && s.tintMap === tintMap) return s.rgba;
    const rgba = s && s.rgba.length === bytes ? s.rgba : new Uint8ClampedArray(bytes);
    scanConvertLut(display, lut, rgba, tintMap);
    this.stripSector = { frameId, lut, tintMap, rgba };
    return rgba;
  }

  composite(ctx: CompositeContext): CompositeResult | null {
    const { input: inp, beam, spec, frozen, cf } = ctx;
    const W = Math.max(64, inp.display.width);
    const H = Math.max(64, inp.display.height);
    const modality = frozen ? (cf?.acquisition.modality ?? inp.modality) : inp.modality;
    const colorSettings =
      frozen && cf?.acquisition.color
        ? resolveFrozenColorSettings(cf.acquisition.color, inp.color)
        : inp.color;
    const isStrip = isStripModality(modality);
    const sectorH = isStrip ? Math.round(H * 0.42) : H;
    const display = cf ? cf.display : ctx.display;
    const fspec = cf ? cf.spec : spec;
    const frameId = cf ? cf.frameId : ctx.frameId;
    if (!display) return null;

    const mapping = computeSectorMapping(
      fspec,
      W,
      sectorH,
      inp.settings.invertLR,
      inp.settings.zoom,
    );
    const key = lutKey(fspec, mapping);
    if (!this.lut || this.lut.key !== key) this.lut = buildScanLut(fspec, mapping);

    const colorVel = cf
      ? cf.colorVel
      : inp.modality === 'color' && ctx.colorCurrent
        ? ctx.colorCurrent.vel
        : null;
    const colorVar = cf
      ? cf.colorVar
      : inp.modality === 'color' && ctx.colorCurrent
        ? ctx.colorCurrent.variance
        : null;
    const colorPower = cf
      ? (cf.colorPower ?? undefined)
      : inp.modality === 'color' && ctx.colorCurrent
        ? ctx.colorCurrent.power
        : undefined;
    const tintMap = inp.settings.tintMap;

    let bitmap: ImageBitmap | null = null;
    const tp = performance.now();
    if (!frozen && !isStrip && ctx.displayOnGpu && ctx.gpu) {
      const color =
        modality === 'color' && ctx.dopplerOnGpu && ctx.gpu.hasDeferredDoppler
          ? ({ deferred: true, settings: colorSettings } as const)
          : colorVel && colorVar
            ? {
                vel: colorVel,
                variance: colorVar,
                power: colorPower,
                version: ctx.colorVersion,
                settings: colorSettings,
              }
            : null;
      bitmap = ctx.gpu.present({ lut: this.lut, width: W, height: sectorH, color, tintMap });
      if (ctx.gpu.contextLost) ctx.onGpuContextLost();
    }
    ctx.timing.presentMs = bitmap ? performance.now() - tp : 0;

    const buffer = bitmap ? new ArrayBuffer(0) : this.takeBuffer(W * H * 4);
    const rgba = new Uint8ClampedArray(buffer);
    ctx.timing.overlayColorMs = 0;
    if (!bitmap) {
      const sectorRgba = new Uint8ClampedArray(buffer, 0, W * sectorH * 4);
      if (isSpectralModality(modality) && !(colorVel && colorVar))
        sectorRgba.set(this.sectorAboveStrip(display, frameId, this.lut, tintMap, sectorRgba.length));
      else scanConvertLut(display, this.lut, sectorRgba, tintMap);
      if (colorVel && colorVar) {
        const tOv = performance.now();
        overlayColorField(sectorRgba, this.lut, colorVel, colorVar, colorSettings, colorPower);
        ctx.timing.overlayColorMs = performance.now() - tOv;
      }
    }
    if (isStrip) {
      rgba.fill(0, W * sectorH * 4);
      for (let i = W * sectorH * 4 + 3; i < rgba.length; i += 4) rgba[i] = 255;
    }

    let strip: StripInfo = {
      x: 0,
      y: sectorH,
      width: W,
      height: H - sectorH,
      secondsPerColumn: 0,
      headColumn: 0,
      columns: 0,
      topValue: 0,
      bottomValue: 0,
      kind: null,
    };
    if (isStrip && (!cf || cf.acquisition.stripAcquisitionId === ctx.strips.historyId))
      strip = ctx.strips.drawStrip(rgba, W, H, sectorH, tintMap, inp.color, inp.spectral);

    const view = cf ? cf.analysis : ctx.lastAnalysis;
    const c = ctx.clockCurrent;
    const structure = cf ? cf.structure : ctx.frame ? ctx.frame.structure : new Uint8Array(0);
    const segment = cf ? cf.segment : ctx.frame ? ctx.frame.segment : new Uint8Array(0);
    const gate = cf
      ? cf.gate
      : isStrip
        ? ctx.strips.gateInfo(beam, fspec, c.phase, structure, ctx.stripCtx())
        : null;
    const ecgTail = ctx.ecgRing.extractTail(frozen, ctx.timeS, ctx.cine.oldest?.timeS);
    const sector = { ...mapping, x: 0, y: 0 };
    const displayedFrame = cf ?? ctx.cine.newest;
    const f = ctx.formation;

    const output: SimOutput = {
      acquisition: cf?.acquisition,
      frameId,
      width: W,
      height: H,
      rgba: buffer,
      bitmap,
      sector,
      strip,
      polar: {
        lines: fspec.lines,
        samples: fspec.samples,
        sectorRad: fspec.sectorRad,
        depthCm: fspec.depthCm,
      },
      structure:
        frozen || inp.needStructure !== false ? new Uint8Array(structure) : new Uint8Array(0),
      segment: frozen || inp.needSegments !== false ? new Uint8Array(segment) : new Uint8Array(0),
      gate,
      timeS: cf ? cf.timeS : ctx.timeS,
      phase: cf ? cf.phase : c.phase,
      beatIndex: cf ? cf.beatIndex : c.beatIndex,
      heartRateBpm: 60 / (cf ? cf.rrS : c.rrS),
      rrS: cf ? cf.rrS : c.rrS,
      simulatedFps: ctx.acquisitionFps,
      cadenceHz: 1 / ctx.frameIntervalS,
      ecg: ecgTail,
      ecgHead: ctx.timeS,
      view,
      spectrumColumn: frozen && cf ? cf.spectrumColumn : ctx.strips.spectrumColumn,
      spectralRange:
        frozen && cf
          ? cf.spectralRange
          : (ctx.strips.velocityRange ?? spectralRange(inp.spectral)),
      frozen,
      cineLength: ctx.cine.length,
      cineOffset: cf ? inp.cineOffset : 0,
      cineFramePhase: cf ? cf.phase : c.phase,
      cineWindow: {
        startS: ctx.cine.oldest?.timeS ?? ctx.timeS,
        endS: ctx.cine.newest?.timeS ?? ctx.timeS,
        frameS: cf ? cf.timeS : ctx.timeS,
      },
      // the stats speak of the frame shown, frozen as it was formed (`FrameFormation`); only the port's state now and
      // what composes and presents this output are live
      stats: {
        ...f.backendStats,
        backend: f.backend,
        anatomy: f.anatomy,
        imagen: f.imagen,
        gpu: f.gpu,
        gpuLive: ctx.gpuLive,
        tier: f.tier,
        lines: spec.lines,
        samples: spec.samples,
        renderFrameMs: Number(f.renderFrameMs.toFixed(1)),
        consoleMs: Number(f.consoleMs.toFixed(1)),
        persistence: Number(f.persistence.toFixed(3)),
        analysisMs: Number(f.analysisMs.toFixed(1)),
        compositeMs: Number(ctx.timing.compositeMs.toFixed(1)),
        cineMs: Number(f.cineMs.toFixed(2)),
        console: f.console,
        dopplerPipe: f.dopplerPipe,
        present: bitmap ? 'gpu' : 'cpu',
        presentMs: Number(ctx.timing.presentMs.toFixed(2)),
      },
      colorFps: ctx.colorFps,
      probeBeam: {
        origin: [beam.origin.x, beam.origin.y, beam.origin.z],
        forward: [beam.forward.x, beam.forward.y, beam.forward.z],
        lateral: [beam.lateral.x, beam.lateral.y, beam.lateral.z],
        normal: [beam.normal.x, beam.normal.y, beam.normal.z],
      },
    };

    return { output, sector, strip, displayedFrame };
  }
}
