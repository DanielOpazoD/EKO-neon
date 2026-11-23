import { overlayColorField, type ColorField } from '@/simulator/doppler/color/colorDoppler';
import { spectralRange } from '@/simulator/doppler/spectral/spectrum';
import type { BeamFrame } from '@/simulator/probe/pose';
import type { Webgl2Renderer } from '@/simulator/renderer/gpu/webgl2Renderer';
import { isStripModality } from '@/simulator/renderer/modality';
import {
  buildScanLut,
  computeSectorMapping,
  lutKey,
  scanConvertLut,
  type ScanLut,
  type SectorMapping,
} from '@/simulator/renderer/scanConvert';
import type { PolarFrame, PolarFrameSpec, RendererBackend } from '@/simulator/renderer/types';
import type { ViewAnalysis } from '@/simulator/view-recognition/viewQuality';
import type { CineBuffer, CineFrame } from './cineBuffer';
import type { EcgRing } from './ecgRing';
import type { ExperimentalLoaders } from './experimentalLoaders';
import type { QualityTier, SimInput, SimOutput, StripInfo } from './protocol';
import type { StripCtx, StripEngine } from './stripEngine';

export interface CompositeTiming {
  renderFrameMs: number;
  compositeMs: number;
  analysisMs: number;
  consoleMs: number;
  cineMs: number;
  presentMs: number;
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
  gpuReason: string;
  onGpuContextLost: (reason: string) => void;
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
  backend: RendererBackend;
  isProceduralBackend: boolean;
  experimental: ExperimentalLoaders;
  tier: QualityTier;
  lastPersistence: number;
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
  private lut: ScanLut | null = null;

  recycle(buffer: ArrayBuffer): void {
    if (buffer.byteLength > 0 && this.rgbaPool.length < 3) this.rgbaPool.push(buffer);
  }

  takeBuffer(size: number): ArrayBuffer {
    for (let i = 0; i < this.rgbaPool.length; i++) {
      const b = this.rgbaPool[i]!;
      if (b.byteLength === size) {
        this.rgbaPool.splice(i, 1);
        return b;
      }
    }
    return new ArrayBuffer(size);
  }

  get poolSize(): number {
    return this.rgbaPool.length;
  }

  clear(): void {
    this.rgbaPool.length = 0;
    this.lut = null;
  }

  composite(ctx: CompositeContext): CompositeResult | null {
    const { input: inp, beam, spec, frozen, cf } = ctx;
    const W = Math.max(64, inp.display.width);
    const H = Math.max(64, inp.display.height);
    const modality = frozen ? (cf?.acquisition.modality ?? inp.modality) : inp.modality;
    const colorSettings = frozen ? (cf?.acquisition.color ?? inp.color) : inp.color;
    const isStrip = isStripModality(modality);
    const sectorH = isStrip ? Math.round(H * 0.42) : H;
    const display = cf ? cf.display : ctx.display;
    const fspec = cf ? cf.spec : spec;
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
    const colorPower =
      !cf && inp.modality === 'color' && ctx.colorCurrent ? ctx.colorCurrent.power : undefined;
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
      if (ctx.gpu.contextLost) ctx.onGpuContextLost('WebGL context lost: CPU tracer');
    }
    ctx.timing.presentMs = bitmap ? performance.now() - tp : 0;

    const buffer = bitmap ? new ArrayBuffer(0) : this.takeBuffer(W * H * 4);
    const rgba = new Uint8ClampedArray(buffer);
    if (!bitmap) {
      const sectorRgba = new Uint8ClampedArray(buffer, 0, W * sectorH * 4);
      scanConvertLut(display, this.lut, sectorRgba, tintMap);
      if (colorVel && colorVar)
        overlayColorField(sectorRgba, this.lut, colorVel, colorVar, colorSettings, colorPower);
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
      strip = ctx.strips.drawStrip(rgba, W, H, sectorH, tintMap);

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

    const output: SimOutput = {
      acquisition: cf?.acquisition,
      frameId: cf ? cf.frameId : ctx.frameId,
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
      stats: {
        ...ctx.backend.stats(),
        backend: ctx.backend.id,
        anatomy: ctx.experimental.anatomyStatus(inp.anatomy, ctx.isProceduralBackend),
        imagen: ctx.experimental.imageFormationStatus(inp.imageFormation),
        gpu: ctx.gpuReason,
        tier: ctx.tier,
        lines: spec.lines,
        samples: spec.samples,
        renderFrameMs: Number(ctx.timing.renderFrameMs.toFixed(1)),
        consoleMs: Number(ctx.timing.consoleMs.toFixed(1)),
        persistence: Number(ctx.lastPersistence.toFixed(3)),
        analysisMs: Number(ctx.timing.analysisMs.toFixed(1)),
        compositeMs: Number(ctx.timing.compositeMs.toFixed(1)),
        cineMs: Number(ctx.timing.cineMs.toFixed(2)),
        console: ctx.displayOnGpu ? 'gpu' : 'cpu',
        dopplerPipe: modality === 'color' ? (ctx.dopplerOnGpu ? 'gpu-deferred' : 'cpu') : 'off',
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
