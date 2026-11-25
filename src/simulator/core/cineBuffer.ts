import type { BeamFrame } from '@/simulator/probe/pose';
import type { FrameAcquisition, GateInfo } from './protocol';
import type { Scene } from '@/simulator/renderer/types';
import type { PolarFrameSpec } from '@/simulator/renderer/types';
import type { ViewAnalysis } from '@/simulator/view-recognition/viewQuality';

export interface CineFrame {
  acquisition: FrameAcquisition;
  /** Models are replaced, not mutated, by the core; retain the acquired pose and model references. */
  scene: Scene;
  gate: GateInfo | null;
  spectrumColumn: Float32Array | null;
  spectralRange: { vMin: number; vMax: number };
  display: Uint8ClampedArray;
  structure: Uint8Array;
  segment: Uint8Array;
  spec: PolarFrameSpec;
  frameId: number;
  rrS: number;
  phase: number;
  timeS: number;
  beatIndex: number;
  colorVel: Float32Array | null;
  colorVar: Float32Array | null;
  colorPower?: Float32Array | null;
  /** Compact G-buffer snapshot for lazy CPU Color Doppler reconstruction when scrubbing frozen GPU-deferred cine. */
  colorTissue?: Uint8Array | null;
  colorTransCode?: Uint8Array | null;
  beam: BeamFrame;
  analysis: ViewAnalysis | null;
}

/** Bounded acquisition history. Capture owns its arrays; later renderer reuse cannot rewrite a frame. */
export class CineBuffer {
  private frames: (CineFrame | undefined)[];
  private head = 0;
  private count = 0;

  constructor(private readonly capacity = 96) {
    if (!Number.isInteger(capacity) || capacity < 1)
      throw new RangeError('Cine capacity must be positive');
    this.frames = new Array<CineFrame | undefined>(capacity);
  }

  get length(): number {
    return this.count;
  }
  get oldest(): CineFrame | undefined {
    return this.at(0);
  }
  get newest(): CineFrame | undefined {
    return this.at(this.count - 1);
  }

  private at(index: number): CineFrame | undefined {
    if (index < 0 || index >= this.count) return undefined;
    return this.frames[(this.head - this.count + index + this.capacity) % this.capacity];
  }

  select(offset: number): CineFrame | undefined {
    const safeOffset = Number.isFinite(offset) ? Math.trunc(offset) : 0;
    return this.at(Math.max(0, Math.min(this.count - 1, this.count - 1 + safeOffset)));
  }

  /**
   * Find the cine offset (`-(count-1)..0`) closest to `targetPhase` (0..1, circular distance),
   * preferring the most recent cardiac cycle when multiple beats match closely.
   */
  findNearestPhaseOffset(targetPhase: number): { offset: number; phase: number } | null {
    if (this.count === 0) return null;
    const t = ((targetPhase % 1) + 1) % 1;
    let bestOffset = 0;
    let bestPhase = 0;
    let bestScore = Infinity;
    // Search newest → oldest; tiny recency tie-breaker (0.012) picks the latest beat's ED/ES cleanly
    for (let i = this.count - 1; i >= 0; i--) {
      const f = this.at(i);
      if (!f) continue;
      const dRaw = Math.abs(f.phase - t);
      const d = Math.min(dRaw, 1 - dRaw);
      const agePenalty = ((this.count - 1 - i) / Math.max(1, this.count)) * 0.012;
      const score = d + agePenalty;
      if (score < bestScore) {
        bestScore = score;
        bestOffset = i - (this.count - 1);
        bestPhase = f.phase;
      }
    }
    return { offset: bestOffset, phase: bestPhase };
  }

  capture(frame: CineFrame): void {
    const prev = this.frames[this.head];
    const display = reuseU8Clamped(prev?.display, frame.display);
    const structure = reuseU8(prev?.structure, frame.structure);
    const segment = reuseU8(prev?.segment, frame.segment);
    const spectrumColumn = frame.spectrumColumn
      ? reuseF32(prev?.spectrumColumn ?? undefined, frame.spectrumColumn)
      : null;
    const colorVel = frame.colorVel ? reuseF32(prev?.colorVel ?? undefined, frame.colorVel) : null;
    const colorVar = frame.colorVar ? reuseF32(prev?.colorVar ?? undefined, frame.colorVar) : null;
    const colorPower = frame.colorPower
      ? reuseF32(prev?.colorPower ?? undefined, frame.colorPower)
      : null;
    const colorTissue = frame.colorTissue
      ? reuseU8(prev?.colorTissue ?? undefined, frame.colorTissue)
      : null;
    const colorTransCode = frame.colorTransCode
      ? reuseU8(prev?.colorTransCode ?? undefined, frame.colorTransCode)
      : null;

    this.frames[this.head] = {
      ...frame,
      acquisition: {
        ...frame.acquisition,
        probe: { ...frame.acquisition.probe },
        patient: { ...frame.acquisition.patient },
        settings: { ...frame.acquisition.settings, tgcDb: [...frame.acquisition.settings.tgcDb] },
        color: { ...frame.acquisition.color },
        spectral: { ...frame.acquisition.spectral },
        phaseMarks: { ...frame.acquisition.phaseMarks },
      },
      gate: frame.gate ? { ...frame.gate, lineStructures: [...frame.gate.lineStructures] } : null,
      spectrumColumn,
      spectralRange: { ...frame.spectralRange },
      display,
      structure,
      segment,
      spec: { ...frame.spec },
      beam: {
        ...frame.beam,
        origin: { ...frame.beam.origin },
        forward: { ...frame.beam.forward },
        lateral: { ...frame.beam.lateral },
        normal: { ...frame.beam.normal },
      },
      colorVel,
      colorVar,
      colorPower,
      colorTissue,
      colorTransCode,
    };
    this.head = (this.head + 1) % this.capacity;
    this.count = Math.min(this.capacity, this.count + 1);
  }
}

function reuseU8Clamped(
  existing: Uint8ClampedArray | undefined,
  src: Uint8ClampedArray,
): Uint8ClampedArray {
  if (existing && existing.length === src.length) {
    existing.set(src);
    return existing;
  }
  return src.slice();
}

function reuseU8(existing: Uint8Array | undefined, src: Uint8Array): Uint8Array {
  if (existing && existing.length === src.length) {
    existing.set(src);
    return existing;
  }
  return src.slice();
}

function reuseF32(existing: Float32Array | undefined, src: Float32Array): Float32Array {
  if (existing && existing.length === src.length) {
    existing.set(src);
    return existing;
  }
  return src.slice();
}

