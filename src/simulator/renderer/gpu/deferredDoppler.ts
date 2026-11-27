import { aliasVelocity } from '@/clinical/formulas';
import { attenuationFrequencyMHz } from '../acoustic/acoustics';
import type { ColorPresentSettings, PolarFrameSpec } from '../types';
import {
  buildProgram,
  makeTexture2D,
  uniformLocations,
  UPLOAD_UNIT,
  type Locations,
} from './glHelpers';
import { COLOR_THRESHOLD } from '@/simulator/doppler/color/colorDoppler';
import {
  wallFilterShares,
  type WallFilterShares,
} from '@/simulator/doppler/color/kasaiEstimator';
import {
  DEFAULT_COLOR_SMOOTHING,
  ST_HYSTERESIS_RATIO,
} from '@/simulator/doppler/color/spatiotemporalFilter';
import {
  GLSL_DOPPLER_AUTOCORR_FRAG,
  GLSL_DOPPLER_SHADE_FRAG,
  GLSL_DOPPLER_SPATIAL_FRAG,
} from './glslDoppler';
import { GLSL_VERT } from './glslPasses';

export const DOPPLER_AUTOCORR_UNIFORMS = [
  'uGbufEnv',
  'uGbufIds',
  'uFlowRaw',
  'uSamples',
  'uLines',
  'uDepthCm',
  'uAttenFreqMHz',
  'uScaleMps',
  'uWallShares',
  'uBloomSamples',
  'uInvert',
  'uRealization',
  'uSeed',
] as const;

export const DOPPLER_SHADE_UNIFORMS = [
  'uCorr',
  'uHist',
  'uBoxBounds',
  'uScaleMps',
  'uBaselineMps',
  'uGainLin',
  'uPersist',
  'uSmoothing',
  'uUseHistory',
] as const;

export const DOPPLER_SPATIAL_UNIFORMS = [
  'uShade',
  'uCorr',
  'uBoxBounds',
  'uSmoothing',
  'uGainDb',
] as const;

/** GPU-resident Polar G-Buffer produced independently by B-mode anatomical reconstruction. */
export interface PolarGBuffer {
  readonly ready: boolean;
  readonly w: number;
  readonly h: number;
  readonly version: number;
  readonly envTex: WebGLTexture | null;
  readonly idsTex: WebGLTexture | null;
  readonly packedTex: WebGLTexture | null;
}

/** Polar texel bounds of the Color Doppler ROI box (inclusive min/max, plus scissor dimensions). */
export interface DopplerBoxBounds {
  siMin: number;
  siMax: number;
  liMin: number;
  liMax: number;
  boxW: number;
  boxH: number;
  totalTexels: number;
  coveragePct: number;
  empty: boolean;
}

/** Full Color Doppler shading parameters required by the deferred GPU pass. */
export interface DeferredDopplerSettings extends ColorPresentSettings {
  baselineShiftMps: number;
  wallFilterMps: number;
  gainDb: number;
  persistence: number;
  smoothing?: number;
  invert: boolean;
}

export interface DeferredDopplerRequest {
  spec: PolarFrameSpec;
  settings: DeferredDopplerSettings;
  frequencyMHz: number;
  harmonics: boolean;
  /**
   * Compact box-only RGBA float buffer (`boxW * boxH * 4`) holding `[vAxial, sigmaVMps, present, vWall]`
   * in row-major line order (`li = liMin..liMax`, `si = siMin..siMax`).
   * Omit to re-shade directly against the already-resident GPU raw flow texture when only Doppler
   * console settings (scale, baseline, wall filter, gain, persistence, invert) changed.
   */
  rawFlowBox?: Float32Array;
  /** Stochastic 8-pulse Kasai estimator realization index and seed. */
  estimate: { realization: number; seed: number };
  /** Explicitly invalidate temporal circular phasor history (e.g. probe movement or modality switch). */
  resetHistory?: boolean;
}

export interface DeferredDopplerStats {
  active: boolean;
  allocated: boolean;
  version: number;
  lastMs: number;
  boxPct: number;
  shadedTexels: number;
  independentPasses: number;
  totalPasses: number;
}

const ZERO4 = new Float32Array([0, 0, 0, 0]);
const WALL_SHARES: WallFilterShares = { mean: 0, trend: 0, quad: 0 };

/**
 * CPU colour power of a present-texture texel from its alpha, the inverse of Pass D3's `encodeDisplayAlpha` (and of
 * `GlPresentUploads.ensureColor`): alpha 0.5 + 0.5·c carries the power confidence c = colorPowerConfidence(p), alpha
 * 1 + b the breakthrough b, which the CPU power encodes as 1 + b too. The confidence is a smoothstep between
 * COLOR_THRESHOLD·ST_HYSTERESIS_RATIO and COLOR_THRESHOLD + 0.3, inverted in closed form (t = ½ − sin(asin(1 − 2c)/3)):
 * `overlayColorField` then gives a read-back field the opacity and breakthrough the present pass gave the texel. Power
 * outside that band is not recoverable and does not need to be: it maps to the same confidence.
 */
export function colorPowerFromPresentAlpha(alpha: number): number {
  if (alpha > 1) return alpha;
  const c = Math.min(1, Math.max(0, (alpha - 0.5) * 2));
  const lo = COLOR_THRESHOLD * ST_HYSTERESIS_RATIO;
  const hi = COLOR_THRESHOLD + 0.3;
  return lo + (hi - lo) * (0.5 - Math.sin(Math.asin(1 - 2 * c) / 3));
}

/**
 * Pure helper computing the exact inclusive polar ROI box bounds and scissor rect matching
 * `computeColorField` in `colorDoppler.ts`.
 */
export function computeDopplerBoxBounds(
  spec: PolarFrameSpec,
  color: Pick<
    ColorPresentSettings,
    'boxThetaMinRad' | 'boxThetaMaxRad' | 'boxRMinCm' | 'boxRMaxCm'
  >,
): DopplerBoxBounds {
  const { lines, samples, sectorRad, depthCm } = spec;
  if (lines <= 0 || samples <= 0 || sectorRad <= 0 || depthCm <= 0) {
    return {
      siMin: 0,
      siMax: -1,
      liMin: 0,
      liMax: -1,
      boxW: 0,
      boxH: 0,
      totalTexels: 0,
      coveragePct: 0,
      empty: true,
    };
  }
  const half = sectorRad / 2;
  const liMin = Math.max(0, Math.floor(((color.boxThetaMinRad + half) / sectorRad) * lines));
  const liMax = Math.min(
    lines - 1,
    Math.ceil(((color.boxThetaMaxRad + half) / sectorRad) * lines),
  );
  const siMin = Math.max(0, Math.floor((color.boxRMinCm / depthCm) * samples));
  const siMax = Math.min(samples - 1, Math.ceil((color.boxRMaxCm / depthCm) * samples));
  const boxW = Math.max(0, siMax - siMin + 1);
  const boxH = Math.max(0, liMax - liMin + 1);
  const totalTexels = boxW * boxH;
  const full = lines * samples;
  const coveragePct = full > 0 ? (totalTexels / full) * 100 : 0;
  return {
    siMin,
    siMax,
    liMin,
    liMax,
    boxW,
    boxH,
    totalTexels,
    coveragePct,
    empty: boxW === 0 || boxH === 0,
  };
}

/**
 * Independent WebGL2 Deferred Color Doppler Shading Pipeline.
 *
 * Operates completely decoupled from B-mode anatomical raymarching: consumes the GPU-resident
 * Polar G-Buffer (`envTex`, `idsTex`) and sparse box-only flow samples, executes scissor-clipped
 * physical Doppler shading (acoustic shadow gate, gain blooming, 8-pulse Kasai packet estimator,
 * regression wall filter, 3×3 resolution-cell averaging, Nyquist aliasing, and ping-pong circular
 * phasor persistence), and outputs an `RGBA16F` complex-phasor texture directly consumable by the
 * final deferred composite pass (`GLSL_PRESENT_FRAG`).
 *
 * All shaders, textures, and FBOs are allocated lazily on first Color Doppler use so 2D B-mode
 * incurs zero extra VRAM or startup compile latency.
 */
export class DeferredDopplerPipeline {
  private progAutocorr: WebGLProgram | null = null;
  private progShade: WebGLProgram | null = null;
  private progSpatial: WebGLProgram | null = null;
  private ua: Locations<typeof DOPPLER_AUTOCORR_UNIFORMS> | null = null;
  private us: Locations<typeof DOPPLER_SHADE_UNIFORMS> | null = null;
  private usp: Locations<typeof DOPPLER_SPATIAL_UNIFORMS> | null = null;

  private texFlowRaw: WebGLTexture | null = null;
  private texCorr: WebGLTexture | null = null;
  private texRawShade: WebGLTexture | null = null;
  private texOut: WebGLTexture | null = null;
  private texHist: [WebGLTexture | null, WebGLTexture | null] = [null, null];

  private fbCorr: WebGLFramebuffer | null = null;
  private fbShade: [WebGLFramebuffer | null, WebGLFramebuffer | null] = [null, null];
  private fbOut: WebGLFramebuffer | null = null;

  private w = 0;
  private h = 0;
  private current: 0 | 1 = 0;
  private _valid = false;
  private _version = 0;
  private historyKey = '';
  private rawFlowBoxValid = false;
  private lastBox: DopplerBoxBounds | null = null;
  private lastGBufferVersion = -1;

  private lastMs = 0;
  private lastBoxPct = 0;
  private lastShadedTexels = 0;
  private independentPasses = 0;
  private totalPasses = 0;
  private readbackBuf = new Float32Array(0);

  constructor(private readonly gl: WebGL2RenderingContext) {}

  get allocated(): boolean {
    return this.progAutocorr !== null && this.progSpatial !== null && this.fbCorr !== null;
  }

  get valid(): boolean {
    return this._valid && this.texOut !== null;
  }

  get version(): number {
    return this._version;
  }

  /** Latest GPU-resident `RGBA16F` complex-phasor Doppler texture `(cos φ, sin φ, var, powConf)`. */
  get outputTexture(): WebGLTexture | null {
    return this._valid ? this.texOut : null;
  }

  get hasRawFlow(): boolean {
    return this.rawFlowBoxValid && this.texFlowRaw !== null;
  }

  stats(): DeferredDopplerStats {
    return {
      active: this._valid,
      allocated: this.allocated,
      version: this._version,
      lastMs: Number(this.lastMs.toFixed(2)),
      boxPct: Number(this.lastBoxPct.toFixed(1)),
      shadedTexels: this.lastShadedTexels,
      independentPasses: this.independentPasses,
      totalPasses: this.totalPasses,
    };
  }

  resetHistory(): void {
    this.historyKey = '';
    this._valid = false;
  }

  /** Invalidate cached Doppler output when Polar G-Buffer dimensions change. */
  onGBufferResize(): void {
    this._valid = false;
    this.rawFlowBoxValid = false;
    this.historyKey = '';
    this.lastBox = null;
    this.lastGBufferVersion = -1;
  }

  private ensurePrograms(): void {
    if (this.progAutocorr && this.progShade && this.progSpatial) return;
    const gl = this.gl;
    this.progAutocorr = buildProgram(gl, GLSL_VERT, GLSL_DOPPLER_AUTOCORR_FRAG);
    this.progShade = buildProgram(gl, GLSL_VERT, GLSL_DOPPLER_SHADE_FRAG);
    this.progSpatial = buildProgram(gl, GLSL_VERT, GLSL_DOPPLER_SPATIAL_FRAG);
    this.ua = uniformLocations(gl, this.progAutocorr, DOPPLER_AUTOCORR_UNIFORMS);
    this.us = uniformLocations(gl, this.progShade, DOPPLER_SHADE_UNIFORMS);
    this.usp = uniformLocations(gl, this.progSpatial, DOPPLER_SPATIAL_UNIFORMS);

    gl.useProgram(this.progAutocorr);
    gl.uniform1i(this.ua.uGbufEnv, 2);
    gl.uniform1i(this.ua.uGbufIds, 3);
    gl.uniform1i(this.ua.uFlowRaw, 4);

    gl.useProgram(this.progShade);
    gl.uniform1i(this.us.uCorr, 2);
    gl.uniform1i(this.us.uHist, 3);

    gl.useProgram(this.progSpatial);
    gl.uniform1i(this.usp.uShade, 2);
    gl.uniform1i(this.usp.uCorr, 3);
  }

  private ensureTargets(w: number, h: number): boolean {
    if (this.w === w && this.h === h && this.fbCorr && this.fbOut) return false;
    this.disposeTargets();
    const gl = this.gl;
    const mk = (internal: number, format: number, type: number): WebGLTexture => {
      const t = makeTexture2D(gl);
      gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, null);
      return t;
    };
    const mkFb = (...targets: WebGLTexture[]): WebGLFramebuffer => {
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      const attachments = targets.map((t, i) => {
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0);
        return gl.COLOR_ATTACHMENT0 + i;
      });
      gl.drawBuffers(attachments);
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (status !== gl.FRAMEBUFFER_COMPLETE)
        throw new Error(`deferred doppler framebuffer incomplete: ${status}`);
      return fb;
    };

    this.texFlowRaw = mk(gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT);
    this.texCorr = mk(gl.RGBA32F, gl.RGBA, gl.FLOAT);
    this.texRawShade = mk(gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT);
    this.texOut = mk(gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT);
    const hist0 = mk(gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT);
    const hist1 = mk(gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT);
    this.texHist = [hist0, hist1];
    this.fbCorr = mkFb(this.texCorr);
    this.fbShade = [mkFb(this.texRawShade, hist0), mkFb(this.texRawShade, hist1)];
    this.fbOut = mkFb(this.texOut);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    this.w = w;
    this.h = h;
    this._valid = false;
    this.rawFlowBoxValid = false;
    this.historyKey = '';
    this.lastBox = null;
    return true;
  }

  private bindAt(unit: number, tex: WebGLTexture | null): void {
    this.gl.activeTexture(this.gl.TEXTURE0 + unit);
    this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
  }

  /**
   * Execute the independent 2-pass scissor-clipped deferred Doppler shading pipeline against the
   * GPU-resident Polar G-Buffer. Does NOT invoke or depend on B-mode geometry/envelope passes.
   */
  shade(gbuffer: PolarGBuffer, vao: WebGLVertexArrayObject, req: DeferredDopplerRequest): boolean {
    const gl = this.gl;
    if (!gbuffer.ready || !gbuffer.envTex || !gbuffer.idsTex) return false;
    const { spec, settings } = req;
    const w = spec.samples;
    const h = spec.lines;
    if (gbuffer.w !== w || gbuffer.h !== h) return false;

    const t0 = performance.now();
    this.ensurePrograms();
    this.ensureTargets(w, h);
    const ua = this.ua!;
    const us = this.us!;
    const usp = this.usp!;

    const box = computeDopplerBoxBounds(spec, settings);
    if (box.empty) {
      const next: 0 | 1 = this.current === 0 ? 1 : 0;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbShade[next]);
      gl.disable(gl.SCISSOR_TEST);
      gl.clearBufferfv(gl.COLOR, 0, ZERO4);
      gl.clearBufferfv(gl.COLOR, 1, ZERO4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbOut);
      gl.clearBufferfv(gl.COLOR, 0, ZERO4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this.current = next;
      this._valid = true;
      this._version++;
      this.lastBox = box;
      this.lastBoxPct = 0;
      this.lastShadedTexels = 0;
      this.lastMs = performance.now() - t0;
      return true;
    }

    // Upload ONLY the sparse ROI box sub-rectangle when new raw flow data is provided
    if (req.rawFlowBox) {
      const expectedLen = box.boxW * box.boxH * 4;
      if (req.rawFlowBox.length < expectedLen)
        throw new Error(
          `rawFlowBox length ${req.rawFlowBox.length} < required ${expectedLen} (${box.boxW}x${box.boxH})`,
        );
      const subData =
        req.rawFlowBox.length === expectedLen
          ? req.rawFlowBox
          : req.rawFlowBox.subarray(0, expectedLen);
      this.bindAt(UPLOAD_UNIT, this.texFlowRaw);
      gl.texSubImage2D(
        gl.TEXTURE_2D,
        0,
        box.siMin,
        box.liMin,
        box.boxW,
        box.boxH,
        gl.RGBA,
        gl.FLOAT,
        subData,
      );
      this.bindAt(UPLOAD_UNIT, null);
      this.rawFlowBoxValid = true;
    } else if (!this.rawFlowBoxValid) {
      return false;
    }

    // Circular phasor history is only valid under identical scale, baseline, inversion & resolution
    const nextHistKey = `${settings.scaleMps.toFixed(4)}|${settings.baselineShiftMps.toFixed(4)}|${settings.invert ? 1 : 0}|${w}x${h}`;
    const useHistory =
      !req.resetHistory &&
      this._valid &&
      this.historyKey === nextHistKey &&
      settings.persistence > 0;

    const bloomSamples = Math.max(0, Math.round((settings.gainDb - 2) * 0.6));
    const gainLin = Math.pow(10, settings.gainDb / 20);
    const attenFreqMHz = attenuationFrequencyMHz(req.frequencyMHz, req.harmonics);

    gl.viewport(0, 0, w, h);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(vao);

    // ---- Pass D1: G-Buffer shadow/blooming gate + Kasai 8-pulse autocorrelation (scissor-clipped!) ----
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbCorr);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(box.siMin, box.liMin, box.boxW, box.boxH);
    gl.useProgram(this.progAutocorr);
    gl.uniform1i(ua.uSamples, w);
    gl.uniform1i(ua.uLines, h);
    gl.uniform1f(ua.uDepthCm, spec.depthCm);
    gl.uniform1f(ua.uAttenFreqMHz, attenFreqMHz);
    gl.uniform1f(ua.uScaleMps, settings.scaleMps);
    // the regression wall filter's shares, from the function the CPU estimator uses: both cut at the same −3 dB point
    const wf = wallFilterShares(settings.wallFilterMps, settings.scaleMps, WALL_SHARES);
    gl.uniform3f(ua.uWallShares, wf.mean, wf.trend, wf.quad);
    gl.uniform1i(ua.uBloomSamples, bloomSamples);
    gl.uniform1i(ua.uInvert, settings.invert ? 1 : 0);
    gl.uniform1ui(ua.uRealization, req.estimate.realization >>> 0);
    gl.uniform1ui(ua.uSeed, req.estimate.seed >>> 0);
    this.bindAt(2, gbuffer.envTex);
    this.bindAt(3, gbuffer.idsTex);
    this.bindAt(4, this.texFlowRaw);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    const smoothing = Math.max(0, Math.min(1, settings.smoothing ?? DEFAULT_COLOR_SMOOTHING));

    // ---- Pass D2: Resolution-cell averaging + aliasing + adaptive temporal hysteresis persistence ----
    const next: 0 | 1 = this.current === 0 ? 1 : 0;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbShade[next]);
    // Fast TBDR tile-clear outside the ROI box so shrinking/moving the box leaves zero ghost pixels
    gl.disable(gl.SCISSOR_TEST);
    gl.clearBufferfv(gl.COLOR, 0, ZERO4);
    gl.clearBufferfv(gl.COLOR, 1, ZERO4);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(box.siMin, box.liMin, box.boxW, box.boxH);
    gl.useProgram(this.progShade);
    gl.uniform4i(us.uBoxBounds, box.siMin, box.liMin, box.siMax, box.liMax);
    gl.uniform1f(us.uScaleMps, settings.scaleMps);
    gl.uniform1f(us.uBaselineMps, settings.baselineShiftMps);
    gl.uniform1f(us.uGainLin, gainLin);
    gl.uniform1f(us.uPersist, useHistory ? settings.persistence : 0);
    gl.uniform1f(us.uSmoothing, smoothing);
    gl.uniform1i(us.uUseHistory, useHistory ? 1 : 0);
    this.bindAt(2, this.texCorr);
    this.bindAt(3, this.texHist[this.current]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // ---- Pass D3: Coherence- & variance-adaptive circular spatial filter (scissor-clipped!) ----
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbOut);
    gl.disable(gl.SCISSOR_TEST);
    gl.clearBufferfv(gl.COLOR, 0, ZERO4);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(box.siMin, box.liMin, box.boxW, box.boxH);
    gl.useProgram(this.progSpatial);
    gl.uniform4i(usp.uBoxBounds, box.siMin, box.liMin, box.siMax, box.liMax);
    gl.uniform1f(usp.uSmoothing, smoothing);
    gl.uniform1f(usp.uGainDb, settings.gainDb);
    this.bindAt(2, this.texRawShade);
    this.bindAt(3, this.texCorr);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.disable(gl.SCISSOR_TEST);
    for (let u = 2; u <= 4; u++) this.bindAt(u, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    if (this.lastGBufferVersion === gbuffer.version) this.independentPasses++;
    this.lastGBufferVersion = gbuffer.version;
    this.totalPasses++;
    this.current = next;
    this._valid = true;
    this._version++;
    this.historyKey = nextHistKey;
    this.lastBox = box;
    this.lastBoxPct = box.coveragePct;
    this.lastShadedTexels = box.totalTexels;
    this.lastMs = performance.now() - t0;
    return true;
  }

  /**
   * Optional on-demand readback of the shaded ROI box into CPU polar buffers (GPU-vs-CPU parity verification, and the
   * freeze of the frame on screen). Not called during live GPU presentation. `powOut` receives the CPU colour power that
   * `overlayColorField` blends exactly as the present pass blends the texel's alpha (`colorPowerFromPresentAlpha`).
   */
  readbackBox(
    spec: PolarFrameSpec,
    scaleMps: number,
    baselineShiftMps: number,
    velOut: Float32Array,
    varOut: Float32Array,
    powOut?: Float32Array,
  ): boolean {
    const gl = this.gl;
    const box = this.lastBox;
    if (!this._valid || !box || !this.fbOut) return false;
    velOut.fill(NaN);
    varOut.fill(0);
    powOut?.fill(0);
    if (box.empty) return true;

    const needed = box.boxW * box.boxH * 4;
    if (this.readbackBuf.length < needed) this.readbackBuf = new Float32Array(needed);
    const buf =
      this.readbackBuf.length === needed ? this.readbackBuf : this.readbackBuf.subarray(0, needed);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbOut);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(box.siMin, box.liMin, box.boxW, box.boxH, gl.RGBA, gl.FLOAT, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (gl.isContextLost()) return false;

    const scale = Math.max(1e-3, scaleMps);
    const invPiScale = scale / Math.PI;
    const samples = spec.samples;
    for (let dy = 0; dy < box.boxH; dy++) {
      const li = box.liMin + dy;
      const rowOff = dy * box.boxW * 4;
      const polarRow = li * samples + box.siMin;
      for (let dx = 0; dx < box.boxW; dx++) {
        const o = rowOff + dx * 4;
        if (buf[o + 3]! < 0.5) continue;
        const vRaw = baselineShiftMps + invPiScale * Math.atan2(buf[o + 1]!, buf[o]!);
        velOut[polarRow + dx] = aliasVelocity(vRaw, scale, baselineShiftMps);
        varOut[polarRow + dx] = buf[o + 2]!;
        if (powOut) powOut[polarRow + dx] = colorPowerFromPresentAlpha(buf[o + 3]!);
      }
    }
    return true;
  }

  private disposeTargets(): void {
    const gl = this.gl;
    for (const t of [
      this.texFlowRaw,
      this.texCorr,
      this.texRawShade,
      this.texOut,
      ...this.texHist,
    ])
      if (t) gl.deleteTexture(t);
    for (const f of [this.fbCorr, ...this.fbShade, this.fbOut]) if (f) gl.deleteFramebuffer(f);
    this.texFlowRaw = null;
    this.texCorr = null;
    this.texRawShade = null;
    this.texOut = null;
    this.texHist = [null, null];
    this.fbCorr = null;
    this.fbShade = [null, null];
    this.fbOut = null;
    this.w = 0;
    this.h = 0;
    this._valid = false;
    this.rawFlowBoxValid = false;
    this.lastBox = null;
  }

  dispose(): void {
    const gl = this.gl;
    if (this.progAutocorr) gl.deleteProgram(this.progAutocorr);
    if (this.progShade) gl.deleteProgram(this.progShade);
    if (this.progSpatial) gl.deleteProgram(this.progSpatial);
    this.progAutocorr = null;
    this.progShade = null;
    this.progSpatial = null;
    this.ua = null;
    this.us = null;
    this.usp = null;
    this.disposeTargets();
    this.historyKey = '';
    this.lastGBufferVersion = -1;
  }
}
