import type { BeamFrame } from '@/simulator/probe/pose';
import { COMPOUND_LOOKS } from '../acoustic/acoustics';
import { buildPsfKernels, psfKey, sideLobeLevelDb } from '../acoustic/psf';
import type { ConsoleState } from '../postprocess/consolePipeline';
import { tintMapIndex } from '../postprocess/tintMap';
import type { ScanLut } from '../scanConvert';
import { TRANS_DECODE } from '../transmissionCode';
import type {
  AcquisitionSettings,
  ColorPresentSettings,
  DisplayConsole,
  PolarFrame,
  PolarFrameSpec,
  RendererBackend,
  RenderHints,
  Scene,
  TintMapId,
} from '../types';
import {
  buildNoiseKernelsCached,
  buildProgram,
  CONSOLE_UNIFORMS,
  initPassAUniforms,
  LOG_UNIFORMS,
  makeTexture2D,
  NOISE_AXIAL_UNIFORMS,
  NOISE_LATERAL_UNIFORMS,
  PRESENT_UNIFORMS,
  SOFTWARE_GL,
  uniformLocations,
  UPLOAD_UNIT,
  uploadKernelTable,
  uploadNoiseLattice3D,
  webglRendererName,
  type Locations,
} from './glHelpers';
import { GLSL_COMMON } from './glslCommon';
import { GLSL_HEART } from './glslHeart';
import {
  GLSL_CONSOLE_FRAG,
  GLSL_LOG_FRAG,
  GLSL_NOISE_AXIAL_FRAG,
  GLSL_NOISE_LATERAL_FRAG,
  GLSL_PRESENT_FRAG,
} from './glslImage';
import {
  GLSL_PASS_A_MAIN,
  GLSL_PASS_A_SIDE_MAIN,
  GLSL_PASS_B_MAIN,
  GLSL_PASS_C_MAIN,
  GLSL_PASS_D_MAIN,
  GLSL_PASS_L_MAIN,
  GLSL_PASS_P_MAIN,
  GLSL_VERT,
} from './glslPasses';
import { GLSL_THORAX } from './glslThorax';
import { GlPresentUploads } from './glPresentUploads';
import { GlRenderTargets } from './glTargets';
import { allocPacked, packScene, PARAM_TEXELS, type PackedScene } from './paramLayout';

export { webglRendererName };

/** Colour field blended by the present pass: velocity per polar sample (NaN = no colour) and variance. */
export interface PresentColor {
  vel: Float32Array;
  variance: Float32Array;
  /** Changes whenever the field is recomputed, so an unchanged field is not uploaded again. */
  version: number;
  settings: ColorPresentSettings;
}

export interface PresentRequest {
  lut: ScanLut;
  width: number;
  height: number;
  color: PresentColor | null;
  tintMap?: TintMapId;
  /** Equivalence checks: read the drawn RGBA (top row first) into this array instead of returning an ImageBitmap. */
  readback?: Uint8Array;
}

/**
 * WebGL2 procedural renderer: the same scanline model and image formation as ProceduralSliceRenderer,
 * evaluated in four fragment passes — classification and local acoustics in parallel (A), per-line
 * transmission march into a complex signal (B), axial PSF (C), lateral PSF and envelope (D) — and read back
 * into the CPU PolarFrame (`render`), or continued on the GPU through the console into a packed display, read back
 * with the ids of pass B for the LV segment map (decision 152), and a present pass that scan-converts it into the
 * canvas (`renderDisplay`, `present`, decision 54).
 */
export class Webgl2Renderer implements RendererBackend {
  readonly id = 'webgl2-procedural' as const;
  private gl: WebGL2RenderingContext;
  private progA: WebGLProgram;
  private progASide: WebGLProgram;
  private passAVariants = new Map<string, WebGLProgram>();
  private passASideVariants = new Map<string, WebGLProgram>();
  private uLooksD: WebGLUniformLocation | null;
  private common = '';
  private progB: WebGLProgram;
  private progL: WebGLProgram;
  private progP: WebGLProgram;
  private progC: WebGLProgram;
  private progD: WebGLProgram;
  private progLog: WebGLProgram;
  private progConsole: WebGLProgram;
  private progPresent: WebGLProgram;
  private progNoiseA: WebGLProgram;
  private progNoiseL: WebGLProgram;
  private una: Locations<typeof NOISE_AXIAL_UNIFORMS>;
  private unl: Locations<typeof NOISE_LATERAL_UNIFORMS>;
  private ul: Locations<typeof LOG_UNIFORMS>;
  private uc: Locations<typeof CONSOLE_UNIFORMS>;
  private up: Locations<typeof PRESENT_UNIFORMS>;
  private noiseKernelTex: WebGLTexture;
  private noiseKernelKey = '';
  private paramsTex: WebGLTexture;
  private noiseTex: WebGLTexture;
  private psfTex: WebGLTexture;
  private psfKeyUploaded = '';
  private noiseSeed = NaN;
  private targets: GlRenderTargets;
  private uploads: GlPresentUploads;
  private histCurrent: 0 | 1 = 0;
  private histUpload = new Float32Array(0);
  private disposed = false;
  private histValid = false;
  private packedFresh = false;
  private lastPresentMs = 0;
  private lastOutput: 'float' | 'display' = 'float';
  private lastSegRead = true;
  private packed: PackedScene = allocPacked();
  private lastMs = 0;
  private lastGpuMs = 0;
  private lastReadMs = 0;
  private uElevK: WebGLUniformLocation | null;
  private uElevKSide: WebGLUniformLocation | null;
  private vao: WebGLVertexArrayObject;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    const ext = gl.getExtension('EXT_color_buffer_float');
    if (!ext) throw new Error('EXT_color_buffer_float unavailable');
    const common = `#version 300 es\n${GLSL_COMMON}`;
    this.common = common;
    this.progA = buildProgram(
      gl,
      GLSL_VERT,
      `${common}${GLSL_HEART}${GLSL_THORAX}${GLSL_PASS_A_MAIN}`,
    );
    this.progASide = buildProgram(
      gl,
      GLSL_VERT,
      `${common}${GLSL_HEART}${GLSL_THORAX}${GLSL_PASS_A_SIDE_MAIN}`,
    );
    this.progB = buildProgram(gl, GLSL_VERT, `${common}${GLSL_PASS_B_MAIN}`);
    this.progL = buildProgram(gl, GLSL_VERT, `${common}${GLSL_PASS_L_MAIN}`);
    this.progP = buildProgram(gl, GLSL_VERT, `${common}${GLSL_PASS_P_MAIN}`);
    this.progC = buildProgram(gl, GLSL_VERT, `${common}${GLSL_PASS_C_MAIN}`);
    this.progD = buildProgram(gl, GLSL_VERT, `${common}${GLSL_PASS_D_MAIN}`);
    this.progLog = buildProgram(gl, GLSL_VERT, GLSL_LOG_FRAG);
    this.progConsole = buildProgram(gl, GLSL_VERT, GLSL_CONSOLE_FRAG);
    this.progPresent = buildProgram(gl, GLSL_VERT, GLSL_PRESENT_FRAG);
    this.progNoiseA = buildProgram(gl, GLSL_VERT, GLSL_NOISE_AXIAL_FRAG);
    this.progNoiseL = buildProgram(gl, GLSL_VERT, GLSL_NOISE_LATERAL_FRAG);
    this.una = uniformLocations(gl, this.progNoiseA, NOISE_AXIAL_UNIFORMS);
    this.unl = uniformLocations(gl, this.progNoiseL, NOISE_LATERAL_UNIFORMS);
    this.ul = uniformLocations(gl, this.progLog, LOG_UNIFORMS);
    this.uc = uniformLocations(gl, this.progConsole, CONSOLE_UNIFORMS);
    this.up = uniformLocations(gl, this.progPresent, PRESENT_UNIFORMS);
    this.targets = new GlRenderTargets(gl);
    this.uploads = new GlPresentUploads(gl);
    this.noiseKernelTex = makeTexture2D(gl);
    this.vao = gl.createVertexArray()!;
    this.paramsTex = makeTexture2D(gl);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, PARAM_TEXELS, 1, 0, gl.RGBA, gl.FLOAT, null);
    this.noiseTex = gl.createTexture()!;
    this.psfTex = makeTexture2D(gl);
    this.uElevK = gl.getUniformLocation(this.progA, 'uElevK');
    this.uElevKSide = gl.getUniformLocation(this.progASide, 'uElevK');
    initPassAUniforms(gl, this.progA);
    initPassAUniforms(gl, this.progASide);
    for (const prog of [this.progL, this.progP, this.progB, this.progC, this.progD]) {
      gl.useProgram(prog);
      gl.uniform1i(gl.getUniformLocation(prog, 'uParams'), 0);
      gl.uniform1i(gl.getUniformLocation(prog, 'uNoise'), 1);
    }
    gl.useProgram(this.progB);
    for (const [name, unit] of [
      ['uPassA', 2],
      ['uPassB', 3],
      ['uSideA', 4],
      ['uSideB', 5],
      ['uPassC', 6],
      ['uDead', 7],
      ['uPassD', 8],
    ] as const) {
      gl.uniform1i(gl.getUniformLocation(this.progB, name), unit);
    }
    gl.useProgram(this.progL);
    gl.uniform1i(gl.getUniformLocation(this.progL, 'uPassA'), 2);
    gl.useProgram(this.progP);
    gl.uniform1i(gl.getUniformLocation(this.progP, 'uPassA'), 2);
    gl.uniform1i(gl.getUniformLocation(this.progP, 'uDead'), 3);
    gl.useProgram(this.progC);
    gl.uniform1i(gl.getUniformLocation(this.progC, 'uSig'), 2);
    gl.uniform1i(gl.getUniformLocation(this.progC, 'uPsf'), 3);
    gl.uniform1i(gl.getUniformLocation(this.progC, 'uSig2'), 4);
    gl.useProgram(this.progD);
    gl.uniform1i(gl.getUniformLocation(this.progD, 'uAx'), 2);
    gl.uniform1i(gl.getUniformLocation(this.progD, 'uPsf'), 3);
    gl.uniform1i(gl.getUniformLocation(this.progD, 'uAx2'), 4);
    this.uLooksD = gl.getUniformLocation(this.progD, 'uLooks');
    gl.useProgram(this.progLog);
    gl.uniform1i(this.ul.uEnv, 2);
    gl.uniform1i(this.ul.uComp, 3);
    gl.uniform1i(this.ul.uNoise, 4);
    gl.useProgram(this.progConsole);
    gl.uniform1i(this.uc.uLog, 2);
    gl.uniform1i(this.uc.uEnv, 3);
    gl.uniform1i(this.uc.uIds, 4);
    gl.uniform1i(this.uc.uHist, 5);
    gl.useProgram(this.progNoiseA);
    gl.uniform1i(this.una.uKernels, 2);
    gl.useProgram(this.progNoiseL);
    gl.uniform1i(this.unl.uNoiseAx, 2);
    gl.uniform1i(this.unl.uKernels, 3);
    gl.useProgram(this.progPresent);
    gl.uniform1i(this.up.uPacked, 2);
    gl.uniform1i(this.up.uLut, 3);
    gl.uniform1i(this.up.uColor, 4);
    gl.uniform1i(this.up.uPolar, 5);
  }

  get contextLost(): boolean {
    return this.disposed || this.gl.isContextLost();
  }

  stats(): Record<string, number | string> {
    return {
      renderMs: Number(this.lastMs.toFixed(2)),
      gpuMs: Number(this.lastGpuMs.toFixed(2)),
      readMs: Number(this.lastReadMs.toFixed(2)),
      presentMs: Number(this.lastPresentMs.toFixed(2)),
      output: this.lastOutput,
      segRead: this.lastSegRead ? 1 : 0,
      gpu: 'webgl2',
    };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const gl = this.gl;
    for (const p of [
      ...this.passAVariants.values(),
      ...this.passASideVariants.values(),
      this.progA,
      this.progASide,
      this.progL,
      this.progP,
      this.progB,
      this.progC,
      this.progD,
      this.progLog,
      this.progConsole,
      this.progPresent,
      this.progNoiseA,
      this.progNoiseL,
    ])
      gl.deleteProgram(p);
    this.uploads.dispose();
    gl.deleteTexture(this.paramsTex);
    gl.deleteTexture(this.noiseTex);
    gl.deleteTexture(this.psfTex);
    gl.deleteTexture(this.noiseKernelTex);
    this.targets.dispose();
    this.histValid = false;
    this.packedFresh = false;
    this.psfKeyUploaded = this.noiseKernelKey = '';
    this.noiseSeed = NaN;
    (gl.getExtension('WEBGL_lose_context') as { loseContext: () => void } | null)?.loseContext();
  }

  private ensureNoise(seed: number): void {
    if (this.noiseSeed === seed) return;
    uploadNoiseLattice3D(this.gl, this.noiseTex, seed);
    this.noiseSeed = seed;
  }

  private ensurePsf(scene: Scene, spec: PolarFrameSpec): void {
    const { frequencyMHz, harmonics } = scene.physics;
    const sg = scene.scatterGpu;
    if (sg) {
      const k = sg.kernels(spec, frequencyMHz, harmonics);
      if (this.psfKeyUploaded === k.key) return;
      uploadKernelTable(this.gl, this.psfTex, k, spec);
      this.psfKeyUploaded = k.key;
      return;
    }
    const bw = scene.physics.beamWidth ?? 0;
    const lobes = sideLobeLevelDb(scene.physics.sideLobe);
    const key = psfKey(spec, frequencyMHz, harmonics, bw, lobes);
    if (this.psfKeyUploaded === key) return;
    uploadKernelTable(
      this.gl,
      this.psfTex,
      buildPsfKernels(spec, frequencyMHz, harmonics, bw, lobes),
      spec,
    );
    this.psfKeyUploaded = key;
  }

  private ensureNoiseKernels(spec: PolarFrameSpec, settings: AcquisitionSettings): void {
    const k = buildNoiseKernelsCached(spec, settings);
    if (this.noiseKernelKey === k.key) return;
    uploadKernelTable(this.gl, this.noiseKernelTex, k, spec);
    this.noiseKernelKey = k.key;
  }

  private passAFor(scene: Scene): WebGLProgram {
    const hv = scene.heartVolumeGpu;
    const sg = scene.scatterGpu;
    if (!hv && !sg) return this.progA;
    const key = `${hv?.key ?? 'analytic'}|${sg?.key ?? 'lattice'}`;
    let prog = this.passAVariants.get(key);
    if (!prog) {
      const gl = this.gl;
      const defines = `${hv ? '#define VOLUME_HEART\n' : ''}${sg ? '#define SCATTER_FIELD\n' : ''}`;
      prog = buildProgram(
        gl,
        GLSL_VERT,
        `${this.common}${defines}${GLSL_HEART}${GLSL_THORAX}${hv?.glsl ?? ''}${sg?.glsl ?? ''}${GLSL_PASS_A_MAIN}`,
      );
      initPassAUniforms(gl, prog);
      hv?.upload(gl);
      this.passAVariants.set(key, prog);
    }
    return prog;
  }

  private passASideFor(scene: Scene): WebGLProgram {
    const hv = scene.heartVolumeGpu;
    if (!hv) return this.progASide;
    let prog = this.passASideVariants.get(hv.key);
    if (!prog) {
      const gl = this.gl;
      prog = buildProgram(
        gl,
        GLSL_VERT,
        `${this.common}#define VOLUME_HEART\n${GLSL_HEART}${GLSL_THORAX}${hv.glsl}${GLSL_PASS_A_SIDE_MAIN}`,
      );
      initPassAUniforms(gl, prog);
      hv.upload(gl);
      this.passASideVariants.set(hv.key, prog);
    }
    return prog;
  }

  private bindAt(unit: number, tex: WebGLTexture | null): void {
    this.gl.activeTexture(this.gl.TEXTURE0 + unit);
    this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
  }

  private formEnvelope(scene: Scene, beam: BeamFrame, spec: PolarFrameSpec): void {
    const gl = this.gl;
    const t = this.targets;
    const w = spec.samples,
      h = spec.lines;
    this.ensureNoise(scene.physics.seed);
    if (t.ensure(w, h)) {
      this.histValid = false;
      this.packedFresh = false;
    }
    this.ensurePsf(scene, spec);
    packScene(scene, beam, spec, this.packed);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.paramsTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, PARAM_TEXELS, 1, gl.RGBA, gl.FLOAT, this.packed.data);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, this.noiseTex);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(this.vao);

    // Pass A side planes (lightweight single RGBA16F target)
    if (spec.elevationSamples > 1) {
      const progSide = this.passASideFor(scene);
      gl.useProgram(progSide);
      const uElevKSide =
        progSide === this.progASide ? this.uElevKSide : gl.getUniformLocation(progSide, 'uElevK');
      if (scene.heartVolumeGpu) {
        scene.heartVolumeGpu.bind(gl, progSide, 2);
        scene.heartVolumeGpu.frame(gl, progSide, scene.heartPose);
      }
      gl.uniform1f(uElevKSide, -1);
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbS0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.uniform1f(uElevKSide, 1);
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbS1);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    // Pass A central plane
    const progA = this.passAFor(scene);
    gl.useProgram(progA);
    const uElevK = progA === this.progA ? this.uElevK : gl.getUniformLocation(progA, 'uElevK');
    if (scene.heartVolumeGpu) {
      scene.heartVolumeGpu.bind(gl, progA, 2);
      scene.heartVolumeGpu.frame(gl, progA, scene.heartPose);
    }
    scene.scatterGpu?.frame(gl, progA, scene.physics);
    gl.uniform1f(uElevK, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbA);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Pass L: pencil transmission, pleural entry index and body-sample count
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbL);
    gl.useProgram(this.progL);
    this.bindAt(2, t.texA0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Pass P: beam-width averaged transmission
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbP);
    gl.useProgram(this.progP);
    this.bindAt(2, t.texA0);
    this.bindAt(3, t.texL0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Pass B: complex signal + structure/tissue/segment ids
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbB);
    gl.useProgram(this.progB);
    this.bindAt(2, t.texP0);
    this.bindAt(3, t.texA1);
    this.bindAt(4, t.texS0);
    this.bindAt(5, t.texS1);
    this.bindAt(6, t.texA2);
    this.bindAt(7, t.texL0);
    this.bindAt(8, t.texA3);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Pass C: axial PSF
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbC);
    gl.useProgram(this.progC);
    this.bindAt(2, t.texB0);
    this.bindAt(3, this.psfTex);
    this.bindAt(4, t.texB2);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Pass D: lateral PSF and compounded envelope
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbD);
    gl.useProgram(this.progD);
    gl.uniform1f(this.uLooksD, scene.scatterGpu?.looks ?? COMPOUND_LOOKS);
    this.bindAt(2, t.texC0);
    this.bindAt(3, this.psfTex);
    this.bindAt(4, t.texC1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    for (let u = 2; u <= 8; u++) this.bindAt(u, null);
  }

  render(
    scene: Scene,
    beam: BeamFrame,
    spec: PolarFrameSpec,
    _phase: number,
    out: PolarFrame,
  ): void {
    if (this.disposed) throw new Error('Webgl2Renderer used after dispose');
    const t0 = performance.now();
    const gl = this.gl;
    const t = this.targets;
    const w = spec.samples,
      h = spec.lines;
    this.formEnvelope(scene, beam, spec);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    const tRead = performance.now();
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, t.readAmp);
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbB);
    gl.readBuffer(gl.COLOR_ATTACHMENT1);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, t.readIds);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const tDone = performance.now();
    this.lastGpuMs = tRead - t0;
    this.lastReadMs = tDone - tRead;
    const n = w * h;
    const amp = out.amplitude,
      tr = out.transmission,
      st = out.structure,
      ti = out.tissue,
      sg = out.segment;
    const ra = t.readAmp,
      ri = t.readIds;
    for (let i = 0; i < n; i++) {
      amp[i] = ra[i * 4]!;
      tr[i] = ra[i * 4 + 1]!;
      st[i] = ri[i * 4]!;
      ti[i] = ri[i * 4 + 1]!;
      sg[i] = ri[i * 4 + 3]!;
    }
    this.packedFresh = false;
    this.lastOutput = 'float';
    this.lastMs = performance.now() - t0;
  }

  renderDisplay(
    scene: Scene,
    beam: BeamFrame,
    spec: PolarFrameSpec,
    _phase: number,
    out: PolarFrame,
    _hints: RenderHints | undefined,
    con: DisplayConsole,
    display: Uint8ClampedArray,
  ): boolean {
    const gl = this.gl;
    if (this.disposed || gl.isContextLost()) return false;
    const t0 = performance.now();
    const t = this.targets;
    const w = spec.samples,
      h = spec.lines;
    this.formEnvelope(scene, beam, spec);
    const s = con.settings;
    this.uploads.ensureCompensation(s, spec);
    let useHistory = con.state.gpuHistory && this.histValid && s.persistence > 0;
    if (!con.state.gpuHistory && s.persistence > 0 && con.state.prev?.length === w * h) {
      this.uploadHistory(con.state.prev, w, h);
      useHistory = true;
    }
    // Receiver noise: axial + lateral passes
    this.ensureNoiseKernels(spec, s);
    gl.viewport(0, 0, w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbNA);
    gl.useProgram(this.progNoiseA);
    gl.uniform1i(this.una.uSamples, w);
    gl.uniform1ui(this.una.uSeed, con.state.seed >>> 0);
    gl.uniform1ui(this.una.uFrameIndex, con.state.frameIndex >>> 0);
    this.bindAt(2, this.noiseKernelTex);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbNL);
    gl.useProgram(this.progNoiseL);
    gl.uniform1i(this.unl.uLines, h);
    this.bindAt(2, t.texNA);
    this.bindAt(3, this.noiseKernelTex);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Console pass 1: detection + compensation + log compression
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbLog);
    gl.useProgram(this.progLog);
    gl.uniform1f(this.ul.uDynRange, s.dynamicRangeDb);
    this.bindAt(2, t.texD0);
    this.bindAt(3, this.uploads.compTex);
    this.bindAt(4, t.texNL);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Console pass 2: anisotropic SRI + edge enhancement + persistence + grey map
    const next = this.histCurrent === 0 ? 1 : 0;
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbConsole[next]);
    gl.useProgram(this.progConsole);
    const u = this.uc;
    gl.uniform1i(u.uSamples, w);
    gl.uniform1i(u.uLines, h);
    gl.uniform1f(u.uSri, s.sri && s.sri > 0 ? Math.min(1, s.sri) : 0);
    gl.uniform1f(u.uEdge, s.edgeEnhance > 0 ? s.edgeEnhance * 0.8 : 0);
    gl.uniform1f(u.uPersist, useHistory ? s.persistence : 0);
    gl.uniform1i(
      u.uGrayMap,
      s.grayMap === 's-curve'
        ? 1
        : s.grayMap === 'high-contrast'
          ? 2
          : s.grayMap === 'clinical'
            ? 3
            : 0,
    );
    this.bindAt(2, t.texLog);
    this.bindAt(3, t.texD0);
    this.bindAt(4, t.texB1);
    this.bindAt(5, t.texHist[this.histCurrent]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    for (let unit = 2; unit <= 5; unit++) this.bindAt(unit, null);

    const readSeg = con.needSegments !== false;
    gl.readBuffer(gl.COLOR_ATTACHMENT1);
    const tRead = performance.now();
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, t.readPacked);
    if (readSeg) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbB);
      gl.readBuffer(gl.COLOR_ATTACHMENT1);
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, t.readIds);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const tDone = performance.now();
    this.lastGpuMs = tRead - t0;
    this.lastReadMs = tDone - tRead;
    if (gl.isContextLost()) return false;
    this.histCurrent = next;
    this.histValid = true;
    this.packedFresh = true;
    con.state.gpuHistory = true;
    con.state.prev = null;
    con.state.frameIndex++;
    const n = w * h;
    const rp = t.readPacked;
    const st = out.structure,
      ti = out.tissue,
      tr = out.transmission,
      sg = out.segment;
    if (readSeg) {
      const ri = t.readIds;
      for (let i = 0, o = 0; i < n; i++, o += 4) {
        display[i] = rp[o]!;
        st[i] = rp[o + 1]!;
        ti[i] = rp[o + 2]!;
        tr[i] = TRANS_DECODE[rp[o + 3]!]!;
        sg[i] = ri[o + 3]!;
      }
    } else {
      for (let i = 0, o = 0; i < n; i++, o += 4) {
        display[i] = rp[o]!;
        st[i] = rp[o + 1]!;
        ti[i] = rp[o + 2]!;
        tr[i] = TRANS_DECODE[rp[o + 3]!]!;
      }
      sg.fill(0);
    }
    this.lastSegRead = readSeg;
    this.lastOutput = 'display';
    this.lastMs = performance.now() - t0;
    return true;
  }

  present(req: PresentRequest): ImageBitmap | null {
    const gl = this.gl;
    const t = this.targets;
    if (this.disposed || !this.packedFresh || !t.texPacked || gl.isContextLost()) return null;
    const t0 = performance.now();
    const { lut, width: W, height: H, color } = req;
    const canvas = gl.canvas;
    if (canvas.width !== W || canvas.height !== H) {
      canvas.width = W;
      canvas.height = H;
    }
    this.uploads.ensureLut(lut, W, H);
    if (color) {
      this.uploads.ensureColor(color, lut.lines, lut.samples);
      this.uploads.ensurePolar(lut, W, H);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.bindVertexArray(this.vao);
    gl.useProgram(this.progPresent);
    const u = this.up;
    gl.uniform1i(u.uHeightPx, H);
    gl.uniform1i(u.uLines, lut.lines);
    gl.uniform1i(u.uSamples, lut.samples);
    gl.uniform1i(u.uColorOn, color ? 1 : 0);
    gl.uniform1i(u.uTintMap, tintMapIndex(req.tintMap));
    if (color) {
      const c = color.settings;
      gl.uniform4f(u.uBox, c.boxRMinCm, c.boxRMaxCm, c.boxThetaMinRad, c.boxThetaMaxRad);
      gl.uniform2f(u.uColorMap, c.scaleMps, c.showVariance ? 1 : 0);
    }
    this.bindAt(2, t.texPacked);
    this.bindAt(3, this.uploads.lutTex);
    this.bindAt(4, this.uploads.colorTex);
    this.bindAt(5, this.uploads.polarTex);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    for (let unit = 2; unit <= 5; unit++) this.bindAt(unit, null);
    if (req.readback) {
      const tmp = new Uint8Array(W * H * 4);
      gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, tmp);
      for (let y = 0; y < H; y++)
        req.readback.set(tmp.subarray((H - 1 - y) * W * 4, (H - y) * W * 4), y * W * 4);
      this.lastPresentMs = performance.now() - t0;
      return null;
    }
    const bitmap =
      typeof OffscreenCanvas !== 'undefined' && canvas instanceof OffscreenCanvas
        ? canvas.transferToImageBitmap()
        : null;
    this.lastPresentMs = performance.now() - t0;
    return bitmap;
  }

  restoreCpuHistory(state: ConsoleState, spec: PolarFrameSpec): void {
    if (!state.gpuHistory) return;
    state.gpuHistory = false;
    const gl = this.gl;
    const t = this.targets;
    const w = spec.samples,
      h = spec.lines;
    if (this.disposed || gl.isContextLost() || !this.histValid || t.w !== w || t.h !== h) return;
    const n = w * h;
    if (this.histUpload.length !== n * 4) this.histUpload = new Float32Array(n * 4);
    const buf = this.histUpload;
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbConsole[this.histCurrent]);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (gl.isContextLost()) return;
    const prev = new Float32Array(n);
    for (let i = 0; i < n; i++) prev[i] = buf[i * 4]!;
    state.prev = prev;
  }

  private uploadHistory(prev: Float32Array, w: number, h: number): void {
    const n = w * h;
    if (this.histUpload.length !== n * 4) this.histUpload = new Float32Array(n * 4);
    const buf = this.histUpload;
    buf.fill(0);
    for (let i = 0; i < n; i++) buf[i * 4] = prev[i] ?? 0;
    const gl = this.gl;
    this.bindAt(UPLOAD_UNIT, this.targets.texHist[this.histCurrent]);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RGBA, gl.FLOAT, buf);
    this.bindAt(UPLOAD_UNIT, null);
    this.histValid = true;
  }
}

/**
 * Create the GPU renderer on an OffscreenCanvas (worker) or a canvas element (main thread).
 * Returns null when WebGL2 or float render targets are unavailable, and also on software WebGL
 * unless `allowSoftware` is set (decision 53).
 */
export function createWebgl2Renderer(
  canvas?: OffscreenCanvas | HTMLCanvasElement,
  options: { allowSoftware?: boolean } = {},
): { renderer: Webgl2Renderer | null; reason: string } {
  try {
    const c =
      canvas ??
      (typeof OffscreenCanvas !== 'undefined'
        ? new OffscreenCanvas(4, 4)
        : typeof document !== 'undefined'
          ? document.createElement('canvas')
          : null);
    if (!c) return { renderer: null, reason: 'no canvas available' };
    const gl = c.getContext('webgl2', {
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: false,
      premultipliedAlpha: false,
    });
    if (!gl) return { renderer: null, reason: 'WebGL2 unavailable' };
    const name = webglRendererName(gl);
    if (!options.allowSoftware && SOFTWARE_GL.test(name)) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      return { renderer: null, reason: `software WebGL (${name.slice(0, 60)}): CPU tracer` };
    }
    return { renderer: new Webgl2Renderer(gl), reason: 'ok' };
  } catch (e) {
    return { renderer: null, reason: e instanceof Error ? e.message : String(e) };
  }
}
