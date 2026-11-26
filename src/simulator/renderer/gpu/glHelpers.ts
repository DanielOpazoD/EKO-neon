import { sceneLattice } from '@/core/noise';
import { TISSUE_PROPS } from '@/simulator/anatomy/tissue';
import {
  buildNoiseKernels,
  LATERAL_TAPS,
  MAX_LATERAL_RADIUS,
  psfKey,
  type PsfKernels,
} from '../acoustic/psf';
import type { AcquisitionSettings, PolarFrameSpec } from '../types';

/** Texture unit used only to upload data, never sampled. */
export const UPLOAD_UNIT = 9;

export const LOG_UNIFORMS = ['uEnv', 'uComp', 'uNoise', 'uDynRange'] as const;
export const CONSOLE_UNIFORMS = [
  'uLog',
  'uEnv',
  'uIds',
  'uHist',
  'uSamples',
  'uLines',
  'uSri',
  'uEdge',
  'uPersist',
  'uGrayMap',
] as const;
export const NOISE_AXIAL_UNIFORMS = ['uKernels', 'uSamples', 'uSeed', 'uFrameIndex'] as const;
export const NOISE_LATERAL_UNIFORMS = ['uNoiseAx', 'uKernels', 'uLines'] as const;
export const PRESENT_UNIFORMS = [
  'uPacked',
  'uLut',
  'uColor',
  'uPolar',
  'uHeightPx',
  'uLines',
  'uSamples',
  'uColorOn',
  'uTintMap',
  'uBox',
  'uColorMap',
  'uColorPalette',
] as const;

export type Locations<T extends readonly string[]> = Record<T[number], WebGLUniformLocation | null>;

export function uniformLocations<T extends readonly string[]>(
  gl: WebGL2RenderingContext,
  prog: WebGLProgram,
  names: T,
): Locations<T> {
  const o: Record<string, WebGLUniformLocation | null> = {};
  for (const n of names) o[n] = gl.getUniformLocation(prog, n);
  return o as Locations<T>;
}

let noiseKernelMemo: PsfKernels | null = null;
/** The noise response of a frame geometry, rebuilt only when geometry or probe settings change. */
export function buildNoiseKernelsCached(
  spec: PolarFrameSpec,
  settings: AcquisitionSettings,
): PsfKernels {
  const key = `noise|${psfKey(spec, settings.frequencyMHz, settings.harmonics, 0, null)}`;
  if (noiseKernelMemo?.key !== key)
    noiseKernelMemo = buildNoiseKernels(spec, settings.frequencyMHz, settings.harmonics);
  return noiseKernelMemo;
}

export function makeTexture2D(gl: WebGL2RenderingContext): WebGLTexture {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

export function buildProgram(
  gl: WebGL2RenderingContext,
  vs: string,
  fs: string,
): WebGLProgram {
  const compile = (type: number, src: string): WebGLShader => {
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh) ?? '';
      gl.deleteShader(sh);
      throw new Error(`shader compile error: ${log}`);
    }
    return sh;
  };
  const v = compile(gl.VERTEX_SHADER, vs);
  const f = compile(gl.FRAGMENT_SHADER, fs);
  const prog = gl.createProgram();
  gl.attachShader(prog, v);
  gl.attachShader(prog, f);
  gl.linkProgram(prog);
  gl.deleteShader(v);
  gl.deleteShader(f);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog) ?? '';
    gl.deleteProgram(prog);
    throw new Error(`program link error: ${log}`);
  }
  return prog;
}

/** Pass A's constant uniforms: the tissue table, the parameter and noise units. */
export function initPassAUniforms(gl: WebGL2RenderingContext, prog: WebGLProgram): void {
  gl.useProgram(prog);
  const tp = new Float32Array(20 * 4);
  for (let t = 0; t < 20; t++) {
    const p = TISSUE_PROPS[t];
    if (!p) continue;
    tp[t * 4] = p.reflect;
    tp[t * 4 + 1] = p.specular;
    tp[t * 4 + 2] = p.attenuation;
    tp[t * 4 + 3] = p.grain;
  }
  gl.uniform4fv(gl.getUniformLocation(prog, 'uTissue'), tp);
  gl.uniform1i(gl.getUniformLocation(prog, 'uParams'), 0);
  gl.uniform1i(gl.getUniformLocation(prog, 'uNoise'), 1);
}

/** Upload a PSF/noise kernel table: row 0 axial taps, row 1 + sample lateral taps. */
export function uploadKernelTable(
  gl: WebGL2RenderingContext,
  tex: WebGLTexture,
  k: PsfKernels,
  spec: PolarFrameSpec,
): void {
  const rows = spec.samples + 1;
  const data = new Float32Array(LATERAL_TAPS * rows * 4);
  const c = MAX_LATERAL_RADIUS;
  for (let j = -k.axialRadius; j <= k.axialRadius; j++)
    data[(c + j) * 4] = k.axial[j + k.axialRadius]!;
  data[c * 4 + 1] = k.axialRadius;
  for (let si = 0; si < spec.samples; si++) {
    const row = (si + 1) * LATERAL_TAPS;
    for (let t = 0; t < LATERAL_TAPS; t++)
      data[(row + t) * 4] = k.lateral[si * LATERAL_TAPS + t]!;
    data[(row + c) * 4 + 1] = k.lateralRadius[si]!;
  }
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, LATERAL_TAPS, rows, 0, gl.RGBA, gl.FLOAT, data);
}

/**
 * The RGBA8 texels of the 128³ noise lattice texture for a seed, as uploadNoiseLattice3D uploads them: texel (x, y, z)
 * at ((z·128 + y)·128 + x)·4, channel k holding the byte there of the scene's lattice k (core/noise.ts sceneLattice:
 * the CPU's speckle lattices A, B and C, then the heart's wall lattice), which the shaders read as lat(p, k).
 * cpuGpuParity.test.ts reads the shaders' texelFetch from this array.
 */
export function noiseLatticeTexels(seed: number): Uint8Array {
  const [a, b, c, w] = [sceneLattice(seed, 0), sceneLattice(seed, 1), sceneLattice(seed, 2), sceneLattice(seed, 3)];
  const n = 128 * 128 * 128;
  const data = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    data[i * 4] = a[i]!;
    data[i * 4 + 1] = b[i]!;
    data[i * 4 + 2] = c[i]!;
    data[i * 4 + 3] = w[i]!;
  }
  return data;
}

/** Upload the 4-channel 128^3 3D speckle/scatter noise lattice texture for a given seed. */
export function uploadNoiseLattice3D(
  gl: WebGL2RenderingContext,
  tex: WebGLTexture,
  seed: number,
): void {
  const data = noiseLatticeTexels(seed);
  gl.bindTexture(gl.TEXTURE_3D, tex);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.REPEAT);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA8, 128, 128, 128, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
}

/** Renderer string of a WebGL context, unmasked when the browser exposes it. */
export function webglRendererName(gl: WebGL2RenderingContext): string {
  const dbg = gl.getExtension('WEBGL_debug_renderer_info') as {
    UNMASKED_RENDERER_WEBGL: number;
  } | null;
  const name: unknown = dbg
    ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)
    : gl.getParameter(gl.RENDERER);
  return typeof name === 'string' ? name : '';
}

/** Software rasterisers behind WebGL (Chromium without GPU, virtual machines, remote desktops, Playwright). */
export const SOFTWARE_GL = /swiftshader|llvmpipe|softpipe|software|basic render/i;
