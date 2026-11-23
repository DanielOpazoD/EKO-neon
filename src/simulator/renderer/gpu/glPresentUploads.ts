import { COLOR_THRESHOLD } from '@/simulator/doppler/color/colorDoppler';
import { colorPowerConfidence } from '@/simulator/doppler/color/spatiotemporalFilter';
import { consoleCompensation } from '../postprocess/consolePipeline';
import { packScanLutTexels, type ScanLut } from '../scanConvert';
import type { AcquisitionSettings, PolarFrameSpec } from '../types';
import { makeTexture2D, UPLOAD_UNIT } from './glHelpers';
import type { PresentColorCpu } from './webgl2Renderer';

/**
 * Manages cached WebGL2 texture uploads for console depth compensation, scan-conversion LUT,
 * polar coordinates (for exact colour-box boundaries), and complex-phase Color Doppler data.
 */
export class GlPresentUploads {
  readonly compTex: WebGLTexture;
  readonly lutTex: WebGLTexture;
  readonly colorTex: WebGLTexture;
  readonly polarTex: WebGLTexture;

  private compKey = '';
  private compData = new Float32Array(0);
  private lutKeyUploaded = '';
  private lutTexels = new Uint16Array(0);
  private colorKey = '';
  private colorTexels = new Float32Array(0);
  private polarKeyUploaded = '';
  private polarTexels = new Float32Array(0);

  constructor(private readonly gl: WebGL2RenderingContext) {
    this.compTex = makeTexture2D(gl);
    this.lutTex = makeTexture2D(gl);
    this.colorTex = makeTexture2D(gl);
    this.polarTex = makeTexture2D(gl);
  }

  private bindUpload(tex: WebGLTexture | null): void {
    this.gl.activeTexture(this.gl.TEXTURE0 + UPLOAD_UNIT);
    this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
  }

  ensureCompensation(s: AcquisitionSettings, spec: PolarFrameSpec): void {
    const key = `${spec.samples}|${spec.depthCm}|${spec.focusCm}|${s.frequencyMHz}|${s.gainDb}|${s.depthCompensationDbPerCmMHz ?? ''}|${s.tgcDb.join(',')}`;
    if (key === this.compKey) return;
    if (this.compData.length !== spec.samples) this.compData = new Float32Array(spec.samples);
    consoleCompensation(s, spec, this.compData);
    const gl = this.gl;
    this.bindUpload(this.compTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, spec.samples, 1, 0, gl.RED, gl.FLOAT, this.compData);
    this.bindUpload(null);
    this.compKey = key;
  }

  ensureLut(lut: ScanLut, W: number, H: number): void {
    if (lut.key === this.lutKeyUploaded) return;
    this.lutTexels = packScanLutTexels(lut, this.lutTexels);
    const gl = this.gl;
    this.bindUpload(this.lutTex);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA16UI,
      W,
      H,
      0,
      gl.RGBA_INTEGER,
      gl.UNSIGNED_SHORT,
      this.lutTexels,
    );
    this.bindUpload(null);
    this.lutKeyUploaded = lut.key;
  }

  /**
   * The colour box is tested against the LUT's own polar coordinates, not recomputed per pixel:
   * GLSL `atan` is approximate on some GPUs.
   */
  ensurePolar(lut: ScanLut, W: number, H: number): void {
    if (lut.key === this.polarKeyUploaded) return;
    const n = W * H;
    if (this.polarTexels.length !== n * 2) this.polarTexels = new Float32Array(n * 2);
    const t = this.polarTexels;
    for (let p = 0; p < n; p++) {
      t[p * 2] = lut.rCm[p] ?? 0;
      t[p * 2 + 1] = lut.theta[p] ?? 0;
    }
    const gl = this.gl;
    this.bindUpload(this.polarTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, W, H, 0, gl.RG, gl.FLOAT, t);
    this.bindUpload(null);
    this.polarKeyUploaded = lut.key;
  }

  ensureColor(c: PresentColorCpu, lines: number, samples: number): void {
    const scale = Math.max(1e-3, c.settings.scaleMps);
    const baseline = c.settings.baselineShiftMps ?? 0;
    const key = `${c.version}|${lines}x${samples}|${scale.toFixed(4)}|${baseline.toFixed(4)}`;
    if (key === this.colorKey) return;
    const n = lines * samples;
    if (this.colorTexels.length !== n * 4) this.colorTexels = new Float32Array(n * 4);
    const t = this.colorTexels;
    const kPhi = Math.PI / scale;
    const powArr = c.power;
    for (let i = 0, o = 0; i < n; i++, o += 4) {
      const v = c.vel[i] ?? NaN;
      if (!Number.isNaN(v)) {
        const ph = (v - baseline) * kPhi;
        t[o] = Math.cos(ph);
        t[o + 1] = Math.sin(ph);
        t[o + 2] = c.variance[i] ?? 0;
        if (powArr) {
          t[o + 3] = 0.5 + 0.5 * colorPowerConfidence(powArr[i] ?? 1, COLOR_THRESHOLD);
        } else {
          t[o + 3] = 1;
        }
      } else {
        t[o] = 0;
        t[o + 1] = 0;
        t[o + 2] = 0;
        t[o + 3] = 0;
      }
    }
    const gl = this.gl;
    this.bindUpload(this.colorTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, samples, lines, 0, gl.RGBA, gl.FLOAT, t);
    this.bindUpload(null);
    this.colorKey = key;
  }

  dispose(): void {
    const gl = this.gl;
    for (const t of [this.compTex, this.lutTex, this.colorTex, this.polarTex]) gl.deleteTexture(t);
    this.compKey = this.lutKeyUploaded = this.colorKey = this.polarKeyUploaded = '';
  }
}
