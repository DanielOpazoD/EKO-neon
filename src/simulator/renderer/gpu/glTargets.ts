import { makeTexture2D } from './glHelpers';

/**
 * Owns all polar-grid size-dependent WebGL2 render target textures, framebuffers, and
 * CPU readback buffers for Webgl2Renderer. Reallocates cleanly on resolution changes.
 */
export class GlRenderTargets {
  texA0: WebGLTexture | null = null;
  texA1: WebGLTexture | null = null;
  texA2: WebGLTexture | null = null;
  texA3: WebGLTexture | null = null;
  texS0: WebGLTexture | null = null;
  texS1: WebGLTexture | null = null;
  texB0: WebGLTexture | null = null;
  texB1: WebGLTexture | null = null;
  texB2: WebGLTexture | null = null;
  texL0: WebGLTexture | null = null;
  texP0: WebGLTexture | null = null;
  texC0: WebGLTexture | null = null;
  texC1: WebGLTexture | null = null;
  texD0: WebGLTexture | null = null;
  texNA: WebGLTexture | null = null;
  texNL: WebGLTexture | null = null;
  texLog: WebGLTexture | null = null;
  texHist: [WebGLTexture | null, WebGLTexture | null] = [null, null];
  texPacked: WebGLTexture | null = null;

  fbA: WebGLFramebuffer | null = null;
  fbS0: WebGLFramebuffer | null = null;
  fbS1: WebGLFramebuffer | null = null;
  fbB: WebGLFramebuffer | null = null;
  fbL: WebGLFramebuffer | null = null;
  fbP: WebGLFramebuffer | null = null;
  fbC: WebGLFramebuffer | null = null;
  fbD: WebGLFramebuffer | null = null;
  fbNA: WebGLFramebuffer | null = null;
  fbNL: WebGLFramebuffer | null = null;
  fbLog: WebGLFramebuffer | null = null;
  fbConsole: [WebGLFramebuffer | null, WebGLFramebuffer | null] = [null, null];

  w = 0;
  h = 0;
  readAmp = new Float32Array(0);
  readIds = new Uint8Array(0);
  readPacked = new Uint8Array(0);

  constructor(private readonly gl: WebGL2RenderingContext) {}

  /** Allocate or resize all polar FBOs and readback buffers. Returns true if reallocated. */
  ensure(w: number, h: number): boolean {
    if (this.w === w && this.h === h && this.fbA) return false;
    this.dispose();
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
      if (status !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`framebuffer incomplete: ${status}`);
      return fb;
    };
    const f32 = (): WebGLTexture => mk(gl.RGBA32F, gl.RGBA, gl.FLOAT);
    const f16 = (): WebGLTexture => mk(gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT);

    // Half-float (RGBA16F) cuts memory bandwidth in half on bounded-range targets; full float32
    // stays on the complex signal/envelope/march targets to preserve dynamic range down to 1e-6.
    this.texA0 = f16();
    this.texA1 = mk(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);
    this.texA2 = f16();
    this.texA3 = f16();
    this.texS0 = f16();
    this.texS1 = f16();
    this.texB0 = f32();
    this.texB1 = mk(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);
    this.texB2 = f32();
    this.texL0 = f32();
    this.texP0 = f16();
    this.texC0 = f32();
    this.texC1 = f32();
    this.texD0 = f32();
    this.texNA = f32();
    this.texNL = f32();
    this.texLog = mk(gl.R16F, gl.RED, gl.HALF_FLOAT);

    this.fbA = mkFb(this.texA0, this.texA1, this.texA2, this.texA3);
    this.fbS0 = mkFb(this.texS0);
    this.fbS1 = mkFb(this.texS1);
    this.fbB = mkFb(this.texB0, this.texB1, this.texB2);
    this.fbL = mkFb(this.texL0);
    this.fbP = mkFb(this.texP0);
    this.fbC = mkFb(this.texC0, this.texC1);
    this.fbD = mkFb(this.texD0);
    this.fbNA = mkFb(this.texNA);
    this.fbNL = mkFb(this.texNL);
    this.fbLog = mkFb(this.texLog);

    const hist0 = f16();
    const hist1 = f16();
    const packed = mk(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);
    this.texHist = [hist0, hist1];
    this.texPacked = packed;
    this.fbConsole = [mkFb(hist0, packed), mkFb(hist1, packed)];

    this.readPacked = new Uint8Array(w * h * 4);
    this.readAmp = new Float32Array(w * h * 4);
    this.readIds = new Uint8Array(w * h * 4);
    this.w = w;
    this.h = h;
    return true;
  }

  dispose(): void {
    const gl = this.gl;
    for (const t of [
      this.texA0,
      this.texA1,
      this.texA2,
      this.texA3,
      this.texS0,
      this.texS1,
      this.texB0,
      this.texL0,
      this.texP0,
      this.texB1,
      this.texB2,
      this.texC0,
      this.texC1,
      this.texD0,
      this.texNA,
      this.texNL,
      this.texLog,
      ...this.texHist,
      this.texPacked,
    ])
      if (t) gl.deleteTexture(t);
    for (const f of [
      this.fbA,
      this.fbS0,
      this.fbS1,
      this.fbL,
      this.fbP,
      this.fbB,
      this.fbC,
      this.fbD,
      this.fbNA,
      this.fbNL,
      this.fbLog,
      ...this.fbConsole,
    ])
      if (f) gl.deleteFramebuffer(f);
    this.texLog = null;
    this.fbLog = null;
    this.texHist = [null, null];
    this.fbConsole = [null, null];
    this.texPacked = null;
    this.texA0 = this.texA1 = this.texA2 = this.texA3 = this.texS0 = this.texS1 = null;
    this.texL0 =
      this.texP0 =
      this.texB0 =
      this.texB1 =
      this.texB2 =
      this.texC0 =
      this.texC1 =
      this.texD0 =
      this.texNA =
      this.texNL =
        null;
    this.fbA =
      this.fbS0 =
      this.fbS1 =
      this.fbL =
      this.fbP =
      this.fbB =
      this.fbC =
      this.fbD =
      this.fbNA =
      this.fbNL =
        null;
    this.w = 0;
    this.h = 0;
  }
}
