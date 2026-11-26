import type { Webgl2Renderer } from '@/simulator/renderer/gpu/webgl2Renderer';

/**
 * Builds the GPU port: on `canvas`, a canvas whose lost context the browser has restored, or on a new canvas without it.
 * `createWebgl2Renderer` in the app; tests pass one over a simulated GL.
 */
export type GpuFactory = (canvas?: OffscreenCanvas | HTMLCanvasElement) => {
  renderer: Webgl2Renderer | null;
  reason: string;
};

/**
 * The wait (ms) before each attempt to bring the GPU port back: the first counts from the loss, each next one from the
 * attempt that failed before it. Five attempts at 1, 3, 7, 15 and 31 s after the loss, and then the CPU tracer stays: a
 * GPU that fails for good (WebGL blocked after repeated GPU crashes, a driver that loses every context) costs five
 * context creations, not one per frame.
 */
export const GPU_RETRY_DELAYS_MS: readonly number[] = [1000, 2000, 4000, 8000, 16000];

/**
 * A rebuilt port that keeps its context this long has recovered, and a later loss has all the attempts again. One lost
 * sooner did not: its attempt keeps counting, so a context lost again on every restoration cannot rebuild forever.
 */
export const GPU_STABLE_MS = 10_000;

/**
 * Brings the GPU port back after its WebGL context was lost: a suspended laptop, a restarted GPU process, a driver reset.
 * Everything the port holds on the GPU is derived (programs from GLSL strings, textures from keys the new renderer starts
 * without), so a new renderer is the whole recovery; the core gives it a new atlas and a colour history to start.
 *
 * The lost renderer is kept until it is replaced, so its listener still calls preventDefault() on a webglcontextlost
 * dispatched after the core saw `isContextLost()` (the event comes in a later task), and hears the browser restore the
 * context. The first restoration is used at once and on the same canvas, without a new context; otherwise, and after
 * any failed attempt, the attempts follow GPU_RETRY_DELAYS_MS, on the restored canvas if there is one by then and on a
 * new canvas if not. Time is the wall clock: the browser restores in wall time, and the simulated clock crawls on the CPU
 * tracer (steps of at most 0.1 s however long a frame takes).
 */
export class GpuRecovery {
  /** The renderer whose context was lost, kept for its listeners; disposed when replaced or given up. */
  private lostRenderer: Webgl2Renderer | null = null;
  private state: 'idle' | 'waiting' | 'failed' = 'idle';
  /** Attempts since the port was last stable. */
  private attempts = 0;
  private nextAt = 0;
  private recoveredAt = -Infinity;
  private lastFailure = '';

  constructor(
    private readonly create: GpuFactory,
    private readonly now: () => number,
  ) {}

  /** The port's context was lost: keep `renderer` for the browser's restoration and schedule the next attempt. */
  lost(renderer: Webgl2Renderer): void {
    const t = this.now();
    if (t - this.recoveredAt >= GPU_STABLE_MS) this.attempts = 0;
    this.releaseLost();
    this.lostRenderer = renderer;
    if (this.attempts >= GPU_RETRY_DELAYS_MS.length) {
      this.giveUp();
      return;
    }
    this.state = 'waiting';
    this.nextAt = t + GPU_RETRY_DELAYS_MS[this.attempts]!;
  }

  /** A new GPU port when an attempt is due and succeeds; null otherwise. Cheap when nothing is pending. */
  poll(): Webgl2Renderer | null {
    if (this.state !== 'waiting') return null;
    const t = this.now();
    const restored = this.lostRenderer?.contextRestored === true;
    if (t < this.nextAt && !(restored && this.attempts === 0)) return null;
    this.attempts++;
    let canvas: OffscreenCanvas | HTMLCanvasElement | undefined;
    if (restored) {
      canvas = this.lostRenderer!.canvas;
      // its listeners off before the new renderer puts its own on the same canvas
      this.releaseLost();
    }
    const g = this.create(canvas);
    if (g.renderer) {
      this.releaseLost();
      this.state = 'idle';
      this.recoveredAt = t;
      return g.renderer;
    }
    this.lastFailure = g.reason;
    if (this.attempts >= GPU_RETRY_DELAYS_MS.length) this.giveUp();
    else this.nextAt = t + GPU_RETRY_DELAYS_MS[this.attempts]!;
    return null;
  }

  /** What the HUD says of the GPU while the port is lost (null when no loss is pending): waiting, or given up. */
  get reason(): string | null {
    const n = GPU_RETRY_DELAYS_MS.length;
    const lost = 'Contexto WebGL perdido';
    if (this.state === 'waiting')
      return `${lost}: trazador CPU hasta que vuelva la GPU (intento ${this.attempts + 1} de ${n})`;
    if (this.state === 'failed') {
      const why = this.lastFailure || 'contexto perdido de nuevo';
      return `${lost}: la GPU no volvió tras ${n} intentos (${why}); recarga el caso`;
    }
    return null;
  }

  dispose(): void {
    this.releaseLost();
    this.state = 'idle';
  }

  private giveUp(): void {
    this.releaseLost();
    this.state = 'failed';
  }

  private releaseLost(): void {
    this.lostRenderer?.dispose();
    this.lostRenderer = null;
  }
}
