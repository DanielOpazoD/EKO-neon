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
 * Why the CPU tracer forms the frames after the port's context was lost: what each of those frames keeps of it
 * (`FrameFormation.gpu`), whatever the port does later. The port's own state goes apart (`GpuRecovery.status`): kept with
 * the frame, «trazador CPU hasta que vuelva la GPU (intento 1 de 5)» was still the tip over a frozen frame once the GPU
 * was back, or five attempts later.
 */
export const GPU_LOST = 'Contexto WebGL perdido';

/**
 * Brings the GPU port back after its WebGL context was lost: a suspended laptop, a restarted GPU process, a driver reset.
 * Everything the port holds on the GPU is derived (programs from GLSL strings, textures from keys the new renderer starts
 * without), so a new renderer is the whole recovery; the core gives it a new atlas and a colour history to start.
 *
 * The lost renderer is kept until it is replaced, so its listener still calls preventDefault() on a webglcontextlost
 * dispatched after the core saw `isContextLost()` (the event comes in a later task), and hears the browser restore the
 * context. The first restoration is used at once and on the same canvas, without a new context; otherwise, and after
 * any failed attempt, the attempts follow GPU_RETRY_DELAYS_MS, on the restored canvas if there is one by then and on a
 * new canvas if not. A lost context nobody builds on again (replaced on a new canvas, given up, disposed with the core)
 * is given back, should the browser still restore it (`Webgl2Renderer.dispose`). Time is the wall clock: the browser
 * restores in wall time, and the simulated clock crawls on the CPU tracer (steps of at most 0.1 s however long a frame
 * takes).
 *
 * The core follows the context on every step, live or frozen (a frozen tick every 80 ms), so the attempts count from
 * the loss. Paused (`SimInput.paused`: nobody looks at the scanner, another screen or a hidden tab) the worker does not
 * step at all: a loss is seen on waking, and its attempts count from then; a restoration that came meanwhile is built on
 * at that first step, which costs no CPU frame.
 */
export class GpuRecovery {
  /** The renderer whose context was lost, kept for its listeners; disposed when replaced or given up. */
  private lostRenderer: Webgl2Renderer | null = null;
  private state: 'idle' | 'waiting' | 'failed' = 'idle';
  /** Attempts since the port was last stable. */
  private attempts = 0;
  private nextAt = 0;
  private recoveredAt = -Infinity;

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
    const g = this.create(restored ? this.lostRenderer!.canvas : undefined);
    if (g.renderer) {
      // the lost renderer lets go once the new one is built (no event can come in between): of a restored context as it
      // is, the new renderer's now, and of a lost one by giving it back
      this.releaseLost(restored);
      this.state = 'idle';
      this.recoveredAt = t;
      return g.renderer;
    }
    // a restored context the port could not be built on (a shader its driver fails) is given back as well: it was left
    // alive with no owner, and the next attempts are on new canvases
    if (restored) this.releaseLost();
    if (this.attempts >= GPU_RETRY_DELAYS_MS.length) this.giveUp();
    else this.nextAt = t + GPU_RETRY_DELAYS_MS[this.attempts]!;
    return null;
  }

  /**
   * The port's state in the HUD's words (null when no loss is pending): the attempt next, or given up and what to do. It
   * names no tracer and no quality, which belong to the frame shown (`FrameFormation`). One text for the badge over a
   * frozen frame the GPU formed and for the end of the tip over one the CPU tracer formed, live or frozen, so it keeps to
   * one line of the badge (42 characters: 320 px of 12 px mono). The driver's message of the last failed attempt
   * («WebGL2 unavailable», a software GL's name of up to 60 characters) is left out: what the learner can do about any of
   * them is the same: reload the page (the same case does not rebuild the core, and the exam locks the case selector).
   */
  get status(): string | null {
    const n = GPU_RETRY_DELAYS_MS.length;
    if (this.state === 'waiting') return `GPU perdida (intento ${this.attempts + 1} de ${n})`;
    if (this.state === 'failed') return `Sin GPU tras ${n} intentos; recarga la página`;
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

  /** Lets go of the lost renderer: with `keepContext`, of the restored context a new renderer was built on, untouched. */
  private releaseLost(keepContext = false): void {
    this.lostRenderer?.dispose(keepContext);
    this.lostRenderer = null;
  }
}
