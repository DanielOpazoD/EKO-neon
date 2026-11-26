import { createWebgl2Renderer, type Webgl2Renderer } from './webgl2Renderer';

/**
 * A canvas with a simulated WebGL2 context, for tests of the GPU port's lifecycle in Node: nothing is shaded (draws and
 * reads do nothing), but the context lives and dies as a browser's does. Losing it sets isContextLost() at once and
 * dispatches 'webglcontextlost' (synchronously here, in a later task in a browser); it comes back only if a listener
 * called preventDefault() on the loss; while lost, getExtension returns null and shaders do not compile; and its objects
 * die with it: isTexture is false for an object of a lost context, even once the context is restored. Outside any
 * *.test.ts file so that importing it does not run another file's tests (as `core/baseInput.ts`).
 */
export class SimulatedCanvas {
  width = 4;
  height = 4;
  /** No WebGL2 on this canvas (getContext → null), as on a GPU the browser blocked. */
  refuse = false;
  /** GL calls a test checks: what a dispose deletes, whether it released the context, and draws read back. */
  readonly calls = { createTexture: 0, deleteProgram: 0, deleteTexture: 0, loseContext: 0, readPixels: 0 };
  readonly gl: WebGL2RenderingContext;
  private lost = false;
  private prevented = false;
  /** Bumped by every loss: an object made before it belongs to no context any more. */
  private generation = 0;
  private readonly listeners = new Map<string, Set<(e: Event) => void>>();

  constructor(options: { floatTargets?: boolean } = {}) {
    this.gl = simulatedContext(this, options.floatTargets ?? true);
  }

  get contextLost(): boolean {
    return this.lost;
  }

  getContext(type: string): WebGL2RenderingContext | null {
    return type === 'webgl2' && !this.refuse ? this.gl : null;
  }

  addEventListener(type: string, fn: (e: Event) => void): void {
    let set = this.listeners.get(type);
    if (!set) this.listeners.set(type, (set = new Set()));
    set.add(fn);
  }

  removeEventListener(type: string, fn: (e: Event) => void): void {
    this.listeners.get(type)?.delete(fn);
  }

  /** Listeners on the canvas, of one event type or of all. */
  listenerCount(type?: string): number {
    if (type) return this.listeners.get(type)?.size ?? 0;
    let n = 0;
    for (const set of this.listeners.values()) n += set.size;
    return n;
  }

  /** The context is lost and its loss dispatched; whether a listener prevented the default (asked for a restore). */
  lose(): boolean {
    this.loseSilently();
    return this.dispatchLost();
  }

  /** The context is lost, its event not dispatched yet: what the core can see first in a browser. */
  loseSilently(): void {
    if (this.lost) return;
    this.lost = true;
    this.prevented = false;
    this.generation++;
  }

  /** Dispatch 'webglcontextlost'; whether a listener prevented the default. */
  dispatchLost(): boolean {
    let prevented = false;
    const e = {
      preventDefault: () => {
        prevented = true;
      },
    } as unknown as Event;
    for (const fn of [...(this.listeners.get('webglcontextlost') ?? [])]) fn(e);
    this.prevented ||= prevented;
    return prevented;
  }

  /** The browser restores the context, which it does only after a prevented loss; whether it did. */
  restore(): boolean {
    if (!this.lost || !this.prevented) return false;
    this.lost = false;
    this.prevented = false;
    for (const fn of [...(this.listeners.get('webglcontextrestored') ?? [])]) fn({} as Event);
    return true;
  }

  /** A new GL object of this context as it is now. */
  makeObject(): object {
    return { canvas: this, generation: this.generation };
  }

  /** Whether `obj` is a live object of this context: made by it, since its last loss, and the context not lost. */
  owns(obj: unknown): boolean {
    const o = obj as { canvas?: unknown; generation?: number } | null;
    return !this.lost && o?.canvas === this && o.generation === this.generation;
  }
}

function simulatedContext(canvas: SimulatedCanvas, floatTargets: boolean): WebGL2RenderingContext {
  const object = () => canvas.makeObject();
  const target: Record<string, unknown> = {
    canvas,
    COMPILE_STATUS: 0x8b81,
    LINK_STATUS: 0x8b82,
    VERTEX_SHADER: 0x8b31,
    FRAGMENT_SHADER: 0x8b30,
    FRAMEBUFFER_COMPLETE: 0x8cd5,
    TEXTURE0: 0x84c0,
    COLOR_ATTACHMENT0: 0x8ce0,
    RENDERER: 0x1f01,
    isContextLost: () => canvas.contextLost,
    getExtension: (name: string) => {
      if (canvas.contextLost) return null;
      if (name === 'EXT_color_buffer_float') return floatTargets ? {} : null;
      if (name === 'WEBGL_lose_context')
        return {
          loseContext: () => {
            canvas.calls.loseContext++;
            canvas.lose();
          },
          restoreContext: () => canvas.restore(),
        };
      return null;
    },
    getParameter: () => 'Simulated GPU',
    createShader: object,
    createProgram: object,
    createTexture: () => {
      canvas.calls.createTexture++;
      return object();
    },
    createFramebuffer: object,
    createVertexArray: object,
    isTexture: (t: unknown) => canvas.owns(t),
    deleteProgram: () => {
      canvas.calls.deleteProgram++;
    },
    deleteTexture: () => {
      canvas.calls.deleteTexture++;
    },
    getShaderParameter: () => !canvas.contextLost,
    getProgramParameter: () => !canvas.contextLost,
    checkFramebufferStatus: () => 0x8cd5,
    getUniformLocation: () => ({}),
    readPixels: () => {
      canvas.calls.readPixels++;
    },
  };
  return new Proxy(target, {
    get(obj, prop) {
      if (prop in obj) return obj[prop as string];
      return () => {};
    },
  }) as unknown as WebGL2RenderingContext;
}

/**
 * Builds the GPU port as `createWebgl2Renderer` does in the app (the core's `createGpu`): on the canvas given, one whose
 * context was restored, or on a new simulated canvas, which has no WebGL2 while `refuse` is set.
 */
export function simulatedGpuFactory() {
  const factory = {
    /** Every canvas a renderer was asked of without one given, in order. */
    canvases: [] as SimulatedCanvas[],
    /** One entry per call: on a restored canvas or on a new one. */
    calls: [] as Array<'restored' | 'new'>,
    refuse: false,
    create(canvas?: OffscreenCanvas | HTMLCanvasElement): {
      renderer: Webgl2Renderer | null;
      reason: string;
    } {
      factory.calls.push(canvas ? 'restored' : 'new');
      if (canvas) return createWebgl2Renderer(canvas);
      const c = new SimulatedCanvas();
      c.refuse = factory.refuse;
      factory.canvases.push(c);
      return createWebgl2Renderer(c as unknown as OffscreenCanvas);
    },
  };
  return factory;
}
