/** The timers a throttle runs on (injectable for tests). */
export interface ThrottleClock {
  now(): number;
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(id: unknown): void;
}

const systemClock: ThrottleClock = {
  now: () => performance.now(),
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (id) => clearTimeout(id as ReturnType<typeof setTimeout>),
};

export interface LatestThrottle<T> {
  /** Offer the newest value: published now if `intervalMs` has passed since the last one, else when it has. */
  push(value: T): void;
  /** Forget a value waiting for its turn. */
  cancel(): void;
}

/**
 * Publishes the newest of a stream of values at most once per `intervalMs`, and never leaves the last one unpublished: a
 * value that arrives inside the interval goes out when it ends, or the newer one that replaced it meanwhile. The HUD
 * needs this since the worker sends a frozen frame only when it changes: kept only if 120 ms had passed, the frame
 * that followed the freeze, or the last one of a cine scroll, never reached the HUD, which went on describing the live
 * frame before it (not frozen, its phase, its view).
 */
export function throttleLatest<T>(
  intervalMs: number,
  publish: (value: T) => void,
  clock: ThrottleClock = systemClock,
): LatestThrottle<T> {
  let lastMs = -Infinity;
  let pending: { value: T } | null = null;
  let timer: unknown = null;
  const flush = (): void => {
    timer = null;
    if (!pending) return;
    const { value } = pending;
    pending = null;
    lastMs = clock.now();
    publish(value);
  };
  return {
    push(value) {
      pending = { value };
      if (timer !== null) return;
      const waitMs = intervalMs - (clock.now() - lastMs);
      if (waitMs <= 0) flush();
      else timer = clock.setTimeout(flush, waitMs);
    },
    cancel() {
      if (timer !== null) clock.clearTimeout(timer);
      timer = null;
      pending = null;
    },
  };
}
