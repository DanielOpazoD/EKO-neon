import { useSyncExternalStore } from 'react';

interface PendingHandlers {
  undo: () => void;
  finish: () => void;
}

let pendingCount = 0;
let handlers: PendingHandlers | null = null;
const listeners = new Set<() => void>();

function emit(): void {
  for (const fn of listeners) fn();
}

/**
 * Tiny imperative bridge so the active capture bar and MeasureTab reflect the pending vertex count
 * and can trigger "Deshacer punto" / "Finalizar trazado" without storing transient mouse clicks in
 * the main simulation store or triggering re-renders during live frames.
 */
export const pendingSession = {
  getCount: (): number => pendingCount,
  setCount(next: number): void {
    if (next === pendingCount) return;
    pendingCount = next;
    emit();
  },
  register(next: PendingHandlers | null): void {
    handlers = next;
  },
  undo(): boolean {
    if (!handlers || pendingCount <= 0) return false;
    handlers.undo();
    return true;
  },
  finish(): boolean {
    if (!handlers || pendingCount <= 0) return false;
    handlers.finish();
    return true;
  },
  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  },
};

export function usePendingPointCount(): number {
  return useSyncExternalStore(pendingSession.subscribe, pendingSession.getCount, () => 0);
}
