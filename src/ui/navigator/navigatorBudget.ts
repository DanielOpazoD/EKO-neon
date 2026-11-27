/**
 * Pure render-budget and invalidation helpers for the 3D torso navigator (`TorsoView`).
 * Keeps software-WebGL throttling, visibility gating, and selective store invalidation
 * decoupled from Three.js so they can be verified deterministically in Node unit tests.
 */

export const SOFTWARE_WEBGL_RE =
  /swiftshader|llvmpipe|softpipe|mesa offscreen|microsoft basic render|angle.*google.*swiftshader/i;

export function isSoftwareWebglRenderer(rendererName: string): boolean {
  return SOFTWARE_WEBGL_RE.test(rendererName);
}

/** Probe the active WebGL2 renderer string from an already-created context. */
export function isSoftwareWebglContext(gl: WebGLRenderingContext | WebGL2RenderingContext): boolean {
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  const raw = ext
    ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)
    : gl.getParameter(gl.RENDERER);
  return isSoftwareWebglRenderer(String(raw ?? ''));
}

export interface NavigatorFrameBudgetInput {
  hidden: boolean;
  width: number;
  height: number;
  hasThorax: boolean;
  anatomy: boolean;
  dirty: boolean;
  phaseIdx: number;
  shownPhaseIdx: number;
  phasesReady: number;
  shownPhasesReady: number;
  totalPhases: number;
  nowMs: number;
  lastRenderAtMs: number;
  lastCostMs: number;
  softwareGl: boolean;
  /** True after a genuine canvas resize so the newly cleared WebGL framebuffer repaints immediately. */
  forceImmediate?: boolean;
}

export interface NavigatorFrameDecision {
  shouldRender: boolean;
  minIntervalMs: number;
  nextDirty: boolean;
}

/** Minimum ms between consecutive 3D navigator frames given recent render cost and backend. */
export function computeNavigatorMinIntervalMs(lastCostMs: number, softwareGl: boolean): number {
  if (softwareGl) return Math.max(350, Math.min(1200, lastCostMs * 5));
  return lastCostMs > 25 ? Math.min(500, lastCostMs * 4) : 0;
}

/**
 * Decide whether the 3D navigator should render on the current rAF tick.
 * - Never renders while the tab is hidden or the container is collapsed (`<= 0` px).
 * - Before `thorax` arrives, renders only when `dirty` (eliminates 60 Hz empty renders).
 * - Only marks phase updates dirty when heart anatomy is visible and all mesh phases are loaded.
 * - Never performs idle keepalive renders when nothing changed.
 */
export function shouldRenderNavigatorFrame(
  inp: NavigatorFrameBudgetInput,
): NavigatorFrameDecision {
  const minIntervalMs = computeNavigatorMinIntervalMs(inp.lastCostMs, inp.softwareGl);
  if (inp.hidden || inp.width <= 0 || inp.height <= 0) {
    return { shouldRender: false, minIntervalMs, nextDirty: inp.dirty };
  }
  let dirty = inp.dirty;
  if (inp.hasThorax) {
    const phasesChanged = inp.phasesReady !== inp.shownPhasesReady;
    const canBeat = inp.anatomy && inp.phasesReady >= inp.totalPhases;
    if (phasesChanged || (canBeat && inp.phaseIdx !== inp.shownPhaseIdx)) {
      dirty = true;
    }
  }
  if (!dirty) {
    return { shouldRender: false, minIntervalMs, nextDirty: false };
  }
  const elapsed = inp.nowMs - inp.lastRenderAtMs;
  if (!inp.forceImmediate && inp.lastRenderAtMs > 0 && elapsed < minIntervalMs) {
    return { shouldRender: false, minIntervalMs, nextDirty: true };
  }
  return { shouldRender: true, minIntervalMs, nextDirty: false };
}

/** Minimal slice of `useSimStore` state that affects the 3D torso navigator. */
export interface NavigatorStoreSlice {
  probe: unknown;
  mode: unknown;
  settings: { depthCm: number; sectorDeg: number };
  reviewMarkers: unknown;
  reviewSelectedId: unknown;
  ui: {
    showSkeleton: boolean;
    navSkin: boolean;
    navHeart: boolean;
    navChambers: boolean;
    navValves: boolean;
    navVessels: boolean;
    navSegments: boolean;
    navWindows: boolean;
    navAxes: boolean;
    navCut: boolean;
    navSplit: boolean;
    navLabels: boolean;
    segmentModel: unknown;
    selectedSegment: unknown;
    reviewMode: boolean;
  };
}

/** Returns true only when a navigator-visible store slice changed (ignores `fpsUi`, HUD counters, etc.). */
export function navigatorStoreChanged(
  prev: NavigatorStoreSlice,
  next: NavigatorStoreSlice,
): boolean {
  if (
    prev.probe !== next.probe ||
    prev.mode !== next.mode ||
    prev.settings.depthCm !== next.settings.depthCm ||
    prev.settings.sectorDeg !== next.settings.sectorDeg ||
    prev.reviewMarkers !== next.reviewMarkers ||
    prev.reviewSelectedId !== next.reviewSelectedId
  ) {
    return true;
  }
  const a = prev.ui;
  const b = next.ui;
  if (a === b) return false;
  return (
    a.showSkeleton !== b.showSkeleton ||
    a.navSkin !== b.navSkin ||
    a.navHeart !== b.navHeart ||
    a.navChambers !== b.navChambers ||
    a.navValves !== b.navValves ||
    a.navVessels !== b.navVessels ||
    a.navSegments !== b.navSegments ||
    a.navWindows !== b.navWindows ||
    a.navAxes !== b.navAxes ||
    a.navCut !== b.navCut ||
    a.navSplit !== b.navSplit ||
    a.navLabels !== b.navLabels ||
    a.segmentModel !== b.segmentModel ||
    a.selectedSegment !== b.selectedSegment ||
    a.reviewMode !== b.reviewMode
  );
}
