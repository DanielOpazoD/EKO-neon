import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  computeNavigatorMinIntervalMs,
  isSoftwareWebglRenderer,
  navigatorStoreChanged,
  shouldRenderNavigatorFrame,
  type NavigatorFrameBudgetInput,
  type NavigatorStoreSlice,
} from './navigatorBudget';

function baseBudget(overrides: Partial<NavigatorFrameBudgetInput> = {}): NavigatorFrameBudgetInput {
  return {
    hidden: false,
    width: 360,
    height: 480,
    hasThorax: true,
    anatomy: true,
    dirty: false,
    phaseIdx: 0,
    shownPhaseIdx: 0,
    phasesReady: 10,
    shownPhasesReady: 10,
    totalPhases: 10,
    nowMs: 1000,
    lastRenderAtMs: 500,
    lastCostMs: 5,
    softwareGl: false,
    ...overrides,
  };
}

function baseSlice(): NavigatorStoreSlice {
  return {
    probe: { x: 0 },
    mode: 'free',
    settings: { depthCm: 16, sectorDeg: 80 },
    reviewMarkers: [],
    reviewSelectedId: null,
    ui: {
      showSkeleton: true,
      navSkin: true,
      navHeart: true,
      navChambers: true,
      navValves: true,
      navVessels: true,
      navSegments: false,
      navWindows: false,
      navAxes: false,
      navCut: true,
      navSplit: false,
      navLabels: false,
      segmentModel: 'aha17',
      selectedSegment: null,
      reviewMode: false,
    },
  };
}

describe('Torso 3D navigator render budget & invalidation', () => {
  it('detects software WebGL renderers and enforces a protective frame interval floor', () => {
    assert.equal(isSoftwareWebglRenderer('ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device))'), true);
    assert.equal(isSoftwareWebglRenderer('llvmpipe (LLVM 15.0.6, 256 bits)'), true);
    assert.equal(isSoftwareWebglRenderer('ANGLE (Apple, Apple M3 Pro, OpenGL 4.1)'), false);

    assert.equal(computeNavigatorMinIntervalMs(4, false), 0);
    assert.equal(computeNavigatorMinIntervalMs(40, false), 160);
    assert.equal(computeNavigatorMinIntervalMs(4, true), 350);
    assert.equal(computeNavigatorMinIntervalMs(120, true), 600);
  });

  it('skips idle frames, hidden tabs, collapsed containers, and non-beating partial phases', () => {
    // Clean state with no changes must NOT render (no idle keepalive)
    assert.equal(shouldRenderNavigatorFrame(baseBudget()).shouldRender, false);

    // Pre-thorax renders once when dirty, never continuously when clean
    assert.equal(
      shouldRenderNavigatorFrame(baseBudget({ hasThorax: false, dirty: true })).shouldRender,
      true,
    );
    assert.equal(
      shouldRenderNavigatorFrame(baseBudget({ hasThorax: false, dirty: false })).shouldRender,
      false,
    );

    // Hidden tab or collapsed panel preserves dirty flag without drawing
    const hiddenRes = shouldRenderNavigatorFrame(baseBudget({ hidden: true, dirty: true }));
    assert.equal(hiddenRes.shouldRender, false);
    assert.equal(hiddenRes.nextDirty, true);

    const collapsedRes = shouldRenderNavigatorFrame(baseBudget({ width: 0, dirty: true }));
    assert.equal(collapsedRes.shouldRender, false);
    assert.equal(collapsedRes.nextDirty, true);

    // Partial phases (< totalPhases) or hidden anatomy must not trigger beating redraws
    assert.equal(
      shouldRenderNavigatorFrame(
        baseBudget({ phasesReady: 4, shownPhasesReady: 4, phaseIdx: 3, shownPhaseIdx: 2 }),
      ).shouldRender,
      false,
    );
    assert.equal(
      shouldRenderNavigatorFrame(
        baseBudget({ anatomy: false, phaseIdx: 3, shownPhaseIdx: 2 }),
      ).shouldRender,
      false,
    );

    // Once all phases are ready and anatomy is visible, a new phase renders on hardware GL,
    // while software GL defers until minIntervalMs elapses
    assert.equal(
      shouldRenderNavigatorFrame(
        baseBudget({ phaseIdx: 3, shownPhaseIdx: 2, nowMs: 600, lastRenderAtMs: 500 }),
      ).shouldRender,
      true,
    );
    const swDeferred = shouldRenderNavigatorFrame(
      baseBudget({
        phaseIdx: 3,
        shownPhaseIdx: 2,
        nowMs: 600,
        lastRenderAtMs: 500,
        softwareGl: true,
      }),
    );
    assert.equal(swDeferred.shouldRender, false);
    assert.equal(swDeferred.nextDirty, true);
  });

  it('ignores unrelated store updates (such as fpsUi) and flags navigator-relevant changes', () => {
    const a = baseSlice();
    const b = { ...a, settings: { ...a.settings } };
    assert.equal(navigatorStoreChanged(a, b), false);

    assert.equal(
      navigatorStoreChanged(a, { ...a, settings: { depthCm: 18, sectorDeg: 80 } }),
      true,
    );
    assert.equal(
      navigatorStoreChanged(a, { ...a, ui: { ...a.ui, navCut: false } }),
      true,
    );
  });
});
