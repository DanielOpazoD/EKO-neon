import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { useSimStore } from '@/app/store';
import { MAX_TEACHING_ARROWS } from '@/app/store/uiSlice';
import { DEFAULT_ACQUISITION } from '@/simulator/renderer/types';
import { polarToPixel, type SectorMapping } from '@/simulator/renderer/scanConvert';
import { activeBmodePresetId, BMODE_IMAGE_PRESETS } from '@/ui/console/imagePresets';
import {
  COLOR_BOX_MIN_R_CM,
  COLOR_BOX_MIN_SPAN_CM,
  COLOR_BOX_MIN_SPAN_RAD,
  colorBoxCornerPixels,
  colorBoxHandleCursor,
  hitTestColorBoxHandle,
  resizeColorBoxFromHandle,
} from './colorBoxHandles';

const MOCK_SECTOR: SectorMapping = {
  width: 640,
  height: 480,
  apexX: 320,
  apexY: 20,
  pxPerCm: 26,
  depthCm: 16,
  sectorRad: (80 * Math.PI) / 180,
  invertLR: false,
};

describe('PR C: 2D texture presets, Color box corner handles, and frozen teaching pointer', () => {
  it('matches DEFAULT_ACQUISITION to standard preset and round-trips all 4 clinical B-mode presets', () => {
    assert.equal(activeBmodePresetId(DEFAULT_ACQUISITION), 'standard');
    assert.equal(BMODE_IMAGE_PRESETS.length, 4);

    for (const preset of BMODE_IMAGE_PRESETS) {
      const merged = { ...DEFAULT_ACQUISITION, ...preset.patch };
      assert.equal(activeBmodePresetId(merged), preset.id);
    }

    assert.equal(
      activeBmodePresetId({ ...DEFAULT_ACQUISITION, dynamicRangeDb: 62 }),
      null,
      'Custom settings should not match any preset',
    );
  });

  it('hit-tests all 4 Color ROI corners and resizes cleanly with minimum span and sector clamping', () => {
    const box = { t0: -0.35, t1: 0.35, r0: 3.0, r1: 10.5 };
    const corners = colorBoxCornerPixels(MOCK_SECTOR, box);
    assert.equal(corners.length, 4);

    for (const c of corners) {
      assert.equal(hitTestColorBoxHandle({ x: c.x + 2, y: c.y - 2 }, MOCK_SECTOR, box), c.id);
    }
    assert.equal(hitTestColorBoxHandle({ x: 10, y: 10 }, MOCK_SECTOR, box), null);

    // Cursor orientation flips cleanly with invertLR
    assert.equal(colorBoxHandleCursor('tl', false), 'nwse-resize');
    assert.equal(colorBoxHandleCursor('tr', false), 'nesw-resize');
    assert.equal(colorBoxHandleCursor('tl', true), 'nesw-resize');
    assert.equal(colorBoxHandleCursor('br', true), 'nesw-resize');

    // Shrinking top-left past opposite edges enforces min span
    const shrunk = resizeColorBoxFromHandle('tl', box, { thetaRad: 0.9, rCm: 14 }, MOCK_SECTOR);
    assert.ok(shrunk.boxThetaMaxRad - shrunk.boxThetaMinRad >= COLOR_BOX_MIN_SPAN_RAD - 1e-6);
    assert.ok(shrunk.boxRMaxCm - shrunk.boxRMinCm >= COLOR_BOX_MIN_SPAN_CM - 1e-6);

    // Expanding bottom-right outside sector clamps neatly inside sector boundaries
    const clamped = resizeColorBoxFromHandle('br', box, { thetaRad: 2.0, rCm: 25 }, MOCK_SECTOR);
    assert.ok(clamped.boxThetaMaxRad <= MOCK_SECTOR.sectorRad / 2 + 1e-6);
    assert.ok(clamped.boxRMaxCm <= MOCK_SECTOR.depthCm + 1e-6);

    // Top depth clamps above minimum depth floor
    const topClamped = resizeColorBoxFromHandle('tr', box, { thetaRad: 0.2, rCm: 0.1 }, MOCK_SECTOR);
    assert.ok(topClamped.boxRMinCm >= COLOR_BOX_MIN_R_CM - 1e-6);

    // Corner pixels match polarToPixel directly
    const pBr = polarToPixel(MOCK_SECTOR, box.r1, box.t1);
    const brCorner = corners.find((c) => c.id === 'br')!;
    assert.ok(Math.hypot(brCorner.x - pBr.x, brCorner.y - pBr.y) < 1e-6);
  });

  it('enforces ephemeral lifecycle and exam privacy for frozen-frame teaching pointer arrows', () => {
    const st = useSimStore.getState();
    st.setMode('sandbox');
    if (useSimStore.getState().frozen) useSimStore.getState().toggleFreeze();
    useSimStore.getState().clearTeachingArrows();

    // Cannot arm pointer or add arrows while live
    useSimStore.getState().setTeachingPointer(true);
    assert.equal(useSimStore.getState().teachingPointerOn, false);
    useSimStore.getState().addTeachingArrow({ x0: 10, y0: 10, x1: 80, y1: 60, onStrip: false });
    assert.equal(useSimStore.getState().teachingArrows.length, 0);

    // Freeze -> arm caliper -> enabling teaching pointer disarms caliper
    useSimStore.getState().toggleFreeze();
    useSimStore.getState().setActiveTool('caliper');
    assert.equal(useSimStore.getState().activeTool, 'caliper');
    useSimStore.getState().setTeachingPointer(true);
    assert.equal(useSimStore.getState().teachingPointerOn, true);
    assert.equal(useSimStore.getState().activeTool, 'none');

    // Adding arrows caps at MAX_TEACHING_ARROWS
    for (let i = 0; i < MAX_TEACHING_ARROWS + 2; i++) {
      useSimStore.getState().addTeachingArrow({
        x0: 20 + i * 10,
        y0: 30,
        x1: 90 + i * 10,
        y1: 80,
        onStrip: false,
      });
    }
    assert.equal(useSimStore.getState().teachingArrows.length, MAX_TEACHING_ARROWS);

    // Arming a measurement tool disarms pointer mode while keeping drawn arrows visible on the frozen frame
    useSimStore.getState().setActiveTool('caliper');
    assert.equal(useSimStore.getState().teachingPointerOn, false);
    assert.equal(useSimStore.getState().teachingArrows.length, MAX_TEACHING_ARROWS);

    // Unfreezing automatically clears arrows and pointer mode
    useSimStore.getState().setTeachingPointer(true);
    useSimStore.getState().toggleFreeze();
    assert.equal(useSimStore.getState().frozen, false);
    assert.equal(useSimStore.getState().teachingPointerOn, false);
    assert.equal(useSimStore.getState().teachingArrows.length, 0);

    // Exam mode refuses teaching pointer even when frozen
    useSimStore.getState().setMode('exam', 'normal-excellent-window');
    useSimStore.getState().toggleFreeze();
    useSimStore.getState().setTeachingPointer(true);
    assert.equal(useSimStore.getState().teachingPointerOn, false);
    useSimStore.getState().addTeachingArrow({ x0: 10, y0: 10, x1: 80, y1: 60, onStrip: false });
    assert.equal(useSimStore.getState().teachingArrows.length, 0);

    // Restore clean sandbox state
    useSimStore.getState().setMode('sandbox');
  });
});
