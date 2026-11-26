import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { frameSizeForBox, MAX_FRAME, MIN_FRAME } from './frameSize';

/** Horizontal over vertical scale of the frame stretched over the box: 1 is an undistorted image. */
const anisotropy = (box: { w: number; h: number }, f: { width: number; height: number }) =>
  box.w / f.width / (box.h / f.height);

describe('frame size of the display', () => {
  it('forms the box itself when it fits', () => {
    assert.deepEqual(frameSizeForBox(1012, 341), { width: 1012, height: 341 });
    assert.deepEqual(frameSizeForBox(640.4, 519.6), { width: 640, height: 520 });
  });

  it('scales a large box down alike in both axes', () => {
    // 1920 px screens: the box measured in the browser with the default and the collapsed rail
    for (const box of [
      { w: 1272, h: 840 },
      { w: 1272, h: 970 },
      { w: 1528, h: 840 },
      { w: 2200, h: 1300 },
    ]) {
      const f = frameSizeForBox(box.w, box.h);
      assert.ok(f.width <= MAX_FRAME.width && f.height <= MAX_FRAME.height, `${box.w}×${box.h}`);
      assert.ok(
        Math.abs(anisotropy(box, f) - 1) < 0.004,
        `${box.w}×${box.h} → ${f.width}×${f.height} keeps the proportions`,
      );
    }
    assert.deepEqual(frameSizeForBox(1272, 840), { width: 1024, height: 676 });
  });

  it('scales a small box up alike in both axes, and falls back on a box with no size', () => {
    const f = frameSizeForBox(300, 200);
    assert.deepEqual(f, { width: 360, height: 240 });
    assert.ok(f.width >= MIN_FRAME.width && f.height >= MIN_FRAME.height);
    assert.deepEqual(frameSizeForBox(0, 0), { ...MIN_FRAME });
    assert.deepEqual(frameSizeForBox(Number.NaN, 400), { ...MIN_FRAME });
  });
});
