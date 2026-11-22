import assert from 'node:assert/strict';
import test from 'node:test';
import { FrameCompositor } from './frameCompositor';

test('FrameCompositor reuses matching recycled buffers and allocates fresh ones on size mismatch', () => {
  const compositor = new FrameCompositor();
  assert.equal(compositor.poolSize, 0);

  const bufA = compositor.takeBuffer(1024);
  assert.equal(bufA.byteLength, 1024);

  compositor.recycle(bufA);
  assert.equal(compositor.poolSize, 1);

  // Matching size returns the exact pooled buffer instance
  const reused = compositor.takeBuffer(1024);
  assert.equal(reused, bufA);
  assert.equal(compositor.poolSize, 0);

  // Non-matching size leaves pooled buffer intact and allocates a fresh one
  compositor.recycle(bufA);
  const bigger = compositor.takeBuffer(2048);
  assert.equal(bigger.byteLength, 2048);
  assert.notEqual(bigger, bufA);
  assert.equal(compositor.poolSize, 1);
});

test('FrameCompositor ignores zero-byte buffers, caps pool at 3, and clears cleanly', () => {
  const compositor = new FrameCompositor();

  compositor.recycle(new ArrayBuffer(0));
  assert.equal(compositor.poolSize, 0);

  for (let i = 0; i < 6; i++) {
    compositor.recycle(new ArrayBuffer(512));
  }
  assert.equal(compositor.poolSize, 3);

  compositor.clear();
  assert.equal(compositor.poolSize, 0);
});
