import assert from 'node:assert/strict';
import test from 'node:test';
import { FrameCompositor } from './frameCompositor';

test('FrameCompositor reuses matching recycled buffers and evicts the old size on a resize', () => {
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

  // A new size allocates a fresh buffer and drops the pooled ones of the old size: nothing will ask for them again
  compositor.recycle(bufA);
  const bigger = compositor.takeBuffer(2048);
  assert.equal(bigger.byteLength, 2048);
  assert.notEqual(bigger, bufA);
  assert.equal(compositor.poolSize, 0);
  // and a buffer of the old size that comes back from the main thread afterwards is not kept either
  compositor.recycle(bufA);
  assert.equal(compositor.poolSize, 0);
  compositor.recycle(bigger);
  assert.equal(compositor.takeBuffer(2048), bigger);
});

test('FrameCompositor ignores zero-byte buffers, caps pool at 3, and clears cleanly', () => {
  const compositor = new FrameCompositor();

  compositor.recycle(new ArrayBuffer(0));
  assert.equal(compositor.poolSize, 0);

  compositor.takeBuffer(512);
  for (let i = 0; i < 6; i++) {
    compositor.recycle(new ArrayBuffer(512));
  }
  assert.equal(compositor.poolSize, 3);

  compositor.clear();
  assert.equal(compositor.poolSize, 0);
});

test('FrameCompositor keeps reusing buffers after many display sizes (the pool is never left full of stale ones)', () => {
  // the worker loop: a buffer per frame, two frames in flight, each returned when the main thread has drawn it. Three
  // buffers of earlier sizes filled the pool for good and from the fourth size on every frame allocated a new one.
  const compositor = new FrameCompositor();
  const sizes = [640 * 520, 800 * 650, 1024 * 820, 960 * 780, 900 * 730, 1000 * 812].map((p) => p * 4);
  const seen = new Set<ArrayBuffer>();
  const inFlight: ArrayBuffer[] = [];
  for (const size of sizes) {
    let allocated = 0;
    for (let frame = 0; frame < 12; frame++) {
      while (inFlight.length >= 2) compositor.recycle(inFlight.shift()!);
      const b = compositor.takeBuffer(size);
      assert.equal(b.byteLength, size);
      if (!seen.has(b)) {
        seen.add(b);
        allocated++;
      }
      inFlight.push(b);
    }
    assert.ok(allocated <= 2, `${size} B: ${allocated} buffers allocated in 12 frames (2 in flight)`);
    assert.ok(compositor.poolSize <= 3);
  }
});
