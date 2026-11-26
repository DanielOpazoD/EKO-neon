import assert from 'node:assert/strict';
import test from 'node:test';
import { throttleLatest, type ThrottleClock } from './throttleLatest';

/** A clock that only moves when told to, firing the timers that fall due. */
function manualClock(): ThrottleClock & { advance(ms: number): void } {
  let now = 1000;
  let timers: { at: number; fn: () => void; id: number }[] = [];
  let nextId = 1;
  return {
    now: () => now,
    setTimeout(fn, ms) {
      const id = nextId++;
      timers.push({ at: now + ms, fn, id });
      return id;
    },
    clearTimeout(id) {
      timers = timers.filter((t) => t.id !== id);
    },
    advance(ms) {
      const until = now + ms;
      for (;;) {
        const due = timers.filter((t) => t.at <= until).sort((a, b) => a.at - b.at)[0];
        if (!due) break;
        timers = timers.filter((t) => t !== due);
        now = due.at;
        due.fn();
      }
      now = until;
    },
  };
}

test('throttleLatest publishes at once, then the newest value when the interval ends', () => {
  const clock = manualClock();
  const shown: string[] = [];
  const t = throttleLatest<string>(120, (v) => shown.push(v), clock);
  t.push('live');
  assert.deepEqual(shown, ['live']);
  // frozen 30 ms later: the worker sends this frame once and nothing after it
  clock.advance(30);
  t.push('frozen');
  assert.deepEqual(shown, ['live'], 'inside the interval it waits');
  clock.advance(89);
  assert.deepEqual(shown, ['live']);
  clock.advance(1);
  assert.deepEqual(shown, ['live', 'frozen'], 'the last frame is published when the interval ends');
  clock.advance(1000);
  assert.deepEqual(shown, ['live', 'frozen'], 'and only once');
});

test('throttleLatest keeps the newest of a burst, at most once per interval', () => {
  const clock = manualClock();
  const shown: number[] = [];
  const t = throttleLatest<number>(120, (v) => shown.push(v), clock);
  // a cine scroll: a frame every 15 ms for 300 ms
  for (let i = 0; i <= 20; i++) {
    t.push(i);
    clock.advance(15);
  }
  clock.advance(500);
  assert.equal(shown[0], 0);
  assert.equal(shown[shown.length - 1], 20, 'the frame the scroll stopped on is the one the HUD keeps');
  assert.ok(shown.length <= 4, `${shown.length} publications in 315 ms`);
});

test('throttleLatest.cancel forgets the value waiting for its turn', () => {
  const clock = manualClock();
  const shown: string[] = [];
  const t = throttleLatest<string>(120, (v) => shown.push(v), clock);
  t.push('a');
  t.push('b');
  t.cancel();
  clock.advance(500);
  assert.deepEqual(shown, ['a']);
  t.push('c');
  assert.deepEqual(shown, ['a', 'c'], 'a push after the interval publishes at once');
});
