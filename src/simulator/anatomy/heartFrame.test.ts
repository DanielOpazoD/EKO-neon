import assert from 'node:assert/strict';
import test from 'node:test';
import { listCases, loadCaseById } from '@/cases';
import { cross, dot, length, sub, v3 } from '@/core/vec3';
import { buildHeartFrame, heartDirToTorso, heartToTorso, torsoToHeart } from './heartFrame';

test('buildHeartFrame constructs an orthonormal right-handed basis for every clinical case', () => {
  const cases = listCases();
  assert.ok(cases.length >= 12);

  for (const meta of cases) {
    const c = loadCaseById(meta.id);
    const frame = buildHeartFrame(c.anatomy, v3(0.4, -0.3, 0.2));

    assert.ok(Math.abs(length(frame.ex) - 1) < 1e-6, `${meta.id}: |ex| != 1`);
    assert.ok(Math.abs(length(frame.ey) - 1) < 1e-6, `${meta.id}: |ey| != 1`);
    assert.ok(Math.abs(length(frame.ez) - 1) < 1e-6, `${meta.id}: |ez| != 1`);

    assert.ok(Math.abs(dot(frame.ex, frame.ey)) < 1e-6, `${meta.id}: ex·ey != 0`);
    assert.ok(Math.abs(dot(frame.ey, frame.ez)) < 1e-6, `${meta.id}: ey·ez != 0`);
    assert.ok(Math.abs(dot(frame.ez, frame.ex)) < 1e-6, `${meta.id}: ez·ex != 0`);

    // right-handed: ex × ey = ez
    const handed = dot(cross(frame.ex, frame.ey), frame.ez);
    assert.ok(Math.abs(handed - 1) < 1e-6, `${meta.id}: expected ex×ey=ez, got ${handed}`);
  }
});

test('torsoToHeart and heartToTorso are exact inverses and heartDirToTorso preserves lengths', () => {
  const c = loadCaseById('normal-excellent');
  const frame = buildHeartFrame(c.anatomy, v3(0.5, -0.2, -0.4));

  const torsoPoints = [v3(0, 0, 0), v3(3.2, -1.7, 5.8), v3(-2.4, 4.1, -6.5), frame.origin];
  for (const p of torsoPoints) {
    const roundTrip = heartToTorso(frame, torsoToHeart(frame, p));
    assert.ok(length(sub(roundTrip, p)) < 1e-6, `round-trip drift for ${JSON.stringify(p)}`);
  }

  const dirs = [v3(1, 0, 0), v3(0, 1, 0), v3(0, 0, 1), v3(1.5, -2.2, 0.7)];
  for (const d of dirs) {
    const td = heartDirToTorso(frame, d);
    assert.ok(Math.abs(length(td) - length(d)) < 1e-6, 'direction length must be invariant under rotation');
  }
});
