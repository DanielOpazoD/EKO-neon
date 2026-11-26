import assert from 'node:assert/strict';
import test from 'node:test';
import { CASE_INPUTS, loadCaseById } from '@/cases';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import { buildCaseModels, REST_PATIENT } from './caseModels';
import { computeChamberPose, lvInflowCavitySdf } from './heartPose';
import {
  mitralFreeEdge,
  mitralHingeZ,
  mitralLeafletPoint,
  SAM_CONTACT_SEVERITY,
  SAM_DRAG_RISE,
  SAM_MAX_BLEND,
  SAM_RELEASE,
  SAM_RELEASE_START,
  samDrag,
} from './mitralValve';
import { fitOpenRotation, openProfile, openWallGap } from './openLeafletFit';
import { skirtTip } from './valveSkirt';

/**
 * The fastest leaflet motion of a normal heart is the early-diastolic opening of the anterior mitral leaflet, 0.40 ±
 * 0.06 m/s at its peak (instantaneous M-mode velocity in 20 normal subjects; Upton, Gibson & Brown, Br Heart J
 * 1976;38:51–58). Mean + 3 SD bounds every leaflet motion that is not the valve's own opening or closing.
 */
const LEAFLET_MAX_MPS = 0.6;
/**
 * The cycle's opening law (CycleState.mvOpen, tvOpen) carries a free edge between its closed and fully open positions at
 * its own pace, fastest at the instants of opening and closing, so a step may add that excursion times the change of
 * opening. No free edge travels farther between the two than 4 cm: 3.9 cm the elongated anterior leaflet of the HOCM
 * case, 3.6 cm the anterior tricuspid leaflet of the pulmonary hypertension case. The SAM is not an opening (mvOpen is 0
 * through systole): the anterior leaflet keeps to the speed bound alone while the outflow drags it.
 */
const OPENING_EXCURSION_CM = 4;
/** Phases per cycle: the pose cache's bins (simulatorCore), at their centres. */
const N = 256;
const MV_Q = Array.from({ length: 19 }, (_, i) => -0.9 + i * 0.1);
const TV_OFF = [-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9];

interface Frame {
  mvOpen: number;
  tvOpen: number;
  sam: number;
  ejection: number;
  /** Free edges [x, y, z, …]: anterior and posterior mitral fibres at MV_Q, tricuspid zones at TV_OFF of their span. */
  tips: [number[], number[], number[]];
  /** Largest signed distance to the ventricular cavity of the drawn open mitral leaflets (cm, > 0 outside), and where. */
  wall: number;
  wallAt: string;
}
interface Sweep {
  id: string;
  stepS: number;
  mitralOpenPhase: number;
  frames: Frame[];
}

let sweeps: Sweep[] | null = null;
function allSweeps(): Sweep[] {
  if (sweeps) return sweeps;
  sweeps = CASE_INPUTS.map((input) => {
    const { heart, tables } = buildCaseModels(loadCaseById(input.id), REST_PATIENT);
    const tm = tables.timings;
    const p = [0, 0, 0];
    const frames: Frame[] = [];
    for (let i = 0; i < N; i++) {
      const state = cycleStateAt(tables, (i + 0.5) / N);
      const pose = computeChamberPose(heart, state);
      const mv = pose.valves.mitral;
      const tips: Frame['tips'] = [[], [], []];
      for (const leaflet of [0, 1] as const)
        for (const q of MV_Q) {
          mitralFreeEdge(mv, leaflet, q, p);
          tips[leaflet].push(p[0]!, p[1]!, p[2]!);
        }
      for (const zone of pose.valves.tv.zones)
        for (const off of TV_OFF) {
          skirtTip(pose.valves.tv, zone, off * zone.halfSpan, p);
          tips[2].push(p[0]!, p[1]!, p[2]!);
        }
      // the open leaflets in diastole, beyond their first tenth and apical of the annulus (basal to it is the atrium)
      let wall = -Infinity,
        wallAt = '';
      if (mv.open > 0 && mv.samBlend === 0)
        for (const leaflet of [0, 1] as const)
          for (const q of MV_Q)
            for (let s = 0.1; s <= 1.0001; s += 0.05) {
              mitralLeafletPoint(mv, leaflet, q, s, p);
              if (p[2]! < mitralHingeZ(p[0]!, p[1]!, mv)) continue;
              const d = lvInflowCavitySdf(heart, pose, mv, p[0]!, p[1]!, p[2]!);
              if (d > wall) {
                wall = d;
                wallAt = `phase ${((i + 0.5) / N).toFixed(3)} leaflet ${leaflet} q ${q.toFixed(1)} at ${s.toFixed(2)} of its length`;
              }
            }
      frames.push({
        mvOpen: state.mvOpen,
        tvOpen: state.tvOpen,
        sam: mv.samBlend,
        // from the beat's own valve events, not from the state
        ejection: (state.timeInBeatS - tm.ejectionStartS) / (tm.ejectionEndS - tm.ejectionStartS),
        tips,
        wall,
        wallAt,
      });
    }
    return {
      id: input.id,
      stepS: tables.rrS / N,
      mitralOpenPhase: tables.timings.mitralOpenS / tables.rrS,
      frames,
    };
  });
  return sweeps;
}

test('leaflet free edges move no faster than a leaflet can, beyond the opening law (12 cases, 256 phases)', () => {
  const groups = ['anterior mitral', 'posterior mitral', 'tricuspid'] as const;
  const failures: string[] = [];
  let worst = 0,
    worstAt = '';
  for (const sw of allSweeps())
    for (let i = 0; i < N; i++) {
      const a = sw.frames[i]!,
        b = sw.frames[(i + 1) % N]!;
      for (let g = 0; g < 3; g++) {
        const dOpen = g === 2 ? Math.abs(b.tvOpen - a.tvOpen) : Math.abs(b.mvOpen - a.mvOpen);
        const bound = LEAFLET_MAX_MPS * 100 * sw.stepS + OPENING_EXCURSION_CM * dOpen;
        const A = a.tips[g]!,
          B = b.tips[g]!;
        for (let j = 0; j < A.length; j += 3) {
          const step = Math.hypot(A[j]! - B[j]!, A[j + 1]! - B[j + 1]!, A[j + 2]! - B[j + 2]!);
          const where = `${sw.id} ${groups[g]} tip ${j / 3}, phase ${((i + 0.5) / N).toFixed(3)}: ${(step * 10).toFixed(1)} mm in ${(sw.stepS * 1000).toFixed(2)} ms (opening ${dOpen.toFixed(3)}), bound ${(bound * 10).toFixed(1)} mm`;
          if (step / bound > worst) {
            worst = step / bound;
            worstAt = where;
          }
          if (step > bound) failures.push(where);
        }
      }
    }
  assert.equal(
    failures.length,
    0,
    `${failures.length} jumps; worst ${worst.toFixed(2)}× the bound: ${worstAt}\n${failures.slice(0, 8).join('\n')}`,
  );
});

test('the open mitral leaflets stay inside the ventricle (12 cases, 256 phases)', () => {
  // the hinge sits on the annulus and the closed commissural fibres along the inflow outline, ≤ 0.4 mm outside it
  const TOLERANCE_CM = 0.05;
  for (const sw of allSweeps()) {
    let worst = -Infinity,
      worstAt = '';
    for (const f of sw.frames)
      if (f.wall > worst) {
        worst = f.wall;
        worstAt = f.wallAt;
      }
    assert.ok(
      worst <= TOLERANCE_CM,
      `${sw.id}: an open leaflet lies ${(worst * 10).toFixed(1)} mm into the wall (${worstAt})`,
    );
  }
});

test('SAM: the outflow drags the anterior leaflet progressively from the ejection onset and lets it go before the mitral valve opens', () => {
  const sw = allSweeps().find((s) => s.id === 'hocm-sam')!;
  const c = loadCaseById('hocm-sam');
  const peak = Math.min(SAM_MAX_BLEND, c.anatomy.mitral.samSeverity / SAM_CONTACT_SEVERITY);
  for (let i = 0; i < N; i++) {
    const f = sw.frames[i]!,
      next = sw.frames[(i + 1) % N]!;
    const u = f.ejection;
    const at = `phase ${((i + 0.5) / N).toFixed(3)}, ${u.toFixed(2)} ejection times from aortic opening`;
    if (u <= 0) assert.equal(f.sam, 0, `SAM before the ejection (${at})`);
    // no step at the onset: a twentieth into the ejection the leaflet has gone a twentieth of its way, not a quarter
    if (u > 0 && u < 0.05) assert.ok(f.sam < 0.05 * peak, `SAM ${f.sam.toFixed(2)} at ${at}`);
    if (u >= SAM_DRAG_RISE && u <= SAM_RELEASE_START)
      assert.ok(Math.abs(f.sam - peak) < 1e-9, `the contact does not hold through ${at}`);
    if (u > 0 && u < SAM_DRAG_RISE && next.ejection < SAM_DRAG_RISE)
      assert.ok(next.sam >= f.sam, `the drag recedes while the ejection builds (${at})`);
    if ((i + 0.5) / N >= sw.mitralOpenPhase && (i + 0.5) / N < 0.99)
      assert.equal(f.sam, 0, `SAM still on as the mitral valve opens (${at})`);
  }
});

test('samDrag is smooth: zero slope where it starts and ends, a plateau between', () => {
  const h = 1e-4;
  assert.equal(samDrag(0), 0);
  assert.equal(samDrag(-0.3), 0);
  assert.equal(samDrag(Number.NaN), 0);
  assert.ok(samDrag(h) / h < 1e-2, 'the drag starts with a slope');
  assert.equal(samDrag(SAM_DRAG_RISE), 1);
  assert.equal(samDrag((SAM_DRAG_RISE + SAM_RELEASE_START) / 2), 1);
  assert.ok((1 - samDrag(SAM_RELEASE_START + h)) / h < 1e-2, 'the release starts with a slope');
  assert.equal(samDrag(SAM_RELEASE_START + SAM_RELEASE), 0);
  // the steepest slope of a smoothstep is 1.5 over its width
  for (let u = 0; u < 1.5; u += 1e-3)
    assert.ok(Math.abs(samDrag(u + 1e-3) - samDrag(u)) / 1e-3 <= 1.5 / Math.min(SAM_DRAG_RISE, SAM_RELEASE) + 1e-6);
});

/** A wall slab: `outer` cm outward of the hinge (where the leaflet opens), `inner` cm inward (the opposite wall). */
const slab =
  (outer: number, inner: number) =>
  (a: number): number =>
    Math.max(-a - outer, a - inner);
const PROFILE = openProfile([-0.61, -0.7, -0.79], 0.8);
/** Worst wall penetration plus gap of PROFILE turned by `rot` against `wall` (≤ 0 when clear). */
function violation(rot: number, wall: (a: number, z: number) => number): number {
  let worst = -Infinity,
    pa = 0,
    pz = 0;
  for (let i = 0; i < 3; i++) {
    const na = PROFILE[i * 2]!,
      nz = PROFILE[i * 2 + 1]!;
    for (const [a0, z0] of [
      [(pa + na) / 2, (pz + nz) / 2],
      [na, nz],
    ] as const) {
      const need = openWallGap(Math.hypot(a0, z0));
      if (need <= 0) continue;
      const a = a0 * Math.cos(rot) + z0 * Math.sin(rot),
        z = z0 * Math.cos(rot) - a0 * Math.sin(rot);
      worst = Math.max(worst, wall(a, z) + need);
    }
    pa = na;
    pz = nz;
  }
  return worst;
}

test('fitOpenRotation: the least clear rotation, continuous as the wall moves', () => {
  assert.equal(fitOpenRotation(PROFILE, slab(5, 5)), 0, 'a profile clear of the wall turns not at all');
  let prev = Number.NaN;
  for (let outer = 2.2; outer >= 0.2; outer -= 0.01) {
    const wall = slab(outer, 5);
    const rot = fitOpenRotation(PROFILE, wall);
    assert.ok(violation(rot, wall) <= 1e-9, `not clear at outer ${outer.toFixed(2)}`);
    // the least: a little less is not clear
    if (rot > 0.01) assert.ok(violation(rot - 0.01, wall) > 0, `not the least at outer ${outer.toFixed(2)}`);
    if (!Number.isNaN(prev))
      assert.ok(Math.abs(rot - prev) < 0.03, `jumped ${prev.toFixed(3)} → ${rot.toFixed(3)} rad at outer ${outer.toFixed(2)}`);
    prev = rot;
  }
});

test('fitOpenRotation: the least violated rotation when none clears, continuous as the channel closes', () => {
  let prev = Number.NaN;
  let closed = false;
  for (let inner = 2.0; inner >= 0.05; inner -= 0.005) {
    const wall = slab(0.3, inner);
    const rot = fitOpenRotation(PROFILE, wall);
    const v = violation(rot, wall);
    if (v > 0) {
      closed = true;
      // the minimum of the violation: no better angle on a fine scan
      for (let r = 0; r <= 1.5; r += 0.005)
        assert.ok(violation(r, wall) >= v - 2e-3, `rotation ${rot.toFixed(3)} is not the least violated (inner ${inner.toFixed(3)})`);
    }
    if (!Number.isNaN(prev))
      assert.ok(Math.abs(rot - prev) < 0.03, `jumped ${prev.toFixed(3)} → ${rot.toFixed(3)} rad at inner ${inner.toFixed(3)}`);
    prev = rot;
  }
  assert.ok(closed, 'the channel never closed');
});
