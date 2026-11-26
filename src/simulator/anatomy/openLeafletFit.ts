/**
 * Open atrioventricular leaflets kept off the ventricular wall: one fitter for the fibres of the mitral fans and the
 * zones of the tricuspid skirt.
 *
 * An open leaflet is a three-segment profile hanging from its hinge in the plane of its fibre, `a` inward from the hinge
 * (toward the orifice) and `z` apical, built from per-segment angles turned outward from the long axis. The angles are
 * the same for every fibre of a leaflet, so where the wall runs closer than the profile would reach, the profile turns
 * inward about its hinge — every segment angle by the same rotation — the least that leaves a gap of blood between it and
 * the wall (`openWallGap`). The valves differ only in where a point of the profile lies in the heart, which their `wall`
 * callbacks encode.
 *
 * The least clear rotation was the first clear angle of a 0.1 rad scan refined by four bisections, written twice (mitral
 * and tricuspid), and the mitral fibres were also kept 3.5 mm off the papillary muscles. A muscle tip standing just
 * inward of a fibre's wall-limited position splits its clear rotations into two intervals, between the wall and the
 * muscle and inward of it, and from one 1/256 of the cycle to the next the scan fell into one or the other: the anterior
 * free edge jumped 0.6–2.3 cm in a step (1.6–6.6 m/s) in ten of the twelve cases, 2.34 cm in the HOCM case (openRot
 * 1.50 → 0.40 rad), and the posterior one 0.55 cm in the prolapse case. Against the wall alone the violation is
 * quasi-convex in the rotation — it falls as the profile leaves the wall it opens toward and rises as it nears the
 * opposite one — so the clear rotations form one interval whose least element moves continuously with the wall. When no
 * rotation clears it, the scan returned its 1.5 rad limit; the profile now takes the least violated rotation, which is
 * where the clear interval closes, so it does not jump there either. Outside the valves' opening and closing the free
 * edges now move at most 0.53 m/s, and the fitter costs 0.07 ms per pose (0.12 ms before): its scan only asks whether
 * each angle is clear, and the least violated angle is sought only when none is.
 */

/** Blood (cm) between an open leaflet and the ventricular wall, beyond the first few millimetres of leaflet. */
export const OPEN_WALL_GAP_CM = 0.25;
/** Largest inward rotation of an open profile (rad). */
export const OPEN_ROT_MAX = 1.5;
/** Coarse scan of the rotation (rad); the interval found is then refined. */
const SCAN_STEP = 0.1;
const SCAN_N = Math.round(OPEN_ROT_MAX / SCAN_STEP);
/** Bisections on a scan step (0.1 rad / 2⁵ ≈ 0.003 rad, 0.08 mm at the free edge of a 2.5 cm leaflet). */
const BISECTIONS = 5;
/** Golden-section steps for the least violated rotation (0.2 rad · 0.618¹⁰ ≈ 0.0016 rad). */
const GOLDEN_STEPS = 10;
const GOLDEN = 0.381966011250105;

/** Gap (cm) the wall must leave at `s` cm from the hinge: the leaflet leaves the wall there and needs more blood over its first half centimetre. */
export function openWallGap(s: number): number {
  return Math.min(OPEN_WALL_GAP_CM, 0.4 * (s - 0.15));
}

/**
 * Open profile [a1, z1, a2, z2, a3, z3] of three segments of length `segLen` at angles (rad) from the long axis, positive
 * toward the orifice: `a` inward from the hinge, `z` apical.
 */
export function openProfile(angles: readonly number[], segLen: number): Float64Array {
  const out = new Float64Array(6);
  let a = 0,
    z = 0;
  for (let i = 0; i < 3; i++) {
    a += Math.sin(angles[i]!) * segLen;
    z += Math.cos(angles[i]!) * segLen;
    out[i * 2] = a;
    out[i * 2 + 1] = z;
  }
  return out;
}

// sample points of the profile being fitted (vertices and segment midpoints) and the gap each needs (single-threaded)
const pts = new Float64Array(12);
const gaps = new Float64Array(6);
let wallAt: (a: number, z: number) => number = () => 0;

/**
 * Worst wall penetration plus gap over the sample points turned inward by `rot` (≤ 0: clear). Once it exceeds
 * `cutoff` the rest cannot change what the caller does with it, so it returns there; the free edge, which meets the
 * wall first, is tested first.
 */
function violation(rot: number, cutoff = Infinity): number {
  const cr = Math.cos(rot),
    sr = Math.sin(rot);
  let worst = -Infinity;
  for (let j = 5; j >= 0; j--) {
    const need = gaps[j]!;
    if (need <= 0) continue;
    const a0 = pts[j * 2]!,
      z0 = pts[j * 2 + 1]!;
    const v = wallAt(a0 * cr + z0 * sr, z0 * cr - a0 * sr) + need;
    if (v > worst) {
      worst = v;
      if (worst > cutoff) return worst;
    }
  }
  return worst;
}

/** Least clear rotation in (lo, hi], `violation(lo)` > 0 and `violation(hi)` ≤ 0. */
function firstClear(lo: number, hi: number): number {
  for (let i = 0; i < BISECTIONS; i++) {
    const mid = (lo + hi) / 2;
    if (violation(mid, 0) <= 0) hi = mid;
    else lo = mid;
  }
  return hi;
}

/**
 * Least inward rotation (rad, 0…OPEN_ROT_MAX) of the open profile `prof` [a1, z1, a2, z2, a3, z3] that keeps its
 * vertices and segment midpoints `openWallGap` inside the wall, or the least violated rotation when none does.
 * `wall(a, z)` is the signed distance (cm, negative inside) of the cavity at a point of the leaflet's plane.
 */
export function fitOpenRotation(
  prof: ArrayLike<number>,
  wall: (a: number, z: number) => number,
): number {
  let pa = 0,
    pz = 0;
  for (let i = 0; i < 3; i++) {
    const na = prof[i * 2]!,
      nz = prof[i * 2 + 1]!;
    pts[i * 4] = (pa + na) / 2;
    pts[i * 4 + 1] = (pz + nz) / 2;
    pts[i * 4 + 2] = na;
    pts[i * 4 + 3] = nz;
    pa = na;
    pz = nz;
  }
  for (let j = 0; j < 6; j++) gaps[j] = openWallGap(Math.hypot(pts[j * 2]!, pts[j * 2 + 1]!));
  wallAt = wall;
  // the first clear scanned angle: only whether each is clear matters, so the first blocked point settles it
  for (let k = 0; k <= SCAN_N; k++) {
    const r = k * SCAN_STEP;
    if (violation(r, 0) <= 0) return k === 0 ? 0 : firstClear(r - SCAN_STEP, r);
  }
  // No scanned angle is clear: the minimum lies within a step of the least violated one (the violation is
  // quasi-convex). Clear there, the interval it opens is narrower than the scan and its least angle lies below the
  // minimum; otherwise the least violated angle is the answer.
  let best = 0,
    vBest = violation(0);
  for (let k = 1; k <= SCAN_N; k++) {
    const v = violation(k * SCAN_STEP, vBest);
    if (v < vBest) {
      vBest = v;
      best = k * SCAN_STEP;
    }
  }
  const lo0 = Math.max(0, best - SCAN_STEP);
  let lo = lo0,
    hi = Math.min(OPEN_ROT_MAX, best + SCAN_STEP);
  let x1 = lo + GOLDEN * (hi - lo),
    x2 = hi - GOLDEN * (hi - lo);
  let f1 = violation(x1),
    f2 = violation(x2, f1);
  // each new point only has to be compared with the other one: past it, its value is a bound that decides the same
  for (let i = 0; i < GOLDEN_STEPS && f1 > 0 && f2 > 0; i++) {
    if (f1 <= f2) {
      hi = x2;
      x2 = x1;
      f2 = f1;
      x1 = lo + GOLDEN * (hi - lo);
      f1 = violation(x1, f2);
    } else {
      lo = x1;
      x1 = x2;
      f1 = f2;
      x2 = hi - GOLDEN * (hi - lo);
      f2 = violation(x2, f1);
    }
  }
  const rMin = f1 <= f2 ? x1 : x2;
  return Math.min(f1, f2) <= 0 ? firstClear(lo0, rMin) : rMin;
}
