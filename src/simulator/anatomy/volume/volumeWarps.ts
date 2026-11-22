import { heartAnchors } from '../anchors';
import { classifyHeart, type HeartModel, type HeartPose } from '../heartModel';
import { makeSample, Structure } from '../tissue';
import { dot, type LabelVolumeHeader, type V3 } from './volumeFields';

export const AORTIC_CUSP_REACH_CM: readonly [number, number] = [-0.5, 2.5];

/** The CT's aortic valve plane in the heart frame and the map of a root point to the analytic valve (decision 373). */
export interface AorticWarp {
  /** The disc's centre (cm) and unit axis toward the aorta, at end diastole. */
  centre: V3;
  axis: V3;
  /** Row-major 3×3 and offset: analytic point = m · reference point + o. */
  m: number[];
  o: V3;
}

export const MITRAL_SECTORS = 36;
export const MITRAL_FADE_CM: readonly [number, number] = [1, 2.5];

/**
 * The map of the left heart's points onto the analytic mitral apparatus at a pose (decision 374): per sector of
 * MITRAL_SECTORS, the CT's hinge radius and height and the analytic ones (cm), and the two centres (x, y).
 */
export interface MitralWarp {
  /** Per sector: CT radius, CT height, analytic radius, analytic height. */
  sectors: Float64Array;
  /** The CT's centre (the analytic valve's at end diastole) and the analytic valve's now. */
  centre: [number, number, number, number];
}

/**
 * Where, along the LV (cm from the mitral centre at end diastole), lies the tissue that is now u cm apical of the
 * descended base (decision 362), when the longitudinal stretch is lc up to z0, grows linearly to la at z1 and stays la
 * beyond: the inverse of the stretch's integral.
 */
export function apicalStretchInverse(
  u: number,
  lc: number,
  la: number,
  z0: number,
  z1: number,
): number {
  const w = z1 - z0,
    u1 = lc * z0,
    u2 = u1 + ((lc + la) * w) / 2;
  if (u <= u1) return u / lc;
  if (u >= u2) return z1 + (u - u2) / la;
  const a = (la - lc) / (2 * w),
    c = u1 - u;
  return z0 + (-2 * c) / (lc + Math.sqrt(lc * lc - 4 * a * c));
}

const ATRIAL_GRID_CM = 0.5;
export const ATRIAL_PHASE_BINS = 48;
const atrialCache = new WeakMap<HeartModel, Map<number, { la: number; ra: number }>>();

/**
 * Volumes (mL) of the analytic heart's left and right atrial cavities at `pose`, counted on a grid of ATRIAL_GRID_CM
 * around each atrium: the curves the volume's atria follow (decision 350).
 */
export function analyticAtrialVolumes(
  heart: HeartModel,
  pose: HeartPose,
): { la: number; ra: number } {
  const bin = Math.round(pose.state.phase * ATRIAL_PHASE_BINS) % ATRIAL_PHASE_BINS;
  let bins = atrialCache.get(heart);
  if (!bins) atrialCache.set(heart, (bins = new Map<number, { la: number; ra: number }>()));
  const kept = bins.get(bin);
  if (kept) return kept;
  const A = heartAnchors(heart);
  const q = makeSample();
  const count = (
    c: { x: number; y: number; z: number },
    r: { x: number; y: number; z: number },
    want: Structure,
  ) => {
    let n = 0;
    const h = ATRIAL_GRID_CM;
    for (let x = c.x - r.x - 1.5; x <= c.x + r.x + 1.5; x += h)
      for (let y = c.y - r.y - 1.5; y <= c.y + r.y + 1.5; y += h)
        for (let z = c.z - r.z - 2; z <= c.z + r.z + 2; z += h)
          if (classifyHeart(heart, pose, x + pose.swingX, y, z, q) && q.structure === want) n++;
    return n * h ** 3;
  };
  const v = {
    la: count(A.laCenter, A.laR, Structure.LaCavity),
    ra: count(A.raCenter, A.raR, Structure.RaCavity),
  };
  bins.set(bin, v);
  return v;
}

/**
 * The CT's mitral hinge (decision 374): per sector of MITRAL_SECTORS around the analytic valve's centre at end diastole,
 * the inner radius of the annulus (its 10th percentile, where the leaflets attach) and its most atrial height, in the
 * heart frame (cm).
 */
export function takeMitralRing(
  h: LabelVolumeHeader,
  label: Uint8Array,
  M: { o: V3; ex: V3; ey: V3; ez: V3 },
  F: { origin: V3; ex: V3; ey: V3; ez: V3 },
  k: number,
  ed: HeartPose,
): { cx: number; cy: number; radius: Float64Array; height: Float64Array } | null {
  const [nx, ny] = h.dims;
  const cx = ed.valves.mitral.cx,
    cy = ed.valves.mitral.cy;
  const rs: number[][] = Array.from({ length: MITRAL_SECTORS }, () => []);
  const height = new Float64Array(MITRAL_SECTORS).fill(Infinity);
  for (let v = 0; v < label.length; v++) {
    if (label[v] !== 37) continue;
    const i = v % nx,
      j = Math.floor(v / nx) % ny,
      kk = Math.floor(v / (nx * ny));
    const m: V3 = [
      h.origin[0] + (i + 0.5) * h.voxelMm - M.o[0],
      h.origin[1] + (j + 0.5) * h.voxelMm - M.o[1],
      h.origin[2] + (kk + 0.5) * h.voxelMm - M.o[2],
    ];
    const a: V3 = [dot(m, M.ex) * k, dot(m, M.ey) * k, dot(m, M.ez) * k];
    const p = [0, 1, 2].map(
      (c) => F.origin[c]! + a[0] * F.ex[c]! + a[1] * F.ey[c]! + a[2] * F.ez[c]!,
    );
    const t =
      Math.floor(
        ((Math.atan2(p[1]! - cy, p[0]! - cx) + Math.PI) / (2 * Math.PI)) * MITRAL_SECTORS,
      ) % MITRAL_SECTORS;
    rs[t]!.push(Math.hypot(p[0]! - cx, p[1]! - cy));
    height[t] = Math.min(height[t]!, p[2]!);
  }
  if (rs.every((r) => r.length === 0)) return null;
  const radius = new Float64Array(MITRAL_SECTORS);
  rs.forEach((r, t) => {
    r.sort((a, b) => a - b);
    radius[t] = r.length ? r[Math.floor(0.1 * (r.length - 1))]! : NaN;
  });
  for (let t = 0; t < MITRAL_SECTORS; t++)
    for (let d = 1; !Number.isFinite(radius[t]) && d < MITRAL_SECTORS; d++)
      for (const u of [t + d, t - d]) {
        const w = (u + MITRAL_SECTORS) % MITRAL_SECTORS;
        if (!Number.isFinite(radius[t]) && rs[w]!.length) {
          radius[t] = radius[w]!;
          height[t] = height[w]!;
        }
      }
  return { cx, cy, radius, height };
}

/**
 * Where a left-heart point is classified against the analytic mitral apparatus (decision 374), into `out`.
 */
export function mitralWarpPoint(
  w: MitralWarp,
  rx: number,
  ry: number,
  rz: number,
  hx: number,
  hy: number,
  hz: number,
  out: V3,
): void {
  const [cxR, cyR, cxA, cyA] = w.centre;
  const dx = rx - cxR,
    dy = ry - cyR;
  const f = ((Math.atan2(dy, dx) + Math.PI) / (2 * Math.PI)) * MITRAL_SECTORS - 0.5;
  const i0 = Math.floor(f);
  const fr = f - i0;
  const a = ((i0 % MITRAL_SECTORS) + MITRAL_SECTORS) % MITRAL_SECTORS,
    b = (a + 1) % MITRAL_SECTORS;
  const S = w.sectors;
  const rR = S[4 * a]! + fr * (S[4 * b]! - S[4 * a]!),
    zR = S[4 * a + 1]! + fr * (S[4 * b + 1]! - S[4 * a + 1]!),
    rA = S[4 * a + 2]! + fr * (S[4 * b + 2]! - S[4 * a + 2]!),
    zA = S[4 * a + 3]! + fr * (S[4 * b + 3]! - S[4 * a + 3]!);
  const depth = rz - zR;
  const wt =
    depth <= MITRAL_FADE_CM[0]
      ? 1
      : depth >= MITRAL_FADE_CM[1]
        ? 0
        : (MITRAL_FADE_CM[1] - depth) / (MITRAL_FADE_CM[1] - MITRAL_FADE_CM[0]);
  const sc = rA / rR;
  out[0] = wt * (cxA + dx * sc) + (1 - wt) * hx;
  out[1] = wt * (cyA + dy * sc) + (1 - wt) * hy;
  out[2] = wt * (rz + zA - zR) + (1 - wt) * hz;
}

/**
 * The CT's aortic or pulmonary valve plane (decisions 373 & 375): its centre, axis and radius in the heart frame (cm),
 * and its voxels converted to vessel blood. Null when the volume has none.
 */
export function takeValvePlane(
  h: LabelVolumeHeader,
  label: Uint8Array,
  M: { o: V3; ex: V3; ey: V3; ez: V3 },
  F: { origin: V3; ex: V3; ey: V3; ez: V3 },
  k: number,
  toward: { x: number; y: number; z: number },
  valve: number,
  blood: number,
): { centre: V3; axis: V3; radius: number } | null {
  const [nx, ny] = h.dims;
  const pts: V3[] = [];
  for (let v = 0; v < label.length; v++) {
    if (label[v] !== valve) continue;
    const i = v % nx,
      j = Math.floor(v / nx) % ny,
      kk = Math.floor(v / (nx * ny));
    const m: V3 = [
      h.origin[0] + (i + 0.5) * h.voxelMm - M.o[0],
      h.origin[1] + (j + 0.5) * h.voxelMm - M.o[1],
      h.origin[2] + (kk + 0.5) * h.voxelMm - M.o[2],
    ];
    const a: V3 = [dot(m, M.ex) * k, dot(m, M.ey) * k, dot(m, M.ez) * k];
    pts.push(
      [0, 1, 2].map(
        (c) => F.origin[c]! + a[0] * F.ex[c]! + a[1] * F.ey[c]! + a[2] * F.ez[c]!,
      ) as V3,
    );
    label[v] = blood;
  }
  if (pts.length < 10) return null;
  const centre: V3 = [0, 0, 0];
  for (const p of pts) for (let c = 0; c < 3; c++) centre[c]! += p[c]! / pts.length;
  const C = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (const p of pts)
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 3; c++)
        C[3 * r + c]! += ((p[r]! - centre[r]!) * (p[c]! - centre[c]!)) / pts.length;
  const power = (ortho: V3[]): V3 => {
    let x: V3 = [0.3, 0.5, 0.8];
    for (let it = 0; it < 100; it++) {
      let y: V3 = [0, 1, 2].map(
        (r) => C[3 * r]! * x[0] + C[3 * r + 1]! * x[1] + C[3 * r + 2]! * x[2],
      ) as V3;
      for (const o of ortho) {
        const d = dot(y, o);
        y = [y[0] - d * o[0], y[1] - d * o[1], y[2] - d * o[2]];
      }
      const l = Math.hypot(...y);
      x = [y[0] / l, y[1] / l, y[2] / l];
    }
    return x;
  };
  const e1 = power([]),
    e2 = power([e1]);
  let axis: V3 = [
    e1[1] * e2[2] - e1[2] * e2[1],
    e1[2] * e2[0] - e1[0] * e2[2],
    e1[0] * e2[1] - e1[1] * e2[0],
  ];
  if (axis[0] * toward.x + axis[1] * toward.y + axis[2] * toward.z < 0)
    axis = [-axis[0], -axis[1], -axis[2]];
  const radial = pts
    .map((p) => {
      const d: V3 = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
      const t = dot(d, axis);
      return Math.hypot(d[0] - t * axis[0], d[1] - t * axis[1], d[2] - t * axis[2]);
    })
    .sort((a, b) => a - b);
  return { centre, axis, radius: radial[Math.floor(0.98 * (radial.length - 1))]! };
}
