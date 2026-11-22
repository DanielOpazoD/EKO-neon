/**
 * A heart read from a label volume (decision 346), for the CPU tracer's `Scene.heartClassifier`: the mean heart of Rodero
 * et al. 2021 voxelised by `tools/offline/anatomy2/rodero-voxelize.ts` (the data stay outside the repository), placed in the model's heart frame by the anatomical frame both share
 * (`atlasFrame.ts`: origin the mitral centre, z to the LV apex, x from the tricuspid centre to the mitral one, y toward the
 * aortic valve) and scaled so that its LV length is the model's. The end-diastolic shape of the CT, moved by the case's cycle when asked (decision 350).
 *
 * What the mesh does not have is added the way the analytic heart draws it: the epicardial fat that fills the grooves
 * (the heart's silhouette closed with a ball of EPI_CLOSING_MM, kept where it lies between an atrium and a ventricle or
 * between the ventricles: decision 358), the pericardium PERICARDIUM_CM outside it, and on a miss the distance beyond the
 * sac for the thorax. The atrial walls, which the segmentation draws thicker than CT measures them, keep ATRIAL_WALL_MM
 * from their cavity. The CT has no leaflets, chords or papillary muscles: its atrioventricular
 * valve planes are flat discs, taken as blood, and the analytic heart's apparatus is drawn in the cavities (APPARATUS).
 * The pulmonary valve plane is closed valve, as at end diastole; the aortic one is the root's blood, in which the analytic
 * cusps are drawn over the CT's annulus (decision 373). Where the CT segmentation cut the great vessels they carry on as
 * the analytic heart's (decision 372).
 */
import { atlasMarks } from './atlasFrame';
import { anchorsCached } from './anchors';
import { rootRadiusAt } from './aorticValve';
import { ROOT_EXCURSION } from './heartFrame';
import { hingeHeight } from './mitralValve';
import { makeSample } from './tissue';
import { PERICARDIUM_CM } from './classify/pericardium';
import { classifyGreatVessels } from './greatVessels';
import { classifyHeart, type HeartModel, type HeartPose } from './heartModel';
import type { HeartClassifier } from './sceneClassifier';
import { Structure, Tissue, type TissueSample } from './tissue';
import {
  AORTIC_VALVE_LABEL,
  APPARATUS,
  APPARATUS_HOSTS,
  ATRIAL_TISSUE,
  ATRIAL_WALL_MM,
  ATRIAL_WALLS,
  bakeApparatus,
  cross,
  CRUX_REACH_MM,
  CURTAIN,
  CURTAIN_MM,
  dot,
  edt,
  EPI_CLOSING_MM,
  FAT,
  GROOVE_REACH_MM,
  LABELS,
  LEAFLETS,
  LEFT_HOSTS,
  meshFrame,
  MITRAL_ANNULUS_MM,
  PULMONARY_VALVE_LABEL,
  ROOT_BLOOD,
  SEPTUM_GAP_MM,
  SEPTUM_MM,
  SEPTUM_REACH_MM,
  TRICUSPID_ANNULUS_MM,
  TRUNK_BLOOD,
  unit,
  volumeFields,
  type LabelVolume,
  type LabelVolumeHeader,
  type V3,
} from './volume/volumeFields';
import {
  analyticAtrialVolumes,
  AORTIC_CUSP_REACH_CM,
  apicalStretchInverse,
  ATRIAL_PHASE_BINS,
  MITRAL_FADE_CM,
  MITRAL_SECTORS,
  mitralWarpPoint,
  takeMitralRing,
  takeValvePlane,
  type AorticWarp,
  type MitralWarp,
} from './volume/volumeWarps';

export type { AorticWarp, LabelVolume, LabelVolumeHeader, MitralWarp };
export {
  analyticAtrialVolumes,
  AORTIC_CUSP_REACH_CM,
  AORTIC_VALVE_LABEL,
  apicalStretchInverse,
  APPARATUS,
  APPARATUS_HOSTS,
  ATRIAL_PHASE_BINS,
  ATRIAL_TISSUE,
  ATRIAL_WALL_MM,
  CRUX_REACH_MM,
  CURTAIN,
  CURTAIN_MM,
  edt,
  EPI_CLOSING_MM,
  FAT,
  GROOVE_REACH_MM,
  LABELS,
  LEAFLETS,
  LEFT_HOSTS,
  MITRAL_ANNULUS_MM,
  MITRAL_FADE_CM,
  MITRAL_SECTORS,
  mitralWarpPoint,
  PULMONARY_VALVE_LABEL,
  ROOT_BLOOD,
  SEPTUM_GAP_MM,
  SEPTUM_MM,
  TRICUSPID_ANNULUS_MM,
  TRUNK_BLOOD,
};

/** Reach (voxels) and width of the Gaussian that weighs the interface normal on the voxels that border an interface. */
const NORMAL_REACH = 5;
const NORMAL_SIGMA = 2.5;
/** The neighbours the border normal sums, with their Gaussian weights: (dx, dy, dz, weight) in a row. */
const NORMAL_OFFSETS = (() => {
  const o: number[] = [];
  for (let w = -NORMAL_REACH; w <= NORMAL_REACH; w++)
    for (let u = -NORMAL_REACH; u <= NORMAL_REACH; u++)
      for (let t = -NORMAL_REACH; t <= NORMAL_REACH; t++) {
        const r2 = t * t + u * u + w * w;
        if (r2 === 0 || r2 > NORMAL_REACH * NORMAL_REACH) continue;
        o.push(t, u, w, Math.exp(-r2 / (2 * NORMAL_SIGMA * NORMAL_SIGMA)));
      }
  return Float64Array.from(o);
})();
/**
 * Rotation of the LV at end-systole (degrees, positive counterclockwise seen from the apex), at its base and its apex,
 * linear in between: Kocabay et al. 2014 (healthy adults by speckle tracking, base −6.9 ± 3.5°, apex +13.0 ± 6.5°).
 */
export const TWIST_BASE_DEG = -6.9;
export const TWIST_APEX_DEG = 13.0;
/** Height of the slices of the motion tables (cm), and their angular sectors around the LV slice's centre. */
export const SLICE_CM = 0.25;
export const SECTORS = 36;
/** Width (cm) of the band beyond the LV epicardium over which the LV's radial motion and twist fade out. */
export const LV_FADE_CM = 1.5;
/** Heights (cm, from the mitral centre toward the apex) over which the LV's motion starts below the annulus. */
export const LV_BAND_CM: readonly [number, number] = [-1, 0];
/** Heights (cm) over which the atria's radial motion fades out toward the annulus. */
export const ATRIAL_BAND_CM: readonly [number, number] = [0.5, -0.5];
/** Distances (over the RV's slice radius) over which the RV free wall's inward motion fades out. */
export const RV_REACH: readonly [number, number] = [1.6, 2.4];
export const CAVITY_TIP_CM = 0.1;
export const CAP_WALL_CM = 1;
/** Share of the LV length beyond which the LV wall is the apex. */
export const APEX_SHARE = 0.85;
/** Septal → lateral coordinate (cm) over which the base's descent goes from MAPSE (LV side) to TAPSE (RV free wall). */
export const RV_DESCENT_FROM_CM = -1.5;
export const RV_DESCENT_TO_CM = -3.5;
/** Voxels of each atrium's cavity over which the radial scale that gives its volume is solved. */
const ATRIAL_SAMPLES = 4000;


export interface VolumeHeartOptions {
  /**
   * The analytic heart's pose whose leaflets and chords are drawn in the volume's cavities. Its papillary muscles and
   * moderator band are those of `motion.endDiastole` carried with the walls when the volume moves, of this pose when not.
   */
  apparatus?: HeartPose;
  /** The cycle's pose that moves the volume (decision 350), and the end-diastolic pose the volume stands for. */
  motion?: { pose: HeartPose; endDiastole: HeartPose };
}

/**
 * The bound volume as the GPU samples it (decision 353): its labels (baked apparatus, annuli and groove fat included),
 * per voxel the distance beyond the epicardium and to the nearest interface (mm) and the unit normal in mesh axes
 * (octahedral), the LV and RV slice tables (GPU_SLICE_COLUMNS per slice: LV centre, RV centre and radius, then the
 * endocardial and epicardial radius of each sector) and the geometry the shader maps points with.
 */
export interface VolumeGpuData {
  dims: V3;
  labels: Uint8Array;
  fields: Float32Array;
  slices: Float32Array;
  nSlices: number;
  /** Grid origin (mm, mesh), voxel (mm), mesh frame (origin, axes), anatomical frame (origin, axes, heart frame). */
  gridOrigin: V3;
  voxelMm: number;
  mesh: { o: V3; ex: V3; ey: V3; ez: V3; far: number };
  frame: { origin: V3; ex: V3; ey: V3; ez: V3 };
  /** cm of the model per mm of the mesh, LV length (cm), handedness of the twist. */
  k: number;
  lvLength: number;
  hand: number;
  /** Per atrium (left, right): axis centre x, y (cm), reach (cm), 1 when the volume has it. */
  atria: number[];
  /** Where the longitudinal stretch's ramp starts (the cavity's tip) and ends (the epicardial apex), cm (decision 362). */
  apicalRamp: [number, number];
}
/** Columns of a slice in `VolumeGpuData.slices`: LV cx, cy, RV cx, cy, radius, then 36 endocardial and 36 epicardial radii. */
export const GPU_SLICE_COLUMNS = 5 + 2 * SECTORS;

/**
 * The motion of one frame (decision 350), what the backward map needs besides the volume's own geometry: the base's descent
 * on the left and the right (cm), the LV's longitudinal stretch and emptied fraction, the twist at full weight (rad per
 * degree of the profile), the atrial roof (cm), each atrium's radial scale and the RV free wall's.
 */
export interface MotionParams {
  sL: number;
  sR: number;
  lambda: number;
  /**
   * The longitudinal stretch at the epicardial apex (decision 362): it grows linearly from the cavity's at its tip, and
   * its mean over the cap is the thickening of the wall CAP_WALL_CM before the tip.
   */
  capLambda: number;
  alpha: number;
  twist: number;
  roof: number;
  atrialScales: number[];
  rvRatio: number;
}

/** A point carried back to end diastole, in the anatomical frame (cm), and the twist (rad) that carried it. */
export interface ReferencePoint {
  x: number;
  y: number;
  z: number;
  twist: number;
}

export interface VolumeHeart {
  /** The volume's heart: static at end diastole, or moved by `motion`; with the analytic valve apparatus if asked. */
  classifier(options?: VolumeHeartOptions): HeartClassifier;
  /**
   * The backward map of the motion at `pose` (decision 350): where the tissue now at anatomical point (x, y, z) (cm) lay
   * at end diastole. Identity at end diastole.
   */
  toReference(
    pose: HeartPose,
    endDiastole: HeartPose,
  ): (x: number, y: number, z: number, out: ReferencePoint) => void;
  /**
   * The LV cross-sections of the volume at end diastole, every SLICE_CM from the mitral centre: the cavity's centre and,
   * per sector of 360°/SECTORS around it, the endocardial and epicardial radii of the sector's area (cm; slice-major).
   */
  lvSlices: { cx: Float64Array; cy: Float64Array; rEndo: Float64Array; rEpi: Float64Array };
  /** The motion of the frame at `pose` (decision 350), the numbers the GPU's backward map takes (decision 353). */
  motionParams(pose: HeartPose, endDiastole: HeartPose): MotionParams;
  /** What the GPU samples (decision 353): built on first request and kept. */
  gpu(): VolumeGpuData;
  /** The volume's label at a heart-frame point (cm), the groove fat as 254; −1 beyond the grid. */
  labelAt(hx: number, hy: number, hz: number): number;
  /** cm of the model per mm of the mesh. */
  cmPerMm: number;
  /**
   * The crux (decision 360): where the interatrial septum meets the atrioventricular junction at end diastole, in the
   * heart frame (cm); computed on first request. The four-chamber plane of this heart passes through it.
   */
  crux(): [number, number, number];
  /** The map of the root onto the analytic aortic valve at `pose` (decision 373); null when the CT has no valve plane. */
  aorticWarp(pose: HeartPose): AorticWarp | null;
  /** The map of the trunk onto the analytic pulmonary valve at `pose` (decision 375); null without a valve plane. */
  pulmonaryWarp(pose: HeartPose): AorticWarp | null;
  /** The map of the left heart onto the analytic mitral apparatus at `pose` (decision 374); null without an annulus. */
  mitralWarp(pose: HeartPose): MitralWarp | null;
}

/** A label volume in the frame of `heart`, aligned by its end-diastolic pose `endDiastole`. */
export function volumeHeart(
  vol: LabelVolume,
  heart: HeartModel,
  endDiastole: HeartPose,
): VolumeHeart {
  const h0 = vol.header;
  const M = meshFrame(h0);
  const marks = atlasMarks(heart, endDiastole);
  const F = marks.frame;
  const k = marks.lvLength / M.far;
  const label = bakeApparatus(vol, heart, endDiastole, M, F, k);
  const anchors0 = anchorsCached(heart);
  const aorticPlane = takeValvePlane(
    vol.header,
    label,
    M,
    F,
    k,
    anchors0.avAxis,
    AORTIC_VALVE_LABEL,
    ROOT_BLOOD,
  );
  const pulmonaryPlane = takeValvePlane(
    vol.header,
    label,
    M,
    F,
    k,
    anchors0.paDir,
    PULMONARY_VALVE_LABEL,
    TRUNK_BLOOD,
  );
  const { h, dEpi, dIf } = volumeFields(h0, label);
  const mitralRing = takeMitralRing(h, label, M, F, k, endDiastole);
  const dims = h.dims;
  const [nx, ny, nz] = dims;
  const vox = h.voxelMm;
  const cls = (l: number): number => (l === FAT ? 1 : (LABELS[l]?.[0] ?? 0) + 2);

  const at = (i: number, j: number, kk: number): number => i + nx * (j + ny * kk);
  const clampI = (i: number, n: number): number => (i < 0 ? 0 : i >= n ? n - 1 : i);
  /** The twist (rad) of the sample being classified, applied to its normal. */
  let normalTwist = 0;
  /** Material coordinates: where the tissue lay at end diastole, so the speckle moves with it. */
  const setMaterial = (q: TissueSample, x: number, y: number, z: number): void => {
    q.segment = 0;
    q.transmural = -1;
    q.mx = x;
    q.my = y;
    q.mz = z;
    q.extraReflect = 0;
  };
  /** A gradient in mesh axes turned into the heart frame, unit, into `q`. */
  const toHeart = (gx: number, gy: number, gz: number, q: TissueSample): void => {
    const g: V3 = [gx, gy, gz];
    // mesh → anatomical frame → heart frame
    const a0 = dot(g, M.ex),
      a1 = dot(g, M.ey);
    // turned with the twist that carried the tissue here (about the anatomical z)
    const ct = Math.cos(normalTwist),
      st = Math.sin(normalTwist);
    const a: V3 = [a0 * ct - a1 * st, a0 * st + a1 * ct, dot(g, M.ez)];
    const n: V3 = [
      a[0] * F.ex[0] + a[1] * F.ey[0] + a[2] * F.ez[0],
      a[0] * F.ex[1] + a[1] * F.ey[1] + a[2] * F.ez[1],
      a[0] * F.ex[2] + a[1] * F.ey[2] + a[2] * F.ez[2],
    ];
    const l = Math.hypot(...n);
    if (l === 0) {
      q.nx = 0;
      q.ny = 0;
      q.nz = 1;
      return;
    }
    q.nx = n[0] / l;
    q.ny = n[1] / l;
    q.nz = n[2] / l;
  };
  /** Outward normal of a smooth field (central differences). */
  const fieldNormal = (
    fld: Float32Array,
    i: number,
    j: number,
    kk: number,
    q: TissueSample,
  ): void =>
    toHeart(
      fld[at(clampI(i + 1, nx), j, kk)]! - fld[at(clampI(i - 1, nx), j, kk)]!,
      fld[at(i, clampI(j + 1, ny), kk)]! - fld[at(i, clampI(j - 1, ny), kk)]!,
      fld[at(i, j, clampI(kk + 1, nz))]! - fld[at(i, j, clampI(kk - 1, nz))]!,
      q,
    );
  /**
   * Normal of the interfaces around a voxel's tissue, in mesh axes (not unit). Off the interface, the gradient of the
   * distance to it (the direction to the nearest interface, smooth: Euclidean). On the voxels that border it, where that
   * distance has a kink and no gradient, the Gaussian-weighted gradient of the indicator of the voxel's own class over
   * ±NORMAL_REACH voxels: a 3-voxel Sobel followed the staircase of an inclined surface and missed its normal by 10°. Each
   * bordering voxel's normal is worked out once and kept. The GPU's table holds the same normals (`gpu()`).
   */
  const borderNormals = new Map<number, V3>();
  const meshNormal = (i: number, j: number, kk: number): V3 => {
    const v = at(i, j, kk);
    if (dIf[v]! > vox)
      return [
        dIf[at(clampI(i + 1, nx), j, kk)]! - dIf[at(clampI(i - 1, nx), j, kk)]!,
        dIf[at(i, clampI(j + 1, ny), kk)]! - dIf[at(i, clampI(j - 1, ny), kk)]!,
        dIf[at(i, j, clampI(kk + 1, nz))]! - dIf[at(i, j, clampI(kk - 1, nz))]!,
      ];
    const kept = borderNormals.get(v);
    if (kept) return kept;
    const c = cls(label[v]!);
    let gx = 0,
      gy = 0,
      gz = 0;
    const inner =
      i >= NORMAL_REACH &&
      j >= NORMAL_REACH &&
      kk >= NORMAL_REACH &&
      i < nx - NORMAL_REACH &&
      j < ny - NORMAL_REACH &&
      kk < nz - NORMAL_REACH;
    for (let o = 0; o < NORMAL_OFFSETS.length; o += 4) {
      const t = NORMAL_OFFSETS[o]!,
        u = NORMAL_OFFSETS[o + 1]!,
        w = NORMAL_OFFSETS[o + 2]!;
      const l = inner
        ? label[v + t + nx * (u + ny * w)]!
        : label[at(clampI(i + t, nx), clampI(j + u, ny), clampI(kk + w, nz))]!;
      if (cls(l) === c) continue;
      // the other tissue lies along +r: the outward normal points there
      const wt = NORMAL_OFFSETS[o + 3]!;
      gx += wt * t;
      gy += wt * u;
      gz += wt * w;
    }
    const n: V3 = [gx, gy, gz];
    borderNormals.set(v, n);
    return n;
  };
  const classNormal = (i: number, j: number, kk: number, q: TissueSample): void => {
    const n = meshNormal(i, j, kk);
    toHeart(n[0], n[1], n[2], q);
  };

  // heart frame (cm) → anatomical frame (mm of the mesh) → grid (voxels); the last point mapped, in `g`
  const g = { ax: 0, ay: 0, az: 0, fi: 0, fj: 0, fk: 0 };
  const toGrid = (hx: number, hy: number, hz: number): void => {
    const d: V3 = [hx - F.origin[0], hy - F.origin[1], hz - F.origin[2]];
    g.ax = dot(d, F.ex) / k;
    g.ay = dot(d, F.ey) / k;
    g.az = dot(d, F.ez) / k;
    g.fi = (M.o[0] + g.ax * M.ex[0] + g.ay * M.ey[0] + g.az * M.ez[0] - h.origin[0]) / vox;
    g.fj = (M.o[1] + g.ax * M.ex[1] + g.ay * M.ey[1] + g.az * M.ez[1] - h.origin[1]) / vox;
    g.fk = (M.o[2] + g.ax * M.ex[2] + g.ay * M.ey[2] + g.az * M.ez[2] - h.origin[2]) / vox;
  };
  const labelAt = (hx: number, hy: number, hz: number): number => {
    toGrid(hx, hy, hz);
    const i = Math.floor(g.fi),
      j = Math.floor(g.fj),
      kk = Math.floor(g.fk);
    return i < 0 || j < 0 || kk < 0 || i >= nx || j >= ny || kk >= nz ? -1 : label[at(i, j, kk)]!;
  };

  // ---- motion (decision 350): geometry of the volume at end diastole, in the anatomical frame (cm) ----
  const L = marks.lvLength;
  const nSlices = Math.ceil(L / SLICE_CM) + 1;
  const voxCm3 = (vox * k) ** 3;
  /** Every voxel of the labels asked, in the anatomical frame (cm). */
  const eachVoxel = (
    want: Set<number>,
    f: (l: number, x: number, y: number, z: number) => void,
  ): void => {
    for (let kk = 0; kk < nz; kk++)
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++) {
          const l = label[at(i, j, kk)]!;
          if (!want.has(l)) continue;
          const m: V3 = [
            h.origin[0] + (i + 0.5) * vox - M.o[0],
            h.origin[1] + (j + 0.5) * vox - M.o[1],
            h.origin[2] + (kk + 0.5) * vox - M.o[2],
          ];
          f(l, dot(m, M.ex) * k, dot(m, M.ey) * k, dot(m, M.ez) * k);
        }
  };
  const sliceOf = (z: number): number => Math.round(z / SLICE_CM);
  // first pass: the cavities' centres per slice (LV, RV) and per atrium, and the atrial roof
  const cnt = { lv: new Float64Array(nSlices), rv: new Float64Array(nSlices) };
  const sum = {
    lvx: new Float64Array(nSlices),
    lvy: new Float64Array(nSlices),
    rvx: new Float64Array(nSlices),
    rvy: new Float64Array(nSlices),
  };
  const atria = [
    { cav: 33, n: 0, x: 0, y: 0, cx: 0, cy: 0, r: 0 },
    { cav: 34, n: 0, x: 0, y: 0, cx: 0, cy: 0, r: 0 },
  ];
  let zRoof = 0;
  eachVoxel(new Set([31, 32, 3, 4, 33, 34]), (l, x, y, z) => {
    const si = sliceOf(z);
    if (l === 31 || l === 32) {
      if (si < 0 || si >= nSlices) return;
      if (l === 31) {
        cnt.lv[si]!++;
        sum.lvx[si]! += x;
        sum.lvy[si]! += y;
      } else {
        cnt.rv[si]!++;
        sum.rvx[si]! += x;
        sum.rvy[si]! += y;
      }
      return;
    }
    zRoof = Math.min(zRoof, z);
    for (const a2 of atria)
      if (l === a2.cav) {
        a2.n++;
        a2.x += x;
        a2.y += y;
      }
  });
  const centre = (n: Float64Array, sx: Float64Array, sy: Float64Array) => {
    const cx = new Float64Array(nSlices),
      cy = new Float64Array(nSlices);
    let last = -1;
    for (let si = 0; si < nSlices; si++)
      if (n[si]! > 0) {
        cx[si] = sx[si]! / n[si]!;
        cy[si] = sy[si]! / n[si]!;
        if (last < 0) for (let t = 0; t < si; t++) [cx[t], cy[t]] = [cx[si]!, cy[si]!];
        last = si;
      } else if (last >= 0) [cx[si], cy[si]] = [cx[last]!, cy[last]!];
    return { cx, cy };
  };
  const lvC = centre(cnt.lv, sum.lvx, sum.lvy);
  const rvC = centre(cnt.rv, sum.rvx, sum.rvy);
  for (const a2 of atria) {
    a2.cx = a2.x / Math.max(1, a2.n);
    a2.cy = a2.y / Math.max(1, a2.n);
  }
  // second pass: the LV's sector areas around its slice centre, the RV's radius, the atria's reach
  const secCav = new Float64Array(nSlices * SECTORS),
    secMyo = new Float64Array(nSlices * SECTORS);
  const rvArea = new Float64Array(nSlices);
  const atrialDist: number[][] = [[], []];
  const atrialZ: number[][] = [[], []];
  eachVoxel(new Set([1, 31, 32, 33, 34]), (l, x, y, z) => {
    if (l === 33 || l === 34) {
      const a2 = atria[l - 33]!;
      atrialDist[l - 33]!.push(Math.hypot(x - a2.cx, y - a2.cy));
      atrialZ[l - 33]!.push(z);
      return;
    }
    const si = sliceOf(z);
    if (si < 0 || si >= nSlices) return;
    if (l === 32) {
      rvArea[si]! += voxCm3 / SLICE_CM;
      return;
    }
    const th = Math.atan2(y - lvC.cy[si]!, x - lvC.cx[si]!);
    const ti = (Math.floor(((th + Math.PI) / (2 * Math.PI)) * SECTORS) + SECTORS) % SECTORS;
    (l === 31 ? secCav : secMyo)[si * SECTORS + ti]! += voxCm3 / SLICE_CM;
  });
  for (let a = 0; a < 2; a++) {
    const d = [...atrialDist[a]!].sort((p, q) => p - q);
    atria[a]!.r = d.length ? d[Math.floor(0.9 * (d.length - 1))]! : 1;
  }
  // radii of the sectors' areas (a wedge of angle Δθ and area a: r = √(2a/Δθ)), smoothed over the neighbouring sectors
  // and slices so that the noise of the voxels does not ripple the wall
  const dth = (2 * Math.PI) / SECTORS;
  const rawEndo = new Float64Array(nSlices * SECTORS),
    rawEpi = new Float64Array(nSlices * SECTORS);
  for (let c = 0; c < rawEndo.length; c++) {
    rawEndo[c] = Math.sqrt((2 * secCav[c]!) / dth);
    rawEpi[c] = Math.sqrt((2 * (secCav[c]! + secMyo[c]!)) / dth);
  }
  const smoothTable = (t: Float64Array): Float64Array => {
    const o = new Float64Array(t.length);
    for (let si = 0; si < nSlices; si++)
      for (let ti = 0; ti < SECTORS; ti++) {
        let acc = 0,
          wsum = 0;
        for (let ds = -1; ds <= 1; ds++)
          for (let dt = -1; dt <= 1; dt++) {
            const s2 = si + ds;
            if (s2 < 0 || s2 >= nSlices) continue;
            const w = (2 - Math.abs(ds)) * (2 - Math.abs(dt));
            acc += w * t[s2 * SECTORS + ((ti + dt + SECTORS) % SECTORS)]!;
            wsum += w;
          }
        o[si * SECTORS + ti] = acc / wsum;
      }
    return o;
  };
  const lvSlices = {
    cx: lvC.cx,
    cy: lvC.cy,
    rEndo: smoothTable(rawEndo),
    rEpi: smoothTable(rawEpi),
  };
  const rvR = Float64Array.from(rvArea, (a) => Math.sqrt(a / Math.PI));
  // the cavity's tip, where the cap's ramp of the longitudinal stretch starts (decision 362)
  const meanRow = (t: Float64Array, si: number): number => {
    let acc = 0;
    for (let ti = 0; ti < SECTORS; ti++) acc += t[si * SECTORS + ti]!;
    return acc / SECTORS;
  };
  let zTip = L - SLICE_CM;
  for (let si = 1; si < nSlices; si++) {
    const m0 = meanRow(lvSlices.rEndo, si - 1),
      m1 = meanRow(lvSlices.rEndo, si);
    if (m0 >= CAVITY_TIP_CM && m1 < CAVITY_TIP_CM) {
      zTip = Math.min(L - SLICE_CM, (si - 1 + (m0 - CAVITY_TIP_CM) / (m0 - m1)) * SLICE_CM);
      break;
    }
  }
  const capCm = L - zTip;
  /** The cavity's longitudinal stretch for a base descent s0 and the stretch la at the apex: the LV's end stays at L. */
  const cavityLambda = (s0: number, la: number): number =>
    (L - s0 - (la * capCm) / 2) / (zTip + capCm / 2);
  /** The wall's thickening at height z under the cavity's stretch lc and emptied fraction alpha, mean of the sectors. */
  const wallThickening = (z: number, lc: number, alpha: number): number => {
    const si = Math.min(nSlices - 1, Math.max(0, Math.round(z / SLICE_CM)));
    let acc = 0;
    for (let ti = 0; ti < SECTORS; ti++) {
      const re = lvSlices.rEndo[si * SECTORS + ti]!,
        rp = lvSlices.rEpi[si * SECTORS + ti]!;
      const endo = re * Math.sqrt((1 - alpha) / lc),
        epi = Math.sqrt(Math.max(0, rp * rp - alpha * re * re) / lc);
      acc += rp > re ? (epi - endo) / (rp - re) : 1;
    }
    return acc / SECTORS;
  };
  const slice = (z: number, t: Float64Array): number => {
    const f = Math.min(nSlices - 1, Math.max(0, z / SLICE_CM));
    const i0 = Math.floor(f),
      i1 = Math.min(nSlices - 1, i0 + 1);
    return t[i0]! + (t[i1]! - t[i0]!) * (f - i0);
  };
  /** A sector table at height z and angle th, interpolated in both. */
  const sector = (z: number, th: number, t: Float64Array): number => {
    const f = Math.min(nSlices - 1, Math.max(0, z / SLICE_CM));
    const i0 = Math.floor(f),
      i1 = Math.min(nSlices - 1, i0 + 1);
    const g2 = ((th + Math.PI) / dth - 0.5 + SECTORS) % SECTORS;
    const t0 = Math.floor(g2),
      t1 = (t0 + 1) % SECTORS,
      ft = g2 - t0;
    const at2 = (si: number): number =>
      t[si * SECTORS + t0]! + (t[si * SECTORS + t1]! - t[si * SECTORS + t0]!) * ft;
    return at2(i0) + (at2(i1) - at2(i0)) * (f - i0);
  };
  const smooth = (e0: number, e1: number, x: number): number => {
    const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
    return t * t * (3 - 2 * t);
  };
  /** The atrial radial motion's weight at height z and normalised distance d (distance over the atrium's reach). */
  const atrialWeight = (z: number, d: number, r: number): number =>
    smooth(ATRIAL_BAND_CM[0], ATRIAL_BAND_CM[1], z) * (1 - smooth(1, 1 + 1 / r, d));
  /**
   * How much the atrium's cavity grows in cross-section under a radial scale `sc` with the weights the map applies: the
   * mean of the area element's change (1 + (sc − 1)w)(1 + (sc − 1)(w + d·w′)) over a sample of its voxels.
   */
  const atrialSamples = atria.map((a2, i) => {
    const d = atrialDist[i]!,
      z = atrialZ[i]!;
    const stride = Math.max(1, Math.floor(d.length / ATRIAL_SAMPLES));
    const out: [number, number][] = [];
    for (let n = 0; n < d.length; n += stride) out.push([d[n]! / a2.r, z[n]!]);
    return out;
  });
  const areaRatio = (i: number, sc: number): number => {
    const r = atria[i]!.r,
      pts = atrialSamples[i]!;
    if (!pts.length) return 1;
    let m = 0;
    for (const [d, z] of pts) {
      const w = atrialWeight(z, d, r);
      const dw = (atrialWeight(z, d + 1e-3, r) - atrialWeight(z, d - 1e-3, r)) / 2e-3;
      m += (1 + (sc - 1) * w) * (1 + (sc - 1) * (w + d * dw));
    }
    return m / pts.length;
  };
  /** The radial scale that makes the atrium's cross-section grow by `target`, by bisection. */
  const atrialScaleFor = (i: number, target: number): number => {
    let lo = 0.5,
      hi = 2.5;
    for (let it = 0; it < 30; it++) {
      const mid = (lo + hi) / 2;
      if (areaRatio(i, mid) < target) lo = mid;
      else hi = mid;
    }
    return (lo + hi) / 2;
  };
  // right-handed heart frame: positive twist turns from x (lateral) toward y (anterior) seen from the apex
  const det3 = (a: V3, b: V3, c: V3): number => dot(a, cross(b, c));
  const hand = Math.sign(det3(F.ex, F.ey, F.ez)) || 1;

  /** The volume's label at an anatomical point (cm); −1 beyond the grid. */
  const labelAtAnat = (x: number, y: number, z: number): number => {
    const ax = x / k,
      ay = y / k,
      az = z / k;
    const i = Math.floor((M.o[0] + ax * M.ex[0] + ay * M.ey[0] + az * M.ez[0] - h.origin[0]) / vox);
    const j = Math.floor((M.o[1] + ax * M.ex[1] + ay * M.ey[1] + az * M.ez[1] - h.origin[1]) / vox);
    const kk = Math.floor(
      (M.o[2] + ax * M.ex[2] + ay * M.ey[2] + az * M.ez[2] - h.origin[2]) / vox,
    );
    return i < 0 || j < 0 || kk < 0 || i >= nx || j >= ny || kk >= nz ? -1 : label[at(i, j, kk)]!;
  };
  // the atria's scales of each bin of the cycle, solved once (they depend on the phase alone, as their curve does)
  const atrialScaleBins = new Map<number, number[]>();
  const motionParams = (pose: HeartPose, ed: HeartPose): MotionParams => {
    const sL = pose.zAnn - ed.zAnn;
    const sR = pose.tvZ - ed.tvZ;
    const alpha = Math.max(0, 1 - pose.state.lvVolumeMl / pose.state.edvMl);
    // the apical cap thickens as the wall before the cavity's tip does (its mean stretch, the ramp's mid value), and the
    // cavity takes the rest of the base's descent
    let lambda = 1 - sL / L,
      capLambda = 1;
    for (let it = 0; it < 4; it++) {
      capLambda = 2 * wallThickening(zTip - CAP_WALL_CM, lambda, alpha) - lambda;
      lambda = cavityLambda(sL, capLambda);
    }
    const twist = pose.state.contraction * hand * (Math.PI / 180);
    const roof = Math.min(-0.5, zRoof);
    // each atrium follows the volume curve of the analytic atrium: the volume's longitudinal step already stretches it
    // by 1 + s/|roof|, and the radial scale does the rest
    const vNow = analyticAtrialVolumes(heart, pose),
      vEd = analyticAtrialVolumes(heart, ed);
    const ratios = [vNow.la / vEd.la, vNow.ra / vEd.ra];
    const bin = Math.round(pose.state.phase * ATRIAL_PHASE_BINS) % ATRIAL_PHASE_BINS;
    const keptScales = atrialScaleBins.get(bin);
    const atrialScales =
      keptScales ??
      atria.map((a2, i) => {
        // the longitudinal step stretches the atrium above the annular plane by 1 + s/|roof| and compresses what of it lies
        // below the plane (the right atrium reaches below the mitral centre) by 1 − s/L: its mean over the atrium's voxels
        const sA = sL + (sR - sL) * smooth(RV_DESCENT_FROM_CM, RV_DESCENT_TO_CM, a2.cx);
        const pts = atrialSamples[i]!;
        let stretch = 0;
        for (const [, z] of pts) stretch += z < 0 ? 1 + sA / -roof : 1 - sA / L;
        stretch = pts.length ? stretch / pts.length : 1;
        return atrialScaleFor(i, ratios[i]! / stretch);
      });
    if (!keptScales) atrialScaleBins.set(bin, atrialScales);
    const rvRatio = pose.rvScale / ed.rvScale;
    return { sL, sR, lambda, capLambda, alpha, twist, roof, atrialScales, rvRatio };
  };
  const backward = (
    mp: MotionParams,
  ): ((x: number, y: number, z: number, out: ReferencePoint) => void) => {
    const { sL, sR, lambda, capLambda, alpha, twist, roof, atrialScales, rvRatio } = mp;
    /** The longitudinal stretch at reference height zr (decision 362): the cavity's lc, then the cap's ramp to the apex. */
    const stretchAt = (zr: number, lc: number): number =>
      zr <= zTip ? lc : lc + (capLambda - lc) * Math.min(1, (zr - zTip) / capCm);
    return (x, y, z, out) => {
      // the LV's radial motion and twist, in the cross-section, toward the slice's centre
      // (the longitudinal step leaves x and y alone, so the slice is found by undoing it first)
      const s0 = sL + (sR - sL) * smooth(RV_DESCENT_FROM_CM, RV_DESCENT_TO_CM, x);
      // the cavity's stretch where the base descends by s0 (the septum and the RV side descend toward the tricuspid's)
      const lc = cavityLambda(s0, capLambda);
      const zr =
        z >= L || z <= roof
          ? z
          : z >= s0
            ? apicalStretchInverse(z - s0, lc, capLambda, zTip, L)
            : (z - s0) / (1 + s0 / -roof);
      // the radial step keeps the cross-section's area times its height with the LV's stretch (the base's descent at the
      // mitral annulus). With the descent of each point the map shears where the descent turns toward the tricuspid's,
      // and the wall grew 2.6 % instead of 1.5 % by end systole (decision 363)
      const lz = stretchAt(zr, lambda);
      let px = x,
        py = y,
        rot = 0,
        lvReach = 0;
      const wz = smooth(LV_BAND_CM[0], LV_BAND_CM[1], zr);
      if (wz > 0) {
        const cx = slice(zr, lvSlices.cx),
          cy = slice(zr, lvSlices.cy);
        const dx = x - cx,
          dy = y - cy;
        const rNow = Math.hypot(dx, dy);
        // the sector the point lies in now: the twist turns the tissue by at most a sector or two
        const th0 = Math.atan2(dy, dx);
        const re = sector(zr, th0, lvSlices.rEndo),
          rp = sector(zr, th0, lvSlices.rEpi);
        const A = alpha * re * re;
        const rEpiNow = Math.sqrt(Math.max(0, rp * rp - A) / lz);
        lvReach = 1 - smooth(rEpiNow, rEpiNow + LV_FADE_CM, rNow);
        const w = wz * lvReach;
        if (w > 0) {
          const r = rNow + (Math.sqrt(lz * rNow * rNow + A) - rNow) * w;
          const zeta = Math.min(1, Math.max(0, zr / L));
          rot = -twist * (TWIST_BASE_DEG + (TWIST_APEX_DEG - TWIST_BASE_DEG) * zeta) * w;
          const th = Math.atan2(dy, dx) + rot;
          const f = rNow > 1e-9 ? r : 0;
          px = cx + f * Math.cos(th);
          py = cy + f * Math.sin(th);
        }
      }
      // the RV free wall's inward motion (the model's RV scale) toward the RV's slice centre, outside the LV's reach
      if (wz > 0 && rvRatio !== 1 && lvReach < 1) {
        const rcx = slice(zr, rvC.cx),
          rcy = slice(zr, rvC.cy),
          rr = slice(zr, rvR);
        if (rr > 0) {
          const d = Math.hypot(px - rcx, py - rcy) / rr;
          const w = wz * (1 - lvReach) * (1 - smooth(RV_REACH[0], RV_REACH[1], d));
          if (w > 0) {
            const sc = 1 + (rvRatio - 1) * w;
            px = rcx + (px - rcx) / sc;
            py = rcy + (py - rcy) / sc;
          }
        }
      }
      // the atria: their reservoir and booster, radial about each atrium's axis, above the annulus. A point belongs to the
      // atrium whose own scaling carries it back onto that atrium's tissue (the left first); elsewhere to the nearer one,
      // by distance over each one's reach. By distance alone the larger right atrium took the left's septal side and
      // scaled it about its own axis: the left grew 1.15 times where 1.26 was asked.
      const wa = smooth(ATRIAL_BAND_CM[0], ATRIAL_BAND_CM[1], zr);
      if (wa > 0 && (atria[0]!.n || atria[1]!.n)) {
        const back = (ai: number): [number, number] => {
          const a2 = atria[ai]!;
          const d = Math.hypot(px - a2.cx, py - a2.cy) / a2.r;
          const w = atrialWeight(zr, d, a2.r);
          const sc = 1 + (atrialScales[ai]! - 1) * w;
          return [a2.cx + (px - a2.cx) / sc, a2.cy + (py - a2.cy) / sc];
        };
        let chosen: [number, number] | null = null;
        for (let ai = 0; ai < atria.length && !chosen; ai++) {
          if (!atria[ai]!.n) continue;
          const c = back(ai);
          if (ATRIAL_TISSUE[ai]!.has(labelAtAnat(c[0], c[1], zr))) chosen = c;
        }
        if (!chosen) {
          let best = Infinity,
            ai = 0;
          for (let a = 0; a < atria.length; a++) {
            if (!atria[a]!.n) continue;
            const d = Math.hypot(px - atria[a]!.cx, py - atria[a]!.cy) / atria[a]!.r;
            if (d < best) {
              best = d;
              ai = a;
            }
          }
          chosen = back(ai);
        }
        px = chosen[0];
        py = chosen[1];
      }
      out.x = px;
      out.y = py;
      out.z = zr;
      out.twist = rot;
    };
  };
  const toReference = (pose: HeartPose, ed: HeartPose) => backward(motionParams(pose, ed));

  // the anatomical frame (cm) of a heart-frame point, and back
  const toAnat = (hx: number, hy: number, hz: number, o: V3): void => {
    const d: V3 = [hx - F.origin[0], hy - F.origin[1], hz - F.origin[2]];
    o[0] = dot(d, F.ex);
    o[1] = dot(d, F.ey);
    o[2] = dot(d, F.ez);
  };
  const toHeartPoint = (a: ReferencePoint, o: V3): void => {
    for (let c = 0; c < 3; c++)
      o[c] = F.origin[c]! + a.x * F.ex[c]! + a.y * F.ey[c]! + a.z * F.ez[c]!;
  };

  const classifier = (options: VolumeHeartOptions = {}): HeartClassifier => {
    const back = options.motion
      ? toReference(options.motion.pose, options.motion.endDiastole)
      : null;
    const cur: V3 = [0, 0, 0],
      refH: V3 = [0, 0, 0];
    const ref: ReferencePoint = { x: 0, y: 0, z: 0, twist: 0 };
    const warp = options.apparatus ? aorticWarp(options.apparatus) : null;
    const pWarp = options.apparatus ? pulmonaryWarp(options.apparatus) : null;
    const mWarp = options.apparatus ? mitralWarp(options.apparatus) : null;
    const mq: V3 = [0, 0, 0];
    return (hx, hy, hz, q) => {
      let rx = hx,
        ry = hy,
        rz = hz;
      if (back) {
        toAnat(hx, hy, hz, cur);
        back(cur[0], cur[1], cur[2], ref);
        toHeartPoint(ref, refH);
        rx = refH[0];
        ry = refH[1];
        rz = refH[2];
        normalTwist = -ref.twist;
      } else normalTwist = 0;
      toGrid(rx, ry, rz);
      const { ax, ay, az, fi, fj, fk } = g;
      const i = Math.floor(fi),
        j = Math.floor(fj),
        kk = Math.floor(fk);
      q.segment = 0;
      q.transmural = -1;
      const outside = i < 0 || j < 0 || kk < 0 || i >= nx || j >= ny || kk >= nz;
      const v = outside ? -1 : at(i, j, kk);
      const l = outside ? 0 : label[v]!;
      if (l === 0) {
        // beyond the grid: its margin plus the way out to it; inside it, the distance beyond the epicardium
        const de = outside
          ? (dEpi[at(clampI(i, nx), clampI(j, ny), clampI(kk, nz))]! +
              Math.hypot(
                Math.max(0, -fi, fi - nx),
                Math.max(0, -fj, fj - ny),
                Math.max(0, -fk, fk - nz),
              ) *
                vox) *
            k
          : dEpi[v]! * k;
        // where the CT has nothing, the great vessels it cut carry on as the analytic heart's (decision 372)
        q.sdf = de - PERICARDIUM_CM;
        if (options.apparatus && classifyGreatVessels(heart, options.apparatus, hx, hy, hz, q))
          return true;
        if (outside || de >= PERICARDIUM_CM) return false;
        setMaterial(q, rx, ry, rz);
        q.tissue = Tissue.Pericardium;
        q.structure = Structure.Pericardium;
        q.sdf = -Math.min(de, PERICARDIUM_CM - de);
        fieldNormal(dEpi, i, j, kk, q);
        return true;
      }
      // the analytic leaflets and chords where the analytic heart has them now; the papillary muscles and the moderator
      // band are the volume's own voxels, baked at end diastole and carried with the walls (decision 353). In the root
      // over the CT's annulus, the analytic cusps, carried onto it (decision 373)
      if (options.apparatus) {
        let aortic = false;
        if (warp && l === ROOT_BLOOD) {
          const t =
            (rx - warp.centre[0]) * warp.axis[0] +
            (ry - warp.centre[1]) * warp.axis[1] +
            (rz - warp.centre[2]) * warp.axis[2];
          aortic = t >= AORTIC_CUSP_REACH_CM[0] && t <= AORTIC_CUSP_REACH_CM[1];
        }
        // in the trunk over the CT's pulmonary valve plane, the analytic pulmonary cusps (decision 375)
        let pulmonary = false;
        if (pWarp && l === TRUNK_BLOOD) {
          const t =
            (rx - pWarp.centre[0]) * pWarp.axis[0] +
            (ry - pWarp.centre[1]) * pWarp.axis[1] +
            (rz - pWarp.centre[2]) * pWarp.axis[2];
          pulmonary = t >= AORTIC_CUSP_REACH_CM[0] && t <= AORTIC_CUSP_REACH_CM[1];
        }
        if (aortic || pulmonary || APPARATUS_HOSTS.has(l)) {
          const w = aortic ? warp!.m : pulmonary ? pWarp!.m : [];
          const wo = aortic ? warp!.o : pulmonary ? pWarp!.o : [0, 0, 0];
          // the mitral leaflets and their chords carried onto the CT's annulus (decision 374)
          if (!aortic && !pulmonary && mWarp && LEFT_HOSTS.has(l))
            mitralWarpPoint(mWarp, rx, ry, rz, hx, hy, hz, mq);
          else {
            mq[0] = hx;
            mq[1] = hy;
            mq[2] = hz;
          }
          const semilunar = aortic || pulmonary;
          const px = semilunar ? w[0]! * rx + w[1]! * ry + w[2]! * rz + wo[0]! : mq[0],
            py = semilunar ? w[3]! * rx + w[4]! * ry + w[5]! * rz + wo[1]! : mq[1],
            pz = semilunar ? w[6]! * rx + w[7]! * ry + w[8]! * rz + wo[2]! : mq[2];
          if (
            classifyHeart(heart, options.apparatus, px, py, pz, q) &&
            (aortic
              ? q.structure === Structure.AorticValve
              : pulmonary
                ? q.structure === Structure.PulmonaryValve
                : LEAFLETS.has(q.structure))
          )
            return true;
        }
      }
      setMaterial(q, rx, ry, rz);
      const lab = l === FAT ? null : LABELS[l];
      if (l !== FAT && !lab) {
        q.sdf = PERICARDIUM_CM;
        return false;
      }
      q.tissue = lab ? lab[0] : Tissue.Fat;
      q.structure = lab ? lab[1] : Structure.EpicardialFat;
      if (l === 1) {
        // the LV wall by its sector around the long axis, the apex beyond 85 % of the length
        const phi = Math.atan2(ay, ax);
        q.structure =
          az > APEX_SHARE * M.far
            ? Structure.LvApex
            : Math.abs(phi) < Math.PI / 4
              ? Structure.LvWallLateral
              : Math.abs(phi) > (3 * Math.PI) / 4
                ? Structure.LvWallSeptal
                : phi > 0
                  ? Structure.LvWallAnterior
                  : Structure.LvWallInferior;
      }
      q.sdf = -dIf[v]! * k;
      classNormal(i, j, kk, q);
      return true;
    };
  };
  let gpuData: VolumeGpuData | null = null;
  const gpu = (): VolumeGpuData => {
    if (gpuData) return gpuData;
    const n = label.length;
    const fields = new Float32Array(4 * n);
    for (let kk = 0; kk < nz; kk++)
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++) {
          const v = at(i, j, kk);
          const [gx, gy, gz] = meshNormal(i, j, kk);
          // octahedral code of the unit normal: (0, 0) for none, which decodes to +z
          const l1 = Math.abs(gx) + Math.abs(gy) + Math.abs(gz);
          let ox = 0,
            oy = 0;
          if (l1 > 0) {
            ox = gx / l1;
            oy = gy / l1;
            if (gz < 0) {
              const tx = (1 - Math.abs(oy)) * (ox >= 0 ? 1 : -1),
                ty = (1 - Math.abs(ox)) * (oy >= 0 ? 1 : -1);
              ox = tx;
              oy = ty;
            }
          }
          fields[4 * v] = dEpi[v]!;
          fields[4 * v + 1] = dIf[v]!;
          fields[4 * v + 2] = ox;
          fields[4 * v + 3] = oy;
        }
    const slices = new Float32Array(nSlices * GPU_SLICE_COLUMNS);
    for (let si = 0; si < nSlices; si++) {
      const o = si * GPU_SLICE_COLUMNS;
      slices[o] = lvSlices.cx[si]!;
      slices[o + 1] = lvSlices.cy[si]!;
      slices[o + 2] = rvC.cx[si]!;
      slices[o + 3] = rvC.cy[si]!;
      slices[o + 4] = rvR[si]!;
      for (let t = 0; t < SECTORS; t++) {
        slices[o + 5 + t] = lvSlices.rEndo[si * SECTORS + t]!;
        slices[o + 5 + SECTORS + t] = lvSlices.rEpi[si * SECTORS + t]!;
      }
    }
    gpuData = {
      dims,
      labels: label,
      fields,
      slices,
      nSlices,
      gridOrigin: h.origin,
      voxelMm: vox,
      mesh: M,
      frame: { origin: F.origin, ex: F.ex, ey: F.ey, ez: F.ez },
      k,
      lvLength: L,
      hand,
      atria: atria.flatMap((a2) => [a2.cx, a2.cy, a2.r, a2.n ? 1 : 0]),
      apicalRamp: [zTip, L],
    };
    return gpuData;
  };
  let cruxPoint: [number, number, number] | null = null;
  const crux = (): [number, number, number] => {
    if (cruxPoint) return cruxPoint;
    // the interatrial septum (atrial wall near both atrial cavities) within CRUX_REACH_MM of the atrioventricular planes
    const dAv = edt(
      h.dims,
      (v) => label[v] === 7 || label[v] === 8 || label[v] === 37 || label[v] === 38,
    );
    const dLa = edt(h.dims, (v) => label[v] === 33);
    const dRa = edt(h.dims, (v) => label[v] === 34);
    const reach = CRUX_REACH_MM / vox,
      septum = SEPTUM_REACH_MM / vox;
    let n = 0;
    const a: V3 = [0, 0, 0];
    for (let kk = 0; kk < nz; kk++)
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++) {
          const v = at(i, j, kk);
          if (
            !ATRIAL_WALLS.has(label[v]!) ||
            dAv[v]! > reach ||
            dLa[v]! > septum ||
            dRa[v]! > septum
          )
            continue;
          const m: V3 = [
            h.origin[0] + (i + 0.5) * vox - M.o[0],
            h.origin[1] + (j + 0.5) * vox - M.o[1],
            h.origin[2] + (kk + 0.5) * vox - M.o[2],
          ];
          a[0] += dot(m, M.ex) * k;
          a[1] += dot(m, M.ey) * k;
          a[2] += dot(m, M.ez) * k;
          n++;
        }
    if (!n) throw new Error('volume heart: no interatrial septum near the atrioventricular planes');
    const c: V3 = [a[0] / n, a[1] / n, a[2] / n];
    cruxPoint = [0, 1, 2].map(
      (q) => F.origin[q]! + c[0] * F.ex[q]! + c[1] * F.ey[q]! + c[2] * F.ez[q]!,
    ) as [number, number, number];
    return cruxPoint;
  };
  const aorticWarp = (pose: HeartPose): AorticWarp | null => {
    if (!aorticPlane) return null;
    const A = anchorsCached(heart);
    return semilunarWarp(
      aorticPlane,
      [A.avCenter.x, A.avCenter.y, A.avCenter.z + pose.zAnn * ROOT_EXCURSION],
      [A.avAxis.x, A.avAxis.y, A.avAxis.z],
      rootRadiusAt(pose.valves.root, 0, 0),
    );
  };
  const pulmonaryWarp = (pose: HeartPose): AorticWarp | null => {
    if (!pulmonaryPlane) return null;
    const A = anchorsCached(heart);
    return semilunarWarp(
      pulmonaryPlane,
      [A.rvotB.x, A.rvotB.y, A.rvotB.z + pose.pvZ],
      [A.paDir.x, A.paDir.y, A.paDir.z],
      A.pvR,
    );
  };
  /** A CT valve plane taken onto an analytic valve centred at `ca`, on axis `na`, of annulus radius `ra`. */
  const semilunarWarp = (
    plane: { centre: V3; axis: V3; radius: number },
    ca: V3,
    na: V3,
    ra: number,
  ): AorticWarp => {
    const nr = plane.axis;
    // the shortest turn from the CT's axis onto the analytic one (Rodrigues), scaled to the analytic annulus
    const v: V3 = [
      nr[1] * na[2] - nr[2] * na[1],
      nr[2] * na[0] - nr[0] * na[2],
      nr[0] * na[1] - nr[1] * na[0],
    ];
    const c = dot(nr, na);
    const f = 1 / (1 + c);
    const sc = ra / plane.radius;
    const R = [
      1 - f * (v[1] * v[1] + v[2] * v[2]),
      -v[2] + f * v[0] * v[1],
      v[1] + f * v[0] * v[2],
      v[2] + f * v[0] * v[1],
      1 - f * (v[0] * v[0] + v[2] * v[2]),
      -v[0] + f * v[1] * v[2],
      -v[1] + f * v[0] * v[2],
      v[0] + f * v[1] * v[2],
      1 - f * (v[0] * v[0] + v[1] * v[1]),
    ];
    const m = R.map((x) => x * sc);
    const C = plane.centre;
    const o: V3 = [0, 1, 2].map(
      (r) => ca[r]! - (m[3 * r]! * C[0] + m[3 * r + 1]! * C[1] + m[3 * r + 2]! * C[2]),
    ) as V3;
    return { centre: C, axis: nr, m, o };
  };

  const mitralWarp = (pose: HeartPose): MitralWarp | null => {
    if (!mitralRing) return null;
    const mv = pose.valves.mitral;
    const sectors = new Float64Array(4 * MITRAL_SECTORS);
    for (let t = 0; t < MITRAL_SECTORS; t++) {
      const a = -Math.PI + ((t + 0.5) * 2 * Math.PI) / MITRAL_SECTORS;
      const dx = Math.cos(a),
        dy = Math.sin(a);
      // the D-shaped annulus: the circle of radius R cut by the straight curtain at D toward the aortic valve
      const c = dx * mv.ux + dy * mv.uy;
      const r = c > 0 && mv.D / c < mv.R ? mv.D / c : mv.R;
      const v = r * c,
        u = r * (dx * -mv.uy + dy * mv.ux);
      sectors[4 * t] = mitralRing.radius[t]!;
      sectors[4 * t + 1] = mitralRing.height[t]!;
      sectors[4 * t + 2] = r;
      sectors[4 * t + 3] = mv.cz + hingeHeight(mv.R, mv.D, mv.saddle, mv.lift, u, v);
    }
    return { sectors, centre: [mitralRing.cx, mitralRing.cy, mv.cx, mv.cy] };
  };

  return {
    classifier,
    toReference,
    motionParams,
    gpu,
    lvSlices,
    labelAt,
    cmPerMm: k,
    crux,
    aorticWarp,
    pulmonaryWarp,
    mitralWarp,
  };
}
