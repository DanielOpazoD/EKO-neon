import { classifyHeart, type HeartModel, type HeartPose } from '../heartModel';
import { makeSample, Structure, Tissue } from '../tissue';

export type V3 = [number, number, number];

/** The header the voxeliser writes next to the labels: the grid (mm, mesh coordinates) and the frame's landmarks. */
export interface LabelVolumeHeader {
  voxelMm: number;
  origin: V3;
  dims: V3;
  landmarks: Record<'mitral' | 'tricuspid' | 'aortic' | 'pulmonary' | 'apex', V3>;
}

/** A label volume: one byte per voxel, x fastest, the labels of rodero-voxelize.ts. */
export interface LabelVolume {
  header: LabelVolumeHeader;
  labels: Uint8Array;
}

export interface Fields {
  h: LabelVolumeHeader;
  label: Uint8Array;
  dEpi: Float32Array;
  dIf: Float32Array;
}

/** Radius (mm, mesh scale) of the ball that closes the silhouette: what it adds outside the walls is epicardial fat. */
export const EPI_CLOSING_MM = 4;

/** Tissue and structure of each label of the volume (rodero-voxelize.ts); 0 and unknown labels are outside. */
export const LABELS: Record<number, [Tissue, Structure]> = {
  1: [Tissue.Myocardium, Structure.LvWallLateral],
  2: [Tissue.Myocardium, Structure.RvWall],
  3: [Tissue.Myocardium, Structure.LaWall],
  4: [Tissue.Myocardium, Structure.RaWall],
  5: [Tissue.VesselWall, Structure.AorticRoot],
  6: [Tissue.VesselWall, Structure.PulmonaryArtery],
  7: [Tissue.Blood, Structure.LvCavity],
  8: [Tissue.Blood, Structure.RvCavity],
  9: [Tissue.Valve, Structure.AorticValve],
  10: [Tissue.Valve, Structure.PulmonaryValve],
  11: [Tissue.Blood, Structure.LaAppendage],
  12: [Tissue.Blood, Structure.PulmonaryVein],
  13: [Tissue.Blood, Structure.PulmonaryVein],
  14: [Tissue.Blood, Structure.PulmonaryVein],
  15: [Tissue.Blood, Structure.PulmonaryVein],
  16: [Tissue.Blood, Structure.Svc],
  17: [Tissue.Blood, Structure.Ivc],
  18: [Tissue.Myocardium, Structure.LaAppendage],
  19: [Tissue.VesselWall, Structure.PulmonaryVein],
  20: [Tissue.VesselWall, Structure.PulmonaryVein],
  21: [Tissue.VesselWall, Structure.PulmonaryVein],
  22: [Tissue.VesselWall, Structure.PulmonaryVein],
  23: [Tissue.VesselWall, Structure.Svc],
  24: [Tissue.VesselWall, Structure.Ivc],
  40: [Tissue.Myocardium, Structure.PapillaryMuscle],
  41: [Tissue.Myocardium, Structure.RvPapillary],
  42: [Tissue.Myocardium, Structure.ModeratorBand],
  37: [Tissue.Fibrous, Structure.MitralAnnulus],
  38: [Tissue.Fibrous, Structure.TricuspidAnnulus],
  39: [Tissue.Fibrous, Structure.AortomitralCurtain],
  31: [Tissue.Blood, Structure.LvCavity],
  32: [Tissue.Blood, Structure.RvCavity],
  33: [Tissue.Blood, Structure.LaCavity],
  34: [Tissue.Blood, Structure.RaCavity],
  35: [Tissue.Blood, Structure.AorticRoot],
  36: [Tissue.Blood, Structure.PulmonaryArtery],
};

export const MITRAL_ANNULUS_MM = 1.1;
export const TRICUSPID_ANNULUS_MM = 0.9;
const ANNULUS_WALLS = new Set([1, 2, 3, 4, 5]);
export const FAT = 254;
export const ATRIAL_WALL_MM = 2;
export const ATRIAL_WALLS = new Set([3, 4]);
const ATRIAL_BLOOD = new Set([33, 34, 11, 12, 13, 14, 15, 16, 17]);
export const GROOVE_REACH_MM = 6;
export const SEPTUM_MM = 4;
export const SEPTUM_GAP_MM = 16;
const SEPTUM_FACING_COS = -0.8;
const LA_BLOOD = new Set([33, 11, 12, 13, 14, 15]);
const RA_BLOOD = new Set([34, 16, 17]);
const HINGE_SECTORS = 36;
const HINGE_TOLERANCE_MM = 0.5;
const JUNCTION_REACH_MM = 5;
const JUNCTION_PASSES = 6;
export const CURTAIN = 39;
export const CURTAIN_MM = 2;
export const AORTIC_VALVE_LABEL = 9;
export const ROOT_BLOOD = 35;
export const PULMONARY_VALVE_LABEL = 10;
export const TRUNK_BLOOD = 36;
const INNER_WALLS = [1, 2, 3, 4, 5];
export const CRUX_REACH_MM = 10;
export const SEPTUM_REACH_MM = 5;
const GROOVE_ATRIAL = new Set([3, 4, 18, 37, 38]);
const GROOVE_VENTRICULAR = new Set([1, 2]);

export const APPARATUS = new Set<Structure>([
  Structure.MitralAnterior,
  Structure.MitralPosterior,
  Structure.TricuspidValve,
  Structure.Chordae,
  Structure.PapillaryMuscle,
  Structure.RvPapillary,
  Structure.ModeratorBand,
]);

export const LEAFLETS = new Set<Structure>([
  Structure.MitralAnterior,
  Structure.MitralPosterior,
  Structure.TricuspidValve,
  Structure.Chordae,
]);

export const APPARATUS_HOSTS = new Set([7, 8, 31, 32, 33, 34]);
export const LEFT_HOSTS = new Set([7, 31, 33]);

export const ATRIAL_TISSUE = [
  new Set([33, 3, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22]),
  new Set([34, 4, 16, 17, 23, 24]),
];

export const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const scale = (a: V3, k: number): V3 => [a[0] * k, a[1] * k, a[2] * k];
export const unit = (a: V3): V3 => scale(a, 1 / Math.hypot(...a));
export const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

/** The anatomical frame of the mesh (mm): origin, axes and LV length. */
export function meshFrame(h: LabelVolumeHeader): { o: V3; ex: V3; ey: V3; ez: V3; far: number } {
  const L = h.landmarks;
  const far = Math.hypot(...sub(L.apex, L.mitral));
  const ez = unit(sub(L.apex, L.mitral));
  let ex = sub(L.mitral, L.tricuspid);
  ex = unit(sub(ex, scale(ez, dot(ex, ez))));
  let ey = cross(ez, ex);
  if (dot(sub(L.aortic, L.mitral), ey) < 0) ey = scale(ey, -1);
  return { o: L.mitral, ex, ey, ez, far };
}

/** Squared Euclidean distance transform along one line (Felzenszwalb & Huttenlocher 2012). */
function edtLine(
  f: Float64Array,
  n: number,
  d: Float64Array,
  v: Int32Array,
  z: Float64Array,
): void {
  let k = 0;
  v[0] = 0;
  z[0] = -Infinity;
  z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s = (f[q]! + q * q - (f[v[k]!]! + v[k]! * v[k]!)) / (2 * q - 2 * v[k]!);
    while (s <= z[k]!) {
      k--;
      s = (f[q]! + q * q - (f[v[k]!]! + v[k]! * v[k]!)) / (2 * q - 2 * v[k]!);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1]! < q) k++;
    d[q] = (q - v[k]!) ** 2 + f[v[k]!]!;
  }
}

/** Distance (voxels) from every voxel to the nearest voxel where `feature` holds. */
export function edt(dims: V3, feature: (v: number) => boolean): Float32Array {
  const [nx, ny, nz] = dims;
  const out = new Float32Array(nx * ny * nz);
  const BIG = 1e20;
  for (let v = 0; v < out.length; v++) out[v] = feature(v) ? 0 : BIG;
  const m = Math.max(nx, ny, nz);
  const f = new Float64Array(m),
    d = new Float64Array(m),
    z = new Float64Array(m + 1);
  const w = new Int32Array(m);
  const pass = (n: number, count: number, at: (line: number, q: number) => number): void => {
    for (let line = 0; line < count; line++) {
      for (let q = 0; q < n; q++) f[q] = out[at(line, q)]!;
      edtLine(f, n, d, w, z);
      for (let q = 0; q < n; q++) out[at(line, q)] = d[q]!;
    }
  };
  pass(nx, ny * nz, (l, q) => q + nx * l);
  pass(ny, nx * nz, (l, q) => (l % nx) + nx * (q + ny * Math.floor(l / nx)));
  pass(nz, nx * ny, (l, q) => l + nx * ny * q);
  for (let v = 0; v < out.length; v++) out[v] = Math.sqrt(out[v]!);
  return out;
}

/**
 * The volume's fields over labels the caller owns (the bound heart's copy): the fibrous annuli and the groove fat written
 * into them, the distance beyond the epicardium and to the nearest interface (mm).
 */
export function volumeFields(h: LabelVolumeHeader, label: Uint8Array): Fields {
  const dims = h.dims;
  const [nx, ny] = dims;
  const vox = h.voxelMm;
  const cls = (l: number): number => (l === FAT ? 1 : (LABELS[l]?.[0] ?? 0) + 2);
  // the fibrous annuli: the voxels of the mitral (7) and tricuspid (8) planes within reach of a wall
  {
    const nz = dims[2];
    const at = (i: number, j: number, k: number): number => i + nx * (j + ny * k);
    const rim: [number, number][] = [];
    for (let k = 0; k < nz; k++)
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++) {
          const l = label[at(i, j, k)]!;
          if (l !== 7 && l !== 8) continue;
          const reach = Math.max((l === 7 ? MITRAL_ANNULUS_MM : TRICUSPID_ANNULUS_MM) / vox, 1);
          const n = Math.ceil(reach);
          let near = false;
          for (let w = -n; w <= n && !near; w++)
            for (let u = -n; u <= n && !near; u++)
              for (let t = -n; t <= n && !near; t++) {
                if (t * t + u * u + w * w > reach * reach) continue;
                const a = i + t,
                  b = j + u,
                  c = k + w;
                if (a < 0 || b < 0 || c < 0 || a >= nx || b >= ny || c >= nz) continue;
                if (ANNULUS_WALLS.has(label[at(a, b, c)]!)) near = true;
              }
          if (near) rim.push([at(i, j, k), l === 7 ? 37 : 38]);
        }
    for (const [v, l] of rim) label[v] = l;
  }
  // the atrial walls thinned from outside to ATRIAL_WALL_MM from their cavity (decision 358)
  {
    const dCav = edt(dims, (v) => ATRIAL_BLOOD.has(label[v]!));
    const keep = Math.max(ATRIAL_WALL_MM / vox, 2);
    for (let v = 0; v < label.length; v++)
      if (ATRIAL_WALLS.has(label[v]!) && dCav[v]! > keep) label[v] = 0;
  }
  // the heart's inner walls (decision 364)
  {
    const [sx, sy, sz] = [1, nx, nx * ny];
    const faces6 = [sx, -sx, sy, -sy, sz, -sz];
    const facing = (a: Float32Array, b: Float32Array, v: number): boolean => {
      const ax = a[v + sx]! - a[v - sx]!,
        ay = a[v + sy]! - a[v - sy]!,
        az = a[v + sz]! - a[v - sz]!,
        bx = b[v + sx]! - b[v - sx]!,
        by = b[v + sy]! - b[v - sy]!,
        bz = b[v + sz]! - b[v - sz]!;
      const n = Math.hypot(ax, ay, az) * Math.hypot(bx, by, bz);
      return n > 0 && (ax * bx + ay * by + az * bz) / n < SEPTUM_FACING_COS;
    };
    const reach = SEPTUM_GAP_MM / 2 / vox;
    const apply = (changes: [number, number][]): void => {
      for (const [v, l] of changes) label[v] = l;
    };
    {
      const Mf = meshFrame(h);
      const toAnat = (v: number): V3 => {
        const i = v % nx,
          j = Math.floor(v / nx) % ny,
          kk = Math.floor(v / (nx * ny));
        const m: V3 = [
          h.origin[0] + (i + 0.5) * vox - Mf.o[0],
          h.origin[1] + (j + 0.5) * vox - Mf.o[1],
          h.origin[2] + (kk + 0.5) * vox - Mf.o[2],
        ];
        return [dot(m, Mf.ex), dot(m, Mf.ey), dot(m, Mf.ez)];
      };
      const sector = (a: V3): number =>
        Math.floor(((Math.atan2(a[1], a[0]) + Math.PI) / (2 * Math.PI)) * HINGE_SECTORS) %
        HINGE_SECTORS;
      const hinge = new Array<number>(HINGE_SECTORS).fill(Infinity);
      for (let v = 0; v < label.length; v++)
        if (label[v] === 37) {
          const a = toAnat(v);
          const t = sector(a);
          hinge[t] = Math.min(hinge[t]!, a[2]);
        }
      const aboveHinge = (v: number): boolean => {
        const a = toAnat(v);
        const hz = hinge[sector(a)]!;
        return Number.isFinite(hz) && a[2] <= hz - HINGE_TOLERANCE_MM;
      };
      const own = new Set([33, 3, 7, 31, 37, CURTAIN, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22]);
      const outflow = (v: number): boolean => label[v] === 31 && aboveHinge(v);
      const dL0 = edt(dims, (v) => LA_BLOOD.has(label[v]!));
      const dR0 = edt(dims, (v) => RA_BLOOD.has(label[v]!));
      const leftSide = (v: number): boolean => dL0[v]! <= dR0[v]!;
      for (let pass = 0; pass < JUNCTION_PASSES; pass++) {
        const dL = edt(dims, (v) => LA_BLOOD.has(label[v]!));
        const sheet = new Uint8Array(label.length);
        for (let v = sz; v < label.length - sz; v++)
          if (label[v] === 1 && dL[v]! <= reach && leftSide(v) && aboveHinge(v)) sheet[v] = 1;
        const dSheet = edt(dims, (v) => sheet[v] === 1);
        const taken: number[] = [];
        for (let v = sz; v < label.length - sz; v++) {
          const l = label[v]!;
          if (sheet[v]) taken.push(v);
          else if (
            (l === 0 || l === 3) &&
            dSheet[v]! * vox <= JUNCTION_REACH_MM &&
            dL[v]! <= reach &&
            leftSide(v) &&
            aboveHinge(v)
          )
            taken.push(v);
        }
        const isTaken = new Uint8Array(label.length);
        for (const v of taken) isTaken[v] = 1;
        const dOutflow = edt(dims, outflow);
        const dForeign = edt(dims, (v) => (!own.has(label[v]!) || outflow(v)) && !isTaken[v]);
        for (const v of taken)
          label[v] =
            dOutflow[v]! * vox <= CURTAIN_MM
              ? CURTAIN
              : dForeign[v]! * vox <= ATRIAL_WALL_MM
                ? 3
                : 33;
        if (!taken.some((v) => sheet[v])) break;
      }
    }
    {
      const dL = edt(dims, (v) => LA_BLOOD.has(label[v]!));
      const dR = edt(dims, (v) => RA_BLOOD.has(label[v]!));
      const changes: [number, number][] = [];
      for (let v = sz; v < label.length - sz; v++) {
        const l = label[v]!;
        if (l !== 0 && !ATRIAL_WALLS.has(l)) continue;
        if (dL[v]! > reach || dR[v]! > reach || !facing(dL, dR, v)) continue;
        const d = (dL[v]! - dR[v]!) * vox;
        changes.push([v, Math.abs(d) <= SEPTUM_MM ? (d <= 0 ? 3 : 4) : d < 0 ? 33 : 34]);
      }
      apply(changes);
    }
    {
      const dL = edt(dims, (v) => LA_BLOOD.has(label[v]!));
      const dR = edt(dims, (v) => RA_BLOOD.has(label[v]!));
      const dAo = edt(dims, (v) => label[v] === 5);
      const changes: [number, number][] = [];
      for (let v = sz; v < label.length - sz; v++) {
        const l = label[v]!;
        if ((l !== 0 && !ATRIAL_WALLS.has(l)) || dAo[v]! > reach) continue;
        const left = dL[v]! <= dR[v]!;
        const dA = left ? dL : dR;
        if (dA[v]! > reach || !facing(dA, dAo, v)) continue;
        changes.push([v, dAo[v]! * vox <= ATRIAL_WALL_MM ? (left ? 3 : 4) : left ? 33 : 34]);
      }
      apply(changes);
    }
    {
      const changes: [number, number][] = [];
      for (let v = sz; v < label.length - sz; v++) {
        const l = label[v]!;
        if ((l === 33 || l === 34) && faces6.some((st) => label[v + st] === 0))
          changes.push([v, l === 33 ? 3 : 4]);
      }
      apply(changes);
    }
    {
      const dL = edt(dims, (v) => LA_BLOOD.has(label[v]!));
      const dR = edt(dims, (v) => RA_BLOOD.has(label[v]!));
      const r0 = EPI_CLOSING_MM / vox;
      const out0 = edt(dims, (v) => label[v] !== 0);
      const inside = edt(dims, (v) => out0[v]! > r0);
      const walls = INNER_WALLS.map((w) => edt(dims, (v) => label[v] === w));
      const changes: [number, number][] = [];
      for (let v = sz; v < label.length - sz; v++) {
        if (label[v] !== 0 || dL[v]! > reach || dR[v]! > reach || inside[v]! <= r0) continue;
        let best = 0;
        for (let w = 1; w < walls.length; w++) if (walls[w]![v]! < walls[best]![v]!) best = w;
        changes.push([v, INNER_WALLS[best]!]);
      }
      apply(changes);
    }
  }
  // epicardial fat in the grooves
  const r = EPI_CLOSING_MM / vox;
  const dOut = edt(dims, (v) => label[v] !== 0);
  const dIn = edt(dims, (v) => dOut[v]! > r);
  const reach = GROOVE_REACH_MM / vox;
  const dAtrium = edt(dims, (v) => GROOVE_ATRIAL.has(label[v]!));
  const dVentricle = edt(dims, (v) => GROOVE_VENTRICULAR.has(label[v]!));
  const dLv = edt(dims, (v) => label[v] === 1);
  const dRv = edt(dims, (v) => label[v] === 2);
  const dLa = edt(dims, (v) => label[v] === 3);
  const dRa = edt(dims, (v) => label[v] === 4);
  for (let v = 0; v < label.length; v++)
    if (
      label[v] === 0 &&
      dIn[v]! > r &&
      ((dAtrium[v]! <= reach && dVentricle[v]! <= reach) ||
        (dLv[v]! <= reach && dRv[v]! <= reach)) &&
      !(dLa[v]! <= reach && dRa[v]! <= reach)
    )
      label[v] = FAT;
  const dEpi = edt(dims, (v) => label[v] !== 0);
  for (let v = 0; v < dEpi.length; v++) dEpi[v] = dEpi[v]! * vox;
  const step = [1, -1, nx, -nx, nx * ny, -nx * ny];
  const dIf = edt(dims, (v) => {
    const c = cls(label[v]!);
    for (const s of step) {
      const w = v + s;
      if (w >= 0 && w < label.length && cls(label[w]!) !== c) return true;
    }
    return false;
  });
  for (let v = 0; v < dIf.length; v++) dIf[v] = (dIf[v]! + 0.5) * vox;

  return { h, label, dEpi, dIf };
}

const BAKE_STRIDE = 3;
const BAKED: Partial<Record<Structure, number>> = {
  [Structure.PapillaryMuscle]: 40,
  [Structure.RvPapillary]: 41,
  [Structure.ModeratorBand]: 42,
};

/**
 * A copy of the volume's labels with the analytic heart's papillary muscles and moderator band at end diastole written
 * into its cavity voxels.
 */
export function bakeApparatus(
  vol: LabelVolume,
  heart: HeartModel,
  ed: HeartPose,
  M: { o: V3; ex: V3; ey: V3; ez: V3 },
  F: { origin: V3; ex: V3; ey: V3; ez: V3 },
  k: number,
): Uint8Array {
  const h = vol.header;
  const label = Uint8Array.from(vol.labels);
  const [nx, ny, nz] = h.dims;
  const q = makeSample();
  const bakedAt = (i: number, j: number, kk: number): number => {
    const m: V3 = [
      h.origin[0] + (i + 0.5) * h.voxelMm - M.o[0],
      h.origin[1] + (j + 0.5) * h.voxelMm - M.o[1],
      h.origin[2] + (kk + 0.5) * h.voxelMm - M.o[2],
    ];
    const ax = dot(m, M.ex) * k,
      ay = dot(m, M.ey) * k,
      az = dot(m, M.ez) * k;
    const hx = F.origin[0] + ax * F.ex[0] + ay * F.ey[0] + az * F.ez[0];
    const hy = F.origin[1] + ax * F.ex[1] + ay * F.ey[1] + az * F.ez[1];
    const hz = F.origin[2] + ax * F.ex[2] + ay * F.ey[2] + az * F.ez[2];
    if (!classifyHeart(heart, ed, hx + ed.swingX, hy, hz, q)) return 0;
    return BAKED[q.structure] ?? 0;
  };
  const S = BAKE_STRIDE;
  const bx = Math.ceil(nx / S),
    by = Math.ceil(ny / S),
    bz = Math.ceil(nz / S);
  const hit = new Uint8Array(bx * by * bz);
  for (let kk = 0; kk < nz; kk += S)
    for (let j = 0; j < ny; j += S)
      for (let i = 0; i < nx; i += S)
        if (APPARATUS_HOSTS.has(label[i + nx * (j + ny * kk)]!) && bakedAt(i, j, kk))
          hit[i / S + bx * (j / S + by * (kk / S))] = 1;
  for (let c = 0; c < bz; c++)
    for (let b = 0; b < by; b++)
      for (let a = 0; a < bx; a++) {
        let near = false;
        for (let w = -1; w <= 1 && !near; w++)
          for (let u = -1; u <= 1 && !near; u++)
            for (let t = -1; t <= 1 && !near; t++) {
              const A = a + t,
                B = b + u,
                C = c + w;
              if (
                A >= 0 &&
                B >= 0 &&
                C >= 0 &&
                A < bx &&
                B < by &&
                C < bz &&
                hit[A + bx * (B + by * C)]
              )
                near = true;
            }
        if (!near) continue;
        for (let kk = c * S; kk < Math.min(nz, (c + 1) * S); kk++)
          for (let j = b * S; j < Math.min(ny, (b + 1) * S); j++)
            for (let i = a * S; i < Math.min(nx, (a + 1) * S); i++) {
              const v = i + nx * (j + ny * kk);
              if (!APPARATUS_HOSTS.has(label[v]!)) continue;
              const l = bakedAt(i, j, kk);
              if (l) label[v] = l;
            }
      }
  return label;
}
