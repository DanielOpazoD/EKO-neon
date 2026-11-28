import type { HeartModel, HeartPose } from './heartModel';
import { classifyHeart } from './heartModel';
import { heartToTorso, torsoToHeart } from './heartFrame';
import {
  classifyThorax,
  descAortaScale,
  diaphragmY,
  DESC_AORTA_X,
  DESC_AORTA_Z,
  type ThoraxModel,
} from './thoraxModel';
import { makeSample, Structure, Tissue } from './tissue';
import { buildCoronaryMeshes } from './coronaryMesh';

/**
 * Surface meshes of the heart, extracted from the same implicit model the ultrasound beam cuts
 * (`classifyHeart`). The 3D navigator therefore cannot drift from the image: both read one anatomy.
 *
 * Extraction uses surface nets over an occupancy grid with volume-preserving Taubin smoothing,
 * geometric concavity + inter-group sulcus ambient occlusion, physiological separation of arteries,
 * veins, and atria, and epicardial coronary vessels snapped onto the active surface.
 */
export interface MeshGroup {
  id: string;
  label: string;
  color: number;
  opacity: number;
  positions: Float32Array;
  /** Per-vertex normals, area-weighted from the triangles: the only rule that survives a one-cell wall. */
  normals: Float32Array;
  indices: Uint32Array;
  /**
   * Per-vertex ambient occlusion factor (~0.55–1.05): darkens anatomical sulci, inter-chamber creases,
   * and great-vessel roots so surfaces read with sculptural 3D depth instead of flat uniform shading.
   */
  ao?: Float32Array;
  /**
   * Per-vertex inter-structure sulcus proximity (0..1) on myocardium: identifies the interventricular
   * and atrioventricular grooves for subtle subepicardial tissue tinting.
   */
  groove?: Float32Array;
  /**
   * LV myocardium only: the segment code (`lvSegments.ts`, decision 152) of the tissue each vertex bounds, from the
   * classifier at this phase's grid (0 on the papillary muscles). Each phase is classified anew, so the colour follows
   * the tissue without assuming that vertex indices match between phases.
   */
  segments?: Uint8Array;
}

export interface HeartMeshOptions {
  /** Grid spacing in cm (0.26–0.30 is crisp and fast enough for a navigator). */
  stepCm?: number;
  /** Extraction bounds in the heart frame (cm). */
  bounds?: { min: [number, number, number]; max: [number, number, number] };
  smoothing?: number;
  /** Shared thorax supplies the descending segment, trimmed neatly to the cardiac silhouette. */
  thorax?: ThoraxModel;
  /** Set false on secondary cardiac phases to reuse the thoracic aorta mesh from phase 0. */
  includeAorta?: boolean;
  /** Set false to omit epicardial coronary vessels (defaults to true). */
  includeCoronaries?: boolean;
}

type GroupSpec = {
  id: string;
  label: string;
  color: number;
  opacity: number;
  has: (s: Structure, t: Tissue) => boolean;
  /**
   * For a wall thinner than the grid: the structures the mesh may grow into, `cells` deep, from the side facing them.
   * The extracted surface on that side moves inward by as much; the outer (epicardial) surface stays where it is.
   */
  thicken?: { into: (s: Structure) => boolean; cells: number };
  /** Laplacian passes for this group, over the default. */
  smoothing?: number;
};

const isLvWall = (s: Structure): boolean =>
  s === Structure.LvWallSeptal ||
  s === Structure.LvWallLateral ||
  s === Structure.LvWallAnterior ||
  s === Structure.LvWallInferior ||
  s === Structure.LvApex;

/** Groups the navigator can show or hide independently (solid, natural anatomical palette). */
export const MESH_GROUPS: GroupSpec[] = [
  {
    // LV + RV myocardium fused into ONE continuous biventricular surface (with a constrained sulcus
    // fillet bridging the RV–LV insertions). Eliminates the detached "shell/cape" overhang seam
    // between separate LV and RV meshes while keeping both interior cavities hollow for cut views
    // and preserving exact AHA 17-segment tagging on LV wall vertices.
    id: 'lv-myocardium',
    label: 'Miocardio (VI + VD)',
    color: 0x8e242a,
    opacity: 1,
    has: (s) =>
      isLvWall(s) ||
      s === Structure.PapillaryMuscle ||
      s === Structure.RvWall ||
      s === Structure.ModeratorBand ||
      s === Structure.RvPapillary,
    smoothing: 3,
  },
  {
    id: 'lv-cavity',
    label: 'Cavidad VI',
    color: 0x7c1a24,
    opacity: 0.55,
    has: (s) => s === Structure.LvCavity || s === Structure.Lvot,
  },
  {
    id: 'rv-cavity',
    label: 'Cavidad VD',
    color: 0x2c4788,
    opacity: 0.52,
    has: (s) => s === Structure.RvCavity || s === Structure.Rvot,
  },
  {
    id: 'left-atrium',
    label: 'Aurícula izquierda',
    color: 0x7d2633,
    opacity: 1,
    has: (s) =>
      s === Structure.LaCavity || s === Structure.LaWall || s === Structure.LaAppendage,
  },
  {
    id: 'right-atrium',
    label: 'Aurícula derecha',
    color: 0x514166,
    opacity: 1,
    has: (s) =>
      s === Structure.RaCavity || s === Structure.RaWall || s === Structure.RaAppendage,
  },
  {
    id: 'valves',
    label: 'Válvulas',
    color: 0xe3d4a3,
    opacity: 1,
    has: (s, t) =>
      t === Tissue.Valve ||
      s === Structure.Chordae ||
      s === Structure.MitralAnnulus ||
      s === Structure.TricuspidAnnulus,
  },
  {
    id: 'aortic-root',
    label: 'Raíz aórtica',
    color: 0xb03036,
    opacity: 1,
    has: (s) => s === Structure.AorticRoot,
  },
  {
    id: 'pulmonary-artery',
    label: 'Arteria pulmonar',
    color: 0x455b85,
    opacity: 1,
    has: (s) => s === Structure.PulmonaryArtery,
  },
  {
    id: 'venae-cavae',
    label: 'Venas cavas',
    color: 0x36507a,
    opacity: 1,
    has: (s) => s === Structure.Svc || s === Structure.Ivc || s === Structure.CoronarySinus,
  },
  {
    id: 'pulmonary-veins',
    label: 'Venas pulmonares',
    color: 0xa03842,
    opacity: 1,
    has: (s) => s === Structure.PulmonaryVein,
  },
];

export const AORTA_GROUP: GroupSpec = {
  id: 'thoracic-aorta',
  label: 'Arco y aorta torácica',
  color: 0xb03036,
  opacity: 1,
  // the arch and its branches by name: `s >= AscendingAorta` took in every structure added after them (the right atrial
  // appendage, the esophagus, the mediastinum)
  has: (s) =>
    s === Structure.DescendingAorta ||
    (s >= Structure.AscendingAorta && s <= Structure.LeftSubclavian),
};

const DEFAULT_BOUNDS = {
  min: [-8, -7, -7] as [number, number, number],
  max: [7, 8, 12] as [number, number, number],
};

export function buildHeartMeshes(
  heart: HeartModel,
  pose: HeartPose,
  opts: HeartMeshOptions = {},
): MeshGroup[] {
  const ivcFloorY = heart.frame.origin.y - 4.8;
  const groups = extractMeshes(heart, pose, opts, MESH_GROUPS, ivcFloorY);
  const out: MeshGroup[] = [...groups];

  if (opts.includeAorta !== false) {
    // Explicit caller bounds remain authoritative (tests and offline crops).
    const min = { ...pose.aorta.tube.min },
      max = { ...pose.aorta.tube.max };
    const aortaFloorY = heart.frame.origin.y - 5.4;
    if (opts.thorax) {
      const z = DESC_AORTA_Z + opts.thorax.columnShiftCm;
      const bottom = Math.max(diaphragmY(opts.thorax, DESC_AORTA_X, z), aortaFloorY);
      for (const x of [DESC_AORTA_X - 1.5, DESC_AORTA_X + 1.5])
        for (const zz of [z - 1.5, z + 1.5])
          for (const y of [bottom, opts.thorax.descAortaTopY]) {
            const p = torsoToHeart(heart.frame, { x, y, z: zz });
            min.x = Math.min(min.x, p.x);
            min.y = Math.min(min.y, p.y);
            min.z = Math.min(min.z, p.z);
            max.x = Math.max(max.x, p.x);
            max.y = Math.max(max.y, p.y);
            max.z = Math.max(max.z, p.z);
          }
    }
    const aortaStep = Math.max(opts.stepCm ?? 0.35, 0.42);
    const pad = aortaStep * 2;
    const vascular = extractMeshes(
      heart,
      pose,
      {
        ...opts,
        stepCm: aortaStep,
        bounds: opts.bounds ?? {
          min: [min.x - pad, min.y - pad, min.z - pad],
          max: [max.x + pad, max.y + pad, max.z + pad],
        },
      },
      [AORTA_GROUP],
      aortaFloorY,
    );
    out.push(...vascular);
  }

  if (opts.includeCoronaries !== false) {
    const myoTargets = groups.filter((g) => g.id === 'lv-myocardium');
    out.push(...buildCoronaryMeshes(heart, pose, myoTargets));
  }
  return out;
}

function extractMeshes(
  heart: HeartModel,
  pose: HeartPose,
  opts: HeartMeshOptions,
  specs: GroupSpec[],
  inferiorTrimTorsoY?: number,
): MeshGroup[] {
  const step = opts.stepCm ?? 0.35;
  const b = opts.bounds ?? DEFAULT_BOUNDS;
  const nx = Math.ceil((b.max[0] - b.min[0]) / step) + 1;
  const ny = Math.ceil((b.max[1] - b.min[1]) / step) + 1;
  const nz = Math.ceil((b.max[2] - b.min[2]) / step) + 1;
  // one classification pass for every group: the structure id per grid point
  const ids = new Uint8Array(nx * ny * nz);
  const tissues = new Uint8Array(nx * ny * nz);
  const segs = new Uint8Array(nx * ny * nz);
  const s = makeSample();
  const bc = heart.boundCenter;
  const br2 = (heart.boundRadius + 0.6) * (heart.boundRadius + 0.6);
  const wantDescAorta = Boolean(opts.thorax) && specs.some((g) => g.id === 'thoracic-aorta');
  const descScale = opts.thorax ? descAortaScale(opts.thorax, pose.state.aorticPressure) : 1;

  for (let k = 0; k < nz; k++) {
    const z = b.min[2] + k * step;
    const dz = z - bc.z;
    const dz2 = dz * dz;
    for (let j = 0; j < ny; j++) {
      const y = b.min[1] + j * step;
      const dy = y - bc.y;
      const dyz2 = dy * dy + dz2;
      for (let i = 0; i < nx; i++) {
        const x = b.min[0] + i * step;
        const dx = x - bc.x;
        let hit = false;
        if (dx * dx + dyz2 <= br2) {
          hit = classifyHeart(heart, pose, x, y, z, s);
        }
        if (hit && s.structure === Structure.Ivc && inferiorTrimTorsoY !== undefined) {
          const p = heartToTorso(heart.frame, { x, y, z });
          if (p.y < inferiorTrimTorsoY) hit = false;
        }
        if (!hit) {
          if (!wantDescAorta || !opts.thorax) continue;
          const p = heartToTorso(heart.frame, { x, y, z });
          if (inferiorTrimTorsoY !== undefined && p.y < inferiorTrimTorsoY) continue;
          if (
            !classifyThorax(opts.thorax, p.x, p.y, p.z, s, s.sdf, descScale) ||
            s.structure !== Structure.DescendingAorta
          )
            continue;
        }
        const o = (k * ny + j) * nx + i;
        ids[o] = s.structure;
        tissues[o] = s.tissue;
        segs[o] = s.segment;
      }
    }
  }
  return specs.map((g) =>
    surfaceNet(
      g,
      ids,
      tissues,
      nx,
      ny,
      nz,
      b.min,
      step,
      g.smoothing ?? opts.smoothing ?? 2,
      g.id === 'lv-myocardium' ? segs : null,
    ),
  );
}

/** Whether a neighboring structure constitutes a meaningful anatomical sulcus against the ventricles. */
function isSulcusNeighbour(groupId: string, other: Structure): boolean {
  if (
    other === Structure.None ||
    other === Structure.Pericardium ||
    other === Structure.EpicardialFat ||
    other === Structure.PericardialFat
  )
    return false;
  if (groupId === 'lv-myocardium') {
    return (
      other === Structure.LaWall ||
      other === Structure.LaCavity ||
      other === Structure.LaAppendage ||
      other === Structure.RaWall ||
      other === Structure.RaCavity ||
      other === Structure.RaAppendage ||
      other === Structure.AorticRoot ||
      other === Structure.PulmonaryArtery
    );
  }
  return false;
}

const isExteriorFillable = (s: Structure): boolean =>
  s === Structure.None ||
  s === Structure.EpicardialFat ||
  s === Structure.Pericardium ||
  s === Structure.PericardialFat;

/**
 * Builds the seamless unified biventricular occupancy mask:
 * 1. Thickens the thin RV free wall 2 cells inward into RV cavity/RVOT so the epicardium stays watertight.
 * 2. Applies constrained exterior sulcus filleting exclusively in the concave notch where RV meets LV
 *    (never invading blood cavities, atria, or great vessels) so the ventricles fuse into one continuous
 *    sculptural epicardial mass with zero detached "cape/shell" step.
 */
function buildUnifiedVentricleMask(
  ids: Uint8Array,
  nx: number,
  ny: number,
  nz: number,
): Uint8Array {
  const n = ids.length;
  const lvZone = new Uint8Array(n);
  const rvZone = new Uint8Array(n);
  for (let o = 0; o < n; o++) {
    const s = ids[o] as Structure;
    if (isLvWall(s) || s === Structure.PapillaryMuscle) lvZone[o] = 1;
    else if (
      s === Structure.RvWall ||
      s === Structure.ModeratorBand ||
      s === Structure.RvPapillary
    )
      rvZone[o] = 1;
  }

  // 1. Watertight RV wall: grow RV 2 cells inward into its own cavity/RVOT only (epicardium stays exact)
  const grow = new Uint8Array(n);
  for (let pass = 0; pass < 2; pass++) {
    grow.fill(0);
    for (let k = 1; k < nz - 1; k++)
      for (let j = 1; j < ny - 1; j++)
        for (let i = 1; i < nx - 1; i++) {
          const o = (k * ny + j) * nx + i;
          if (rvZone[o]) continue;
          const s = ids[o] as Structure;
          if (s !== Structure.RvCavity && s !== Structure.Rvot) continue;
          if (
            rvZone[o - 1] ||
            rvZone[o + 1] ||
            rvZone[o - nx] ||
            rvZone[o + nx] ||
            rvZone[o - nx * ny] ||
            rvZone[o + nx * ny]
          )
            grow[o] = 1;
        }
    for (let o = 0; o < n; o++) if (grow[o]) rvZone[o] = 1;
  }

  const inside = new Uint8Array(n);
  for (let o = 0; o < n; o++) inside[o] = lvZone[o] || rvZone[o] ? 1 : 0;

  // 2. Constrained LV↔RV sulcus fillet (3 passes): smoothly bridges only the exterior V-notch / overhang
  //    between LV and RV (anterior & inferior interventricular grooves and apical RV insertion).
  const strideZ = nx * ny;
  const fill = new Uint8Array(n);
  for (let pass = 0; pass < 3; pass++) {
    fill.fill(0);
    for (let k = 1; k < nz - 1; k++)
      for (let j = 1; j < ny - 1; j++)
        for (let i = 1; i < nx - 1; i++) {
          const o = k * strideZ + j * nx + i;
          if (inside[o] || !isExteriorFillable(ids[o] as Structure)) continue;
          let nLv = 0;
          let nRv = 0;
          for (let dk = -1; dk <= 1; dk++) {
            const ok = o + dk * strideZ;
            for (let dj = -1; dj <= 1; dj++) {
              const oj = ok + dj * nx;
              for (let di = -1; di <= 1; di++) {
                if (di === 0 && dj === 0 && dk === 0) continue;
                const nb = oj + di;
                if (lvZone[nb]) nLv++;
                if (rvZone[nb]) nRv++;
              }
            }
          }
          if (nLv >= 1 && nRv >= 1 && nLv + nRv >= 4) fill[o] = 1;
        }
    for (let o = 0; o < n; o++) {
      if (fill[o]) {
        inside[o] = 1;
        lvZone[o] = 1;
        rvZone[o] = 1;
      }
    }
  }

  // 3. One pass of concave-only exterior morphological closing (>= 14 of 26 neighbors inside)
  //    smooths any residual step along the apical RV-to-LV junction without inflating convex walls.
  fill.fill(0);
  for (let k = 1; k < nz - 1; k++)
    for (let j = 1; j < ny - 1; j++)
      for (let i = 1; i < nx - 1; i++) {
        const o = k * strideZ + j * nx + i;
        if (inside[o] || !isExteriorFillable(ids[o] as Structure)) continue;
        let cnt = 0;
        for (let dk = -1; dk <= 1; dk++) {
          const ok = o + dk * strideZ;
          for (let dj = -1; dj <= 1; dj++) {
            const oj = ok + dj * nx;
            for (let di = -1; di <= 1; di++) {
              if (di === 0 && dj === 0 && dk === 0) continue;
              if (inside[oj + di]) cnt++;
            }
          }
        }
        if (cnt >= 14) fill[o] = 1;
      }
  for (let o = 0; o < n; o++) if (fill[o]) inside[o] = 1;

  return inside;
}

function surfaceNet(
  g: GroupSpec,
  ids: Uint8Array,
  tissues: Uint8Array,
  nx: number,
  ny: number,
  nz: number,
  min: [number, number, number],
  step: number,
  smoothing: number,
  segs: Uint8Array | null = null,
): MeshGroup {
  const inside =
    g.id === 'lv-myocardium'
      ? buildUnifiedVentricleMask(ids, nx, ny, nz)
      : (() => {
          const mask = new Uint8Array(nx * ny * nz);
          for (let o = 0; o < ids.length; o++)
            mask[o] = g.has(ids[o] as Structure, tissues[o] as Tissue) ? 1 : 0;
          if (g.thicken) {
            const grow = new Uint8Array(mask.length);
            for (let pass = 0; pass < g.thicken.cells; pass++) {
              grow.fill(0);
              for (let k = 1; k < nz - 1; k++)
                for (let j = 1; j < ny - 1; j++)
                  for (let i = 1; i < nx - 1; i++) {
                    const o = (k * ny + j) * nx + i;
                    if (mask[o] || !g.thicken.into(ids[o] as Structure)) continue;
                    if (
                      mask[o - 1] ||
                      mask[o + 1] ||
                      mask[o - nx] ||
                      mask[o + nx] ||
                      mask[o - nx * ny] ||
                      mask[o + nx * ny]
                    )
                      grow[o] = 1;
                  }
              for (let o = 0; o < mask.length; o++) if (grow[o]) mask[o] = 1;
            }
          }
          return mask;
        })();
  const at = (i: number, j: number, k: number): number => inside[(k * ny + j) * nx + i] ?? 0;
  const cellVertex = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1);
  const pos: number[] = [];
  const vertexSeg: number[] = [];
  const trackGroove = g.id === 'lv-myocardium' || g.id === 'rv-myocardium';
  const vertexGroove: number[] = [];
  const votes = new Uint8Array(21);
  const cellIndex = (i: number, j: number, k: number): number => (k * (ny - 1) + j) * (nx - 1) + i;
  for (let k = 0; k < nz - 1; k++)
    for (let j = 0; j < ny - 1; j++)
      for (let i = 0; i < nx - 1; i++) {
        let count = 0;
        for (let dk = 0; dk < 2; dk++)
          for (let dj = 0; dj < 2; dj++)
            for (let di = 0; di < 2; di++) if (at(i + di, j + dj, k + dk)) count++;
        if (count === 0 || count === 8) continue; // fully outside or fully inside: no surface here
        let cx = 0,
          cy = 0,
          cz = 0,
          crossings = 0;
        for (let dk = 0; dk < 2; dk++)
          for (let dj = 0; dj < 2; dj++)
            for (let di = 0; di < 2; di++) {
              const here = at(i + di, j + dj, k + dk);
              if (di === 0 && here !== at(i + 1, j + dj, k + dk)) {
                cx += 0.5;
                cy += dj;
                cz += dk;
                crossings++;
              }
              if (dj === 0 && here !== at(i + di, j + 1, k + dk)) {
                cx += di;
                cy += 0.5;
                cz += dk;
                crossings++;
              }
              if (dk === 0 && here !== at(i + di, j + dj, k + 1)) {
                cx += di;
                cy += dj;
                cz += 0.5;
                crossings++;
              }
            }
        if (!crossings) continue;
        if (segs) {
          votes.fill(0);
          for (let dk = 0; dk < 2; dk++)
            for (let dj = 0; dj < 2; dj++)
              for (let di = 0; di < 2; di++)
                if (at(i + di, j + dj, k + dk)) {
                  const c = segs[(k + dk) * ny * nx + (j + dj) * nx + i + di]!;
                  if (c > 0 && c < 21) votes[c]!++;
                }
          let best = 0;
          for (let c = 1; c < 21; c++) if (votes[c]! > votes[best]!) best = c;
          vertexSeg.push(best);
        }
        if (trackGroove) {
          let sulcus = 0;
          const k0 = Math.max(0, k - 1),
            k1 = Math.min(nz - 1, k + 2);
          const j0 = Math.max(0, j - 1),
            j1 = Math.min(ny - 1, j + 2);
          const i0 = Math.max(0, i - 1),
            i1 = Math.min(nx - 1, i + 2);
          outer: for (let kk = k0; kk <= k1; kk++)
            for (let jj = j0; jj <= j1; jj++)
              for (let ii = i0; ii <= i1; ii++) {
                const st = ids[(kk * ny + jj) * nx + ii] as Structure;
                if (isSulcusNeighbour(g.id, st)) {
                  sulcus = 1;
                  break outer;
                }
              }
          vertexGroove.push(sulcus);
        }
        cellVertex[cellIndex(i, j, k)] = pos.length / 3;
        pos.push(
          min[0] + (i + cx / crossings) * step,
          min[1] + (j + cy / crossings) * step,
          min[2] + (k + cz / crossings) * step,
        );
      }
  // quads across every edge whose endpoints differ, built from the four cells sharing that edge
  const idx: number[] = [];
  const vertexAt = (i: number, j: number, k: number): number => {
    if (i < 0 || j < 0 || k < 0 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1) return -1;
    const v = cellVertex[cellIndex(i, j, k)];
    return v === undefined ? -1 : v;
  };
  const quad = (a: number, b2: number, c: number, d: number, flip: boolean): void => {
    if (a < 0 || b2 < 0 || c < 0 || d < 0) return;
    if (flip) idx.push(a, c, b2, a, d, c);
    else idx.push(a, b2, c, a, c, d);
  };
  for (let k = 1; k < nz - 1; k++)
    for (let j = 1; j < ny - 1; j++)
      for (let i = 1; i < nx - 1; i++) {
        const here = at(i, j, k);
        if (here !== at(i + 1, j, k))
          quad(
            vertexAt(i, j - 1, k - 1),
            vertexAt(i, j, k - 1),
            vertexAt(i, j, k),
            vertexAt(i, j - 1, k),
            here === 0,
          );
        if (here !== at(i, j + 1, k))
          quad(
            vertexAt(i - 1, j, k - 1),
            vertexAt(i, j, k - 1),
            vertexAt(i, j, k),
            vertexAt(i - 1, j, k),
            here !== 0,
          );
        if (here !== at(i, j, k + 1))
          quad(
            vertexAt(i - 1, j - 1, k),
            vertexAt(i, j - 1, k),
            vertexAt(i, j, k),
            vertexAt(i - 1, j, k),
            here === 0,
          );
      }
  const positions = new Float32Array(pos);
  smoothTaubin(positions, idx, smoothing);
  const normals = vertexNormals(positions, idx);
  smoothNormals(normals, idx, 2);
  const groove = trackGroove ? smoothScalar(Float32Array.from(vertexGroove), idx, 3) : undefined;
  const ao = computeAmbientOcclusion(positions, normals, idx, step, groove);
  return {
    id: g.id,
    label: g.label,
    color: g.color,
    opacity: g.opacity,
    positions,
    normals,
    indices: new Uint32Array(idx),
    ao,
    ...(groove ? { groove } : {}),
    ...(segs ? { segments: Uint8Array.from(vertexSeg) } : {}),
  };
}

/**
 * Area-weighted normals from the triangles themselves.
 *
 * These used to come from the gradient of the occupancy field, which is well defined only where the
 * structure is several cells thick. Across a wall one or two cells wide the smoothed field rises and falls
 * again, so the central difference subtracts two samples on the *same* side of the wall and collapses.
 * Measured on the normal case at 0.28 cm, the median gradient magnitude was 0.71 on the LV wall (median 6
 * cells thick) but 0.50 on the RV wall (2 cells) and 0.12 on the valves (1 cell), with a quarter of the RV
 * vertices under 0.32 — those normals are ridge noise, and they shaded as a lattice of dark patches on
 * exactly the thin groups. Sampling the gradient at the vertex's true position rather than the nearest grid
 * point does not help; the ridge is real, not a rounding artefact.
 *
 * Geometric normals do not care how thick the wall is. They do inherit the winding of each quad, which is
 * why the field was used at first, but the winding has since been measured against the gradient wherever
 * the gradient is trustworthy: 100% agreement over 16 790 vertices, in all six groups that have any. The
 * handful of vertices whose triangles cancel exactly are folds of a one-cell sheet, where no normal is
 * meaningful anyway.
 */
function vertexNormals(positions: Float32Array, indices: number[]): Float32Array {
  const out = new Float32Array(positions.length);
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t]! * 3,
      b = indices[t + 1]! * 3,
      c = indices[t + 2]! * 3;
    const ux = positions[b]! - positions[a]!,
      uy = positions[b + 1]! - positions[a + 1]!,
      uz = positions[b + 2]! - positions[a + 2]!;
    const vx = positions[c]! - positions[a]!,
      vy = positions[c + 1]! - positions[a + 1]!,
      vz = positions[c + 2]! - positions[a + 2]!;
    // the cross product is twice the triangle area, so accumulating it unnormalised weights by area
    const nx = uy * vz - uz * vy,
      ny = uz * vx - ux * vz,
      nz = ux * vy - uy * vx;
    for (const o of [a, b, c]) {
      out[o] = out[o]! + nx;
      out[o + 1] = out[o + 1]! + ny;
      out[o + 2] = out[o + 2]! + nz;
    }
  }
  for (let v = 0; v < out.length; v += 3) {
    const len = Math.hypot(out[v]!, out[v + 1]!, out[v + 2]!);
    if (len < 1e-12) {
      out[v + 2] = 1; // a fold with no surface: any unit vector will do, and there are ~19 of them
      continue;
    }
    out[v] = out[v]! / len;
    out[v + 1] = out[v + 1]! / len;
    out[v + 2] = out[v + 2]! / len;
  }
  return out;
}

/**
 * Two passes of neighbourhood averaging over the normals.
 *
 * Surface nets gives one vertex per straddling cell, so where a wall is as thin as the cell that vertex
 * serves *both* of its surfaces: the triangles of the outer and the inner face meet there with opposing
 * normals, which cancel, and the result shades as a black facet. Measured on the normal case, the fraction
 * of vertices whose triangle normals cancel is 2.3% of the RV wall and 11.4% of the valves at 0.28 cm, and
 * a finer grid does not remove it — 1.2% at 0.22 cm, 0.9% at 0.15 cm for four times the extraction cost,
 * with the valves flat at 11-15% because a 0.18 cm leaflet is one cell thick at every step. The folds are
 * few and isolated, though, so each one is surrounded by vertices whose normals are sound; averaging over
 * the neighbourhood pulls it back to the local surface, which is what the eye expects on a thin shell.
 * The cost is slightly softer creases, which a navigator can afford.
 */
function smoothNormals(normals: Float32Array, indices: number[], passes: number): void {
  if (!passes || !indices.length) return;
  const n = normals.length / 3;
  const sum = new Float32Array(n * 3);
  for (let p = 0; p < passes; p++) {
    sum.set(normals);
    for (let t = 0; t < indices.length; t += 3)
      for (let e = 0; e < 3; e++) {
        const a = indices[t + e]!,
          b = indices[t + ((e + 1) % 3)]!;
        for (const [from, to] of [
          [a, b],
          [b, a],
        ] as const) {
          const fi = from * 3,
            ti = to * 3;
          sum[fi] = sum[fi]! + normals[ti]!;
          sum[fi + 1] = sum[fi + 1]! + normals[ti + 1]!;
          sum[fi + 2] = sum[fi + 2]! + normals[ti + 2]!;
        }
      }
    for (let v = 0; v < n; v++) {
      const len = Math.hypot(sum[v * 3]!, sum[v * 3 + 1]!, sum[v * 3 + 2]!);
      if (len < 1e-12) continue; // a whole neighbourhood that cancels: keep what it had
      for (let c = 0; c < 3; c++) normals[v * 3 + c] = sum[v * 3 + c]! / len;
    }
  }
}

/**
 * Volume-preserving Taubin smoothing (alternating shrink λ = 0.52 and inflate μ = -0.54 passes).
 * Removes voxel stair-stepping without ballooning/shrinking chambers or erasing anatomical sulci.
 */
function smoothTaubin(positions: Float32Array, indices: number[], passes: number): void {
  if (!passes || !indices.length) return;
  const n = positions.length / 3;
  const sum = new Float32Array(n * 3);
  const deg = new Uint16Array(n);
  const stepPass = (weight: number) => {
    sum.fill(0);
    deg.fill(0);
    for (let t = 0; t < indices.length; t += 3)
      for (let e = 0; e < 3; e++) {
        const a = indices[t + e]!,
          b = indices[t + ((e + 1) % 3)]!;
        for (const [from, to] of [
          [a, b],
          [b, a],
        ] as const) {
          const fi = from * 3,
            ti = to * 3;
          sum[fi] = (sum[fi] ?? 0) + (positions[ti] ?? 0);
          sum[fi + 1] = (sum[fi + 1] ?? 0) + (positions[ti + 1] ?? 0);
          sum[fi + 2] = (sum[fi + 2] ?? 0) + (positions[ti + 2] ?? 0);
          deg[from] = (deg[from] ?? 0) + 1;
        }
      }
    for (let v = 0; v < n; v++) {
      const d = deg[v]!;
      if (!d) continue;
      for (let c = 0; c < 3; c++) {
        const cur = positions[v * 3 + c]!;
        const avg = sum[v * 3 + c]! / d;
        positions[v * 3 + c] = cur + weight * (avg - cur);
      }
    }
  };
  for (let p = 0; p < passes; p++) {
    stepPass(0.52);
    stepPass(-0.535);
  }
}

/** Smooths a per-vertex scalar attribute over triangle adjacency. */
function smoothScalar(values: Float32Array, indices: number[], passes: number): Float32Array {
  if (!passes || !indices.length) return values;
  const n = values.length;
  const sum = new Float32Array(n);
  const deg = new Uint16Array(n);
  for (let p = 0; p < passes; p++) {
    sum.fill(0);
    deg.fill(0);
    for (let t = 0; t < indices.length; t += 3)
      for (let e = 0; e < 3; e++) {
        const a = indices[t + e]!,
          b = indices[t + ((e + 1) % 3)]!;
        sum[a] = sum[a]! + values[b]!;
        deg[a] = deg[a]! + 1;
        sum[b] = sum[b]! + values[a]!;
        deg[b] = deg[b]! + 1;
      }
    for (let v = 0; v < n; v++) {
      const d = deg[v]!;
      if (d) values[v] = values[v]! * 0.45 + (sum[v]! / d) * 0.55;
    }
  }
  return values;
}

/**
 * Per-vertex ambient occlusion from local surface concavity (Laplacian displacement projected onto the
 * outward normal) combined with inter-structure sulcus contact (`groove`). Darkens crevices, sulci, and
 * vessel roots while giving convex domes a crisp sculptural lift.
 */
function computeAmbientOcclusion(
  positions: Float32Array,
  normals: Float32Array,
  indices: number[],
  step: number,
  groove?: Float32Array,
): Float32Array {
  const n = positions.length / 3;
  const ao = new Float32Array(n).fill(1);
  if (!indices.length) return ao;
  const sum = new Float32Array(n * 3);
  const deg = new Uint16Array(n);
  for (let t = 0; t < indices.length; t += 3)
    for (let e = 0; e < 3; e++) {
      const a = indices[t + e]!,
        b = indices[t + ((e + 1) % 3)]!;
      for (const [from, to] of [
        [a, b],
        [b, a],
      ] as const) {
        const fi = from * 3,
          ti = to * 3;
        sum[fi] = sum[fi]! + positions[ti]!;
        sum[fi + 1] = sum[fi + 1]! + positions[ti + 1]!;
        sum[fi + 2] = sum[fi + 2]! + positions[ti + 2]!;
        deg[from] = deg[from]! + 1;
      }
    }
  const invStep = 1 / Math.max(0.1, step);
  const concavity = new Float32Array(n);
  for (let v = 0; v < n; v++) {
    const d = deg[v]!;
    if (!d) continue;
    const o = v * 3;
    const lx = sum[o]! / d - positions[o]!;
    const ly = sum[o + 1]! / d - positions[o + 1]!;
    const lz = sum[o + 2]! / d - positions[o + 2]!;
    // Positive projection onto outward normal = concave valley/sulcus
    concavity[v] = (lx * normals[o]! + ly * normals[o + 1]! + lz * normals[o + 2]!) * invStep;
  }
  smoothScalar(concavity, indices, 2);
  for (let v = 0; v < n; v++) {
    const c = concavity[v]!;
    let factor = c > 0 ? 1.02 - c * 0.55 : 1.02 + Math.min(0.12, -c * 0.14);
    if (groove) factor *= 1 - 0.24 * groove[v]!;
    ao[v] = Math.max(0.54, Math.min(1.06, factor));
  }
  return ao;
}
