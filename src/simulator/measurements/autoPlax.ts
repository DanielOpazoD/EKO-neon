import { Structure } from '@/simulator/anatomy/tissue';
import type { CineBuffer, CineFrame } from '@/simulator/core/cineBuffer';
import type { FrameAcquisition, PhaseMarks } from '@/simulator/core/protocol';
import { isStripModality } from '@/simulator/renderer/modality';
import {
  computeSectorMapping,
  pixelToPolar,
  type SectorMapping,
} from '@/simulator/renderer/scanConvert';
import type { AcquisitionSettings, PolarFrameSpec } from '@/simulator/renderer/types';
import { sectorCmToPixel } from './autoVolume';

/**
 * Pure 2D polar Parasternal Long-Axis (PLAX) automatic linear measurements (ASE/EACVI 2015/2019):
 *  - End-Diastole (ED): IVSd (`ivsd`), LVEDD (`lv-edd`), LVPWd (`lvpwd`), Aortic Root Sinuses (`aortic-root`)
 *  - Mid-Systole: LVOT diameter (`lvot-diameter`, ~0.5 cm proximal to aortic annulus, inner-to-inner)
 *  - End-Systole (ES): LVESD (`lv-esd`), LA anteroposterior diameter (`la-ap`)
 *
 * Reads strictly from the acquired 2D/Color polar `structure` and `display` frames in the cine buffer —
 * zero ground-truth access. All returned calipers are editable on their frozen frames.
 */

export type PlaxMeasurementId =
  | 'ivsd'
  | 'lv-edd'
  | 'lvpwd'
  | 'lv-esd'
  | 'lvot-diameter'
  | 'aortic-root'
  | 'la-ap';

export type PlaxPhaseBucket = 'ed' | 'mid-systole' | 'es';

export interface AutoPlaxCaliper {
  measurementId: PlaxMeasurementId;
  phaseBucket: PlaxPhaseBucket;
  valueCm: number;
  /** Caliper endpoints in display pixel coordinates for `sector`. */
  aPx: { x: number; y: number };
  bPx: { x: number; y: number };
  offset: number;
  frameId: number;
  phase: number;
  timeS: number;
  beatIndex: number;
  rrS: number;
  sector: SectorMapping;
  viewId: string | null;
  viewScore: number | null;
  gainScore: number | null;
  acquisition: FrameAcquisition;
  /** Sampled structures along the caliper and 3 px beyond each end for exact technique evaluation. */
  segmentStructures: number[];
  segmentEndsOutside: [number, number];
}

export interface AutoPlaxSuccess {
  ok: true;
  calipers: AutoPlaxCaliper[];
  edFrameId: number;
  midSysFrameId: number | null;
  esFrameId: number | null;
  edOffset: number;
}

export interface AutoPlaxFailure {
  ok: false;
  reason: 'no-frame' | 'wrong-modality' | 'incomplete-cycle' | 'not-plax' | 'no-calipers';
  message: string;
}

export type AutoPlaxResult = AutoPlaxSuccess | AutoPlaxFailure;

const SEPTAL_WALL_SET = new Set<number>([Structure.LvWallSeptal, Structure.LvWallAnterior]);
const POSTERIOR_WALL_SET = new Set<number>([Structure.LvWallInferior, Structure.LvWallLateral]);
const LV_CAVITY_RUN_SET = new Set<number>([
  Structure.LvCavity,
  Structure.Chordae,
  Structure.PapillaryMuscle,
]);
const MITRAL_SET = new Set<number>([Structure.MitralAnterior, Structure.MitralPosterior]);
const LVOT_LUMEN_SET = new Set<number>([
  Structure.Lvot,
  Structure.AorticRoot,
  Structure.LvCavity,
  Structure.AorticValve,
]);
const BLOOD_SET = new Set<number>([
  Structure.LvCavity,
  Structure.RvCavity,
  Structure.LaCavity,
  Structure.RaCavity,
  Structure.Lvot,
  Structure.AorticRoot,
  Structure.AscendingAorta,
  Structure.Rvot,
  Structure.LaAppendage,
  Structure.RaAppendage,
  Structure.PulmonaryVein,
  Structure.CoronarySinus,
  Structure.DescendingAorta,
  Structure.PulmonaryArtery,
]);

const MAX_PHASE_DIST = 0.1;

interface PolarGrid {
  structure: Uint8Array;
  display?: Uint8ClampedArray;
  lines: number;
  samples: number;
  sectorRad: number;
  depthCm: number;
  dr: number;
  dth: number;
  sinTable: Float64Array;
  cosTable: Float64Array;
}

interface CmPoint {
  x: number;
  y: number;
}

interface CmSegment {
  a: CmPoint;
  b: CmPoint;
  lengthCm: number;
}

function makeGrid(
  structure: Uint8Array,
  spec: Pick<PolarFrameSpec, 'lines' | 'samples' | 'sectorRad' | 'depthCm'>,
  display?: Uint8ClampedArray,
): PolarGrid {
  const { lines, samples, sectorRad, depthCm } = spec;
  const dr = depthCm / samples;
  const dth = sectorRad / lines;
  const sinTable = new Float64Array(lines);
  const cosTable = new Float64Array(lines);
  for (let li = 0; li < lines; li++) {
    const th = -0.5 * sectorRad + (li + 0.5) * dth;
    sinTable[li] = Math.sin(th);
    cosTable[li] = Math.cos(th);
  }
  return { structure, display, lines, samples, sectorRad, depthCm, dr, dth, sinTable, cosTable };
}

function polarIndexAtCm(g: PolarGrid, x: number, y: number): number {
  const r = Math.hypot(x, y);
  if (r <= 0 || r >= g.depthCm) return -1;
  const th = Math.atan2(x, y);
  const half = g.sectorRad / 2;
  if (th < -half || th > half) return -1;
  const li = Math.min(g.lines - 1, Math.max(0, Math.floor(((th + half) / g.sectorRad) * g.lines)));
  const si = Math.min(g.samples - 1, Math.max(0, Math.floor((r / g.depthCm) * g.samples)));
  return li * g.samples + si;
}

/** Sample structure id at un-inverted sector cm coordinates `(x, y)`. */
function sampleAtCm(g: PolarGrid, x: number, y: number): number {
  const idx = polarIndexAtCm(g, x, y);
  return idx >= 0 ? (g.structure[idx] ?? 0) : 0;
}

/** Sample B-mode grayscale brightness (0..255) at un-inverted sector cm coordinates `(x, y)`. */
function sampleDisplayAtCm(g: PolarGrid, x: number, y: number): number {
  if (!g.display) return 0;
  const idx = polarIndexAtCm(g, x, y);
  return idx >= 0 ? (g.display[idx] ?? 0) : 0;
}

function collectPoints(
  g: PolarGrid,
  match: (s: number) => boolean,
): { pts: CmPoint[]; cx: number; cy: number; areaCm2: number } {
  const pts: CmPoint[] = [];
  let sx = 0,
    sy = 0,
    sa = 0;
  for (let li = 0; li < g.lines; li++) {
    const sn = g.sinTable[li]!;
    const cs = g.cosTable[li]!;
    const base = li * g.samples;
    for (let si = 0; si < g.samples; si++) {
      if (!match(g.structure[base + si]!)) continue;
      const r = (si + 0.5) * g.dr;
      const x = r * sn;
      const y = r * cs;
      const a = r * g.dr * g.dth;
      pts.push({ x, y });
      sx += x * a;
      sy += y * a;
      sa += a;
    }
  }
  return {
    pts,
    cx: sa > 0 ? sx / sa : 0,
    cy: sa > 0 ? sy / sa : 0,
    areaCm2: sa,
  };
}

/** Unit vector perpendicular to `(ux, uy)` oriented shallow → deep (`py > 0`). */
function perpDown(ux: number, uy: number): { px: number; py: number } {
  let px = -uy;
  let py = ux;
  if (py < 0) {
    px = -px;
    py = -py;
  }
  return { px, py };
}

/** Find all contiguous runs along a ray `origin + s * dir` matching `pred` for `s ∈ [sMin, sMax]`. */
function rayRuns(
  g: PolarGrid,
  ox: number,
  oy: number,
  dx: number,
  dy: number,
  sMin: number,
  sMax: number,
  stepCm: number,
  pred: (st: number, disp: number) => boolean,
): { s0: number; s1: number }[] {
  const runs: { s0: number; s1: number }[] = [];
  let inRun = false;
  let runStart = 0;
  for (let s = sMin; s <= sMax; s += stepCm) {
    const x = ox + dx * s;
    const y = oy + dy * s;
    const idx = polarIndexAtCm(g, x, y);
    const st = idx >= 0 ? (g.structure[idx] ?? 0) : 0;
    const disp = idx >= 0 && g.display ? (g.display[idx] ?? 0) : 0;
    if (pred(st, disp)) {
      if (!inRun) {
        inRun = true;
        runStart = s;
      }
    } else if (inRun) {
      runs.push({ s0: runStart, s1: s - stepCm });
      inRun = false;
    }
  }
  if (inRun) runs.push({ s0: runStart, s1: sMax });
  return runs;
}

function segFromRay(
  ox: number,
  oy: number,
  dx: number,
  dy: number,
  s0: number,
  s1: number,
): CmSegment {
  const a = { x: ox + dx * s0, y: oy + dy * s0 };
  const b = { x: ox + dx * s1, y: oy + dy * s1 };
  return { a, b, lengthCm: Math.abs(s1 - s0) };
}

function isTissue(st: number): boolean {
  return st !== Structure.None && !BLOOD_SET.has(st);
}

/** Estimate the B-mode lumen/vessel-wall brightness cutoff from interior lumen blood cells. */
function estimateWallBrightnessThreshold(
  g: PolarGrid,
  ox: number,
  oy: number,
  px: number,
  py: number,
  s0: number,
  s1: number,
): number {
  if (!g.display) return 256;
  const midVals: number[] = [];
  const lo = s0 + (s1 - s0) * 0.25;
  const hi = s0 + (s1 - s0) * 0.75;
  for (let s = lo; s <= hi; s += 0.05) {
    midVals.push(sampleDisplayAtCm(g, ox + px * s, oy + py * s));
  }
  if (!midVals.length) return 90;
  midVals.sort((a, b) => a - b);
  const medianBlood = midVals[Math.floor(midVals.length * 0.5)]!;
  return Math.max(55, Math.min(135, medianBlood + 38));
}

/**
 * Trim bright fibrous/vessel-wall echoes (`Structure.AorticRoot` with `Tissue.VesselWall` shares
 * the `AorticRoot` structure id with blood lumen) from both ends of a blood-lumen run so inner-edge
 * to inner-edge diameters exclude the ~2.5 mm aortic walls cleanly.
 */
function trimBrightVesselWall(
  g: PolarGrid,
  ox: number,
  oy: number,
  px: number,
  py: number,
  s0: number,
  s1: number,
  maxTrimCm = 0.34,
  stepCm = 0.02,
): { s0: number; s1: number } {
  if (!g.display || s1 - s0 < 1.0) return { s0, s1 };
  const thr = estimateWallBrightnessThreshold(g, ox, oy, px, py, s0, s1);
  let a = s0;
  const aLimit = Math.min(s1 - 0.8, s0 + maxTrimCm);
  while (a < aLimit && sampleDisplayAtCm(g, ox + px * a, oy + py * a) >= thr) {
    a += stepCm;
  }
  let b = s1;
  const bLimit = Math.max(a + 0.8, s1 - maxTrimCm);
  while (b > bLimit && sampleDisplayAtCm(g, ox + px * b, oy + py * b) >= thr) {
    b -= stepCm;
  }
  return { s0: a, s1: b };
}

/**
 * Verify PLAX-like long-axis geometry strictly from the 2D polar structure map (zero view-ID cheating):
 * LV cavity, LA cavity, and aortic root/LVOT must all be visible with LV lateral to the LA/Ao base
 * and the aortic root anterior (shallower) to the left atrium. Uses anatomical chamber centroids for
 * axes so spherical dilated ventricles (HFrEF) and stubby aortic roots never suffer PCA 90° flips.
 */
export function detectPlaxFrameAxes(
  structure: Uint8Array,
  spec: Pick<PolarFrameSpec, 'lines' | 'samples' | 'sectorRad' | 'depthCm'>,
  display?: Uint8ClampedArray,
): {
  grid: PolarGrid;
  lv: { cx: number; cy: number; areaCm2: number; pts: CmPoint[] };
  la: { cx: number; cy: number; areaCm2: number; pts: CmPoint[] };
  ao: { cx: number; cy: number; areaCm2: number; pts: CmPoint[] };
  ascAo: { cx: number; cy: number; areaCm2: number; pts: CmPoint[] };
  lvot: { cx: number; cy: number; areaCm2: number; pts: CmPoint[] };
  mv: { cx: number; cy: number; areaCm2: number; pts: CmPoint[] };
  uLv: { ux: number; uy: number };
  pLv: { px: number; py: number };
  uAo: { ux: number; uy: number };
  pAo: { px: number; py: number };
} | null {
  const g = makeGrid(structure, spec, display);
  const lv = collectPoints(g, (s) => s === Structure.LvCavity);
  const la = collectPoints(g, (s) => s === Structure.LaCavity);
  const ao = collectPoints(g, (s) => s === Structure.AorticRoot);
  const ascAo = collectPoints(g, (s) => s === Structure.AscendingAorta);
  const lvot = collectPoints(g, (s) => s === Structure.Lvot);
  const mv = collectPoints(g, (s) => MITRAL_SET.has(s));

  if (lv.areaCm2 < 4.0 || la.areaCm2 < 1.5 || ao.areaCm2 + lvot.areaCm2 < 0.8) return null;

  const aoRefX = ao.areaCm2 > 0.3 ? ao.cx : lvot.cx;
  const aoRefY = ao.areaCm2 > 0.3 ? ao.cy : lvot.cy;
  // In PLAX, aortic root sits anteriorly (shallower) over the posterior left atrium
  if (la.cy - aoRefY < 0.5) return null;

  // LV and LA must be separated predominantly laterally (rejects apical A4C/A2C/A5C/A3C where LV is shallow of LA)
  const lvToLaX = la.cx - lv.cx;
  const lvToLaY = la.cy - lv.cy;
  const lvToLaDist = Math.hypot(lvToLaX, lvToLaY);
  if (lvToLaDist < 1.8 || Math.abs(lvToLaX) / lvToLaDist < 0.5) return null;

  // Aortic root and LA must lie on the same basal side of LV
  const lvToAoX = aoRefX - lv.cx;
  if (lvToAoX * lvToLaX <= 0) return null;

  // Anatomical LV long axis: from LV cavity centroid toward the mitral/LVOT basal midpoint
  const baseAnchorX =
    lvot.areaCm2 > 0.2 ? 0.5 * (la.cx + lvot.cx) : 0.55 * la.cx + 0.45 * aoRefX;
  const baseAnchorY =
    lvot.areaCm2 > 0.2 ? 0.5 * (la.cy + lvot.cy) : 0.55 * la.cy + 0.45 * aoRefY;
  const lvDx = baseAnchorX - lv.cx;
  const lvDy = baseAnchorY - lv.cy;
  const lvNorm = Math.hypot(lvDx, lvDy);
  if (lvNorm < 1.0) return null;
  const uLv = { ux: lvDx / lvNorm, uy: lvDy / lvNorm };
  const pLv = perpDown(uLv.ux, uLv.uy);

  // Anatomical aortic root / LVOT centerline: LVOT -> AorticRoot (-> AscendingAorta), immune to stubby-root PCA flips
  let aoDx = 0;
  let aoDy = 0;
  if (ao.areaCm2 > 0.25 && lvot.areaCm2 > 0.15) {
    const dx = ao.cx - lvot.cx;
    const dy = ao.cy - lvot.cy;
    const n = Math.hypot(dx, dy);
    if (n > 0.25) {
      aoDx += (dx / n) * 2.0;
      aoDy += (dy / n) * 2.0;
    }
  }
  if (ascAo.areaCm2 > 0.25 && ao.areaCm2 > 0.25) {
    const dx = ascAo.cx - ao.cx;
    const dy = ascAo.cy - ao.cy;
    const n = Math.hypot(dx, dy);
    if (n > 0.25) {
      aoDx += dx / n;
      aoDy += dy / n;
    }
  }
  const aoNorm = Math.hypot(aoDx, aoDy);
  const uAo = aoNorm > 0.2 ? { ux: aoDx / aoNorm, uy: aoDy / aoNorm } : uLv;
  const pAo = perpDown(uAo.ux, uAo.uy);

  return { grid: g, lv, la, ao, ascAo, lvot, mv, uLv, pLv, uAo, pAo };
}

export interface PlaxLvTransectResult {
  ivsd?: CmSegment;
  cavity?: CmSegment;
  lvpwd?: CmSegment;
  /** Anatomical ED transect origin & axes so ES LVESD measures on the exact same line (ASE 2015). */
  transect?: {
    ox: number;
    oy: number;
    uLv: { ux: number; uy: number };
    pLv: { px: number; py: number };
  };
}

/**
 * Measure ASE PLAX LV linear dimensions (IVSd, LVEDD, LVPWd on ED; or LVESD on ES) along a
 * perpendicular transect at the level of the mitral valve leaflet tips.
 */
export function extractPlaxLvTransect(
  structure: Uint8Array,
  spec: Pick<PolarFrameSpec, 'lines' | 'samples' | 'sectorRad' | 'depthCm'>,
  includeWalls: boolean,
  display?: Uint8ClampedArray,
): PlaxLvTransectResult | null {
  const axes = detectPlaxFrameAxes(structure, spec, display);
  if (!axes) return null;
  const { grid: g, lv, mv, uLv, pLv } = axes;

  // Target ASE level along LV long axis uLv:
  // - ED (`includeWalls = true`): mitral leaflets are open into LV, so tips sit near the basal-mid widest cavity.
  // - ES (`includeWalls = false`): mitral leaflets are closed at the annulus, so measure ~0.65 cm apical to them.
  let tTarget = 0;
  if (mv.pts.length >= 4) {
    let minMvT = Infinity;
    for (const p of mv.pts) {
      const t = (p.x - lv.cx) * uLv.ux + (p.y - lv.cy) * uLv.uy;
      if (t < minMvT) minMvT = t;
    }
    tTarget = includeWalls ? minMvT + 0.15 : minMvT - 0.65;
  }

  let lvTMin = Infinity,
    lvTMax = -Infinity;
  for (const p of lv.pts) {
    const t = (p.x - lv.cx) * uLv.ux + (p.y - lv.cy) * uLv.uy;
    if (t < lvTMin) lvTMin = t;
    if (t > lvTMax) lvTMax = t;
  }
  const tLo = Math.max(lvTMin + 0.6, tTarget - 1.1);
  const tHi = Math.min(lvTMax - 0.2, tTarget + 1.1);
  if (tHi < tLo) return null;

  let bestScore = -Infinity;
  let bestResult: PlaxLvTransectResult | null = null;

  const stepCm = 0.025;
  for (let t = tLo; t <= tHi; t += 0.1) {
    const ox = lv.cx + uLv.ux * t;
    const oy = lv.cy + uLv.uy * t;

    const cavRuns = rayRuns(
      g,
      ox,
      oy,
      pLv.px,
      pLv.py,
      -5.5,
      5.5,
      stepCm,
      (s) => LV_CAVITY_RUN_SET.has(s),
    );
    if (!cavRuns.length) continue;
    let cav = cavRuns[0]!;
    for (const r of cavRuns) {
      if (r.s1 - r.s0 > cav.s1 - cav.s0) cav = r;
    }
    const cavLen = cav.s1 - cav.s0;
    if (cavLen < 1.5) continue;

    // Verify myocardium right outside both endocardial ends (excludes mitral leaflet / open LVOT)
    const antEdge = sampleAtCm(g, ox + pLv.px * (cav.s0 - 0.12), oy + pLv.py * (cav.s0 - 0.12));
    const postEdge = sampleAtCm(g, ox + pLv.px * (cav.s1 + 0.12), oy + pLv.py * (cav.s1 + 0.12));
    if (!SEPTAL_WALL_SET.has(antEdge) || !POSTERIOR_WALL_SET.has(postEdge)) continue;

    // Trace septal wall directly anterior to cav.s0
    let ivsS0 = cav.s0;
    for (let s = cav.s0 - stepCm; s >= cav.s0 - 3.6; s -= stepCm) {
      const st = sampleAtCm(g, ox + pLv.px * s, oy + pLv.py * s);
      if (SEPTAL_WALL_SET.has(st)) ivsS0 = s;
      else break;
    }
    const ivsLen = cav.s0 - ivsS0;

    // Trace posterior wall directly posterior to cav.s1 (stops cleanly before pericardium/fat)
    let pwS1 = cav.s1;
    for (let s = cav.s1 + stepCm; s <= cav.s1 + 3.2; s += stepCm) {
      const st = sampleAtCm(g, ox + pLv.px * s, oy + pLv.py * s);
      if (POSTERIOR_WALL_SET.has(st)) pwS1 = s;
      else break;
    }
    const pwLen = pwS1 - cav.s1;

    if (includeWalls && (ivsLen < 0.4 || pwLen < 0.4)) continue;

    // Prefer proximity to ASE mitral-tip station + full clean wall borders + widest representative cavity span
    const distFromTarget = Math.abs(t - tTarget);
    const score =
      -distFromTarget * 1.1 +
      Math.min(cavLen, 8.0) * 1.15 +
      (ivsLen >= 0.45 && pwLen >= 0.45 ? 1.2 : 0);
    if (score > bestScore) {
      bestScore = score;
      const ivsInner = cav.s0 - stepCm;
      const pwInner = cav.s1 + stepCm;
      bestResult = {
        cavity: segFromRay(ox, oy, pLv.px, pLv.py, cav.s0, cav.s1),
        ...(includeWalls && ivsLen >= 0.4
          ? { ivsd: segFromRay(ox, oy, pLv.px, pLv.py, ivsS0, ivsInner) }
          : {}),
        ...(includeWalls && pwLen >= 0.4
          ? { lvpwd: segFromRay(ox, oy, pLv.px, pLv.py, pwInner, pwS1) }
          : {}),
        transect: { ox, oy, uLv, pLv },
      };
    }
  }

  return bestResult;
}

/**
 * Measure LVESD on the end-systolic frame at the EXACT same PLAX transect line and perpendicularity
 * established for LVEDD/IVSd/LVPWd at end-diastole (ASE/EACVI 2015), falling back by ≤ 0.35 cm
 * only if basal longitudinal descent shifts the mitral leaflet onto the exact ED line.
 */
function extractPlaxLvEsAtTransect(
  structure: Uint8Array,
  spec: Pick<PolarFrameSpec, 'lines' | 'samples' | 'sectorRad' | 'depthCm'>,
  transect: NonNullable<PlaxLvTransectResult['transect']>,
  display?: Uint8ClampedArray,
): CmSegment | null {
  const g = makeGrid(structure, spec, display);
  const { ox, oy, uLv, pLv } = transect;
  const stepCm = 0.025;
  // Prefer dt = 0 (exact ED line), then step gently apically along -uLv to clear systolic
  // basal annular descent (MAPSE) while staying strictly 0° parallel to the ED transect.
  const shifts = [
    0, -0.1, -0.25, -0.4, -0.55, -0.7, -0.85, -1.0, -1.15, -1.3, -1.45, -1.6, -1.75, -1.9, -2.05,
    0.15,
  ];
  for (const dt of shifts) {
    const cx = ox + uLv.ux * dt;
    const cy = oy + uLv.uy * dt;
    const cavRuns = rayRuns(
      g,
      cx,
      cy,
      pLv.px,
      pLv.py,
      -5.5,
      5.5,
      stepCm,
      (s) => LV_CAVITY_RUN_SET.has(s),
    );
    if (!cavRuns.length) continue;
    let cav = cavRuns[0]!;
    for (const r of cavRuns) if (r.s1 - r.s0 > cav.s1 - cav.s0) cav = r;
    const cavLen = cav.s1 - cav.s0;
    if (cavLen < 1.2) continue;
    const antEdge = sampleAtCm(g, cx + pLv.px * (cav.s0 - 0.12), cy + pLv.py * (cav.s0 - 0.12));
    const postEdge = sampleAtCm(g, cx + pLv.px * (cav.s1 + 0.12), cy + pLv.py * (cav.s1 + 0.12));
    if (!SEPTAL_WALL_SET.has(antEdge) || !POSTERIOR_WALL_SET.has(postEdge)) continue;
    return segFromRay(cx, cy, pLv.px, pLv.py, cav.s0, cav.s1);
  }
  return null;
}

/** Measure ASE LVOT diameter (~0.5 cm proximal to aortic annulus, inner-edge to inner-edge) on mid-systole PLAX. */
export function extractPlaxLvotDiameter(
  structure: Uint8Array,
  spec: Pick<PolarFrameSpec, 'lines' | 'samples' | 'sectorRad' | 'depthCm'>,
  display?: Uint8ClampedArray,
): CmSegment | null {
  const axes = detectPlaxFrameAxes(structure, spec, display);
  if (!axes || axes.lvot.pts.length < 6) return null;
  const { grid: g, lvot, ao, uAo, pAo } = axes;

  // Locate aortic valve / annular plane along uAo (projection relative to lvot centroid)
  const av = collectPoints(g, (s) => s === Structure.AorticValve);
  let tAnn = 0.5;
  if (av.pts.length >= 2) {
    let minAvT = Infinity;
    for (const p of av.pts) {
      const t = (p.x - lvot.cx) * uAo.ux + (p.y - lvot.cy) * uAo.uy;
      if (t < minAvT) minAvT = t;
    }
    tAnn = minAvT;
  } else if (ao.pts.length >= 4) {
    let minAoT = Infinity;
    for (const p of ao.pts) {
      const t = (p.x - lvot.cx) * uAo.ux + (p.y - lvot.cy) * uAo.uy;
      if (t < minAoT) minAoT = t;
    }
    tAnn = minAoT;
  }

  // ASE 2015/2019 station: ~0.4–0.5 cm proximal (toward LV) from aortic leaflet insertion.
  // Allow scanning slightly closer to the annulus (+0.48 cm) for severely dilated LVs (HFrEF)
  // where the basal septum meets the outflow tract closer to the aortic valve hinge.
  const tTarget = tAnn - 0.42;
  const stepCm = 0.02;
  let best: { seg: CmSegment; score: number } | null = null;

  for (let dt = -0.3; dt <= 0.48; dt += 0.06) {
    const t = tTarget + dt;
    const ox = lvot.cx + uAo.ux * t;
    const oy = lvot.cy + uAo.uy * t;

    // Scan LVOT lumen (including LvCavity at proximal station in dilated ventricles, stopping at septum/mitral/aortic wall)
    const runs = rayRuns(
      g,
      ox,
      oy,
      pAo.px,
      pAo.py,
      -2.6,
      2.6,
      stepCm,
      (s) => LVOT_LUMEN_SET.has(s),
    );
    if (!runs.length) continue;

    // Choose the run closest to s = 0 (LVOT centerline)
    let run = runs[0]!;
    let bestCenterDist = Math.abs(0.5 * (run.s0 + run.s1));
    for (const r of runs) {
      const d = Math.abs(0.5 * (r.s0 + r.s1));
      if (d < bestCenterDist) {
        bestCenterDist = d;
        run = r;
      }
    }

    // Trim bright vessel wall echoes (`Structure.AorticRoot` VesselWall at the posterior aortomitral continuity)
    const trimmed = trimBrightVesselWall(g, ox, oy, pAo.px, pAo.py, run.s0, run.s1, 0.35, stepCm);
    const len = trimmed.s1 - trimmed.s0;
    if (len < 1.3 || len > 3.2) continue;

    // Count Lvot/AorticRoot structure share along trimmed lumen (required >= 50% by technique.ts)
    let targetCount = 0;
    const samplesN = 16;
    for (let k = 0; k < samplesN; k++) {
      const s = trimmed.s0 + ((k + 0.5) / samplesN) * len;
      const st = sampleAtCm(g, ox + pAo.px * s, oy + pAo.py * s);
      if (st === Structure.Lvot || st === Structure.AorticRoot) targetCount++;
    }
    const targetShare = targetCount / samplesN;
    if (targetShare < 0.5) continue;

    // Anterior border must be ventricular septum (or vessel wall), posterior border aortic wall/AML/curtain
    const antSt = sampleAtCm(g, ox + pAo.px * (trimmed.s0 - 0.14), oy + pAo.py * (trimmed.s0 - 0.14));
    const postSt = sampleAtCm(g, ox + pAo.px * (trimmed.s1 + 0.12), oy + pAo.py * (trimmed.s1 + 0.12));
    const postDisp = sampleDisplayAtCm(
      g,
      ox + pAo.px * (trimmed.s1 + 0.06),
      oy + pAo.py * (trimmed.s1 + 0.06),
    );
    const antOk = isTissue(antSt) || antSt === Structure.AorticRoot;
    const postOk =
      isTissue(postSt) ||
      postSt === Structure.AorticRoot ||
      (g.display !== undefined && postDisp >= 60);
    if (!antOk || !postOk) continue;

    const score = targetShare * 2.5 - Math.abs(dt) * 3.0 - bestCenterDist * 0.5;
    if (!best || score > best.score) {
      best = { seg: segFromRay(ox, oy, pAo.px, pAo.py, trimmed.s0, trimmed.s1), score };
    }
  }
  return best?.seg ?? null;
}

/** Measure Aortic Root Sinuses of Valsalva maximum perpendicular diameter on end-diastolic PLAX. */
export function extractPlaxAorticRoot(
  structure: Uint8Array,
  spec: Pick<PolarFrameSpec, 'lines' | 'samples' | 'sectorRad' | 'depthCm'>,
  display?: Uint8ClampedArray,
): CmSegment | null {
  const axes = detectPlaxFrameAxes(structure, spec, display);
  if (!axes || axes.ao.pts.length < 10) return null;
  const { grid: g, ao, uAo, pAo } = axes;

  let tMin = Infinity,
    tMax = -Infinity;
  for (const p of ao.pts) {
    const t = (p.x - ao.cx) * uAo.ux + (p.y - ao.cy) * uAo.uy;
    if (t < tMin) tMin = t;
    if (t > tMax) tMax = t;
  }
  if (tMax - tMin < 0.5) return null;

  const stepCm = 0.02;
  let best: CmSegment | null = null;
  // Scan through the sinus bulge between aortic annulus and sinotubular junction
  const lo = tMin + 0.25;
  const hi = tMax - 0.15;
  for (let t = lo; t <= hi; t += 0.07) {
    const ox = ao.cx + uAo.ux * t;
    const oy = ao.cy + uAo.uy * t;
    const runs = rayRuns(
      g,
      ox,
      oy,
      pAo.px,
      pAo.py,
      -3.4,
      3.4,
      stepCm,
      (s) => s === Structure.AorticRoot || s === Structure.AorticValve,
    );
    if (!runs.length) continue;
    let run = runs[0]!;
    for (const r of runs) if (r.s1 - r.s0 > run.s1 - run.s0) run = r;

    // Trim bright anterior & posterior aortic root walls (`Tissue.VesselWall` shares `Structure.AorticRoot`)
    const trimmed = trimBrightVesselWall(g, ox, oy, pAo.px, pAo.py, run.s0, run.s1, 0.36, stepCm);
    const len = trimmed.s1 - trimmed.s0;
    if (len < 1.8 || len > 5.8) continue;

    if (!best || len > best.lengthCm) {
      best = segFromRay(ox, oy, pAo.px, pAo.py, trimmed.s0, trimmed.s1);
    }
  }
  return best;
}

/** Measure LA Anteroposterior diameter along the heart AP axis (perpendicular to LV-LA long axis) behind the aortic root on ES PLAX. */
export function extractPlaxLaAp(
  structure: Uint8Array,
  spec: Pick<PolarFrameSpec, 'lines' | 'samples' | 'sectorRad' | 'depthCm'>,
  display?: Uint8ClampedArray,
): CmSegment | null {
  const axes = detectPlaxFrameAxes(structure, spec, display);
  if (!axes || axes.la.pts.length < 16) return null;
  const { grid: g, la, ao, uLv, pLv } = axes;

  let tMin = Infinity,
    tMax = -Infinity;
  for (const p of la.pts) {
    const t = (p.x - la.cx) * uLv.ux + (p.y - la.cy) * uLv.uy;
    if (t < tMin) tMin = t;
    if (t > tMax) tMax = t;
  }
  const aoT = ao.areaCm2 > 0 ? (ao.cx - la.cx) * uLv.ux + (ao.cy - la.cy) * uLv.uy : 0;
  const stepCm = 0.025;
  let best: { seg: CmSegment; score: number } | null = null;

  for (let t = tMin + 0.25; t <= tMax - 0.25; t += 0.08) {
    const ox = la.cx + uLv.ux * t;
    const oy = la.cy + uLv.uy * t;
    const runs = rayRuns(
      g,
      ox,
      oy,
      pLv.px,
      pLv.py,
      -4.5,
      4.5,
      stepCm,
      (s) => s === Structure.LaCavity,
    );
    if (!runs.length) continue;
    let run = runs[0]!;
    for (const r of runs) if (r.s1 - r.s0 > run.s1 - run.s0) run = r;
    const len = run.s1 - run.s0;
    if (len < 1.8 || len > 7.5) continue;

    // Posterior edge must sit on LA wall / mediastinum / pericardium
    const e1 = sampleAtCm(g, ox + pLv.px * (run.s1 + 0.14), oy + pLv.py * (run.s1 + 0.14));
    if (!isTissue(e1)) continue;
    // Prefer rays whose anterior edge lies behind the aortic root wall (classic ASE LA AP station)
    const e0 = sampleAtCm(g, ox + pLv.px * (run.s0 - 0.16), oy + pLv.py * (run.s0 - 0.16));
    const behindAo = e0 === Structure.AorticRoot || e0 === Structure.LaWall;

    const score = len * 0.45 + (behindAo ? 1.5 : 0) - Math.abs(t - aoT) * 0.9;
    if (!best || score > best.score) {
      best = { seg: segFromRay(ox, oy, pLv.px, pLv.py, run.s0, run.s1), score };
    }
  }
  return best?.seg ?? null;
}

function sameAcquisitionPose(a: FrameAcquisition, b: FrameAcquisition): boolean {
  if (a.modality !== '2d' && a.modality !== 'color') return false;
  if (a.modality !== b.modality) return false;
  if (Math.abs(a.settings.depthCm - b.settings.depthCm) > 0.25) return false;
  if (Math.abs(a.settings.sectorDeg - b.settings.sectorDeg) > 1) return false;
  const pa = a.probe;
  const pb = b.probe;
  return (
    Math.abs(pa.u - pb.u) <= 0.15 &&
    Math.abs(pa.v - pb.v) <= 0.15 &&
    Math.abs(pa.rotationDeg - pb.rotationDeg) <= 3 &&
    Math.abs(pa.tiltDeg - pb.tiltDeg) <= 2 &&
    Math.abs(pa.rockDeg - pb.rockDeg) <= 2
  );
}

function sampleSegmentOnFrame(
  cf: CineFrame,
  sector: SectorMapping,
  aPx: { x: number; y: number },
  bPx: { x: number; y: number },
): { along: number[]; ends: [number, number] } {
  const p = cf.spec;
  const atPx = (x: number, y: number): number => {
    const { rCm, thetaRad } = pixelToPolar(sector, x, y);
    if (rCm < 0 || rCm > p.depthCm || Math.abs(thetaRad) > p.sectorRad / 2) return 0;
    const li = Math.min(
      p.lines - 1,
      Math.max(0, Math.floor(((thetaRad + p.sectorRad / 2) / p.sectorRad) * p.lines)),
    );
    const si = Math.min(p.samples - 1, Math.max(0, Math.floor((rCm / p.depthCm) * p.samples)));
    return cf.structure[li * p.samples + si] ?? 0;
  };
  const n = 24;
  const along: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = (i + 0.5) / n;
    along.push(atPx(aPx.x + (bPx.x - aPx.x) * t, aPx.y + (bPx.y - aPx.y) * t));
  }
  const dx = bPx.x - aPx.x;
  const dy = bPx.y - aPx.y;
  const len = Math.hypot(dx, dy) || 1;
  const ux = (dx / len) * 3;
  const uy = (dy / len) * 3;
  return {
    along,
    ends: [atPx(aPx.x - ux, aPx.y - uy), atPx(bPx.x + ux, bPx.y + uy)],
  };
}

/**
 * Automatic ASE/EACVI PLAX linear dimensions across ED (IVSd, LVEDD, LVPWd, Aortic Root),
 * mid-systole (LVOT diameter), and ES (LVESD, LA AP) over the acquired 2D/Color cine buffer.
 * Pure 2D image/structure analysis — zero ground-truth access.
 */
export function computeAutoPlaxMeasurements(
  cine: CineBuffer,
  displayed: CineFrame | undefined,
  phaseMarks: PhaseMarks,
  displaySize: { width: number; height: number },
  settings: Pick<AcquisitionSettings, 'invertLR' | 'zoom'>,
): AutoPlaxResult {
  if (!displayed || cine.length === 0) {
    return {
      ok: false,
      reason: 'no-frame',
      message: 'Aún no hay cuadros adquiridos en el cine.',
    };
  }
  const mod = displayed.acquisition.modality;
  if (mod !== '2d' && mod !== 'color') {
    return {
      ok: false,
      reason: 'wrong-modality',
      message: 'Cambia a modo 2D (o Color) en vista PLAX para medir las dimensiones automáticas.',
    };
  }

  const refAcq = displayed.acquisition;
  const matcher = (f: CineFrame) => sameAcquisitionPose(f.acquisition, refAcq);
  const edPick = cine.findNearestPhaseOffsetMatching(0, matcher);
  const midSysPhase =
    phaseMarks.ejectionStart + 0.42 * (phaseMarks.ejectionEnd - phaseMarks.ejectionStart);
  const midSysPick = cine.findNearestPhaseOffsetMatching(midSysPhase, matcher);
  const esPick = cine.findNearestPhaseOffsetMatching(phaseMarks.endSystole, matcher);

  if (!edPick || !esPick || edPick.distance > MAX_PHASE_DIST || esPick.distance > MAX_PHASE_DIST) {
    return {
      ok: false,
      reason: 'incomplete-cycle',
      message:
        'Deja correr al menos un latido completo en PLAX para capturar TD, mesosístole y TS.',
    };
  }

  const edFrame = cine.select(edPick.offset);
  const midSysFrame =
    midSysPick && midSysPick.distance <= MAX_PHASE_DIST ? cine.select(midSysPick.offset) : undefined;
  const esFrame = cine.select(esPick.offset);
  if (!edFrame || !esFrame) {
    return {
      ok: false,
      reason: 'no-frame',
      message: 'No se pudieron recuperar los cuadros de TD/TS del cine.',
    };
  }

  // Geometry guard strictly from rendered 2D pixels (no viewId cheating)
  if (!detectPlaxFrameAxes(edFrame.structure, edFrame.spec, edFrame.display)) {
    return {
      ok: false,
      reason: 'not-plax',
      message:
        'No se detecta un plano paraesternal eje largo (PLAX) completo con VI, AI y raíz aórtica.',
    };
  }

  const W = Math.max(64, displaySize.width);
  const H = Math.max(64, displaySize.height);
  const sectorH = isStripModality(mod) ? Math.round(H * 0.42) : H;

  const makeCaliper = (
    measurementId: PlaxMeasurementId,
    phaseBucket: PlaxPhaseBucket,
    cf: CineFrame,
    offset: number,
    seg: CmSegment,
  ): AutoPlaxCaliper => {
    const sector = computeSectorMapping(cf.spec, W, sectorH, settings.invertLR, settings.zoom);
    const aPx = sectorCmToPixel(sector, seg.a);
    const bPx = sectorCmToPixel(sector, seg.b);
    const sampled = sampleSegmentOnFrame(cf, sector, aPx, bPx);
    return {
      measurementId,
      phaseBucket,
      valueCm: seg.lengthCm,
      aPx,
      bPx,
      offset,
      frameId: cf.frameId,
      phase: cf.phase,
      timeS: cf.timeS,
      beatIndex: cf.beatIndex,
      rrS: cf.rrS,
      sector,
      viewId: cf.analysis?.bestViewId ?? null,
      viewScore: cf.analysis?.score ?? null,
      gainScore: cf.analysis ? Math.round(cf.analysis.components.gain * 100) : null,
      acquisition: cf.acquisition,
      segmentStructures: sampled.along,
      segmentEndsOutside: sampled.ends,
    };
  };

  const calipers: AutoPlaxCaliper[] = [];

  // 1. End-diastole: IVSd, LVEDD, LVPWd, Aortic Root
  const edLv = extractPlaxLvTransect(edFrame.structure, edFrame.spec, true, edFrame.display);
  if (edLv?.ivsd) calipers.push(makeCaliper('ivsd', 'ed', edFrame, edPick.offset, edLv.ivsd));
  if (edLv?.cavity) calipers.push(makeCaliper('lv-edd', 'ed', edFrame, edPick.offset, edLv.cavity));
  if (edLv?.lvpwd) calipers.push(makeCaliper('lvpwd', 'ed', edFrame, edPick.offset, edLv.lvpwd));

  const aoRoot = extractPlaxAorticRoot(edFrame.structure, edFrame.spec, edFrame.display);
  if (aoRoot) calipers.push(makeCaliper('aortic-root', 'ed', edFrame, edPick.offset, aoRoot));

  // 2. Mid-systole: LVOT diameter (0.5 cm proximal to open AV annulus)
  let midSysFrameId: number | null = null;
  if (midSysFrame && midSysPick) {
    const lvotSeg = extractPlaxLvotDiameter(
      midSysFrame.structure,
      midSysFrame.spec,
      midSysFrame.display,
    );
    if (lvotSeg) {
      calipers.push(
        makeCaliper('lvot-diameter', 'mid-systole', midSysFrame, midSysPick.offset, lvotSeg),
      );
      midSysFrameId = midSysFrame.frameId;
    }
  }

  // 3. End-systole: LVESD on the exact same ED transect line & perpendicularity + LA AP diameter
  let esFrameId: number | null = null;
  const esCavity =
    (edLv?.transect
      ? extractPlaxLvEsAtTransect(esFrame.structure, esFrame.spec, edLv.transect, esFrame.display)
      : null) ?? extractPlaxLvTransect(esFrame.structure, esFrame.spec, false, esFrame.display)?.cavity;
  if (esCavity) {
    calipers.push(makeCaliper('lv-esd', 'es', esFrame, esPick.offset, esCavity));
    esFrameId = esFrame.frameId;
  }
  const laAp = extractPlaxLaAp(esFrame.structure, esFrame.spec, esFrame.display);
  if (laAp) {
    calipers.push(makeCaliper('la-ap', 'es', esFrame, esPick.offset, laAp));
    esFrameId = esFrame.frameId;
  }

  if (!calipers.length) {
    return {
      ok: false,
      reason: 'no-calipers',
      message: 'No se pudieron trazar dimensiones válidas sobre este plano PLAX; optimiza la vista.',
    };
  }

  return {
    ok: true,
    calipers,
    edFrameId: edFrame.frameId,
    midSysFrameId,
    esFrameId,
    edOffset: edPick.offset,
  };
}
