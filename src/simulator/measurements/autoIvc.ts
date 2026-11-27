import { Structure } from '@/simulator/anatomy/tissue';
import type { CineBuffer, CineFrame } from '@/simulator/core/cineBuffer';
import type { FrameAcquisition } from '@/simulator/core/protocol';
import { isStripModality } from '@/simulator/renderer/modality';
import {
  computeSectorMapping,
  pixelToPolar,
  type SectorMapping,
} from '@/simulator/renderer/scanConvert';
import type { AcquisitionSettings, PolarFrameSpec } from '@/simulator/renderer/types';
import { sectorCmToPixel } from './autoVolume';

/**
 * Pure 2D polar Inferior Vena Cava (IVC) automatic lumen measurement (ASE/EACVI Right Heart 2025):
 *  - Locates the subcostal longitudinal IVC (`Structure.Ivc`), orients along its long axis toward
 *    the right atrium (`Structure.RaCavity`), and measures inner-edge to inner-edge perpendicular
 *    diameter 1–2 cm proximal to the IVC–RA junction (avoiding hepatic vein confluence).
 *  - Trims bright B-mode `VesselWall` echoes from `display` so the caliper measures the true
 *    blood lumen in both expiration (`ivc-diameter`) and sniff inspiration (`ivc-diameter-inspiration`).
 *  - Zero ground-truth or view-ID access.
 */

export type IvcMeasurementId = 'ivc-diameter' | 'ivc-diameter-inspiration';

export interface AutoIvcSuccess {
  ok: true;
  measurementId: IvcMeasurementId;
  respiration: FrameAcquisition['patient']['respiration'];
  valueCm: number;
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
  segmentStructures: number[];
  segmentEndsOutside: [number, number];
}

export interface AutoIvcFailure {
  ok: false;
  reason: 'no-frame' | 'wrong-modality' | 'not-ivc';
  message: string;
}

export type AutoIvcResult = AutoIvcSuccess | AutoIvcFailure;

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
  Structure.Svc,
  Structure.Ivc,
  Structure.HepaticVein,
]);

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

function sampleAtCm(g: PolarGrid, x: number, y: number): number {
  const idx = polarIndexAtCm(g, x, y);
  return idx >= 0 ? (g.structure[idx] ?? 0) : 0;
}

function sampleDisplayAtCm(g: PolarGrid, x: number, y: number): number {
  if (!g.display) return 0;
  const idx = polarIndexAtCm(g, x, y);
  return idx >= 0 ? (g.display[idx] ?? 0) : 0;
}

function isNonBloodTissue(st: number): boolean {
  return st !== Structure.None && !BLOOD_SET.has(st);
}

/**
 * Trim bright `Tissue.VesselWall` pixels from both ends of a `Structure.Ivc` run using the
 * rendered B-mode grayscale image (zero ground-truth access).
 */
function trimBrightIvcWalls(
  g: PolarGrid,
  ox: number,
  oy: number,
  px: number,
  py: number,
  s0: number,
  s1: number,
  stepCm = 0.02,
): { s0: number; s1: number } {
  if (!g.display || s1 - s0 < 0.45) return { s0, s1 };
  const midVals: number[] = [];
  const lo = s0 + (s1 - s0) * 0.28;
  const hi = s0 + (s1 - s0) * 0.72;
  for (let s = lo; s <= hi; s += 0.04) {
    midVals.push(sampleDisplayAtCm(g, ox + px * s, oy + py * s));
  }
  if (!midVals.length) return { s0, s1 };
  midVals.sort((a, b) => a - b);
  const medianBlood = midVals[Math.floor(midVals.length * 0.5)]!;
  const thr = Math.max(48, Math.min(130, medianBlood + 34));
  const maxTrim = Math.min(0.28, (s1 - s0) * 0.35);

  let a = s0;
  const aLim = s0 + maxTrim;
  while (a < aLim && sampleDisplayAtCm(g, ox + px * a, oy + py * a) >= thr) a += stepCm;
  let b = s1;
  const bLim = s1 - maxTrim;
  while (b > bLim && sampleDisplayAtCm(g, ox + px * b, oy + py * b) >= thr) b -= stepCm;
  return b - a >= 0.25 ? { s0: a, s1: b } : { s0, s1 };
}

/**
 * Extract the perpendicular ASE subcostal IVC lumen diameter (1–2 cm proximal to the RA junction,
 * avoiding the hepatic vein confluence) directly from a 2D polar slice.
 */
export function extractIvcDiameter(
  structure: Uint8Array,
  spec: Pick<PolarFrameSpec, 'lines' | 'samples' | 'sectorRad' | 'depthCm'>,
  display?: Uint8ClampedArray,
): { a: { x: number; y: number }; b: { x: number; y: number }; valueCm: number } | null {
  const { lines, samples } = spec;
  const total = lines * samples;
  if (structure.length < total || lines < 8 || samples < 16) return null;

  const g = makeGrid(structure, spec, display);

  // 1. Largest 4-connected component of Structure.Ivc (allows collapsed inspiratory IVC ≥ 0.45 cm²)
  const visited = new Uint8Array(total);
  const queue = new Int32Array(total);
  let bestCells: number[] = [];
  let liverCount = 0;
  let raSx = 0,
    raSy = 0,
    raArea = 0;

  for (let idx = 0; idx < total; idx++) {
    const st = structure[idx]!;
    if (st === Structure.Liver) {
      liverCount++;
    } else if (st === Structure.RaCavity) {
      const li = Math.floor(idx / samples);
      const si = idx - li * samples;
      const r = (si + 0.5) * g.dr;
      const a = r * g.dr * g.dth;
      raSx += r * g.sinTable[li]! * a;
      raSy += r * g.cosTable[li]! * a;
      raArea += a;
    }
    if (visited[idx] || st !== Structure.Ivc) continue;
    let qH = 0;
    let qT = 0;
    queue[qT++] = idx;
    visited[idx] = 1;
    const comp: number[] = [];
    while (qH < qT) {
      const cur = queue[qH++]!;
      comp.push(cur);
      const li = Math.floor(cur / samples);
      const si = cur - li * samples;
      if (li > 0 && !visited[cur - samples] && structure[cur - samples] === Structure.Ivc) {
        visited[cur - samples] = 1;
        queue[qT++] = cur - samples;
      }
      if (li + 1 < lines && !visited[cur + samples] && structure[cur + samples] === Structure.Ivc) {
        visited[cur + samples] = 1;
        queue[qT++] = cur + samples;
      }
      if (si > 0 && !visited[cur - 1] && structure[cur - 1] === Structure.Ivc) {
        visited[cur - 1] = 1;
        queue[qT++] = cur - 1;
      }
      if (si + 1 < samples && !visited[cur + 1] && structure[cur + 1] === Structure.Ivc) {
        visited[cur + 1] = 1;
        queue[qT++] = cur + 1;
      }
    }
    if (comp.length > bestCells.length) bestCells = comp;
  }

  // Must see a meaningful longitudinal IVC segment alongside liver or right atrium
  if (bestCells.length < 30 || (liverCount < 60 && raArea < 0.8)) return null;

  let sx = 0,
    sy = 0,
    sa = 0;
  const pts: { x: number; y: number; li: number; si: number }[] = [];
  for (let k = 0; k < bestCells.length; k++) {
    const idx = bestCells[k]!;
    const li = Math.floor(idx / samples);
    const si = idx - li * samples;
    const r = (si + 0.5) * g.dr;
    const x = r * g.sinTable[li]!;
    const y = r * g.cosTable[li]!;
    const a = r * g.dr * g.dth;
    pts.push({ x, y, li, si });
    sx += x * a;
    sy += y * a;
    sa += a;
  }
  if (sa < 0.45) return null;
  const cx = sx / sa;
  const cy = sy / sa;

  // 2. Principal axis of IVC oriented toward RA cavity (or rightward if RA is out of frame)
  let sxx = 0,
    sxy = 0,
    syy = 0;
  for (const p of pts) {
    const dx = p.x - cx;
    const dy = p.y - cy;
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }
  const ang = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  let ux = Math.cos(ang);
  let uy = Math.sin(ang);
  const dirX = raArea > 0.5 ? raSx / raArea - cx : 1;
  const dirY = raArea > 0.5 ? raSy / raArea - cy : 0;
  if (ux * dirX + uy * dirY < 0) {
    ux = -ux;
    uy = -uy;
  }
  // Perpendicular axis oriented shallow -> deep (py > 0)
  let px = -uy;
  let py = ux;
  if (py < 0) {
    px = -px;
    py = -py;
  }

  let tMin = Infinity,
    tMax = -Infinity,
    sMinAll = Infinity,
    sMaxAll = -Infinity;
  for (const p of pts) {
    const t = (p.x - cx) * ux + (p.y - cy) * uy;
    const s = (p.x - cx) * px + (p.y - cy) * py;
    if (t < tMin) tMin = t;
    if (t > tMax) tMax = t;
    if (s < sMinAll) sMinAll = s;
    if (s > sMaxAll) sMaxAll = s;
  }
  const spanT = tMax - tMin;
  const spanS = sMaxAll - sMinAll;
  // Require longitudinal view of IVC (reject cross-sectional short-axis blobs)
  if (spanT < 2.2 || spanT / Math.max(0.3, spanS) < 1.35) return null;

  // 3. Locate IVC-RA junction along uIvc (`tMax` of IVC component toward RA) and hepatic vein confluence
  let hvTMin = Infinity;
  let hvTMax = -Infinity;
  let hvCount = 0;
  for (let li = 0; li < lines; li++) {
    const base = li * samples;
    for (let si = 0; si < samples; si++) {
      if (structure[base + si] !== Structure.HepaticVein) continue;
      const r = (si + 0.5) * g.dr;
      const x = r * g.sinTable[li]!;
      const y = r * g.cosTable[li]!;
      const t = (x - cx) * ux + (y - cy) * uy;
      if (t < hvTMin) hvTMin = t;
      if (t > hvTMax) hvTMax = t;
      hvCount++;
    }
  }

  // ASE 2025 station: 1.5 cm proximal (caudal) to IVC-RA junction (`tMax`), clear of hepatic vein ostium
  const tTarget = Math.max(tMin + 0.6, tMax - 1.7);
  const tLo = Math.max(tMin + 0.45, tMax - 3.4);
  const tHi = Math.min(tMax - 0.8, tMax - 0.9);
  if (tHi < tLo) return null;

  const stepCm = 0.02;
  let best: {
    a: { x: number; y: number };
    b: { x: number; y: number };
    valueCm: number;
    score: number;
  } | null = null;

  for (let t = tLo; t <= tHi; t += 0.1) {
    const ox = cx + ux * t;
    const oy = cy + uy * t;

    // Find contiguous Structure.Ivc runs along perpendicular ray
    const runs: { s0: number; s1: number }[] = [];
    let inRun = false;
    let rStart = 0;
    for (let s = -2.6; s <= 2.6; s += stepCm) {
      const st = sampleAtCm(g, ox + px * s, oy + py * s);
      if (st === Structure.Ivc) {
        if (!inRun) {
          inRun = true;
          rStart = s;
        }
      } else if (inRun) {
        runs.push({ s0: rStart, s1: s - stepCm });
        inRun = false;
      }
    }
    if (inRun) runs.push({ s0: rStart, s1: 2.6 });
    if (!runs.length) continue;

    // Pick run closest to vessel centerline (s = 0)
    let run = runs[0]!;
    let bestD = Math.abs(0.5 * (run.s0 + run.s1));
    for (const r of runs) {
      const d = Math.abs(0.5 * (r.s0 + r.s1));
      if (d < bestD) {
        bestD = d;
        run = r;
      }
    }

    // Both outside walls must be solid tissue (Liver/Diaphragm/Mediastinum — NOT HepaticVein or RA blood!)
    const antOut = sampleAtCm(g, ox + px * (run.s0 - 0.16), oy + py * (run.s0 - 0.16));
    const postOut = sampleAtCm(g, ox + px * (run.s1 + 0.16), oy + py * (run.s1 + 0.16));
    if (!isNonBloodTissue(antOut) || !isNonBloodTissue(postOut)) continue;

    // Trim bright vessel wall echoes so caliper sits strictly inner-edge to inner-edge
    const trimmed = trimBrightIvcWalls(g, ox, oy, px, py, run.s0, run.s1, stepCm);
    const len = trimmed.s1 - trimmed.s0;
    if (len < 0.25 || len > 4.2) continue;

    // Mild penalty if right inside the hepatic vein axial band so we prefer clean parallel IVC walls
    const inHvBand = hvCount >= 8 && t >= hvTMin - 0.35 && t <= hvTMax + 0.35;
    const score = -Math.abs(t - tTarget) * 1.4 - (inHvBand ? 1.5 : 0);
    if (!best || score > best.score) {
      best = {
        a: { x: ox + px * trimmed.s0, y: oy + py * trimmed.s0 },
        b: { x: ox + px * trimmed.s1, y: oy + py * trimmed.s1 },
        valueCm: len,
        score,
      };
    }
  }

  return best ? { a: best.a, b: best.b, valueCm: best.valueCm } : null;
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
  // Probe slightly beyond the thin vessel wall (~6 px ≈ 0.25 cm) to confirm solid surrounding tissue
  const ux = (dx / len) * 6;
  const uy = (dy / len) * 6;
  return {
    along,
    ends: [atPx(aPx.x - ux, aPx.y - uy), atPx(bPx.x + ux, bPx.y + uy)],
  };
}

/**
 * Automatic 2D subcostal IVC lumen measurement from the displayed cine frame. Automatically
 * commits `ivc-diameter-inspiration` during sniff inspiration and `ivc-diameter` otherwise.
 */
export function computeAutoIvc(
  cine: CineBuffer,
  displayed: CineFrame | undefined,
  displaySize: { width: number; height: number },
  settings: Pick<AcquisitionSettings, 'invertLR' | 'zoom'>,
): AutoIvcResult {
  const cf = displayed ?? cine.newest;
  if (!cf || cine.length === 0) {
    return {
      ok: false,
      reason: 'no-frame',
      message: 'Aún no hay cuadros adquiridos en el cine.',
    };
  }
  const mod = cf.acquisition.modality;
  if (mod !== '2d' && mod !== 'color') {
    return {
      ok: false,
      reason: 'wrong-modality',
      message: 'Cambia a modo 2D (o Color) en vista subcostal VCI para medir automáticamente la cava.',
    };
  }

  const ext = extractIvcDiameter(cf.structure, cf.spec, cf.display);
  if (!ext) {
    return {
      ok: false,
      reason: 'not-ivc',
      message:
        'No se identifica la vena cava inferior en eje largo; adquiere la vista subcostal VCI mostrando hígado y desembocadura en AD.',
    };
  }

  const W = Math.max(64, displaySize.width);
  const H = Math.max(64, displaySize.height);
  const sectorH = isStripModality(mod) ? Math.round(H * 0.42) : H;
  const sector = computeSectorMapping(cf.spec, W, sectorH, settings.invertLR, settings.zoom);
  const aPx = sectorCmToPixel(sector, ext.a);
  const bPx = sectorCmToPixel(sector, ext.b);
  const sampled = sampleSegmentOnFrame(cf, sector, aPx, bPx);
  const respiration = cf.acquisition.patient.respiration;
  const measurementId: IvcMeasurementId =
    respiration === 'inspiration' ? 'ivc-diameter-inspiration' : 'ivc-diameter';

  const found = cine.findFrameOffset(cf.frameId);
  return {
    ok: true,
    measurementId,
    respiration,
    valueCm: ext.valueCm,
    aPx,
    bPx,
    offset: found?.offset ?? 0,
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
}
