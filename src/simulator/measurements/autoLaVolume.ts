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
import { sectorCmToPixel, type AutoChamberFailureReason } from './autoVolume';
import { cutBySectorDepth, discProfileFromContour, volumeFromProfileMl } from './simpson';

/**
 * Pure 2D polar Left Atrial end-systolic volume (ASE/EACVI 2015 method of discs).
 *
 * Traces strictly `Structure.LaCavity` — excluding `Structure.LaAppendage` and
 * `Structure.PulmonaryVein` per ASE guidelines — closing straight across the mitral
 * annular hinge plane at ventricular end-systole (maximal LA volume).
 * Never reads ground truth.
 */

const LA_CAVITY_SET = new Set<number>([Structure.LaCavity]);
const LA_BASE_SET = new Set<number>([
  Structure.MitralAnnulus,
  Structure.MitralAnterior,
  Structure.MitralPosterior,
  Structure.LvCavity,
  Structure.AortomitralCurtain,
]);

const MIN_LA_AREA_CM2 = 3.0;
const MIN_LA_BASE_SPAN_CM = 1.1;
const MIN_LA_LONG_AXIS_CM = 2.0;
/** In apical views (A4C/A2C), the LA dome sits deeper than the mitral plane (opposite orientation to LV/RV). */
const MIN_LA_APICAL_ALIGNMENT = 0.72;
const MIN_LA_BASE_HORIZONTAL = 0.65;
const CONTOUR_BINS = 20;
const MAX_PHASE_DIST = 0.1;

export interface ExtractedLaContour {
  areaCm2: number;
  volumeMl: number;
  longAxisCm: number;
  baseSpanCm: number;
  /** Ordered open endocardial contour in un-inverted sector cm: hinge1 -> left wall -> roof -> right wall -> hinge2. */
  contourCm: { x: number; y: number }[];
  cutByDepth: boolean;
}

export interface AutoLaVolumeSuccess {
  ok: true;
  volumeMl: number;
  areaCm2: number;
  longAxisCm: number;
  cutByDepth: boolean;
  offset: number;
  frameId: number;
  phase: number;
  timeS: number;
  beatIndex: number;
  rrS: number;
  contourPx: { x: number; y: number }[];
  sector: SectorMapping;
  viewId: string | null;
  viewScore: number | null;
  gainScore: number | null;
  acquisition: FrameAcquisition;
  interiorStructures: number[];
}

export interface AutoLaVolumeFailure {
  ok: false;
  reason: AutoChamberFailureReason;
  message: string;
}

export type AutoLaVolumeResult = AutoLaVolumeSuccess | AutoLaVolumeFailure;

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

function pointInPolygon(x: number, y: number, pts: readonly { x: number; y: number }[]): boolean {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const a = pts[i]!;
    const b = pts[j]!;
    if (a.y > y !== b.y > y && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

function sampleInteriorStructures(
  cf: CineFrame,
  sector: SectorMapping,
  contourPx: readonly { x: number; y: number }[],
  grid = 12,
): number[] {
  if (contourPx.length < 3 || cf.structure.length === 0) return [];
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (const p of contourPx) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const p = cf.spec;
  const out: number[] = [];
  for (let i = 0; i < grid; i++) {
    for (let j = 0; j < grid; j++) {
      const x = minX + ((i + 0.5) / grid) * (maxX - minX);
      const y = minY + ((j + 0.5) / grid) * (maxY - minY);
      if (!pointInPolygon(x, y, contourPx)) continue;
      const { rCm, thetaRad } = pixelToPolar(sector, x, y);
      if (rCm < 0 || rCm > p.depthCm || Math.abs(thetaRad) > p.sectorRad / 2) continue;
      const li = Math.min(
        p.lines - 1,
        Math.max(0, Math.floor(((thetaRad + p.sectorRad / 2) / p.sectorRad) * p.lines)),
      );
      const si = Math.min(p.samples - 1, Math.max(0, Math.floor((rCm / p.depthCm) * p.samples)));
      out.push(cf.structure[li * p.samples + si] ?? 0);
    }
  }
  return out;
}

/** Extract the apical Left Atrial endocardial contour (excluding LAA & pulmonary veins) from a 2D polar slice. */
export function extractLaContour(
  structure: Uint8Array,
  spec: Pick<PolarFrameSpec, 'lines' | 'samples' | 'sectorRad' | 'depthCm'>,
): ExtractedLaContour | null {
  const { lines, samples, sectorRad, depthCm } = spec;
  const total = lines * samples;
  if (structure.length < total || lines < 8 || samples < 16) return null;

  // 1. Largest 4-connected component of LaCavity (strictly excluding LaAppendage & PulmonaryVein)
  const visited = new Uint8Array(total);
  const queue = new Int32Array(total);
  let bestCells: number[] = [];
  for (let idx = 0; idx < total; idx++) {
    if (visited[idx] || !LA_CAVITY_SET.has(structure[idx]!)) continue;
    let qHead = 0;
    let qTail = 0;
    queue[qTail++] = idx;
    visited[idx] = 1;
    const comp: number[] = [];
    while (qHead < qTail) {
      const cur = queue[qHead++]!;
      comp.push(cur);
      const li = Math.floor(cur / samples);
      const si = cur - li * samples;
      if (li > 0) {
        const n = cur - samples;
        if (!visited[n] && LA_CAVITY_SET.has(structure[n]!)) {
          visited[n] = 1;
          queue[qTail++] = n;
        }
      }
      if (li + 1 < lines) {
        const n = cur + samples;
        if (!visited[n] && LA_CAVITY_SET.has(structure[n]!)) {
          visited[n] = 1;
          queue[qTail++] = n;
        }
      }
      if (si > 0) {
        const n = cur - 1;
        if (!visited[n] && LA_CAVITY_SET.has(structure[n]!)) {
          visited[n] = 1;
          queue[qTail++] = n;
        }
      }
      if (si + 1 < samples) {
        const n = cur + 1;
        if (!visited[n] && LA_CAVITY_SET.has(structure[n]!)) {
          visited[n] = 1;
          queue[qTail++] = n;
        }
      }
    }
    if (comp.length > bestCells.length) bestCells = comp;
  }
  if (bestCells.length < 24) return null;

  const dr = depthCm / samples;
  const dth = sectorRad / lines;
  const sinTable = new Float64Array(lines);
  const cosTable = new Float64Array(lines);
  for (let li = 0; li < lines; li++) {
    const th = -0.5 * sectorRad + (li + 0.5) * dth;
    sinTable[li] = Math.sin(th);
    cosTable[li] = Math.cos(th);
  }

  let areaCm2 = 0;
  const pts: { x: number; y: number; li: number; si: number }[] = [];
  const basePts: { x: number; y: number; li: number; si: number }[] = [];
  // Check shallower/adjacent cells for the mitral annular plane
  const baseOffsets: readonly [number, number][] = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
    [-1, -1],
    [1, -1],
    [0, -2],
  ];

  for (let k = 0; k < bestCells.length; k++) {
    const idx = bestCells[k]!;
    const li = Math.floor(idx / samples);
    const si = idx - li * samples;
    const r = (si + 0.5) * dr;
    areaCm2 += r * dr * dth;
    const pt = { x: r * sinTable[li]!, y: r * cosTable[li]!, li, si };
    pts.push(pt);
    for (let o = 0; o < baseOffsets.length; o++) {
      const nl = li + baseOffsets[o]![0];
      const ns = si + baseOffsets[o]![1];
      if (
        nl >= 0 &&
        nl < lines &&
        ns >= 0 &&
        ns < samples &&
        LA_BASE_SET.has(structure[nl * samples + ns]!)
      ) {
        basePts.push(pt);
        break;
      }
    }
  }
  if (areaCm2 < MIN_LA_AREA_CM2 || basePts.length < 2) return null;

  // 2. Locate the two mitral annular hinges (widest span along the mitral boundary)
  let h1 = basePts[0]!;
  let h2 = basePts[1]!;
  let baseSpanCm = -1;
  for (let i = 0; i < basePts.length; i++) {
    const a = basePts[i]!;
    for (let j = i + 1; j < basePts.length; j++) {
      const b = basePts[j]!;
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (d > baseSpanCm) {
        baseSpanCm = d;
        h1 = a;
        h2 = b;
      }
    }
  }
  if (baseSpanCm < MIN_LA_BASE_SPAN_CM) return null;

  const bx = 0.5 * (h1.x + h2.x);
  const by = 0.5 * (h1.y + h2.y);

  // 3. Locate the LA roof (posterior dome) as the LaCavity point farthest from the mitral midpoint
  let roof = pts[0]!;
  let longAxisCm = -1;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i]!;
    const d = Math.hypot(p.x - bx, p.y - by);
    if (d > longAxisCm) {
      longAxisCm = d;
      roof = p;
    }
  }
  if (longAxisCm < MIN_LA_LONG_AXIS_CM) return null;

  // Apical orientation guards: LA roof sits deeper than a horizontally oriented mitral plane
  // (rejects PLAX where LA lies horizontally behind aortic root & mitral valve).
  if (roof.li <= 2 || roof.li >= lines - 3) return null;
  if ((roof.y - by) / longAxisCm < MIN_LA_APICAL_ALIGNMENT) return null;
  if (Math.abs(h2.x - h1.x) / baseSpanCm < MIN_LA_BASE_HORIZONTAL) return null;

  // Require LV cavity visible shallower than the mitral plane (confirms apical 4C/2C/3C view)
  let lvShallowCount = 0;
  for (let li = 0; li < lines; li++) {
    const base = li * samples;
    for (let si = 0; si < samples; si++) {
      if (structure[base + si] !== Structure.LvCavity) continue;
      const r = (si + 0.5) * dr;
      if (r * cosTable[li]! < by) lvShallowCount++;
    }
  }
  if (lvShallowCount < 20) return null;

  const ax = (roof.x - bx) / longAxisCm;
  const ay = (roof.y - by) / longAxisCm;
  const px = -ay;
  const py = ax;

  // Order hinges consistently along perpendicular axis
  if ((h1.x - bx) * px + (h1.y - by) * py > (h2.x - bx) * px + (h2.y - by) * py) {
    [h1, h2] = [h2, h1];
  }

  // 4. Trace left & right LA endocardial walls in CONTOUR_BINS slices from mitral base to roof.
  // Expand each cell center by half its polar lateral footprint so the contour traces the actual
  // outer LaCavity endocardial pixel boundary rather than the inset cell-center locus.
  const sMin = new Array<number>(CONTOUR_BINS).fill(Infinity);
  const sMax = new Array<number>(CONTOUR_BINS).fill(-Infinity);
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i]!;
    const t = (p.x - bx) * ax + (p.y - by) * ay;
    if (t < -0.25 || t > longAxisCm + 0.15) continue;
    const bin = Math.min(CONTOUR_BINS - 1, Math.max(0, Math.floor((t / longAxisCm) * CONTOUR_BINS)));
    const r = (p.si + 0.5) * dr;
    const halfCellW = 0.5 * Math.max(dr, r * dth);
    const s = (p.x - bx) * px + (p.y - by) * py;
    if (s - halfCellW < sMin[bin]!) sMin[bin] = s - halfCellW;
    if (s + halfCellW > sMax[bin]!) sMax[bin] = s + halfCellW;
  }

  const fillGaps = (arr: number[], edgeStart: number, edgeEnd: number): number[] => {
    const out = [...arr];
    for (let i = 0; i < out.length; i++) {
      if (Number.isFinite(out[i]!)) continue;
      let lo = i - 1;
      while (lo >= 0 && !Number.isFinite(out[lo]!)) lo--;
      let hi = i + 1;
      while (hi < out.length && !Number.isFinite(out[hi]!)) hi++;
      const vLo = lo >= 0 ? out[lo]! : edgeStart;
      const vHi = hi < out.length ? out[hi]! : edgeEnd;
      const span = (hi < out.length ? hi : out.length) - (lo >= 0 ? lo : -1);
      const frac = (i - (lo >= 0 ? lo : -1)) / Math.max(1, span);
      out[i] = vLo + (vHi - vLo) * frac;
    }
    return out;
  };
  const h1S = (h1.x - bx) * px + (h1.y - by) * py;
  const h2S = (h2.x - bx) * px + (h2.y - by) * py;
  const leftRaw = fillGaps(sMin, h1S, 0);
  const rightRaw = fillGaps(sMax, h2S, 0);

  // Light spike-removing filter that preserves convex atrial wall expansion
  const smoothOuter = (arr: number[], isLeft: boolean): number[] =>
    arr.map((v, i) => {
      if (i === 0 || i === arr.length - 1) return v;
      const avg = 0.2 * arr[i - 1]! + 0.6 * v + 0.2 * arr[i + 1]!;
      return isLeft ? Math.min(v, avg) : Math.max(v, avg);
    });
  const leftS = smoothOuter(leftRaw, true);
  const rightS = smoothOuter(rightRaw, false);

  let maxWidthCm = 0;
  const leftWall: { x: number; y: number }[] = [{ x: h1.x, y: h1.y }];
  const rightWall: { x: number; y: number }[] = [{ x: h2.x, y: h2.y }];
  for (let i = 0; i < CONTOUR_BINS; i++) {
    const w = rightS[i]! - leftS[i]!;
    if (w > maxWidthCm) maxWidthCm = w;
    if (w <= 0.05) continue;
    const t = ((i + 0.5) / CONTOUR_BINS) * longAxisCm;
    leftWall.push({ x: bx + ax * t + px * leftS[i]!, y: by + ay * t + py * leftS[i]! });
    rightWall.push({ x: bx + ax * t + px * rightS[i]!, y: by + ay * t + py * rightS[i]! });
  }
  if (maxWidthCm < 1.2) return null;

  const roofEdge = { x: roof.x + ax * 0.5 * dr, y: roof.y + ay * 0.5 * dr };
  const contourCm = [...leftWall, roofEdge, ...rightWall.reverse()];
  const prof = discProfileFromContour(contourCm, 1);
  if (!prof) return null;

  const volumeMl = volumeFromProfileMl(prof);
  if (!(volumeMl > 0) || !Number.isFinite(volumeMl)) return null;

  return {
    areaCm2,
    volumeMl,
    longAxisCm: prof.longAxisCm,
    baseSpanCm,
    contourCm,
    cutByDepth: cutBySectorDepth({ structure, polar: spec }, [...LA_CAVITY_SET]),
  };
}

/**
 * Automatic end-systolic Left Atrial volume (ASE/EACVI method of discs) directly from the
 * acquired 2D/Color cine buffer on A4C or A2C. Repeat on both planes to derive biplane LA volume
 * and LAVI (mL/m²). Never reads ground truth.
 */
export function computeAutoLaVolume(
  cine: CineBuffer,
  displayed: CineFrame | undefined,
  phaseMarks: PhaseMarks,
  displaySize: { width: number; height: number },
  settings: Pick<AcquisitionSettings, 'invertLR' | 'zoom'>,
): AutoLaVolumeResult {
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
      message: 'Cambia a modo 2D (o Color) en vista apical (A4C o A2C) para medir el volumen de la AI.',
    };
  }

  const refAcq = displayed.acquisition;
  const matcher = (f: CineFrame) => sameAcquisitionPose(f.acquisition, refAcq);
  // Ventricular end-systole = maximal LA volume just before mitral opening
  const esPick = cine.findNearestPhaseOffsetMatching(phaseMarks.endSystole, matcher);
  if (!esPick || esPick.distance > MAX_PHASE_DIST) {
    return {
      ok: false,
      reason: 'incomplete-cycle',
      message:
        'Deja correr al menos un latido completo en A4C o A2C para capturar la telesístole (AI máxima).',
    };
  }

  const esFrame = cine.select(esPick.offset);
  if (!esFrame) {
    return {
      ok: false,
      reason: 'no-frame',
      message: 'No se pudo recuperar el cuadro de telesístole del cine.',
    };
  }

  const ext = extractLaContour(esFrame.structure, esFrame.spec);
  if (!ext) {
    return {
      ok: false,
      reason: 'chamber-not-visible',
      message:
        'No se detecta el contorno apical completo de la AI en este corte; adquiere A4C o A2C centrando la aurícula.',
    };
  }

  const W = Math.max(64, displaySize.width);
  const H = Math.max(64, displaySize.height);
  const sectorH = isStripModality(mod) ? Math.round(H * 0.42) : H;
  const sector = computeSectorMapping(esFrame.spec, W, sectorH, settings.invertLR, settings.zoom);
  const contourPx = ext.contourCm.map((pt) => sectorCmToPixel(sector, pt));
  const interiorStructures = sampleInteriorStructures(esFrame, sector, contourPx);

  return {
    ok: true,
    volumeMl: ext.volumeMl,
    areaCm2: ext.areaCm2,
    longAxisCm: ext.longAxisCm,
    cutByDepth: ext.cutByDepth,
    offset: esPick.offset,
    frameId: esFrame.frameId,
    phase: esFrame.phase,
    timeS: esFrame.timeS,
    beatIndex: esFrame.beatIndex,
    rrS: esFrame.rrS,
    contourPx,
    sector,
    viewId: esFrame.analysis?.bestViewId ?? null,
    viewScore: esFrame.analysis?.score ?? null,
    gainScore: esFrame.analysis ? Math.round(esFrame.analysis.components.gain * 100) : null,
    acquisition: esFrame.acquisition,
    interiorStructures,
  };
}
