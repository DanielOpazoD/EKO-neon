import { fractionalAreaChange, rvVolumeFromAreaMl } from '@/clinical/formulas';
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
import { cutBySectorDepth, discProfileFromContour, volumeFromProfileMl } from './simpson';

export type ChamberId = 'lv' | 'rv';

/** Endocardial blood + intracavitary structures included inside the traced border (ASE 2015 / 2025). */
const LV_CAVITY_SET = new Set<number>([
  Structure.LvCavity,
  Structure.PapillaryMuscle,
  Structure.Chordae,
]);
const LV_BASE_SET = new Set<number>([
  Structure.MitralAnnulus,
  Structure.MitralAnterior,
  Structure.MitralPosterior,
  Structure.LaCavity,
  Structure.AortomitralCurtain,
]);

const RV_CAVITY_SET = new Set<number>([
  Structure.RvCavity,
  Structure.RvPapillary,
  Structure.ModeratorBand,
]);
const RV_BASE_SET = new Set<number>([
  Structure.TricuspidAnnulus,
  Structure.TricuspidValve,
  Structure.RaCavity,
]);

const MIN_CAVITY_AREA_CM2 = 3.0;
const MIN_BASE_SPAN_CM = 1.2;
const MIN_LONG_AXIS_CM = 3.0;
/** Apical orientation guards (base deeper than apex, horizontal annular plane) so PLAX/PSAX are rejected from 2D geometry alone. */
const MIN_APICAL_ALIGNMENT = 0.78;
const MIN_BASE_HORIZONTAL = 0.72;
const CONTOUR_BINS = 20;
const MAX_PHASE_DIST = 0.09;

export interface ExtractedChamberContour {
  chamber: ChamberId;
  areaCm2: number;
  volumeMl: number;
  longAxisCm: number;
  baseSpanCm: number;
  /** Ordered open endocardial contour in un-inverted sector cm: hinge1 -> left wall -> apex -> right wall -> hinge2. */
  contourCm: { x: number; y: number }[];
  cutByDepth: boolean;
}

export type AutoChamberFailureReason =
  | 'no-frame'
  | 'wrong-modality'
  | 'incomplete-cycle'
  | 'chamber-not-visible'
  | 'invalid-contour';

export interface AutoChamberVolumeFailure {
  ok: false;
  chamber: ChamberId;
  reason: AutoChamberFailureReason;
  message: string;
}

export interface AutoChamberFrameSnapshot {
  offset: number;
  frameId: number;
  phase: number;
  timeS: number;
  beatIndex: number;
  volumeMl: number;
  areaCm2: number;
  longAxisCm: number;
  cutByDepth: boolean;
  /** Display-pixel endocardial contour for `sector`. */
  contourPx: { x: number; y: number }[];
  sector: SectorMapping;
  viewId: string | null;
  viewScore: number | null;
  gainScore: number | null;
  acquisition: FrameAcquisition;
  /** Structure ids sampled inside `contourPx` on this frame for technique evaluation. */
  interiorStructures: number[];
}

export interface AutoChamberVolumeSuccess {
  ok: true;
  chamber: ChamberId;
  edvMl: number;
  esvMl: number;
  svMl: number;
  efPct: number;
  edAreaCm2: number;
  esAreaCm2: number;
  facPct: number;
  rrS: number;
  ed: AutoChamberFrameSnapshot;
  es: AutoChamberFrameSnapshot;
}

export type AutoChamberVolumeResult = AutoChamberVolumeSuccess | AutoChamberVolumeFailure;

/** Shoelace polygon area in cm² from display-pixel vertices and `pxPerCm`. */
export function polygonAreaCm2(
  points: readonly { x: number; y: number }[],
  pxPerCm: number,
): number {
  if (points.length < 3 || !(pxPerCm > 0)) return 0;
  let twice = 0;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const a = points[j]!;
    const b = points[i]!;
    twice += a.x * b.y - b.x * a.y;
  }
  return Math.abs(twice) * 0.5 / (pxPerCm * pxPerCm);
}

/** Map an un-inverted sector-cm contour point to display pixels using `SectorMapping`. */
export function sectorCmToPixel(
  m: SectorMapping,
  p: { x: number; y: number },
): { x: number; y: number } {
  return {
    x: m.apexX + p.x * m.pxPerCm * (m.invertLR ? -1 : 1),
    y: m.apexY + p.y * m.pxPerCm,
  };
}

/** Extract the largest 4-connected cavity component on a polar slice and trace its endocardial border. */
export function extractChamberContour(
  structure: Uint8Array,
  spec: Pick<PolarFrameSpec, 'lines' | 'samples' | 'sectorRad' | 'depthCm'>,
  chamber: ChamberId,
): ExtractedChamberContour | null {
  const { lines, samples, sectorRad, depthCm } = spec;
  const total = lines * samples;
  if (structure.length < total || lines < 8 || samples < 16) return null;

  const cavSet = chamber === 'lv' ? LV_CAVITY_SET : RV_CAVITY_SET;
  const baseSet = chamber === 'lv' ? LV_BASE_SET : RV_BASE_SET;

  // 1. Largest 4-connected component in polar grid to discard isolated noise cells.
  const visited = new Uint8Array(total);
  const queue = new Int32Array(total);
  let bestCells: number[] = [];
  for (let idx = 0; idx < total; idx++) {
    if (visited[idx] || !cavSet.has(structure[idx]!)) continue;
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
        if (!visited[n] && cavSet.has(structure[n]!)) {
          visited[n] = 1;
          queue[qTail++] = n;
        }
      }
      if (li + 1 < lines) {
        const n = cur + samples;
        if (!visited[n] && cavSet.has(structure[n]!)) {
          visited[n] = 1;
          queue[qTail++] = n;
        }
      }
      if (si > 0) {
        const n = cur - 1;
        if (!visited[n] && cavSet.has(structure[n]!)) {
          visited[n] = 1;
          queue[qTail++] = n;
        }
      }
      if (si + 1 < samples) {
        const n = cur + 1;
        if (!visited[n] && cavSet.has(structure[n]!)) {
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
  const pts: { x: number; y: number; li: number }[] = [];
  const basePts: { x: number; y: number; li: number }[] = [];
  const baseOffsets: readonly [number, number][] = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
    [-1, 1],
    [1, 1],
    [0, 2],
  ];

  for (let k = 0; k < bestCells.length; k++) {
    const idx = bestCells[k]!;
    const li = Math.floor(idx / samples);
    const si = idx - li * samples;
    const r = (si + 0.5) * dr;
    areaCm2 += r * dr * dth;
    const pt = { x: r * sinTable[li]!, y: r * cosTable[li]!, li };
    pts.push(pt);
    for (let o = 0; o < baseOffsets.length; o++) {
      const nl = li + baseOffsets[o]![0];
      const ns = si + baseOffsets[o]![1];
      if (nl >= 0 && nl < lines && ns >= 0 && ns < samples && baseSet.has(structure[nl * samples + ns]!)) {
        basePts.push(pt);
        break;
      }
    }
  }

  if (areaCm2 < MIN_CAVITY_AREA_CM2 || basePts.length < 2) return null;

  // 2. Locate the two annular hinges as the widest span along the valve/atrial boundary.
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
  if (baseSpanCm < MIN_BASE_SPAN_CM) return null;

  const bx = 0.5 * (h1.x + h2.x);
  const by = 0.5 * (h1.y + h2.y);

  // 3. Locate visible apex as the cavity point farthest from the annular midpoint.
  let apex = pts[0]!;
  let longAxisCm = -1;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i]!;
    const d = Math.hypot(p.x - bx, p.y - by);
    if (d > longAxisCm) {
      longAxisCm = d;
      apex = p;
    }
  }
  if (longAxisCm < MIN_LONG_AXIS_CM) return null;

  // Apical orientation guards: apex must lie inside the sector (not chopped by a lateral border
  // as in PLAX) and shallower than a horizontally oriented AV annular plane.
  if (apex.li <= 2 || apex.li >= lines - 3) return null;
  if ((by - apex.y) / longAxisCm < MIN_APICAL_ALIGNMENT) return null;
  if (Math.abs(h2.x - h1.x) / baseSpanCm < MIN_BASE_HORIZONTAL) return null;

  const ax = (apex.x - bx) / longAxisCm;
  const ay = (apex.y - by) / longAxisCm;
  const px = -ay;
  const py = ax;

  // Order hinges so h1 lies on the negative perpendicular side and h2 on the positive side.
  if ((h1.x - bx) * px + (h1.y - by) * py > (h2.x - bx) * px + (h2.y - by) * py) {
    [h1, h2] = [h2, h1];
  }

  // 4. Trace left & right endocardial walls in CONTOUR_BINS slices along the long axis.
  const sMin = new Array<number>(CONTOUR_BINS).fill(Infinity);
  const sMax = new Array<number>(CONTOUR_BINS).fill(-Infinity);
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i]!;
    const t = (p.x - bx) * ax + (p.y - by) * ay;
    if (t < 0 || t > longAxisCm) continue;
    const bin = Math.min(CONTOUR_BINS - 1, Math.max(0, Math.floor((t / longAxisCm) * CONTOUR_BINS)));
    const s = (p.x - bx) * px + (p.y - by) * py;
    if (s < sMin[bin]!) sMin[bin] = s;
    if (s > sMax[bin]!) sMax[bin] = s;
  }

  // Fill any sparse bin by linear interpolation from neighboring valid bins.
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

  // Light 3-point smoothing on interior bins removes polar sampling staircase without shrinking volume.
  const smooth = (arr: number[]): number[] =>
    arr.map((v, i) =>
      i === 0 || i === arr.length - 1 ? v : 0.25 * arr[i - 1]! + 0.5 * v + 0.25 * arr[i + 1]!,
    );
  const leftS = smooth(leftRaw);
  const rightS = smooth(rightRaw);

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
  if (maxWidthCm < 0.8 || longAxisCm / maxWidthCm < 0.85) return null;

  const contourCm = [...leftWall, { x: apex.x, y: apex.y }, ...rightWall.reverse()];
  const prof = discProfileFromContour(contourCm, 1);
  if (!prof) return null;

  const volumeMl = chamber === 'lv' ? volumeFromProfileMl(prof) : rvVolumeFromAreaMl(areaCm2);
  if (!(volumeMl > 0) || !Number.isFinite(volumeMl)) return null;

  return {
    chamber,
    areaCm2,
    volumeMl,
    longAxisCm: prof.longAxisCm,
    baseSpanCm,
    contourCm,
    cutByDepth: cutBySectorDepth({ structure, polar: spec }, [...cavSet]),
  };
}

/** Whether two frames come from the same 2D/Color probe window and sector geometry. */
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

/**
 * Automatic end-diastolic & end-systolic LV/RV contour extraction, volume, EF and RV FAC directly
 * from the acquired 2D/Color cine buffer (never reads ground truth).
 */
export function computeAutoChamberVolume(
  cine: CineBuffer,
  displayed: CineFrame | undefined,
  phaseMarks: PhaseMarks,
  displaySize: { width: number; height: number },
  settings: Pick<AcquisitionSettings, 'invertLR' | 'zoom'>,
  chamber: ChamberId,
): AutoChamberVolumeResult {
  const chamberLabel = chamber === 'lv' ? 'del VI' : 'del VD';
  const viewHint = chamber === 'lv' ? 'A4C o A2C' : 'apical enfocada en VD o A4C';

  if (!displayed || cine.length === 0) {
    return {
      ok: false,
      chamber,
      reason: 'no-frame',
      message: 'Aún no hay cuadros adquiridos en el cine.',
    };
  }
  const mod = displayed.acquisition.modality;
  if (mod !== '2d' && mod !== 'color') {
    return {
      ok: false,
      chamber,
      reason: 'wrong-modality',
      message: `Cambia a modo 2D (o Color) en una vista apical (${viewHint}) para calcular el volumen automático ${chamberLabel}.`,
    };
  }

  const refAcq = displayed.acquisition;
  const matcher = (f: CineFrame) => sameAcquisitionPose(f.acquisition, refAcq);
  const edPick = cine.findNearestPhaseOffsetMatching(0, matcher);
  const esPick = cine.findNearestPhaseOffsetMatching(phaseMarks.endSystole, matcher);

  if (
    !edPick ||
    !esPick ||
    edPick.distance > MAX_PHASE_DIST ||
    esPick.distance > MAX_PHASE_DIST ||
    edPick.offset === esPick.offset
  ) {
    return {
      ok: false,
      chamber,
      reason: 'incomplete-cycle',
      message: `Deja correr al menos un latido completo en esta vista (${viewHint}) para capturar telediástole y telesístole.`,
    };
  }

  const edFrame = cine.select(edPick.offset);
  const esFrame = cine.select(esPick.offset);
  if (!edFrame || !esFrame) {
    return {
      ok: false,
      chamber,
      reason: 'no-frame',
      message: 'No se pudieron recuperar los cuadros de telediástole y telesístole del cine.',
    };
  }

  const edExt = extractChamberContour(edFrame.structure, edFrame.spec, chamber);
  const esExt = extractChamberContour(esFrame.structure, esFrame.spec, chamber);
  if (!edExt || !esExt) {
    return {
      ok: false,
      chamber,
      reason: 'chamber-not-visible',
      message: `No se detecta el contorno apical completo ${chamberLabel} en este corte; adquiere un plano apical (${viewHint}) sin cortar el ápex ni el anillo.`,
    };
  }
  if (!(edExt.volumeMl > 0) || !(esExt.volumeMl >= 0)) {
    return {
      ok: false,
      chamber,
      reason: 'invalid-contour',
      message: `El contorno detectado ${chamberLabel} no es válido para calcular volumen.`,
    };
  }

  const W = Math.max(64, displaySize.width);
  const H = Math.max(64, displaySize.height);
  const sectorH = isStripModality(mod) ? Math.round(H * 0.42) : H;

  const buildSnapshot = (
    cf: CineFrame,
    offset: number,
    ext: ExtractedChamberContour,
  ): AutoChamberFrameSnapshot => {
    const sector = computeSectorMapping(cf.spec, W, sectorH, settings.invertLR, settings.zoom);
    const contourPx = ext.contourCm.map((pt) => sectorCmToPixel(sector, pt));
    const interiorStructures = sampleInteriorStructures(cf, sector, contourPx);
    return {
      offset,
      frameId: cf.frameId,
      phase: cf.phase,
      timeS: cf.timeS,
      beatIndex: cf.beatIndex,
      volumeMl: ext.volumeMl,
      areaCm2: ext.areaCm2,
      longAxisCm: ext.longAxisCm,
      cutByDepth: ext.cutByDepth,
      contourPx,
      sector,
      viewId: cf.analysis?.bestViewId ?? null,
      viewScore: cf.analysis?.score ?? null,
      gainScore: cf.analysis ? Math.round(cf.analysis.components.gain * 100) : null,
      acquisition: cf.acquisition,
      interiorStructures,
    };
  };

  const ed = buildSnapshot(edFrame, edPick.offset, edExt);
  const es = buildSnapshot(esFrame, esPick.offset, esExt);
  const svMl = ed.volumeMl - es.volumeMl;
  const efPct = (svMl / ed.volumeMl) * 100;
  const facPct = fractionalAreaChange(ed.areaCm2, es.areaCm2);

  return {
    ok: true,
    chamber,
    edvMl: ed.volumeMl,
    esvMl: es.volumeMl,
    svMl,
    efPct,
    edAreaCm2: ed.areaCm2,
    esAreaCm2: es.areaCm2,
    facPct,
    rrS: edFrame.rrS,
    ed,
    es,
  };
}
