import {
  imageSegmentsOn,
  useSegmentHover,
  useSegmentOrientation,
  useStructureHover,
  type SimStore,
} from '@/app/store';
import type { ReviewMarker } from '@/app/review';
import { cellAt, structureComponent } from '@/app/reviewRegion';
import { Structure } from '@/simulator/anatomy/tissue';
import type { SimOutput } from '@/simulator/core/protocol';
import { canvasFont } from '../canvasFonts';
import {
  computeStructureStats,
  nearestSampleLut,
  paintStructureOutline,
  placeLabels,
  withoutOverlaps,
} from '../cutMap';
import { imageUpOnPolarMap, rvInsertionPoints, SegmentAnchors, type RvInsertions } from '../segmentAnchors';
import { paintSegmentOverlay, segmentIds } from '../segmentMap';

/**
 * Shared pixel-to-polar-sample LUT keyed by sector geometry.
 * Picked-region, structure-outline, and LV-segment overlays all project onto the exact same
 * sector grid; sharing one LUT avoids 3x redundant allocation (~2 MB each) and 3x rebuild work
 * whenever depth, zoom, or canvas size changes.
 */
const sharedLut: { key: string; lut: Int32Array | null } = { key: '', lut: null };

function sectorGeometryKey(hud: SimOutput, w: number, h: number): string {
  const m = hud.sector;
  const p = hud.polar;
  return `${p.lines}x${p.samples}|${p.sectorRad.toFixed(4)}|${p.depthCm}|${w}x${h}|${m.apexX.toFixed(1)},${m.apexY.toFixed(1)}|${m.pxPerCm.toFixed(3)}|${m.invertLR ? 1 : 0}`;
}

function getSharedSectorLut(hud: SimOutput, w: number, h: number, key: string): Int32Array {
  if (key !== sharedLut.key || !sharedLut.lut) {
    sharedLut.lut = nearestSampleLut(hud.polar, { ...hud.sector, width: w, height: h });
    sharedLut.key = key;
  }
  return sharedLut.lut;
}

interface OffscreenSurface {
  key: string;
  canvas: HTMLCanvasElement | null;
  img: ImageData | null;
}

function ensureOffscreenSurface(
  surface: OffscreenSurface,
  key: string,
  w: number,
  h: number,
): { canvas: HTMLCanvasElement; img: ImageData; reset: boolean } {
  if (key !== surface.key || !surface.canvas || !surface.img) {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    surface.canvas = canvas;
    surface.img = new ImageData(w, h);
    surface.key = key;
    return { canvas, img: surface.img, reset: true };
  }
  return { canvas: surface.canvas, img: surface.img, reset: false };
}

/** Layer of the picked region (decision 340): the piece's mask and offscreen surface. */
const pickLayer: OffscreenSurface & {
  maskKey: string;
  mask: Uint8Array | null;
} = { key: '', maskKey: '', mask: null, canvas: null, img: null };

/**
 * The piece of the structure under the selected click (decision 340), tinted and outlined in the markers' colour, so the
 * reader sees what the click picked: the connected run of that structure on the frame's own map, not every piece of it.
 */
export function drawPickedRegion(
  ctx: CanvasRenderingContext2D,
  hud: SimOutput,
  mk: ReviewMarker,
): void {
  const m = hud.sector;
  const p = hud.polar;
  if (hud.structure.length !== p.lines * p.samples || m.width < 8 || m.height < 8) return;
  const w = Math.round(m.width),
    h = Math.round(m.height);
  const key = sectorGeometryKey(hud, w, h);
  const lut = getSharedSectorLut(hud, w, h, key);
  const { canvas, img, reset } = ensureOffscreenSurface(pickLayer, key, w, h);
  if (reset) pickLayer.maskKey = '';

  const maskKey = `${key}|${hud.frameId}|${mk.id}|${mk.rCm}|${mk.thetaRad}`;
  if (maskKey !== pickLayer.maskKey || !pickLayer.mask) {
    const pm = { ...p, structure: hud.structure };
    const cell = cellAt(pm, mk.rCm!, mk.thetaRad!);
    const mask = new Uint8Array(p.lines * p.samples);
    for (const c of structureComponent(pm, cell)) mask[c] = 1;
    pickLayer.mask = mask;
    pickLayer.maskKey = maskKey;
  }
  const mask = pickLayer.mask;
  img.data.fill(0);
  // a light tint over the piece, then its edge
  for (let i = 0; i < lut.length; i++) {
    if (lut[i]! < 0 || !mask[lut[i]!]) continue;
    const o = i * 4;
    img.data[o] = 255;
    img.data[o + 1] = 106;
    img.data[o + 2] = 213;
    img.data[o + 3] = 46;
  }
  if (!paintStructureOutline(img.data, lut, mask, w, 1, [255, 106, 213])) return;
  const off = canvas.getContext('2d');
  if (!off) return;
  off.putImageData(img, 0, 0);
  ctx.drawImage(canvas, m.x, m.y);
}

/** Offscreen surface for the outline of the hovered structure, kept while the geometry stays. */
const structLayer: OffscreenSurface = { key: '', canvas: null, img: null };

/**
 * The structure under the pointer on the cut map, outlined on the image in the accent (decision 202): the learner
 * points at «AI» on the drawing and sees where the atrium is in the echo. Nothing is drawn while nothing is pointed at.
 */
export function drawStructureOutline(ctx: CanvasRenderingContext2D, hud: SimOutput): void {
  const id = useStructureHover.getState().id;
  if (id === null) return;
  const m = hud.sector;
  const p = hud.polar;
  if (hud.structure.length !== p.lines * p.samples || m.width < 8 || m.height < 8) return;
  const w = Math.round(m.width),
    h = Math.round(m.height);
  const key = sectorGeometryKey(hud, w, h);
  const lut = getSharedSectorLut(hud, w, h, key);
  const { canvas, img } = ensureOffscreenSurface(structLayer, key, w, h);

  img.data.fill(0);
  if (!paintStructureOutline(img.data, lut, hud.structure, w, id)) return;
  const off = canvas.getContext('2d');
  if (!off) return;
  off.putImageData(img, 0, 0);
  ctx.drawImage(canvas, m.x, m.y);
}

/**
 * The LV segment layer of the image (decision 153): the frame's segment codes read in the chosen model, painted
 * translucent over the echo and numbered.
 */
const segLayer: OffscreenSurface & {
  ids: Uint8Array | undefined;
  anchors: SegmentAnchors;
} = {
  key: '',
  canvas: null,
  img: null,
  ids: undefined,
  anchors: new SegmentAnchors(),
};

/**
 * The RV insertions of a short-axis cut (decision 181): an amber pointer outside each, aimed at it from the side away
 * from the centre of the numbered myocardium. The septum (2, 3, 8, 9) lies between them, whatever the clock position.
 */
function drawRvInsertions(
  ctx: CanvasRenderingContext2D,
  m: { x: number; y: number },
  stats: readonly { id: number; count: number; cx: number; cy: number }[],
  ins: RvInsertions,
): void {
  let n = 0,
    cx = 0,
    cy = 0;
  for (const s of stats)
    if (s.id >= 1 && s.id <= 12) {
      cx += s.cx * s.count;
      cy += s.cy * s.count;
      n += s.count;
    }
  if (!n) return;
  cx /= n;
  cy /= n;
  ctx.save();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.85)';
  ctx.fillStyle = '#ffb020';
  for (const p of [ins.anterior, ins.inferior]) {
    const dx = p.x - cx,
      dy = p.y - cy;
    const d = Math.hypot(dx, dy) || 1;
    const ux = dx / d,
      uy = dy / d;
    // tip on the insertion, body 11 px outward, 8 px wide
    const tx = m.x + p.x,
      ty = m.y + p.y;
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx + ux * 11 - uy * 4, ty + uy * 11 + ux * 4);
    ctx.lineTo(tx + ux * 11 + uy * 4, ty + uy * 11 - ux * 4);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

export function drawSegmentLayer(ctx: CanvasRenderingContext2D, hud: SimOutput, st: SimStore): void {
  const m = hud.sector;
  const p = hud.polar;
  if (hud.segment.length !== p.lines * p.samples || m.width < 8 || m.height < 8) return;
  const w = Math.round(m.width),
    h = Math.round(m.height);
  const key = sectorGeometryKey(hud, w, h);
  const lut = getSharedSectorLut(hud, w, h, key);
  const { canvas, img } = ensureOffscreenSurface(segLayer, key, w, h);

  const model = st.ui.segmentModel;
  segLayer.ids = segmentIds(hud.segment, model, segLayer.ids);
  const hovered = useSegmentHover.getState().id;
  const stats = paintSegmentOverlay(
    img.data,
    lut,
    segLayer.ids,
    w,
    st.ui.selectedSegment,
    hovered,
  );
  const off = canvas.getContext('2d');
  if (!off) return;
  off.putImageData(img, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(canvas, m.x, m.y);
  // numbers on each segment and the RV insertions: the mean of their places over one beat of an unchanged probe,
  // geometry and model, then still (decision 181)
  ctx.font = canvasFont(12, 'ui', 700);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const pr = st.probe;
  const layoutKey = `${key}|${model}|${st.caseId}|${[pr.u, pr.v, pr.rotationDeg, pr.tiltDeg, pr.rockDeg, pr.pressure].map((v) => v.toFixed(3)).join(',')}`;
  const anchors = segLayer.anchors;
  if (anchors.wants(layoutKey, hud.frameId)) {
    const shown = new Set(anchors.view().labels.map((l) => l.id));
    const labels = placeLabels(
      stats,
      lut,
      segLayer.ids,
      w,
      Math.max(30, 0.0005 * w * h),
      (t) => ({ w: ctx.measureText(t).width + 6, h: 16 }),
      shown,
      (id) => String(id),
    );
    anchors.add(layoutKey, hud, labels, rvInsertionPoints(lut, segLayer.ids, w, stats));
  }
  const view = anchors.view();
  useSegmentOrientation.getState().set(imageUpOnPolarMap(view.labels), view.insertions !== null);
  // where the numbers and the insertions were drawn, rounded to the pixel, for the E2E that checks they hold still
  const placed = JSON.stringify({
    labels: view.labels.map((l) => [l.id, Math.round(l.x), Math.round(l.y)]),
    insertions:
      view.insertions &&
      [view.insertions.anterior, view.insertions.inferior].map((q) => [
        Math.round(q.x),
        Math.round(q.y),
      ]),
  });
  if (ctx.canvas.dataset.segmentLabels !== placed) ctx.canvas.dataset.segmentLabels = placed;
  if (view.insertions) drawRvInsertions(ctx, m, stats, view.insertions);
  ctx.lineJoin = 'round';
  ctx.lineWidth = 3;
  for (const l of view.labels) {
    const x = m.x + l.x,
      y = m.y + l.y;
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.85)';
    ctx.strokeText(l.text, x, y);
    ctx.fillStyle = Number(l.text) === hovered ? '#5cc8ff' : '#ffffff';
    ctx.fillText(l.text, x, y);
  }
  ctx.restore();
}

/**
 * Concise clinical labels for direct 2D sector anatomy overlay (teaching aid; disabled in exam mode).
 * Focused on cardiac chambers, valves, great vessels, pericardial effusion, and key acoustic landmarks
 * while omitting noisy exterior background regions (ribs/mediastinal fat) so the 2D image stays readable.
 */
export const IMAGE_ANATOMY_LABELS: Partial<Record<Structure, string>> = {
  [Structure.LvCavity]: 'VI',
  [Structure.LvWallSeptal]: 'Septo',
  [Structure.LvWallLateral]: 'P. lat',
  [Structure.LvWallAnterior]: 'P. ant',
  [Structure.LvWallInferior]: 'P. inf',
  [Structure.LvApex]: 'Ápex',
  [Structure.RvCavity]: 'VD',
  [Structure.RvWall]: 'Pared VD',
  [Structure.LaCavity]: 'AI',
  [Structure.RaCavity]: 'AD',
  [Structure.MitralAnterior]: 'VM ant',
  [Structure.MitralPosterior]: 'VM post',
  [Structure.AorticValve]: 'VAo',
  [Structure.TricuspidValve]: 'VT',
  [Structure.PulmonaryValve]: 'VP',
  [Structure.AorticRoot]: 'Ao',
  [Structure.Lvot]: 'TSVI',
  [Structure.Rvot]: 'TSVD',
  [Structure.PapillaryMuscle]: 'MP',
  [Structure.InteratrialSeptum]: 'TIA',
  [Structure.Pericardium]: 'Pericardio',
  [Structure.PericardialEffusion]: 'Derrame',
  [Structure.DescendingAorta]: 'Ao desc',
  [Structure.AscendingAorta]: 'Ao asc',
  [Structure.AorticArch]: 'Arco Ao',
  [Structure.Ivc]: 'VCI',
  [Structure.Svc]: 'VCS',
  [Structure.PulmonaryArtery]: 'AP',
  [Structure.PulmonaryVein]: 'V. pulm',
  [Structure.CoronarySinus]: 'SC',
  [Structure.HepaticVein]: 'V. hep',
  [Structure.Liver]: 'Hígado',
  [Structure.Diaphragm]: 'Diafragma',
  [Structure.ModeratorBand]: 'Banda mod.',
  [Structure.LaAppendage]: 'Orejuela',
};

const LV_WALL_STRUCTURES = new Set<Structure>([
  Structure.LvWallSeptal,
  Structure.LvWallLateral,
  Structure.LvWallAnterior,
  Structure.LvWallInferior,
  Structure.LvApex,
]);

const anatomyAnchors = new SegmentAnchors();

/**
 * Draws stabilized, collision-free anatomical structure pills directly onto the 2D ultrasound sector.
 * Averages label anchors over one beat while the probe rests (matching CutMapView / SegmentAnchors)
 * and suppresses LV wall labels whenever the 17/16-segment overlay is simultaneously active.
 */
export function drawAnatomyLabelsLayer(
  ctx: CanvasRenderingContext2D,
  hud: SimOutput,
  st: SimStore,
): void {
  const m = hud.sector;
  const p = hud.polar;
  if (hud.structure.length !== p.lines * p.samples || m.width < 8 || m.height < 8) return;
  const w = Math.round(m.width),
    h = Math.round(m.height);
  const key = sectorGeometryKey(hud, w, h);
  const lut = getSharedSectorLut(hud, w, h, key);
  const segOn = imageSegmentsOn(st);
  const pr = hud.frozen && hud.acquisition ? hud.acquisition.probe : st.probe;
  const layoutKey = `${key}|anat|${segOn ? 1 : 0}|${st.caseId}|${[pr.u, pr.v, pr.rotationDeg, pr.tiltDeg, pr.rockDeg, pr.pressure].map((v) => v.toFixed(3)).join(',')}`;

  ctx.save();
  ctx.beginPath();
  ctx.rect(m.x, m.y, m.width, m.height);
  ctx.clip();
  ctx.font = canvasFont(11, 'ui', 650);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const measure = (text: string) => ({ w: ctx.measureText(text).width + 12, h: 18 });
  const labelOf = (id: Structure): string | undefined => {
    if (segOn && LV_WALL_STRUCTURES.has(id)) return undefined;
    return IMAGE_ANATOMY_LABELS[id];
  };

  if (anatomyAnchors.wants(layoutKey, hud.frameId)) {
    const stats = computeStructureStats(lut, hud.structure, w);
    const shown = new Set(anatomyAnchors.view().labels.map((l) => l.id));
    const labels = placeLabels(
      stats,
      lut,
      hud.structure,
      w,
      Math.max(120, 0.0022 * w * h),
      measure,
      shown,
      labelOf,
    );
    anatomyAnchors.add(layoutKey, hud, labels, null);
  }

  const labelsShown = withoutOverlaps(anatomyAnchors.view().labels, measure);
  const padX = 5;
  const pillH = 16;
  const r = 4;
  for (const l of labelsShown) {
    const tw = ctx.measureText(l.text).width;
    const pw = tw + padX * 2;
    const x = Math.max(m.x + pw / 2 + 4, Math.min(m.x + m.width - pw / 2 - 4, m.x + l.x));
    const y = Math.max(m.y + pillH / 2 + 4, Math.min(m.y + m.height - pillH / 2 - 4, m.y + l.y));
    const x0 = x - pw / 2;
    const y0 = y - pillH / 2;
    ctx.beginPath();
    ctx.roundRect(x0, y0, pw, pillH, r);
    ctx.fillStyle = 'rgba(8, 15, 28, 0.72)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(92, 200, 255, 0.45)';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = '#e6f4ff';
    ctx.fillText(l.text, x, y + 0.5);
  }
  ctx.restore();
}
