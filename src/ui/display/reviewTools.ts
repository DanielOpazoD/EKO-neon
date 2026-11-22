import { frameBus } from '@/app/frameBus';
import { structureAtPixel } from '@/app/measurementCapture';
import type { ReviewMarker } from '@/app/review';
import {
  cellAt,
  lassoCells,
  lassoCentroid,
  regionSummary,
  structureComponent,
  thinLasso,
  type LassoPoint,
  type PolarMap,
} from '@/app/reviewRegion';
import { useSimStore } from '@/app/store';
import type { SimOutput } from '@/simulator/core/protocol';
import { pixelToPolar, polarToPixel } from '@/simulator/renderer/scanConvert';
import { markerScreenPosition } from './overlayDraw';

/** A review marker's id: time and a random tail, unique within a session. */
export function newMarkerId(): string {
  return `r${Date.now().toString(36)}${Math.floor(Math.random() * 1e4)}`;
}

/** The frame's polar structure map, or null when the frame carries none. */
export function polarMapOf(hud: SimOutput): PolarMap | null {
  return hud.structure.length === hud.polar.lines * hud.polar.samples
    ? { ...hud.polar, structure: hud.structure }
    : null;
}

/** The displayed grey at a polar point, read once from the image canvas for a whole region. */
export function greyReader(
  canvas: HTMLCanvasElement | null,
  hud: SimOutput,
): (q: LassoPoint) => number | null {
  const ctx = canvas?.getContext('2d');
  if (!canvas || !ctx || !canvas.width || !canvas.height) return () => null;
  const d = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  return (q) => {
    const px = polarToPixel(hud.sector, q.rCm, q.thetaRad);
    const x = Math.round(px.x),
      y = Math.round(px.y);
    if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return null;
    const i = (y * canvas.width + x) * 4;
    return Math.round((d[i]! + d[i + 1]! + d[i + 2]!) / 3);
  };
}

/** The fields of a marker that depend on where a display point lies: sector (polar, structure) or strip. */
export function reviewFieldsAt(
  p: { x: number; y: number },
  hud: SimOutput,
): Pick<
  ReviewMarker,
  'x' | 'y' | 'captureSector' | 'rCm' | 'thetaRad' | 'strip' | 'structure'
> | null {
  const m = hud.sector;
  const strip = hud.strip;
  const inSector = p.y < m.height;
  const onStrip = !inSector && strip.kind !== null && p.y >= strip.y;
  if (!inSector && !onStrip) return null;
  const polar = inSector ? pixelToPolar(m, p.x, p.y) : null;
  if (polar && (polar.rCm > m.depthCm || Math.abs(polar.thetaRad) > m.sectorRad / 2)) return null;
  return {
    x: p.x,
    y: p.y,
    captureSector: { ...m },
    rCm: polar ? polar.rCm : null,
    thetaRad: polar ? polar.thetaRad : null,
    strip:
      onStrip && strip.kind
        ? {
            kind: strip.kind,
            column: p.x - strip.x,
            value:
              strip.topValue +
              ((p.y - strip.y) / strip.height) * (strip.bottomValue - strip.topValue),
          }
        : null,
    structure: inSector ? structureAtPixel(hud, p.x, p.y) : 0,
  };
}

/** The image marker under a display point (within 12 px), if any. */
export function reviewMarkerAt(
  p: { x: number; y: number },
  hud: SimOutput,
  markers: ReviewMarker[],
): ReviewMarker | null {
  for (const mk of markers) {
    if (mk.space !== 'image') continue;
    const q = markerScreenPosition(mk, hud.sector);
    if (Math.hypot(q.x - p.x, q.y - p.y) <= 12) return mk;
  }
  return null;
}

/** The worker's classification of a marker's point (the coordinates the anatomy code reasons in). */
function askWorkerProbe(id: string, rCm: number | null, thetaRad: number | null): void {
  if (rCm === null || thetaRad === null) return;
  void frameBus
    .request({ kind: 'probePoint', rCm, thetaRad })
    .then((res) => {
      if (res && res.kind === 'probePoint')
        useSimStore.getState().updateReviewMarker(id, { point: res.point });
    })
    .catch((e: unknown) => {
      console.warn('probe point request failed', e instanceof Error ? e.message : e);
    });
}

/** Where the echo at a marker comes from (decision 340): the worker re-forms a wedge without each source. */
function askWorkerExplain(id: string, rCm: number | null, thetaRad: number | null): void {
  if (rCm === null || thetaRad === null) return;
  void frameBus
    .request({ kind: 'explainPoint', rCm, thetaRad })
    .then((res) => {
      if (res && res.kind === 'explainPoint')
        useSimStore.getState().updateReviewMarker(id, { explain: res.explain });
    })
    .catch((e: unknown) => {
      console.warn('explain point request failed', e instanceof Error ? e.message : e);
    });
}

/** A click's identification (decision 340): the piece of the structure there, the model's reading and the echo's sources. */
export function identifyReviewMarker(
  id: string,
  rCm: number | null,
  thetaRad: number | null,
  hud: SimOutput | null,
  imgCanvas: HTMLCanvasElement | null,
): void {
  const pm = hud ? polarMapOf(hud) : null;
  let region = null;
  if (hud && pm && rCm !== null && thetaRad !== null) {
    const cell = cellAt(pm, rCm, thetaRad);
    if (cell >= 0) region = regionSummary(pm, structureComponent(pm, cell), greyReader(imgCanvas, hud));
  }
  useSimStore.getState().updateReviewMarker(id, { region, explain: null });
  askWorkerProbe(id, rCm, thetaRad);
  askWorkerExplain(id, rCm, thetaRad);
}

/** A numbered marker where the click fell, with the structure the frame's own map holds there. */
export function placeReviewMarker(
  p: { x: number; y: number },
  hud: SimOutput,
  imgCanvas: HTMLCanvasElement | null,
  parentId: string | null = null,
): void {
  const st = useSimStore.getState();
  const f = reviewFieldsAt(p, hud);
  if (!f) return;
  const id = newMarkerId();
  const parent = parentId ? st.reviewMarkers.find((x) => x.id === parentId) : undefined;
  st.addReviewMarker({
    id,
    n: 0,
    parentId: parent ? parent.id : null,
    space: 'image',
    torso: null,
    group: null,
    ...f,
    frameId: hud.frameId,
    phase: hud.phase,
    modality: hud.acquisition?.modality ?? st.modality,
    point: null,
    note: '',
    category: parent ? parent.category : 'anatomia',
  });
  identifyReviewMarker(id, f.rCm, f.thetaRad, hud, imgCanvas);
}

/** Closes the pencil's outline into a marker at its centroid, with what it holds (decision 340). */
export function commitReviewLasso(
  drawn: LassoPoint[],
  hud: SimOutput | null,
  imgCanvas: HTMLCanvasElement | null,
): void {
  const pm = hud ? polarMapOf(hud) : null;
  if (!hud || !pm || drawn.length < 3) return;
  const poly = thinLasso(drawn);
  const cells = lassoCells(pm, poly);
  if (!cells.length) return;
  const c = lassoCentroid(pm, cells, poly);
  const at = polarToPixel(hud.sector, c.rCm, c.thetaRad);
  const f = reviewFieldsAt(at, hud);
  if (!f) return;
  const st = useSimStore.getState();
  const id = newMarkerId();
  st.addReviewMarker({
    id,
    n: 0,
    parentId: null,
    space: 'image',
    torso: null,
    group: null,
    ...f,
    frameId: hud.frameId,
    phase: hud.phase,
    modality: hud.acquisition?.modality ?? st.modality,
    point: null,
    note: '',
    category: 'textura',
    lasso: poly,
    region: regionSummary(pm, cells, greyReader(imgCanvas, hud)),
    explain: null,
  });
  askWorkerProbe(id, f.rCm, f.thetaRad);
  askWorkerExplain(id, f.rCm, f.thetaRad);
}
