import { polarToPixel, type SectorMapping } from '@/simulator/renderer/scanConvert';

export type ColorBoxHandleId = 'tl' | 'tr' | 'bl' | 'br';

export interface ColorBoxBounds {
  t0: number;
  t1: number;
  r0: number;
  r1: number;
}

export interface ColorBoxCornerPixel {
  id: ColorBoxHandleId;
  x: number;
  y: number;
}

/** Minimum angular width (rad, ~9°) and radial height (cm) when resizing the Color ROI box. */
export const COLOR_BOX_MIN_SPAN_RAD = 0.16;
export const COLOR_BOX_MIN_SPAN_CM = 1.5;
export const COLOR_BOX_MIN_R_CM = 0.5;

/** Compute display pixel coordinates of the 4 polar corners of the Color Doppler ROI box. */
export function colorBoxCornerPixels(
  sector: SectorMapping,
  box: ColorBoxBounds,
): ColorBoxCornerPixel[] {
  const tl = polarToPixel(sector, box.r0, box.t0);
  const tr = polarToPixel(sector, box.r0, box.t1);
  const bl = polarToPixel(sector, box.r1, box.t0);
  const br = polarToPixel(sector, box.r1, box.t1);
  return [
    { id: 'tl', x: tl.x, y: tl.y },
    { id: 'tr', x: tr.x, y: tr.y },
    { id: 'bl', x: bl.x, y: bl.y },
    { id: 'br', x: br.x, y: br.y },
  ];
}

/** Hit-test the 4 Color ROI corner handles in display pixel space. */
export function hitTestColorBoxHandle(
  pt: { x: number; y: number },
  sector: SectorMapping,
  box: ColorBoxBounds,
  radiusPx = 12,
): ColorBoxHandleId | null {
  const corners = colorBoxCornerPixels(sector, box);
  let best: ColorBoxHandleId | null = null;
  let bestD = radiusPx;
  for (const c of corners) {
    const d = Math.hypot(pt.x - c.x, pt.y - c.y);
    if (d <= bestD) {
      bestD = d;
      best = c.id;
    }
  }
  return best;
}

/** CSS resize cursor matching each polar corner handle (accounts for L/R sector inversion). */
export function colorBoxHandleCursor(
  handle: ColorBoxHandleId,
  invertLR: boolean,
): 'nwse-resize' | 'nesw-resize' {
  const diag = handle === 'tl' || handle === 'br';
  return (invertLR ? !diag : diag) ? 'nwse-resize' : 'nesw-resize';
}

/** Resize the Color ROI box cleanly from any dragged polar corner while preserving min span and sector bounds. */
export function resizeColorBoxFromHandle(
  handle: ColorBoxHandleId,
  startBox: ColorBoxBounds,
  curPolar: { rCm: number; thetaRad: number },
  sector: Pick<SectorMapping, 'depthCm' | 'sectorRad'>,
): {
  boxThetaMinRad: number;
  boxThetaMaxRad: number;
  boxRMinCm: number;
  boxRMaxCm: number;
} {
  const half = sector.sectorRad / 2;
  const clampTheta = (th: number) => Math.max(-half, Math.min(half, th));
  const clampR = (r: number) => Math.max(COLOR_BOX_MIN_R_CM, Math.min(sector.depthCm, r));

  let t0 = startBox.t0;
  let t1 = startBox.t1;
  let r0 = startBox.r0;
  let r1 = startBox.r1;

  const isLeft = handle === 'tl' || handle === 'bl';
  const isTop = handle === 'tl' || handle === 'tr';

  if (isLeft) {
    t0 = clampTheta(Math.min(curPolar.thetaRad, t1 - COLOR_BOX_MIN_SPAN_RAD));
  } else {
    t1 = clampTheta(Math.max(curPolar.thetaRad, t0 + COLOR_BOX_MIN_SPAN_RAD));
  }
  if (isTop) {
    r0 = clampR(Math.min(curPolar.rCm, r1 - COLOR_BOX_MIN_SPAN_CM));
  } else {
    r1 = clampR(Math.max(curPolar.rCm, r0 + COLOR_BOX_MIN_SPAN_CM));
  }

  return {
    boxThetaMinRad: t0,
    boxThetaMaxRad: t1,
    boxRMinCm: r0,
    boxRMaxCm: r1,
  };
}
