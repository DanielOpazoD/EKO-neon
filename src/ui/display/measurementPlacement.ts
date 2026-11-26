import type { SimOutput, StripInfo } from '@/simulator/core/protocol';
import { reprojectGeometry } from '@/simulator/measurements/geometry';
import type { Measurement } from '@/simulator/measurements/types';
import { ecgTimeAtX, ecgX, type EcgLayout } from '../ecgTrace';

type Point = { x: number; y: number };

/** The time axis of a strip: a sweep of `columns` columns ending at its newest one, as its ECG is (decision 231). */
function sweepOf(st: StripInfo): EcgLayout | null {
  if (!st.kind || !(st.columns > 0) || !(st.secondsPerColumn > 0) || st.headTimeS === undefined)
    return null;
  return {
    x0: st.x,
    width: st.width,
    y: st.y,
    height: st.height,
    spanS: st.secondsPerColumn * st.columns,
    headS: st.headTimeS,
    sweep: { headColumn: st.headColumn, columns: st.columns },
  };
}

/** Slack on the sweep window, in seconds: rounding of the head column, never a column's worth. */
const SWEEP_EPS_S = 1e-6;

/**
 * Strip points from the strip they were taken on to the strip on screen, through what they stand for: the acquisition
 * time across, the velocity (spectral) or the depth (M-mode) down. One acquisition redrawn at another size (the torso
 * hidden, the window resized) spreads the same seconds and m/s over other pixels. Null once the sweep has written over
 * any of the measured columns: the image was frozen again later, and other beats stand where they were. A point beyond
 * the strip's edge (the baseline intercept of a deceleration time) keeps its distance in time to the points on it.
 */
export function reprojectStripGeometry(
  points: readonly Point[],
  from: StripInfo,
  to: StripInfo,
): Point[] | null {
  const a = sweepOf(from);
  const b = sweepOf(to);
  const fromSpan = from.bottomValue - from.topValue;
  const toSpan = to.bottomValue - to.topValue;
  if (!a || !b || fromSpan === 0 || toSpan === 0 || !(from.height > 0) || !(to.height > 0))
    return points.map((p) => ({ ...p }));
  const onStrip = (p: Point) => p.x >= from.x && p.x <= from.x + from.width;
  const anchor = points.find(onStrip);
  if (!anchor) return null;
  for (const p of points) {
    if (!onStrip(p)) continue;
    const age = b.headS - ecgTimeAtX(p.x, a);
    if (age < -SWEEP_EPS_S || age > b.spanS + SWEEP_EPS_S) return null;
  }
  // one point on the strip is placed by its time; the others keep their distance in time to it, as the measurement's
  // own value was read (pixels × seconds per column)
  const x0 = ecgX(ecgTimeAtX(anchor.x, a), b);
  const xPerX = from.secondsPerColumn / to.secondsPerColumn;
  return points.map((p) => ({
    x: x0 + (p.x - anchor.x) * xPerX,
    y: to.y + ((from.topValue + ((p.y - from.y) / from.height) * fromSpan - to.topValue) / toSpan) * to.height,
  }));
}

/**
 * Where a saved measurement is drawn on the frame on screen, or null when that frame is not the acquisition it was
 * taken on. Calipers belong to a frozen image, as on a scanner: a sector measurement to its cine frame (reprojected in
 * centimetres if the display changed size), a strip measurement to its strip acquisition (reprojected in seconds and
 * m/s or cm). Drawn on every later frame of the modality, they stood on another view or phase, or on a sweep that had
 * written other beats under them; a strip one kept its capture pixels whatever the strip's size.
 */
export function measurementGeometryOn(ms: Measurement, hud: SimOutput): Point[] | null {
  if (!hud.frozen) return null;
  if (ms.captureStrip) {
    const id = ms.captureStrip.acquisitionId;
    if (id === undefined || hud.strip.acquisitionId !== id) return null;
    return reprojectStripGeometry(ms.geometry, ms.captureStrip, hud.strip);
  }
  if (ms.frameId !== hud.frameId) return null;
  return reprojectGeometry(ms.geometry, ms.captureSector, hud.sector);
}

/**
 * The acquisition the points of a measurement in progress belong to: the frozen cine frame for a sector tool, the
 * frozen strip at its size for a strip tool. When it changes (another frame scrubbed to, the image resumed, the strip
 * redrawn) the points start over: each end of a two-click measurement would otherwise come from another image.
 */
export function pendingAcquisitionKey(hud: SimOutput, sectorTool: boolean): string {
  if (!hud.frozen) return 'live';
  return sectorTool
    ? `frame ${hud.frameId}`
    : `strip ${hud.strip.acquisitionId ?? '-'} ${hud.width}x${hud.height}`;
}
