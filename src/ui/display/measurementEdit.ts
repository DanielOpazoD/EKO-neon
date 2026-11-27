import { evaluateCapture, specFor } from '@/app/measurementCapture';
import { rvVolumeFromAreaMl } from '@/clinical/formulas';
import type { PhaseMarks, SimOutput } from '@/simulator/core/protocol';
import { polygonAreaCm2 } from '@/simulator/measurements/autoVolume';
import { discProfileFromContour, volumeFromProfileMl } from '@/simulator/measurements/simpson';
import type { Measurement } from '@/simulator/measurements/types';
import type { ImagingModality } from '@/simulator/renderer/types';
import { measurementGeometryOn } from './measurementPlacement';
import {
  colorPisaAliasingVelocityMps,
  computeDecelerationTime,
  computeLinearCm,
  computeMModeExcursionCm,
  isPhtMeasurementId,
  pressureHalfTimeFromDtMs,
  stripVelocityAtY,
  summarizeManualVti,
} from './measurementTools';

export interface MeasurementHandle {
  measurementId: string;
  vertexIndex: number;
  point: { x: number; y: number };
  space: 'sector' | 'strip';
}

export interface RecomputeEditOptions {
  phaseMarks: PhaseMarks | null;
  lvLengthCm: number | null;
  color: { scaleMps: number; baselineShiftMps: number };
}

export interface RecomputeEditResult {
  patch: Partial<Measurement>;
  /** Updated vertex index in `patch.geometry` (can shift when manual VTI vertices reorder along x). */
  vertexIndex: number;
}

/** Whether a saved measurement belongs to the modality currently drawn on screen. */
function measurementMatchesModality(ms: Measurement, displayedModality: ImagingModality): boolean {
  return (
    ms.modality === displayedModality ||
    (ms.modality === '2d' && displayedModality === 'color') ||
    (ms.modality === 'color' && displayedModality === '2d')
  );
}

/**
 * Number of directly draggable vertices for a saved measurement.
 * Dense auto-traced contours/envelopes return 0 (no individual point dragging), and 3-point
 * deceleration slopes expose only the peak (0) and slope (1) points because point 2 is derived.
 */
export function editableVertexCount(ms: Measurement): number {
  if (ms.kind === 'volume' && ms.geometry.length > 20) return 0;
  if (ms.kind === 'vti' && ms.geometry.length > 24) return 0;
  if (ms.kind === 'time' && ms.geometry.length === 3) return 2;
  return ms.geometry.length;
}

/** Clamp a dragged screen point inside its legal ultrasound display region (sector or strip). */
export function clampEditPoint(
  p: { x: number; y: number },
  space: 'sector' | 'strip',
  hud: Pick<SimOutput, 'width' | 'sector' | 'strip'>,
): { x: number; y: number } {
  if (space === 'strip') {
    const s = hud.strip;
    return {
      x: Math.max(s.x, Math.min(s.x + s.width, p.x)),
      y: Math.max(s.y, Math.min(s.y + s.height, p.y)),
    };
  }
  return {
    x: Math.max(2, Math.min(hud.width - 2, p.x)),
    y: Math.max(2, Math.min(Math.max(2, hud.sector.height - 2), p.y)),
  };
}

/**
 * Find the closest editable vertex of any measurement currently displayed on a frozen frame.
 * Pure & fast (zero allocations beyond reprojected coordinates), safe to run on pointer hover.
 */
export function hitTestMeasurementVertex(
  p: { x: number; y: number },
  measurements: readonly Measurement[],
  hud: SimOutput,
  liveModality: ImagingModality,
  radiusPx = 11,
): MeasurementHandle | null {
  if (!hud.frozen || !measurements.length) return null;
  const displayedModality = hud.acquisition?.modality ?? liveModality;
  const maxD2 = radiusPx * radiusPx;
  let best: MeasurementHandle | null = null;
  let bestD2 = maxD2;

  // Iterate newest-first so the topmost painted caliper wins on overlapping vertices.
  for (let i = measurements.length - 1; i >= 0; i--) {
    const ms = measurements[i]!;
    const count = editableVertexCount(ms);
    if (count === 0) continue;
    if (!measurementMatchesModality(ms, displayedModality)) continue;
    const pts = measurementGeometryOn(ms, hud);
    if (!pts || pts.length < count) continue;
    for (let vi = 0; vi < count; vi++) {
      const q = pts[vi]!;
      const d2 = (p.x - q.x) ** 2 + (p.y - q.y) ** 2;
      if (d2 <= bestD2) {
        bestD2 = d2;
        best = {
          measurementId: ms.id,
          vertexIndex: vi,
          point: q,
          space: ms.captureStrip ? 'strip' : 'sector',
        };
      }
    }
  }
  return best;
}

/**
 * Recompute a saved measurement live when one of its vertices is dragged on a frozen image.
 * Uses the exact same clinical formulas and `evaluateCapture` grading as initial placement.
 */
export function recomputeEditedMeasurement(
  ms: Measurement,
  vertexIndex: number,
  nextScreenPt: { x: number; y: number },
  hud: SimOutput,
  opts: RecomputeEditOptions,
): RecomputeEditResult | null {
  if (!hud.frozen) return null;
  const editable = editableVertexCount(ms);
  if (vertexIndex < 0 || vertexIndex >= editable) return null;
  const currentPts = measurementGeometryOn(ms, hud);
  if (!currentPts || currentPts.length < editable) return null;

  const space: 'sector' | 'strip' = ms.captureStrip ? 'strip' : 'sector';
  const clamped = clampEditPoint(nextScreenPt, space, hud);
  const pts = currentPts.map((q) => ({ ...q }));
  pts[vertexIndex] = clamped;

  const spec = specFor(ms.measurementId ?? null);
  const acquiredMode = hud.acquisition?.modality ?? ms.modality;

  // 1. Sector linear caliper (2D / Color)
  if (!ms.captureStrip && ms.kind === 'linear' && pts.length === 2) {
    const [a, b] = pts as [{ x: number; y: number }, { x: number; y: number }];
    const cm = computeLinearCm(a, b, hud.sector.pxPerCm);
    const derived =
      spec?.id === 'mr-pisa-radius'
        ? { ...ms.derived, aliasingVelocityMps: colorPisaAliasingVelocityMps(opts.color) }
        : ms.derived;
    const technique = evaluateCapture(spec, hud, acquiredMode, opts.phaseMarks, {
      segment: [a, b],
      userAssisted: ms.userAssisted,
    });
    return {
      vertexIndex,
      patch: {
        geometry: [a, b],
        captureSector: { ...hud.sector },
        value: cm,
        technique,
        ...(derived ? { derived } : {}),
      },
    };
  }

  // 2. Manual Simpson / 2D chamber volume contour
  if (!ms.captureStrip && ms.kind === 'volume' && pts.length >= 5) {
    const prof = discProfileFromContour(pts, hud.sector.pxPerCm);
    if (!prof) return null;
    const isRv = spec?.id === 'rv-edv' || spec?.id === 'rv-esv';
    const areaCm2 = polygonAreaCm2(pts, hud.sector.pxPerCm);
    const vol = isRv && areaCm2 > 0 ? rvVolumeFromAreaMl(areaCm2) : volumeFromProfileMl(prof);
    const trueL =
      opts.lvLengthCm !== null && spec?.id !== 'la-volume' && !isRv
        ? spec?.phase === 'es'
          ? opts.lvLengthCm - 1.2
          : opts.lvLengthCm
        : null;
    const technique = evaluateCapture(spec, hud, acquiredMode, opts.phaseMarks, {
      contour: pts,
      longAxisCm: prof.longAxisCm,
      trueLongAxisCm: trueL,
      userAssisted: ms.userAssisted,
    });
    return {
      vertexIndex,
      patch: {
        geometry: pts,
        captureSector: { ...hud.sector },
        value: vol,
        derived: {
          ...ms.derived,
          longAxisCm: prof.longAxisCm,
          discs: prof.diametersCm.length,
          ...(areaCm2 > 0 ? { areaCm2 } : {}),
        },
        technique,
      },
    };
  }

  // 3. M-mode linear excursion (TAPSE / M-mode depth excursion)
  if (ms.captureStrip && ms.kind === 'linear' && pts.length === 2 && hud.strip.kind === 'm-mode') {
    const [a, b] = pts as [{ x: number; y: number }, { x: number; y: number }];
    const cm = computeMModeExcursionCm(a, b, hud.strip);
    return {
      vertexIndex,
      patch: {
        geometry: [a, b],
        captureStrip: { ...hud.strip },
        value: cm,
      },
    };
  }

  // 4. Spectral velocity peak (single point)
  if (ms.captureStrip && ms.kind === 'velocity' && pts.length >= 1 && hud.strip.kind === 'spectral') {
    const pt = pts[0]!;
    const vMps = Math.abs(stripVelocityAtY(pt.y, hud.strip));
    const scale = ms.units === 'cm/s' ? 100 : 1;
    return {
      vertexIndex: 0,
      patch: {
        geometry: [pt],
        captureStrip: { ...hud.strip },
        value: vMps * scale,
        derived: { ...ms.derived, gradientMmHg: 4 * vMps * vMps },
      },
    };
  }

  // 5. Strip time interval (2 points)
  if (ms.captureStrip && ms.kind === 'time' && pts.length === 2 && hud.strip.kind !== null) {
    const [a, b] = pts as [{ x: number; y: number }, { x: number; y: number }];
    const msVal = Math.abs(a.x - b.x) * hud.strip.secondsPerColumn * 1000;
    return {
      vertexIndex,
      patch: {
        geometry: [a, b],
        captureStrip: { ...hud.strip },
        value: msVal,
      },
    };
  }

  // 6. Spectral deceleration time / PHT slope (peak + slope -> extrapolated intercept)
  if (ms.captureStrip && ms.kind === 'time' && pts.length === 3 && hud.strip.kind === 'spectral') {
    const a = pts[0]!;
    const b = pts[1]!;
    const dt = computeDecelerationTime(a, b, hud.strip);
    if (!dt) return null;
    const isPht = isPhtMeasurementId(ms.measurementId);
    return {
      vertexIndex,
      patch: {
        geometry: [a, b, dt.intercept],
        captureStrip: { ...hud.strip },
        value: isPht ? pressureHalfTimeFromDtMs(dt.dtMs) : dt.dtMs,
        derived: isPht
          ? { ...ms.derived, peakMps: dt.peakMps, dtMs: dt.dtMs }
          : { ...ms.derived, peakMps: dt.peakMps },
      },
    };
  }

  // 7. Manual VTI polyline (2..24 vertices)
  if (ms.captureStrip && ms.kind === 'vti' && pts.length >= 2 && hud.strip.kind === 'spectral') {
    const movedRef = pts[vertexIndex]!;
    const { sorted, summary } = summarizeManualVti(pts, hud.strip);
    const nextIdx = Math.max(0, sorted.indexOf(movedRef));
    return {
      vertexIndex: nextIdx,
      patch: {
        geometry: sorted,
        captureStrip: { ...hud.strip },
        value: summary.vtiCm,
        derived: {
          ...ms.derived,
          vmaxMps: summary.vmaxMps,
          meanGradientMmHg: summary.meanGradientMmHg,
          peakGradientMmHg: summary.peakGradientMmHg,
        },
      },
    };
  }

  return null;
}
