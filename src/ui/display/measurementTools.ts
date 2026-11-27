import { rvVolumeFromAreaMl } from '@/clinical/formulas';
import { evaluateTechnique } from '@/education/technique';
import { frameBus } from '@/app/frameBus';
import { evaluateCapture, specFor, type CaptureExtras } from '@/app/measurementCapture';
import { useHudStore, useSimStore } from '@/app/store';
import type { SimOutput, StripInfo } from '@/simulator/core/protocol';
import type { AutoTraceBeatResult } from '@/simulator/core/strip/autoTrace';
import type {
  AutoStripMeasureResult,
  AutoStripMeasureTarget,
} from '@/simulator/core/strip/autoMeasureStrip';
import type {
  AutoDopplerTargetResult,
  DopplerTargetId,
} from '@/simulator/measurements/autoDopplerTarget';
import type { AutoLvotGateResult } from '@/simulator/measurements/autoLvotGate';
import {
  polygonAreaCm2,
  type AutoChamberFrameSnapshot,
  type AutoChamberVolumeResult,
  type AutoChamberVolumeSuccess,
  type ChamberId,
} from '@/simulator/measurements/autoVolume';
import { reprojectGeometry } from '@/simulator/measurements/geometry';
import { discProfileFromContour, volumeFromProfileMl } from '@/simulator/measurements/simpson';
import type { Measurement } from '@/simulator/measurements/types';
import { summarizeEnvelope, type VtiSummary } from '@/simulator/measurements/vti';
import { isSpectralModality, MODALITIES } from '@/simulator/renderer/modality';
import type { SectorMapping } from '@/simulator/renderer/scanConvert';

export interface PendingMeasurement {
  points: { x: number; y: number }[];
  captureSector?: SectorMapping;
  context?: string;
}

/** Linear distance in cm between two sector pixels. */
export function computeLinearCm(
  a: { x: number; y: number },
  b: { x: number; y: number },
  pxPerCm: number,
): number {
  return Math.hypot(a.x - b.x, a.y - b.y) / pxPerCm;
}

/** Vertical excursion in cm between two M-mode strip pixels. */
export function computeMModeExcursionCm(
  a: { x: number; y: number },
  b: { x: number; y: number },
  strip: Pick<StripInfo, 'height' | 'topValue' | 'bottomValue'>,
): number {
  return (Math.abs(a.y - b.y) / strip.height) * Math.abs(strip.bottomValue - strip.topValue);
}

/** Velocity value (m/s, signed) at a y pixel on a spectral strip. */
export function stripVelocityAtY(
  y: number,
  strip: Pick<StripInfo, 'y' | 'height' | 'topValue' | 'bottomValue'>,
): number {
  return strip.topValue + ((y - strip.y) / strip.height) * (strip.bottomValue - strip.topValue);
}

/** Hatle linear-slope conversion factor from deceleration time to pressure half-time (1 − 1/√2 ≈ 0.29). */
export const PHT_FROM_DT_FACTOR = 0.29;

/** Pressure half-time (ms) derived from linear Doppler deceleration time (ms): PHT = 0.29 × DT. */
export function pressureHalfTimeFromDtMs(dtMs: number): number {
  return dtMs * PHT_FROM_DT_FACTOR;
}

/** Whether a protocol measurement id represents a pressure half-time (PHT) rather than a full deceleration time. */
export function isPhtMeasurementId(id: string | null | undefined): boolean {
  return id === 'ar-pht' || id === 'mitral-pht';
}

/** Effective Color PISA aliasing velocity (m/s) toward the shifted baseline edge: |v_N| − |baselineShift|. */
export function colorPisaAliasingVelocityMps(
  color: { scaleMps: number; baselineShiftMps: number },
): number {
  return Math.max(0.05, color.scaleMps - Math.abs(color.baselineShiftMps));
}

/**
 * Deceleration time from peak click `a` to slope click `b`, extrapolated linearly to the zero baseline.
 * Returns null if `b` is not strictly later and lower in speed than `a`.
 */
export function computeDecelerationTime(
  a: { x: number; y: number },
  b: { x: number; y: number },
  strip: Pick<StripInfo, 'y' | 'height' | 'topValue' | 'bottomValue' | 'secondsPerColumn'>,
): { dtMs: number; peakMps: number; intercept: { x: number; y: number } } | null {
  const v0 = Math.abs(stripVelocityAtY(a.y, strip));
  const v1 = Math.abs(stripVelocityAtY(b.y, strip));
  const t0 = a.x * strip.secondsPerColumn;
  const t1 = b.x * strip.secondsPerColumn;
  if (!(v0 > v1 && t1 > t0)) return null;
  const tInt = t1 + (v1 * (t1 - t0)) / (v0 - v1);
  const yBase =
    strip.y + ((0 - strip.topValue) / (strip.bottomValue - strip.topValue)) * strip.height;
  return {
    dtMs: (tInt - t0) * 1000,
    peakMps: v0,
    intercept: { x: tInt / strip.secondsPerColumn, y: yBase },
  };
}

/** Resamples a manually clicked spectral VTI polyline onto 1-pixel columns as positive velocities (m/s). */
export function resampleManualVti(
  points: readonly { x: number; y: number }[],
  strip: Pick<StripInfo, 'y' | 'height' | 'topValue' | 'bottomValue'>,
): { sorted: { x: number; y: number }[]; velocitiesMps: number[] } {
  const sorted = [...points].sort((a, b) => a.x - b.x);
  const velocitiesMps: number[] = [];
  if (sorted.length < 2) return { sorted, velocitiesMps };
  const x0 = sorted[0]!.x;
  const x1 = sorted[sorted.length - 1]!.x;
  for (let x = x0; x <= x1; x += 1) {
    let i = 0;
    while (i < sorted.length - 2 && sorted[i + 1]!.x < x) i++;
    const a = sorted[i]!;
    const b = sorted[i + 1]!;
    const t = b.x === a.x ? 0 : (x - a.x) / (b.x - a.x);
    const y = a.y + (b.y - a.y) * t;
    velocitiesMps.push(Math.abs(stripVelocityAtY(y, strip)));
  }
  return { sorted, velocitiesMps };
}

/**
 * Exact continuous piecewise-linear ASE VTI & gradient summary directly over manually clicked
 * spectral vertices `(x, y)`. Avoids 1-pixel endpoint truncation when `x1 - x0` is non-integer
 * and guarantees `VTI (cm) = ∫ |v(cm/s)| · dt(ms) / 1000` is invariant to sweep speed.
 */
export function summarizeManualVti(
  points: readonly { x: number; y: number }[],
  strip: Pick<StripInfo, 'y' | 'height' | 'topValue' | 'bottomValue' | 'secondsPerColumn'>,
): { sorted: { x: number; y: number }[]; summary: VtiSummary } {
  const sorted = [...points].sort((a, b) => a.x - b.x);
  if (sorted.length < 2) {
    return {
      sorted,
      summary: { vtiCm: 0, vmaxMps: 0, meanGradientMmHg: 0, peakGradientMmHg: 0 },
    };
  }
  const vSigned = sorted.map((pt) => stripVelocityAtY(pt.y, strip));
  let vmaxMps = 0;
  for (const v of vSigned) vmaxMps = Math.max(vmaxMps, Math.abs(v));
  let areaM = 0;
  let gradTimeSum = 0;
  let flowTimeS = 0;
  for (let i = 1; i < sorted.length; i++) {
    const dtS = (sorted[i]!.x - sorted[i - 1]!.x) * strip.secondsPerColumn;
    if (!(dtS > 0)) continue;
    const va = vSigned[i - 1]!;
    const vb = vSigned[i]!;
    // Split if the segment crosses the zero baseline so |v(t)| integrates exactly over both triangles
    if ((va > 0 && vb < 0) || (va < 0 && vb > 0)) {
      const f = Math.abs(va) / (Math.abs(va) + Math.abs(vb));
      const dt1 = dtS * f;
      const dt2 = dtS * (1 - f);
      areaM += 0.5 * Math.abs(va) * dt1 + 0.5 * Math.abs(vb) * dt2;
      gradTimeSum += (4 / 3) * (va * va * dt1 + vb * vb * dt2);
      flowTimeS += dtS;
    } else {
      const a = Math.abs(va);
      const b = Math.abs(vb);
      areaM += 0.5 * (a + b) * dtS;
      if (a > 0 || b > 0) {
        gradTimeSum += (4 / 3) * (a * a + a * b + b * b) * dtS;
        flowTimeS += dtS;
      }
    }
  }
  return {
    sorted,
    summary: {
      vtiCm: areaM * 100,
      vmaxMps,
      meanGradientMmHg: flowTimeS > 0 ? gradTimeSum / flowTimeS : 0,
      peakGradientMmHg: 4 * vmaxMps * vmaxMps,
    },
  };
}

/** Whether the armed tool measures at `p` on this frame: the sector for the caliper and Simpson, its strip for the rest. */
export function toolActsAt(
  tool: string,
  p: { x: number; y: number },
  hud: Pick<SimOutput, 'sector' | 'strip'>,
): boolean {
  const strip = hud.strip;
  const onStrip = strip.kind !== null && p.y >= strip.y;
  switch (tool) {
    case 'caliper':
    case 'simpson':
      return p.y < hud.sector.height;
    case 'velocity':
    case 'vti':
    case 'auto-vti':
    case 'slope':
      return onStrip && strip.kind === 'spectral';
    case 'tapse':
      return onStrip && strip.kind === 'm-mode';
    case 'time':
      return onStrip;
    default:
      return false;
  }
}

/**
 * Estimate which cardiac beat on a frozen Doppler/M-mode strip was clicked so measuring
 * 3–5 consecutive beats in atrial fibrillation assigns distinct beatIndex values for ASE averaging.
 */
function estimateStripBeatIndex(
  hud: SimOutput,
  geometry: { x: number; y: number }[],
): number {
  const { strip } = hud;
  if (
    !geometry.length ||
    !(strip.width > 0) ||
    !(strip.columns > 0) ||
    !(strip.secondsPerColumn > 0)
  ) {
    return hud.beatIndex;
  }
  const cx = geometry.reduce((acc, p) => acc + p.x, 0) / geometry.length;
  const col = Math.max(
    0,
    Math.min(strip.columns - 1, Math.round(((cx - strip.x) / strip.width) * strip.columns)),
  );
  const ageCols = (((strip.headColumn - col) % strip.columns) + strip.columns) % strip.columns;
  const ageS = ageCols * strip.secondsPerColumn;
  const rr = Math.max(0.35, hud.rrS);
  const elapsedInBeatS = Math.max(0, Math.min(1, hud.phase)) * rr;
  if (ageS <= elapsedInBeatS + 0.08) return hud.beatIndex;
  const beatsAgo = 1 + Math.floor((ageS - elapsedInBeatS) / rr);
  return Math.max(0, hud.beatIndex - beatsAgo);
}

/** Handles clicks on the ultrasound canvas when any measurement tool is armed. */
export function handleMeasurementToolClick(
  p: { x: number; y: number },
  detail: number,
  hud: SimOutput,
  pend: PendingMeasurement,
  redrawOverlay: () => void,
): void {
  const st = useSimStore.getState();
  // a click where the armed tool does not measure does nothing, as before
  if (!toolActsAt(st.activeTool, p, hud)) return;
  // calipers go on a frozen image, as on a scanner: a click on the live image freezes it, and the points go on the frame
  // it froze on. Live, each end of a two-click tool landed on another frame or another sweep, and the measurement
  // recorded the frame and phase of the second
  if (!hud.frozen) {
    if (!st.frozen) st.toggleFreeze();
    return;
  }
  const m = hud.sector;
  const strip = hud.strip;
  const sectorTool = st.activeTool === 'caliper' || st.activeTool === 'simpson';
  if (sectorTool) {
    pend.points = reprojectGeometry(pend.points, pend.captureSector, m);
    pend.captureSector = { ...m };
  } else pend.captureSector = undefined;
  const spec = specFor(st.activeMeasurementId);
  const acquiredMode = hud.acquisition?.modality ?? st.modality;
  // Preserve 'color' for Color-specific protocol specs (vena contracta, PISA); collapse to '2d' for 2D/free measurements.
  const modality =
    acquiredMode === 'color' && spec?.modalities.includes('color')
      ? 'color'
      : acquiredMode === 'color'
        ? '2d'
        : acquiredMode;
  const redraw = () => {
    redrawOverlay();
    useHudStore.getState().setHud({ ...hud });
  };
  const commit = (
    ms: Omit<
      Measurement,
      | 'id'
      | 'createdAt'
      | 'frameId'
      | 'phase'
      | 'timeS'
      | 'sourceViewId'
      | 'viewScore'
      | 'imageQualityScore'
      | 'userAssisted'
      | 'referenceGuidelineIds'
      | 'measurementId'
      | 'technique'
    >,
    extras: CaptureExtras = {},
  ) => {
    // pending points go before the store update: the store subscription redraws the overlay synchronously, and
    // with the points still pending their cyan crosses covered the measurement's yellow ones until the next frozen
    // frame (80 ms), which the caliper E2E read as a crosshair drawn elsewhere
    pend.points = [];
    st.addMeasurement({
      ...ms,
      captureSector: sectorTool ? { ...m } : undefined,
      captureStrip: sectorTool ? undefined : { ...strip },
      acquisition: hud.acquisition,
      beatIndex:
        ms.beatIndex ?? (sectorTool ? hud.beatIndex : estimateStripBeatIndex(hud, ms.geometry)),
      rrS: hud.rrS,
      label: spec ? spec.label : ms.label,
      measurementId: spec ? spec.id : null,
      technique: evaluateCapture(
        spec,
        hud,
        acquiredMode,
        hud.acquisition?.phaseMarks ?? st.phaseMarks,
        extras,
      ),
      id: `m${Date.now().toString(36)}${Math.floor(Math.random() * 1e4)}`,
      createdAt: new Date().toISOString(),
      frameId: hud.frameId,
      phase: hud.phase,
      timeS: hud.timeS,
      sourceViewId: hud.view?.bestViewId ?? null,
      viewScore: hud.view?.score ?? null,
      imageQualityScore: hud.view ? Math.round(hud.view.components.gain * 100) : null,
      userAssisted: extras.userAssisted ?? false,
      referenceGuidelineIds: spec ? spec.referenceIds : ['ase-tte-2019'],
    });
    if (spec) st.setActiveMeasurement(null);
  };
  const velocityUnits = MODALITIES[modality].velocityUnits;
  const velocityScale = velocityUnits === 'cm/s' ? 100 : 1;

  if (st.activeTool === 'caliper') {
    if (p.y >= m.height) return;
    pend.points.push(p);
    if (pend.points.length === 2) {
      const [a, b] = pend.points as [{ x: number; y: number }, { x: number; y: number }];
      const cm = computeLinearCm(a, b, m.pxPerCm);
      const colorAcq = hud.acquisition?.color ?? st.color;
      const derived =
        spec?.id === 'mr-pisa-radius'
          ? { aliasingVelocityMps: colorPisaAliasingVelocityMps(colorAcq) }
          : undefined;
      commit(
        {
          kind: 'linear',
          label: 'Distancia',
          value: cm,
          units: 'cm',
          modality,
          geometry: [a, b],
          ...(derived ? { derived } : {}),
        },
        { segment: [a, b] },
      );
    }
    redraw();
    return;
  }
  if (st.activeTool === 'velocity') {
    if (strip.kind !== 'spectral' || p.y < strip.y) return;
    const v = Math.abs(stripVelocityAtY(p.y, strip));
    commit({
      kind: 'velocity',
      label: 'Velocidad',
      value: v * velocityScale,
      units: velocityUnits,
      modality,
      geometry: [p],
      derived: { gradientMmHg: 4 * v * v },
    });
    return;
  }
  if (st.activeTool === 'time') {
    if (strip.kind === null || p.y < strip.y) return;
    pend.points.push(p);
    if (pend.points.length === 2) {
      const [a, b] = pend.points as [{ x: number; y: number }, { x: number; y: number }];
      const ms = Math.abs(a.x - b.x) * strip.secondsPerColumn * 1000;
      commit({
        kind: 'time',
        label: 'Tiempo',
        value: ms,
        units: 'ms',
        modality,
        geometry: [a, b],
      });
    }
    redraw();
    return;
  }
  if (st.activeTool === 'slope') {
    // deceleration time: first click on the peak, second on the slope; extrapolated to the baseline
    if (strip.kind !== 'spectral' || p.y < strip.y) return;
    pend.points.push(p);
    if (pend.points.length === 2) {
      const [a, b] = pend.points as [{ x: number; y: number }, { x: number; y: number }];
      const dt = computeDecelerationTime(a, b, strip);
      if (dt) {
        const isPht = isPhtMeasurementId(spec?.id);
        commit({
          kind: 'time',
          label: isPht ? 'Tiempo de hemipresión (PHT)' : 'Tiempo de desaceleración',
          value: isPht ? pressureHalfTimeFromDtMs(dt.dtMs) : dt.dtMs,
          units: 'ms',
          modality,
          geometry: [a, b, dt.intercept],
          derived: isPht ? { peakMps: dt.peakMps, dtMs: dt.dtMs } : { peakMps: dt.peakMps },
        });
      } else pend.points = [];
    }
    redraw();
    return;
  }
  if (st.activeTool === 'tapse') {
    if (strip.kind !== 'm-mode' || p.y < strip.y) return;
    pend.points.push(p);
    if (pend.points.length === 2) {
      const [a, b] = pend.points as [{ x: number; y: number }, { x: number; y: number }];
      const cm = computeMModeExcursionCm(a, b, strip);
      commit({
        kind: 'linear',
        label: 'Excursión (modo M)',
        value: cm,
        units: 'cm',
        modality,
        geometry: [a, b],
      });
    }
    redraw();
    return;
  }
  if (st.activeTool === 'simpson') {
    if (p.y >= m.height) return;
    if (detail >= 2 && pend.points.length >= 5) {
      finalizePendingMeasurement(hud, pend, redrawOverlay);
      return;
    }
    pend.points.push(p);
    redraw();
    return;
  }
  if (st.activeTool === 'auto-vti') {
    if (strip.kind !== 'spectral' || p.y < strip.y) return;
    pend.points.push(p);
    if (pend.points.length === 2) {
      const [a, b] = pend.points as [{ x: number; y: number }, { x: number; y: number }];
      pend.points = [];
      const x0 = Math.min(a.x, b.x),
        x1 = Math.max(a.x, b.x);
      void frameBus
        .request({
          kind: 'autoTrace',
          x0: Math.round(x0 - strip.x),
          x1: Math.round(x1 - strip.x),
        })
        .then((res) => {
          if (!res || res.kind !== 'autoTrace' || res.velocitiesMps.length < 2) return;
          const s = summarizeEnvelope(res.velocitiesMps, res.secondsPerColumn);
          const geometry = res.velocitiesMps.map((v, i) => ({
            x: strip.x + res.x0 + i * res.pixelsPerColumn,
            y:
              strip.y +
              ((v - strip.topValue) / (strip.bottomValue - strip.topValue)) * strip.height,
          }));
          commit(
            {
              kind: 'vti',
              label: 'VTI auto',
              value: s.vtiCm,
              units: 'cm',
              modality,
              geometry,
              derived: {
                vmaxMps: s.vmaxMps,
                meanGradientMmHg: s.meanGradientMmHg,
                peakGradientMmHg: s.peakGradientMmHg,
              },
            },
            { userAssisted: true },
          );
          redraw();
        })
        .catch((e: unknown) => {
          // a dead or reloaded worker rejects (client.ts): the measurement is simply not taken
          console.warn('auto-trace request failed', e instanceof Error ? e.message : e);
        });
    }
    redraw();
    return;
  }
  if (st.activeTool === 'vti') {
    if (strip.kind !== 'spectral' || p.y < strip.y) return;
    if (detail >= 2 && pend.points.length >= 2) {
      finalizePendingMeasurement(hud, pend, redrawOverlay);
      return;
    }
    pend.points.push(p);
    redraw();
  }
}

/**
 * Commit an open multi-point trace (`simpson` ≥ 5 points, or manual `vti` ≥ 2 points) from
 * double-click, the Enter key, or the "Finalizar" button through one single code path.
 */
export function finalizePendingMeasurement(
  hud: SimOutput,
  pend: PendingMeasurement,
  redrawOverlay: () => void,
): boolean {
  const st = useSimStore.getState();
  if (!hud.frozen) return false;
  const spec = specFor(st.activeMeasurementId);
  const acquiredMode = hud.acquisition?.modality ?? st.modality;
  const modality =
    acquiredMode === 'color' && spec?.modalities.includes('color')
      ? 'color'
      : acquiredMode === 'color'
        ? '2d'
        : acquiredMode;
  const m = hud.sector;
  const strip = hud.strip;

  const commitMulti = (
    sectorTool: boolean,
    ms: Omit<
      Measurement,
      | 'id'
      | 'createdAt'
      | 'frameId'
      | 'phase'
      | 'timeS'
      | 'sourceViewId'
      | 'viewScore'
      | 'imageQualityScore'
      | 'userAssisted'
      | 'referenceGuidelineIds'
      | 'measurementId'
      | 'technique'
    >,
    extras: CaptureExtras = {},
  ) => {
    pend.points = [];
    st.addMeasurement({
      ...ms,
      captureSector: sectorTool ? { ...m } : undefined,
      captureStrip: sectorTool ? undefined : { ...strip },
      acquisition: hud.acquisition,
      beatIndex:
        ms.beatIndex ?? (sectorTool ? hud.beatIndex : estimateStripBeatIndex(hud, ms.geometry)),
      rrS: hud.rrS,
      label: spec ? spec.label : ms.label,
      measurementId: spec ? spec.id : null,
      technique: evaluateCapture(
        spec,
        hud,
        acquiredMode,
        hud.acquisition?.phaseMarks ?? st.phaseMarks,
        extras,
      ),
      id: `m${Date.now().toString(36)}${Math.floor(Math.random() * 1e4)}`,
      createdAt: new Date().toISOString(),
      frameId: hud.frameId,
      phase: hud.phase,
      timeS: hud.timeS,
      sourceViewId: hud.view?.bestViewId ?? null,
      viewScore: hud.view?.score ?? null,
      imageQualityScore: hud.view ? Math.round(hud.view.components.gain * 100) : null,
      userAssisted: extras.userAssisted ?? false,
      referenceGuidelineIds: spec ? spec.referenceIds : ['ase-tte-2019'],
    });
    if (spec) st.setActiveMeasurement(null);
    redrawOverlay();
    useHudStore.getState().setHud({ ...hud });
  };

  if (st.activeTool === 'simpson' && pend.points.length >= 5) {
    const pts = reprojectGeometry([...pend.points], pend.captureSector, m);
    const prof = discProfileFromContour(pts, m.pxPerCm);
    if (!prof) {
      pend.points = [];
      redrawOverlay();
      return false;
    }
    const isRv = spec?.id === 'rv-edv' || spec?.id === 'rv-esv';
    const areaCm2 = polygonAreaCm2(pts, m.pxPerCm);
    const vol = isRv && areaCm2 > 0 ? rvVolumeFromAreaMl(areaCm2) : volumeFromProfileMl(prof);
    const trueL =
      st.lvLengthCm !== null && spec?.id !== 'la-volume' && !isRv
        ? spec?.phase === 'es'
          ? st.lvLengthCm - 1.2
          : st.lvLengthCm
        : null;
    commitMulti(
      true,
      {
        kind: 'volume',
        label: isRv ? 'Volumen VD (2D)' : 'Volumen VI (Simpson monoplano)',
        value: vol,
        units: 'mL',
        modality,
        geometry: pts,
        derived: {
          longAxisCm: prof.longAxisCm,
          discs: prof.diametersCm.length,
          ...(areaCm2 > 0 ? { areaCm2 } : {}),
        },
      },
      { contour: pts, longAxisCm: prof.longAxisCm, trueLongAxisCm: trueL },
    );
    return true;
  }

  if (st.activeTool === 'vti' && strip.kind === 'spectral' && pend.points.length >= 2) {
    const { sorted, summary: s } = summarizeManualVti(pend.points, strip);
    commitMulti(false, {
      kind: 'vti',
      label: 'VTI',
      value: s.vtiCm,
      units: 'cm',
      modality,
      geometry: sorted,
      derived: {
        vmaxMps: s.vmaxMps,
        meanGradientMmHg: s.meanGradientMmHg,
        peakGradientMmHg: s.peakGradientMmHg,
      },
    });
    return true;
  }

  return false;
}

/**
 * Commit an automatic ED + ES LV or RV volume measurement pair to the store, freezing on the ED
 * traced frame so the endocardial contour is immediately visible for visual inspection.
 */
export function commitAutoChamberVolume(res: AutoChamberVolumeSuccess): void {
  const st = useSimStore.getState();
  const isLv = res.chamber === 'lv';
  const edSpec = specFor(isLv ? 'lv-edv-simpson' : 'rv-edv');
  const esSpec = specFor(isLv ? 'lv-esv-simpson' : 'rv-esv');
  const now = new Date().toISOString();
  const baseId = Date.now().toString(36);

  const buildRecord = (
    snap: AutoChamberFrameSnapshot,
    spec: ReturnType<typeof specFor>,
    phaseTag: 'ed' | 'es',
  ): Measurement => {
    const acquiredMode = snap.acquisition.modality;
    const modality = acquiredMode === 'color' ? '2d' : acquiredMode;
    const trueL =
      isLv && st.lvLengthCm !== null
        ? phaseTag === 'es'
          ? st.lvLengthCm - 1.2
          : st.lvLengthCm
        : undefined;
    const technique = spec
      ? evaluateTechnique(spec, {
          modality: acquiredMode,
          viewId: snap.viewId,
          viewScore: snap.viewScore,
          phase: snap.phase,
          phaseMarks: snap.acquisition.phaseMarks ?? st.phaseMarks,
          gate: null,
          segmentStructures: snap.interiorStructures,
          cutByDepth: snap.cutByDepth,
          longAxisCm: snap.longAxisCm,
          trueLongAxisCm: trueL,
          userAssisted: true,
        })
      : null;

    return {
      id: `m${baseId}${phaseTag}${Math.floor(Math.random() * 1e3)}`,
      kind: 'volume',
      measurementId: spec ? spec.id : null,
      technique,
      label: spec ? spec.label : isLv ? 'Volumen VI auto' : 'Volumen VD auto',
      value: snap.volumeMl,
      units: 'mL',
      modality,
      sourceViewId: snap.viewId,
      viewScore: snap.viewScore,
      frameId: snap.frameId,
      phase: snap.phase,
      timeS: snap.timeS,
      geometry: snap.contourPx,
      captureSector: { ...snap.sector },
      acquisition: snap.acquisition,
      beatIndex: snap.beatIndex,
      rrS: res.rrS,
      derived: {
        areaCm2: snap.areaCm2,
        longAxisCm: snap.longAxisCm,
        efPct: res.efPct,
        ...(isLv ? {} : { facPct: res.facPct }),
      },
      imageQualityScore: snap.gainScore,
      userAssisted: true,
      referenceGuidelineIds: spec ? spec.referenceIds : ['ase-eacvi-chamber-2015'],
      createdAt: now,
    };
  };

  if (!st.frozen) st.toggleFreeze();
  st.addMeasurement(buildRecord(res.ed, edSpec, 'ed'));
  st.addMeasurement(buildRecord(res.es, esSpec, 'es'));
  // Jump strictly by ED frameId so the overlaid contour stays aligned even if live frames elapsed during the RPC.
  useSimStore.getState().jumpCinePhase('ed', res.ed.frameId);
}

/** Request automatic 2D LV or RV volume & ejection fraction calculation over the current cine loop. */
export async function runAutoChamberVolume(chamber: ChamberId): Promise<AutoChamberVolumeResult> {
  const res = await frameBus.request({ kind: 'autoChamberVolume', chamber });
  if (!res || res.kind !== 'autoChamberVolume') {
    return {
      ok: false,
      chamber,
      reason: 'no-frame',
      message: 'El simulador no respondió a la solicitud de volumen automático.',
    };
  }
  if (res.result.ok) commitAutoChamberVolume(res.result);
  return res.result;
}

/**
 * Locate the LVOT PW sample volume from the rendered 2D polar frame (~0.5 cm proximal to the AV),
 * snap the cursor/gate there, and switch to live PW Doppler so the LVOT spectrum sweeps immediately.
 */
export async function runAutoLvotGate(): Promise<AutoLvotGateResult> {
  const res = await frameBus.request({ kind: 'autoLvotGate' });
  if (!res || res.kind !== 'autoLvotGate') {
    return {
      ok: false,
      reason: 'no-frame',
      message: 'El simulador no respondió al intento de colocar el gate en el TSVI.',
    };
  }
  if (res.result.ok) {
    const st = useSimStore.getState();
    st.setCursor(res.result.thetaRad, res.result.rCm);
    if (st.modality !== 'pw') st.setModality('pw');
    else if (st.frozen) st.toggleFreeze();
    // Shift the PW velocity window downward (`baselineShiftMps = -0.25`, raising the zero line)
    // so negative apical LVOT ejection (~1.0–1.2 m/s away from probe) clears Nyquist without aliasing.
    const after = useSimStore.getState();
    const maxShift = Math.round(after.spectral.scaleMps * 80) / 100;
    const shift = Math.max(-maxShift, Math.min(maxShift, -0.25));
    if (after.spectral.baselineShiftMps > shift + 0.02) {
      after.setSpectral({ baselineShiftMps: shift });
    }
  }
  return res.result;
}

/**
 * Pure-2D guided placement for clinical Doppler and M-mode targets (LVOT PW, Mitral E/A PW,
 * TDI septal/lateral e′, TAPSE M-mode, TR CW): snaps cursor/gate from the rendered 2D polar frame,
 * activates the target modality live, and sets a sensible baseline shift for aliasing headroom.
 */
export async function runAutoDopplerTarget(
  target: DopplerTargetId,
): Promise<AutoDopplerTargetResult> {
  const res = await frameBus.request({ kind: 'autoDopplerTarget', target });
  if (!res || res.kind !== 'autoDopplerTarget') {
    return {
      ok: false,
      target,
      reason: 'no-frame',
      message: 'El simulador no respondió al intento de posicionar el cursor.',
    };
  }
  if (res.result.ok) {
    const r = res.result;
    const st = useSimStore.getState();
    st.setCursor(r.thetaRad, r.rCm);
    if (st.modality !== r.modality) st.setModality(r.modality);
    else if (st.frozen) st.toggleFreeze();
    if (r.recommendedBaselineShiftMps !== undefined && isSpectralModality(r.modality)) {
      const after = useSimStore.getState();
      const maxShift = Math.round(after.spectral.scaleMps * 80) / 100;
      const shift = Math.max(-maxShift, Math.min(maxShift, r.recommendedBaselineShiftMps));
      after.setSpectral({ baselineShiftMps: shift });
    }
  }
  return res.result;
}

/**
 * Freeze the current spectral strip and automatically trace the newest complete systolic beat
 * envelope (LVOT VTI in PW, or AV VTI if in CW), committing a full protocol measurement.
 */
export async function runAutoSpectralVti(
  measurementId: 'lvot-vti' | 'av-vti' = 'lvot-vti',
): Promise<AutoTraceBeatResult> {
  const st = useSimStore.getState();
  const initialHud = useHudStore.getState().hud;
  const acquiredMod = initialHud?.acquisition?.modality ?? st.modality;
  const expectedMod = measurementId === 'av-vti' ? 'cw' : 'pw';
  if (acquiredMod !== expectedMod) {
    return {
      ok: false,
      reason: 'wrong-modality',
      message:
        measurementId === 'lvot-vti'
          ? 'Activa Doppler pulsado (PW) sobre el TSVI para calcular el VTI automático.'
          : 'Activa Doppler continuo (CW) sobre la válvula aórtica para calcular el VTI/Vmáx automático.',
    };
  }
  // Freeze first and yield one animation frame so the worker's strip head and HUD snapshot match exactly.
  if (!st.frozen) {
    st.toggleFreeze();
    await new Promise<void>((resolve) => {
      if (typeof requestAnimationFrame === 'function') {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      } else {
        setTimeout(resolve, 25);
      }
    });
  }
  const res = await frameBus.request({ kind: 'autoTraceBeat' });
  if (!res || res.kind !== 'autoTraceBeat') {
    return {
      ok: false,
      reason: 'incomplete-beat',
      message: 'El simulador no respondió al cálculo automático de VTI.',
    };
  }
  if (!res.result.ok) return res.result;
  const traced = res.result;

  const hud = useHudStore.getState().hud;
  if (!hud || hud.strip.kind !== 'spectral') {
    return {
      ok: false,
      reason: 'wrong-modality',
      message: 'No hay una tira espectral visible para registrar el VTI.',
    };
  }
  const strip = hud.strip;
  const spec = specFor(measurementId);
  const modeNow = hud.acquisition?.modality ?? useSimStore.getState().modality;
  const geometry = traced.velocitiesMps.map((v, i) => ({
    x: strip.x + traced.x0 + i * traced.pixelsPerColumn,
    y:
      strip.y +
      ((v - strip.topValue) / (strip.bottomValue - strip.topValue)) * strip.height,
  }));
  const stateNow = useSimStore.getState();
  const baseId = Date.now().toString(36);
  const nowIso = new Date().toISOString();
  stateNow.addMeasurement({
    id: `m${baseId}${Math.floor(Math.random() * 1e4)}`,
    kind: 'vti',
    label: spec ? spec.label : 'VTI auto',
    measurementId: spec ? spec.id : null,
    technique: evaluateCapture(
      spec,
      hud,
      modeNow,
      hud.acquisition?.phaseMarks ?? stateNow.phaseMarks,
      { userAssisted: true },
    ),
    value: traced.vtiCm,
    units: 'cm',
    modality: modeNow,
    geometry,
    captureStrip: { ...strip },
    acquisition: hud.acquisition,
    beatIndex: traced.beatIndex,
    rrS: hud.rrS,
    derived: {
      vmaxMps: traced.vmaxMps,
      meanGradientMmHg: traced.meanGradientMmHg,
      peakGradientMmHg: traced.peakGradientMmHg,
    },
    frameId: hud.frameId,
    phase: hud.phase,
    timeS: hud.timeS,
    sourceViewId: hud.view?.bestViewId ?? null,
    viewScore: hud.view?.score ?? null,
    imageQualityScore: hud.view ? Math.round(hud.view.components.gain * 100) : null,
    userAssisted: true,
    referenceGuidelineIds: spec ? spec.referenceIds : ['ase-tte-2019'],
    createdAt: nowIso,
  });

  // For transaortic CW auto-VTI, also record the drag-editable AV Vmax peak marker so continuity AVA,
  // velocity ratio (DVI), and peak transaortic gradient populate in one clinical step.
  if (measurementId === 'av-vti' && modeNow === 'cw' && geometry.length > 0) {
    let peakIdx = 0;
    let peakAbs = -1;
    for (let i = 0; i < traced.velocitiesMps.length; i++) {
      const av = Math.abs(traced.velocitiesMps[i]!);
      if (av > peakAbs) {
        peakAbs = av;
        peakIdx = i;
      }
    }
    const peakPt = geometry[peakIdx]!;
    const vmaxSpec = specFor('av-vmax');
    stateNow.addMeasurement({
      id: `m${baseId}v${Math.floor(Math.random() * 1e3)}`,
      kind: 'velocity',
      label: vmaxSpec ? vmaxSpec.label : 'Vmax VAo',
      measurementId: vmaxSpec ? vmaxSpec.id : 'av-vmax',
      technique: evaluateCapture(
        vmaxSpec,
        hud,
        modeNow,
        hud.acquisition?.phaseMarks ?? stateNow.phaseMarks,
        { userAssisted: true },
      ),
      value: traced.vmaxMps,
      units: 'm/s',
      modality: modeNow,
      geometry: [peakPt],
      captureStrip: { ...strip },
      acquisition: hud.acquisition,
      beatIndex: traced.beatIndex,
      rrS: hud.rrS,
      derived: { gradientMmHg: traced.peakGradientMmHg },
      frameId: hud.frameId,
      phase: hud.phase,
      timeS: hud.timeS,
      sourceViewId: hud.view?.bestViewId ?? null,
      viewScore: hud.view?.score ?? null,
      imageQualityScore: hud.view ? Math.round(hud.view.components.gain * 100) : null,
      userAssisted: true,
      referenceGuidelineIds: vmaxSpec ? vmaxSpec.referenceIds : ['ase-eacvi-aortic-stenosis-2017'],
      createdAt: nowIso,
    });
  }
  return traced;
}

/**
 * Freeze the active Doppler or M-mode strip and automatically measure Mitral E/A/TD, TDI e′,
 * TR Vmax, or M-mode TAPSE purely from the rendered strip signal, committing drag-editable
 * protocol measurements with full technique evaluation.
 */
export async function runAutoStripMeasure(
  target: AutoStripMeasureTarget,
  opts?: { tdiSite?: 'septal' | 'lateral' },
): Promise<AutoStripMeasureResult> {
  const st = useSimStore.getState();
  const initialHud = useHudStore.getState().hud;
  const acquiredMod = initialHud?.acquisition?.modality ?? st.modality;
  const modOk =
    target === 'tapse'
      ? acquiredMod === 'm-mode' || acquiredMod === 'cmm'
      : target === 'tdi-e-prime'
        ? acquiredMod === 'tdi'
        : target === 'tr-vmax'
          ? acquiredMod === 'cw'
          : acquiredMod === 'pw' || acquiredMod === 'cw';
  if (!modOk) {
    return {
      ok: false,
      target,
      reason: 'wrong-modality',
      message:
        target === 'tapse'
          ? 'Activa Modo M sobre el anillo tricuspídeo lateral para medir TAPSE automáticamente.'
          : target === 'tdi-e-prime'
            ? 'Activa Doppler tisular (TDI) sobre el anillo mitral para medir e′ automáticamente.'
            : target === 'tr-vmax'
              ? 'Activa Doppler continuo (CW) sobre la válvula tricúspide para medir Vmax IT.'
              : 'Activa Doppler pulsado (PW) sobre las puntas mitrales para medir E / A / TD.',
    };
  }

  if (!st.frozen) {
    st.toggleFreeze();
    await new Promise<void>((resolve) => {
      if (typeof requestAnimationFrame === 'function') {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      } else {
        setTimeout(resolve, 25);
      }
    });
  }

  const res = await frameBus.request({ kind: 'autoMeasureStrip', target });
  if (!res || res.kind !== 'autoMeasureStrip') {
    return {
      ok: false,
      target,
      reason: 'incomplete-beat',
      message: 'El simulador no respondió a la medición automática de la tira.',
    };
  }
  if (!res.result.ok) return res.result;

  const hud = useHudStore.getState().hud;
  const expectedKind = target === 'tapse' ? 'm-mode' : 'spectral';
  if (!hud || hud.strip.kind !== expectedKind) {
    return {
      ok: false,
      target,
      reason: 'wrong-modality',
      message: 'No hay una tira compatible visible para registrar la medición.',
    };
  }

  const strip = hud.strip;
  const stateNow = useSimStore.getState();
  const modeNow = hud.acquisition?.modality ?? stateNow.modality;
  const pm = hud.acquisition?.phaseMarks ?? stateNow.phaseMarks;
  const baseId = Date.now().toString(36);
  const nowIso = new Date().toISOString();

  const yForStripVal = (val: number): number =>
    strip.y + ((val - strip.topValue) / (strip.bottomValue - strip.topValue)) * strip.height;

  const pushRecord = (
    suffix: string,
    specId: string,
    kind: Measurement['kind'],
    value: number,
    units: string,
    geometry: { x: number; y: number }[],
    beatIndex: number,
    derived?: Record<string, number>,
  ) => {
    const spec = specFor(specId);
    stateNow.addMeasurement({
      id: `m${baseId}${suffix}${Math.floor(Math.random() * 1e3)}`,
      kind,
      label: spec ? spec.label : specId,
      measurementId: spec ? spec.id : specId,
      technique: evaluateCapture(spec, hud, modeNow, pm, { userAssisted: true }),
      value,
      units,
      modality: modeNow,
      geometry,
      captureStrip: { ...strip },
      acquisition: hud.acquisition,
      beatIndex,
      rrS: hud.rrS,
      ...(derived ? { derived } : {}),
      frameId: hud.frameId,
      phase: hud.phase,
      timeS: hud.timeS,
      sourceViewId: hud.view?.bestViewId ?? null,
      viewScore: hud.view?.score ?? null,
      imageQualityScore: hud.view ? Math.round(hud.view.components.gain * 100) : null,
      userAssisted: true,
      referenceGuidelineIds: spec ? spec.referenceIds : ['ase-tte-2019'],
      createdAt: nowIso,
    });
  };

  const out = res.result;
  if (out.target === 'mitral-inflow') {
    const ePt = { x: strip.x + out.e.xPx, y: yForStripVal(out.e.velocityMps) };
    const eAbs = Math.abs(out.e.velocityMps);
    pushRecord('e', 'mitral-e', 'velocity', eAbs, 'm/s', [ePt], out.beatIndex, {
      gradientMmHg: 4 * eAbs * eAbs,
    });
    if (out.a) {
      const aPt = { x: strip.x + out.a.xPx, y: yForStripVal(out.a.velocityMps) };
      const aAbs = Math.abs(out.a.velocityMps);
      pushRecord('a', 'mitral-a', 'velocity', aAbs, 'm/s', [aPt], out.beatIndex, {
        gradientMmHg: 4 * aAbs * aAbs,
      });
    }
    if (out.dt) {
      const p0 = { x: strip.x + out.dt.peakXPx, y: yForStripVal(out.dt.peakVelocityMps) };
      const p1 = { x: strip.x + out.dt.slopeXPx, y: yForStripVal(out.dt.slopeVelocityMps) };
      const dtCalc = computeDecelerationTime(p0, p1, strip);
      if (dtCalc) {
        pushRecord(
          'dt',
          'mitral-dt',
          'time',
          dtCalc.dtMs,
          'ms',
          [p0, p1, dtCalc.intercept],
          out.beatIndex,
          { peakMps: dtCalc.peakMps },
        );
      }
    }
    return out;
  }

  if (out.target === 'tdi-e-prime') {
    const site = opts?.tdiSite ?? 'septal';
    const specId = site === 'lateral' ? 'e-prime-lateral' : 'e-prime-septal';
    const pt = { x: strip.x + out.ePrime.xPx, y: yForStripVal(out.ePrime.velocityMps) };
    const vMps = Math.abs(out.ePrime.velocityMps);
    pushRecord('ep', specId, 'velocity', vMps * 100, 'cm/s', [pt], out.beatIndex, {
      gradientMmHg: 4 * vMps * vMps,
    });
    return out;
  }

  if (out.target === 'tr-vmax') {
    const pt = { x: strip.x + out.vmax.xPx, y: yForStripVal(out.vmax.velocityMps) };
    const vMps = Math.abs(out.vmax.velocityMps);
    pushRecord('tr', 'tr-vmax', 'velocity', vMps, 'm/s', [pt], out.beatIndex, {
      gradientMmHg: out.vmax.peakGradientMmHg,
    });
    return out;
  }

  // target === 'tapse'
  const edPt = { x: strip.x + out.ed.xPx, y: yForStripVal(out.ed.depthCm) };
  const esPt = { x: strip.x + out.es.xPx, y: yForStripVal(out.es.depthCm) };
  pushRecord('tp', 'tapse', 'linear', out.tapseCm, 'cm', [edPt, esPt], out.beatIndex);
  return out;
}
