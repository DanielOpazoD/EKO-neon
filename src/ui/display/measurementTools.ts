import { frameBus } from '@/app/frameBus';
import { evaluateCapture, specFor, type CaptureExtras } from '@/app/measurementCapture';
import { useHudStore, useSimStore } from '@/app/store';
import type { SimOutput, StripInfo } from '@/simulator/core/protocol';
import { reprojectGeometry } from '@/simulator/measurements/geometry';
import { discProfileFromContour, volumeFromProfileMl } from '@/simulator/measurements/simpson';
import type { Measurement } from '@/simulator/measurements/types';
import { summarizeEnvelope, type VtiSummary } from '@/simulator/measurements/vti';
import { MODALITIES } from '@/simulator/renderer/modality';
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

/** Handles clicks on the ultrasound canvas when any measurement tool is armed. */
export function handleMeasurementToolClick(
  p: { x: number; y: number },
  detail: number,
  hud: SimOutput,
  pend: PendingMeasurement,
  redrawOverlay: () => void,
): void {
  const st = useSimStore.getState();
  const m = hud.sector;
  const strip = hud.strip;
  const sectorTool = st.activeTool === 'caliper' || st.activeTool === 'simpson';
  if (!sectorTool && !strip.kind) return;
  if (sectorTool) {
    pend.points = reprojectGeometry(pend.points, pend.captureSector, m);
    pend.captureSector = { ...m };
  } else pend.captureSector = undefined;
  const spec = specFor(st.activeMeasurementId);
  const acquiredMode = hud.acquisition?.modality ?? st.modality;
  const modality = acquiredMode === 'color' ? '2d' : acquiredMode;
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
      beatIndex: hud.beatIndex,
      rrS: hud.rrS,
      label: spec ? spec.label : ms.label,
      measurementId: spec ? spec.id : null,
      technique: evaluateCapture(
        spec,
        hud,
        modality,
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
      commit(
        {
          kind: 'linear',
          label: 'Distancia',
          value: cm,
          units: 'cm',
          modality,
          geometry: [a, b],
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
        commit({
          kind: 'time',
          label: 'Tiempo de desaceleración',
          value: dt.dtMs,
          units: 'ms',
          modality,
          geometry: [a, b, dt.intercept],
          derived: { peakMps: dt.peakMps },
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
      const pts = [...pend.points];
      const prof = discProfileFromContour(pts, m.pxPerCm);
      if (prof) {
        const vol = volumeFromProfileMl(prof);
        // the foreshortening check compares against the LV length: an atrial trace has none (decision 180)
        const trueL =
          st.lvLengthCm !== null && spec?.id !== 'la-volume'
            ? spec?.phase === 'es'
              ? st.lvLengthCm - 1.2
              : st.lvLengthCm
            : null;
        commit(
          {
            kind: 'volume',
            label: 'Volumen VI (Simpson monoplano)',
            value: vol,
            units: 'mL',
            modality,
            geometry: pts,
            derived: { longAxisCm: prof.longAxisCm, discs: prof.diametersCm.length },
          },
          { contour: pts, longAxisCm: prof.longAxisCm, trueLongAxisCm: trueL },
        );
      } else pend.points = [];
      redraw();
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
      const { sorted, summary: s } = summarizeManualVti(pend.points, strip);
      commit({
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
      return;
    }
    pend.points.push(p);
    redraw();
  }
}
