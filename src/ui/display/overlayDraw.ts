import { markerLabel, structureLabel, type ReviewMarker } from '@/app/review';
import { imageSegmentsOn, useSegmentOrientation, type SimStore } from '@/app/store';
import { classifyDopplerAngle } from '@/education/technique';
import { resolveFrozenColorSettings } from '@/simulator/core/acquisitionKeys';
import type { SimOutput } from '@/simulator/core/protocol';
import { reprojectGeometry } from '@/simulator/measurements/geometry';
import {
  hasGate,
  isSpectralModality,
  isStripModality,
  MODALITIES,
} from '@/simulator/renderer/modality';
import { tgcAtDepth } from '@/simulator/renderer/postprocess/consolePipeline';
import { polarToPixel, type SectorMapping } from '@/simulator/renderer/scanConvert';
import { canvasFont } from '../canvasFonts';
import { ecgLayoutOf, ecgTracePoints, ecgX } from '../ecgTrace';
import { drawColorBar } from './colorBarOverlay';
import { measurementGeometryOn } from './measurementPlacement';
import type { LivePreview } from './measurementPreview';
import { drawPickedRegion, drawSegmentLayer, drawStructureOutline } from './rasterLayers';

/** Where an image marker sits on the current display: strip markers keep their pixel, sector markers reproject. */
export function markerScreenPosition(
  mk: ReviewMarker,
  m: SectorMapping,
): { x: number; y: number } {
  if (mk.strip || !mk.captureSector) return { x: mk.x, y: mk.y };
  return reprojectGeometry([{ x: mk.x, y: mk.y }], mk.captureSector, m)[0] ?? { x: mk.x, y: mk.y };
}

/**
 * The orientation marker beside the apex shows the side of the image the probe's index mark points to: the +lateral side
 * (θ > 0), screen right, and screen left once «Invertir izq/der» mirrors the image, as `polarToPixel` maps it. It was
 * drawn on the right whatever the setting, so with the image inverted it pointed at the side opposite the notch, which is
 * the one thing it is there to tell. The label sits outside the dot, on the same side.
 */
export function orientationMarker(m: SectorMapping): {
  x: number;
  y: number;
  labelX: number;
  labelAlign: 'left' | 'right';
} {
  const side = m.invertLR ? -1 : 1;
  return {
    x: m.apexX + 16 * side,
    y: m.apexY + 4,
    labelX: m.apexX + 24 * side,
    labelAlign: side > 0 ? 'left' : 'right',
  };
}

export interface SpectralAxisTick {
  valueMps: number;
  frac: number; // 0 = top of strip, 1 = bottom of strip
  label: string;
}

const ASE_CMPS_STEPS = [5, 10, 20, 25, 50, 100, 200] as const;
const ASE_MPS_STEPS = [0.2, 0.25, 0.5, 1, 1.5, 2, 3, 4] as const;

/**
 * ASE-standard spectral velocity axis ticks anchored strictly at 0 (baseline), so shifting the
 * baseline or depth-limiting PRF always keeps `0` locked on the baseline with clean clinical steps.
 */
export function computeSpectralTicks(
  topValueMps: number,
  bottomValueMps: number,
  useCmps: boolean,
): SpectralAxisTick[] {
  const spanMps = topValueMps - bottomValueMps;
  if (!(spanMps > 1e-6)) return [];
  const scale = useCmps ? 100 : 1;
  const top = topValueMps * scale;
  const bottom = bottomValueMps * scale;
  const steps = useCmps ? ASE_CMPS_STEPS : ASE_MPS_STEPS;
  let step = steps[steps.length - 1]!;
  for (const candidate of steps) {
    const count = Math.floor(top / candidate) - Math.ceil(bottom / candidate) + 1;
    if (count <= 7) {
      step = candidate;
      break;
    }
  }
  const kMin = Math.ceil(bottom / step - 1e-6);
  const kMax = Math.floor(top / step + 1e-6);
  const ticks: SpectralAxisTick[] = [];
  for (let k = kMax; k >= kMin; k--) {
    const val = k * step;
    const valueMps = val / scale;
    const frac = (topValueMps - valueMps) / spanMps;
    if (frac < -1e-4 || frac > 1 + 1e-4) continue;
    const text = useCmps
      ? `${Math.round(val)}`
      : Math.abs(val) < 1e-6
        ? '0'
        : Number.isInteger(val)
          ? `${val.toFixed(0)}`
          : `${val.toFixed(1)}`;
    ticks.push({ valueMps, frac: Math.max(0, Math.min(1, frac)), label: text });
  }
  if (ticks.length > 0) {
    ticks[0] = { ...ticks[0]!, label: `${ticks[0]!.label} ${useCmps ? 'cm/s' : 'm/s'}` };
  }
  return ticks;
}

export interface ActiveMeasurementHandle {
  measurementId: string;
  vertexIndex: number;
  dragging: boolean;
}

export function drawOverlay(
  canvas: HTMLCanvasElement,
  hud: SimOutput,
  st: SimStore,
  pending: { x: number; y: number }[],
  lassoDraft: { x: number; y: number }[] = [],
  livePreview: LivePreview | null = null,
  activeHandle: ActiveMeasurementHandle | null = null,
): void {
  const { ui, activeTool, measurements, reviewMarkers } = st;
  const {
    modality,
    color: acqColor,
    settings,
    cursorThetaRad: cursorTheta,
    gateDepthCm: gateDepth,
    spectral,
  } = hud.frozen && hud.acquisition ? hud.acquisition : st;
  const color =
    hud.frozen && hud.acquisition ? resolveFrozenColorSettings(acqColor, st.color) : acqColor;
  const dpr = window.devicePixelRatio || 1;
  const W = hud.width,
    H = hud.height;
  if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
    canvas.width = W * dpr;
    canvas.height = H * dpr;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (imageSegmentsOn(st)) drawSegmentLayer(ctx, hud, st);
  else useSegmentOrientation.getState().set(null, false);
  drawStructureOutline(ctx, hud);
  const m = hud.sector;
  ctx.font = canvasFont(11);
  ctx.textBaseline = 'middle';
  const sectorH = m.height;
  const half = m.sectorRad / 2;
  const rightX = Math.min(W - 6, m.apexX + m.depthCm * m.pxPerCm * Math.sin(half) + 10);
  ctx.strokeStyle = '#f59e0b';
  ctx.fillStyle = '#fbbf24';
  ctx.lineWidth = 1;
  // the depth scale as on a scanner (decision 200): a tick every centimetre, a longer one with its figure every five
  // when the width limits the sector its edge is 14 px from the canvas edge: the scale then grows inwards and the
  // figures sit left of the ticks, otherwise both fell off the canvas (decision 201)
  const inward = rightX + 24 > W;
  const dir = inward ? -1 : 1;
  ctx.font = canvasFont(10, 'mono');
  ctx.textAlign = inward ? 'right' : 'left';
  for (let d = 1; d <= m.depthCm + 1e-6; d += 1) {
    const y = m.apexY + d * m.pxPerCm;
    if (y > sectorH - 2) break;
    const major = d % 5 === 0;
    ctx.beginPath();
    ctx.moveTo(rightX, y);
    ctx.lineTo(rightX + dir * (major ? 8 : 4), y);
    ctx.stroke();
    if (major) ctx.fillText(String(d), rightX + dir * 11, y);
  }
  ctx.textAlign = 'left';
  ctx.font = canvasFont(11);
  const fy = m.apexY + settings.focusCm * m.pxPerCm;
  if (fy < sectorH) {
    ctx.fillStyle = '#ffc857';
    ctx.beginPath();
    ctx.moveTo(rightX - 2, fy);
    ctx.lineTo(rightX - 8, fy - 4);
    ctx.lineTo(rightX - 8, fy + 4);
    ctx.closePath();
    ctx.fill();
  }
  const mark = orientationMarker(m);
  ctx.fillStyle = '#5cc8ff';
  ctx.beginPath();
  ctx.arc(mark.x, mark.y, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#9aa4b5';
  ctx.textAlign = mark.labelAlign;
  ctx.fillText('R', mark.labelX, mark.y);
  ctx.textAlign = 'left';
  ctx.strokeStyle = 'rgba(92,200,255,0.5)';
  ctx.beginPath();
  for (let i = 0; i <= 20; i++) {
    const d = (i / 20) * m.depthCm;
    const y = m.apexY + d * m.pxPerCm;
    const db = tgcAtDepth(settings, d);
    const x = 14 + (db / 15) * 10;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  if (modality === 'color' || modality === 'cmm') {
    if (modality === 'color') {
      drawArcBox(
        ctx,
        m,
        color.boxThetaMinRad,
        color.boxThetaMaxRad,
        color.boxRMinCm,
        color.boxRMaxCm,
        'rgba(102, 224, 160, 0.88)',
      );
    }
    const rawHudBottom = canvas.parentElement?.style.getPropertyValue('--hud-tr-b');
    const hudTopClearY = rawHudBottom ? Number(rawHudBottom) || 0 : 0;
    drawColorBar(ctx, modality, color, m, W, sectorH, rightX, inward, hudTopClearY);
  }
  if (isStripModality(modality)) {
    const p0 = polarToPixel(m, 0.3, cursorTheta);
    const p1 = polarToPixel(m, m.depthCm, cursorTheta);
    ctx.strokeStyle = MODALITIES[modality].strip === 'm-mode' ? '#5cc8ff' : '#ffc857';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    ctx.lineTo(p1.x, p1.y);
    ctx.stroke();
    ctx.setLineDash([]);
    if (hasGate(modality)) {
      const g0 = polarToPixel(m, gateDepth - spectral.gateLengthCm / 2, cursorTheta);
      const g1 = polarToPixel(m, gateDepth + spectral.gateLengthCm / 2, cursorTheta);
      ctx.strokeStyle = '#ffc857';
      ctx.lineWidth = 2;
      const nx = -(g1.y - g0.y),
        ny = g1.x - g0.x;
      const nl = Math.hypot(nx, ny) || 1;
      for (const g of [g0, g1]) {
        ctx.beginPath();
        ctx.moveTo(g.x - (nx / nl) * 6, g.y - (ny / nl) * 6);
        ctx.lineTo(g.x + (nx / nl) * 6, g.y + (ny / nl) * 6);
        ctx.stroke();
      }
      ctx.lineWidth = 1;
    }
    if (
      isSpectralModality(modality) &&
      hud.gate &&
      hud.gate.flowPresent &&
      hud.gate.flowAngleDeg !== null &&
      Number.isFinite(hud.gate.flowAngleDeg)
    ) {
      const angleDeg = Math.round(hud.gate.flowAngleDeg);
      const quality = classifyDopplerAngle(hud.gate.flowAngleDeg);
      const anchorDepth = hasGate(modality) ? gateDepth : m.depthCm * 0.52;
      const anchor = polarToPixel(m, anchorDepth, cursorTheta);
      drawAngleBadge(ctx, angleDeg, quality, anchor.x, anchor.y, W, m.apexY, sectorH, cursorTheta);
    }
    const strip = hud.strip;
    if (strip.kind === 'spectral') {
      const useCmpsAxis = Boolean(strip.aliasing);
      const ticks = computeSpectralTicks(strip.topValue, strip.bottomValue, useCmpsAxis);
      ctx.font = canvasFont(10, 'mono');
      ctx.textAlign = 'right';
      for (const tick of ticks) {
        const y = strip.y + strip.height * tick.frac;
        // Right-edge calibration tick mark
        ctx.strokeStyle =
          Math.abs(tick.valueMps) < 1e-6 ? 'rgba(130,210,145,0.75)' : 'rgba(154,164,181,0.45)';
        ctx.beginPath();
        ctx.moveTo(W - 4, y);
        ctx.lineTo(W, y);
        ctx.stroke();
        const labelY = Math.min(H - 8, Math.max(strip.y + 8, y));
        ctx.lineWidth = 3;
        ctx.strokeStyle = 'rgba(0,0,0,0.82)';
        ctx.strokeText(tick.label, W - 7, labelY);
        ctx.lineWidth = 1;
        ctx.fillStyle = Math.abs(tick.valueMps) < 1e-6 ? '#b7e4c2' : '#aeb8c7';
        ctx.fillText(tick.label, W - 7, labelY);
      }
      ctx.textAlign = 'left';
      ctx.font = canvasFont(10.5, 'mono');
      const scaleHalf = (strip.topValue - strip.bottomValue) / 2;
      const wfText = strip.wallFilterMps === undefined ? '—' : Math.round(strip.wallFilterMps * 100);
      const prfKhz = strip.prfHz ? `  PRF ${(strip.prfHz / 1000).toFixed(1)} kHz` : '';
      const pulsedMeta =
        strip.aliasing && strip.gateDepthCm !== undefined
          ? `  ±${Math.round((strip.nyquistMps ?? scaleHalf) * 100)} cm/s${prfKhz}  gate ${strip.gateDepthCm.toFixed(1)} cm`
          : `  ±${scaleHalf.toFixed(2)} m/s`;
      const headerText = `${modality.toUpperCase()}  ${strip.sweepSpeedMmPerS ?? '—'} mm/s (200 ms/div)${pulsedMeta}  WF ${wfText} cm/s`;
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(0,0,0,0.8)';
      ctx.strokeText(headerText, 6, strip.y + 8);
      ctx.lineWidth = 1;
      ctx.fillStyle = '#aeb8c7';
      ctx.fillText(headerText, 6, strip.y + 8);
      // ASE time-scale calibration ticks directly on the zero baseline (200 ms minor, 1000 ms major):
      // matches clinical echo monitors and keeps the bottom ECG strip completely unobstructed.
      const cols200ms = strip.secondsPerColumn > 0 ? 0.2 / strip.secondsPerColumn : 0;
      const spanMps = strip.topValue - strip.bottomValue;
      const zeroY =
        spanMps > 1e-6
          ? strip.y + Math.max(0, Math.min(1, strip.topValue / spanMps)) * strip.height
          : strip.y + strip.height / 2;
      if (cols200ms > 4) {
        for (let k = 1; k * cols200ms < strip.width; k++) {
          const x = strip.x + k * cols200ms;
          const major = k % 5 === 0;
          const halfTick = major ? 4.5 : 2.2;
          ctx.strokeStyle = major ? 'rgba(165,220,175,0.62)' : 'rgba(130,190,145,0.34)';
          ctx.beginPath();
          ctx.moveTo(x, zeroY - halfTick);
          ctx.lineTo(x, zeroY + halfTick);
          ctx.stroke();
        }
      }
      ctx.font = canvasFont(11);
    } else if (strip.kind === 'm-mode') {
      ctx.font = canvasFont(10, 'mono');
      ctx.textAlign = 'right';
      for (let d = 0; d <= strip.bottomValue; d += 5) {
        const y = strip.y + (d / strip.bottomValue) * strip.height;
        const labelY = Math.min(H - 8, Math.max(strip.y + 8, y));
        ctx.lineWidth = 3;
        ctx.strokeStyle = 'rgba(0,0,0,0.82)';
        ctx.strokeText(`${d}`, W - 5, labelY);
        ctx.lineWidth = 1;
        ctx.fillStyle = '#aeb8c7';
        ctx.fillText(`${d}`, W - 5, labelY);
      }
      ctx.textAlign = 'left';
      ctx.font = canvasFont(10.5, 'mono');
      const mHeader = `${modality === 'cmm' ? 'COLOR M-MODE' : 'M-MODE'}  ${strip.sweepSpeedMmPerS ?? '—'} mm/s`;
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(0,0,0,0.82)';
      ctx.strokeText(mHeader, 6, strip.y + 8);
      ctx.lineWidth = 1;
      ctx.fillStyle = '#aeb8c7';
      ctx.fillText(mHeader, 6, strip.y + 8);
      ctx.font = canvasFont(11);
    } else if (hud.frozen && hud.acquisition?.stripAcquisitionId != null) {
      ctx.fillStyle = '#9aa4b5';
      ctx.fillText('La tira de esta adquisición ya no está disponible.', 12, strip.y + 24);
    }
  }
  if (ui.showPhysics) {
    ctx.strokeStyle = 'rgba(92,200,255,0.25)';
    const lines = Number(hud.stats['lines'] ?? 0);
    const n = Math.max(8, Math.min(48, Math.round(lines / 4)));
    for (let i = 0; i <= n; i++) {
      const th = -half + (m.sectorRad * i) / n;
      const p1 = polarToPixel(m, m.depthCm, th);
      ctx.beginPath();
      ctx.moveTo(m.apexX, m.apexY);
      ctx.lineTo(p1.x, p1.y);
      ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(255,200,87,0.6)';
    drawArc(ctx, m, settings.focusCm - 1.5, -half, half);
    drawArc(ctx, m, settings.focusCm + 1.5, -half, half);
    ctx.fillStyle = '#ffc857';
    ctx.fillText('zona focal', m.apexX - 30, m.apexY + (settings.focusCm + 2.2) * m.pxPerCm);
  }
  if (ui.showEcg && hud.ecg.length > 3) {
    const layout = ecgLayoutOf(hud, modality);
    const { y: ey, height: eh } = layout;
    if (hud.frozen) {
      // the cine buffer as a band on the ECG and the frame shown as a playhead (decision 190)
      const x0 = ecgX(hud.cineWindow.startS, layout);
      const x1 = ecgX(hud.cineWindow.endS, layout);
      ctx.fillStyle = 'rgba(255,200,87,0.09)';
      // on a strip's sweep the band may run past the right edge and on from the left (decision 231)
      if (x1 >= x0) ctx.fillRect(x0, ey, Math.max(1, x1 - x0), eh);
      else {
        ctx.fillRect(x0, ey, layout.x0 + layout.width - x0, eh);
        ctx.fillRect(layout.x0, ey, Math.max(1, x1 - layout.x0), eh);
      }
    }
    ctx.strokeStyle = '#57d38c';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    // one polyline, or two on a strip's sweep, split at its head (decision 231)
    for (const line of ecgTracePoints(hud.ecg, layout))
      line.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.fillStyle = '#ffc857';
    if (hud.frozen) {
      const px = Math.round(ecgX(hud.cineWindow.frameS, layout));
      ctx.fillRect(px - 1, ey - 2, 2, eh + 2);
      ctx.beginPath();
      ctx.moveTo(px - 5, ey - 7);
      ctx.lineTo(px + 5, ey - 7);
      ctx.lineTo(px, ey - 1);
      ctx.closePath();
      ctx.fill();
    } else ctx.fillRect(layout.sweep ? Math.round(ecgX(layout.headS, layout)) : W - 9, ey, 2, eh);
  }
  ctx.lineWidth = 1.5;
  // the value chips first and every caliper over them: with the 12 px minimum of the design review a chip could cover
  // a cross of another measurement, or of its own after a zoom
  const stripBaselineY =
    hud.strip.kind === 'spectral' && hud.strip.topValue > hud.strip.bottomValue
      ? hud.strip.y +
        Math.max(
          0,
          Math.min(1, hud.strip.topValue / (hud.strip.topValue - hud.strip.bottomValue)),
        ) *
          hud.strip.height
      : undefined;
  // each measurement on the acquisition it was taken on, and nowhere else (measurementGeometryOn)
  const shown: { ms: (typeof measurements)[number]; geometry: { x: number; y: number }[] }[] = [];
  for (const ms of measurements) {
    const sameModality =
      ms.modality === modality ||
      (ms.modality === '2d' && modality === 'color') ||
      (ms.modality === 'color' && modality === '2d');
    const geometry = sameModality ? measurementGeometryOn(ms, hud) : null;
    if (geometry) shown.push({ ms, geometry });
  }
  for (const { ms, geometry } of shown) {
    drawGeometry(ctx, geometry, ms.kind, '#ffc857', ms.kind === 'vti' ? stripBaselineY : undefined);
    if (activeHandle && activeHandle.measurementId === ms.id) {
      const pt = geometry[activeHandle.vertexIndex];
      if (pt) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, activeHandle.dragging ? 8.5 : 7, 0, Math.PI * 2);
        ctx.fillStyle = activeHandle.dragging
          ? 'rgba(92, 200, 255, 0.28)'
          : 'rgba(92, 200, 255, 0.16)';
        ctx.fill();
        ctx.strokeStyle = '#5cc8ff';
        ctx.lineWidth = 1.6;
        ctx.stroke();
        ctx.restore();
      }
    }
  }
  for (const { ms, geometry } of shown) {
    let anchor = geometry[geometry.length - 1];
    if (ms.kind === 'vti' && geometry.length > 2 && stripBaselineY !== undefined) {
      let bestDist = -1;
      for (const q of geometry) {
        const d = Math.abs(q.y - stripBaselineY);
        if (d > bestDist) {
          bestDist = d;
          anchor = q;
        }
      }
    }
    if (anchor) {
      const vmaxText =
        ms.kind === 'vti' && ms.derived?.vmaxMps !== undefined && ms.derived.vmaxMps > 0
          ? ms.modality === 'pw' || ms.modality === 'tdi'
            ? ` · Vmáx ${Math.round(ms.derived.vmaxMps * 100)} cm/s`
            : ` · Vmáx ${ms.derived.vmaxMps.toFixed(2)} m/s`
          : '';
      const gradText =
        ms.kind === 'vti' &&
        ms.modality === 'cw' &&
        ms.derived?.meanGradientMmHg !== undefined &&
        ms.derived.meanGradientMmHg >= 1
          ? ` · ΔPm ${Math.round(ms.derived.meanGradientMmHg)} mmHg`
          : '';
      const decimals =
        ms.kind === 'time' || ms.kind === 'volume' ? 0 : ms.kind === 'vti' || ms.units === 'cm/s' ? 1 : 2;
      drawValueChip(
        ctx,
        `${ms.label} ${ms.value.toFixed(decimals)} ${ms.units}${vmaxText}${gradText}`,
        anchor.x,
        anchor.y,
        '#ffc857',
      );
    }
  }
  if (pending.length)
    drawGeometry(
      ctx,
      pending,
      activeTool === 'caliper' || activeTool === 'tapse'
        ? 'linear'
        : activeTool === 'vti' || activeTool === 'simpson'
          ? 'vti'
          : 'time',
      '#5cc8ff',
      activeTool === 'vti' ? stripBaselineY : undefined,
    );
  if (livePreview) drawLivePreview(ctx, livePreview);
  ctx.lineWidth = 1;
  // review markers (decision 134): numbered, with the structure the model holds under each one.
  if (ui.reviewMode) {
    const tagY = (modality === '2d' || modality === 'color' ? sectorH : H) - 50;
    ctx.fillStyle = '#ff6ad5';
    ctx.fillText(
      st.reviewLinkParentId
        ? `REVISIÓN · punto secundario de ${reviewMarkers.find((x) => x.id === st.reviewLinkParentId)?.n ?? '?'} · Esc termina`
        : 'REVISIÓN',
      12,
      tagY,
    );
  }
  if (ui.reviewMode || reviewMarkers.length) {
    const sel = reviewMarkers.find((x) => x.id === st.reviewSelectedId);
    if (
      sel &&
      sel.space === 'image' &&
      !sel.lasso &&
      !sel.strip &&
      sel.rCm !== null &&
      sel.thetaRad !== null
    )
      drawPickedRegion(ctx, hud, sel);
    for (const mk of reviewMarkers) {
      if (!mk.lasso || mk.lasso.length < 3) continue;
      const on = mk.id === st.reviewSelectedId;
      ctx.beginPath();
      mk.lasso.forEach((q, i) => {
        const px = polarToPixel(m, q.rCm, q.thetaRad);
        if (i === 0) ctx.moveTo(px.x, px.y);
        else ctx.lineTo(px.x, px.y);
      });
      ctx.closePath();
      ctx.fillStyle = on ? 'rgba(255,106,213,0.18)' : 'rgba(255,106,213,0.08)';
      ctx.fill();
      ctx.strokeStyle = on ? '#ff6ad5' : 'rgba(255,106,213,0.7)';
      ctx.lineWidth = on ? 2 : 1.2;
      ctx.stroke();
    }
    if (lassoDraft.length > 1) {
      ctx.beginPath();
      lassoDraft.forEach((q, i) => (i === 0 ? ctx.moveTo(q.x, q.y) : ctx.lineTo(q.x, q.y)));
      ctx.strokeStyle = '#ff6ad5';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.lineWidth = 1;
  }
  // links from each secondary point to its primary (decision 136), under the rings
  for (const mk of reviewMarkers) {
    if (mk.space !== 'image' || !mk.parentId) continue;
    const parent = reviewMarkers.find((x) => x.id === mk.parentId);
    if (!parent || parent.space !== 'image') continue;
    const a = markerScreenPosition(parent, m);
    const b = markerScreenPosition(mk, m);
    ctx.strokeStyle = 'rgba(255,106,213,0.8)';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const mk of reviewMarkers) {
    if (mk.space !== 'image') continue;
    const q = markerScreenPosition(mk, m);
    const selected = mk.id === st.reviewSelectedId;
    const radius = mk.parentId ? 7 : 9;
    if (selected) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(q.x, q.y, 14, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#ff6ad5';
    ctx.fillStyle = 'rgba(20,20,28,0.75)';
    ctx.beginPath();
    ctx.arc(q.x, q.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#ff6ad5';
    ctx.font = `bold ${mk.parentId ? 9 : 11}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText(markerLabel(mk, reviewMarkers), q.x, q.y + 0.5);
    ctx.textAlign = 'left';
    ctx.font = canvasFont(11);
    if (!mk.strip) {
      const note = mk.note.trim();
      const label =
        (mk.lasso ? 'lazo · ' : '') +
        structureLabel(mk.point ? mk.point.structure : mk.structure) +
        (note ? ` — ${note.length > 28 ? `${note.slice(0, 27)}…` : note}` : '');
      const w = ctx.measureText(label).width + 8;
      ctx.fillStyle = 'rgba(20,20,28,0.75)';
      ctx.fillRect(q.x + 12, q.y - 8, w, 16);
      ctx.fillStyle = '#ff6ad5';
      ctx.fillText(label, q.x + 16, q.y);
    }
    ctx.lineWidth = 1;
  }
}

function drawArc(
  ctx: CanvasRenderingContext2D,
  m: SimOutput['sector'],
  rCm: number,
  t0: number,
  t1: number,
): void {
  ctx.beginPath();
  const n = 24;
  for (let i = 0; i <= n; i++) {
    const th = t0 + ((t1 - t0) * i) / n;
    const p = polarToPixel(m, rCm, th);
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  ctx.stroke();
}

function drawArcBox(
  ctx: CanvasRenderingContext2D,
  m: SimOutput['sector'],
  t0: number,
  t1: number,
  r0: number,
  r1: number,
  color: string,
): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  const n = 16;
  for (let i = 0; i <= n; i++) {
    const p = polarToPixel(m, r0, t0 + ((t1 - t0) * i) / n);
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  for (let i = n; i >= 0; i--) {
    const p = polarToPixel(m, r1, t0 + ((t1 - t0) * i) / n);
    ctx.lineTo(p.x, p.y);
  }
  ctx.closePath();
  ctx.stroke();
  ctx.lineWidth = 1;
}

function drawGeometry(
  ctx: CanvasRenderingContext2D,
  pts: { x: number; y: number }[],
  kind: string,
  color: string,
  baselineY?: number,
): void {
  if (!pts.length) return;
  const denseVti = kind === 'vti' && pts.length > 24;
  const denseVolume = kind === 'volume' && pts.length > 20;
  let drawPts = pts;
  let caliperPts = pts;
  if (denseVti) {
    let first = 0;
    let last = pts.length - 1;
    if (baselineY !== undefined) {
      const idxFirst = pts.findIndex((p) => Math.abs(p.y - baselineY) > 1.5);
      if (idxFirst >= 0) {
        first = Math.max(0, idxFirst - 1);
        for (let i = pts.length - 1; i >= idxFirst; i--) {
          if (Math.abs(pts[i]!.y - baselineY) > 1.5) {
            last = Math.min(pts.length - 1, i + 1);
            break;
          }
        }
        drawPts = pts.slice(first, last + 1);
      }
    }
    caliperPts = [drawPts[0]!, drawPts[drawPts.length - 1]!];
  } else if (denseVolume) {
    // Auto-traced endocardial contours highlight only the 3 anatomical landmarks (2 annular hinges + apex).
    caliperPts = [pts[0]!, pts[Math.floor(pts.length / 2)]!, pts[pts.length - 1]!];
  }
  // Dense auto-traced spectral envelopes draw a sleek clinical curve + translucent area fill to the
  // zero baseline, placing caliper crosses only at the active flow boundaries instead of every column.
  if (kind === 'vti' && drawPts.length >= 2 && baselineY !== undefined) {
    ctx.beginPath();
    ctx.moveTo(drawPts[0]!.x, baselineY);
    for (const p of drawPts) ctx.lineTo(p.x, p.y);
    ctx.lineTo(drawPts[drawPts.length - 1]!.x, baselineY);
    ctx.closePath();
    ctx.fillStyle = color === '#ffc857' ? 'rgba(255, 200, 87, 0.14)' : 'rgba(92, 200, 255, 0.14)';
    ctx.fill();
  } else if (denseVolume && drawPts.length >= 5) {
    ctx.beginPath();
    ctx.moveTo(drawPts[0]!.x, drawPts[0]!.y);
    for (const p of drawPts.slice(1)) ctx.lineTo(p.x, p.y);
    ctx.closePath();
    ctx.fillStyle = color === '#ffc857' ? 'rgba(255, 200, 87, 0.11)' : 'rgba(92, 200, 255, 0.12)';
    ctx.fill();
  }
  // calipers as on a scanner (decision 200): a cross at each point and a line between them, both over a dark
  // halo so they read on bright tissue as well as in the cavity
  const crosses = () => {
    ctx.beginPath();
    for (const p of caliperPts) {
      ctx.moveTo(p.x - 4.5, p.y);
      ctx.lineTo(p.x + 4.5, p.y);
      ctx.moveTo(p.x, p.y - 4.5);
      ctx.lineTo(p.x, p.y + 4.5);
    }
    ctx.stroke();
  };
  const joined =
    drawPts.length >= 2 &&
    (kind === 'linear' || kind === 'time' || kind === 'vti' || kind === 'volume');
  const line = () => {
    ctx.beginPath();
    ctx.moveTo(drawPts[0]!.x, drawPts[0]!.y);
    for (const p of drawPts.slice(1)) ctx.lineTo(p.x, p.y);
    if (kind === 'volume' && !denseVolume) ctx.closePath();
    ctx.stroke();
  };
  const annularBase = () => {
    if (!denseVolume || drawPts.length < 3) return;
    ctx.beginPath();
    ctx.moveTo(drawPts[drawPts.length - 1]!.x, drawPts[drawPts.length - 1]!.y);
    ctx.lineTo(drawPts[0]!.x, drawPts[0]!.y);
    ctx.stroke();
  };
  const cap = ctx.lineCap;
  const join = ctx.lineJoin;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.72)';
  ctx.lineWidth = denseVti || denseVolume ? 3.2 : 3.5;
  crosses();
  if (joined) line();
  if (denseVolume) annularBase();
  ctx.strokeStyle = color;
  ctx.lineWidth = denseVti || denseVolume ? 1.8 : 1.5;
  crosses();
  if (joined) {
    if (!denseVti && !denseVolume) ctx.setLineDash([5, 4]);
    line();
    ctx.setLineDash([]);
  }
  if (denseVolume) {
    ctx.setLineDash([4, 4]);
    annularBase();
    ctx.setLineDash([]);
  }
  ctx.lineCap = cap;
  ctx.lineJoin = join;
}

/** A measurement's value beside its last point: a small dark chip with the figure in the mono face (decision 200). */
function drawValueChip(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  color: string,
): void {
  const font = ctx.font;
  ctx.font = canvasFont(10.5, 'mono');
  const w = ctx.measureText(text).width + 12;
  const h = 18;
  const left = Math.max(
    2,
    Math.min(ctx.canvas.width / (window.devicePixelRatio || 1) - w - 2, x + 8),
  );
  // above the point, or under it when the point is at the top edge (decision 201)
  const top = y - h - 6 < 2 ? y + 8 : y - h - 6;
  ctx.fillStyle = 'rgba(8, 11, 16, 0.8)';
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.14)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(left, top, w, h, 5);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, left + 6, top + h / 2 + 0.5);
  ctx.font = font;
}

/** Compact color-coded beam–flow angle badge (θ XX°) beside the PW/TDI gate or CW cursor ray. */
function drawAngleBadge(
  ctx: CanvasRenderingContext2D,
  angleDeg: number,
  quality: 'ok' | 'warn' | 'invalid',
  anchorX: number,
  anchorY: number,
  W: number,
  apexY: number,
  sectorH: number,
  cursorTheta: number,
): void {
  const font = ctx.font;
  const textAlign = ctx.textAlign;
  const text = `θ ${angleDeg}°`;
  ctx.font = canvasFont(10.5, 'mono');
  const textW = ctx.measureText(text).width;
  const padX = 6;
  const w = Math.ceil(textW + padX * 2);
  const h = 17;
  // Place on the opposite side of the cursor away from the sector center when tilted, otherwise to the right.
  const placeLeft = cursorTheta > 0.25 && anchorX - 14 - w >= 6;
  const rawLeft = placeLeft ? anchorX - 14 - w : anchorX + 14;
  const left = Math.max(6, Math.min(W - w - 6, rawLeft));
  const top = Math.max(apexY + 8, Math.min(sectorH - h - 6, Math.round(anchorY - h / 2)));
  const fg = quality === 'ok' ? '#57d38c' : quality === 'warn' ? '#fbbf24' : '#f87171';
  const border =
    quality === 'ok'
      ? 'rgba(87, 211, 140, 0.42)'
      : quality === 'warn'
        ? 'rgba(251, 191, 36, 0.45)'
        : 'rgba(248, 113, 113, 0.48)';
  ctx.fillStyle = 'rgba(8, 12, 18, 0.78)';
  ctx.strokeStyle = border;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(left, top, w, h, 4);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = fg;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, left + padX, top + h / 2 + 0.5);
  ctx.font = font;
  ctx.textAlign = textAlign;
}

/** Real-time cyan rubber-band guide and live clinical value chip while placing a measurement. */
function drawLivePreview(ctx: CanvasRenderingContext2D, preview: LivePreview): void {
  const pts = preview.points;
  if (!pts.length) return;
  ctx.save();
  if (preview.shape === 'span' && pts.length >= 2) {
    const x0 = Math.min(pts[0]!.x, pts[1]!.x);
    const x1 = Math.max(pts[0]!.x, pts[1]!.x);
    const y0 = pts[0]!.y;
    const h = pts[1]!.y - y0;
    ctx.fillStyle = 'rgba(92, 200, 255, 0.14)';
    ctx.fillRect(x0, y0, Math.max(1, x1 - x0), h);
    ctx.strokeStyle = 'rgba(92, 200, 255, 0.85)';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x0, y0 + h);
    ctx.moveTo(x1, y0);
    ctx.lineTo(x1, y0 + h);
    ctx.stroke();
  } else if (preview.shape === 'guide' && pts.length >= 2) {
    ctx.strokeStyle = 'rgba(92, 200, 255, 0.52)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(pts[0]!.x, pts[0]!.y);
    ctx.lineTo(pts[1]!.x, pts[1]!.y);
    ctx.stroke();
  } else if (pts.length >= 2) {
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.65)';
    ctx.lineWidth = 3;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pts[0]!.x, pts[0]!.y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i]!.x, pts[i]!.y);
    ctx.stroke();
    ctx.strokeStyle = '#5cc8ff';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(pts[0]!.x, pts[0]!.y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i]!.x, pts[i]!.y);
    ctx.stroke();
    const target = preview.anchor;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(target.x - 4, target.y);
    ctx.lineTo(target.x + 4, target.y);
    ctx.moveTo(target.x, target.y - 4);
    ctx.lineTo(target.x, target.y + 4);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  drawValueChip(ctx, preview.label, preview.anchor.x, preview.anchor.y, '#5cc8ff');
  ctx.restore();
}
