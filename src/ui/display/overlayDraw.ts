import { markerLabel, structureLabel, type ReviewMarker } from '@/app/review';
import { imageSegmentsOn, useSegmentOrientation, type SimStore } from '@/app/store';
import type { SimOutput } from '@/simulator/core/protocol';
import { reprojectGeometry } from '@/simulator/measurements/geometry';
import { hasGate, isStripModality, MODALITIES } from '@/simulator/renderer/modality';
import { tgcAtDepth } from '@/simulator/renderer/postprocess/consolePipeline';
import { polarToPixel, type SectorMapping } from '@/simulator/renderer/scanConvert';
import { canvasFont } from '../canvasFonts';
import { ecgLayoutOf, ecgTracePoints, ecgX } from '../ecgTrace';
import { drawPickedRegion, drawSegmentLayer, drawStructureOutline } from './rasterLayers';

/** Where an image marker sits on the current display: strip markers keep their pixel, sector markers reproject. */
export function markerScreenPosition(
  mk: ReviewMarker,
  m: SectorMapping,
): { x: number; y: number } {
  if (mk.strip || !mk.captureSector) return { x: mk.x, y: mk.y };
  return reprojectGeometry([{ x: mk.x, y: mk.y }], mk.captureSector, m)[0] ?? { x: mk.x, y: mk.y };
}

export function drawOverlay(
  canvas: HTMLCanvasElement,
  hud: SimOutput,
  st: SimStore,
  pending: { x: number; y: number }[],
  lassoDraft: { x: number; y: number }[] = [],
): void {
  const { ui, activeTool, measurements, reviewMarkers } = st;
  const {
    modality,
    color,
    settings,
    cursorThetaRad: cursorTheta,
    gateDepthCm: gateDepth,
    spectral,
  } = hud.frozen && hud.acquisition ? hud.acquisition : st;
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
  ctx.fillStyle = '#5cc8ff';
  ctx.beginPath();
  ctx.arc(m.apexX + 16, m.apexY + 4, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#9aa4b5';
  ctx.fillText('R', m.apexX + 24, m.apexY + 4);
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
  if (modality === 'color') {
    drawArcBox(
      ctx,
      m,
      color.boxThetaMinRad,
      color.boxThetaMaxRad,
      color.boxRMinCm,
      color.boxRMaxCm,
      '#57d38c',
    );
    // below the telemetry the HUD prints in the top-right corner (two lines, ~60 px), clear of the focus marker
    const bx = W - 30,
      by = Math.max(m.apexY + 20, 76),
      bh = 90;
    const grad = ctx.createLinearGradient(0, by, 0, by + bh);
    grad.addColorStop(0, '#ffd23c');
    grad.addColorStop(0.45, '#c81e1e');
    grad.addColorStop(0.5, '#000');
    grad.addColorStop(0.55, '#1e3cc8');
    grad.addColorStop(1, '#3ce0ff');
    ctx.fillStyle = grad;
    ctx.fillRect(bx, by, 8, bh);
    ctx.fillStyle = '#b9c3d0';
    ctx.font = canvasFont(10, 'mono');
    ctx.textAlign = 'right';
    ctx.fillText(`+${color.scaleMps.toFixed(2)}`, bx - 5, by + 7);
    ctx.fillText(`−${color.scaleMps.toFixed(2)}`, bx - 5, by + bh);
    ctx.fillStyle = '#7f8a9a';
    ctx.font = canvasFont(9.5);
    ctx.fillText('hacia', bx - 5, by + bh / 2 - 6);
    ctx.fillText('desde', bx - 5, by + bh / 2 + 12);
    ctx.textAlign = 'left';
    ctx.font = canvasFont(11);
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
    const strip = hud.strip;
    if (strip.kind === 'spectral') {
      ctx.fillStyle = '#9aa4b5';
      ctx.textAlign = 'right';
      const n = 4;
      for (let i = 0; i <= n; i++) {
        const v = strip.topValue + ((strip.bottomValue - strip.topValue) * i) / n;
        const y = strip.y + (strip.height * i) / n;
        ctx.fillText(`${v.toFixed(2)}`, W - 4, Math.min(H - 8, Math.max(strip.y + 6, y)));
      }
      ctx.textAlign = 'left';
      ctx.fillText(
        `${modality.toUpperCase()}  ${strip.sweepSpeedMmPerS ?? '—'} mm/s  escala ±${((strip.topValue - strip.bottomValue) / 2).toFixed(2)} m/s  WF ${strip.wallFilterMps === undefined ? '—' : Math.round(strip.wallFilterMps * 100)} cm/s`,
        6,
        strip.y + 8,
      );
      const cols1s = strip.secondsPerColumn > 0 ? 1 / strip.secondsPerColumn : 0;
      if (cols1s > 0) {
        ctx.strokeStyle = 'rgba(154,164,181,0.35)';
        for (let x = 0; x < strip.width; x += cols1s) {
          ctx.beginPath();
          ctx.moveTo(strip.x + x, strip.y + strip.height - 8);
          ctx.lineTo(strip.x + x, strip.y + strip.height);
          ctx.stroke();
        }
      }
    } else if (strip.kind === 'm-mode') {
      ctx.fillStyle = '#9aa4b5';
      ctx.textAlign = 'right';
      for (let d = 0; d <= strip.bottomValue; d += 5) {
        const y = strip.y + (d / strip.bottomValue) * strip.height;
        ctx.fillText(`${d}`, W - 4, Math.min(H - 8, Math.max(strip.y + 6, y)));
      }
      ctx.textAlign = 'left';
      ctx.fillText(`M-MODE  ${strip.sweepSpeedMmPerS ?? '—'} mm/s`, 6, strip.y + 8);
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
  const shown = measurements
    .filter(
      (ms) =>
        ms.modality === modality ||
        (ms.modality === '2d' && modality === 'color') ||
        (ms.modality === 'color' && modality === '2d'),
    )
    .map((ms) => ({ ms, geometry: reprojectGeometry(ms.geometry, ms.captureSector, m) }));
  for (const { ms, geometry } of shown) {
    const p = geometry[geometry.length - 1];
    if (p)
      drawValueChip(
        ctx,
        `${ms.label} ${ms.value.toFixed(ms.kind === 'time' ? 0 : ms.kind === 'vti' ? 1 : 2)} ${ms.units}`,
        p.x,
        p.y,
        '#ffc857',
      );
  }
  for (const { ms, geometry } of shown) drawGeometry(ctx, geometry, ms.kind, '#ffc857');
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
    );
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
): void {
  // calipers as on a scanner (decision 200): a cross at each point and a dashed line between them, both over a dark
  // halo so they read on bright tissue as well as in the cavity
  const crosses = () => {
    ctx.beginPath();
    for (const p of pts) {
      ctx.moveTo(p.x - 5, p.y);
      ctx.lineTo(p.x + 5, p.y);
      ctx.moveTo(p.x, p.y - 5);
      ctx.lineTo(p.x, p.y + 5);
    }
    ctx.stroke();
  };
  const joined =
    pts.length >= 2 &&
    (kind === 'linear' || kind === 'time' || kind === 'vti' || kind === 'volume');
  const line = () => {
    ctx.beginPath();
    ctx.moveTo(pts[0]!.x, pts[0]!.y);
    for (const p of pts.slice(1)) ctx.lineTo(p.x, p.y);
    if (kind === 'volume') ctx.closePath();
    ctx.stroke();
  };
  const cap = ctx.lineCap;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.65)';
  ctx.lineWidth = 3.5;
  crosses();
  if (joined) line();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  crosses();
  if (joined) {
    ctx.setLineDash([5, 4]);
    line();
    ctx.setLineDash([]);
  }
  ctx.lineCap = cap;
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
