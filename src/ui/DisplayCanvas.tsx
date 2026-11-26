import { frameBus } from '@/app/frameBus';
import type { LassoPoint } from '@/app/reviewRegion';
import {
  imageSegmentsOn,
  useSegmentHover,
  useSimStore,
  useStructureHover,
} from '@/app/store';
import type { SimOutput } from '@/simulator/core/protocol';
import { reprojectGeometry } from '@/simulator/measurements/geometry';
import { hasGate, isStripModality } from '@/simulator/renderer/modality';
import { pixelToPolar, polarToPixel } from '@/simulator/renderer/scanConvert';
import { useCallback, useEffect, useRef, useState } from 'react';
import { cineOffsetAtX, cineOffsetOnEcg, ecgLayoutOf } from './ecgTrace';
import { ImageHud } from './ImageHud';
import { ImageQuickBar } from './ImageQuickBar';
import { SegmentImageToggle } from './SegmentImageToggle';
import { sampleIndexAt, segmentCss, segmentIdOf, segmentNames } from './segmentMap';
import { coverageText } from './segmentCoverage';
import { handleMeasurementToolClick, type PendingMeasurement } from './display/measurementTools';
import { drawOverlay } from './display/overlayDraw';
import { frameSizeForBox } from './display/frameSize';
import { pendingAcquisitionKey } from './display/measurementPlacement';
import {
  commitReviewLasso,
  identifyReviewMarker,
  placeReviewMarker,
  reviewFieldsAt,
  reviewMarkerAt,
} from './display/reviewTools';

/**
 * Ultrasound display: draws the composite frame from the simulator and the overlays (depth scale,
 * orientation marker, focus, TGC curve, ECG, colour box, Doppler cursor/gate, calipers). Direct
 * manipulation of box/cursor/gate/calipers happens here (spec 29.3).
 */
export interface DisplayHandle {
  pushFrame: (out: SimOutput) => void;
}

export function DisplayCanvas(props: { onSize: (s: { width: number; height: number }) => void }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLCanvasElement>(null);
  const ovRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ width: 640, height: 520 });
  const pendingRef = useRef<PendingMeasurement>({ points: [] });
  const lastOutRef = useRef<SimOutput | null>(null);
  /** the overlay redraw of the drawing effect, so tool clicks refresh the overlay without waiting for a frame */
  const redrawOverlayRef = useRef<() => void>(() => {});
  const dragRef = useRef<{
    kind: 'box-move' | 'box-resize' | 'cursor' | 'marker' | 'cine' | 'lasso' | 'none';
    startX: number;
    startY: number;
    box?: { t0: number; t1: number; r0: number; r1: number };
    /** review marker being dragged (decision 135) and whether it has moved */
    markerId?: string;
    moved?: boolean;
  }>({ kind: 'none', startX: 0, startY: 0 });
  /** the lasso being drawn in review mode (decision 340), polar */
  const lassoRef = useRef<LassoPoint[]>([]);
  const reviewMode = useSimStore((s) => s.ui.reviewMode);
  const reviewTool = useSimStore((s) => s.ui.reviewTool);
  const frozen = useSimStore((s) => s.frozen);
  const armed = useSimStore((s) => s.activeTool !== 'none');
  const segTipRef = useRef<HTMLDivElement>(null);
  const onSize = props.onSize;

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      // the frame keeps the box's proportions: the canvases stretch over the box, and must stretch alike in x and y
      const next = frameSizeForBox(r.width, r.height);
      setSize(next);
      onSize(next);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [onSize]);

  // Imperative drawing: image on every frame, overlay on every frame and on relevant store changes.
  useEffect(() => {
    const drawImage = (out: SimOutput) => {
      const canvas = imgRef.current;
      if (!canvas) return;
      if (canvas.width !== out.width || canvas.height !== out.height) {
        canvas.width = out.width;
        canvas.height = out.height;
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      if (out.bitmap) {
        // formed on the GPU (decision 54): a texture copy, then release the bitmap's GPU memory
        ctx.drawImage(out.bitmap, 0, 0);
        out.bitmap.close();
      } else
        ctx.putImageData(
          new ImageData(new Uint8ClampedArray(out.rgba), out.width, out.height),
          0,
          0,
        );
    };
    const redrawOverlay = () => {
      const out = lastOutRef.current;
      const canvas = ovRef.current;
      if (!out || !canvas) return;
      const state = useSimStore.getState();
      // the ↔ of the ECG strip goes with the scrub: live again, a tool or review mode, before the pointer moves
      if (
        canvas.style.cursor &&
        (!out.frozen || state.activeTool !== 'none' || state.ui.reviewMode)
      )
        canvas.style.cursor = '';
      const pending = pendingRef.current;
      const context = [
        state.caseId,
        state.modality,
        state.activeTool,
        state.activeMeasurementId ?? '',
        pendingAcquisitionKey(out, state.activeTool === 'caliper' || state.activeTool === 'simpson'),
      ].join('|');
      if (pending.context !== context) {
        pending.points = [];
        pending.captureSector = undefined;
        pending.context = context;
      }
      drawOverlay(
        canvas,
        out,
        state,
        reprojectGeometry(pending.points, pending.captureSector, out.sector),
        lassoRef.current.map((q) => polarToPixel(out.sector, q.rCm, q.thetaRad)),
      );
    };
    const unsubFrame = frameBus.subscribe((out) => {
      const t0 = performance.now();
      drawImage(out);
      const t1 = performance.now();
      frameBus.latest = null;
      frameBus.recycle(out.rgba);
      lastOutRef.current = out;
      redrawOverlay();
      frameBus.diag.drawMs = t1 - t0;
      frameBus.diag.overlayMs = performance.now() - t1;
      frameBus.diag.frames++;
    });
    redrawOverlayRef.current = redrawOverlay;
    const unsubStore = useSimStore.subscribe((next, prev) => {
      if (
        next.caseId === prev.caseId &&
        next.modality === prev.modality &&
        next.frozen === prev.frozen &&
        next.cineOffset === prev.cineOffset &&
        next.settings === prev.settings &&
        next.color === prev.color &&
        next.spectral === prev.spectral &&
        next.cursorThetaRad === prev.cursorThetaRad &&
        next.gateDepthCm === prev.gateDepthCm &&
        next.activeTool === prev.activeTool &&
        next.activeMeasurementId === prev.activeMeasurementId &&
        next.measurements === prev.measurements &&
        next.reviewMarkers === prev.reviewMarkers &&
        next.reviewSelectedId === prev.reviewSelectedId &&
        next.ui === prev.ui
      ) {
        return;
      }
      redrawOverlay();
    });
    // the segment under the pointer in another view (cut map, 3D heart, polar map) is highlighted here too
    const unsubHover = useSegmentHover.subscribe(redrawOverlay);
    // the structure under the pointer on the cut map is outlined here (decision 202)
    const unsubStructHover = useStructureHover.subscribe(redrawOverlay);
    return () => {
      unsubFrame();
      unsubStore();
      unsubHover();
      unsubStructHover();
    };
  }, []);

  const toLocal = useCallback((e: React.MouseEvent): { x: number; y: number } => {
    const r = ovRef.current!.getBoundingClientRect();
    const out = lastOutRef.current;
    const sx = (out?.width ?? r.width) / r.width;
    const sy = (out?.height ?? r.height) / r.height;
    return { x: (e.clientX - r.left) * sx, y: (e.clientY - r.top) * sy };
  }, []);

  const onMouseDown = (e: React.MouseEvent) => {
    const hud = lastOutRef.current;
    if (!hud) return;
    const p = toLocal(e);
    const st = useSimStore.getState();
    const m = hud.sector;
    const inSector = p.y < m.height;
    // review mode (decision 134): a plain click marks the image, a click on a marker selects it and drags it
    // (decision 135); Shift keeps the ordinary behaviour
    if (st.ui.reviewMode && st.activeTool === 'none' && !e.shiftKey) {
      const hit = reviewMarkerAt(p, hud, st.reviewMarkers);
      if (hit) {
        st.selectReviewMarker(hit.id);
        // a lasso keeps its outline where it was drawn: its number selects it, it does not drag
        if (!hit.lasso)
          dragRef.current = {
            kind: 'marker',
            startX: p.x,
            startY: p.y,
            markerId: hit.id,
            moved: false,
          };
      } else if (st.ui.reviewTool === 'lasso' && inSector) {
        // the pencil (decision 340): press, draw the outline, release to close it
        lassoRef.current = [pixelToPolar(m, p.x, p.y)];
        dragRef.current = { kind: 'lasso', startX: p.x, startY: p.y };
      } else {
        // Alt+click with a marker selected, or the armed «+ secundario», links the new point to that primary
        const sel = st.reviewMarkers.find((x) => x.id === st.reviewSelectedId);
        const parent = st.reviewLinkParentId ?? (e.altKey && sel ? (sel.parentId ?? sel.id) : null);
        placeReviewMarker(p, hud, imgRef.current, parent);
      }
      return;
    }
    if (st.activeTool !== 'none') {
      handleMeasurementToolClick(p, e.detail, hud, pendingRef.current, () =>
        redrawOverlayRef.current(),
      );
      return;
    }
    // frozen, a press on the ECG strip picks the cine frame at that instant and a drag scrubs (decision 190)
    const cineOffset = cineOffsetOnEcg(p, hud, st);
    if (cineOffset !== null) {
      st.setCineOffset(cineOffset);
      dragRef.current = { kind: 'cine', startX: p.x, startY: p.y };
      return;
    }
    if (st.modality === 'color' && inSector) {
      const { rCm, thetaRad } = pixelToPolar(m, p.x, p.y);
      const c = st.color;
      const inside =
        rCm >= c.boxRMinCm &&
        rCm <= c.boxRMaxCm &&
        thetaRad >= c.boxThetaMinRad &&
        thetaRad <= c.boxThetaMaxRad;
      dragRef.current = {
        kind: inside && !e.shiftKey ? 'box-move' : 'box-resize',
        startX: p.x,
        startY: p.y,
        box: { t0: c.boxThetaMinRad, t1: c.boxThetaMaxRad, r0: c.boxRMinCm, r1: c.boxRMaxCm },
      };
      if (!inside && !e.shiftKey) {
        // start a new box centred on the click
        st.setColor({
          boxThetaMinRad: thetaRad - 0.25,
          boxThetaMaxRad: thetaRad + 0.25,
          boxRMinCm: Math.max(0.5, rCm - 3),
          boxRMaxCm: Math.min(m.depthCm, rCm + 3),
        });
        dragRef.current.kind = 'none';
      }
      return;
    }
    if (isStripModality(st.modality) && inSector) {
      const { rCm, thetaRad } = pixelToPolar(m, p.x, p.y);
      st.setCursor(
        Math.max(-m.sectorRad / 2, Math.min(m.sectorRad / 2, thetaRad)),
        hasGate(st.modality) ? Math.max(1, Math.min(m.depthCm - 0.5, rCm)) : undefined,
      );
      dragRef.current = { kind: 'cursor', startX: p.x, startY: p.y };
    }
  };

  /** Name of the LV segment under the pointer, when the image shows the segments (decision 153). */
  const updateSegmentTip = (e: React.MouseEvent | null) => {
    const tip = segTipRef.current;
    const hud = lastOutRef.current;
    const st = useSimStore.getState();
    const hide = () => {
      if (tip) tip.style.display = 'none';
      useSegmentHover.getState().setHover(null, 'image');
    };
    if (!e || !tip || !hud || !imageSegmentsOn(st)) return hide();
    const p = toLocal(e);
    const m = hud.sector;
    const p0 = hud.polar;
    if (p.y >= m.height || hud.segment.length !== p0.lines * p0.samples) return hide();
    const k = sampleIndexAt(p0, m, p.x, p.y);
    const id = k >= 0 ? segmentIdOf(hud.segment[k] ?? 0, st.ui.segmentModel) : 0;
    if (id <= 0) return hide();
    const names = segmentNames(id, st.ui.segmentModel);
    const view = hud.view?.segments;
    const cov = (st.ui.segmentModel === 'LV_AHA17' ? view?.aha17 : view?.lv16)?.find(
      (c) => c.segmentId === id,
    );
    const swatch = tip.children[0] as HTMLElement;
    const title = tip.children[1] as HTMLElement;
    const sub = tip.children[2] as HTMLElement;
    swatch.style.background = segmentCss(id);
    title.textContent = `${id} · ${names.es}`;
    sub.textContent = `${names.en} · ${coverageText(cov)}`;
    const wrap = wrapRef.current!.getBoundingClientRect();
    const x = e.clientX - wrap.left,
      y = e.clientY - wrap.top;
    tip.style.display = 'grid';
    // beside the pointer, flipped to its left near the right edge of the image
    const left = x + 16 + 260 > wrap.width ? x - 16 - tip.offsetWidth : x + 16;
    tip.style.left = `${Math.max(4, left)}px`;
    tip.style.top = `${Math.max(4, y + 14)}px`;
    useSegmentHover.getState().setHover(id, 'image');
  };

  const onMouseMove = (e: React.MouseEvent) => {
    const hud = lastOutRef.current;
    if (dragRef.current.kind === 'none') {
      updateSegmentTip(e);
      // the strip says it can be scrubbed; an armed tool keeps its crosshair from the stylesheet
      if (hud && ovRef.current)
        ovRef.current.style.cursor =
          cineOffsetOnEcg(toLocal(e), hud, useSimStore.getState()) !== null ? 'ew-resize' : '';
      return;
    }
    if (!hud) return;
    const p = toLocal(e);
    const st = useSimStore.getState();
    const m = hud.sector;
    if (dragRef.current.kind === 'cine') {
      const layout = ecgLayoutOf(hud, st.modality);
      st.setCineOffset(cineOffsetAtX(p.x, layout, hud.cineWindow, hud.cineLength));
      return;
    }
    if (dragRef.current.kind === 'lasso') {
      const q = pixelToPolar(m, p.x, Math.min(p.y, m.height - 1));
      const last = lassoRef.current[lassoRef.current.length - 1];
      const step = last
        ? Math.hypot(
            q.rCm * Math.sin(q.thetaRad) - last.rCm * Math.sin(last.thetaRad),
            q.rCm * Math.cos(q.thetaRad) - last.rCm * Math.cos(last.thetaRad),
          )
        : Infinity;
      if (step >= 0.05) {
        lassoRef.current.push(q);
        redrawOverlayRef.current();
      }
      return;
    }
    if (dragRef.current.kind === 'marker') {
      const f = reviewFieldsAt(p, hud);
      if (f && dragRef.current.markerId) {
        st.updateReviewMarker(dragRef.current.markerId, { ...f, point: null });
        dragRef.current.moved = true;
      }
      return;
    }
    if (dragRef.current.kind === 'cursor') {
      const { rCm, thetaRad } = pixelToPolar(m, p.x, Math.min(p.y, m.height - 1));
      st.setCursor(
        Math.max(-m.sectorRad / 2, Math.min(m.sectorRad / 2, thetaRad)),
        hasGate(st.modality) ? Math.max(1, Math.min(m.depthCm - 0.5, rCm)) : undefined,
      );
    } else if (dragRef.current.kind === 'box-move' && dragRef.current.box) {
      const a = pixelToPolar(m, dragRef.current.startX, dragRef.current.startY);
      const b = pixelToPolar(m, p.x, p.y);
      const dt = b.thetaRad - a.thetaRad,
        dr = b.rCm - a.rCm;
      const bx = dragRef.current.box;
      st.setColor({
        boxThetaMinRad: bx.t0 + dt,
        boxThetaMaxRad: bx.t1 + dt,
        boxRMinCm: Math.max(0.5, bx.r0 + dr),
        boxRMaxCm: Math.min(m.depthCm, bx.r1 + dr),
      });
    } else if (dragRef.current.kind === 'box-resize' && dragRef.current.box) {
      const b = pixelToPolar(m, p.x, p.y);
      const bx = dragRef.current.box;
      st.setColor({
        boxThetaMaxRad: Math.max(bx.t0 + 0.08, b.thetaRad),
        boxRMaxCm: Math.max(bx.r0 + 1, Math.min(m.depthCm, b.rCm)),
      });
    }
  };

  const onMouseUp = () => {
    const d = dragRef.current;
    if (d.kind === 'marker' && d.moved && d.markerId) {
      // moved: ask the worker again for what lies under the new position, and outline the piece there
      const mk = useSimStore.getState().reviewMarkers.find((m) => m.id === d.markerId);
      if (mk) identifyReviewMarker(mk.id, mk.rCm, mk.thetaRad, lastOutRef.current, imgRef.current);
    }
    if (d.kind === 'lasso') {
      const drawn = lassoRef.current;
      lassoRef.current = [];
      redrawOverlayRef.current();
      commitReviewLasso(drawn, lastOutRef.current, imgRef.current);
    }
    dragRef.current = { kind: 'none', startX: 0, startY: 0 };
  };

  return (
    <div className={`display-wrap${frozen ? ' frozen' : ''}`} ref={wrapRef}>
      <canvas
        ref={imgRef}
        width={size.width}
        height={size.height}
        aria-label="Imagen ecográfica simulada"
      />
      <canvas
        ref={ovRef}
        className={`overlay${reviewMode ? ' review' : ''}${reviewMode && reviewTool === 'lasso' ? ' lasso' : ''}${armed ? ' armed' : ''}`}
        style={{ width: '100%', height: '100%' }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={() => {
          onMouseUp();
          updateSegmentTip(null);
        }}
        aria-label="Superposiciones y herramientas de medición"
      />
      <ImageHud />
      <SegmentImageToggle />
      <ImageQuickBar />
      <div className="seg-tip" ref={segTipRef} role="status" aria-live="polite">
        <i className="seg-tip-swatch" />
        <b />
        <span />
      </div>
    </div>
  );
}
