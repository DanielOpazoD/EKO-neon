import { frameBus } from '@/app/frameBus';
import { examClosed } from '@/app/modePolicy';
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
import { useCallback, useEffect, useRef } from 'react';
import { cineOffsetAtX, cineOffsetOnEcg, ecgLayoutOf } from './ecgTrace';
import { ImageHud } from './ImageHud';
import { ImageQuickBar } from './ImageQuickBar';
import { SegmentImageToggle } from './SegmentImageToggle';
import { sampleIndexAt, segmentCss, segmentIdOf, segmentNames } from './segmentMap';
import { coverageText } from './segmentCoverage';
import {
  finalizePendingMeasurement,
  handleMeasurementToolClick,
  type PendingMeasurement,
} from './display/measurementTools';
import { CaptureBar } from './display/CaptureBar';
import {
  colorBoxHandleCursor,
  hitTestColorBoxHandle,
  resizeColorBoxFromHandle,
  type ColorBoxHandleId,
} from './display/colorBoxHandles';
import { hitTestMeasurementVertex, recomputeEditedMeasurement } from './display/measurementEdit';
import { computeLivePreview } from './display/measurementPreview';
import { drawOverlay } from './display/overlayDraw';
import { frameSizeForBox } from './display/frameSize';
import { pendingAcquisitionKey } from './display/measurementPlacement';
import { pendingSession } from './display/pendingSession';
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
  const lastSizeRef = useRef<{ width: number; height: number } | null>(null);
  const pendingRef = useRef<PendingMeasurement>({ points: [] });
  /** Pointer coordinates over the canvas while a measurement tool is armed (ref-only: zero React re-renders). */
  const hoverPtRef = useRef<{ x: number; y: number } | null>(null);
  /** Hovered or dragged measurement caliper vertex on a frozen frame (ref-only: zero React re-renders). */
  const hoveredHandleRef = useRef<{ measurementId: string; vertexIndex: number } | null>(null);
  /** Hovered or dragged Color Doppler ROI corner handle (ref-only: zero React re-renders). */
  const hoveredColorHandleRef = useRef<ColorBoxHandleId | 'move' | null>(null);
  /** Ephemeral teaching arrow currently being dragged on a frozen frame (ref-only). */
  const arrowDraftRef = useRef<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const lastOutRef = useRef<SimOutput | null>(null);
  /** the overlay redraw of the drawing effect, so tool clicks refresh the overlay without waiting for a frame */
  const redrawOverlayRef = useRef<() => void>(() => {});
  const dragRef = useRef<{
    kind:
      | 'box-move'
      | 'box-resize'
      | 'box-handle'
      | 'cursor'
      | 'marker'
      | 'cine'
      | 'lasso'
      | 'measure-vertex'
      | 'teach-arrow'
      | 'none';
    startX: number;
    startY: number;
    box?: { t0: number; t1: number; r0: number; r1: number };
    handleId?: ColorBoxHandleId;
    /** review marker being dragged (decision 135) and whether it has moved */
    markerId?: string;
    /** measurement vertex being dragged on a frozen image */
    measureId?: string;
    vertexIndex?: number;
    moved?: boolean;
  }>({ kind: 'none', startX: 0, startY: 0 });
  /** the lasso being drawn in review mode (decision 340), polar */
  const lassoRef = useRef<LassoPoint[]>([]);
  const reviewMode = useSimStore((s) => s.ui.reviewMode);
  const reviewTool = useSimStore((s) => s.ui.reviewTool);
  const frozen = useSimStore((s) => s.frozen);
  const armed = useSimStore((s) => s.activeTool !== 'none');
  const pointerOn = useSimStore((s) => s.teachingPointerOn);
  const segTipRef = useRef<HTMLDivElement>(null);
  const onSize = props.onSize;

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      // the frame keeps the box's proportions: the canvases stretch over the box, and must stretch alike in x and y
      const next = frameSizeForBox(r.width, r.height);
      const prev = lastSizeRef.current;
      if (prev && prev.width === next.width && prev.height === next.height) return;
      lastSizeRef.current = next;
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
      const keepColorCursor =
        !out.frozen &&
        state.modality === 'color' &&
        (hoveredColorHandleRef.current !== null ||
          dragRef.current.kind === 'box-handle' ||
          dragRef.current.kind === 'box-move');
      if (
        canvas.style.cursor &&
        (state.activeTool !== 'none' ||
          state.ui.reviewMode ||
          state.teachingPointerOn ||
          (!out.frozen && !keepColorCursor))
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
      pendingSession.setCount(pending.points.length);
      const pendingReproj = reprojectGeometry(pending.points, pending.captureSector, out.sector);
      const livePreview = computeLivePreview(
        state.activeTool,
        state.activeMeasurementId,
        pendingReproj,
        hoverPtRef.current,
        out,
        state.modality,
      );
      const activeHandle =
        out.frozen &&
        state.activeTool === 'none' &&
        !state.teachingPointerOn &&
        !state.ui.reviewMode &&
        !examClosed(state)
          ? dragRef.current.kind === 'measure-vertex' &&
            dragRef.current.measureId !== undefined &&
            dragRef.current.vertexIndex !== undefined
            ? {
                measurementId: dragRef.current.measureId,
                vertexIndex: dragRef.current.vertexIndex,
                dragging: true,
              }
            : hoveredHandleRef.current
              ? { ...hoveredHandleRef.current, dragging: false }
              : null
          : null;
      const activeColorHover =
        !out.frozen && state.modality === 'color' && state.activeTool === 'none' && !state.ui.reviewMode
          ? dragRef.current.kind === 'box-handle' && dragRef.current.handleId
            ? dragRef.current.handleId
            : dragRef.current.kind === 'box-move'
              ? 'move'
              : hoveredColorHandleRef.current
          : null;
      drawOverlay(
        canvas,
        out,
        state,
        pendingReproj,
        lassoRef.current.map((q) => polarToPixel(out.sector, q.rCm, q.thetaRad)),
        livePreview,
        activeHandle,
        activeColorHover,
        arrowDraftRef.current,
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
    pendingSession.register({
      undo: () => {
        if (pendingRef.current.points.length > 0) {
          pendingRef.current.points.pop();
          pendingSession.setCount(pendingRef.current.points.length);
          redrawOverlay();
        }
      },
      finish: () => {
        const out = lastOutRef.current;
        if (!out) return;
        finalizePendingMeasurement(out, pendingRef.current, redrawOverlay);
        pendingSession.setCount(pendingRef.current.points.length);
      },
    });
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
        next.teachingPointerOn === prev.teachingPointerOn &&
        next.teachingArrows === prev.teachingArrows &&
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
      pendingSession.register(null);
      pendingSession.setCount(0);
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
      pendingSession.setCount(pendingRef.current.points.length);
      return;
    }
    // frozen teaching pointer: drag anywhere on the sector or strip to draw an ephemeral callout arrow
    if (hud.frozen && st.teachingPointerOn && !examClosed(st) && st.mode !== 'exam') {
      arrowDraftRef.current = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
      dragRef.current = { kind: 'teach-arrow', startX: p.x, startY: p.y };
      redrawOverlayRef.current();
      return;
    }
    // frozen, dragging any caliper/velocity/time/manual-contour vertex edits the measurement live
    if (hud.frozen && !examClosed(st)) {
      const handle = hitTestMeasurementVertex(p, st.measurements, hud, st.modality);
      if (handle) {
        dragRef.current = {
          kind: 'measure-vertex',
          startX: p.x,
          startY: p.y,
          measureId: handle.measurementId,
          vertexIndex: handle.vertexIndex,
        };
        hoveredHandleRef.current = {
          measurementId: handle.measurementId,
          vertexIndex: handle.vertexIndex,
        };
        if (ovRef.current) ovRef.current.style.cursor = 'grabbing';
        redrawOverlayRef.current();
        return;
      }
    }
    // frozen, a press on the ECG strip picks the cine frame at that instant and a drag scrubs (decision 190)
    const cineOffset = cineOffsetOnEcg(p, hud, st);
    if (cineOffset !== null) {
      st.setCineOffset(cineOffset);
      dragRef.current = { kind: 'cine', startX: p.x, startY: p.y };
      return;
    }
    if (!hud.frozen && st.modality === 'color' && inSector) {
      const { rCm, thetaRad } = pixelToPolar(m, p.x, p.y);
      const c = st.color;
      const box = { t0: c.boxThetaMinRad, t1: c.boxThetaMaxRad, r0: c.boxRMinCm, r1: c.boxRMaxCm };
      const corner = hitTestColorBoxHandle(p, m, box);
      if (corner) {
        dragRef.current = {
          kind: 'box-handle',
          handleId: corner,
          startX: p.x,
          startY: p.y,
          box,
        };
        hoveredColorHandleRef.current = corner;
        if (ovRef.current) ovRef.current.style.cursor = colorBoxHandleCursor(corner, m.invertLR);
        redrawOverlayRef.current();
        return;
      }
      const inside =
        rCm >= c.boxRMinCm &&
        rCm <= c.boxRMaxCm &&
        thetaRad >= c.boxThetaMinRad &&
        thetaRad <= c.boxThetaMaxRad;
      dragRef.current = {
        kind: inside && !e.shiftKey ? 'box-move' : 'box-resize',
        startX: p.x,
        startY: p.y,
        box,
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
    if (!hud.frozen && isStripModality(st.modality) && inSector) {
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
      const st = useSimStore.getState();
      const p = hud ? toLocal(e) : null;
      if (hud && p && st.activeTool !== 'none') {
        hoverPtRef.current = p;
        if (hud.frozen) redrawOverlayRef.current();
      } else if (hoverPtRef.current !== null) {
        hoverPtRef.current = null;
      }
      // Highlight draggable caliper handles on frozen images (zero React re-renders)
      if (
        hud &&
        p &&
        hud.frozen &&
        st.activeTool === 'none' &&
        !st.teachingPointerOn &&
        !st.ui.reviewMode &&
        !examClosed(st)
      ) {
        const handle = hitTestMeasurementVertex(p, st.measurements, hud, st.modality);
        const prev = hoveredHandleRef.current;
        const changed =
          (handle?.measurementId ?? null) !== (prev?.measurementId ?? null) ||
          (handle?.vertexIndex ?? null) !== (prev?.vertexIndex ?? null);
        hoveredHandleRef.current = handle
          ? { measurementId: handle.measurementId, vertexIndex: handle.vertexIndex }
          : null;
        if (changed) redrawOverlayRef.current();
        if (ovRef.current) {
          ovRef.current.style.cursor = handle
            ? 'grab'
            : cineOffsetOnEcg(p, hud, st) !== null
              ? 'ew-resize'
              : '';
        }
        return;
      }
      if (hoveredHandleRef.current !== null) {
        hoveredHandleRef.current = null;
        redrawOverlayRef.current();
      }
      // Highlight draggable Color Doppler ROI box corners and body on live color frames
      if (
        hud &&
        p &&
        !hud.frozen &&
        st.modality === 'color' &&
        st.activeTool === 'none' &&
        !st.ui.reviewMode &&
        p.y < hud.sector.height
      ) {
        const m = hud.sector;
        const c = st.color;
        const box = { t0: c.boxThetaMinRad, t1: c.boxThetaMaxRad, r0: c.boxRMinCm, r1: c.boxRMaxCm };
        const corner = hitTestColorBoxHandle(p, m, box);
        let nextColorHover: ColorBoxHandleId | 'move' | null = corner;
        if (!nextColorHover) {
          const { rCm, thetaRad } = pixelToPolar(m, p.x, p.y);
          const inside =
            rCm >= c.boxRMinCm &&
            rCm <= c.boxRMaxCm &&
            thetaRad >= c.boxThetaMinRad &&
            thetaRad <= c.boxThetaMaxRad;
          if (inside) nextColorHover = 'move';
        }
        if (nextColorHover !== hoveredColorHandleRef.current) {
          hoveredColorHandleRef.current = nextColorHover;
          redrawOverlayRef.current();
        }
        if (ovRef.current) {
          ovRef.current.style.cursor =
            nextColorHover && nextColorHover !== 'move'
              ? colorBoxHandleCursor(nextColorHover, m.invertLR)
              : nextColorHover === 'move'
                ? 'move'
                : '';
        }
        return;
      }
      if (hoveredColorHandleRef.current !== null) {
        hoveredColorHandleRef.current = null;
        redrawOverlayRef.current();
      }
      // the strip says it can be scrubbed; an armed tool keeps its crosshair from the stylesheet
      if (hud && p && ovRef.current)
        ovRef.current.style.cursor = cineOffsetOnEcg(p, hud, st) !== null ? 'ew-resize' : '';
      return;
    }
    if (!hud) return;
    const p = toLocal(e);
    const st = useSimStore.getState();
    const m = hud.sector;
    if (
      dragRef.current.kind === 'measure-vertex' &&
      dragRef.current.measureId !== undefined &&
      dragRef.current.vertexIndex !== undefined
    ) {
      const ms = st.measurements.find((x) => x.id === dragRef.current.measureId);
      if (ms) {
        const edited = recomputeEditedMeasurement(ms, dragRef.current.vertexIndex, p, hud, {
          phaseMarks: hud.acquisition?.phaseMarks ?? st.phaseMarks,
          lvLengthCm: st.lvLengthCm,
          color: hud.acquisition?.color ?? st.color,
        });
        if (edited) {
          dragRef.current.vertexIndex = edited.vertexIndex;
          hoveredHandleRef.current = {
            measurementId: ms.id,
            vertexIndex: edited.vertexIndex,
          };
          st.updateMeasurement(ms.id, edited.patch);
        }
      }
      return;
    }
    if (dragRef.current.kind === 'teach-arrow') {
      arrowDraftRef.current = {
        x0: dragRef.current.startX,
        y0: dragRef.current.startY,
        x1: p.x,
        y1: p.y,
      };
      redrawOverlayRef.current();
      return;
    }
    if (dragRef.current.kind === 'box-handle' && dragRef.current.box && dragRef.current.handleId) {
      const b = pixelToPolar(m, p.x, Math.min(p.y, m.height - 1));
      st.setColor(resizeColorBoxFromHandle(dragRef.current.handleId, dragRef.current.box, b, m));
      return;
    }
    if (dragRef.current.kind === 'cine') {
      const layout = ecgLayoutOf(hud, hud.acquisition?.modality ?? st.modality);
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
    if (d.kind === 'teach-arrow') {
      const draft = arrowDraftRef.current;
      arrowDraftRef.current = null;
      dragRef.current = { kind: 'none', startX: 0, startY: 0 };
      if (draft && Math.hypot(draft.x1 - draft.x0, draft.y1 - draft.y0) >= 8) {
        const hud = lastOutRef.current;
        const onStrip = hud ? draft.y0 >= hud.sector.height && draft.y1 >= hud.sector.height : false;
        useSimStore.getState().addTeachingArrow({
          x0: draft.x0,
          y0: draft.y0,
          x1: draft.x1,
          y1: draft.y1,
          onStrip,
          ...(hud && !onStrip ? { captureSector: hud.sector } : {}),
        });
      }
      redrawOverlayRef.current();
      return;
    }
    if (d.kind === 'box-handle') {
      dragRef.current = { kind: 'none', startX: 0, startY: 0 };
      redrawOverlayRef.current();
      return;
    }
    if (d.kind === 'measure-vertex') {
      dragRef.current = { kind: 'none', startX: 0, startY: 0 };
      if (ovRef.current) ovRef.current.style.cursor = hoveredHandleRef.current ? 'grab' : '';
      redrawOverlayRef.current();
      return;
    }
    dragRef.current = { kind: 'none', startX: 0, startY: 0 };
  };

  return (
    <div className={`display-wrap${frozen ? ' frozen' : ''}`} ref={wrapRef}>
      {/* imgRef dimensions are updated exclusively inside drawImage(out) right before painting so
          layout/resize events never clear the ultrasound canvas to black before the next frame arrives. */}
      <canvas ref={imgRef} aria-label="Imagen ecográfica simulada" />
      <canvas
        ref={ovRef}
        className={`overlay${reviewMode ? ' review' : ''}${reviewMode && reviewTool === 'lasso' ? ' lasso' : ''}${armed ? ' armed' : ''}${pointerOn ? ' pointer' : ''}`}
        style={{ width: '100%', height: '100%' }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={() => {
          onMouseUp();
          updateSegmentTip(null);
          if (hoverPtRef.current !== null) {
            hoverPtRef.current = null;
            if (useSimStore.getState().activeTool !== 'none') redrawOverlayRef.current();
          }
          if (hoveredHandleRef.current !== null) {
            hoveredHandleRef.current = null;
            redrawOverlayRef.current();
          }
          if (hoveredColorHandleRef.current !== null) {
            hoveredColorHandleRef.current = null;
            redrawOverlayRef.current();
          }
        }}
        aria-label="Superposiciones y herramientas de medición"
      />
      <ImageHud />
      <CaptureBar />
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
