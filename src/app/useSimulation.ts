import { useEffect, useRef } from 'react';
import { SimClient } from '@/simulator/core/client';
import type { SimInput, SimOutput } from '@/simulator/core/protocol';
import { useSimStore } from './store';
import { modePolicy } from './modePolicy';
import { loadCaseById } from '@/cases';
import { frameBus } from './frameBus';

/**
 * Drives the simulator from requestAnimationFrame: snapshots the store into a SimInput, sends it,
 * and publishes outputs back (the HUD state). The heavy work happens in the worker.
 */
export function useSimulation(
  displaySize: { width: number; height: number },
  onFrame: (out: SimOutput) => void,
): void {
  const clientRef = useRef<SimClient | null>(null);
  const sizeRef = useRef(displaySize);
  const onFrameRef = useRef(onFrame);
  const caseId = useSimStore((s) => s.caseId);
  // the case the client last loaded: the mount loads it, and the case effect only reloads a different one (decision
  // 172; both loaded it at start, and the worker built the first core twice)
  const loadedCaseRef = useRef<string | null>(null);
  useEffect(() => {
    sizeRef.current = displaySize;
    clientRef.current?.send(buildInput(displaySize));
  }, [displaySize]);
  useEffect(() => {
    onFrameRef.current = onFrame;
  }, [onFrame]);

  useEffect(() => {
    const store = useSimStore.getState();
    const client = new SimClient({
      onFrame: (out) => {
        onFrameRef.current(out);
      },
      onReady: (truth, id, phaseMarks, lvLengthCm) => {
        useSimStore.getState().setTruth(truth, id);
        useSimStore.getState().setCycleInfo(phaseMarks, lvLengthCm);
        useSimStore.getState().setWorkerMode(client.mode);
        // free practice opens on the canonical PLAX once the case's models exist (decision 343)
        useSimStore.getState().openOnStartView();
      },
      onError: (m) => useSimStore.getState().setError(m),
    });
    clientRef.current = client;
    frameBus.recycle = (b) => client.recycle(b);
    frameBus.request = (req) => client.request(req);
    const caseDef = loadCaseById(store.caseId);
    loadedCaseRef.current = store.caseId;
    void client.loadCase(caseDef, buildInput(sizeRef.current));
    // Inputs are pushed on simulation-relevant store changes (ignoring fpsUi/HUD-only updates).
    const push = () => client.send(buildInput(sizeRef.current));
    const unsub = useSimStore.subscribe((next, prev) => {
      if (
        next.probe === prev.probe &&
        next.patient === prev.patient &&
        next.settings === prev.settings &&
        next.modality === prev.modality &&
        next.frozen === prev.frozen &&
        next.cineOffset === prev.cineOffset &&
        next.color === prev.color &&
        next.spectral === prev.spectral &&
        next.cursorThetaRad === prev.cursorThetaRad &&
        next.gateDepthCm === prev.gateDepthCm &&
        next.quality === prev.quality &&
        next.rendererBackend === prev.rendererBackend &&
        next.anatomy === prev.anatomy &&
        next.imageFormation === prev.imageFormation &&
        next.artifactLab === prev.artifactLab &&
        next.activeTool === prev.activeTool &&
        next.activeMeasurementId === prev.activeMeasurementId &&
        next.mode === prev.mode &&
        next.ui === prev.ui
      ) {
        return;
      }
      push();
    });
    const safety = setInterval(() => {
      if (!document.hidden) push();
    }, 500);
    let raf = 0;
    let frames = 0;
    let fpsT = performance.now();
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      if (useSimStore.getState().presetAnim) useSimStore.getState().tickPresetAnimation(now);
      frames++;
      if (now - fpsT > 1000) {
        useSimStore.getState().setFpsUi(frames);
        frames = 0;
        fpsT = now;
      }
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      clearInterval(safety);
      unsub();
      client.dispose();
      clientRef.current = null;
      loadedCaseRef.current = null;
    };
  }, []);

  useEffect(() => {
    const client = clientRef.current;
    if (!client || loadedCaseRef.current === caseId) return;
    loadedCaseRef.current = caseId;
    void client.loadCase(loadCaseById(caseId), buildInput(sizeRef.current));
  }, [caseId]);
}

export function buildInput(display: { width: number; height: number }): SimInput {
  const s = useSimStore.getState();
  return {
    probe: s.probe,
    patient: s.patient,
    settings: s.settings,
    modality: s.modality,
    frozen: s.frozen,
    cineOffset: s.cineOffset,
    color: s.color,
    spectral: s.spectral,
    cursorThetaRad: s.cursorThetaRad,
    gateDepthCm: s.gateDepthCm,
    quality: s.quality,
    display,
    rendererBackend: s.rendererBackend,
    anatomy: s.anatomy,
    imageFormation: s.imageFormation,
    artifactOverrides: s.artifactLab,
    needSegments: s.ui.imageSegments || s.ui.navSegments || s.ui.segmentsOpen || s.ui.reviewMode,
    needStructure:
      s.frozen ||
      s.ui.reviewMode ||
      s.activeTool !== 'none' ||
      s.activeMeasurementId !== null ||
      (!s.ui.minimal && s.ui.showTorso && !s.ui.railMini && modePolicy(s.mode).navigatorAnatomy),
  };
}
