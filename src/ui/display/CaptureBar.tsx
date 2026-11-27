import { examClosed } from '@/app/modePolicy';
import { useSimStore } from '@/app/store';
import { MEASUREMENT_SPECS } from '@/simulator/measurements/protocol';
import { modalityLabel } from '@/simulator/renderer/modality';
import { useShallow } from 'zustand/shallow';
import { captureModalityCheck } from '../console/measureContext';
import { captureStepPrompt, minPointsToFinish } from './measurementPreview';
import { pendingSession, usePendingPointCount } from './pendingSession';

const TOOL_TITLE: Record<string, string> = {
  caliper: 'Caliper (2D)',
  simpson: 'Simpson (volumen)',
  tapse: 'TAPSE (modo M)',
  time: 'Tiempo',
  velocity: 'Velocidad',
  vti: 'VTI manual',
  'auto-vti': 'VTI auto',
  slope: 'TD / PHT',
};

/**
 * Compact dark-glass floating capture bar in the upper-left margin outside the ultrasound cone.
 * Appears only while a measurement tool is armed so the operator's eyes stay on the image.
 */
export function CaptureBar() {
  const s = useSimStore(
    useShallow((st) => ({
      activeTool: st.activeTool,
      activeMeasurementId: st.activeMeasurementId,
      modality: st.modality,
      frozen: st.frozen,
      closed: examClosed(st),
      setModality: st.setModality,
      toggleFreeze: st.toggleFreeze,
      setActiveMeasurement: st.setActiveMeasurement,
      setActiveTool: st.setActiveTool,
    })),
  );
  const pendingCount = usePendingPointCount();

  if (s.activeTool === 'none') return null;

  const spec = s.activeMeasurementId
    ? MEASUREMENT_SPECS.find((m) => m.id === s.activeMeasurementId)
    : undefined;
  const modCheck = captureModalityCheck(spec, s.activeTool, s.modality);
  const targetModLabel = modCheck.suggestedModality
    ? modalityLabel(modCheck.suggestedModality)
    : null;
  const prompt = captureStepPrompt(
    s.activeTool,
    pendingCount,
    s.frozen,
    modCheck.ok,
    targetModLabel,
  );
  const minFinish = minPointsToFinish(s.activeTool);
  const canFinish = Number.isFinite(minFinish) && pendingCount >= minFinish;
  const cancel = () =>
    s.activeMeasurementId ? s.setActiveMeasurement(null) : s.setActiveTool('none');

  return (
    <div className="capture-bar" role="status" aria-live="polite">
      <div className="capture-bar-main">
        <span className="capture-dot" aria-hidden="true" />
        <b className="capture-bar-title">
          {spec ? spec.shortLabel : (TOOL_TITLE[s.activeTool] ?? s.activeTool)}
        </b>
        <span className="capture-bar-step">{prompt}</span>
      </div>
      <div className="capture-bar-actions">
        {!modCheck.ok && modCheck.suggestedModality && !s.closed && (
          <button
            type="button"
            className="capture-bar-primary"
            onClick={() => s.setModality(modCheck.suggestedModality!)}
          >
            Cambiar a {targetModLabel}
          </button>
        )}
        {modCheck.ok && !s.frozen && !s.closed && (
          <button
            type="button"
            className="capture-bar-primary"
            onClick={() => s.toggleFreeze()}
          >
            Congelar
          </button>
        )}
        {pendingCount > 0 && (
          <button type="button" onClick={() => pendingSession.undo()} title="Deshacer último punto (Retroceso)">
            Deshacer
          </button>
        )}
        {canFinish && (
          <button
            type="button"
            className="capture-bar-primary"
            onClick={() => pendingSession.finish()}
            title="Cerrar y calcular trazado (Enter)"
          >
            Finalizar
          </button>
        )}
        <button type="button" onClick={cancel} title="Cancelar medición (Esc)">
          Esc
        </button>
      </div>
    </div>
  );
}
