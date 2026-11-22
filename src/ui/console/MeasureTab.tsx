import { modePolicy } from '@/app/modePolicy';
import { useSimStore, type SimStore } from '@/app/store';
import { MEASUREMENT_SPECS } from '@/simulator/measurements/protocol';
import { modalityLabel } from '@/simulator/renderer/modality';
import { viewLabel } from '@/simulator/windows/viewLabels';
import { useEffect, useRef, useState } from 'react';
import { useShallow } from 'zustand/shallow';
import { Section } from '../controls';
import { MeasurementPanel } from '../MeasurementPanel';
import { useTip } from './useTip';

const FREE_TOOL_HINT: Partial<Record<SimStore['activeTool'], string>> = {
  caliper: 'Marca los dos extremos de la distancia sobre la imagen 2D.',
  velocity: 'Haz clic en el pico del espectro Doppler.',
  vti: 'Traza la envolvente punto a punto; doble clic cierra el trazado.',
  'auto-vti': 'Marca el inicio y el fin del flujo (la eyección o el llenado) sobre el espectro.',
  time: 'Marca los dos instantes del intervalo sobre el strip.',
  slope: 'Marca el pico y luego un punto sobre la pendiente de caída.',
  simpson: 'Traza el endocardio con al menos 5 puntos; doble clic cierra.',
  tapse: 'Marca la excursión del anillo tricuspídeo en el modo M.',
};

/** Protocol measurements, free tools and the captured list. */
export function MeasureTab() {
  const s = useSimStore(
    useShallow((st) => ({
      activeMeasurementId: st.activeMeasurementId,
      activeTool: st.activeTool,
      measurements: st.measurements,
      mode: st.mode,
      removeMeasurement: st.removeMeasurement,
      setActiveMeasurement: st.setActiveMeasurement,
      setActiveTool: st.setActiveTool,
    })),
  );
  const tip = useTip();
  // the image tools first (2D, then M-mode), the spectrum ones after; nothing armed is the catalogue itself, so it
  // needs no button of its own (decision 189)
  const tools: { id: typeof s.activeTool; label: string; title?: string }[] = [
    { id: 'caliper', label: 'Caliper', title: 'Distancia lineal (2D)' },
    {
      id: 'simpson',
      label: 'Simpson',
      title: 'Volumen del VI por discos (trazado del endocardio)',
    },
    { id: 'tapse', label: 'TAPSE', title: 'Excursión vertical en modo M' },
    { id: 'time', label: 'Tiempo', title: 'Intervalo de tiempo' },
    { id: 'velocity', label: 'Velocidad', title: 'Velocidad pico sobre el espectro' },
    { id: 'vti', label: 'VTI', title: 'Trazado manual del envelope' },
    { id: 'auto-vti', label: 'VTI auto', title: 'Envolvente automática entre dos instantes' },
    { id: 'slope', label: 'TD', title: 'Tiempo de desaceleración (pico → pendiente)' },
  ];
  const capturing = Boolean(s.activeMeasurementId) || s.activeTool !== 'none';
  if (capturing) {
    const spec = s.activeMeasurementId
      ? MEASUREMENT_SPECS.find((m) => m.id === s.activeMeasurementId)
      : undefined;
    const tool = tools.find((t) => t.id === s.activeTool);
    const cancel = () =>
      s.activeMeasurementId ? s.setActiveMeasurement(null) : s.setActiveTool('none');
    return (
      <>
        <Section title="Midiendo">
          {/* a live region: screen readers and the E2E flow learn which tool is armed and how to use it */}
          <div role="status">
            <div className="capture-head">
              <span className="capture-dot" aria-hidden="true" />
              <b>{spec ? spec.label : (tool?.title ?? tool?.label ?? s.activeTool)}</b>
            </div>
            {spec && (
              <div className="small">
                {spec.modalities.map(modalityLabel).join('/')} ·{' '}
                {spec.views.map(viewLabel).join('/')}
              </div>
            )}
            <p className="capture-hint">{spec ? spec.instruction : FREE_TOOL_HINT[s.activeTool]}</p>
          </div>
          <button className="capture-cancel ghost" onClick={cancel}>
            Cancelar
            <kbd className="kbd" aria-hidden="true">
              Esc
            </kbd>
          </button>
        </Section>
        <Section title={`Mediciones (${s.measurements.length})`}>
          <MeasureList s={s} showView={modePolicy(s.mode).showViewFeedback} />
        </Section>
      </>
    );
  }
  return (
    <>
      <Section title="Protocolo de medición">
        <MeasurementPanel />
      </Section>
      <Section
        title="Herramientas libres"
        tip={tip('Medidas sin evaluación de técnica. Congela (Espacio) y elige una herramienta.')}
      >
        <div className="tool-grid" role="group" aria-label="Herramienta de medición">
          {tools.map((t) => (
            <button key={t.id} data-tip={t.title} onClick={() => s.setActiveTool(t.id)}>
              {t.label}
            </button>
          ))}
        </div>
        <MeasureList s={s} showView={modePolicy(s.mode).showViewFeedback} />
      </Section>
    </>
  );
}

/** `showView` false (exam) leaves out the view each measurement was taken in: it names the view (decision 154). */
function MeasureList({
  s,
  showView,
}: {
  s: Pick<SimStore, 'measurements' | 'removeMeasurement'>;
  showView: boolean;
}) {
  const line = (m: SimStore['measurements'][number]) =>
    `${m.value.toFixed(m.kind === 'time' || m.kind === 'volume' ? 0 : 2)} ${m.units}` +
    (m.derived?.gradientMmHg !== undefined ? ` · ${m.derived.gradientMmHg.toFixed(0)} mmHg` : '') +
    (m.derived?.meanGradientMmHg !== undefined
      ? ` · media ${m.derived.meanGradientMmHg.toFixed(0)} mmHg`
      : '');
  const [status, setStatus] = useState<string | null>(null);
  const statusTimer = useRef<number | null>(null);
  useEffect(() => () => window.clearTimeout(statusTimer.current ?? undefined), []);
  const copy = async () => {
    const text = s.measurements.map((m) => `${m.label}: ${line(m)}`).join('\n');
    // the browser says whether it took the text: over plain http there is no clipboard at all (decision 201)
    try {
      await navigator.clipboard.writeText(text);
      setStatus('Copiadas');
    } catch {
      setStatus('El navegador no dejó copiar');
    }
    window.clearTimeout(statusTimer.current ?? undefined);
    statusTimer.current = window.setTimeout(() => setStatus(null), 2500);
  };
  if (!s.measurements.length) return null;
  return (
    <div className="measure-list">
      {/* the figures in a column of their own, and the list as text for a report (decision 200) */}
      <div className="measure-head">
        {status && (
          <span className="small" role="status">
            {status}
          </span>
        )}
        <button className="ghost" onClick={() => void copy()}>
          Copiar mediciones
        </button>
      </div>
      {s.measurements.map((m) => (
        <div key={m.id} className="measure-row">
          <span className="m-label">
            {m.label}
            <span className="m-src">
              {modalityLabel(m.modality)}
              {showView && m.sourceViewId ? ` · ${viewLabel(m.sourceViewId)}` : ''}
            </span>
          </span>
          <span className="m-value">{line(m)}</span>
          <button
            className="icon-btn"
            aria-label="Eliminar medición"
            onClick={() => s.removeMeasurement(m.id)}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
