import { examClosed, modePolicy } from '@/app/modePolicy';
import { useSimStore, type SimStore } from '@/app/store';
import { fractionalAreaChange } from '@/clinical/formulas';
import { biplaneVolume } from '@/education/report';
import { MEASUREMENT_SPECS } from '@/simulator/measurements/protocol';
import { modalityLabel } from '@/simulator/renderer/modality';
import { viewLabel } from '@/simulator/windows/viewLabels';
import { useEffect, useRef, useState } from 'react';
import { useShallow } from 'zustand/shallow';
import { Section } from '../controls';
import { runAutoChamberVolume } from '../display/measurementTools';
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
      frozen: st.frozen,
      measurements: st.measurements,
      mode: st.mode,
      removeMeasurement: st.removeMeasurement,
      setActiveMeasurement: st.setActiveMeasurement,
      setActiveTool: st.setActiveTool,
      // a finished exam keeps the measurements it was scored on (`examClosed`)
      closed: examClosed(st),
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
            {!s.frozen && (
              <p className="capture-hint">
                Se mide sobre la imagen congelada: el primer clic sobre la imagen en vivo la congela.
              </p>
            )}
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
      <Section
        title="Volumen y FE automáticos"
        tip={tip(
          'Traza automáticamente el endocardio en TD y TS sobre el cine 2D actual (A4C/A2C para VI · enfocada en VD o A4C para VD).',
        )}
      >
        <AutoEfPanel />
      </Section>
      <Section title="Protocolo de medición">
        <MeasurementPanel />
      </Section>
      <Section
        title="Herramientas libres"
        tip={tip('Medidas sin evaluación de técnica. Congela (Espacio) y elige una herramienta.')}
      >
        <div className="tool-grid" role="group" aria-label="Herramienta de medición">
          {tools.map((t) => (
            <button
              key={t.id}
              data-tip={t.title}
              onClick={() => s.setActiveTool(t.id)}
              disabled={s.closed}
            >
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
  s: Pick<SimStore, 'measurements' | 'removeMeasurement'> & { closed: boolean };
  showView: boolean;
}) {
  const primaryValue = (m: SimStore['measurements'][number]) => {
    const decimals = m.kind === 'time' || m.kind === 'volume' ? 0 : m.kind === 'vti' ? 1 : 2;
    return `${m.value.toFixed(decimals)} ${m.units}`;
  };
  const secondaryMeta = (m: SimStore['measurements'][number]) => {
    const parts: string[] = [];
    if (m.derived?.gradientMmHg !== undefined) {
      parts.push(`ΔP ${m.derived.gradientMmHg.toFixed(0)} mmHg`);
    }
    if (m.derived?.vmaxMps !== undefined && m.derived.vmaxMps > 0) {
      parts.push(
        m.modality === 'pw' || m.modality === 'tdi'
          ? `Vmáx ${Math.round(m.derived.vmaxMps * 100)} cm/s`
          : `Vmáx ${m.derived.vmaxMps.toFixed(2)} m/s`,
      );
    }
    if (m.derived?.meanGradientMmHg !== undefined) {
      const mg = m.derived.meanGradientMmHg;
      parts.push(`ΔPm ${mg < 10 ? mg.toFixed(1) : mg.toFixed(0)} mmHg`);
    }
    if (m.derived?.areaCm2 !== undefined && m.derived.areaCm2 > 0) {
      parts.push(`Área ${m.derived.areaCm2.toFixed(1)} cm²`);
    }
    return parts.join(' · ');
  };
  const line = (m: SimStore['measurements'][number]) => {
    const sub = secondaryMeta(m);
    return sub ? `${primaryValue(m)} · ${sub}` : primaryValue(m);
  };
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
      {s.measurements.map((m) => {
        const sub = secondaryMeta(m);
        return (
          <div key={m.id} className="measure-row">
            <span className="m-label">
              {m.label}
              <span className="m-src">
                {modalityLabel(m.modality)}
                {showView && m.sourceViewId ? ` · ${viewLabel(m.sourceViewId)}` : ''}
              </span>
              {sub && <span className="m-sub">{sub}</span>}
            </span>
            <span className="m-value">{primaryValue(m)}</span>
            <button
              className="icon-btn"
              aria-label="Eliminar medición"
              onClick={() => s.removeMeasurement(m.id)}
              disabled={s.closed}
            >
              ×
            </button>
          </div>
        );
      })}
    </div>
  );
}

function latestById(ms: SimStore['measurements'], id: string) {
  for (let i = ms.length - 1; i >= 0; i--) if (ms[i]!.measurementId === id) return ms[i]!;
  return null;
}

/** One-click automatic LV & RV endocardial tracing, volume, EF, and RV FAC over the acquired 2D cine. */
function AutoEfPanel() {
  const s = useSimStore(
    useShallow((st) => ({
      measurements: st.measurements,
      frozen: st.frozen,
      mode: st.mode,
      closed: examClosed(st),
      toggleFreeze: st.toggleFreeze,
      jumpCinePhase: st.jumpCinePhase,
    })),
  );
  const [busy, setBusy] = useState<'lv' | 'rv' | null>(null);
  const [notice, setNotice] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);

  const run = async (chamber: 'lv' | 'rv') => {
    if (busy || s.closed) return;
    setBusy(chamber);
    setNotice(null);
    try {
      const res = await runAutoChamberVolume(chamber);
      if (!res.ok) {
        setNotice({ kind: 'err', text: res.message });
      } else {
        setNotice({
          kind: 'ok',
          text:
            chamber === 'lv'
              ? `VI trazado en TD y TS (FEVI ${res.efPct.toFixed(0)} %).`
              : `VD trazado en TD y TS (FEVD ${res.efPct.toFixed(0)} %, CAF ${res.facPct.toFixed(0)} %).`,
        });
      }
    } catch (e: unknown) {
      setNotice({
        kind: 'err',
        text: e instanceof Error ? e.message : 'Error al calcular volumen automático.',
      });
    } finally {
      setBusy(null);
    }
  };

  const inspectFrame = (target: 'ed' | 'es', frameId: number) => {
    if (s.closed) return;
    if (!s.frozen) s.toggleFreeze();
    s.jumpCinePhase(target, frameId);
  };

  const lvEdvM = latestById(s.measurements, 'lv-edv-simpson');
  const lvEsvM = latestById(s.measurements, 'lv-esv-simpson');
  const lvEdvBi = biplaneVolume(s.measurements, 'lv-edv-simpson');
  const lvEsvBi = biplaneVolume(s.measurements, 'lv-esv-simpson');
  const hasLvBiplane = Boolean(lvEdvBi && lvEsvBi && lvEdvBi.volumeMl > 0);
  const lvEdv = hasLvBiplane ? lvEdvBi!.volumeMl : (lvEdvM?.value ?? null);
  const lvEsv = hasLvBiplane ? lvEsvBi!.volumeMl : (lvEsvM?.value ?? null);
  const lvEf = lvEdv !== null && lvEsv !== null && lvEdv > 0 ? ((lvEdv - lvEsv) / lvEdv) * 100 : null;

  const rvEdvM = latestById(s.measurements, 'rv-edv');
  const rvEsvM = latestById(s.measurements, 'rv-esv');
  const rvEdv = rvEdvM?.value ?? null;
  const rvEsv = rvEsvM?.value ?? null;
  const rvEf = rvEdv !== null && rvEsv !== null && rvEdv > 0 ? ((rvEdv - rvEsv) / rvEdv) * 100 : null;
  const rvEdArea = rvEdvM?.derived?.areaCm2 ?? null;
  const rvEsArea = rvEsvM?.derived?.areaCm2 ?? null;
  const rvFac =
    rvEdArea !== null && rvEsArea !== null && rvEdArea > 0
      ? fractionalAreaChange(rvEdArea, rvEsArea)
      : null;

  return (
    <div className="auto-ef">
      <div className="auto-ef-actions" role="group" aria-label="Cálculo automático de volumen y FE">
        <button
          type="button"
          onClick={() => void run('lv')}
          disabled={s.closed || busy !== null}
          data-tip="Calcula VTD, VTS y FEVI (Simpson) desde el contorno endocárdico en A4C o A2C; repite en ambos planos para biplano"
        >
          {busy === 'lv' ? 'Midiendo VI…' : 'Auto-FE VI'}
        </button>
        <button
          type="button"
          onClick={() => void run('rv')}
          disabled={s.closed || busy !== null}
          data-tip="Calcula VTD, VTS, FEVD estimada (2D) y CAF del VD desde vista enfocada en VD o A4C"
        >
          {busy === 'rv' ? 'Midiendo VD…' : 'Auto-FE VD'}
        </button>
      </div>
      {notice && (
        <div
          className={`auto-ef-notice ${notice.kind === 'err' ? 'err' : 'ok'}`}
          role="status"
        >
          {notice.text}
        </div>
      )}
      {lvEf !== null && lvEdv !== null && lvEsv !== null && (
        <div className="auto-ef-card" data-auto-ef="lv">
          <div className="auto-ef-head">
            <span className="auto-ef-title">
              VI · {hasLvBiplane ? 'Simpson biplano' : 'Simpson'}
            </span>
            <span className="auto-ef-primary">FEVI {lvEf.toFixed(0)} %</span>
          </div>
          <div className="auto-ef-metrics">
            <span>VTD {lvEdv.toFixed(0)} mL</span>
            <span>VTS {lvEsv.toFixed(0)} mL</span>
            <span>VS {(lvEdv - lvEsv).toFixed(0)} mL</span>
          </div>
          {lvEdvM && lvEsvM && (
            <div className="auto-ef-jump">
              <button
                type="button"
                className="ghost"
                disabled={s.closed}
                onClick={() => inspectFrame('ed', lvEdvM.frameId)}
              >
                Ver TD
              </button>
              <button
                type="button"
                className="ghost"
                disabled={s.closed}
                onClick={() => inspectFrame('es', lvEsvM.frameId)}
              >
                Ver TS
              </button>
            </div>
          )}
        </div>
      )}
      {rvEf !== null && rvEdv !== null && rvEsv !== null && (
        <div className="auto-ef-card" data-auto-ef="rv">
          <div className="auto-ef-head">
            <span className="auto-ef-title">VD · 2D</span>
            <span className="auto-ef-primary">
              FEVD {rvEf.toFixed(0)} %{rvFac !== null ? ` · CAF ${rvFac.toFixed(0)} %` : ''}
            </span>
          </div>
          <div className="auto-ef-metrics">
            <span>VTD {rvEdv.toFixed(0)} mL</span>
            <span>VTS {rvEsv.toFixed(0)} mL</span>
            <span>VS {(rvEdv - rvEsv).toFixed(0)} mL</span>
            {rvEdArea !== null && rvEsArea !== null && (
              <span>
                Área {rvEdArea.toFixed(1)}/{rvEsArea.toFixed(1)} cm²
              </span>
            )}
          </div>
          {rvEdvM && rvEsvM && (
            <div className="auto-ef-jump">
              <button
                type="button"
                className="ghost"
                disabled={s.closed}
                onClick={() => inspectFrame('ed', rvEdvM.frameId)}
              >
                Ver TD
              </button>
              <button
                type="button"
                className="ghost"
                disabled={s.closed}
                onClick={() => inspectFrame('es', rvEsvM.frameId)}
              >
                Ver TS
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
