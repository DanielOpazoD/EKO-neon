import { examClosed, modePolicy, showsCaseIdentity } from '@/app/modePolicy';
import { formatCaptureCaption } from '@/app/studyCapture';
import { useHudStore, useSimStore, type SimStore } from '@/app/store';
import { strokeVolume } from '@/clinical/formulas';
import { aggregateProtocolMeasurement, deriveCalculations } from '@/education/report';
import {
  MEASUREMENT_SPECS,
  type MeasurementTool,
} from '@/simulator/measurements/protocol';
import { modalityLabel } from '@/simulator/renderer/modality';
import { viewLabel } from '@/simulator/windows/viewLabels';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useShallow } from 'zustand/shallow';
import { Section } from '../controls';
import {
  AUTO_MEASURE_SPECS,
  cancelGuidedAutoMeasure,
  runGuidedAutoMeasure,
  type AutoMeasureActionId,
  type OrchestratorProgress,
} from '../display/autoMeasureOrchestrator';
import { captureStepPrompt, minPointsToFinish } from '../display/measurementPreview';
import { pendingSession, usePendingPointCount } from '../display/pendingSession';
import { MeasurementPanel } from '../MeasurementPanel';
import { ViewQuickMenu } from '../ViewQuickMenu';
import {
  captureModalityCheck,
  isFreeToolCompatible,
  isTimeInCineWindow,
  phaseTag,
  summarizeAutoEf,
} from './measureContext';
import { useTip } from './useTip';

type MeasureSubTab = 'medir' | 'capturas';

const FREE_TOOL_HINT: Partial<Record<MeasurementTool, string>> = {
  caliper: 'Marca los dos extremos de la distancia sobre la imagen 2D.',
  velocity: 'Haz clic en el pico del espectro Doppler.',
  vti: 'Traza la envolvente punto a punto; doble clic o Enter cierra el trazado.',
  'auto-vti': 'Marca el inicio y el fin del flujo sobre el espectro.',
  time: 'Marca los dos instantes del intervalo sobre la tira.',
  slope: 'Marca el pico y luego un punto más abajo sobre la pendiente de caída.',
  simpson: 'Traza el endocardio con al menos 5 puntos; doble clic o Enter cierra.',
  tapse: 'Marca la excursión del anillo tricuspídeo en el modo M.',
};

const TOOLS_2D: { id: MeasurementTool; label: string; title: string }[] = [
  { id: 'caliper', label: 'Caliper', title: 'Distancia lineal sobre imagen 2D' },
  {
    id: 'simpson',
    label: 'Simpson',
    title: 'Trazado endocárdico manual (volumen por discos)',
  },
];

const TOOLS_STRIP: { id: MeasurementTool; label: string; title: string }[] = [
  { id: 'velocity', label: 'Velocidad', title: 'Velocidad pico y gradiente en PW/CW/TDI' },
  { id: 'auto-vti', label: 'VTI auto', title: 'Envolvente automática entre dos instantes' },
  { id: 'vti', label: 'VTI manual', title: 'Trazado manual de la envolvente espectral' },
  { id: 'slope', label: 'TD / PHT', title: 'Tiempo de desaceleración o hemipresión (pico → pendiente)' },
  { id: 'time', label: 'Tiempo', title: 'Intervalo temporal sobre tira Doppler o modo M' },
  { id: 'tapse', label: 'TAPSE', title: 'Excursión vertical sobre modo M' },
];

const ALL_TOOLS = [...TOOLS_2D, ...TOOLS_STRIP];

function primaryValue(m: SimStore['measurements'][number]): string {
  const decimals =
    m.kind === 'time' || m.kind === 'volume' ? 0 : m.kind === 'vti' || m.units === 'cm/s' ? 1 : 2;
  return `${m.value.toFixed(decimals)} ${m.units}`;
}

function secondaryMeta(m: SimStore['measurements'][number]): string {
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
}

function techniqueBadge(
  m: SimStore['measurements'][number],
  showFeedback: boolean,
): { level: 'ok' | 'warn' | 'bad'; label: string; tip: string } | null {
  if (!showFeedback || !m.technique) return null;
  const bad = m.technique.findings.some((f) => f.level === 'invalid');
  const warn = !bad && m.technique.findings.some((f) => f.level === 'warn');
  const notes = m.technique.findings
    .filter((f) => f.level !== 'ok')
    .map((f) => f.message)
    .join(' ');
  return {
    level: bad ? 'bad' : warn ? 'warn' : 'ok',
    label: bad ? 'inválida' : warn ? 'revisar' : 'ok',
    tip: notes || 'Técnica adecuada',
  };
}

/**
 * Simplified 1-click clinical measurement workspace:
 * - «Medir»: compact context bar + unified 1-click «Automedición» hub (auto-switches window,
 *   posture, and modality while deactivating conflicting modes) + collapsible manual protocol/tools.
 * - «Capturas (N)»: live derived calculations + measurement worksheet with frame jump, copy & Informe.
 * - «Midiendo»: step-by-step guided capture card when using manual caliper/trace tools.
 */
export function MeasureTab() {
  const s = useSimStore(
    useShallow((st) => ({
      activeMeasurementId: st.activeMeasurementId,
      activeTool: st.activeTool,
      captures: st.captures,
      caseId: st.caseId,
      examFinished: st.examFinished,
      frozen: st.frozen,
      measurements: st.measurements,
      modality: st.modality,
      mode: st.mode,
      truth: st.truth,
      jumpCinePhase: st.jumpCinePhase,
      removeCapture: st.removeCapture,
      removeMeasurement: st.removeMeasurement,
      setActiveMeasurement: st.setActiveMeasurement,
      setActiveTool: st.setActiveTool,
      setModality: st.setModality,
      setUi: st.setUi,
      toggleFreeze: st.toggleFreeze,
      closed: examClosed(st),
    })),
  );
  const [subTab, setSubTab] = useState<MeasureSubTab>('medir');
  const [progress, setProgress] = useState<OrchestratorProgress | null>(null);
  const [notice, setNotice] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);
  const tip = useTip();
  const policy = modePolicy(s.mode);
  const showView = policy.showViewFeedback;
  const showIdentity = showsCaseIdentity(s.mode, s.examFinished);

  // Cancel any running guided auto-measurement on unmount or exam closure
  useEffect(() => () => cancelGuidedAutoMeasure(), []);
  useEffect(() => {
    if (s.closed) cancelGuidedAutoMeasure();
  }, [s.closed]);

  // Narrow Zustand HUD selectors: only re-render when view, frozen frameId, or 0.5 s quantized cine window changes.
  const currentViewId = useHudStore((h) => (showView ? (h.hud?.view?.bestViewId ?? null) : null));
  const currentViewScore = useHudStore((h) => (showView ? (h.hud?.view?.score ?? null) : null));
  const displayedFrameId = useHudStore((h) => (h.hud?.frozen ? h.hud.frameId : null));
  const cineBucket = useHudStore((h) =>
    h.hud && h.hud.cineLength > 0
      ? `${Math.floor(h.hud.cineWindow.startS * 2)}:${Math.ceil(h.hud.cineWindow.endS * 2)}`
      : '',
  );
  const cineWindow = useMemo(() => {
    void cineBucket;
    const h = useHudStore.getState().hud;
    return h && h.cineLength > 0 ? h.cineWindow : null;
  }, [cineBucket]);
  const pendingCount = usePendingPointCount();

  const runAction = async (actionId: AutoMeasureActionId) => {
    if (s.closed) return;
    setNotice(null);
    const res = await runGuidedAutoMeasure(actionId, setProgress);
    setProgress(null);
    if (!res.cancelled) {
      setNotice({ kind: res.ok ? 'ok' : 'err', text: res.message });
    }
  };

  const inspectFrame = (frameId: number, target: 'ed' | 'es' = 'ed') => {
    if (s.closed) return;
    cancelGuidedAutoMeasure();
    setProgress(null);
    if (!s.frozen) s.toggleFreeze();
    s.jumpCinePhase(target, frameId);
  };

  const capturing = Boolean(s.activeMeasurementId) || s.activeTool !== 'none';
  if (capturing) {
    const spec = s.activeMeasurementId
      ? MEASUREMENT_SPECS.find((m) => m.id === s.activeMeasurementId)
      : undefined;
    const tool = ALL_TOOLS.find((t) => t.id === s.activeTool);
    const modCheck = captureModalityCheck(spec, s.activeTool, s.modality);
    const targetModLabel = modCheck.suggestedModality
      ? modalityLabel(modCheck.suggestedModality)
      : null;
    const stepText = captureStepPrompt(
      s.activeTool,
      pendingCount,
      s.frozen,
      modCheck.ok,
      targetModLabel,
    );
    const minFinish = minPointsToFinish(s.activeTool);
    const canFinish = Number.isFinite(minFinish) && pendingCount >= minFinish;
    const showPhaseJump =
      s.frozen &&
      (s.modality === '2d' || s.modality === 'color') &&
      (spec?.phase === 'ed' ||
        spec?.phase === 'es' ||
        s.activeTool === 'caliper' ||
        s.activeTool === 'simpson');
    const cancel = () =>
      s.activeMeasurementId ? s.setActiveMeasurement(null) : s.setActiveTool('none');

    return (
      <>
        <Section title="Midiendo">
          <div className="capture-card" role="status">
            <div className="capture-head">
              <span className="capture-dot" aria-hidden="true" />
              <b>{spec ? spec.label : (tool?.title ?? tool?.label ?? s.activeTool)}</b>
            </div>
            {spec && (
              <div className="capture-tags">
                {spec.modalities.map((mod) => (
                  <span key={mod} className={`mod-chip${mod === s.modality ? ' match' : ''}`}>
                    {modalityLabel(mod)}
                  </span>
                ))}
                {phaseTag(spec.phase) && <span className="mod-chip">{phaseTag(spec.phase)}</span>}
                <span className="capture-views">{spec.views.map(viewLabel).join(' · ')}</span>
              </div>
            )}

            <div className="capture-step-box">
              <p className="capture-step-text">{stepText}</p>
              <div className="capture-quick-actions">
                {!modCheck.ok && modCheck.suggestedModality && (
                  <button
                    type="button"
                    className="btn-primary-soft"
                    disabled={s.closed}
                    onClick={() => s.setModality(modCheck.suggestedModality!)}
                  >
                    Cambiar a {targetModLabel}
                  </button>
                )}
                {modCheck.ok && !s.frozen && (
                  <button
                    type="button"
                    className="btn-primary-soft"
                    disabled={s.closed}
                    onClick={() => s.toggleFreeze()}
                  >
                    Congelar cuadro
                  </button>
                )}
                {modCheck.ok && showPhaseJump && (
                  <>
                    <button
                      type="button"
                      className={spec?.phase === 'ed' ? 'btn-primary-soft' : 'ghost'}
                      disabled={s.closed}
                      onClick={() => s.jumpCinePhase('ed')}
                      data-tip="Saltar a telediástole (onda R)"
                    >
                      Ir a TD
                    </button>
                    <button
                      type="button"
                      className={spec?.phase === 'es' ? 'btn-primary-soft' : 'ghost'}
                      disabled={s.closed}
                      onClick={() => s.jumpCinePhase('es')}
                      data-tip="Saltar a telesístole (volumen mínimo)"
                    >
                      Ir a TS
                    </button>
                  </>
                )}
                {pendingCount > 0 && (
                  <button
                    type="button"
                    className="ghost"
                    onClick={() => pendingSession.undo()}
                  >
                    Deshacer punto
                  </button>
                )}
                {canFinish && (
                  <button
                    type="button"
                    className="btn-primary-soft"
                    onClick={() => pendingSession.finish()}
                  >
                    Finalizar trazado
                  </button>
                )}
              </div>
            </div>

            <p className="capture-hint">
              {spec ? spec.instruction : FREE_TOOL_HINT[s.activeTool as MeasurementTool]}
            </p>
          </div>
          <button className="capture-cancel ghost" onClick={cancel}>
            Cancelar
            <kbd className="kbd" aria-hidden="true">
              Esc
            </kbd>
          </button>
        </Section>
        {s.measurements.length > 0 && (
          <Section title={`Capturas (${s.measurements.length})`}>
            <MeasureList
              s={s}
              showView={showView}
              displayedFrameId={displayedFrameId}
              cineWindow={cineWindow}
              onInspect={inspectFrame}
            />
          </Section>
        )}
      </>
    );
  }

  return (
    <div className="measure-workspace">
      <div className="measure-subtabs" role="tablist" aria-label="Vista de mediciones">
        <button
          type="button"
          role="tab"
          aria-selected={subTab === 'medir'}
          className={subTab === 'medir' ? 'active' : ''}
          onClick={() => setSubTab('medir')}
        >
          Medir
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={subTab === 'capturas'}
          className={subTab === 'capturas' ? 'active' : ''}
          onClick={() => setSubTab('capturas')}
        >
          Capturas
          {s.measurements.length + s.captures.length > 0 && (
            <span className="measure-subtab-count">
              {s.measurements.length + s.captures.length}
            </span>
          )}
        </button>
      </div>

      {subTab === 'medir' ? (
        <>
          <div className="measure-ctx">
            <div className="measure-ctx-info">
              <span className="mod-chip match">{modalityLabel(s.modality)}</span>
              {showView && (
                <ViewQuickMenu placement="down-left" currentViewId={currentViewId}>
                  <span className="measure-ctx-view">
                    <i
                      className={`ctx-dot ${
                        (currentViewScore ?? 0) >= 75
                          ? 'ok'
                          : (currentViewScore ?? 0) >= 50
                            ? 'warn'
                            : 'muted'
                      }`}
                      aria-hidden="true"
                    />
                    {currentViewId ? viewLabel(currentViewId) : 'Cambiar vista'}
                  </span>
                </ViewQuickMenu>
              )}
            </div>
            <button
              type="button"
              className={`measure-freeze-chip${s.frozen ? ' frozen' : ''}`}
              disabled={s.closed}
              onClick={() => {
                cancelGuidedAutoMeasure();
                setProgress(null);
                s.toggleFreeze();
              }}
              data-tip={
                s.frozen
                  ? 'Reanudar imagen en vivo (Espacio)'
                  : 'Congelar imagen para medir (Espacio)'
              }
            >
              {s.frozen ? 'Congelada · Reanudar' : 'En vivo · Congelar'}
            </button>
          </div>

          {progress && (
            <div className="auto-run-bar" role="status" aria-live="polite">
              <span className="auto-run-main">
                <span className="capture-dot" aria-hidden="true" />
                <span className="auto-run-text">{progress.phaseLabel}</span>
              </span>
              <button
                type="button"
                className="ghost"
                onClick={() => {
                  cancelGuidedAutoMeasure();
                  setProgress(null);
                }}
              >
                Cancelar
              </button>
            </div>
          )}

          {notice && !progress && (
            <div
              className={`auto-ef-notice auto-notice-row ${notice.kind === 'err' ? 'err' : 'ok'}`}
              role="status"
            >
              <span>{notice.text}</span>
              <button
                type="button"
                className="icon-btn"
                aria-label="Cerrar aviso"
                onClick={() => setNotice(null)}
              >
                ×
              </button>
            </div>
          )}

          {policy.presetsEnabled ? (
            <>
              <Section
                title="Automedición (1 clic)"
                tip={tip(
                  'Un solo clic cambia automáticamente a la ventana correcta, activa el modo necesario (2D, PW, CW, TDI o M), desactiva los demás y coloca la medición editable.',
                )}
              >
                <OneClickAutoHub
                  measurements={s.measurements}
                  modality={s.modality}
                  currentViewId={currentViewId}
                  displayedFrameId={displayedFrameId}
                  cineWindow={cineWindow}
                  showViewFeedback={showView}
                  showIdentity={showIdentity}
                  truth={s.truth}
                  closed={s.closed}
                  runningActionId={progress?.actionId ?? null}
                  onRun={(id) => void runAction(id)}
                  onInspect={inspectFrame}
                />
              </Section>

              <details className="adv manual-details">
                <summary>Medición manual (Protocolo y herramientas)</summary>
                <div className="manual-details-body">
                  <Section title="Protocolo completo">
                    <MeasurementPanel />
                  </Section>
                  <FreeToolsSection
                    modality={s.modality}
                    closed={s.closed}
                    tip={tip}
                    onArmTool={(t) => {
                      cancelGuidedAutoMeasure();
                      setProgress(null);
                      s.setActiveTool(t);
                    }}
                  />
                </div>
              </details>
            </>
          ) : (
            <>
              <Section title="Auto sobre vista actual">
                <ExamAutoRow
                  modality={s.modality}
                  closed={s.closed}
                  runningActionId={progress?.actionId ?? null}
                  onRun={(id) => void runAction(id)}
                />
              </Section>
              <Section title="Protocolo completo">
                <MeasurementPanel />
              </Section>
              <FreeToolsSection
                modality={s.modality}
                closed={s.closed}
                tip={tip}
                onArmTool={(t) => s.setActiveTool(t)}
              />
            </>
          )}
        </>
      ) : (
        <CapturasWorksheet
          s={s}
          showView={showView}
          showIdentity={showIdentity}
          displayedFrameId={displayedFrameId}
          cineWindow={cineWindow}
          onInspect={inspectFrame}
          onBackToMeasure={() => setSubTab('medir')}
          onOpenReport={() => s.setUi({ screen: 'report' })}
        />
      )}
    </div>
  );
}

/** Unified 5-group 1-click auto-measurement hub for Practice & Guided modes. */
function OneClickAutoHub({
  measurements,
  modality,
  currentViewId,
  displayedFrameId,
  cineWindow,
  showViewFeedback,
  showIdentity,
  truth,
  closed,
  runningActionId,
  onRun,
  onInspect,
}: {
  measurements: SimStore['measurements'];
  modality: SimStore['modality'];
  currentViewId: string | null;
  displayedFrameId: number | null;
  cineWindow: { startS: number; endS: number } | null;
  showViewFeedback: boolean;
  showIdentity: boolean;
  truth: SimStore['truth'];
  closed: boolean;
  runningActionId: AutoMeasureActionId | null;
  onRun: (id: AutoMeasureActionId) => void;
  onInspect: (frameId: number, target?: 'ed' | 'es') => void;
}) {
  const summary = useMemo(
    () => summarizeAutoEf(measurements, currentViewId, modality, showViewFeedback),
    [measurements, currentViewId, modality, showViewFeedback],
  );
  const bsaM2 = truth?.bsaM2 && truth.bsaM2 > 0 ? truth.bsaM2 : null;
  const lvmi = summary.plax.massG !== null && bsaM2 ? summary.plax.massG / bsaM2 : null;
  const lavi = summary.la.volumeMl !== null && bsaM2 ? summary.la.volumeMl / bsaM2 : null;

  const agg = useMemo(
    () => ({
      lvotVti: aggregateProtocolMeasurement(measurements, 'lvot-vti'),
      avVti: aggregateProtocolMeasurement(measurements, 'av-vti'),
      lvotDiam: aggregateProtocolMeasurement(measurements, 'lvot-diameter'),
      mitralE: aggregateProtocolMeasurement(measurements, 'mitral-e'),
      mitralA: aggregateProtocolMeasurement(measurements, 'mitral-a'),
      mitralDt: aggregateProtocolMeasurement(measurements, 'mitral-dt'),
      epSept: aggregateProtocolMeasurement(measurements, 'e-prime-septal'),
      epLat: aggregateProtocolMeasurement(measurements, 'e-prime-lateral'),
      rvSPrime: aggregateProtocolMeasurement(measurements, 'rv-s-prime'),
      rvotAt: aggregateProtocolMeasurement(measurements, 'rvot-acceleration-time'),
      trVmax: aggregateProtocolMeasurement(measurements, 'tr-vmax'),
      tapse: aggregateProtocolMeasurement(measurements, 'tapse'),
    }),
    [measurements],
  );

  const dopplerSvMl =
    agg.lvotVti && agg.lvotDiam && agg.lvotDiam.value > 0
      ? strokeVolume(agg.lvotDiam.value, agg.lvotVti.value)
      : null;
  const continuityAvaCm2 =
    dopplerSvMl !== null && agg.avVti && agg.avVti.value > 0
      ? dopplerSvMl / agg.avVti.value
      : null;
  const velocityRatio =
    agg.lvotVti && agg.avVti && agg.avVti.value > 0
      ? agg.lvotVti.value / agg.avVti.value
      : null;

  const eaRatio =
    agg.mitralE && agg.mitralA && agg.mitralA.value > 0
      ? agg.mitralE.value / agg.mitralA.value
      : null;
  const ePrimeAvgCmps =
    agg.epSept && agg.epLat
      ? 0.5 * (agg.epSept.value + agg.epLat.value)
      : (agg.epSept?.value ?? agg.epLat?.value ?? null);
  const eOverEPrime =
    agg.mitralE && ePrimeAvgCmps && ePrimeAvgCmps > 0
      ? (agg.mitralE.value * 100) / ePrimeAvgCmps
      : null;

  const trGradMmHg = agg.trVmax ? 4 * agg.trVmax.value * agg.trVmax.value : null;
  const psvdMmHg =
    trGradMmHg !== null
      ? trGradMmHg + (summary.ivc.rapMmHg ?? 5)
      : null;

  const inCine = (frameId: number | null, timeS: number | null) =>
    frameId !== null && (displayedFrameId === frameId || isTimeInCineWindow(timeS, cineWindow));

  const showAfHint = showIdentity && truth?.rhythm === 'atrial-fibrillation';

  const renderTile = (
    id: AutoMeasureActionId,
    opts: {
      value?: string | null;
      idleSub: string;
      wide?: boolean;
    },
  ) => {
    const spec = AUTO_MEASURE_SPECS[id];
    const running = runningActionId === id;
    const done = Boolean(opts.value);
    return (
      <button
        key={id}
        type="button"
        className={`auto-tile${opts.wide ? ' wide' : ''}${done ? ' done' : ''}${running ? ' running' : ''}`}
        disabled={closed || (runningActionId !== null && !running)}
        data-auto-action={id}
        data-tip={spec.tip}
        onClick={() => onRun(id)}
      >
        <span className="auto-tile-top">
          <span className="auto-tile-name">{spec.label}</span>
          <span className={`mod-chip${running ? ' match' : ''}`}>{spec.modalityBadge}</span>
        </span>
        <span className="auto-tile-sub">
          {running ? 'Midiendo…' : done ? `✓ ${opts.value}` : opts.idleSub}
        </span>
      </button>
    );
  };

  const plaxEdReady = inCine(summary.plax.edFrameId, summary.plax.edTimeS);
  const plaxMidReady = inCine(summary.plax.midSysFrameId, summary.plax.midSysTimeS);
  const plaxEsReady = inCine(summary.plax.esFrameId, summary.plax.esTimeS);
  const lvEdReady = inCine(summary.lv.edFrameId, summary.lv.edTimeS);
  const lvEsReady = inCine(summary.lv.esFrameId, summary.lv.esTimeS);
  const laEsReady = inCine(summary.la.esFrameId, summary.la.esTimeS);
  const rvEdReady = inCine(summary.rv.edFrameId, summary.rv.edTimeS);
  const rvEsReady = inCine(summary.rv.esFrameId, summary.rv.esTimeS);
  const ivcExpReady = inCine(summary.ivc.expFrameId, summary.ivc.expTimeS);
  const ivcInspReady = inCine(summary.ivc.inspFrameId, summary.ivc.inspTimeS);

  return (
    <div className="auto-hub" data-auto-hub="one-click">
      {showAfHint && (
        <p className="small" role="note">
          FA activa: repetir un botón Doppler en 3–5 latidos calcula automáticamente el promedio ASE.
        </p>
      )}

      {/* 1. PLAX 2D */}
      <div className="auto-group">
        <div className="auto-group-head">
          <span className="auto-group-title">PLAX · Dimensiones</span>
          {summary.plax.hasResult && summary.plax.lvEddCm !== null && (
            <span className="auto-group-kpi">
              DTD {summary.plax.lvEddCm.toFixed(1)}
              {summary.plax.fsPct !== null ? ` · FA ${summary.plax.fsPct.toFixed(0)}%` : ''}
            </span>
          )}
        </div>
        <div className="auto-tile-grid">
          {renderTile('plax-all', {
            wide: true,
            idleSub: 'PLAX · SIV, DTDVI, PP, DTSVI, TSVI, Ao, AI',
            value: summary.plax.hasResult
              ? [
                  summary.plax.lvEddCm !== null ? `DTD ${summary.plax.lvEddCm.toFixed(1)}` : null,
                  summary.plax.lvEsdCm !== null ? `DTS ${summary.plax.lvEsdCm.toFixed(1)}` : null,
                  summary.plax.lvotCm !== null ? `TSVI ${summary.plax.lvotCm.toFixed(2)} cm` : null,
                ]
                  .filter(Boolean)
                  .join(' · ')
              : null,
          })}
        </div>
        {summary.plax.hasResult && (
          <div className="auto-group-foot">
            <div className="auto-ef-metrics">
              {summary.plax.ivsdCm !== null && <span>SIV {summary.plax.ivsdCm.toFixed(2)}</span>}
              {summary.plax.lvpwdCm !== null && <span>PP {summary.plax.lvpwdCm.toFixed(2)}</span>}
              {summary.plax.aoRootCm !== null && <span>Ao {summary.plax.aoRootCm.toFixed(1)}</span>}
              {summary.plax.laApCm !== null && <span>AI {summary.plax.laApCm.toFixed(1)}</span>}
              {lvmi !== null ? (
                <span>IMVI {lvmi.toFixed(0)} g/m²</span>
              ) : summary.plax.massG !== null ? (
                <span>Masa {summary.plax.massG.toFixed(0)} g</span>
              ) : null}
              {summary.plax.rwt !== null && <span>GPR {summary.plax.rwt.toFixed(2)}</span>}
            </div>
            {(plaxEdReady || plaxMidReady || plaxEsReady) && (
              <div className="auto-ef-jump">
                {summary.plax.edFrameId !== null && (
                  <button
                    type="button"
                    className={`ghost${displayedFrameId === summary.plax.edFrameId ? ' active' : ''}`}
                    disabled={closed || !plaxEdReady}
                    onClick={() => onInspect(summary.plax.edFrameId!, 'ed')}
                  >
                    Ver TD
                  </button>
                )}
                {summary.plax.midSysFrameId !== null && (
                  <button
                    type="button"
                    className={`ghost${displayedFrameId === summary.plax.midSysFrameId ? ' active' : ''}`}
                    disabled={closed || !plaxMidReady}
                    onClick={() => onInspect(summary.plax.midSysFrameId!, 'ed')}
                  >
                    Ver TSVI
                  </button>
                )}
                {summary.plax.esFrameId !== null && (
                  <button
                    type="button"
                    className={`ghost${displayedFrameId === summary.plax.esFrameId ? ' active' : ''}`}
                    disabled={closed || !plaxEsReady}
                    onClick={() => onInspect(summary.plax.esFrameId!, 'es')}
                  >
                    Ver TS
                  </button>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* 2. VI y AI Volúmenes (Apical) */}
      <div className="auto-group">
        <div className="auto-group-head">
          <span className="auto-group-title">VI y AI · Volúmenes</span>
          {summary.lv.hasResult && summary.lv.efPct !== null ? (
            <span className="auto-group-kpi">
              FEVI {summary.lv.biplane ? 'bipl. ' : ''}
              {summary.lv.efPct.toFixed(0)}%
            </span>
          ) : lavi !== null ? (
            <span className="auto-group-kpi">LAVI {lavi.toFixed(0)} mL/m²</span>
          ) : null}
        </div>
        <div className="auto-tile-grid">
          {renderTile('lv-a4c', {
            idleSub: 'Apical 4C',
            value: summary.lv.a4c
              ? `${summary.lv.a4c.efPct.toFixed(0)}% · ${summary.lv.a4c.edvMl.toFixed(0)}/${summary.lv.a4c.esvMl.toFixed(0)}`
              : null,
          })}
          {renderTile('lv-a2c', {
            idleSub: 'Apical 2C',
            value: summary.lv.a2c
              ? `${summary.lv.a2c.efPct.toFixed(0)}% · ${summary.lv.a2c.edvMl.toFixed(0)}/${summary.lv.a2c.esvMl.toFixed(0)}`
              : null,
          })}
          {renderTile('la-a4c', {
            idleSub: 'Apical 4C',
            value: summary.la.a4c ? `${summary.la.a4c.volumeMl.toFixed(0)} mL` : null,
          })}
          {renderTile('la-a2c', {
            idleSub: 'Apical 2C',
            value: summary.la.a2c ? `${summary.la.a2c.volumeMl.toFixed(0)} mL` : null,
          })}
        </div>
        {(summary.lv.hasResult || summary.la.hasResult) && (
          <div className="auto-group-foot">
            <div className="auto-ef-metrics">
              {summary.lv.edvMl !== null && summary.lv.esvMl !== null && (
                <span>
                  VI {summary.lv.edvMl.toFixed(0)}/{summary.lv.esvMl.toFixed(0)} mL (VS{' '}
                  {(summary.lv.svMl ?? 0).toFixed(0)})
                </span>
              )}
              {summary.la.volumeMl !== null && (
                <span>
                  {lavi !== null
                    ? `LAVI ${lavi.toFixed(0)} mL/m²`
                    : `AI ${summary.la.volumeMl.toFixed(0)} mL`}
                  {summary.la.biplane ? ' bipl.' : ''}
                </span>
              )}
            </div>
            {(summary.lv.lengthMismatch || summary.la.lengthMismatch) && (
              <div className="biplane-warn" role="status">
                ⚠ Longitud dispar entre A4C y A2C (revisa acortamiento apical).
              </div>
            )}
            {(lvEdReady || lvEsReady || laEsReady) && (
              <div className="auto-ef-jump">
                {lvEdReady && summary.lv.edFrameId !== null && (
                  <button
                    type="button"
                    className={`ghost${displayedFrameId === summary.lv.edFrameId ? ' active' : ''}`}
                    onClick={() => onInspect(summary.lv.edFrameId!, 'ed')}
                  >
                    VI TD
                  </button>
                )}
                {lvEsReady && summary.lv.esFrameId !== null && (
                  <button
                    type="button"
                    className={`ghost${displayedFrameId === summary.lv.esFrameId ? ' active' : ''}`}
                    onClick={() => onInspect(summary.lv.esFrameId!, 'es')}
                  >
                    VI TS
                  </button>
                )}
                {laEsReady && summary.la.esFrameId !== null && (
                  <button
                    type="button"
                    className={`ghost${displayedFrameId === summary.la.esFrameId ? ' active' : ''}`}
                    onClick={() => onInspect(summary.la.esFrameId!, 'es')}
                  >
                    AI TS
                  </button>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* 3. TSVI y Válvula Aórtica (Continuidad A5C) */}
      <div className="auto-group">
        <div className="auto-group-head">
          <span className="auto-group-title">TSVI y Aorta · A5C</span>
          {continuityAvaCm2 !== null ? (
            <span className="auto-group-kpi">AVA {continuityAvaCm2.toFixed(2)} cm²</span>
          ) : dopplerSvMl !== null ? (
            <span className="auto-group-kpi">VS {dopplerSvMl.toFixed(0)} mL</span>
          ) : null}
        </div>
        <div className="auto-tile-grid">
          {renderTile('lvot-vti', {
            idleSub: 'A5C · PW TSVI',
            value: agg.lvotVti
              ? `${agg.lvotVti.value.toFixed(1)} cm${agg.lvotVti.isAveraged ? ` (${agg.lvotVti.beatCount}l)` : ''}`
              : null,
          })}
          {renderTile('av-vti', {
            idleSub: 'A5C · CW VAo',
            value: agg.avVti
              ? `${agg.avVti.latest.derived?.vmaxMps ? `${agg.avVti.latest.derived.vmaxMps.toFixed(2)} m/s` : `${agg.avVti.value.toFixed(1)} cm`}`
              : null,
          })}
        </div>
        {(agg.lvotVti || agg.avVti) && (
          <div className="auto-group-foot">
            <div className="auto-ef-metrics">
              {dopplerSvMl !== null && <span>VS {dopplerSvMl.toFixed(0)} mL</span>}
              {agg.avVti && <span>VTI VAo {agg.avVti.value.toFixed(1)} cm</span>}
              {agg.avVti?.latest.derived?.meanGradientMmHg !== undefined && (
                <span>ΔPm {agg.avVti.latest.derived.meanGradientMmHg.toFixed(0)} mmHg</span>
              )}
              {velocityRatio !== null && <span>DVI {velocityRatio.toFixed(2)}</span>}
            </div>
          </div>
        )}
      </div>

      {/* 4. Diástole VI (A4C) */}
      <div className="auto-group">
        <div className="auto-group-head">
          <span className="auto-group-title">Diástole VI · A4C</span>
          {eOverEPrime !== null ? (
            <span className="auto-group-kpi">E/e′ {eOverEPrime.toFixed(1)}</span>
          ) : eaRatio !== null ? (
            <span className="auto-group-kpi">E/A {eaRatio.toFixed(2)}</span>
          ) : null}
        </div>
        <div className="auto-tile-grid">
          {renderTile('mitral-ea', {
            wide: true,
            idleSub: 'A4C · PW puntas mitrales',
            value: agg.mitralE
              ? [
                  `E ${agg.mitralE.value.toFixed(2)}`,
                  agg.mitralA ? `A ${agg.mitralA.value.toFixed(2)}` : null,
                  agg.mitralDt ? `TD ${Math.round(agg.mitralDt.value)} ms` : null,
                ]
                  .filter(Boolean)
                  .join(' · ')
              : null,
          })}
          {renderTile('tdi-septal', {
            idleSub: 'A4C · TDI sept.',
            value: agg.epSept ? `${agg.epSept.value.toFixed(1)} cm/s` : null,
          })}
          {renderTile('tdi-lateral', {
            idleSub: 'A4C · TDI lat.',
            value: agg.epLat ? `${agg.epLat.value.toFixed(1)} cm/s` : null,
          })}
        </div>
        {(eaRatio !== null || eOverEPrime !== null) && (
          <div className="auto-group-foot">
            <div className="auto-ef-metrics">
              {eaRatio !== null && <span>E/A {eaRatio.toFixed(2)}</span>}
              {eOverEPrime !== null && <span>E/e′ prom {eOverEPrime.toFixed(1)}</span>}
            </div>
          </div>
        )}
      </div>

      {/* 5. VD, Pulmonar y VCI */}
      <div className="auto-group">
        <div className="auto-group-head">
          <span className="auto-group-title">VD, Pulmonar y VCI</span>
          {psvdMmHg !== null ? (
            <span className="auto-group-kpi">PSVD {psvdMmHg.toFixed(0)} mmHg</span>
          ) : summary.ivc.rapMmHg !== null ? (
            <span className="auto-group-kpi">PAD {summary.ivc.rapMmHg} mmHg</span>
          ) : summary.rv.efPct !== null ? (
            <span className="auto-group-kpi">FEVD {summary.rv.efPct.toFixed(0)}%</span>
          ) : null}
        </div>
        <div className="auto-tile-grid">
          {renderTile('rv-2d', {
            idleSub: 'Enf. VD · 2D',
            value: summary.rv.hasResult
              ? [
                  summary.rv.efPct !== null ? `${summary.rv.efPct.toFixed(0)}%` : null,
                  summary.rv.facPct !== null ? `CAF ${summary.rv.facPct.toFixed(0)}%` : null,
                ]
                  .filter(Boolean)
                  .join(' · ') || (summary.rv.basalCm !== null ? `${summary.rv.basalCm.toFixed(1)} cm` : null)
              : null,
          })}
          {renderTile('tapse', {
            idleSub: 'Enf. VD · M',
            value: agg.tapse ? `${agg.tapse.value.toFixed(2)} cm` : null,
          })}
          {renderTile('rv-s-prime', {
            idleSub: 'Enf. VD · TDI',
            value: agg.rvSPrime ? `${agg.rvSPrime.value.toFixed(1)} cm/s` : null,
          })}
          {renderTile('tr-vmax', {
            idleSub: 'Enf. VD · CW',
            value: agg.trVmax ? `${agg.trVmax.value.toFixed(2)} m/s` : null,
          })}
          {renderTile('rvot-at', {
            idleSub: 'PSAX Ao · PW',
            value: agg.rvotAt ? `${Math.round(agg.rvotAt.value)} ms` : null,
          })}
          {renderTile('ivc-full', {
            idleSub: 'Subc. VCI · Esp+Insp',
            value: summary.ivc.hasResult
              ? summary.ivc.rapMmHg !== null && summary.ivc.collapsePct !== null
                ? `PAD ${summary.ivc.rapMmHg} · ${summary.ivc.collapsePct.toFixed(0)}%`
                : `${(summary.ivc.expCm ?? summary.ivc.inspCm ?? 0).toFixed(2)} cm`
              : null,
          })}
        </div>
        {(summary.rv.hasResult || summary.ivc.hasResult || psvdMmHg !== null) && (
          <div className="auto-group-foot">
            <div className="auto-ef-metrics">
              {summary.rv.basalCm !== null && <span>Basal {summary.rv.basalCm.toFixed(1)} cm</span>}
              {summary.ivc.expCm !== null && (
                <span>
                  VCI {summary.ivc.expCm.toFixed(2)}
                  {summary.ivc.inspCm !== null ? `→${summary.ivc.inspCm.toFixed(2)}` : ''} cm
                </span>
              )}
              {summary.ivc.rapMmHg !== null && <span>PAD {summary.ivc.rapMmHg} mmHg</span>}
              {psvdMmHg !== null && <span>PSVD {psvdMmHg.toFixed(0)} mmHg</span>}
            </div>
            {(rvEdReady || rvEsReady || ivcExpReady || ivcInspReady) && (
              <div className="auto-ef-jump">
                {rvEdReady && summary.rv.edFrameId !== null && (
                  <button
                    type="button"
                    className={`ghost${displayedFrameId === summary.rv.edFrameId ? ' active' : ''}`}
                    onClick={() => onInspect(summary.rv.edFrameId!, 'ed')}
                  >
                    VD TD
                  </button>
                )}
                {rvEsReady && summary.rv.esFrameId !== null && (
                  <button
                    type="button"
                    className={`ghost${displayedFrameId === summary.rv.esFrameId ? ' active' : ''}`}
                    onClick={() => onInspect(summary.rv.esFrameId!, 'es')}
                  >
                    VD TS
                  </button>
                )}
                {ivcExpReady && summary.ivc.expFrameId !== null && (
                  <button
                    type="button"
                    className={`ghost${displayedFrameId === summary.ivc.expFrameId ? ' active' : ''}`}
                    onClick={() => onInspect(summary.ivc.expFrameId!, 'ed')}
                  >
                    VCI Esp
                  </button>
                )}
                {ivcInspReady && summary.ivc.inspFrameId !== null && (
                  <button
                    type="button"
                    className={`ghost${displayedFrameId === summary.ivc.inspFrameId ? ' active' : ''}`}
                    onClick={() => onInspect(summary.ivc.inspFrameId!, 'ed')}
                  >
                    VCI Insp
                  </button>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Exam-mode pure-signal auto-measure row: NEVER moves probe, NEVER switches views/modalities,
 * and NEVER leaks view labels — strictly offers actions valid for the modality currently on screen.
 */
function ExamAutoRow({
  modality,
  closed,
  runningActionId,
  onRun,
}: {
  modality: SimStore['modality'];
  closed: boolean;
  runningActionId: AutoMeasureActionId | null;
  onRun: (id: AutoMeasureActionId) => void;
}) {
  const items: { id: AutoMeasureActionId; label: string }[] =
    modality === '2d' || modality === 'color'
      ? [
          { id: 'plax-all', label: 'Auto PLAX' },
          { id: 'lv-a4c', label: 'Auto FE VI' },
          { id: 'la-a4c', label: 'Auto Vol AI' },
          { id: 'rv-2d', label: 'Auto FE VD' },
          { id: 'ivc-full', label: 'Auto VCI' },
        ]
      : modality === 'pw'
        ? [
            { id: 'lvot-vti', label: 'Auto VTI TSVI' },
            { id: 'mitral-ea', label: 'Auto E / A / TD' },
            { id: 'rvot-at', label: 'Auto TAc pulm.' },
          ]
        : modality === 'cw'
          ? [
              { id: 'av-vti', label: 'Auto VTI VAo' },
              { id: 'tr-vmax', label: 'Auto Vmáx IT' },
            ]
          : modality === 'tdi'
            ? [
                { id: 'tdi-septal', label: 'Auto e′ sept' },
                { id: 'tdi-lateral', label: 'Auto e′ lat' },
                { id: 'rv-s-prime', label: 'Auto S′ VD' },
              ]
            : [{ id: 'tapse', label: 'Auto TAPSE' }];

  return (
    <div className="exam-auto-row">
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          className="ghost"
          disabled={closed || runningActionId !== null}
          onClick={() => onRun(item.id)}
        >
          {runningActionId === item.id ? 'Midiendo…' : item.label}
        </button>
      ))}
    </div>
  );
}

/** Manual free measurement tools grouped by 2D vs Doppler/M-mode. */
function FreeToolsSection({
  modality,
  closed,
  tip,
  onArmTool,
}: {
  modality: SimStore['modality'];
  closed: boolean;
  tip: (t: string) => string | undefined;
  onArmTool: (t: MeasurementTool) => void;
}) {
  return (
    <Section
      title="Herramientas libres"
      tip={tip('Medidas directas sin evaluación de técnica. Se activan según la modalidad en pantalla.')}
    >
      <div className="free-tool-group">
        <span className="free-tool-cap">Imagen 2D</span>
        <div className="tool-grid two" role="group" aria-label="Herramientas 2D">
          {TOOLS_2D.map((t) => {
            const compat = isFreeToolCompatible(t.id, modality);
            return (
              <button
                key={t.id}
                type="button"
                className={compat ? '' : 'tool-dim'}
                data-tip={compat ? t.title : `${t.title} · óptima en 2D`}
                onClick={() => onArmTool(t.id)}
                disabled={closed}
              >
                {t.label}
              </button>
            );
          })}
        </div>
      </div>
      <div className="free-tool-group">
        <span className="free-tool-cap">Doppler y modo M</span>
        <div className="tool-grid" role="group" aria-label="Herramientas Doppler y modo M">
          {TOOLS_STRIP.map((t) => {
            const compat = isFreeToolCompatible(t.id, modality);
            return (
              <button
                key={t.id}
                type="button"
                className={compat ? '' : 'tool-dim'}
                data-tip={
                  compat
                    ? t.title
                    : `${t.title} · requiere ${t.id === 'tapse' ? 'Modo M' : 'Doppler espectral (PW/CW/TDI)'}`
                }
                onClick={() => onArmTool(t.id)}
                disabled={closed}
              >
                {t.label}
              </button>
            );
          })}
        </div>
      </div>
    </Section>
  );
}

/** Dedicated «Capturas» worksheet: live derived KPIs + clickable measurement rows + copy/report actions. */
function CapturasWorksheet({
  s,
  showView,
  showIdentity,
  displayedFrameId,
  cineWindow,
  onInspect,
  onBackToMeasure,
  onOpenReport,
}: {
  s: Pick<SimStore, 'captures' | 'measurements' | 'removeCapture' | 'removeMeasurement' | 'truth'> & {
    closed: boolean;
  };
  showView: boolean;
  showIdentity: boolean;
  displayedFrameId: number | null;
  cineWindow: { startS: number; endS: number } | null;
  onInspect: (frameId: number, target?: 'ed' | 'es') => void;
  onBackToMeasure: () => void;
  onOpenReport: () => void;
}) {
  const derived = useMemo(
    () => deriveCalculations(s.measurements, s.truth?.bsaM2 ?? null),
    [s.measurements, s.truth?.bsaM2],
  );
  const showAfTip = showIdentity && s.truth?.rhythm === 'atrial-fibrillation';

  if (!s.measurements.length && !s.captures.length) {
    return (
      <div className="empty measure-empty">
        <p>Todavía no has tomado mediciones ni capturas en este estudio.</p>
        {showAfTip && (
          <p className="small">
            Fibrilación auricular: mide 3–5 latidos distintos; EKO-neon promedia automáticamente por
            latido (ASE).
          </p>
        )}
        <div className="quick">
          <button type="button" className="btn-primary-soft" onClick={onBackToMeasure}>
            Ir a medir
          </button>
        </div>
      </div>
    );
  }

  return (
    <>
      {showAfTip && (
        <p className="small" role="note">
          FA activa: al medir 2–5 latidos distintos del mismo parámetro, el sistema calcula
          automáticamente el promedio ASE.
        </p>
      )}
      {s.captures.length > 0 && (
        <Section title={`Imágenes guardadas (${s.captures.length})`}>
          <div className="capture-gallery" data-capture-gallery="worksheet">
            {s.captures.map((cap, idx) => (
              <figure key={cap.id} className="capture-thumb">
                <img
                  src={cap.dataUrl}
                  alt={`Captura ${idx + 1}: ${formatCaptureCaption(cap, showIdentity)}`}
                />
                <figcaption>
                  <span>{formatCaptureCaption(cap, showIdentity)}</span>
                  {!s.closed && (
                    <button
                      type="button"
                      className="capture-del"
                      aria-label={`Eliminar captura ${idx + 1}`}
                      onClick={() => s.removeCapture(cap.id)}
                    >
                      ×
                    </button>
                  )}
                </figcaption>
              </figure>
            ))}
          </div>
          {!s.measurements.length && (
            <div className="measure-head" style={{ marginTop: 8 }}>
              <button type="button" className="btn-primary-soft" onClick={onOpenReport}>
                Ver en Informe →
              </button>
            </div>
          )}
        </Section>
      )}
      {derived.length > 0 && (
        <Section title="Cálculos derivados">
          <div className="derived-kpi-grid">
            {derived.map((d) => (
              <div key={d.id} className="derived-kpi" data-tip={`${d.formula} (${d.inputs})`}>
                <span className="derived-kpi-label">{d.label}</span>
                <span className="derived-kpi-val">{d.value}</span>
              </div>
            ))}
          </div>
        </Section>
      )}
      {s.measurements.length > 0 && (
        <Section title={`Mediciones tomadas (${s.measurements.length})`}>
          <MeasureList
            s={s}
            showView={showView}
            displayedFrameId={displayedFrameId}
            cineWindow={cineWindow}
            onInspect={onInspect}
            derivedText={derived.map((d) => `${d.label}: ${d.value}`).join('\n')}
            onOpenReport={onOpenReport}
          />
        </Section>
      )}
    </>
  );
}

/** `showView` false (exam) leaves out the view each measurement was taken in: it names the view (decision 154). */
function MeasureList({
  s,
  showView,
  displayedFrameId,
  cineWindow,
  onInspect,
  derivedText,
  onOpenReport,
}: {
  s: Pick<SimStore, 'measurements' | 'removeMeasurement'> & { closed: boolean };
  showView: boolean;
  displayedFrameId: number | null;
  cineWindow: { startS: number; endS: number } | null;
  onInspect: (frameId: number, target?: 'ed' | 'es') => void;
  derivedText?: string;
  onOpenReport?: () => void;
}) {
  const line = (m: SimStore['measurements'][number]) => {
    const sub = secondaryMeta(m);
    return sub ? `${primaryValue(m)} · ${sub}` : primaryValue(m);
  };
  const [status, setStatus] = useState<string | null>(null);
  const statusTimer = useRef<number | null>(null);
  useEffect(() => () => window.clearTimeout(statusTimer.current ?? undefined), []);

  const copy = async () => {
    const body = s.measurements.map((m) => `${m.label}: ${line(m)}`).join('\n');
    const text = derivedText ? `${body}\n\nCálculos derivados:\n${derivedText}` : body;
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
      <div className="measure-head">
        {status && (
          <span className="small" role="status">
            {status}
          </span>
        )}
        <button type="button" className="ghost" onClick={() => void copy()}>
          Copiar resumen
        </button>
        {onOpenReport && (
          <button type="button" className="btn-primary-soft" onClick={onOpenReport}>
            Informe →
          </button>
        )}
      </div>
      {s.measurements.map((m) => {
        const sub = secondaryMeta(m);
        const badge = techniqueBadge(m, showView);
        const onScreen = displayedFrameId !== null && displayedFrameId === m.frameId;
        const canInspect =
          Boolean(m.captureSector) &&
          m.frameId > 0 &&
          !s.closed &&
          (onScreen || isTimeInCineWindow(m.timeS, cineWindow));
        return (
          <div key={m.id} className={`measure-row${onScreen ? ' on-screen' : ''}`}>
            <span className="m-label">
              <span className="m-title-line">
                {m.label}
                {badge && (
                  <span
                    className={`m-tech-dot ${badge.level}`}
                    data-tip={badge.tip}
                    aria-label={badge.label}
                  />
                )}
              </span>
              <span className="m-meta-line">
                <span className="m-src">
                  {modalityLabel(m.modality)}
                  {showView && m.sourceViewId ? ` · ${viewLabel(m.sourceViewId)}` : ''}
                </span>
                {sub && <span className="m-sub">{sub}</span>}
              </span>
            </span>
            <span className="m-right">
              <span className="m-value">{primaryValue(m)}</span>
              {canInspect && (
                <button
                  type="button"
                  className={`m-inspect${onScreen ? ' active' : ''}`}
                  onClick={() => onInspect(m.frameId)}
                  data-tip="Ver el fotograma y contorno de esta medición"
                >
                  {onScreen ? 'En vista' : 'Ver'}
                </button>
              )}
            </span>
            <button
              type="button"
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
