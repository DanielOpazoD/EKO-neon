import { examClosed, modePolicy, showsCaseIdentity } from '@/app/modePolicy';
import { useHudStore, useSimStore, type SimStore } from '@/app/store';
import { loadCaseById } from '@/cases';
import { strokeVolume } from '@/clinical/formulas';
import { deriveCalculations } from '@/education/report';
import { Structure } from '@/simulator/anatomy/tissue';
import {
  MEASUREMENT_SPECS,
  type MeasurementSpec,
  type MeasurementTool,
} from '@/simulator/measurements/protocol';
import { modalityLabel } from '@/simulator/renderer/modality';
import { viewLabel } from '@/simulator/windows/viewLabels';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useShallow } from 'zustand/shallow';
import { Section } from '../controls';
import { captureStepPrompt, minPointsToFinish } from '../display/measurementPreview';
import {
  runAutoChamberVolume,
  runAutoLvotGate,
  runAutoSpectralVti,
} from '../display/measurementTools';
import { pendingSession, usePendingPointCount } from '../display/pendingSession';
import { MeasurementPanel } from '../MeasurementPanel';
import { ViewQuickMenu } from '../ViewQuickMenu';
import {
  captureModalityCheck,
  isFreeToolCompatible,
  isTimeInCineWindow,
  phaseTag,
  specsForView,
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
 * Context-first clinical measurement workspace:
 * - «Medir»: current context strip, view-matched quick measurements, guided Auto-FE VI/VD,
 *   full protocol catalogue, and modality-grouped free tools.
 * - «Capturas (N)»: live derived calculations (FEVI, FEVD, CAF, AVA, E/e′, PSAP…) +
 *   interactive measurement worksheet with frame jump, copy, and direct link to Informe.
 * - «Midiendo»: step-by-step guided capture card with 1-click modality/freeze/phase helpers.
 */
export function MeasureTab() {
  const s = useSimStore(
    useShallow((st) => ({
      activeMeasurementId: st.activeMeasurementId,
      activeTool: st.activeTool,
      caseId: st.caseId,
      examFinished: st.examFinished,
      frozen: st.frozen,
      measurements: st.measurements,
      modality: st.modality,
      mode: st.mode,
      truth: st.truth,
      jumpCinePhase: st.jumpCinePhase,
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
  const tip = useTip();
  const policy = modePolicy(s.mode);
  const showView = policy.showViewFeedback;
  const showIdentity = showsCaseIdentity(s.mode, s.examFinished);

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

  const inspectFrame = (frameId: number, target: 'ed' | 'es' = 'ed') => {
    if (s.closed) return;
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
      (spec?.phase === 'ed' || spec?.phase === 'es' || s.activeTool === 'caliper' || s.activeTool === 'simpson');
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
              {spec
                ? spec.instruction
                : FREE_TOOL_HINT[s.activeTool as MeasurementTool]}
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
          {s.measurements.length > 0 && (
            <span className="measure-subtab-count">{s.measurements.length}</span>
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
              onClick={() => s.toggleFreeze()}
              data-tip={s.frozen ? 'Reanudar imagen en vivo (Espacio)' : 'Congelar imagen para medir (Espacio)'}
            >
              {s.frozen ? 'Congelada · Reanudar' : 'En vivo · Congelar'}
            </button>
          </div>

          {showView && currentViewId && (
            <ViewQuickSection
              viewId={currentViewId}
              modality={s.modality}
              caseId={s.caseId}
              showIdentity={showIdentity}
              showFeedback={showView}
              measurements={s.measurements}
              closed={s.closed}
              onArm={(id) => s.setActiveMeasurement(id)}
            />
          )}

          <Section
            title="Volumen y FE automáticos"
            tip={tip(
              'Traza automáticamente el endocardio en TD y TS sobre el cine 2D actual (A4C y A2C para Simpson biplano del VI · enfocada en VD o A4C para VD).',
            )}
          >
            <AutoEfPanel
              currentViewId={currentViewId}
              displayedFrameId={displayedFrameId}
              cineWindow={cineWindow}
              showViewFeedback={showView}
              presetsEnabled={policy.presetsEnabled}
            />
          </Section>

          <Section
            title="Auto-VTI del TSVI (A5C)"
            tip={tip(
              'Coloca el volumen de muestra PW a 0,5 cm proximal a la válvula aórtica en A5C/A3C y traza automáticamente el latido sistólico más reciente.',
            )}
          >
            <AutoLvotVtiPanel
              currentViewId={currentViewId}
              showViewFeedback={showView}
              presetsEnabled={policy.presetsEnabled}
            />
          </Section>

          <Section title="Protocolo completo">
            <MeasurementPanel />
          </Section>

          <Section
            title="Herramientas libres"
            tip={tip('Medidas directas sin evaluación de técnica. Se activan según la modalidad en pantalla.')}
          >
            <div className="free-tool-group">
              <span className="free-tool-cap">Imagen 2D</span>
              <div className="tool-grid two" role="group" aria-label="Herramientas 2D">
                {TOOLS_2D.map((t) => {
                  const compat = isFreeToolCompatible(t.id, s.modality);
                  return (
                    <button
                      key={t.id}
                      type="button"
                      className={compat ? '' : 'tool-dim'}
                      data-tip={compat ? t.title : `${t.title} · óptima en 2D`}
                      onClick={() => s.setActiveTool(t.id)}
                      disabled={s.closed}
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
                  const compat = isFreeToolCompatible(t.id, s.modality);
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
                      onClick={() => s.setActiveTool(t.id)}
                      disabled={s.closed}
                    >
                      {t.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </Section>
        </>
      ) : (
        <CapturasWorksheet
          s={s}
          showView={showView}
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

/** Context-matched 1-click measurements for the currently recognized view (hidden during exam). */
function ViewQuickSection({
  viewId,
  modality,
  caseId,
  showIdentity,
  showFeedback,
  measurements,
  closed,
  onArm,
}: {
  viewId: string;
  modality: SimStore['modality'];
  caseId: string;
  showIdentity: boolean;
  showFeedback: boolean;
  measurements: SimStore['measurements'];
  closed: boolean;
  onArm: (id: string) => void;
}) {
  const requiredIds = useMemo(() => {
    if (!showIdentity) return [];
    return loadCaseById(caseId).requiredMeasurements.map((r) => r.measurementId);
  }, [caseId, showIdentity]);
  const specs = useMemo(
    () => specsForView(viewId, modality, requiredIds),
    [viewId, modality, requiredIds],
  );
  const [showOtherMods, setShowOtherMods] = useState(false);
  useEffect(() => {
    setShowOtherMods(false);
  }, [viewId, modality]);
  if (!specs.length) return null;

  const sameMod = specs.filter((spec) => spec.modalities.includes(modality));
  const otherMod = specs.filter((spec) => !spec.modalities.includes(modality));
  // Show active-modality measurements immediately; keep other modalities collapsed when there are many so
  // Auto-FE stays above the fold on views like A4C without scrolling.
  const visibleSpecs =
    sameMod.length > 0 && otherMod.length > 0 && !showOtherMods ? sameMod : specs;

  const reqSet = new Set(requiredIds);
  const latestOf = (id: string) => {
    for (let i = measurements.length - 1; i >= 0; i--) {
      if (measurements[i]!.measurementId === id) return measurements[i]!;
    }
    return null;
  };

  return (
    <Section title={`En esta vista · ${viewLabel(viewId)}`}>
      <div className="view-quick-grid">
        {visibleSpecs.map((spec: MeasurementSpec) => {
          const m = latestOf(spec.id);
          const modMatch = spec.modalities.includes(modality);
          const isReq = reqSet.has(spec.id);
          const phase = phaseTag(spec.phase);
          const badge = m ? techniqueBadge(m, showFeedback) : null;
          return (
            <button
              key={spec.id}
              type="button"
              className={`quick-meas-btn${modMatch ? ' ready' : ''}${m ? ' done' : ''}${isReq && !m ? ' req' : ''}`}
              disabled={closed}
              data-tip={spec.instruction}
              onClick={() => onArm(spec.id)}
            >
              <span className="qm-top">
                <span className="qm-name">
                  {isReq && <i className="qm-req-dot" aria-label="Requerida" />}
                  {spec.shortLabel}
                </span>
                <span className={`mod-chip${modMatch ? ' match' : ''}`}>
                  {modalityLabel(spec.modalities[0] ?? '2d')}
                </span>
              </span>
              <span className="qm-bottom">
                {m ? (
                  <>
                    <span className="qm-val">{primaryValue(m)}</span>
                    {badge && <i className={`qm-status ${badge.level}`} aria-hidden="true" />}
                  </>
                ) : (
                  <span className="qm-phase">{phase ?? 'Medir'}</span>
                )}
              </span>
            </button>
          );
        })}
      </div>
      {sameMod.length > 0 && otherMod.length > 0 && (
        <button
          type="button"
          className="view-quick-more ghost"
          onClick={() => setShowOtherMods((v) => !v)}
        >
          {showOtherMods
            ? `Mostrar solo ${modalityLabel(modality)}`
            : `+ ${otherMod.length} en otras modalidades (Doppler / M)`}
        </button>
      )}
    </Section>
  );
}

/** Dedicated «Capturas» worksheet: live derived KPIs + clickable measurement rows + copy/report actions. */
function CapturasWorksheet({
  s,
  showView,
  displayedFrameId,
  cineWindow,
  onInspect,
  onBackToMeasure,
  onOpenReport,
}: {
  s: Pick<SimStore, 'measurements' | 'removeMeasurement' | 'truth'> & { closed: boolean };
  showView: boolean;
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

  if (!s.measurements.length) {
    return (
      <div className="empty measure-empty">
        <p>Todavía no has tomado mediciones en este estudio.</p>
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
                  <span className={`m-tech-dot ${badge.level}`} data-tip={badge.tip} aria-label={badge.label} />
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

/** Guided one-click automatic LV & RV endocardial tracing, biplane progression, and TD/TS inspection. */
function AutoEfPanel({
  currentViewId,
  displayedFrameId,
  cineWindow,
  showViewFeedback,
  presetsEnabled,
}: {
  currentViewId: string | null;
  displayedFrameId: number | null;
  cineWindow: { startS: number; endS: number } | null;
  showViewFeedback: boolean;
  presetsEnabled: boolean;
}) {
  const s = useSimStore(
    useShallow((st) => ({
      measurements: st.measurements,
      modality: st.modality,
      frozen: st.frozen,
      closed: examClosed(st),
      setModality: st.setModality,
      startPresetView: st.startPresetView,
      toggleFreeze: st.toggleFreeze,
      jumpCinePhase: st.jumpCinePhase,
    })),
  );
  const [busy, setBusy] = useState<'lv' | 'rv' | null>(null);
  const [notice, setNotice] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);

  const summary = useMemo(
    () => summarizeAutoEf(s.measurements, currentViewId, s.modality, showViewFeedback),
    [s.measurements, currentViewId, s.modality, showViewFeedback],
  );

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

  const jumpPreset = (viewId: string) => {
    if (s.closed || !presetsEnabled) return;
    setNotice(null);
    if (!summary.is2dReady) s.setModality('2d');
    s.startPresetView(viewId);
  };

  const showApicalShortcuts =
    presetsEnabled &&
    showViewFeedback &&
    summary.is2dReady &&
    !summary.lv.viewReady &&
    !summary.rv.viewReady;
  const showBiplanePrompt =
    presetsEnabled &&
    showViewFeedback &&
    summary.lv.a4cDone &&
    !summary.lv.a2cDone &&
    currentViewId !== 'a2c';

  const showRvPrompt =
    presetsEnabled &&
    showViewFeedback &&
    summary.is2dReady &&
    !summary.rv.hasResult &&
    !summary.rv.viewReady &&
    summary.lv.hasResult;
  const lvA4cEdInCine =
    summary.lv.a4c !== null &&
    (displayedFrameId === summary.lv.a4c.edFrameId ||
      isTimeInCineWindow(summary.lv.a4c.edTimeS, cineWindow));
  const lvA4cEsInCine =
    summary.lv.a4c !== null &&
    (displayedFrameId === summary.lv.a4c.esFrameId ||
      isTimeInCineWindow(summary.lv.a4c.esTimeS, cineWindow));
  const lvA2cEdInCine =
    summary.lv.a2c !== null &&
    (displayedFrameId === summary.lv.a2c.edFrameId ||
      isTimeInCineWindow(summary.lv.a2c.edTimeS, cineWindow));
  const lvA2cEsInCine =
    summary.lv.a2c !== null &&
    (displayedFrameId === summary.lv.a2c.esFrameId ||
      isTimeInCineWindow(summary.lv.a2c.esTimeS, cineWindow));
  const lvEdInCine =
    displayedFrameId === summary.lv.edFrameId || isTimeInCineWindow(summary.lv.edTimeS, cineWindow);
  const lvEsInCine =
    displayedFrameId === summary.lv.esFrameId || isTimeInCineWindow(summary.lv.esTimeS, cineWindow);
  const rvEdInCine =
    displayedFrameId === summary.rv.edFrameId || isTimeInCineWindow(summary.rv.edTimeS, cineWindow);
  const rvEsInCine =
    displayedFrameId === summary.rv.esFrameId || isTimeInCineWindow(summary.rv.esTimeS, cineWindow);

  return (
    <div className="auto-ef">
      {/* LV chamber block */}
      <div className="auto-ef-card" data-auto-ef="lv">
        <div className="auto-ef-row-head">
          <div className="auto-ef-chamber">
            <span className="auto-ef-title">
              VI · {summary.lv.biplane ? 'Simpson biplano' : 'Simpson'}
            </span>
            <span className="plane-pills" aria-label="Planos capturados del VI">
              <span
                className={`plane-pill${summary.lv.a4cDone ? ' done' : currentViewId === 'a4c' ? ' here' : ''}`}
              >
                {summary.lv.a4cDone ? '✓ A4C' : 'A4C'}
              </span>
              <span
                className={`plane-pill${summary.lv.a2cDone ? ' done' : currentViewId === 'a2c' ? ' here' : ''}`}
              >
                {summary.lv.a2cDone ? '✓ A2C' : 'A2C'}
              </span>
            </span>
          </div>
          <button
            type="button"
            className={summary.lv.viewReady ? 'btn-primary-soft' : ''}
            onClick={() => void run('lv')}
            disabled={s.closed || busy !== null}
            data-tip="Traza el endocardio del VI en TD y TS sobre A4C o A2C; repite en ambos planos para Simpson biplano"
          >
            {busy === 'lv' ? 'Midiendo…' : 'Auto-FE VI'}
          </button>
        </div>

        {summary.lv.hasResult &&
          summary.lv.efPct !== null &&
          summary.lv.edvMl !== null &&
          summary.lv.esvMl !== null &&
          summary.lv.svMl !== null && (
            <>
              <div className="auto-ef-result-line">
                <span className="auto-ef-primary">FEVI {summary.lv.efPct.toFixed(0)} %</span>
                <span className="auto-ef-metrics">
                  <span>VTD {summary.lv.edvMl.toFixed(0)}</span>
                  <span>VTS {summary.lv.esvMl.toFixed(0)}</span>
                  <span>VS {summary.lv.svMl.toFixed(0)} mL</span>
                </span>
              </div>

              {/* Explicit A4C & A2C component breakdown so biplane provenance is crystal clear */}
              {(summary.lv.a4c || summary.lv.a2c) && (
                <div className="biplane-breakdown" aria-label="Desglose Simpson A4C y A2C">
                  {([
                    {
                      id: 'a4c' as const,
                      label: 'A4C',
                      plane: summary.lv.a4c,
                      edIn: lvA4cEdInCine,
                      esIn: lvA4cEsInCine,
                    },
                    {
                      id: 'a2c' as const,
                      label: 'A2C',
                      plane: summary.lv.a2c,
                      edIn: lvA2cEdInCine,
                      esIn: lvA2cEsInCine,
                    },
                  ]).map(({ id, label, plane, edIn, esIn }) => (
                    <div key={id} className={`biplane-row${plane ? ' done' : ''}`}>
                      <span className="bp-tag">{plane ? `✓ ${label}` : `○ ${label}`}</span>
                      {plane ? (
                        <>
                          <span className="bp-vals">
                            {plane.edvMl.toFixed(0)}/{plane.esvMl.toFixed(0)} mL ·{' '}
                            <b>{plane.efPct.toFixed(0)}%</b>
                            {plane.edLengthCm !== null ? ` · L ${plane.edLengthCm.toFixed(1)}` : ''}
                          </span>
                          <span className="bp-jumps">
                            <button
                              type="button"
                              className={`ghost${displayedFrameId === plane.edFrameId ? ' active' : ''}`}
                              disabled={s.closed || !edIn}
                              data-tip={edIn ? `Ver ${label} en TD` : 'Fuera del cine actual'}
                              onClick={() => inspectFrame('ed', plane.edFrameId)}
                            >
                              TD
                            </button>
                            <button
                              type="button"
                              className={`ghost${displayedFrameId === plane.esFrameId ? ' active' : ''}`}
                              disabled={s.closed || !esIn}
                              data-tip={esIn ? `Ver ${label} en TS` : 'Fuera del cine actual'}
                              onClick={() => inspectFrame('es', plane.esFrameId)}
                            >
                              TS
                            </button>
                          </span>
                        </>
                      ) : (
                        <>
                          <span className="bp-pending">Pendiente para biplano</span>
                          {presetsEnabled && showViewFeedback && !s.closed && (
                            <button
                              type="button"
                              className="ghost bp-go"
                              onClick={() => jumpPreset(id)}
                            >
                              Ir a {label}
                            </button>
                          )}
                        </>
                      )}
                    </div>
                  ))}
                  {summary.lv.lengthMismatch && (
                    <div className="biplane-warn" role="status">
                      ⚠ Longitud dispar entre A4C y A2C: revisa acortamiento apical.
                    </div>
                  )}
                </div>
              )}

              {/* Fallback TD/TS buttons when LV Simpson was traced outside A4C/A2C (e.g. exam mode or A3C) */}
              {!summary.lv.a4c &&
                !summary.lv.a2c &&
                summary.lv.edFrameId !== null &&
                summary.lv.esFrameId !== null && (
                  <div className="auto-ef-jump">
                    <button
                      type="button"
                      className={`ghost${displayedFrameId === summary.lv.edFrameId ? ' active' : ''}`}
                      disabled={s.closed || !lvEdInCine}
                      data-tip={lvEdInCine ? 'Inspeccionar contorno en TD' : 'El cuadro TD salió del bucle de cine actual'}
                      onClick={() => inspectFrame('ed', summary.lv.edFrameId!)}
                    >
                      Ver TD
                    </button>
                    <button
                      type="button"
                      className={`ghost${displayedFrameId === summary.lv.esFrameId ? ' active' : ''}`}
                      disabled={s.closed || !lvEsInCine}
                      data-tip={lvEsInCine ? 'Inspeccionar contorno en TS' : 'El cuadro TS salió del bucle de cine actual'}
                      onClick={() => inspectFrame('es', summary.lv.esFrameId!)}
                    >
                      Ver TS
                    </button>
                    {showBiplanePrompt && (
                      <button
                        type="button"
                        className="ghost auto-ef-next"
                        disabled={s.closed}
                        onClick={() => jumpPreset('a2c')}
                        data-tip="Mueve la sonda a A2C para completar Simpson biplano"
                      >
                        + A2C biplano
                      </button>
                    )}
                  </div>
                )}
            </>
          )}
      </div>

      {/* RV chamber block */}
      <div className="auto-ef-card" data-auto-ef="rv">
        <div className="auto-ef-row-head">
          <div className="auto-ef-chamber">
            <span className="auto-ef-title">VD · 2D</span>
            <span className="plane-pills" aria-label="Vista del VD">
              <span
                className={`plane-pill${summary.rv.hasResult ? ' done' : summary.rv.viewReady ? ' here' : ''}`}
              >
                {summary.rv.hasResult ? '✓ Enf. VD' : 'Enf. VD'}
              </span>
            </span>
          </div>
          <div className="auto-ef-presets">
            {showRvPrompt && (
              <button
                type="button"
                className="ghost"
                disabled={s.closed}
                onClick={() => jumpPreset('rv-focused')}
                data-tip="Mueve la sonda a la vista apical enfocada en VD"
              >
                Ir a VD
              </button>
            )}
            <button
              type="button"
              className={summary.rv.viewReady ? 'btn-primary-soft' : ''}
              onClick={() => void run('rv')}
              disabled={s.closed || busy !== null}
              data-tip="Traza el endocardio del VD en TD y TS sobre apical enfocada en VD o A4C (calcula FEVD estimada y CAF)"
            >
              {busy === 'rv' ? 'Midiendo…' : 'Auto-FE VD'}
            </button>
          </div>
        </div>

        {summary.rv.hasResult &&
          summary.rv.efPct !== null &&
          summary.rv.edvMl !== null &&
          summary.rv.esvMl !== null &&
          summary.rv.svMl !== null && (
            <>
              <div className="auto-ef-result-line">
                <span className="auto-ef-primary">
                  FEVD {summary.rv.efPct.toFixed(0)} %
                  {summary.rv.facPct !== null ? ` · CAF ${summary.rv.facPct.toFixed(0)} %` : ''}
                </span>
              </div>
              <div className="auto-ef-metrics">
                <span>VTD {summary.rv.edvMl.toFixed(0)} mL</span>
                <span>VTS {summary.rv.esvMl.toFixed(0)} mL</span>
                <span>VS {summary.rv.svMl.toFixed(0)} mL</span>
                {summary.rv.edAreaCm2 !== null && summary.rv.esAreaCm2 !== null && (
                  <span>
                    Área {summary.rv.edAreaCm2.toFixed(1)}/{summary.rv.esAreaCm2.toFixed(1)} cm²
                  </span>
                )}
              </div>
              {summary.rv.edFrameId !== null && summary.rv.esFrameId !== null && (
                <div className="auto-ef-jump">
                  <button
                    type="button"
                    className={`ghost${displayedFrameId === summary.rv.edFrameId ? ' active' : ''}`}
                    disabled={s.closed || !rvEdInCine}
                    data-tip={rvEdInCine ? 'Inspeccionar contorno del VD en TD' : 'El cuadro TD salió del bucle de cine actual'}
                    onClick={() => inspectFrame('ed', summary.rv.edFrameId!)}
                  >
                    Ver TD
                  </button>
                  <button
                    type="button"
                    className={`ghost${displayedFrameId === summary.rv.esFrameId ? ' active' : ''}`}
                    disabled={s.closed || !rvEsInCine}
                    data-tip={rvEsInCine ? 'Inspeccionar contorno del VD en TS' : 'El cuadro TS salió del bucle de cine actual'}
                    onClick={() => inspectFrame('es', summary.rv.esFrameId!)}
                  >
                    Ver TS
                  </button>
                </div>
              )}
            </>
          )}
      </div>

      {/* Contextual one-click helpers when not yet in 2D or apical window */}
      {!summary.is2dReady && !s.closed && (
        <div className="auto-ef-helper">
          <span>Auto-FE requiere imagen 2D</span>
          <button type="button" className="ghost" onClick={() => s.setModality('2d')}>
            Volver a 2D
          </button>
        </div>
      )}
      {showApicalShortcuts && !s.closed && (
        <div className="auto-ef-helper">
          <span>Vista apical:</span>
          <div className="auto-ef-presets">
            <button type="button" className="ghost" onClick={() => jumpPreset('a4c')}>
              A4C
            </button>
            <button type="button" className="ghost" onClick={() => jumpPreset('a2c')}>
              A2C
            </button>
            <button type="button" className="ghost" onClick={() => jumpPreset('rv-focused')}>
              VD
            </button>
          </div>
        </div>
      )}

      {notice && (
        <div
          className={`auto-ef-notice ${notice.kind === 'err' ? 'err' : 'ok'}`}
          role="status"
        >
          {notice.text}
        </div>
      )}
    </div>
  );
}

/** Guided LVOT PW gate placement in A5C/A3C + one-click automatic systolic VTI tracing. */
function AutoLvotVtiPanel({
  currentViewId,
  showViewFeedback,
  presetsEnabled,
}: {
  currentViewId: string | null;
  showViewFeedback: boolean;
  presetsEnabled: boolean;
}) {
  const s = useSimStore(
    useShallow((st) => ({
      measurements: st.measurements,
      modality: st.modality,
      closed: examClosed(st),
      startPresetView: st.startPresetView,
      setModality: st.setModality,
    })),
  );
  // Quantized gate status selector (re-renders only when gate structure or rounded angle actually changes)
  const gateState = useHudStore(
    useShallow((h) => {
      const g = h.hud?.gate;
      if (!g) return { inLvot: false, angleDeg: null as number | null };
      return {
        inLvot: g.structure === Structure.Lvot || g.peakFlowStructure === Structure.Lvot,
        angleDeg: g.flowAngleDeg !== null ? Math.round(g.flowAngleDeg) : null,
      };
    }),
  );
  const [busy, setBusy] = useState<'gate' | 'vti' | null>(null);
  const [notice, setNotice] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);

  const isPw = s.modality === 'pw';
  const onOutflowView = currentViewId === 'a5c' || currentViewId === 'a3c';
  const latestVti = useMemo(() => {
    for (let i = s.measurements.length - 1; i >= 0; i--) {
      if (s.measurements[i]!.measurementId === 'lvot-vti') return s.measurements[i]!;
    }
    return null;
  }, [s.measurements]);
  const latestLvotDiam = useMemo(() => {
    for (let i = s.measurements.length - 1; i >= 0; i--) {
      if (s.measurements[i]!.measurementId === 'lvot-diameter') return s.measurements[i]!;
    }
    return null;
  }, [s.measurements]);
  const dopplerSvMl =
    latestVti && latestLvotDiam && latestLvotDiam.value > 0
      ? strokeVolume(latestLvotDiam.value, latestVti.value)
      : null;

  const placeGate = async () => {
    if (busy || s.closed || !presetsEnabled) return;
    setBusy('gate');
    setNotice(null);
    try {
      const res = await runAutoLvotGate();
      if (!res.ok) {
        setNotice({ kind: 'err', text: res.message });
      } else {
        setNotice({
          kind: 'ok',
          text: `Gate PW colocado en TSVI (${res.rCm.toFixed(1)} cm · ~${res.distToAvCm.toFixed(1)} cm proximal a VAo). Espera 1 latido y pulsa Auto-VTI.`,
        });
      }
    } catch (e: unknown) {
      setNotice({
        kind: 'err',
        text: e instanceof Error ? e.message : 'Error al colocar el gate en el TSVI.',
      });
    } finally {
      setBusy(null);
    }
  };

  const traceVti = async () => {
    if (busy || s.closed) return;
    setBusy('vti');
    setNotice(null);
    try {
      const res = await runAutoSpectralVti('lvot-vti');
      if (!res.ok) {
        setNotice({ kind: 'err', text: res.message });
      } else {
        setNotice({
          kind: 'ok',
          text: `VTI TSVI ${res.vtiCm.toFixed(1)} cm (Vmáx ${Math.round(res.vmaxMps * 100)} cm/s).`,
        });
      }
    } catch (e: unknown) {
      setNotice({
        kind: 'err',
        text: e instanceof Error ? e.message : 'Error al trazar el VTI automático.',
      });
    } finally {
      setBusy(null);
    }
  };

  const goA5c = () => {
    if (s.closed || !presetsEnabled) return;
    setNotice(null);
    if (s.modality !== '2d' && s.modality !== 'pw') s.setModality('2d');
    s.startPresetView('a5c');
  };

  return (
    <div className="auto-ef" data-auto-lvot="panel">
      <div className="auto-ef-card">
        <div className="auto-ef-row-head">
          <span className="auto-ef-title">TSVI · PW</span>
          {showViewFeedback && isPw && (
            <span
              className={`plane-pill${
                gateState.inLvot && (gateState.angleDeg ?? 0) <= 20
                  ? ' done'
                  : gateState.inLvot
                    ? ' here'
                    : ''
              }`}
            >
              {gateState.inLvot
                ? `✓ Gate TSVI${gateState.angleDeg !== null ? ` · θ ${gateState.angleDeg}°` : ''}`
                : 'Gate fuera TSVI'}
            </span>
          )}
        </div>
        <div className="auto-lvot-actions">
          {presetsEnabled && showViewFeedback && !onOutflowView && (
            <button
              type="button"
              className="ghost"
              disabled={s.closed || busy !== null}
              onClick={goA5c}
              data-tip="Mueve la sonda a A5C para alinear el TSVI"
            >
              Ir a A5C
            </button>
          )}
          {presetsEnabled && (
            <button
              type="button"
              className={onOutflowView && !isPw ? 'btn-primary-soft' : ''}
              disabled={s.closed || busy !== null}
              onClick={() => void placeGate()}
              data-tip="Detecta el TSVI en el corte 2D actual, coloca el gate PW a ~0,5 cm de la válvula aórtica y activa PW"
            >
              {busy === 'gate' ? 'Buscando…' : 'Gate TSVI'}
            </button>
          )}
          <button
            type="button"
            className={isPw ? 'btn-primary-soft' : ''}
            disabled={s.closed || busy !== null}
            onClick={() => void traceVti()}
            data-tip="Congela la tira PW y traza automáticamente el latido sistólico más reciente (VTI, Vmáx y gradientes)"
          >
            {busy === 'vti' ? 'Trazando…' : 'Auto-VTI'}
          </button>
        </div>

        {latestVti && (
          <div className="auto-ef-result-line">
            <span className="auto-ef-primary">VTI {latestVti.value.toFixed(1)} cm</span>
            <span className="auto-ef-metrics">
              {latestVti.derived?.vmaxMps !== undefined && (
                <span>Vmáx {Math.round(latestVti.derived.vmaxMps * 100)} cm/s</span>
              )}
              {latestVti.derived?.meanGradientMmHg !== undefined && (
                <span>ΔPm {latestVti.derived.meanGradientMmHg.toFixed(1)} mmHg</span>
              )}
              {dopplerSvMl !== null && <span>VS {dopplerSvMl.toFixed(0)} mL</span>}
            </span>
          </div>
        )}
      </div>

      {notice && (
        <div
          className={`auto-ef-notice ${notice.kind === 'err' ? 'err' : 'ok'}`}
          role="status"
        >
          {notice.text}
        </div>
      )}
    </div>
  );
}
