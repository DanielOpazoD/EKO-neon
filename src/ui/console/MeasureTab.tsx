import { examClosed, modePolicy, showsCaseIdentity } from '@/app/modePolicy';
import { formatCaptureCaption } from '@/app/studyCapture';
import { useHudStore, useSimStore, type SimStore } from '@/app/store';
import { loadCaseById } from '@/cases';
import { strokeVolume } from '@/clinical/formulas';
import { aggregateProtocolMeasurement, deriveCalculations } from '@/education/report';
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
  runAutoDopplerTarget,
  runAutoLaVolume,
  runAutoLvotGate,
  runAutoPlax,
  runAutoSpectralVti,
  runAutoStripMeasure,
} from '../display/measurementTools';
import type { AutoStripMeasureTarget } from '@/simulator/core/strip/autoMeasureStrip';
import type { DopplerTargetId } from '@/simulator/measurements/autoDopplerTarget';
import { pendingSession, usePendingPointCount } from '../display/pendingSession';
import { MeasurementPanel } from '../MeasurementPanel';
import { ViewQuickMenu } from '../ViewQuickMenu';
import {
  captureModalityCheck,
  computeQuickStudySteps,
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
      startPresetView: st.startPresetView,
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

  const quickStudy = useMemo(
    () =>
      policy.presetsEnabled && showView
        ? computeQuickStudySteps(s.measurements, currentViewId)
        : null,
    [policy.presetsEnabled, showView, s.measurements, currentViewId],
  );

  const jumpQuickStudyView = (targetViewId: string) => {
    if (s.closed || !policy.presetsEnabled) return;
    if (currentViewId !== targetViewId && s.modality !== '2d') {
      s.setModality('2d');
    }
    s.startPresetView(targetViewId);
  };

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
              onClick={() => s.toggleFreeze()}
              data-tip={s.frozen ? 'Reanudar imagen en vivo (Espacio)' : 'Congelar imagen para medir (Espacio)'}
            >
              {s.frozen ? 'Congelada · Reanudar' : 'En vivo · Congelar'}
            </button>
          </div>

          {quickStudy && !s.closed && (
            <div
              className="quick-study-strip"
              role="group"
              aria-label="Estudio rápido en 4 pasos"
              data-quick-study="strip"
            >
              <div className="quick-study-head">
                <span>Estudio rápido</span>
                <span className={`quick-study-count${quickStudy.completedCount === 4 ? ' complete' : ''}`}>
                  {quickStudy.completedCount}/4
                </span>
              </div>
              <div className="quick-study-pills">
                {quickStudy.steps.map((step) => (
                  <button
                    key={step.id}
                    type="button"
                    className={`qs-pill ${step.status}${step.active ? ' active' : ''}`}
                    onClick={() => jumpQuickStudyView(step.targetViewId)}
                    data-tip={step.tip}
                  >
                    <span className="qs-label">
                      {step.status === 'done' ? '✓ ' : ''}
                      {step.label}
                    </span>
                    {step.detail && <span className="qs-detail">{step.detail}</span>}
                  </button>
                ))}
              </div>
            </div>
          )}

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
            title="Mediciones 2D automáticas"
            tip={tip(
              'Dimensiones lineales en PLAX (TD/Mesosístole/TS), Simpson biplano del VI, volumen biplano de AI (LAVI) y VD 2D directamente desde la imagen 2D del cine actual.',
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
            title="Asistente Doppler y Modo M"
            tip={tip(
              'Coloca el gate o cursor a partir de la anatomía 2D visible (Mitral E/A, TDI e′ septal/lateral, TAPSE, IT CW y TSVI con Auto-VTI).',
            )}
          >
            <AutoLvotVtiPanel
              currentViewId={currentViewId}
              showViewFeedback={showView}
              showIdentity={showIdentity}
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
                <img src={cap.dataUrl} alt={`Captura ${idx + 1}: ${formatCaptureCaption(cap, showIdentity)}`} />
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

/** Guided one-click automatic 2D measurements: PLAX linear dimensions, LV Simpson, LA volume (LAVI), and RV 2D. */
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
      truth: st.truth,
      closed: examClosed(st),
      setModality: st.setModality,
      startPresetView: st.startPresetView,
      toggleFreeze: st.toggleFreeze,
      jumpCinePhase: st.jumpCinePhase,
    })),
  );
  const [busy, setBusy] = useState<'plax' | 'lv' | 'la' | 'rv' | null>(null);
  const [notice, setNotice] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);

  const summary = useMemo(
    () => summarizeAutoEf(s.measurements, currentViewId, s.modality, showViewFeedback),
    [s.measurements, currentViewId, s.modality, showViewFeedback],
  );
  const bsaM2 = s.truth?.bsaM2 && s.truth.bsaM2 > 0 ? s.truth.bsaM2 : null;
  const lvmi = summary.plax.massG !== null && bsaM2 ? summary.plax.massG / bsaM2 : null;
  const lavi = summary.la.volumeMl !== null && bsaM2 ? summary.la.volumeMl / bsaM2 : null;

  const runPlax = async () => {
    if (busy || s.closed) return;
    setBusy('plax');
    setNotice(null);
    try {
      const res = await runAutoPlax();
      if (!res.ok) {
        setNotice({ kind: 'err', text: res.message });
      } else {
        const byId = new Map(res.calipers.map((c) => [c.measurementId, c.valueCm]));
        const parts: string[] = [];
        if (byId.has('lv-edd')) parts.push(`DTDVI ${byId.get('lv-edd')!.toFixed(1)} cm`);
        if (byId.has('lvot-diameter')) parts.push(`TSVI ${byId.get('lvot-diameter')!.toFixed(2)} cm`);
        if (byId.has('la-ap')) parts.push(`AI ${byId.get('la-ap')!.toFixed(1)} cm`);
        setNotice({
          kind: 'ok',
          text: `PLAX: ${res.calipers.length} dimensiones trazadas${parts.length ? ` (${parts.join(' · ')})` : ''}.`,
        });
      }
    } catch (e: unknown) {
      setNotice({
        kind: 'err',
        text: e instanceof Error ? e.message : 'Error al calcular dimensiones PLAX.',
      });
    } finally {
      setBusy(null);
    }
  };

  const runLa = async () => {
    if (busy || s.closed) return;
    setBusy('la');
    setNotice(null);
    try {
      const res = await runAutoLaVolume();
      if (!res.ok) {
        setNotice({ kind: 'err', text: res.message });
      } else {
        const idxText = bsaM2 ? ` · LAVI ${(res.volumeMl / bsaM2).toFixed(0)} mL/m²` : '';
        const depthWarn = res.cutByDepth
          ? ' ⚠ Fondo de AI recortado por profundidad: aumenta Profundidad.'
          : '';
        setNotice({
          kind: 'ok',
          text: `AI trazada en TS (${res.volumeMl.toFixed(0)} mL${idxText}).${depthWarn}`,
        });
      }
    } catch (e: unknown) {
      setNotice({
        kind: 'err',
        text: e instanceof Error ? e.message : 'Error al calcular el volumen de la AI.',
      });
    } finally {
      setBusy(null);
    }
  };

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

  const showQuickViewShortcuts =
    presetsEnabled &&
    showViewFeedback &&
    summary.is2dReady &&
    !summary.plax.viewReady &&
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
  const showPlaxPrompt =
    presetsEnabled &&
    showViewFeedback &&
    summary.is2dReady &&
    !summary.plax.viewReady &&
    !summary.plax.hasResult;
  const plaxEdInCine =
    displayedFrameId === summary.plax.edFrameId ||
    isTimeInCineWindow(summary.plax.edTimeS, cineWindow);
  const plaxMidInCine =
    displayedFrameId === summary.plax.midSysFrameId ||
    isTimeInCineWindow(summary.plax.midSysTimeS, cineWindow);
  const plaxEsInCine =
    displayedFrameId === summary.plax.esFrameId ||
    isTimeInCineWindow(summary.plax.esTimeS, cineWindow);
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
  const laA4cInCine =
    summary.la.a4c !== null &&
    (displayedFrameId === summary.la.a4c.frameId ||
      isTimeInCineWindow(summary.la.a4c.timeS, cineWindow));
  const laA2cInCine =
    summary.la.a2c !== null &&
    (displayedFrameId === summary.la.a2c.frameId ||
      isTimeInCineWindow(summary.la.a2c.timeS, cineWindow));
  const laEsInCine =
    displayedFrameId === summary.la.esFrameId || isTimeInCineWindow(summary.la.esTimeS, cineWindow);
  const rvEdInCine =
    displayedFrameId === summary.rv.edFrameId || isTimeInCineWindow(summary.rv.edTimeS, cineWindow);
  const rvEsInCine =
    displayedFrameId === summary.rv.esFrameId || isTimeInCineWindow(summary.rv.esTimeS, cineWindow);

  const plaxCard = (
    <div className="auto-ef-card" data-auto-ef="plax">
      <div className="auto-ef-row-head">
        <div className="auto-ef-chamber">
          <span className="auto-ef-title">PLAX · Dimensiones 2D</span>
          <span className="plane-pills" aria-label="Vista PLAX">
            <span
              className={`plane-pill${summary.plax.hasResult ? ' done' : summary.plax.viewReady ? ' here' : ''}`}
            >
              {summary.plax.hasResult ? '✓ PLAX' : 'PLAX'}
            </span>
          </span>
        </div>
        <div className="auto-ef-presets">
          {showPlaxPrompt && (
            <button
              type="button"
              className="ghost"
              disabled={s.closed}
              onClick={() => jumpPreset('plax')}
              data-tip="Mueve la sonda a la vista paraesternal eje largo (PLAX)"
            >
              Ir a PLAX
            </button>
          )}
          <button
            type="button"
            className={summary.plax.viewReady ? 'btn-primary-soft' : ''}
            onClick={() => void runPlax()}
            disabled={s.closed || busy !== null}
            data-tip="Mide SIVd, DTDVI, PPd y raíz Ao en TD, TSVI ⌀ en mesosístole y DTSVI + AI en TS desde la imagen 2D"
          >
            {busy === 'plax' ? 'Midiendo…' : 'Auto-PLAX'}
          </button>
        </div>
      </div>

      {summary.plax.hasResult && (
        <>
          <div className="auto-ef-result-line">
            <span className="auto-ef-primary">
              {summary.plax.lvEddCm !== null
                ? `DTDVI ${summary.plax.lvEddCm.toFixed(1)} cm`
                : summary.plax.lvotCm !== null
                  ? `TSVI ${summary.plax.lvotCm.toFixed(2)} cm`
                  : 'PLAX medido'}
              {summary.plax.fsPct !== null ? ` · FA ${summary.plax.fsPct.toFixed(0)} %` : ''}
            </span>
            <span className="auto-ef-metrics">
              {lvmi !== null ? (
                <span>IMVI {lvmi.toFixed(0)} g/m²</span>
              ) : summary.plax.massG !== null ? (
                <span>Masa {summary.plax.massG.toFixed(0)} g</span>
              ) : null}
              {summary.plax.rwt !== null && <span>GPR {summary.plax.rwt.toFixed(2)}</span>}
            </span>
          </div>
          <div className="auto-ef-metrics">
            {summary.plax.ivsdCm !== null && <span>SIVd {summary.plax.ivsdCm.toFixed(2)}</span>}
            {summary.plax.lvpwdCm !== null && <span>PPd {summary.plax.lvpwdCm.toFixed(2)}</span>}
            {summary.plax.lvEsdCm !== null && <span>DTSVI {summary.plax.lvEsdCm.toFixed(1)}</span>}
            {summary.plax.lvotCm !== null && <span>TSVI {summary.plax.lvotCm.toFixed(2)}</span>}
            {summary.plax.aoRootCm !== null && <span>Ao {summary.plax.aoRootCm.toFixed(1)}</span>}
            {summary.plax.laApCm !== null && <span>AI {summary.plax.laApCm.toFixed(1)} cm</span>}
          </div>
          {(summary.plax.edFrameId !== null ||
            summary.plax.midSysFrameId !== null ||
            summary.plax.esFrameId !== null) && (
            <div className="auto-ef-jump">
              {summary.plax.edFrameId !== null && (
                <button
                  type="button"
                  className={`ghost${displayedFrameId === summary.plax.edFrameId ? ' active' : ''}`}
                  disabled={s.closed || !plaxEdInCine}
                  data-tip={
                    plaxEdInCine
                      ? 'Ver y ajustar SIVd, DTDVI, PPd y raíz Ao en TD'
                      : 'El cuadro TD salió del bucle de cine actual'
                  }
                  onClick={() => inspectFrame('ed', summary.plax.edFrameId!)}
                >
                  Ver TD
                </button>
              )}
              {summary.plax.midSysFrameId !== null && (
                <button
                  type="button"
                  className={`ghost${displayedFrameId === summary.plax.midSysFrameId ? ' active' : ''}`}
                  disabled={s.closed || !plaxMidInCine}
                  data-tip={
                    plaxMidInCine
                      ? 'Ver y ajustar el diámetro del TSVI en mesosístole'
                      : 'El cuadro mesosistólico salió del bucle de cine actual'
                  }
                  onClick={() => inspectFrame('ed', summary.plax.midSysFrameId!)}
                >
                  Ver TSVI
                </button>
              )}
              {summary.plax.esFrameId !== null && (
                <button
                  type="button"
                  className={`ghost${displayedFrameId === summary.plax.esFrameId ? ' active' : ''}`}
                  disabled={s.closed || !plaxEsInCine}
                  data-tip={
                    plaxEsInCine
                      ? 'Ver y ajustar DTSVI y diámetro AP de AI en TS'
                      : 'El cuadro TS salió del bucle de cine actual'
                  }
                  onClick={() => inspectFrame('es', summary.plax.esFrameId!)}
                >
                  Ver TS
                </button>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );

  const laCard = (
    <div className="auto-ef-card" data-auto-ef="la">
      <div className="auto-ef-row-head">
        <div className="auto-ef-chamber">
          <span className="auto-ef-title">
            AI · {summary.la.biplane ? 'Biplano' : 'Volumen'}
          </span>
          <span className="plane-pills" aria-label="Planos capturados de la AI">
            <span
              className={`plane-pill${summary.la.a4cDone ? ' done' : currentViewId === 'a4c' ? ' here' : ''}`}
            >
              {summary.la.a4cDone ? '✓ A4C' : 'A4C'}
            </span>
            <span
              className={`plane-pill${summary.la.a2cDone ? ' done' : currentViewId === 'a2c' ? ' here' : ''}`}
            >
              {summary.la.a2cDone ? '✓ A2C' : 'A2C'}
            </span>
          </span>
        </div>
        <button
          type="button"
          className={summary.la.viewReady ? 'btn-primary-soft' : ''}
          onClick={() => void runLa()}
          disabled={s.closed || busy !== null}
          data-tip="Traza el endocardio de la AI en TS (excluyendo orejuela y venas pulmonares) sobre A4C o A2C; repite en ambos para LAVI biplano"
        >
          {busy === 'la' ? 'Midiendo…' : 'Auto-Vol AI'}
        </button>
      </div>

      {summary.la.hasResult && summary.la.volumeMl !== null && (
        <>
          <div className="auto-ef-result-line">
            <span className="auto-ef-primary">
              {lavi !== null
                ? `LAVI ${lavi.toFixed(0)} mL/m²`
                : `Vol AI ${summary.la.volumeMl.toFixed(0)} mL`}
            </span>
            <span className="auto-ef-metrics">
              <span>Vol {summary.la.volumeMl.toFixed(0)} mL</span>
              <span>{summary.la.biplane ? 'Biplano' : 'Monoplano'}</span>
            </span>
          </div>

          {(summary.la.a4c || summary.la.a2c) && (
            <div className="biplane-breakdown" aria-label="Desglose volumen AI A4C y A2C">
              {([
                {
                  id: 'a4c' as const,
                  label: 'A4C',
                  plane: summary.la.a4c,
                  inCine: laA4cInCine,
                },
                {
                  id: 'a2c' as const,
                  label: 'A2C',
                  plane: summary.la.a2c,
                  inCine: laA2cInCine,
                },
              ]).map(({ id, label, plane, inCine }) => (
                <div key={id} className={`biplane-row${plane ? ' done' : ''}`}>
                  <span className="bp-tag">{plane ? `✓ ${label}` : `○ ${label}`}</span>
                  {plane ? (
                    <>
                      <span className="bp-vals">
                        <b>{plane.volumeMl.toFixed(0)} mL</b>
                        {plane.areaCm2 !== null ? ` · ${plane.areaCm2.toFixed(1)} cm²` : ''}
                        {plane.longAxisCm !== null ? ` · L ${plane.longAxisCm.toFixed(1)}` : ''}
                      </span>
                      <span className="bp-jumps">
                        <button
                          type="button"
                          className={`ghost${displayedFrameId === plane.frameId ? ' active' : ''}`}
                          disabled={s.closed || !inCine}
                          data-tip={inCine ? `Ver AI ${label} en TS` : 'Fuera del cine actual'}
                          onClick={() => inspectFrame('es', plane.frameId)}
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
              {summary.la.lengthMismatch && (
                <div className="biplane-warn" role="status">
                  ⚠ Longitud AI dispar (&gt;0,5 cm) entre A4C y A2C (criterio ASE).
                </div>
              )}
            </div>
          )}

          {!summary.la.a4c && !summary.la.a2c && summary.la.esFrameId !== null && (
            <div className="auto-ef-jump">
              <button
                type="button"
                className={`ghost${displayedFrameId === summary.la.esFrameId ? ' active' : ''}`}
                disabled={s.closed || !laEsInCine}
                data-tip={
                  laEsInCine
                    ? 'Inspeccionar contorno de la AI en TS'
                    : 'El cuadro TS salió del bucle de cine actual'
                }
                onClick={() => inspectFrame('es', summary.la.esFrameId!)}
              >
                Ver TS
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );

  const onPlaxFirst = showViewFeedback && currentViewId === 'plax';

  return (
    <div className="auto-ef">
      {onPlaxFirst && plaxCard}

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

      {/* LA chamber block */}
      {laCard}

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

      {!onPlaxFirst && plaxCard}

      {/* Contextual one-click helpers when not yet in 2D or target window */}
      {!summary.is2dReady && !s.closed && (
        <div className="auto-ef-helper">
          <span>Las mediciones 2D requieren imagen 2D</span>
          <button type="button" className="ghost" onClick={() => s.setModality('2d')}>
            Volver a 2D
          </button>
        </div>
      )}
      {showQuickViewShortcuts && !s.closed && (
        <div className="auto-ef-helper">
          <span>Vista rápida:</span>
          <div className="auto-ef-presets">
            <button type="button" className="ghost" onClick={() => jumpPreset('plax')}>
              PLAX
            </button>
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

interface ViewDopplerPreset {
  id: Exclude<DopplerTargetId, 'lvot'>;
  label: string;
  tip: string;
}

const VIEW_DOPPLER_TARGETS: Record<string, readonly ViewDopplerPreset[]> = {
  a4c: [
    {
      id: 'mitral-inflow',
      label: 'Mitral PW',
      tip: 'Coloca el gate PW en las puntas mitrales para E/A y TD',
    },
    {
      id: 'tdi-septal',
      label: 'e′ sept TDI',
      tip: 'Coloca el gate TDI en el anillo mitral septal',
    },
    {
      id: 'tdi-lateral',
      label: 'e′ lat TDI',
      tip: 'Coloca el gate TDI en el anillo mitral lateral',
    },
    {
      id: 'tapse',
      label: 'TAPSE M',
      tip: 'Alinea el cursor de Modo M con el anillo tricuspídeo lateral',
    },
    {
      id: 'tr-cw',
      label: 'IT CW',
      tip: 'Alinea el cursor CW a través de la válvula tricúspide para Vmax IT',
    },
  ],
  'rv-focused': [
    {
      id: 'tapse',
      label: 'TAPSE M',
      tip: 'Alinea el cursor de Modo M con el anillo tricuspídeo lateral',
    },
    {
      id: 'tr-cw',
      label: 'IT CW',
      tip: 'Alinea el cursor CW a través de la válvula tricúspide para Vmax IT',
    },
  ],
  a2c: [
    {
      id: 'mitral-inflow',
      label: 'Mitral PW',
      tip: 'Coloca el gate PW en las puntas mitrales',
    },
  ],
  a3c: [
    {
      id: 'mitral-inflow',
      label: 'Mitral PW',
      tip: 'Coloca el gate PW en las puntas mitrales',
    },
  ],
  'psax-av': [
    {
      id: 'tr-cw',
      label: 'IT CW',
      tip: 'Alinea el cursor CW a través de la válvula tricúspide',
    },
  ],
  'subcostal-4c': [
    {
      id: 'tr-cw',
      label: 'IT CW',
      tip: 'Alinea el cursor CW a través de la válvula tricúspide',
    },
  ],
};

/** Guided pure-2D Doppler/M-mode cursor placement + LVOT PW gate and 1-click Auto-VTI tracing. */
function AutoLvotVtiPanel({
  currentViewId,
  showViewFeedback,
  showIdentity,
  presetsEnabled,
}: {
  currentViewId: string | null;
  showViewFeedback: boolean;
  showIdentity: boolean;
  presetsEnabled: boolean;
}) {
  const s = useSimStore(
    useShallow((st) => ({
      measurements: st.measurements,
      modality: st.modality,
      truth: st.truth,
      closed: examClosed(st),
      startPresetView: st.startPresetView,
      setModality: st.setModality,
    })),
  );
  // Quantized gate status selector (re-renders only when gate structure or rounded angle actually changes)
  const gateState = useHudStore(
    useShallow((h) => {
      const g = h.hud?.gate;
      if (!g) return { inLvot: false, angleDeg: null as number | null, tdiLateral: false };
      return {
        inLvot: g.structure === Structure.Lvot || g.peakFlowStructure === Structure.Lvot,
        angleDeg: g.flowAngleDeg !== null ? Math.round(g.flowAngleDeg) : null,
        tdiLateral:
          g.structure === Structure.LvWallLateral || g.peakFlowStructure === Structure.LvWallLateral,
      };
    }),
  );
  const [busy, setBusy] = useState<
    'gate' | 'vti' | 'av-vti' | DopplerTargetId | AutoStripMeasureTarget | 'tdi-lat-m' | null
  >(null);
  const [tdiSiteHint, setTdiSiteHint] = useState<'septal' | 'lateral'>('septal');
  const [notice, setNotice] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);

  const isPw = s.modality === 'pw';
  const isCw = s.modality === 'cw';
  const isTdi = s.modality === 'tdi';
  const isMmode = s.modality === 'm-mode' || s.modality === 'cmm';
  const onOutflowView = currentViewId === 'a5c' || currentViewId === 'a3c';
  const viewTargets =
    presetsEnabled && showViewFeedback && currentViewId
      ? (VIEW_DOPPLER_TARGETS[currentViewId] ?? [])
      : [];
  const aggOfId = (id: string) => aggregateProtocolMeasurement(s.measurements, id);
  const latestVti = useMemo(() => aggOfId('lvot-vti'), [s.measurements]);
  const latestAvVti = useMemo(() => aggOfId('av-vti'), [s.measurements]);
  const latestLvotDiam = useMemo(() => aggOfId('lvot-diameter'), [s.measurements]);
  const latestE = useMemo(() => aggOfId('mitral-e'), [s.measurements]);
  const latestA = useMemo(() => aggOfId('mitral-a'), [s.measurements]);
  const latestEpSept = useMemo(() => aggOfId('e-prime-septal'), [s.measurements]);
  const latestEpLat = useMemo(() => aggOfId('e-prime-lateral'), [s.measurements]);
  const latestTr = useMemo(() => aggOfId('tr-vmax'), [s.measurements]);
  const latestTapse = useMemo(() => aggOfId('tapse'), [s.measurements]);
  const showAfHint = showIdentity && s.truth?.rhythm === 'atrial-fibrillation';
  const dopplerSvMl =
    latestVti && latestLvotDiam && latestLvotDiam.value > 0
      ? strokeVolume(latestLvotDiam.value, latestVti.value)
      : null;
  const continuityAvaCm2 =
    dopplerSvMl !== null && latestAvVti && latestAvVti.value > 0
      ? dopplerSvMl / latestAvVti.value
      : null;
  const velocityRatio =
    latestVti && latestAvVti && latestAvVti.value > 0 ? latestVti.value / latestAvVti.value : null;
  const activeTdiSite: 'septal' | 'lateral' = gateState.tdiLateral ? 'lateral' : tdiSiteHint;

  const placeTarget = async (target: Exclude<DopplerTargetId, 'lvot'>) => {
    if (busy || s.closed || !presetsEnabled) return;
    if (target === 'tdi-septal') setTdiSiteHint('septal');
    else if (target === 'tdi-lateral') setTdiSiteHint('lateral');
    setBusy(target);
    setNotice(null);
    try {
      const res = await runAutoDopplerTarget(target);
      if (!res.ok) {
        setNotice({ kind: 'err', text: res.message });
      } else {
        setNotice({ kind: 'ok', text: res.summary });
      }
    } catch (e: unknown) {
      setNotice({
        kind: 'err',
        text: e instanceof Error ? e.message : 'Error al posicionar el cursor.',
      });
    } finally {
      setBusy(null);
    }
  };

  const measureStrip = async (
    target: AutoStripMeasureTarget,
    tdiSite?: 'septal' | 'lateral',
  ) => {
    if (busy || s.closed) return;
    setBusy(target === 'tdi-e-prime' && tdiSite === 'lateral' ? 'tdi-lat-m' : target);
    setNotice(null);
    try {
      const res = await runAutoStripMeasure(target, tdiSite ? { tdiSite } : undefined);
      if (!res.ok) {
        setNotice({ kind: 'err', text: res.message });
      } else if (res.target === 'mitral-inflow') {
        const parts = [`E ${Math.abs(res.e.velocityMps).toFixed(2)} m/s`];
        if (res.a) parts.push(`A ${Math.abs(res.a.velocityMps).toFixed(2)} m/s`);
        if (res.dt) parts.push(`TD ${Math.round(res.dt.dtMs)} ms`);
        setNotice({ kind: 'ok', text: `Llenado mitral: ${parts.join(' · ')}.` });
      } else if (res.target === 'tdi-e-prime') {
        const siteLabel = tdiSite === 'lateral' ? 'lateral' : 'septal';
        setNotice({
          kind: 'ok',
          text: `e′ ${siteLabel}: ${(Math.abs(res.ePrime.velocityMps) * 100).toFixed(1)} cm/s.`,
        });
      } else if (res.target === 'tr-vmax') {
        setNotice({
          kind: 'ok',
          text: `Vmax IT ${Math.abs(res.vmax.velocityMps).toFixed(2)} m/s (ΔP ${res.vmax.peakGradientMmHg.toFixed(0)} mmHg).`,
        });
      } else {
        setNotice({ kind: 'ok', text: `TAPSE ${res.tapseCm.toFixed(2)} cm.` });
      }
    } catch (e: unknown) {
      setNotice({
        kind: 'err',
        text: e instanceof Error ? e.message : 'Error al medir automáticamente sobre la tira.',
      });
    } finally {
      setBusy(null);
    }
  };

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

  const traceVti = async (id: 'lvot-vti' | 'av-vti' = 'lvot-vti') => {
    if (busy || s.closed) return;
    setBusy(id === 'av-vti' ? 'av-vti' : 'vti');
    setNotice(null);
    try {
      const res = await runAutoSpectralVti(id);
      if (!res.ok) {
        setNotice({ kind: 'err', text: res.message });
      } else if (id === 'av-vti') {
        setNotice({
          kind: 'ok',
          text: `VTI VAo ${res.vtiCm.toFixed(1)} cm · Vmáx ${res.vmaxMps.toFixed(2)} m/s · ΔPm ${res.meanGradientMmHg.toFixed(0)} mmHg.`,
        });
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
    if (s.modality !== '2d' && s.modality !== 'pw' && s.modality !== 'cw') s.setModality('2d');
    s.startPresetView('a5c');
  };

  const showStripAutoCard = isPw || isTdi || isCw || isMmode;
  const hasStripSummary = Boolean(
    latestE || latestEpSept || latestEpLat || latestTr || latestTapse,
  );
  const viewTargetsCard =
    viewTargets.length > 0 || showStripAutoCard || hasStripSummary ? (
      <div className="auto-ef-card" data-auto-doppler="view-targets">
        {viewTargets.length > 0 && (
          <>
            <div className="auto-ef-row-head">
              <span className="auto-ef-title">
                Alineación rápida · {currentViewId ? viewLabel(currentViewId) : ''}
              </span>
            </div>
            <div className="doppler-assist-grid">
              {viewTargets.map((t) => (
                <button
                  key={t.id}
                  type="button"
                  className="ghost"
                  disabled={s.closed || busy !== null}
                  onClick={() => void placeTarget(t.id)}
                  data-tip={t.tip}
                >
                  {busy === t.id ? 'Buscando…' : t.label}
                </button>
              ))}
            </div>
          </>
        )}
        {showStripAutoCard && (
          <div className="auto-lvot-actions" data-auto-strip="actions">
            {isPw && (
              <button
                type="button"
                className={!onOutflowView ? 'btn-primary-soft' : 'ghost'}
                disabled={s.closed || busy !== null}
                onClick={() => void measureStrip('mitral-inflow')}
                data-tip="Congela la tira PW y mide automáticamente onda E, onda A y tiempo de desaceleración (TD E)"
              >
                {busy === 'mitral-inflow' ? 'Midiendo…' : 'Auto E / A / TD'}
              </button>
            )}
            {isTdi && (
              <>
                <button
                  type="button"
                  className={activeTdiSite === 'septal' ? 'btn-primary-soft' : 'ghost'}
                  disabled={s.closed || busy !== null}
                  onClick={() => void measureStrip('tdi-e-prime', 'septal')}
                  data-tip="Congela la tira TDI y detecta automáticamente el pico e′ septal"
                >
                  {busy === 'tdi-e-prime' ? 'Midiendo…' : 'Auto e′ sept'}
                </button>
                <button
                  type="button"
                  className={activeTdiSite === 'lateral' ? 'btn-primary-soft' : 'ghost'}
                  disabled={s.closed || busy !== null}
                  onClick={() => void measureStrip('tdi-e-prime', 'lateral')}
                  data-tip="Congela la tira TDI y detecta automáticamente el pico e′ lateral"
                >
                  {busy === 'tdi-lat-m' ? 'Midiendo…' : 'Auto e′ lat'}
                </button>
              </>
            )}
            {isCw && (
              <button
                type="button"
                className={!onOutflowView ? 'btn-primary-soft' : 'ghost'}
                disabled={s.closed || busy !== null}
                onClick={() => void measureStrip('tr-vmax')}
                data-tip="Congela la tira CW y detecta automáticamente la Vmax de la insuficiencia tricuspídea"
              >
                {busy === 'tr-vmax' ? 'Midiendo…' : 'Auto Vmáx IT'}
              </button>
            )}
            {isMmode && (
              <button
                type="button"
                className="btn-primary-soft"
                disabled={s.closed || busy !== null}
                onClick={() => void measureStrip('tapse')}
                data-tip="Congela la tira Modo M y mide automáticamente la excursión sistólica del anillo tricuspídeo (TAPSE)"
              >
                {busy === 'tapse' ? 'Midiendo…' : 'Auto TAPSE'}
              </button>
            )}
          </div>
        )}
        {hasStripSummary && (
          <div className="auto-ef-metrics">
            {latestE && (
              <span>
                E {latestE.value.toFixed(2)}
                {latestE.isAveraged ? ` (${latestE.beatCount}l)` : ''}
                {latestA ? ` / A ${latestA.value.toFixed(2)}` : ''} m/s
              </span>
            )}
            {latestEpSept && (
              <span>
                e′s {latestEpSept.value.toFixed(1)}
                {latestEpSept.isAveraged ? ` (${latestEpSept.beatCount}l)` : ''}
              </span>
            )}
            {latestEpLat && (
              <span>
                e′l {latestEpLat.value.toFixed(1)}
                {latestEpLat.isAveraged ? ` (${latestEpLat.beatCount}l)` : ''}
              </span>
            )}
            {latestTr && (
              <span>
                IT {latestTr.value.toFixed(2)} m/s
                {latestTr.isAveraged ? ` (${latestTr.beatCount}l)` : ''}
              </span>
            )}
            {latestTapse && (
              <span>
                TAPSE {latestTapse.value.toFixed(2)} cm
                {latestTapse.isAveraged ? ` (${latestTapse.beatCount}l)` : ''}
              </span>
            )}
          </div>
        )}
      </div>
    ) : null;

  const lvotCard = (
    <div className="auto-ef-card">
      <div className="auto-ef-row-head">
        <span className="auto-ef-title">TSVI / VAo · Continuidad</span>
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
            data-tip="Mueve la sonda a A5C para alinear el TSVI y la válvula aórtica"
          >
            Ir a A5C
          </button>
        )}
        {presetsEnabled && (
          <button
            type="button"
            className={onOutflowView && !isPw && !isCw ? 'btn-primary-soft' : ''}
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
          onClick={() => void traceVti('lvot-vti')}
          data-tip="Congela la tira PW y traza automáticamente el latido sistólico más reciente (VTI TSVI)"
        >
          {busy === 'vti' ? 'Trazando…' : 'Auto-VTI TSVI'}
        </button>
      </div>
      <div className="auto-lvot-actions">
        {presetsEnabled && (
          <button
            type="button"
            className={onOutflowView && latestVti && !isCw ? 'btn-primary-soft' : 'ghost'}
            disabled={s.closed || busy !== null}
            onClick={() => void placeTarget('av-cw')}
            data-tip="Alinea el cursor CW a través de la válvula aórtica en A5C/A3C y activa Doppler continuo"
          >
            {busy === 'av-cw' ? 'Buscando…' : 'CW VAo'}
          </button>
        )}
        <button
          type="button"
          className={isCw ? 'btn-primary-soft' : ''}
          disabled={s.closed || busy !== null}
          onClick={() => void traceVti('av-vti')}
          data-tip="Congela la tira CW y traza automáticamente VTI, Vmáx y gradientes transaórticos"
        >
          {busy === 'av-vti' ? 'Trazando…' : 'Auto-VTI VAo'}
        </button>
      </div>

      {(latestVti || latestAvVti) && (
        <div className="auto-ef-result-line">
          {continuityAvaCm2 !== null ? (
            <span className="auto-ef-primary">AVA {continuityAvaCm2.toFixed(2)} cm²</span>
          ) : latestAvVti ? (
            <span className="auto-ef-primary">VTI VAo {latestAvVti.value.toFixed(1)} cm</span>
          ) : (
            <span className="auto-ef-primary">VTI TSVI {latestVti!.value.toFixed(1)} cm</span>
          )}
          <span className="auto-ef-metrics">
            {latestVti && (
              <span>
                TSVI {latestVti.value.toFixed(1)} cm
                {latestVti.isAveraged ? ` (${latestVti.beatCount}l)` : ''}
              </span>
            )}
            {dopplerSvMl !== null && <span>VS {dopplerSvMl.toFixed(0)} mL</span>}
            {latestAvVti && (
              <span>
                VAo {latestAvVti.value.toFixed(1)} cm
                {latestAvVti.isAveraged ? ` (${latestAvVti.beatCount}l)` : ''}
                {latestAvVti.latest.derived?.vmaxMps !== undefined
                  ? ` · ${latestAvVti.latest.derived.vmaxMps.toFixed(2)} m/s`
                  : ''}
                {latestAvVti.latest.derived?.meanGradientMmHg !== undefined
                  ? ` · ΔPm ${latestAvVti.latest.derived.meanGradientMmHg.toFixed(0)}`
                  : ''}
              </span>
            )}
            {velocityRatio !== null && <span>DVI {velocityRatio.toFixed(2)}</span>}
          </span>
        </div>
      )}
    </div>
  );

  return (
    <div className="auto-ef" data-auto-lvot="panel">
      {showAfHint && (
        <p className="small" role="note">
          FA activa: mide 3–5 latidos distintos; se promedian automáticamente (ASE).
        </p>
      )}
      {onOutflowView ? (
        <>
          {lvotCard}
          {viewTargetsCard}
        </>
      ) : (
        <>
          {viewTargetsCard}
          {lvotCard}
        </>
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
