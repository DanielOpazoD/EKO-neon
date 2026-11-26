import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';
import { useShallow } from 'zustand/shallow';
import { useHudStore, useSimStore } from './store';
import { frameOfSession } from './store/helpers';
import { modePolicy, screenInMode } from './modePolicy';
import { correctedImpressionScore } from './examSummary';
import { useSimulation } from './useSimulation';
import { useShortcuts } from './shortcuts';
import { DisplayCanvas } from '@/ui/DisplayCanvas';
import { ConsolePanel } from '@/ui/ConsolePanel';
import { TopBar } from '@/ui/TopBar';
import { ModeBar } from '@/ui/ModeBar';
import { ErrorBoundary, firstLine } from '@/ui/ErrorBoundary';
import { evaluateTasks, type LearnerSnapshot } from '@/education/curriculum';
import { IconChevronRight } from '@/ui/icons';
import { TooltipLayer } from '@/ui/Tooltip';
import { Splash } from '@/ui/Splash';
import { isReportSection, ReportSections, type ReportSectionId } from '@/ui/ReportSections';
import type { SimOutput } from '@/simulator/core/protocol';
import type { DopplerAudio } from '@/simulator/doppler/audio/dopplerAudio';
import { frameBus } from './frameBus';
import { isSpectralModality } from '@/simulator/renderer/modality';

// Loaded on demand (engineering audit, B4): the 3D navigator carries three.js (540 kB) and the secondary
// screens are not part of imaging. The entry chunk holds only what the first frame needs.
const TorsoView = lazy(() => import('@/ui/TorsoView').then((m) => ({ default: m.TorsoView })));
const ReferencesScreen = lazy(() =>
  import('@/ui/ReferencesScreen').then((m) => ({ default: m.ReferencesScreen })),
);
const ReportScreen = lazy(() =>
  import('@/ui/ReportScreen').then((m) => ({ default: m.ReportScreen })),
);
const CurriculumScreen = lazy(() =>
  import('@/ui/CurriculumScreen').then((m) => ({ default: m.CurriculumScreen })),
);
const ProgressScreen = lazy(() =>
  import('@/ui/ProgressScreen').then((m) => ({ default: m.ProgressScreen })),
);
const SegmentPanel = lazy(() =>
  import('@/ui/SegmentPanel').then((m) => ({ default: m.SegmentPanel })),
);
const GuidancePanel = lazy(() =>
  import('@/ui/GuidancePanel').then((m) => ({ default: m.GuidancePanel })),
);
const ExamNotice = lazy(() =>
  import('@/ui/GuidancePanel').then((m) => ({ default: m.ExamNotice })),
);
const DevPanel = lazy(() => import('@/ui/DevPanel').then((m) => ({ default: m.DevPanel })));
const ShortcutsDialog = lazy(() =>
  import('@/ui/ShortcutsDialog').then((m) => ({ default: m.ShortcutsDialog })),
);
const Tutorial = lazy(() => import('@/ui/Tutorial').then((m) => ({ default: m.Tutorial })));
const screenFallback = <div className="screen small">Cargando…</div>;
const SECTION_LABEL: Record<ReportSectionId, string> = {
  report: 'El informe',
  curriculum: 'El currículo',
  progress: 'El progreso',
};

export function App() {
  const [size, setSize] = useState({ width: 640, height: 520 });
  const [mobilePanel, setMobilePanel] = useState<'navigation' | 'console' | null>('console');
  const audioRef = useRef<DopplerAudio | null>(null);
  const setHud = useHudStore((s) => s.setHud);
  const lastHudRef = useRef(0);
  const ui = useSimStore(
    useShallow((s) => ({
      screen: s.ui.screen,
      showTorso: s.ui.showTorso,
      minimal: s.ui.minimal,
      railMini: s.ui.railMini,
      roomMode: s.ui.roomMode,
      segmentsOpen: s.ui.segmentsOpen,
      guidanceOpen: s.ui.guidanceOpen,
      devMode: s.ui.devMode,
      devPanel: s.ui.devPanel,
      tutorialDone: s.ui.tutorialDone,
      shortcutsOpen: s.ui.shortcutsOpen,
    })),
  );
  const mode = useSimStore((s) => s.mode);
  const error = useSimStore((s) => s.error);
  const spectral = useSimStore(
    useShallow((s) => ({
      audioOn: s.spectral.audioOn,
      volume: s.spectral.volume,
    })),
  );
  const modality = useSimStore((s) => s.modality);
  // a learning screen the mode closes is never drawn, whatever the store holds (setMode moves off it too)
  const screen = screenInMode(ui.screen, mode);
  const isSim = screen === 'simulator';
  const pageVisible = usePageVisible();
  useShortcuts();

  const onFrame = useCallback(
    (out: SimOutput) => {
      frameBus.latest = out;
      frameBus.emit(out);
      const now = performance.now();
      if (now - lastHudRef.current > 120) {
        lastHudRef.current = now;
        setHud(out);
        const st = useSimStore.getState();
        // only a frame acquired for this case in this session is scored, and a view is judged at the pose it was
        // analysed at (an analysis rides on the next frames, which may show another pose)
        const thisSession = frameOfSession(out.acquisition, st);
        const view = out.view;
        if (thisSession && !out.frozen) {
          st.observeView(view?.bestViewId ?? null, view?.score ?? 0, view?.probe);
          if (view?.bestViewId) st.recordViewScore(view.bestViewId, view.score, view.probe);
        }
        // curriculum: evaluate the automatic task checks against the learner's current state (≈ 8 Hz)
        if (thisSession && modePolicy(st.mode).evaluateCurriculum) {
          const snapshot: LearnerSnapshot = {
            caseId: st.caseId,
            mode: st.mode,
            viewProgress: st.viewProgress,
            handViewProgress: st.handViewProgress,
            bestView: out.view?.bestViewId
              ? { id: out.view.bestViewId, score: out.view.score }
              : null,
            modality: st.modality,
            colorScaleMps: st.color.scaleMps,
            colorGainDb: st.color.gainDb,
            gateStructure: out.gate?.structure ?? null,
            gateFlowAngleDeg: out.gate?.flowAngleDeg ?? null,
            measurements: st.measurements,
            impressionScore: correctedImpressionScore(st),
            settings: {
              depthCm: st.settings.depthCm,
              gainDb: st.settings.gainDb,
              frequencyMHz: st.settings.frequencyMHz,
              harmonics: st.settings.harmonics,
            },
          };
          // a task completes only after the ones it builds on (decision 260)
          const done = evaluateTasks(snapshot, st.progress.completedTasks);
          if (done.length) st.completeTasks(done);
        }
      }
      const audio = audioRef.current;
      if (audio && isSpectralModality(modality))
        audio.update(
          out.frozen || out.strip.kind !== 'spectral' ? null : out.spectrumColumn,
          out.spectralRange.vMin,
          out.spectralRange.vMax,
          (out.strip.frequencyMHz ?? 0) * 1e6,
          {
            aliasing: out.strip.aliasing,
            prfHz: out.strip.prfHz,
            wallFilterMps: out.strip.wallFilterMps,
          },
        );
    },
    [setHud, modality],
  );
  useSimulation(size, onFrame);

  useEffect(() => {
    // the Doppler sound is the scanner's: it stops with the scanner on another screen or in a hidden tab (the simulator
    // pauses there, and the last spectrum would hold as a steady tone)
    const wantAudio = spectral.audioOn && isSpectralModality(modality) && isSim && pageVisible;
    let cancelled = false;
    if (wantAudio && !audioRef.current) {
      import('@/simulator/doppler/audio/dopplerAudio').then(({ DopplerAudio }) => {
        if (cancelled) return;
        const st = useSimStore.getState();
        if (!st.spectral.audioOn || !isSpectralModality(st.modality) || audioRef.current) return;
        if (screenInMode(st.ui.screen, st.mode) !== 'simulator' || document.hidden) return;
        const instance = new DopplerAudio();
        instance.start();
        instance.setVolume(st.spectral.volume);
        audioRef.current = instance;
      });
      return () => {
        cancelled = true;
      };
    }
    if (audioRef.current) {
      if (!wantAudio) {
        audioRef.current.stop();
        audioRef.current = null;
      } else audioRef.current.setVolume(spectral.volume);
    }
    return undefined;
  }, [spectral.audioOn, spectral.volume, modality, isSim, pageVisible]);

  // Clean interface: the whole left rail hides. The exam keeps it, with a torso of skin and ribs and no guide
  // (decision 256): where the probe sits on the chest is the learner's own reference, not a hint.
  const railVisible = ui.showTorso && !ui.minimal;
  const hints = modePolicy(mode).hintsEnabled;
  return (
    <div
      data-mobile-panel={mobilePanel ?? 'none'}
      className={`app ${isSim ? '' : 'doc-screen '}${railVisible ? '' : 'no-torso'} ${railVisible && ui.railMini ? 'rail-mini' : ''}${ui.roomMode ? ' room' : ''}`}
    >
      <Splash />
      <TopBar />
      {/* one key per screen: the same boundary type in the same place would otherwise carry one screen's error
          into the next (decision 154) */}
      {screen === 'references' ? (
        <ErrorBoundary key="references" label="La pantalla de referencias">
          <Suspense fallback={screenFallback}>
            <ReferencesScreen />
          </Suspense>
        </ErrorBoundary>
      ) : isReportSection(screen) ? (
        // the case report, the curriculum and the progress: sections of one tab (decision 244)
        <ReportSections>
          <ErrorBoundary key={screen} label={SECTION_LABEL[screen]}>
            <Suspense fallback={screenFallback}>
              {screen === 'report' ? (
                <ReportScreen />
              ) : screen === 'curriculum' ? (
                <CurriculumScreen />
              ) : (
                <ProgressScreen />
              )}
            </Suspense>
          </ErrorBoundary>
        </ReportSections>
      ) : (
        <>
          <div className="mobile-panels" role="group" aria-label="Paneles de adquisición">
            <button
              aria-expanded={mobilePanel === 'navigation' && railVisible}
              aria-controls="navigation-panel"
              disabled={ui.minimal}
              onClick={() => {
                if (!railVisible) useSimStore.getState().setUi({ showTorso: true });
                setMobilePanel(mobilePanel === 'navigation' && railVisible ? null : 'navigation');
              }}
            >
              Navegación
            </button>
            <button
              aria-expanded={mobilePanel === 'console'}
              aria-controls="console-panel"
              onClick={() => setMobilePanel(mobilePanel === 'console' ? null : 'console')}
            >
              Consola
            </button>
          </div>
          <div
            id="navigation-panel"
            className="left"
            style={{ display: railVisible ? 'flex' : 'none' }}
          >
            <button
              className="rail-collapse"
              onClick={() => useSimStore.getState().setUi({ railMini: !ui.railMini })}
              aria-expanded={!ui.railMini}
              aria-label={
                ui.railMini ? 'Expandir panel de navegación' : 'Colapsar panel de navegación'
              }
              data-tip={
                ui.railMini
                  ? 'Expandir el torso 3D y la guía'
                  : 'Colapsar a una tira con la puntuación'
              }
            >
              <span
                aria-hidden="true"
                style={{
                  display: 'inline-flex',
                  transform: ui.railMini ? 'none' : 'rotate(180deg)',
                }}
              >
                <IconChevronRight size={14} />
              </span>
            </button>
            {/* the torso stays mounted while mini so its mesh worker is not rebuilt on expand */}
            <div className="rail-main" style={{ display: ui.railMini ? 'none' : 'flex' }}>
              {railVisible && (
                <ErrorBoundary label="El navegador 3D">
                  <Suspense fallback={<div className="torso small">Cargando torso 3D…</div>}>
                    <TorsoView />
                  </Suspense>
                </ErrorBoundary>
              )}
              {/* LV segments of the plane (decision 152): a learning aid, hidden like the guide until asked for */}
              {railVisible && hints && (
                <button
                  className="rail-head"
                  onClick={() => useSimStore.getState().setUi({ segmentsOpen: !ui.segmentsOpen })}
                  aria-expanded={ui.segmentsOpen}
                  aria-controls="lv-segments"
                  data-tip="Mapa polar de los segmentos del VI, qué muestra el corte y ficha de cada segmento"
                >
                  <span className="chev" aria-hidden="true">
                    <IconChevronRight size={13} />
                  </span>
                  Segmentos del VI
                </button>
              )}
              {railVisible && ui.segmentsOpen && (
                <div id="lv-segments">
                  <Suspense fallback={<div className="small">Cargando segmentos…</div>}>
                    <SegmentPanel />
                  </Suspense>
                </div>
              )}
              {/* the view guide stays hidden until asked for: the learner reads the image first (decision 141); it
                  sits at the rail's bottom edge, in a smaller header (decision 242) */}
              {railVisible && hints && (
                <button
                  className="rail-head guide-head"
                  onClick={() => useSimStore.getState().setUi({ guidanceOpen: !ui.guidanceOpen })}
                  aria-expanded={ui.guidanceOpen}
                  aria-controls="view-guidance"
                  data-tip="Puntuación de la vista, referencias que faltan y explicaciones"
                >
                  <span className="chev" aria-hidden="true">
                    <IconChevronRight size={13} />
                  </span>
                  Guía de la vista
                  <RailScore />
                </button>
              )}
              {railVisible && hints && ui.guidanceOpen && (
                <div id="view-guidance">
                  <Suspense fallback={<div className="guidance small">Cargando guía…</div>}>
                    <GuidancePanel />
                  </Suspense>
                </div>
              )}
            </div>
            {ui.railMini && <RailMini />}
          </div>
          <div className="center">
            {/* the exam notice sits over the image, where it is never silently lost — the E2E flow caught it hidden
                in the rail after commit 93d30b0; the rail keeps no guide in the exam (decision 256) */}
            {mode === 'exam' && (
              <Suspense fallback={null}>
                <ExamNotice />
              </Suspense>
            )}
            {error && (
              <div className="error" role="alert">
                {firstLine(error)}
                <div>
                  <button onClick={() => useSimStore.getState().setError(null)}>cerrar</button>
                </div>
              </div>
            )}
            <DisplayCanvas onSize={setSize} />
            {modePolicy(mode).devToolsAllowed && ui.devMode && ui.devPanel && (
              <Suspense fallback={null}>
                <DevPanel />
              </Suspense>
            )}
            {isSim && !ui.tutorialDone && (
              <Suspense fallback={null}>
                <Tutorial />
              </Suspense>
            )}
          </div>
          <div id="console-panel" className="right">
            <ConsolePanel />
          </div>
        </>
      )}
      {isSim && <ModeBar />}
      <TooltipLayer />
      {ui.shortcutsOpen && (
        <Suspense fallback={null}>
          <ShortcutsDialog />
        </Suspense>
      )}
    </div>
  );
}

const onVisibility = (cb: () => void) => {
  document.addEventListener('visibilitychange', cb);
  return () => document.removeEventListener('visibilitychange', cb);
};

/** Whether the tab is visible (the Page Visibility API). */
function usePageVisible(): boolean {
  return useSyncExternalStore(onVisibility, () => !document.hidden);
}

/** The view score beside the guide's header, so it is glanceable while the guide is closed. */
function RailScore() {
  const score = useHudStore((h) => h.hud?.view?.score ?? null);
  const mode = useSimStore((s) => s.mode);
  if (!modePolicy(mode).hintsEnabled || score === null) return null;
  const color = score >= 75 ? 'var(--ok)' : score >= 50 ? 'var(--warn)' : 'var(--bad)';
  return (
    <span className="rail-head-score" style={{ color }} aria-hidden="true">
      {score}
    </span>
  );
}

/** Collapsed left rail: the view score stays glanceable while torso and guidance are tucked away. */
function RailMini() {
  const { score, bestViewName } = useHudStore(
    useShallow((h) => ({
      score: h.hud?.view?.score ?? null,
      bestViewName: h.hud?.view?.bestViewName ?? '',
    })),
  );
  const mode = useSimStore((s) => s.mode);
  if (!modePolicy(mode).hintsEnabled || score === null) return <div className="rail-strip" />;
  const color = score >= 75 ? 'var(--ok)' : score >= 50 ? 'var(--warn)' : 'var(--bad)';
  return (
    <div className="rail-strip">
      <div
        className="rail-strip-score"
        style={{ color }}
        data-tip={`${score}/100 — ${bestViewName}`}
      >
        {score}
      </div>
      <div className="rail-strip-bar">
        <i style={{ height: `${score}%`, background: color }} />
      </div>
    </div>
  );
}
