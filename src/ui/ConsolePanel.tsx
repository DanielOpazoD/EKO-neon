import { modePolicy } from '@/app/modePolicy';
import { useSimStore, type ConsoleTab } from '@/app/store';
import { type KeyboardEvent, type ReactNode, lazy, Suspense, useEffect, useRef } from 'react';
import { useShallow } from 'zustand/shallow';
import { AcquireTab } from './console/AcquireTab';
import { DopplerTab } from './console/DopplerTab';
import { ImageTab } from './console/ImageTab';
import { MeasureTab } from './console/MeasureTab';
import { IconAcquire, IconImage, IconLab, IconMeasure, IconReview } from './icons';

const LabTab = lazy(() => import('./console/LabTab').then((m) => ({ default: m.LabTab })));
const ReviewPanel = lazy(() =>
  import('./ReviewPanel').then((m) => ({ default: m.ReviewPanel })),
);

/**
 * Right-hand console streamlined into 3 focused tabs:
 * - «Vistas» (`adquirir`): canonical windows, patient posture/respiration, and optional probe fine-tuning.
 * - «Ajustes» (`imagen`): contextual 2D or active Doppler/M-mode controls (replaces the old empty Doppler tab).
 * - «Medir» (`medir`): 1-click guided auto-measurements, captures worksheet, and manual tools.
 */
export function ConsolePanel() {
  const s = useSimStore(
    useShallow((st) => ({
      measurements: st.measurements,
      modality: st.modality,
      mode: st.mode,
      reviewCount: st.reviewMarkers.length,
      setUi: st.setUi,
      consoleTab: st.ui.consoleTab,
      devMode: st.ui.devMode,
      reviewMode: st.ui.reviewMode,
    })),
  );
  const mod = s.modality;
  const policy = modePolicy(s.mode);

  // Follow intent: activating a Doppler/M-mode modality surfaces its contextual knobs in «Ajustes»
  // unless the operator is actively working in «Medir» or «Revisar».
  useEffect(() => {
    const st = useSimStore.getState();
    if (st.ui.consoleTab === 'doppler') {
      st.setUi({ consoleTab: 'imagen' });
      return;
    }
    if (mod !== '2d' && st.ui.consoleTab !== 'medir' && st.ui.consoleTab !== 'revisar') {
      st.setUi({ consoleTab: 'imagen' });
    }
  }, [mod]);

  const tabs: {
    id: ConsoleTab;
    label: string;
    icon: ReactNode;
    show: boolean;
    badge?: number;
    dot?: boolean;
  }[] = [
    { id: 'adquirir', label: 'Vistas', icon: <IconAcquire />, show: true },
    {
      id: 'imagen',
      label: 'Ajustes',
      icon: <IconImage />,
      show: true,
      dot: mod !== '2d',
    },
    {
      id: 'medir',
      label: 'Medir',
      icon: <IconMeasure />,
      show: true,
      badge: s.measurements.length || undefined,
    },
    { id: 'lab', label: 'Lab', icon: <IconLab />, show: policy.devToolsAllowed && s.devMode },
    {
      id: 'revisar',
      label: 'Revisar',
      icon: <IconReview />,
      show: policy.devToolsAllowed && s.devMode && s.reviewMode,
      badge: s.reviewCount || undefined,
    },
  ];
  const visible = tabs.filter((t) => t.show);
  // Migrate any legacy persisted 'doppler' tab transparently into 'imagen' (Ajustes)
  const normalizedTab: ConsoleTab = s.consoleTab === 'doppler' ? 'imagen' : s.consoleTab;
  const active = visible.some((t) => t.id === normalizedTab) ? normalizedTab : 'adquirir';
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const onTabsKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = visible.findIndex((t) => t.id === active);
    let n = -1;
    if (e.key === 'ArrowRight') n = (i + 1) % visible.length;
    else if (e.key === 'ArrowLeft') n = (i - 1 + visible.length) % visible.length;
    else if (e.key === 'Home') n = 0;
    else if (e.key === 'End') n = visible.length - 1;
    if (n < 0) return;
    e.preventDefault();
    s.setUi({ consoleTab: visible[n]!.id });
    tabRefs.current[n]?.focus();
  };

  return (
    <div className="console">
      <div
        className="console-tabs"
        role="tablist"
        aria-label="Familia de controles"
        onKeyDown={onTabsKeyDown}
      >
        {visible.map((t, i) => (
          <button
            key={t.id}
            ref={(el) => {
              tabRefs.current[i] = el;
            }}
            role="tab"
            id={`console-tab-${t.id}`}
            aria-selected={t.id === active}
            aria-controls={`console-panel-${t.id}`}
            tabIndex={t.id === active ? 0 : -1}
            className={t.id === active ? 'on' : ''}
            onClick={() => s.setUi({ consoleTab: t.id })}
          >
            {t.icon}
            <span>{t.label}</span>
            {t.badge ? (
              <span className="tab-badge" aria-hidden="true">
                {t.badge}
              </span>
            ) : null}
            {t.dot ? <span className="tab-dot" aria-hidden="true" /> : null}
          </button>
        ))}
      </div>
      <div className="console-scroll">
        <div
          key={active}
          role="tabpanel"
          id={`console-panel-${active}`}
          aria-labelledby={`console-tab-${active}`}
        >
          {active === 'adquirir' && <AcquireTab />}
          {active === 'imagen' && (mod === '2d' ? <ImageTab /> : <DopplerTab />)}
          {active === 'medir' && <MeasureTab />}
          {active === 'lab' && (
            <Suspense fallback={<div className="small">Cargando laboratorio…</div>}>
              <LabTab />
            </Suspense>
          )}
          {active === 'revisar' && (
            <Suspense fallback={<div className="small">Cargando revisión…</div>}>
              <ReviewPanel />
            </Suspense>
          )}
        </div>
      </div>
    </div>
  );
}
