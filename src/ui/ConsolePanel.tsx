import { modePolicy } from '@/app/modePolicy';
import { useSimStore, type ConsoleTab } from '@/app/store';
import { MODALITIES } from '@/simulator/renderer/modality';
import { type KeyboardEvent, type ReactNode, lazy, Suspense, useEffect, useRef } from 'react';
import { useShallow } from 'zustand/shallow';
import { AcquireTab } from './console/AcquireTab';
import { DopplerTab } from './console/DopplerTab';
import { ImageTab } from './console/ImageTab';
import { MeasureTab } from './console/MeasureTab';
import { IconAcquire, IconDoppler, IconImage, IconLab, IconMeasure, IconReview } from './icons';
import { useSlidingPill } from './useSlidingPill';

const LabTab = lazy(() => import('./console/LabTab').then((m) => ({ default: m.LabTab })));
const ReviewPanel = lazy(() =>
  import('./ReviewPanel').then((m) => ({ default: m.ReviewPanel })),
);

/**
 * Right-hand console, organised as tabs so only one control family is visible at a time
 * (spec 29.2). The console follows intent: entering a Doppler modality switches to the
 * Doppler tab; every control still alters the render.
 */
export function ConsolePanel() {
  const s = useSimStore(
    useShallow((st) => ({
      measurements: st.measurements,
      modality: st.modality,
      mode: st.mode,
      reviewCount: st.reviewMarkers.length,
      setUi: st.setUi,
      ui: st.ui,
    })),
  );
  const mod = s.modality;
  const policy = modePolicy(s.mode);

  // Follow the modality: enabling a Doppler mode surfaces its controls; returning to 2D
  // leaves the Doppler tab so the user is never left on an empty panel.
  useEffect(() => {
    const st = useSimStore.getState();
    if (mod !== '2d') st.setUi({ consoleTab: 'doppler' });
    else if (st.ui.consoleTab === 'doppler') st.setUi({ consoleTab: 'imagen' });
  }, [mod]);

  const tabs: {
    id: ConsoleTab;
    label: string;
    icon: ReactNode;
    show: boolean;
    badge?: number;
    dot?: boolean;
  }[] = [
    { id: 'adquirir', label: 'Adquirir', icon: <IconAcquire />, show: true },
    { id: 'imagen', label: 'Imagen', icon: <IconImage />, show: true },
    {
      id: 'doppler',
      label: 'Doppler',
      icon: <IconDoppler />,
      show: true,
      dot: MODALITIES[mod].doppler,
    },
    {
      id: 'medir',
      label: 'Medir',
      icon: <IconMeasure />,
      show: true,
      badge: s.measurements.length || undefined,
    },
    { id: 'lab', label: 'Lab', icon: <IconLab />, show: policy.devToolsAllowed },
    {
      id: 'revisar',
      label: 'Revisar',
      icon: <IconReview />,
      show: policy.devToolsAllowed && s.ui.reviewMode,
      badge: s.reviewCount || undefined,
    },
  ];
  const visible = tabs.filter((t) => t.show);
  const active = visible.some((t) => t.id === s.ui.consoleTab) ? s.ui.consoleTab : 'adquirir';
  const tabsRef = useRef<HTMLDivElement>(null);
  // the set of tabs is part of the key: a tab that appears (review) or goes (exam) moves the others
  const ink = useSlidingPill(tabsRef, `${active}|${visible.map((t) => t.id).join(',')}`, '.on');
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
        ref={tabsRef}
      >
        {ink && <span className="tab-ink" style={ink} aria-hidden="true" />}
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
          {active === 'imagen' && <ImageTab />}
          {active === 'doppler' && <DopplerTab />}
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
