import { examClosed, modePolicy } from '@/app/modePolicy';
import { useSimStore } from '@/app/store';
import { useEffect, useRef, useState } from 'react';
import { useShallow } from 'zustand/shallow';

export interface QuickViewOption {
  id: string;
  label: string;
  sub: string;
  title: string;
}

export const QUICK_VIEW_OPTIONS: readonly QuickViewOption[] = [
  { id: 'plax', label: 'PLAX', sub: 'eje largo', title: 'Paraesternal eje largo' },
  { id: 'psax-av', label: 'PSAX', sub: 'aórtica', title: 'Paraesternal eje corto (válvula aórtica)' },
  { id: 'psax-mv', label: 'PSAX', sub: 'mitral', title: 'Paraesternal eje corto (nivel mitral)' },
  { id: 'psax-pm', label: 'PSAX', sub: 'papilar', title: 'Paraesternal eje corto (músculos papilares)' },
  { id: 'a4c', label: 'A4C', sub: '4 cám.', title: 'Apical cuatro cámaras' },
  { id: 'a2c', label: 'A2C', sub: '2 cám.', title: 'Apical dos cámaras' },
  { id: 'a5c', label: 'A5C', sub: '5 cám.', title: 'Apical cinco cámaras' },
  { id: 'a3c', label: 'A3C', sub: 'eje largo', title: 'Apical tres cámaras' },
  { id: 'rv-focused', label: 'VD', sub: 'apical', title: 'Apical enfocada en VD' },
  { id: 'subcostal-4c', label: 'SC 4C', sub: 'subcostal', title: 'Subcostal cuatro cámaras' },
  { id: 'subcostal-ivc', label: 'SC VCI', sub: 'cava inf.', title: 'Subcostal vena cava inferior' },
  { id: 'suprasternal-arch', label: 'SSN', sub: 'arco Ao', title: 'Supraesternal arco aórtico' },
];

/**
 * Compact 1-click view switcher popover anchored to either the on-image HUD view chip or the
 * MeasureTab context strip so the operator switches windows without leaving Medir or Doppler.
 * Strictly disabled/non-interactive in exam mode (`!presetsEnabled`).
 */
export function ViewQuickMenu(props: {
  /** Trigger contents (e.g. "Vista PLAX 95" or "PLAX"). */
  children: React.ReactNode;
  /** Visual placement of the popover relative to the trigger. */
  placement: 'up-right' | 'down-left';
  /** Dark-glass styling when mounted on top of the black ultrasound display. */
  dark?: boolean;
  /** Currently recognized view id for highlighting the active tile. */
  currentViewId: string | null;
}) {
  const s = useSimStore(
    useShallow((st) => ({
      mode: st.mode,
      examFinished: st.examFinished,
      targetViewId: st.targetViewId,
      startPresetView: st.startPresetView,
    })),
  );
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const allowed = modePolicy(s.mode).presetsEnabled && !examClosed(s);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [open]);

  if (!allowed) {
    return <span className="view-picker-static">{props.children}</span>;
  }

  const activeId = props.currentViewId ?? s.targetViewId;

  return (
    <div className={`view-picker${props.dark ? ' dark' : ''}`} ref={wrapRef}>
      <button
        type="button"
        className={`view-picker-trigger${open ? ' open' : ''}`}
        aria-haspopup="menu"
        aria-expanded={open}
        data-tip="Cambiar vista rápidamente sin salir de esta pestaña"
        onClick={() => setOpen((v) => !v)}
      >
        {props.children}
        <span className="view-picker-caret" aria-hidden="true">
          ▾
        </span>
      </button>
      {open && (
        <div
          className={`view-picker-pop ${props.placement}`}
          role="menu"
          aria-label="Cambiar vista ecográfica"
        >
          <div className="view-picker-cap">Cambiar vista</div>
          <div className="view-picker-grid">
            {QUICK_VIEW_OPTIONS.map((v) => (
              <button
                key={v.id}
                type="button"
                role="menuitem"
                className={activeId === v.id ? 'active' : ''}
                title={v.title}
                onClick={() => {
                  s.startPresetView(v.id);
                  setOpen(false);
                }}
              >
                <span>{v.label}</span>
                <small>{v.sub}</small>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
