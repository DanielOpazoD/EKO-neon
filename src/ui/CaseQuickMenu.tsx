import { modePolicy, showsCaseIdentity } from '@/app/modePolicy';
import { useSimStore } from '@/app/store';
import { listCases } from '@/cases';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useShallow } from 'zustand/shallow';

/**
 * Compact 1-click clinical case switcher anchored on the top-left HUD case title so an instructor
 * can switch pathologies live during a class without leaving Medir or Doppler.
 * Strictly non-interactive in exam mode (`mode === 'exam'` or `!showsCaseIdentity`) so exam cases
 * stay locked and anonymous.
 */
export function CaseQuickMenu(props: { children: React.ReactNode; titleAttr?: string }) {
  const s = useSimStore(
    useShallow((st) => ({
      mode: st.mode,
      caseId: st.caseId,
      examFinished: st.examFinished,
      loadCase: st.loadCase,
    })),
  );
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const cases = useMemo(() => listCases(), []);
  const allowed =
    s.mode !== 'exam' && !modePolicy(s.mode).assignsCase && showsCaseIdentity(s.mode, s.examFinished);

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

  return (
    <div className="view-picker dark case-picker" ref={wrapRef}>
      <button
        type="button"
        className={`view-picker-trigger case-picker-trigger${open ? ' open' : ''}`}
        aria-haspopup="menu"
        aria-expanded={open}
        title={props.titleAttr}
        data-tip="Cambiar de caso clínico en 1 clic sin salir de esta pestaña"
        onClick={() => setOpen((v) => !v)}
      >
        {props.children}
        <span className="view-picker-caret" aria-hidden="true">
          ▾
        </span>
      </button>
      {open && (
        <div
          className="view-picker-pop case-picker-pop"
          role="menu"
          aria-label="Cambiar caso clínico"
        >
          <div className="view-picker-cap">Cambiar caso clínico ({cases.length})</div>
          <div className="case-picker-list">
            {cases.map((c) => {
              const active = c.id === s.caseId;
              return (
                <button
                  key={c.id}
                  type="button"
                  role="menuitemradio"
                  aria-checked={active}
                  className={active ? 'active' : ''}
                  onClick={() => {
                    if (!active) s.loadCase(c.id);
                    setOpen(false);
                  }}
                >
                  <span className="case-picker-name">{c.title}</span>
                  <span className="case-picker-diff" aria-label={`Dificultad ${c.difficulty} de 5`}>
                    {'●'.repeat(Math.max(1, Math.min(5, c.difficulty)))}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
