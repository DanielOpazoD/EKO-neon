import { EXAM_CASE_NAME, modePolicy, showsCaseIdentity } from '@/app/modePolicy';
import { useSimStore } from '@/app/store';
import { listCases } from '@/cases';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useShallow } from 'zustand/shallow';

/**
 * Clean 1-click clinical case switcher for TopBar (light surface) or dark overlays.
 * Strictly non-interactive and anonymous (`EXAM_CASE_NAME`) in exam mode.
 */
export function CaseQuickMenu(props: {
  children?: React.ReactNode;
  titleAttr?: string;
  surface?: 'dark' | 'light';
}) {
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
  const identity = showsCaseIdentity(s.mode, s.examFinished);
  const allowed = s.mode !== 'exam' && !modePolicy(s.mode).assignsCase && identity;
  const activeTitle = identity
    ? (cases.find((c) => c.id === s.caseId)?.title ?? s.caseId)
    : EXAM_CASE_NAME;
  const surface = props.surface ?? 'light';

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
    return (
      <span className={`view-picker-static${surface === 'light' ? ' topbar-case-static' : ''}`}>
        {props.children ?? activeTitle}
      </span>
    );
  }

  return (
    <div
      className={`view-picker case-picker ${surface === 'dark' ? 'dark' : 'light'}`}
      ref={wrapRef}
    >
      <button
        type="button"
        className={`view-picker-trigger case-picker-trigger${surface === 'light' ? ' topbar-case-btn' : ''}${open ? ' open' : ''}`}
        aria-haspopup="menu"
        aria-expanded={open}
        title={props.titleAttr ?? activeTitle}
        data-tip="Cambiar de caso clínico"
        onClick={() => setOpen((v) => !v)}
      >
        {props.children ?? <span className="topbar-case-label">{activeTitle}</span>}
        <span className="view-picker-caret" aria-hidden="true">
          ▾
        </span>
      </button>
      {open && (
        <div
          className={`view-picker-pop case-picker-pop${surface === 'light' ? ' light' : ''}`}
          role="menu"
          aria-label="Cambiar caso clínico"
        >
          <div className="view-picker-cap">Casos clínicos ({cases.length})</div>
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
