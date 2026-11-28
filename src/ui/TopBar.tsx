import { useSimStore } from '@/app/store';
import { useShallow } from 'zustand/shallow';
import { modePolicy } from '@/app/modePolicy';
import { useRef } from 'react';
import { useSlidingPill } from './useSlidingPill';
import { isReportSection } from './ReportSections';
import { ANATOMY_LABELS } from './anatomyLabels';
import { IconPencil } from './icons';
import { ANATOMY_CHOICES } from '@/simulator/core/protocol';

/**
 * Minimal top bar: brand on the left, the 3 screens in the center, and product mode on the right.
 * Quality lives in the ⋯ menu so the header stays uncluttered; freeze state lives in ModeBar.
 */
const SCREENS: { id: 'simulator' | 'report' | 'references'; label: string }[] = [
  { id: 'simulator', label: 'Simulador' },
  { id: 'report', label: 'Informe' },
  { id: 'references', label: 'Referencias' },
];

export function TopBar() {
  const navRef = useRef<HTMLElement>(null);
  const s = useSimStore(
    useShallow((st) => ({
      anatomy: st.anatomy,
      mode: st.mode,
      setAnatomy: st.setAnatomy,
      setMode: st.setMode,
      setUi: st.setUi,
      screen: st.ui.screen,
      devMode: st.ui.devMode,
      reviewMode: st.ui.reviewMode,
    })),
  );
  const current = isReportSection(s.screen) ? 'report' : s.screen;
  const isSim = current === 'simulator';
  const pill = useSlidingPill(navRef, current);
  const policy = modePolicy(s.mode);
  return (
    <div className="topbar" role="banner">
      <span className="brand">
        <i className="brand-mark" aria-hidden="true" />
        EchoTwin <span className="brand-sub">TTE</span>
      </span>
      <nav className="nav-seg" aria-label="Pantallas" ref={navRef}>
        {pill && <span className="seg-pill" style={pill} aria-hidden="true" />}
        {SCREENS.map((sc) => (
          <button
            key={sc.id}
            className={current === sc.id ? 'active' : ''}
            aria-current={current === sc.id ? 'page' : undefined}
            onClick={() => {
              if (current !== sc.id) s.setUi({ screen: sc.id });
            }}
            disabled={sc.id === 'references' && !policy.learningScreensEnabled}
          >
            {sc.label}
          </button>
        ))}
      </nav>
      {isSim && policy.devToolsAllowed && s.devMode && (
        <div className="tool-seg" role="group" aria-label="Corazón">
          <span className="tool-seg-cap" aria-hidden="true">
            Corazón
          </span>
          {ANATOMY_CHOICES.map((a) => (
            <button
              key={a}
              className={s.anatomy === a ? 'active' : ''}
              aria-pressed={s.anatomy === a}
              onClick={() => s.setAnatomy(a)}
              data-tip={ANATOMY_LABELS[a].tip}
            >
              {ANATOMY_LABELS[a].short}
            </button>
          ))}
        </div>
      )}
      {isSim && policy.devToolsAllowed && s.devMode && (
        <button
          className={`notes-btn${s.reviewMode ? ' active' : ''}`}
          aria-pressed={s.reviewMode}
          onClick={() =>
            s.setUi({
              reviewMode: !s.reviewMode,
              consoleTab: s.reviewMode ? 'adquirir' : 'revisar',
            })
          }
          data-tip="Modo revisión: marca y anota sobre la imagen lo que ves mal, y copia el informe"
          data-tip-key="R"
        >
          <IconPencil size={15} />
          Notas
        </button>
      )}
      <select
        aria-label="Modo del producto"
        value={s.mode}
        onChange={(e) => s.setMode(e.target.value as typeof s.mode)}
        data-tip="Libre: todo abierto. Guiado: vista objetivo con ayudas. Examen: sin ayudas ni presets"
      >
        <option value="sandbox">Libre</option>
        <option value="guided">Guiado</option>
        <option value="exam">Examen</option>
      </select>
    </div>
  );
}
