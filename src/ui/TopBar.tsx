import { useSimStore } from '@/app/store';
import { useShallow } from 'zustand/shallow';
import { modePolicy } from '@/app/modePolicy';
import { ANATOMY_LABELS } from './anatomyLabels';
import { IconMenu, IconPencil } from './icons';
import { ActionItem, MenuCap, usePopover } from './menu';
import { ANATOMY_CHOICES } from '@/simulator/core/protocol';

/**
 * Ultra-clean top bar: brand on the left (plus «← Simulador» when viewing a secondary screen),
 * and product mode + secondary menu icon (Informe, Currículo, Progreso, Referencias) on the right.
 */
export function TopBar() {
  const s = useSimStore(
    useShallow((st) => ({
      anatomy: st.anatomy,
      mode: st.mode,
      measurementsCount: st.measurements.length,
      capturesCount: st.captures.length,
      setAnatomy: st.setAnatomy,
      setMode: st.setMode,
      setUi: st.setUi,
      screen: st.ui.screen,
      devMode: st.ui.devMode,
      reviewMode: st.ui.reviewMode,
    })),
  );
  const isSim = s.screen === 'simulator';
  const policy = modePolicy(s.mode);

  return (
    <div className="topbar" role="banner">
      <button
        type="button"
        className="brand brand-btn"
        onClick={() => {
          if (!isSim) s.setUi({ screen: 'simulator' });
        }}
        data-tip={isSim ? undefined : 'Volver al simulador'}
      >
        <i className="brand-mark" aria-hidden="true" />
        EchoTwin <span className="brand-sub">TTE</span>
      </button>
      {!isSim && (
        <button
          type="button"
          className="ghost topbar-back"
          onClick={() => s.setUi({ screen: 'simulator' })}
        >
          ← Simulador
        </button>
      )}
      <span className="spacer" style={{ flex: 1 }} />
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
          data-tip="Modo revisión: marca y anota sobre la imagen"
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
      <TopMenu
        screen={s.screen}
        learningEnabled={policy.learningScreensEnabled}
        studyItems={s.measurementsCount + s.capturesCount}
        onNavigate={(screen) => s.setUi({ screen })}
        onOpenShortcuts={() => s.setUi({ shortcutsOpen: true })}
      />
    </div>
  );
}

function TopMenu({
  screen,
  learningEnabled,
  studyItems,
  onNavigate,
  onOpenShortcuts,
}: {
  screen: 'simulator' | 'report' | 'curriculum' | 'progress' | 'references';
  learningEnabled: boolean;
  studyItems: number;
  onNavigate: (screen: 'simulator' | 'report' | 'curriculum' | 'progress' | 'references') => void;
  onOpenShortcuts: () => void;
}) {
  const { open, setOpen, wrap } = usePopover();
  const activeOutside = screen !== 'simulator';
  const go = (target: 'simulator' | 'report' | 'curriculum' | 'progress' | 'references') => {
    onNavigate(target);
    setOpen(false);
  };

  return (
    <div className="menu-wrap" ref={wrap}>
      <button
        type="button"
        className={`menu-btn${open || activeOutside ? ' active' : ''}`}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Menú principal"
        data-tip="Informe, referencias y ayuda"
        onClick={() => setOpen(!open)}
      >
        <IconMenu size={16} />
      </button>
      {open && (
        <div className="menu down" role="menu" aria-label="Pantallas secundarias">
          <MenuCap>Estudio</MenuCap>
          {activeOutside && <ActionItem label="← Volver al simulador" onClick={() => go('simulator')} />}
          <ActionItem
            label={studyItems > 0 ? `Informe (${studyItems})` : 'Informe'}
            onClick={() => go('report')}
          />
          {learningEnabled && (
            <>
              <ActionItem label="Currículo" onClick={() => go('curriculum')} />
              <ActionItem label="Progreso" onClick={() => go('progress')} />
            </>
          )}
          <div className="menu-sep" />
          <MenuCap>Consulta</MenuCap>
          {learningEnabled && (
            <ActionItem label="Referencias clínicas" onClick={() => go('references')} />
          )}
          <ActionItem
            label="Atajos de teclado"
            onClick={() => {
              onOpenShortcuts();
              setOpen(false);
            }}
          />
        </div>
      )}
    </div>
  );
}
