import { useState } from 'react';
import { useSimStore } from '@/app/store';
import { useShallow } from 'zustand/shallow';
import { exportCaseTag, modePolicy } from '@/app/modePolicy';
import { exportDisplayPng, recordDisplayClip } from '@/app/exportImage';
import { ANATOMY_LABELS } from './anatomyLabels';
import { CaseQuickMenu } from './CaseQuickMenu';
import { Segmented } from './controls';
import { IconMenu, IconPencil } from './icons';
import { ActionItem, CheckItem, MenuCap, usePopover } from './menu';
import { ANATOMY_CHOICES } from '@/simulator/core/protocol';

/**
 * Minimal top bar:
 * - Left: brand (clickable to return to simulator) + «← Simulador» on secondary screens.
 * - Center: 1-click clinical case picker (anonymous & locked in exam mode).
 * - Right: product mode + ONE unified «☰» menu holding all secondary screens, quality, display & export options.
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
      <div className="topbar-center">
        <CaseQuickMenu surface="light" />
      </div>
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
      <TopMenu studyItems={s.measurementsCount + s.capturesCount} />
    </div>
  );
}

/** Single unified secondary menu: Informe, Referencias, Currículo, Calidad, Pantalla, Exportar y Desarrollo. */
function TopMenu({ studyItems }: { studyItems: number }) {
  const { open, setOpen, wrap } = usePopover();
  const [recording, setRecording] = useState(false);
  const screen = useSimStore((st) => st.ui.screen);
  const activeOutside = screen !== 'simulator';

  return (
    <div className="menu-wrap" ref={wrap}>
      <button
        type="button"
        className={`menu-btn${open || activeOutside ? ' active' : ''}`}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Menú"
        data-tip="Informe, referencias, calidad y opciones"
        onClick={() => setOpen(!open)}
      >
        <IconMenu size={16} />
      </button>
      {open && (
        <TopMenuContent
          studyItems={studyItems}
          recording={recording}
          setRecording={setRecording}
          close={() => setOpen(false)}
        />
      )}
    </div>
  );
}

function TopMenuContent({
  studyItems,
  recording,
  setRecording,
  close,
}: {
  studyItems: number;
  recording: boolean;
  setRecording: (v: boolean) => void;
  close: () => void;
}) {
  const s = useSimStore(
    useShallow((st) => ({
      caseId: st.caseId,
      examFinished: st.examFinished,
      mode: st.mode,
      quality: st.quality,
      setQuality: st.setQuality,
      setUi: st.setUi,
      ui: st.ui,
    })),
  );
  const policy = modePolicy(s.mode);
  const devOn = policy.devToolsAllowed && s.ui.devMode;
  const activeOutside = s.ui.screen !== 'simulator';
  const go = (target: 'simulator' | 'report' | 'curriculum' | 'progress' | 'references') => {
    s.setUi({ screen: target });
    close();
  };

  return (
    <div className="menu down top-unified-menu" role="menu" aria-label="Menú principal">
      <MenuCap>Estudio</MenuCap>
      {activeOutside && <ActionItem label="← Volver al simulador" onClick={() => go('simulator')} />}
      <ActionItem
        label={studyItems > 0 ? `Informe (${studyItems})` : 'Informe'}
        onClick={() => go('report')}
      />
      {policy.learningScreensEnabled && (
        <>
          <ActionItem label="Currículo" onClick={() => go('curriculum')} />
          <ActionItem label="Progreso" onClick={() => go('progress')} />
          <ActionItem label="Referencias clínicas" onClick={() => go('references')} />
        </>
      )}

      <div className="menu-sep" />
      <MenuCap>Calidad de imagen</MenuCap>
      <div className="menu-quality">
        <Segmented
          ariaLabel="Calidad"
          value={s.quality}
          onChange={(v) => s.setQuality(v)}
          options={[
            { id: 'auto', label: 'Auto' },
            { id: 'low', label: 'Baja' },
            { id: 'medium', label: 'Media' },
            { id: 'high', label: 'Alta' },
          ]}
        />
      </div>

      <div className="menu-sep" />
      <MenuCap>Mostrar</MenuCap>
      <CheckItem
        label="Ayudas de vista"
        checked={s.ui.showHints}
        onToggle={() => s.setUi({ showHints: !s.ui.showHints })}
        disabled={!policy.hintsEnabled}
      />
      <CheckItem
        label="Datos sobre imagen"
        checked={s.ui.showHud}
        onToggle={() => s.setUi({ showHud: !s.ui.showHud })}
      />
      <CheckItem
        label="Modo sala oscura"
        checked={s.ui.roomMode}
        onToggle={() => s.setUi({ roomMode: !s.ui.roomMode })}
      />
      <CheckItem
        label="ECG"
        checked={s.ui.showEcg}
        onToggle={() => s.setUi({ showEcg: !s.ui.showEcg })}
      />

      <div className="menu-sep" />
      <MenuCap>Exportar y ayuda</MenuCap>
      <ActionItem
        label="Guardar imagen PNG"
        onClick={() => {
          exportDisplayPng(exportCaseTag(s.mode, s.examFinished, s.caseId));
          close();
        }}
      />
      <ActionItem
        label={recording ? 'Grabando clip WebM…' : 'Exportar clip WebM (3 s)'}
        onClick={() => {
          if (recording) return;
          setRecording(true);
          close();
          void recordDisplayClip(exportCaseTag(s.mode, s.examFinished, s.caseId), 3000).finally(
            () => setRecording(false),
          );
        }}
      />
      <ActionItem
        label="Atajos de teclado"
        onClick={() => {
          s.setUi({ shortcutsOpen: true });
          close();
        }}
      />
      <ActionItem
        label="Tutorial de controles"
        onClick={() => {
          s.setUi({ tutorialDone: false });
          close();
        }}
      />

      {policy.devToolsAllowed && (
        <>
          <div className="menu-sep" />
          <MenuCap>Desarrollo</MenuCap>
          <CheckItem
            label="Modo desarrollador"
            checked={s.ui.devMode}
            onToggle={() => s.setUi({ devMode: !s.ui.devMode })}
          />
          {devOn && (
            <>
              <CheckItem
                label="Modo revisión"
                checked={s.ui.reviewMode}
                onToggle={() =>
                  s.setUi({
                    reviewMode: !s.ui.reviewMode,
                    consoleTab: s.ui.reviewMode ? 'adquirir' : 'revisar',
                  })
                }
              />
              <CheckItem
                label="Superposición física"
                checked={s.ui.showPhysics}
                onToggle={() => s.setUi({ showPhysics: !s.ui.showPhysics })}
              />
              <CheckItem
                label="Panel Dev"
                checked={s.ui.devPanel}
                onToggle={() => s.setUi({ devPanel: !s.ui.devPanel })}
              />
            </>
          )}
        </>
      )}
    </div>
  );
}
