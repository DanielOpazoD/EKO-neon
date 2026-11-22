import { useEffect, useRef, useState } from 'react';
import { useSlidingPill } from './useSlidingPill';
import { useHudStore, useSimStore } from '@/app/store';
import { useShallow } from 'zustand/shallow';
import { exportCaseTag, modePolicy } from '@/app/modePolicy';
import { exportDisplayPng, recordDisplayClip } from '@/app/exportImage';
import { IconPause, IconPlay, IconSliders, IconTorso } from './icons';
import { MODALITY_LIST } from '@/simulator/renderer/modality';
import { ActionItem, CheckItem, MenuCap, usePopover } from './menu';

const MODES = MODALITY_LIST;

/**
 * Bottom toolbar: only the acquisition controls live in the first row — modality keys, freeze/cine
 * and the torso toggle. Interface toggles and secondary actions live in the ⋯ overflow menu.
 */
export function ModeBar() {
  const s = useSimStore(
    useShallow((st) => ({
      frozen: st.frozen,
      modality: st.modality,
      setModality: st.setModality,
      setUi: st.setUi,
      toggleFreeze: st.toggleFreeze,
      showTorso: st.ui.showTorso,
      minimal: st.ui.minimal,
      workerMode: st.workerMode,
    })),
  );
  const segRef = useRef<HTMLDivElement>(null);
  const pill = useSlidingPill(segRef, s.modality);
  return (
    <div className="modebar" role="toolbar" aria-label="Modalidades y cine">
      <div className="seg-bar" role="group" aria-label="Modalidad" ref={segRef}>
        {pill && <span className="seg-pill accent" style={pill} aria-hidden="true" />}
        {MODES.map((m) => (
          <button
            key={m.id}
            className={s.modality === m.id ? 'active' : ''}
            data-tour={m.id === 'color' ? 'color' : m.id === 'pw' ? 'pw' : undefined}
            onClick={() => s.setModality(m.id)}
            aria-pressed={s.modality === m.id}
          >
            {m.label}
            <kbd className="kbd" aria-hidden="true">
              {m.key.replace('Shift+', '⇧')}
            </kbd>
          </button>
        ))}
      </div>
      <ContextChip />
      <div className="sep" />
      <button
        className={`freeze-btn${s.frozen ? ' active' : ''}`}
        data-tour="freeze"
        onClick={() => s.toggleFreeze()}
        data-tip={s.frozen ? 'Reanudar' : 'Congelar la imagen'}
        data-tip-key="Espacio"
        aria-pressed={s.frozen}
      >
        <i className="run-dot" aria-hidden="true" />
        {s.frozen ? 'Reanudar' : 'Congelar'}
      </button>
      {s.frozen && <CineControls />}
      <span className="spacer" style={{ flex: 1 }} />
      {/* no label (decision 244): the button is an icon, its name stays «Torso 3D» for assistive technology */}
      <button
        onClick={() => s.setUi({ showTorso: !s.showTorso })}
        className={`icon-btn${s.showTorso ? ' active' : ''}`}
        aria-label="Torso 3D"
        data-tip={s.minimal ? undefined : 'Mostrar u ocultar el torso 3D'}
        data-tip-key={s.minimal ? undefined : 'H'}
        aria-pressed={s.showTorso}
        disabled={s.minimal}
      >
        <IconTorso />
      </button>
      <OverflowMenu />
      {s.workerMode === 'inline' && (
        <span
          className="small modebar-status"
          data-tip="La simulación corre en el hilo de la interfaz: puede ir más lenta"
        >
          Sin worker
        </span>
      )}
    </div>
  );
}

/** Mounted only while frozen so live ModeBar never subscribes to high-frequency HUD ticks. */
function CineControls() {
  const s = useSimStore(
    useShallow((st) => ({
      cineOffset: st.cineOffset,
      cineSpeed: st.cineSpeed,
      jumpCinePhase: st.jumpCinePhase,
      setCineOffset: st.setCineOffset,
      setCineSpeed: st.setCineSpeed,
    })),
  );
  const hud = useHudStore(
    useShallow((h) => ({
      hasHud: h.hud !== null,
      cineLength: h.hud?.cineLength ?? 0,
      spanS: h.hud ? h.hud.cineWindow.endS - h.hud.cineWindow.startS : 0,
      cineFramePhase: h.hud?.cineFramePhase ?? 0,
    })),
  );
  if (!hud.hasHud) return null;
  return (
    <div className="cine">
      <CinePlay length={hud.cineLength} spanS={hud.spanS} speed={s.cineSpeed} />
      <button
        type="button"
        className="cine-chip"
        aria-label="Cuadro anterior"
        data-tip="Cuadro anterior"
        disabled={s.cineOffset <= -(hud.cineLength - 1)}
        onClick={() => s.setCineOffset(s.cineOffset - 1)}
      >
        ‹
      </button>
      <button
        type="button"
        className="cine-chip"
        aria-label="Cuadro siguiente"
        data-tip="Cuadro siguiente"
        disabled={s.cineOffset >= 0}
        onClick={() => s.setCineOffset(s.cineOffset + 1)}
      >
        ›
      </button>
      <button
        type="button"
        className="cine-chip"
        aria-label={`Velocidad del cine ${s.cineSpeed}×`}
        data-tip="Cambiar velocidad del bucle (1× / 0.5× / 0.25×)"
        onClick={() => s.setCineSpeed(s.cineSpeed === 1 ? 0.5 : s.cineSpeed === 0.5 ? 0.25 : 1)}
      >
        {s.cineSpeed}×
      </button>
      <button
        type="button"
        className="cine-chip cine-phase"
        aria-label="Ir a telediástole"
        data-tip="Saltar a telediástole (TD · onda R)"
        disabled={hud.cineLength < 2}
        onClick={() => s.jumpCinePhase('ed')}
      >
        TD
      </button>
      <button
        type="button"
        className="cine-chip cine-phase"
        aria-label="Ir a telesístole"
        data-tip="Saltar a telesístole (TS · volumen mínimo VI)"
        disabled={hud.cineLength < 2}
        onClick={() => s.jumpCinePhase('es')}
      >
        TS
      </button>
      <input
        type="range"
        aria-label="Cine"
        min={-(hud.cineLength - 1)}
        max={0}
        value={s.cineOffset}
        onChange={(e) => s.setCineOffset(Number(e.target.value))}
      />
      <span className="small cine-read">
        {hud.cineLength + s.cineOffset}/{hud.cineLength} · {(hud.cineFramePhase * 100).toFixed(0)}%
      </span>
    </div>
  );
}

/**
 * One glanceable chip that states what the active modality is doing — colour Nyquist, the PW/TDI
 * gate depth, the CW cursor or the M-mode sweep. Read-only: the control lives in the Doppler tab.
 */
function ContextChip() {
  const modality = useSimStore((s) => s.modality);
  const scaleMps = useSimStore((s) => s.color.scaleMps);
  const gateDepthCm = useSimStore((s) => s.gateDepthCm);
  const sweepSpeed = useSimStore((s) => s.spectral.sweepSpeedMmPerS);
  const spec: Record<string, { text: string; title: string } | undefined> = {
    color: {
      text: `±${scaleMps.toFixed(2)} m/s`,
      title: 'Escala de Nyquist de la caja de color (pestaña Doppler)',
    },
    pw: { text: `Gate ${gateDepthCm.toFixed(1)} cm`, title: 'Profundidad de la compuerta PW' },
    cw: { text: 'Cursor', title: 'Cursor CW activo — sin compuerta' },
    tdi: { text: `Gate ${gateDepthCm.toFixed(1)} cm`, title: 'Profundidad de la compuerta TDI' },
    'm-mode': { text: `${sweepSpeed} mm/s`, title: 'Velocidad de barrido del modo M' },
    cmm: { text: `${sweepSpeed} mm/s`, title: 'Velocidad de barrido del modo M' },
  };
  const chip = spec[modality];
  if (!chip) return null;
  return (
    <span className="ctx-chip" data-tip={chip.title}>
      {chip.text}
    </span>
  );
}

/** ⋯ menu: interface toggles (checkable) plus the secondary actions, per the minimal-console spec. */
function OverflowMenu() {
  const { open, setOpen, wrap } = usePopover();
  const [recording, setRecording] = useState(false);

  return (
    <div className="menu-wrap" ref={wrap}>
      <button
        className={`menu-btn${open ? ' active' : ''}`}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Más opciones"
        data-tip="Paneles y acciones"
        onClick={() => setOpen(!open)}
      >
        <IconSliders />
      </button>
      {open && (
        <OverflowMenuContent
          recording={recording}
          setRecording={setRecording}
          close={() => setOpen(false)}
        />
      )}
    </div>
  );
}

function OverflowMenuContent({
  recording,
  setRecording,
  close,
}: {
  recording: boolean;
  setRecording: (v: boolean) => void;
  close: () => void;
}) {
  const s = useSimStore(
    useShallow((st) => ({
      caseId: st.caseId,
      examFinished: st.examFinished,
      mode: st.mode,
      setUi: st.setUi,
      ui: st.ui,
    })),
  );
  const policy = modePolicy(s.mode);
  const devOn = policy.devToolsAllowed && s.ui.devMode;

  return (
    <div className="menu" role="menu" aria-label="Paneles y acciones">
      <MenuCap>Mostrar</MenuCap>
      <CheckItem
        label="Ayudas de vista"
        checked={s.ui.showHints}
        onToggle={() => s.setUi({ showHints: !s.ui.showHints })}
        disabled={!policy.hintsEnabled}
        hint="Puntuación y sugerencias de adquisición"
      />
      <CheckItem
        label="Datos sobre imagen"
        checked={s.ui.showHud}
        onToggle={() => s.setUi({ showHud: !s.ui.showHud })}
        hint="Caso, FC y parámetros en las esquinas"
      />
      <CheckItem
        label="Interfaz limpia"
        checked={s.ui.minimal}
        onToggle={() => s.setUi({ minimal: !s.ui.minimal })}
        hint="Solo imagen y consola — oculta el rail izquierdo"
      />
      <CheckItem
        label="Modo sala"
        checked={s.ui.roomMode}
        onToggle={() => s.setUi({ roomMode: !s.ui.roomMode })}
        hint="Atenúa todo salvo la imagen, como en la sala a oscuras"
      />
      <CheckItem
        label="ECG"
        checked={s.ui.showEcg}
        onToggle={() => s.setUi({ showEcg: !s.ui.showEcg })}
      />
      <div className="menu-sep" />
      <MenuCap>Acciones</MenuCap>
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
            hint="Habilita Lab, anatomía experimental, revisión y telemetría"
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
                hint="Marca sobre la imagen para documentar hallazgos (R)"
              />
              <CheckItem
                label="Superposición física"
                checked={s.ui.showPhysics}
                onToggle={() => s.setUi({ showPhysics: !s.ui.showPhysics })}
                hint="Líneas de barrido y zona focal"
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

/**
 * Replays the frozen cine as a loop at the pace it was acquired (decision 191): each tick shows the next frame and the
 * newest wraps to the oldest. The loop lives in the store, so any frame picked by hand, a tool, review mode or the
 * freeze stops it (decision 197); the button keeps one name and says whether it plays with aria-pressed.
 */
function CinePlay({
  length,
  spanS,
  speed,
}: {
  length: number;
  spanS: number;
  speed: 0.25 | 0.5 | 1;
}) {
  const playing = useSimStore((st) => st.cinePlaying);
  const setPlaying = useSimStore((st) => st.setCinePlaying);
  const stepCine = useSimStore((st) => st.stepCine);
  useEffect(() => {
    if (!playing || length < 2) return;
    const frameMs = (1000 * Math.max(0.1, spanS)) / (length - 1) / speed;
    const id = setInterval(stepCine, frameMs);
    return () => clearInterval(id);
  }, [playing, length, spanS, speed, stepCine]);
  return (
    <button
      className={`icon-btn${playing ? ' active' : ''}`}
      aria-label="Reproducir el cine"
      aria-pressed={playing}
      data-tip={playing ? 'Pausar el cine' : 'Reproducir el cine'}
      disabled={length < 2}
      onClick={() => setPlaying(!playing)}
    >
      {playing ? <IconPause size={13} /> : <IconPlay size={13} />}
    </button>
  );
}
