import { useEffect, useRef, useState } from 'react';
import { useSlidingPill } from './useSlidingPill';
import { isClassModeOn, nextClassModeUi, useHudStore, useSimStore } from '@/app/store';
import { useShallow } from 'zustand/shallow';
import { examClosed, exportCaseTag, modePolicy } from '@/app/modePolicy';
import { exportDisplayPng, recordDisplayClip } from '@/app/exportImage';
import { captureCurrentStudyStill } from '@/app/studyCapture';
import {
  IconArrowPointer,
  IconCamera,
  IconPause,
  IconPlay,
  IconPresentation,
  IconSliders,
  IconTorso,
} from './icons';
import { MODALITY_LIST } from '@/simulator/renderer/modality';
import { Segmented } from './controls';
import { ActionItem, CheckItem, MenuCap, usePopover } from './menu';

const PRIMARY_MODES = MODALITY_LIST.filter((m) => m.id !== 'cmm');

/**
 * Clean bottom acquisition bar: the 6 daily modalities (CMM appears on demand via Shift+M or Ajustes),
 * Freeze/Store, cine playback when frozen, and 2 layout icons (Modo clase, Torso 3D) + ⋯ menu.
 */
export function ModeBar() {
  const s = useSimStore(
    useShallow((st) => ({
      closed: examClosed(st),
      captureCount: st.captures.length,
      frozen: st.frozen,
      mode: st.mode,
      modality: st.modality,
      setModality: st.setModality,
      setUi: st.setUi,
      toggleFreeze: st.toggleFreeze,
      teachingPointerOn: st.teachingPointerOn,
      teachingArrowCount: st.teachingArrows.length,
      setTeachingPointer: st.setTeachingPointer,
      clearTeachingArrows: st.clearTeachingArrows,
      showTorso: st.ui.showTorso,
      minimal: st.ui.minimal,
      roomMode: st.ui.roomMode,
      workerMode: st.workerMode,
    })),
  );
  const classMode = isClassModeOn({ minimal: s.minimal, roomMode: s.roomMode });
  const [savedFlash, setSavedFlash] = useState(false);
  const flashTimer = useRef<number | null>(null);
  useEffect(() => () => window.clearTimeout(flashTimer.current ?? undefined), []);
  const onStore = () => {
    if (s.closed) return;
    if (captureCurrentStudyStill()) {
      setSavedFlash(true);
      window.clearTimeout(flashTimer.current ?? undefined);
      flashTimer.current = window.setTimeout(() => setSavedFlash(false), 1200);
    }
  };
  const visibleModes = s.modality === 'cmm' ? MODALITY_LIST : PRIMARY_MODES;
  const segRef = useRef<HTMLDivElement>(null);
  const pill = useSlidingPill(segRef, s.modality);
  return (
    <div className="modebar" role="toolbar" aria-label="Modalidades y cine">
      <div className="seg-bar" role="group" aria-label="Modalidad" ref={segRef}>
        {pill && <span className="seg-pill accent" style={pill} aria-hidden="true" />}
        {visibleModes.map((m) => (
          <button
            key={m.id}
            className={s.modality === m.id ? 'active' : ''}
            data-tour={m.id === 'color' ? 'color' : m.id === 'pw' ? 'pw' : undefined}
            data-tip={m.name}
            data-tip-key={m.key.replace('Shift+', '⇧')}
            onClick={() => s.setModality(m.id)}
            aria-pressed={s.modality === m.id}
            disabled={s.closed}
          >
            {m.label}
          </button>
        ))}
      </div>
      <div className="sep" />
      <button
        className={`freeze-btn${s.frozen ? ' active' : ''}`}
        data-tour="freeze"
        onClick={() => s.toggleFreeze()}
        data-tip={
          s.closed
            ? 'Examen finalizado: la imagen queda congelada'
            : s.frozen
              ? 'Reanudar adquisición en vivo'
              : 'Congelar la imagen'
        }
        data-tip-key={s.closed ? undefined : 'Espacio'}
        aria-pressed={s.frozen}
        disabled={s.closed}
      >
        <i className="run-dot" aria-hidden="true" />
        {s.frozen ? 'Reanudar' : 'Congelar'}
      </button>
      <button
        type="button"
        className={`store-btn${savedFlash ? ' saved' : ''}`}
        onClick={onStore}
        disabled={s.closed}
        data-tip={
          s.closed
            ? 'Examen finalizado: no se pueden guardar más capturas'
            : 'Guardar imagen actual en el estudio e informe'
        }
        data-tip-key={s.closed ? undefined : 'K'}
        aria-label="Guardar captura en el estudio"
      >
        <IconCamera size={14} />
        <span>{savedFlash ? 'Guardada' : 'Guardar'}</span>
        {s.captureCount > 0 && <span className="store-badge">{s.captureCount}</span>}
      </button>
      {s.frozen && !s.closed && s.mode !== 'exam' && (
        <div className="pointer-group">
          <button
            type="button"
            className={`pointer-btn${s.teachingPointerOn ? ' active' : ''}`}
            onClick={() => s.setTeachingPointer(!s.teachingPointerOn)}
            aria-pressed={s.teachingPointerOn}
            data-tip={
              s.teachingPointerOn
                ? 'Arrastra sobre la imagen congelada para señalar con una flecha (Esc para salir)'
                : 'Señalar sobre el cuadro congelado con flechas temporales para clase'
            }
          >
            <IconArrowPointer size={13} />
            <span>Señalar</span>
          </button>
          {s.teachingArrowCount > 0 && (
            <button
              type="button"
              className="pointer-clear-btn"
              onClick={() => s.clearTeachingArrows()}
              aria-label={`Borrar ${s.teachingArrowCount} flechas docentes`}
              data-tip="Borrar flechas temporales (Esc)"
            >
              ×{s.teachingArrowCount}
            </button>
          )}
        </div>
      )}
      {s.frozen && <CineControls />}
      <span className="spacer" style={{ flex: 1 }} />
      <button
        type="button"
        onClick={() => s.setUi(nextClassModeUi({ minimal: s.minimal, roomMode: s.roomMode }))}
        className={`icon-btn${classMode ? ' active' : ''}`}
        aria-label="Modo clase (proyector)"
        data-tip={
          classMode
            ? 'Salir del modo clase'
            : 'Modo clase / proyector: maximiza la imagen y oscurece la interfaz'
        }
        data-tip-key="F"
        aria-pressed={classMode}
      >
        <IconPresentation />
      </button>
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
      <button
        type="button"
        className="cine-chip"
        aria-label={`Velocidad del cine ${s.cineSpeed}×`}
        data-tip="Velocidad del bucle (1× / 0.5× / 0.25×)"
        onClick={() => s.setCineSpeed(s.cineSpeed === 1 ? 0.5 : s.cineSpeed === 0.5 ? 0.25 : 1)}
      >
        {s.cineSpeed}×
      </button>
      <span className="small cine-read">{(hud.cineFramePhase * 100).toFixed(0)}%</span>
    </div>
  );
}

/** ⋯ menu: quality, interface toggles, export actions, and developer tools. */
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
        data-tip="Opciones, calidad y exportación"
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
      quality: st.quality,
      setQuality: st.setQuality,
      setUi: st.setUi,
      ui: st.ui,
    })),
  );
  const policy = modePolicy(s.mode);
  const devOn = policy.devToolsAllowed && s.ui.devMode;

  return (
    <div className="menu" role="menu" aria-label="Paneles y acciones">
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
        hint="Puntuación y sugerencias de adquisición"
      />
      <CheckItem
        label="Datos sobre imagen"
        checked={s.ui.showHud}
        onToggle={() => s.setUi({ showHud: !s.ui.showHud })}
        hint="Caso, FC y parámetros en las esquinas"
      />
      <CheckItem
        label="Modo sala oscura"
        checked={s.ui.roomMode}
        onToggle={() => s.setUi({ roomMode: !s.ui.roomMode })}
        hint="Atenúa los bordes dejando destacar la ecografía"
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
            hint="Lab, anatomía experimental, revisión y telemetría"
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

/** Replays the frozen cine loop at the selected playback speed. */
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
