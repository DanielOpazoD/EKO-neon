import { useEffect, useRef, useState } from 'react';
import { useShallow } from 'zustand/shallow';
import { navigatorLayers, useHudStore, useSimStore } from '@/app/store';
import { modePolicy } from '@/app/modePolicy';
import { CutMapView } from './CutMapView';
import { RotationDial } from './RotationDial';
import { ActionItem, CheckItem, MenuCap, usePopover } from './menu';
import { IconLayers } from './icons';
import { mountNavigatorScene, type NavigatorZoom } from './navigator/navigatorScene';
import type { NavigatorCameraState } from './navigator/navigatorCamera';

export { narrowPanelFovDeg } from './navigator/navigatorScene';

const navigatorCamera: { current: NavigatorCameraState | null } = { current: null };

/**
 * Minimal 3D torso + probe navigator: only 2 top icons (reset view + layers/camera menu),
 * compact 2-pill drag mode toggle (Sonda | Cámara) while live, and zero permanent help-text clutter.
 */
export function TorsoView() {
  const ref = useRef<HTMLDivElement>(null);
  const [dragAction, setDragAction] = useState<'probe' | 'camera'>('probe');
  const dragActionRef = useRef(dragAction);
  const caseId = useSimStore((s) => s.caseId);
  const split = useSimStore((s) => navigatorLayers(s).split);
  const anatomy = useSimStore((s) => modePolicy(s.mode).navigatorAnatomy);
  const livePatient = useSimStore((s) => s.patient);
  const patient = useHudStore(
    useShallow((s) => (s.hud?.frozen ? (s.hud.acquisition?.patient ?? livePatient) : livePatient)),
  );
  const frozen = useHudStore((s) => s.hud?.frozen ?? false);
  const zoomRef = useRef<NavigatorZoom | null>(null);
  const [webglError] = useState(probeWebgl);

  useEffect(() => {
    const el = ref.current;
    if (!el || webglError) return;
    return mountNavigatorScene(el, {
      caseId,
      patient,
      dragAction: dragActionRef,
      zoom: zoomRef,
      camera: navigatorCamera,
    });
  }, [caseId, patient, webglError]);

  const setDrag = (next: 'probe' | 'camera') => {
    dragActionRef.current = next;
    setDragAction(next);
  };

  return (
    <div className={`torso-wrap${split ? ' split' : ''}`}>
      <div className="torso-head">
        {split && <div className="torso-caption top">Sonda y tórax</div>}
        <div className="torso-buttons">
          <button
            type="button"
            className="icon-btn"
            onClick={() => zoomRef.current?.overview()}
            data-tip="Restablecer vista 3D (rueda: rotar · Ctrl+rueda: zoom · clic derecho: orbitar · ? atajos)"
            aria-label="Restablecer vista global del tórax"
          >
            ↺
          </button>
          {anatomy && <LayerMenu zoomRef={zoomRef} />}
        </div>
      </div>
      <div className="torso-3d" ref={ref} aria-label="Torso 3D y sonda virtual" data-tour="torso">
        {webglError && (
          <div className="panel-error" role="alert">
            <b>La vista 3D necesita WebGL 2, que este navegador no ofrece.</b>
            <span className="small">
              La imagen ecográfica, el mapa del corte y los controles de la sonda siguen
              funcionando.
            </span>
          </div>
        )}
      </div>
      {split && <CutMapView />}
      {!frozen && (
        <div
          className="torso-gesture"
          role="group"
          aria-label="Acción al arrastrar en el navegador"
        >
          <button
            type="button"
            className={dragAction === 'probe' ? 'active' : ''}
            aria-pressed={dragAction === 'probe'}
            onClick={() => setDrag('probe')}
            data-tip="Arrastrar mueve la sonda sobre el tórax"
          >
            Sonda
          </button>
          <button
            type="button"
            className={dragAction === 'camera' ? 'active' : ''}
            aria-pressed={dragAction === 'camera'}
            onClick={() => setDrag('camera')}
            data-tip="Arrastrar orbita la cámara sin mover la sonda"
          >
            Cámara
          </button>
        </div>
      )}
      <div className="torso-tools">
        <RotationDial />
      </div>
    </div>
  );
}

function probeWebgl(): string | null {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    if (!gl) return 'este navegador no ofrece un contexto WebGL2';
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return null;
  } catch (e) {
    return e instanceof Error ? e.message : String(e);
  }
}

/** 3D navigator layers + camera framing actions tucked inside one clean popover. */
function LayerMenu({ zoomRef }: { zoomRef: React.RefObject<NavigatorZoom | null> }) {
  const { open, setOpen, wrap } = usePopover();
  return (
    <div className="menu-wrap" ref={wrap}>
      <button
        type="button"
        className={`icon-btn${open ? ' active' : ''}`}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Capas del navegador 3D"
        data-tip="Capas y cámara del navegador 3D"
        onClick={() => setOpen(!open)}
      >
        <IconLayers size={12} />
      </button>
      {open && <LayerMenuContent zoomRef={zoomRef} close={() => setOpen(false)} />}
    </div>
  );
}

function LayerMenuContent({
  zoomRef,
  close,
}: {
  zoomRef: React.RefObject<NavigatorZoom | null>;
  close: () => void;
}) {
  const ui = useSimStore((s) => s.ui);
  const setUi = useSimStore((s) => s.setUi);
  const layer = (label: string, key: keyof typeof ui, hint: string) => (
    <CheckItem
      label={label}
      checked={Boolean(ui[key])}
      onToggle={() => setUi({ [key]: !ui[key] })}
      hint={hint}
    />
  );
  return (
    <div className="menu down" role="menu" aria-label="Capas del navegador 3D">
      <MenuCap>Cuerpo</MenuCap>
      {layer('Piel', 'navSkin', 'Superficie del tórax')}
      {layer('Hueso', 'showSkeleton', 'Costillas y esternón')}
      <MenuCap>Corazón</MenuCap>
      {layer('Miocardio', 'navHeart', 'Paredes del modelo 3D')}
      {layer('Cavidades', 'navChambers', 'Ventrículos y aurículas')}
      {layer('Válvulas', 'navValves', 'Válvulas y cuerdas')}
      {layer('Segmentos del VI', 'navSegments', 'Colorea los segmentos del VI en 3D y en el corte')}
      {layer('Vasos', 'navVessels', 'Aorta, pulmonar y cavas')}
      <MenuCap>Corte</MenuCap>
      {layer('Corte de frente', 'navSplit', 'Plano ecográfico visto de frente')}
      {layer('Etiquetas del corte', 'navLabels', 'Nombres de cámaras y válvulas')}
      {layer('Ventanas', 'navWindows', 'Ventanas acústicas canónicas')}
      {layer('Ejes', 'navAxes', 'Haz, elevación y lateral')}
      {layer('Cortar corazón', 'navCut', 'Seccionar el corazón por el plano')}
      <div className="menu-sep" />
      <MenuCap>Cámara</MenuCap>
      <ActionItem
        label="Centrar en la sonda"
        onClick={() => {
          zoomRef.current?.center();
          close();
        }}
      />
      <ActionItem
        label="Mirar plano de frente"
        onClick={() => {
          zoomRef.current?.plane();
          close();
        }}
      />
    </div>
  );
}
