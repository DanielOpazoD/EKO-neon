import type * as THREE from 'three';
import { useHudStore, useSimStore } from '@/app/store';
import type { ThoraxModel } from '@/simulator/anatomy/thoraxModel';
import { beamFrameFromPose, poseFromControl } from '@/simulator/probe/pose';

/** What the navigator's scene controls the component holds: the camera's zoom and framing. */
export interface NavigatorZoom {
  zoomBy: (f: number) => void;
  center: () => void;
  overview: () => void;
  plane: () => void;
}

/** The acquisition the navigator shows: the frozen one while frozen, the live controls otherwise. */
export function navigatorState() {
  const state = useSimStore.getState();
  const hud = useHudStore.getState().hud;
  return hud?.frozen && hud.acquisition ? { ...state, ...hud.acquisition } : state;
}

/** Vertical field of view of the navigator: 35°, widened below an aspect of 0.5 to keep the horizontal field. */
export function narrowPanelFovDeg(aspect: number): number {
  const half = (35 / 2) * (Math.PI / 180);
  return (2 * Math.atan(Math.tan(half) * Math.max(1, 0.5 / aspect)) * 180) / Math.PI;
}

/** Azimuth and elevation (rad), distance (cm) and target point of the camera's orbit. */
export interface NavigatorOrbit {
  az: number;
  el: number;
  r: number;
  tx: number;
  ty: number;
  tz: number;
}

/** Where the camera stands: carried over when the scene is rebuilt for another patient state or case. */
export interface NavigatorCameraState {
  orbit: NavigatorOrbit;
  up: { x: number; y: number; z: number };
}

/** The orbit of the navigator's camera around a target point, and the framings its buttons offer. */
export interface NavigatorCameraRig {
  /** Change it, then call `update`. */
  readonly orbit: NavigatorOrbit;
  /** Places the camera on the orbit and asks for a redraw. */
  update: () => void;
  readonly zoom: NavigatorZoom;
  /** The camera as it stands, for the next rig to start from. */
  state: () => NavigatorCameraState;
}

const OVERVIEW: NavigatorOrbit = { az: 0.12, el: 0.08, r: 54, tx: 2, ty: -1.5, tz: -6 };

/**
 * The rig starts at the overview, or where `from` left the camera: the scene is rebuilt when the patient's breathing or
 * position changes, and the learner's framing went back to the overview each time.
 */
export function createCameraRig(
  camera: THREE.PerspectiveCamera,
  thorax: () => ThoraxModel | null,
  markDirty: () => void,
  from?: NavigatorCameraState | null,
): NavigatorCameraRig {
  const orbit = { ...(from?.orbit ?? OVERVIEW) };
  if (from) camera.up.set(from.up.x, from.up.y, from.up.z);
  const update = () => {
    markDirty();
    camera.position.set(
      orbit.tx + orbit.r * Math.sin(orbit.az) * Math.cos(orbit.el),
      orbit.ty + orbit.r * Math.sin(orbit.el),
      orbit.tz + orbit.r * Math.cos(orbit.az) * Math.cos(orbit.el),
    );
    camera.lookAt(orbit.tx, orbit.ty, orbit.tz);
  };
  update();
  const zoom: NavigatorZoom = {
    zoomBy: (f) => {
      orbit.r = Math.max(22, Math.min(90, orbit.r * f));
      update();
    },
    overview: () => {
      Object.assign(orbit, OVERVIEW);
      camera.up.set(0, 1, 0);
      update();
    },
    plane: () => {
      const t = thorax();
      if (!t) return;
      const state = navigatorState();
      const beam = beamFrameFromPose(poseFromControl(t, state.probe));
      const d = state.settings.depthCm / 2;
      Object.assign(orbit, {
        tx: beam.origin.x + d * beam.forward.x,
        ty: beam.origin.y + d * beam.forward.y,
        tz: beam.origin.z + d * beam.forward.z,
        az: Math.atan2(beam.normal.x, beam.normal.z),
        el: Math.asin(Math.max(-1, Math.min(1, beam.normal.y))),
        r: Math.max(30, state.settings.depthCm * 2.5),
      });
      camera.up.set(-beam.forward.x, -beam.forward.y, -beam.forward.z);
      update();
    },
    center: () => {
      camera.up.set(0, 1, 0);
      const t = thorax();
      if (!t) return;
      const p = poseFromControl(t, useSimStore.getState().probe).position;
      orbit.tx = p.x;
      orbit.ty = p.y;
      orbit.tz = p.z - 5;
      orbit.r = 30;
      update();
    },
  };
  const state = (): NavigatorCameraState => ({
    orbit: { ...orbit },
    up: { x: camera.up.x, y: camera.up.y, z: camera.up.z },
  });
  return { orbit, update, zoom, state };
}
