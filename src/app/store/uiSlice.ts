import { create } from 'zustand';
import type { SimOutput } from '@/simulator/core/protocol';
import { loadPrefs, savePrefs } from './helpers';
import type { SimSliceCreator, UiSlice } from './types';

/** The structure under the pointer on the cut map (decision 202): the image outlines it in return. */
export const useStructureHover = create<{
  id: number | null;
  setHover: (id: number | null) => void;
}>((set) => ({ id: null, setHover: (id) => set((s) => (s.id === id ? s : { id })) }));

/**
 * The LV segment under the pointer, wherever it is — the image, the cut map, the 3D heart or the polar map — so the
 * others highlight it too (decision 153). Its own store: it changes as the mouse moves and is never persisted.
 */
export const useSegmentHover = create<{
  id: number | null;
  source: 'image' | 'cut' | 'heart' | 'polar' | null;
  setHover: (id: number | null, source: 'image' | 'cut' | 'heart' | 'polar' | null) => void;
}>((set, get) => ({
  id: null,
  source: null,
  setHover: (id, source) => {
    const cur = get();
    // leaving one view clears only that view's hover, so a late mouseleave does not undo another view's
    if (id === null && cur.source !== null && cur.source !== source) return;
    if (cur.id !== id || cur.source !== (id === null ? null : source))
      set({ id, source: id === null ? null : source });
  },
}));

/**
 * How the short-axis cut on the image sits against the polar map (decision 181): the angle of the map (degrees from
 * anterior, counter-clockwise toward the septum) that the top of the image shows, and whether the RV insertions are
 * marked on the image. Null and false when the segment layer is off or the cut is not a short axis.
 */
export const useSegmentOrientation = create<{
  upDeg: number | null;
  insertions: boolean;
  set: (upDeg: number | null, insertions: boolean) => void;
}>((set, get) => ({
  upDeg: null,
  insertions: false,
  set: (upDeg, insertions) => {
    const cur = get();
    const up = upDeg === null ? null : Math.round(upDeg);
    if (cur.upDeg !== up || cur.insertions !== insertions) set({ upDeg: up, insertions });
  },
}));

/** HUD (per-frame light state) lives in its own store so the console does not re-render per frame. */
export const useHudStore = create<{ hud: SimOutput | null; setHud: (h: SimOutput | null) => void }>(
  (set) => ({
    hud: null,
    setHud: (h) => set({ hud: h }),
  }),
);

function devModeQueryParam(): boolean {
  if (typeof window === 'undefined') return false;
  try {
    return new URLSearchParams(window.location.search).get('dev') === '1';
  } catch {
    return false;
  }
}

export const createUiSlice: SimSliceCreator<UiSlice> = (set) => {
  const persisted = loadPrefs();
  const devFromUrl = devModeQueryParam();
  const initialDevMode = devFromUrl || persisted.devMode === true;
  const initialTab =
    !initialDevMode && (persisted.consoleTab === 'lab' || persisted.consoleTab === 'revisar')
      ? 'adquirir'
      : persisted.consoleTab ?? 'adquirir';
  return {
    ui: {
      showTorso: true,
      showSkeleton: true,
      navSkin: true,
      navHeart: true,
      navChambers: true,
      navValves: true,
      navVessels: true,
      navAxes: true,
      navCut: false,
      navWindows: true,
      navSplit: true,
      navLabels: true,
      navSegments: false,
      imageSegments: false,
      segmentsOpen: false,
      segmentModel: 'LV_AHA17',
      selectedSegment: null,
      guidanceOpen: false,
      shortcutsOpen: false,
      showHints: true,
      reviewFreezeOnMark: true,
      reviewTool: 'point',
      showEcg: true,
      tutorialDone: true,
      showHud: true,
      railMini: false,
      minimal: false,
      roomMode: false,
      screen: 'simulator',
      ...persisted,
      devMode: initialDevMode,
      devPanel: initialDevMode ? (persisted.devPanel ?? false) : false,
      showPhysics: initialDevMode ? (persisted.showPhysics ?? false) : false,
      reviewMode: initialDevMode ? (persisted.reviewMode ?? false) : false,
      consoleTab: initialTab,
    },
    phaseMarks: null,
    lvLengthCm: null,
    error: null,
    workerMode: 'starting',
    navigatorReady: false,
    fpsUi: 0,

    setUi: (u) =>
      set((s) => {
        const ui = { ...s.ui, ...u };
        if (u.devMode === false) {
          ui.devPanel = false;
          ui.showPhysics = false;
          ui.reviewMode = false;
          if (ui.consoleTab === 'lab' || ui.consoleTab === 'revisar') {
            ui.consoleTab = 'adquirir';
          }
        }
        savePrefs(ui);
        return u.reviewMode ? { ui, cinePlaying: false } : { ui };
      }),
    setCycleInfo: (marks, lvLengthCm) => set({ phaseMarks: marks, lvLengthCm }),
    setError: (e) => set({ error: e }),
    setWorkerMode: (m) => set({ workerMode: m }),
    setNavigatorReady: (ready) => set({ navigatorReady: ready }),
    setFpsUi: (f) => set({ fpsUi: f }),
  };
};
