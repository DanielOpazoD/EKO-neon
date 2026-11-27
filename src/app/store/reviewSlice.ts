import { constrainAcquisition } from '@/simulator/core/acquisitionInput';
import { renumberMarkers } from '../review';
import { modePolicy } from '../modePolicy';
import type { ReviewSlice, SimSliceCreator } from './types';

export const createReviewSlice: SimSliceCreator<ReviewSlice> = (set, get) => ({
  reviewMarkers: [],
  reviewNote: '',
  reviewSelectedId: null,
  reviewUndo: [],
  reviewSource: null,
  reviewLinkParentId: null,

  addReviewMarker: (m) =>
    set((s) => ({
      reviewMarkers: renumberMarkers([...s.reviewMarkers, m]),
      reviewSelectedId: m.id,
      // the first marker on a live image freezes it, so the markers keep the frame they were put on
      ...(m.space === 'image' && !s.frozen && s.ui.reviewFreezeOnMark
        ? { frozen: true, cineOffset: 0 }
        : {}),
    })),
  updateReviewMarker: (id, patch) =>
    set((s) => ({
      reviewMarkers: s.reviewMarkers.map((m) => (m.id === id ? { ...m, ...patch } : m)),
    })),
  removeReviewMarker: (id) =>
    set((s) => {
      // a primary takes its secondary points with it; the batch comes back together with undo
      const goneIds = new Set([
        id,
        ...s.reviewMarkers.filter((m) => m.parentId === id).map((m) => m.id),
      ]);
      const gone = s.reviewMarkers.filter((m) => goneIds.has(m.id));
      return {
        reviewMarkers: renumberMarkers(s.reviewMarkers.filter((m) => !goneIds.has(m.id))),
        reviewUndo: gone.length ? [...s.reviewUndo.slice(-19), gone] : s.reviewUndo,
        reviewSelectedId: goneIds.has(s.reviewSelectedId ?? '') ? null : s.reviewSelectedId,
        reviewLinkParentId: goneIds.has(s.reviewLinkParentId ?? '') ? null : s.reviewLinkParentId,
      };
    }),
  clearReview: () =>
    set((s) => ({
      reviewMarkers: [],
      reviewNote: '',
      reviewSelectedId: null,
      reviewSource: null,
      reviewLinkParentId: null,
      reviewUndo: s.reviewMarkers.length
        ? [...s.reviewUndo.slice(-19), s.reviewMarkers]
        : s.reviewUndo,
    })),
  selectReviewMarker: (id) => set({ reviewSelectedId: id }),
  armReviewLink: (id) => set({ reviewLinkParentId: id }),
  undoReviewRemove: () =>
    set((s) => {
      const batch = s.reviewUndo[s.reviewUndo.length - 1];
      if (!batch) return s;
      const present = new Set([...s.reviewMarkers, ...batch].map((m) => m.id));
      // a secondary whose primary is gone for good comes back as a primary
      const restored = batch.map((m) =>
        m.parentId && !present.has(m.parentId) ? { ...m, parentId: null } : m,
      );
      return {
        reviewMarkers: renumberMarkers([...s.reviewMarkers, ...restored]),
        reviewUndo: s.reviewUndo.slice(0, -1),
        reviewSelectedId: restored[0]?.id ?? null,
      };
    }),
  setReviewNote: (note) => set({ reviewNote: note }),
  loadReviewReport: (r) => {
    const s = get();
    if (!modePolicy(s.mode).devToolsAllowed) return;
    if (r.caseId !== s.caseId) s.loadCase(r.caseId);
    const i = constrainAcquisition(r.input);
    set({
      probe: { ...i.probe },
      patient: { ...i.patient },
      settings: { ...i.settings, tgcDb: [...i.settings.tgcDb] },
      modality: i.modality,
      color: { ...i.color },
      colorBoxCustom: true,
      spectral: { ...i.spectral },
      cursorThetaRad: i.cursorThetaRad,
      gateDepthCm: i.gateDepthCm,
      quality: i.quality,
      rendererBackend: i.rendererBackend,
      anatomy: i.anatomy ?? 'analytic',
      imageFormation: i.imageFormation ?? 'lattice',
      // a frozen cine cannot be restored: the frame is replayed live at the report's probe and console
      frozen: false,
      cineOffset: 0,
      artifactLab: i.artifactOverrides,
      presetAnim: null,
      reviewMarkers: renumberMarkers(r.markers),
      reviewNote: r.note,
      reviewSelectedId: null,
      reviewLinkParentId: null,
      reviewSource: { author: r.author, createdAt: r.createdAt },
    });
    get().setUi({ reviewMode: true, consoleTab: 'revisar' });
  },
});
