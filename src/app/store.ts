import { create } from 'zustand';
import { createAcquisitionSlice } from './store/acquisitionSlice';
import { createEducationSlice } from './store/educationSlice';
import { createReviewSlice } from './store/reviewSlice';
import {
  createUiSlice,
  useHudStore,
  useSegmentHover,
  useSegmentOrientation,
  useStructureHover,
} from './store/uiSlice';
import type { SimStore } from './store/types';

export type {
  AcquisitionSlice,
  ArtifactLabOverrides,
  ConsoleTab,
  EducationSlice,
  NavigatorLayers,
  PresetAnimationState,
  ProductMode,
  ReviewSlice,
  SimStore,
  StudyCapture,
  UiPrefs,
  UiSlice,
} from './store/types';

export {
  START_PROBE,
  START_VIEW_ID,
  imageSegmentsOn,
  navigatorLayers,
  segmentLayerOn,
} from './store/helpers';

export { useHudStore, useSegmentHover, useSegmentOrientation, useStructureHover };

export const useSimStore = create<SimStore>((...a) => ({
  ...createAcquisitionSlice(...a),
  ...createEducationSlice(...a),
  ...createReviewSlice(...a),
  ...createUiSlice(...a),
}));
