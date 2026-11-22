import type { SimStore } from '@/app/store';

export interface CutMapStoreSlice {
  caseId: SimStore['caseId'];
  mode: SimStore['mode'];
  probe: SimStore['probe'];
  settings: Pick<SimStore['settings'], 'invertLR' | 'zoom'>;
  ui: Pick<SimStore['ui'], 'navLabels' | 'navSegments' | 'segmentModel' | 'selectedSegment'>;
}

/**
 * Pure predicate for `CutMapView`'s store subscription: returns true only when a store mutation can change
 * the cut-map sector LUT, labels, segment coloring, or probe rest-key. Ignores HUD/telemetry ticks (e.g.
 * `fpsUi`), gain/TGC/dynamic-range changes, cine scrubbing, and unrelated navigator layer toggles.
 */
export function cutMapStoreChanged(prev: CutMapStoreSlice, next: CutMapStoreSlice): boolean {
  if (prev === next) return false;
  return (
    prev.caseId !== next.caseId ||
    prev.mode !== next.mode ||
    prev.probe !== next.probe ||
    prev.settings.invertLR !== next.settings.invertLR ||
    prev.settings.zoom !== next.settings.zoom ||
    prev.ui.navLabels !== next.ui.navLabels ||
    prev.ui.navSegments !== next.ui.navSegments ||
    prev.ui.segmentModel !== next.ui.segmentModel ||
    prev.ui.selectedSegment !== next.ui.selectedSegment
  );
}
