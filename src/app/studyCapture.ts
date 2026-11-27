import { modalityLabel } from '@/simulator/renderer/modality';
import { viewLabel } from '@/simulator/windows/viewLabels';
import { captureStudyStillDataUrl } from './exportImage';
import { examClosed } from './modePolicy';
import { useHudStore, useSimStore, type StudyCapture } from './store';

/**
 * Format the clinical caption for a stored study still.
 * Strictly suppresses `viewId` when `revealed === false` (active Exam mode) so stored thumbnails
 * never reveal the recognized view before the exam is finished.
 */
export function formatCaptureCaption(cap: Pick<StudyCapture, 'modality' | 'viewId'>, revealed: boolean): string {
  const mod = modalityLabel(cap.modality);
  if (revealed && cap.viewId) return `${viewLabel(cap.viewId)} · ${mod}`;
  return mod;
}

/**
 * Store the currently displayed composited ultrasound frame (image + overlays + watermark) into the
 * active session's capture gallery. Returns `true` when a capture was stored.
 */
export function captureCurrentStudyStill(): boolean {
  const st = useSimStore.getState();
  if (examClosed(st)) return false;
  const dataUrl = captureStudyStillDataUrl(640);
  if (!dataUrl) return false;
  const hud = useHudStore.getState().hud;
  const modality = hud?.frozen && hud.acquisition ? hud.acquisition.modality : st.modality;
  st.addCapture({
    id: `cap-${Date.now().toString(36)}-${Math.floor(Math.random() * 1e4)}`,
    dataUrl,
    modality,
    viewId: hud?.view?.bestViewId ?? null,
    viewScore: hud?.view?.score ?? null,
    phase: hud?.phase ?? 0,
    frameId: hud?.frameId ?? 0,
    frozen: st.frozen,
    createdAt: new Date().toISOString(),
  });
  return true;
}
