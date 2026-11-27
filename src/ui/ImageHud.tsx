import { useLayoutEffect, useRef } from 'react';
import { useShallow } from 'zustand/shallow';
import { viewLabel } from '@/simulator/windows/viewLabels';
import { useHudStore, useSimStore } from '@/app/store';
import { EXAM_CASE_NAME, modePolicy, showsCaseIdentity } from '@/app/modePolicy';
import { listCases } from '@/cases';
import { isStripModality } from '@/simulator/renderer/modality';
import type { ProductMode } from '@/app/modePolicy';
import { formatMHz } from './format';
import { gpuNotice } from './gpuNotice';

const RHYTHM_LABEL: Record<string, string> = {
  sinus: 'Sinusal',
  'sinus-tachycardia': 'Taquicardia sinusal',
  'sinus-bradycardia': 'Bradicardia sinusal',
  'atrial-fibrillation': 'FA',
};

/**
 * On-image telemetry, where a real scanner prints it: exam name top-left, vitals and acquisition
 * parameters top-right, frame rate and the recognised view bottom-right. Pure overlay — it never
 * intercepts pointer events and disappears with `ui.showHud`.
 */
export function ImageHud() {
  const show = useSimStore((s) => s.ui.showHud);
  const mode = useSimStore((s) => s.mode);
  const caseId = useSimStore((s) => s.caseId);
  const examFinished = useSimStore((s) => s.examFinished);
  const truth = useSimStore((s) => s.truth);
  const liveDepthCm = useSimStore((s) => s.settings.depthCm);
  const liveFrequencyMHz = useSimStore((s) => s.settings.frequencyMHz);
  const liveHarmonics = useSimStore((s) => s.settings.harmonics);
  const liveStrip = useSimStore((s) => isStripModality(s.modality));
  const { hrBpm, fps, gpu, tier, gpuLive, frozenAcq } = useHudStore(
    useShallow((h) => ({
      hrBpm: h.hud ? Math.round(h.hud.heartRateBpm) : null,
      fps: h.hud ? Math.round(h.hud.simulatedFps) : null,
      gpu: h.hud?.stats?.['gpu'] ?? null,
      tier: h.hud?.stats?.['tier'] ?? null,
      gpuLive: h.hud?.stats?.['gpuLive'] ?? null,
      frozenAcq: h.hud?.frozen ? (h.hud.acquisition ?? null) : null,
    })),
  );
  const depthCm = frozenAcq ? frozenAcq.settings.depthCm : liveDepthCm;
  const frequencyMHz = frozenAcq ? frozenAcq.settings.frequencyMHz : liveFrequencyMHz;
  const harmonics = frozenAcq ? frozenAcq.settings.harmonics : liveHarmonics;
  const strip = frozenAcq ? isStripModality(frozenAcq.modality) : liveStrip;
  const notice = gpuNotice({ gpu, tier, gpuLive });
  // the title wraps to two lines on a narrow image: the segment chip below it reads its height (--hud-tl-h) to keep clear
  const titleBoxRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const box = titleBoxRef.current;
    const host = box?.parentElement?.parentElement;
    if (!box || !host) return;
    // and the title leaves the width of the acquisition box on its right (--hud-tr-w), whatever its lines say
    const right = box.parentElement?.querySelector<HTMLElement>('.hud-tr') ?? null;
    const publish = () => {
      host.style.setProperty('--hud-tl-h', `${box.offsetHeight}px`);
      if (right) {
        host.style.setProperty('--hud-tr-w', `${right.offsetWidth}px`);
        host.style.setProperty('--hud-tr-b', `${right.offsetTop + right.offsetHeight}`);
      }
    };
    publish();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(publish);
    ro.observe(box);
    if (right) ro.observe(right);
    return () => ro.disconnect();
  }, [show]);
  if (!show) return null;
  const identity = showsCaseIdentity(mode, examFinished);
  const title = identity
    ? (listCases().find((c) => c.id === caseId)?.title ?? caseId)
    : EXAM_CASE_NAME;
  // the rhythm is a finding the exam asks for (atrial fibrillation): until it is finished the learner reads the ECG
  const rhythm = truth && identity ? ` · ${RHYTHM_LABEL[truth.rhythm] ?? truth.rhythm}` : '';
  // the view line sits bottom-right on the 2D image; with a strip below the sector it joins the top-right group, off
  // the strip's velocity scale
  const viewLine = modePolicy(mode).showViewFeedback ? <ViewLine mode={mode} /> : null;
  return (
    <div className="img-hud">
      <div className="hud-box hud-tl" ref={titleBoxRef}>
        <span className="hud-title" title={title}>
          {title}
        </span>
      </div>
      <div className="hud-box hud-tr">
        <span>
          FC {hrBpm !== null ? hrBpm : '—'} lpm
          {rhythm}
        </span>
        <span>
          {depthCm} cm · {formatMHz(frequencyMHz)}
          {harmonics ? ' THI' : ''}
        </span>
        {/* in colour the frame rate already is the colour one: «· color N» repeated it (decision 195) */}
        <span>FR {fps !== null ? fps : '—'} Hz</span>
        {strip ? viewLine : null}
        {notice ? (
          // the frame shown was formed by the CPU tracer (its tip: why, and the port now); or, frozen over one the GPU
          // formed, the port is lost since
          <span className="hud-warn" data-tip={notice.tip}>
            {notice.text}
          </span>
        ) : null}
      </div>
      {!strip && viewLine ? <div className="hud-box hud-br">{viewLine}</div> : null}
    </div>
  );
}

/** The recognised view and its score, with a ring in the colour of the guide; null in exam mode. */
function ViewLine({ mode }: { mode: ProductMode }) {
  const { hasView, bestViewId, score } = useHudStore(
    useShallow((h) => ({
      hasView: h.hud?.view != null,
      bestViewId: h.hud?.view?.bestViewId ?? null,
      score: h.hud?.view?.score ?? 0,
    })),
  );
  if (!modePolicy(mode).showViewFeedback) return null;
  return (
    <span className="hud-view">
      {hasView ? (
        <i
          className="mini-ring"
          aria-hidden="true"
          style={{
            ['--p' as string]: `${Math.max(0, Math.min(100, score))}%`,
            color: score >= 75 ? 'var(--ok)' : score >= 50 ? 'var(--warn)' : 'var(--bad)',
          }}
        />
      ) : null}
      Vista {bestViewId ? `${viewLabel(bestViewId)} ${score}` : '—'}
    </span>
  );
}
