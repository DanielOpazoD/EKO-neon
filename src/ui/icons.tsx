import type { ReactNode } from 'react';

/**
 * Minimal stroke icon set for the console (24×24 grid, 1.7px stroke, round caps).
 * Kept local and dependency-free; each glyph reads at small sizes in the tab bar.
 */
function I({ children, size = 15 }: { children: ReactNode; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {children}
    </svg>
  );
}

/** Ultrasound sector fan — acquisition / probe work. */
export function IconAcquire({ size }: { size?: number }) {
  return (
    <I size={size}>
      <circle cx="12" cy="3.6" r="1.4" fill="currentColor" stroke="none" />
      <path d="M12 3.6 4.4 20.4" />
      <path d="M12 3.6l7.6 16.8" />
      <path d="M4.4 20.4a18.9 18.9 0 0 0 15.2 0" />
    </I>
  );
}

/** Framed picture — B-mode image controls. */
export function IconImage({ size }: { size?: number }) {
  return (
    <I size={size}>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <circle cx="9" cy="10" r="1.6" />
      <path d="m5.5 18.5 5-5 3 3 3.5-3.5 3.5 3.5" />
    </I>
  );
}

/** Spectral pulse — Doppler modalities. */
export function IconDoppler({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="M3 12h3.5L9 5.5l4.5 13L16 12h5" />
    </I>
  );
}

/** Caliper jaws — measurement tools. */
export function IconMeasure({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="M6 3v4.5L10.5 12 8 21" />
      <path d="M18 3v4.5L13.5 12 16 21" />
      <path d="M6 3h12" />
    </I>
  );
}

/** Conical flask — artifact laboratory. */
export function IconLab({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="M9 3h6" />
      <path d="M10 3v5.2L5.2 18.6A2 2 0 0 0 7 21.4h10a2 2 0 0 0 1.8-2.8L14 8.2V3" />
      <path d="M7.5 14.5h9" />
    </I>
  );
}

/** Crosshair with a dot — review-mode markers (decision 134). */
export function IconReview({ size }: { size?: number }) {
  return (
    <I size={size}>
      <circle cx="12" cy="12" r="7" />
      <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
      <circle cx="12" cy="12" r="1.2" fill="currentColor" />
    </I>
  );
}

/** Pencil — review mode, notes and marks on the image (decision 376). */
export function IconPencil({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="M4 20l1-4L16 5l3 3L8 19z" />
      <path d="M14 7l3 3" />
    </I>
  );
}

/** Horizontal sliders — overflow / display options menu. */
export function IconSliders({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="M4 6h10M18 6h2" />
      <circle cx="16" cy="6" r="2" />
      <path d="M4 12h4M12 12h8" />
      <circle cx="10" cy="12" r="2" />
      <path d="M4 18h13M21 18h-1" />
      <circle cx="19" cy="18" r="2" />
    </I>
  );
}

/** Check mark for menu toggles. */
export function IconCheck({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="M4.5 12.5 10 18 19.5 6.5" />
    </I>
  );
}

/** Chevron pointing right: a collapsible section's header turns it down when open. */
export function IconChevronRight({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="m9 6 6 6-6 6" />
    </I>
  );
}

/** Plus sign: zoom in. */
export function IconPlus({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="M12 5v14M5 12h14" />
    </I>
  );
}

/** Minus sign: zoom out. */
export function IconMinus({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="M5 12h14" />
    </I>
  );
}

/** Crosshair: centre the camera on the probe. */
export function IconCrosshair({ size }: { size?: number }) {
  return (
    <I size={size}>
      <circle cx="12" cy="12" r="6" />
      <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
    </I>
  );
}

/** Stacked sheets: the layers of the 3D navigator. */
export function IconLayers({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="m12 3 9 5-9 5-9-5 9-5Z" />
      <path d="m3 13 9 5 9-5" />
      <path d="m3 17.5 9 5 9-5" />
    </I>
  );
}

/** Play triangle: replay the cine loop. */
export function IconPlay({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="M8 5.5v13l10.5-6.5L8 5.5Z" />
    </I>
  );
}

/** Two bars: pause the cine loop. */
export function IconPause({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="M9 5.5v13M15 5.5v13" />
    </I>
  );
}

/** Head and shoulders: show or hide the 3D torso navigator. */
export function IconTorso({ size }: { size?: number }) {
  return (
    <I size={size}>
      <circle cx="12" cy="6.5" r="3" />
      <path d="M4.5 21v-3.5a5 5 0 0 1 5-5h5a5 5 0 0 1 5 5V21" />
    </I>
  );
}

/** Camera: store a still capture into the study gallery. */
export function IconCamera({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="M4 8.5A2 2 0 0 1 6 6.5h2l1.4-2h5.2l1.4 2H18a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-9Z" />
      <circle cx="12" cy="13" r="3.2" />
    </I>
  );
}

/** Printer: print or export clean clinical PDF report. */
export function IconPrinter({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="M7 9V4h10v5" />
      <rect x="4" y="9" width="16" height="8" rx="2" />
      <path d="M7 14h10v6H7v-6Z" />
    </I>
  );
}

/** Projector screen: 1-click classroom / presentation mode (clean interface + dark room). */
export function IconPresentation({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="M3 4h18" />
      <rect x="4.5" y="4" width="15" height="11" rx="1.5" />
      <path d="M12 15v5M8.5 20h7" />
    </I>
  );
}

/** Speaker on: live Doppler audio enabled. */
export function IconVolume({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="M4.5 9.5h3.5L12.5 6v12L8 14.5H4.5z" />
      <path d="M16 9a4.5 4.5 0 0 1 0 6" />
      <path d="M18.6 6.8a7.8 7.8 0 0 1 0 10.4" />
    </I>
  );
}

/** Speaker muted: live Doppler audio off. */
export function IconVolumeOff({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="M4.5 9.5h3.5L12.5 6v12L8 14.5H4.5z" />
      <path d="m16 9.5 4.5 5M20.5 9.5 16 14.5" />
    </I>
  );
}

/** Tag / anatomical callout icon: on-image 2D anatomy labels toggle. */
export function IconAnatomyTag({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="M4 12.5V5.5A1.5 1.5 0 0 1 5.5 4h7l7.2 7.2a1.6 1.6 0 0 1 0 2.3l-6.2 6.2a1.6 1.6 0 0 1-2.3 0L4 12.5Z" />
      <circle cx="8.5" cy="8.5" r="1.1" fill="currentColor" stroke="none" />
    </I>
  );
}

/** Teaching pointer arrow: draw ephemeral callout arrows on frozen ultrasound frames. */
export function IconArrowPointer({ size }: { size?: number }) {
  return (
    <I size={size}>
      <path d="M5 19 19 5" />
      <path d="M11 5h8v8" />
    </I>
  );
}
