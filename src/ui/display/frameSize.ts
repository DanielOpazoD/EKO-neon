/** The largest frame the simulator forms (px): its cost grows with the area. */
export const MAX_FRAME = { width: 1024, height: 820 } as const;
/** The smallest frame worth forming (px). */
export const MIN_FRAME = { width: 320, height: 240 } as const;

/**
 * The frame size for a display box (CSS px). Both canvases stretch over the box (width and height 100 %), so the frame
 * keeps the box's proportions: one scale for both axes, within MIN_FRAME…MAX_FRAME wherever the proportions allow.
 * Clamped axis by axis, a 1272×840 box (1920 px screen) drew a 1024×820 frame stretched 24 % more across than down: a
 * round short axis showed as an oval, the beam–flow angle drawn differed from the one quoted, and a centimetre across
 * the image was not a centimetre of depth.
 */
export function frameSizeForBox(
  boxWidth: number,
  boxHeight: number,
): { width: number; height: number } {
  const w = Math.round(boxWidth);
  const h = Math.round(boxHeight);
  if (!(w >= 1 && h >= 1)) return { ...MIN_FRAME };
  const up = Math.max(1, MIN_FRAME.width / w, MIN_FRAME.height / h);
  const k = Math.min(up, MAX_FRAME.width / w, MAX_FRAME.height / h);
  return {
    width: Math.min(MAX_FRAME.width, Math.round(w * k)),
    height: Math.min(MAX_FRAME.height, Math.round(h * k)),
  };
}
