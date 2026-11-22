import type { SegmentCoverage } from '@/simulator/view-recognition/segmentCoverage';

export const REASON_ES: Record<string, string> = {
  not_in_plane: 'fuera del plano',
  shadowed: 'en el plano, en sombra acústica',
  insufficient_extent: 'en el plano, corte demasiado pequeño o tangencial',
  insufficient_border_visibility: 'en el plano, borde endocárdico poco visible',
};

/** State of a segment in the current plane, in words (the colour never carries it alone). */
export function coverageText(c: SegmentCoverage | undefined): string {
  if (!c) return 'sin análisis del corte';
  if (c.assessable) return 'evaluable en este corte';
  return REASON_ES[c.reason ?? ''] ?? 'no evaluable';
}
