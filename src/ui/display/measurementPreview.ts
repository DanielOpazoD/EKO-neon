import { rvVolumeFromAreaMl } from '@/clinical/formulas';
import type { SimOutput } from '@/simulator/core/protocol';
import { polygonAreaCm2 } from '@/simulator/measurements/autoVolume';
import type { MeasurementTool } from '@/simulator/measurements/protocol';
import { discProfileFromContour, volumeFromProfileMl } from '@/simulator/measurements/simpson';
import { MODALITIES } from '@/simulator/renderer/modality';
import type { ImagingModality } from '@/simulator/renderer/types';
import {
  computeDecelerationTime,
  computeLinearCm,
  computeMModeExcursionCm,
  isPhtMeasurementId,
  pressureHalfTimeFromDtMs,
  stripVelocityAtY,
  summarizeManualVti,
  toolActsAt,
} from './measurementTools';

export interface LivePreview {
  /** Preview shape rendered on the overlay canvas in cyan. */
  shape: 'line' | 'ray' | 'guide' | 'span' | 'polyline' | 'contour';
  /** Geometry vertices for the preview shape. */
  points: { x: number; y: number }[];
  /** Live clinical readout chip drawn near the cursor. */
  label: string;
  /** Anchor pixel for the live readout chip. */
  anchor: { x: number; y: number };
}

/**
 * Pure real-time rubber-band preview & clinical readout while the pointer moves over a frozen
 * ultrasound display with an armed measurement tool. Uses the exact same clinical formulas as
 * `measurementTools.ts` so the live readout matches the committed value.
 */
export function computeLivePreview(
  tool: MeasurementTool | 'none',
  activeMeasurementId: string | null,
  pending: readonly { x: number; y: number }[],
  cursor: { x: number; y: number } | null,
  hud: Pick<SimOutput, 'frozen' | 'sector' | 'strip' | 'acquisition'>,
  liveModality: ImagingModality,
): LivePreview | null {
  if (!hud.frozen || !cursor || tool === 'none') return null;
  if (!toolActsAt(tool, cursor, hud)) return null;

  const m = hud.sector;
  const strip = hud.strip;
  const modality = hud.acquisition?.modality ?? liveModality;

  if (tool === 'caliper') {
    if (pending.length !== 1) return null;
    const a = pending[0]!;
    const cm = computeLinearCm(a, cursor, m.pxPerCm);
    return {
      shape: 'line',
      points: [a, cursor],
      label: `${cm.toFixed(2)} cm`,
      anchor: cursor,
    };
  }

  if (tool === 'tapse') {
    if (pending.length !== 1) return null;
    const a = pending[0]!;
    const cm = computeMModeExcursionCm(a, cursor, strip);
    return {
      shape: 'line',
      points: [a, { x: a.x, y: cursor.y }],
      label: `${cm.toFixed(2)} cm`,
      anchor: cursor,
    };
  }

  if (tool === 'time') {
    if (pending.length !== 1) return null;
    const a = pending[0]!;
    const ms = Math.abs(cursor.x - a.x) * strip.secondsPerColumn * 1000;
    return {
      shape: 'line',
      points: [a, { x: cursor.x, y: a.y }],
      label: `${Math.round(ms)} ms`,
      anchor: cursor,
    };
  }

  if (tool === 'velocity') {
    const vMps = Math.abs(stripVelocityAtY(cursor.y, strip));
    const units = MODALITIES[modality].velocityUnits;
    const label =
      units === 'cm/s'
        ? `${Math.round(vMps * 100)} cm/s`
        : `${vMps.toFixed(2)} m/s · ΔP ${(4 * vMps * vMps).toFixed(0)} mmHg`;
    return {
      shape: 'guide',
      points: [
        { x: strip.x, y: cursor.y },
        { x: strip.x + strip.width, y: cursor.y },
      ],
      label,
      anchor: cursor,
    };
  }

  if (tool === 'slope') {
    if (pending.length === 0) {
      const vMps = Math.abs(stripVelocityAtY(cursor.y, strip));
      return {
        shape: 'guide',
        points: [
          { x: strip.x, y: cursor.y },
          { x: strip.x + strip.width, y: cursor.y },
        ],
        label: `Pico ${vMps.toFixed(2)} m/s`,
        anchor: cursor,
      };
    }
    if (pending.length === 1) {
      const a = pending[0]!;
      const dt = computeDecelerationTime(a, cursor, strip);
      if (!dt) {
        return {
          shape: 'line',
          points: [a, cursor],
          label: 'Marca un punto más bajo y a la derecha',
          anchor: cursor,
        };
      }
      const isPht = isPhtMeasurementId(activeMeasurementId);
      const val = isPht ? pressureHalfTimeFromDtMs(dt.dtMs) : dt.dtMs;
      return {
        shape: 'ray',
        points: [a, cursor, dt.intercept],
        label: `${isPht ? 'PHT' : 'TD'} ${Math.round(val)} ms`,
        anchor: cursor,
      };
    }
    return null;
  }

  if (tool === 'auto-vti') {
    if (pending.length !== 1) return null;
    const a = pending[0]!;
    const ms = Math.abs(cursor.x - a.x) * strip.secondsPerColumn * 1000;
    return {
      shape: 'span',
      points: [
        { x: a.x, y: strip.y },
        { x: cursor.x, y: strip.y + strip.height },
      ],
      label: `Ventana ${Math.round(ms)} ms`,
      anchor: cursor,
    };
  }

  if (tool === 'vti') {
    if (pending.length === 0) {
      const vMps = Math.abs(stripVelocityAtY(cursor.y, strip));
      return {
        shape: 'guide',
        points: [
          { x: strip.x, y: cursor.y },
          { x: strip.x + strip.width, y: cursor.y },
        ],
        label: `${vMps.toFixed(2)} m/s`,
        anchor: cursor,
      };
    }
    const pts = [...pending, cursor];
    const { summary } = summarizeManualVti(pts, strip);
    return {
      shape: 'polyline',
      points: [pending[pending.length - 1]!, cursor],
      label: `VTI ${summary.vtiCm.toFixed(1)} cm (${pending.length} ptos)`,
      anchor: cursor,
    };
  }

  if (tool === 'simpson') {
    if (pending.length === 0) return null;
    const pts = [...pending, cursor];
    let label = `${pending.length} ${pending.length === 1 ? 'pto' : 'ptos'}`;
    if (pts.length >= 5) {
      const prof = discProfileFromContour(pts, m.pxPerCm);
      if (prof) {
        const isRv = activeMeasurementId === 'rv-edv' || activeMeasurementId === 'rv-esv';
        const area = polygonAreaCm2(pts, m.pxPerCm);
        const vol = isRv && area > 0 ? rvVolumeFromAreaMl(area) : volumeFromProfileMl(prof);
        label = isRv && area > 0 ? `${Math.round(vol)} mL · ${area.toFixed(1)} cm²` : `${Math.round(vol)} mL`;
      }
    }
    return {
      shape: 'contour',
      points: [pending[pending.length - 1]!, cursor, pending[0]!],
      label,
      anchor: cursor,
    };
  }

  return null;
}

/** Minimum pending points needed before a multi-point trace can be closed with Enter or "Finalizar". */
export function minPointsToFinish(tool: MeasurementTool | 'none'): number {
  if (tool === 'simpson') return 5;
  if (tool === 'vti') return 2;
  return Infinity;
}

/** Concise, user-centric step prompt for the on-canvas bar and active capture card. */
export function captureStepPrompt(
  tool: MeasurementTool | 'none',
  pendingCount: number,
  frozen: boolean,
  modalityOk: boolean,
  suggestedModalityLabel: string | null,
): string {
  if (!modalityOk && suggestedModalityLabel) {
    return `Cambia a ${suggestedModalityLabel} para capturar esta medida.`;
  }
  if (!frozen) {
    return 'Congela en el instante deseado (Espacio o clic sobre la imagen).';
  }
  switch (tool) {
    case 'caliper':
      return pendingCount === 0
        ? 'Paso 1/2: marca el primer extremo en la imagen 2D.'
        : 'Paso 2/2: marca el segundo extremo para completar.';
    case 'tapse':
      return pendingCount === 0
        ? 'Paso 1/2: marca el valle telediastólico en el modo M.'
        : 'Paso 2/2: marca el pico telesistólico del anillo.';
    case 'time':
      return pendingCount === 0
        ? 'Paso 1/2: marca el inicio del intervalo en la tira.'
        : 'Paso 2/2: marca el fin del intervalo.';
    case 'slope':
      return pendingCount === 0
        ? 'Paso 1/2: haz clic en el pico de la onda.'
        : 'Paso 2/2: haz clic más abajo sobre la pendiente de caída.';
    case 'velocity':
      return 'Haz clic sobre el pico del espectro Doppler.';
    case 'auto-vti':
      return pendingCount === 0
        ? 'Paso 1/2: marca el inicio del flujo en el espectro.'
        : 'Paso 2/2: marca el fin del flujo para trazar automáticamente.';
    case 'vti':
      return pendingCount < 2
        ? `Traza la envolvente punto a punto (${pendingCount} ptos).`
        : `${pendingCount} ptos · doble clic, Enter o «Finalizar» para cerrar.`;
    case 'simpson':
      return pendingCount < 5
        ? `Traza el endocardio de anillo a anillo (${pendingCount}/5 ptos mín.).`
        : `${pendingCount} ptos · doble clic, Enter o «Finalizar» para cerrar.`;
    default:
      return '';
  }
}
