import { imageAnatomyLabelsOn, imageSegmentsOn, useSimStore } from '@/app/store';
import { modePolicy } from '@/app/modePolicy';
import { IconAnatomyTag } from './icons';
import type { SegmentModelChoice } from './segmentMap';

/**
 * On-image teaching overlay toggles under the case title: LV 17/16 segments (decision 153) and
 * live 2D anatomical structure labels. Strictly hidden in exam mode, where both layers are forced
 * off regardless of persisted preferences. While both are off, fades in on hover like ImageQuickBar.
 */
export function SegmentImageToggle() {
  const mode = useSimStore((s) => s.mode);
  const segOn = useSimStore(imageSegmentsOn);
  const anatomyOn = useSimStore(imageAnatomyLabelsOn);
  const model = useSimStore((s) => s.ui.segmentModel);
  const setUi = useSimStore((s) => s.setUi);
  if (!modePolicy(mode).hintsEnabled) return null;
  const pick = (m: SegmentModelChoice) =>
    setUi({
      segmentModel: m,
      selectedSegment:
        m === 'LV_16' && useSimStore.getState().ui.selectedSegment === 17
          ? null
          : useSimStore.getState().ui.selectedSegment,
    });
  return (
    <div className={segOn || anatomyOn ? 'img-seg on' : 'img-seg'}>
      <button
        type="button"
        className={anatomyOn ? 'img-seg-toggle on' : 'img-seg-toggle'}
        aria-pressed={anatomyOn}
        onClick={() => setUi({ imageAnatomyLabels: !anatomyOn })}
        data-tip="Muestra etiquetas anatómicas (VI, AI, VD, Ao, VM, Ao desc…) directamente sobre la imagen 2D"
        data-tip-key="L"
      >
        <IconAnatomyTag size={13} />
        Anatomía
      </button>
      <button
        type="button"
        className={segOn ? 'img-seg-toggle on' : 'img-seg-toggle'}
        aria-pressed={segOn}
        onClick={() => setUi({ imageSegments: !segOn })}
        data-tip="Colorea en la imagen los segmentos del VI del tejido que corta el plano; pasar el ratón: nombre del segmento"
      >
        <i className="img-seg-icon" aria-hidden="true" />
        Segmentos VI
      </button>
      {segOn && (
        <div
          className="img-seg-model"
          role="group"
          aria-label="Modelo de segmentación de la imagen"
        >
          <button
            type="button"
            className={model === 'LV_AHA17' ? 'active' : ''}
            aria-pressed={model === 'LV_AHA17'}
            onClick={() => pick('LV_AHA17')}
            data-tip="AHA 17: el 17 es el casquete apical, sin cavidad"
          >
            17
          </button>
          <button
            type="button"
            className={model === 'LV_16' ? 'active' : ''}
            aria-pressed={model === 'LV_16'}
            onClick={() => pick('LV_16')}
            data-tip="Motilidad: 16 segmentos, los apicales cubren todo el ápex"
          >
            16
          </button>
        </div>
      )}
    </div>
  );
}
