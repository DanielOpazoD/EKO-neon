import { useSimStore } from '@/app/store';
import { acquisitionLimits } from '@/simulator/core/acquisitionInput';
import type { SpectralModality } from '@/simulator/doppler/spectral/spectrum';
import { useShallow } from 'zustand/shallow';
import { IconMinus, IconPlus, IconVolume, IconVolumeOff } from './icons';
import {
  cycleRespiration,
  cycleSweepSpeed,
  quickBarGroupFor,
  RESPIRATION_SHORT_LABEL,
  stepColorBaseline,
  stepColorScale,
  stepSpectralBaseline,
  stepSpectralScale,
  toggleHdZoom,
} from './imageQuickBarModel';

/**
 * Contextual on-image quick controls (decision 187): surfaces the exact knobs a sonographer turns
 * most for the active modality right over the image so they never have to leave Medir:
 * - 2D / M-mode: Depth, Gain, Zoom 2×, Auto-Opt, Sweep (M-mode), Respiration
 * - Color / CMM: Nyquist Scale, Baseline, Color Gain, Depth, Zoom 2×, Sweep (CMM)
 * - PW / CW / TDI: Scale, Baseline, Spectral Gain, Invert, Sweep (25/50/100), Audio, Resp (PW)
 */
export function ImageQuickBar() {
  const s = useSimStore(
    useShallow((st) => ({
      autoOptimize: st.autoOptimize,
      color: st.color,
      cursorThetaRad: st.cursorThetaRad,
      frozen: st.frozen,
      gateDepthCm: st.gateDepthCm,
      modality: st.modality,
      patient: st.patient,
      setColor: st.setColor,
      setPatient: st.setPatient,
      setSettings: st.setSettings,
      setSpectral: st.setSpectral,
      settings: st.settings,
      spectral: st.spectral,
    })),
  );
  if (s.frozen) return null;

  const group = quickBarGroupFor(s.modality);
  const limits = acquisitionLimits(s);
  const respLabel = RESPIRATION_SHORT_LABEL[s.patient.respiration];
  const respHeld = s.patient.respiration !== 'free-breathing';
  const sweepSpeed = s.spectral.sweepSpeedMmPerS;

  if (group === 'color') {
    const base = s.color.baselineShiftMps;
    return (
      <div className="quick-bar" role="group" aria-label="Ajustes rápidos de Doppler color">
        <Stepper
          label="Escala"
          value={`±${s.color.scaleMps.toFixed(2)}`}
          less="Bajar escala color (Nyquist)"
          more="Subir escala color (Nyquist)"
          onLess={() =>
            s.setColor(stepColorScale(s.color.scaleMps, base, limits.color.maxScaleMps, -1))
          }
          onMore={() =>
            s.setColor(stepColorScale(s.color.scaleMps, base, limits.color.maxScaleMps, 1))
          }
        />
        <i className="quick-sep" aria-hidden="true" />
        <Stepper
          label="Base"
          value={`${base >= 0 ? '+' : '−'}${Math.abs(base).toFixed(2)}`}
          less="Bajar línea base color"
          more="Subir línea base color"
          onLess={() =>
            s.setColor({ baselineShiftMps: stepColorBaseline(s.color.scaleMps, base, -1) })
          }
          onMore={() =>
            s.setColor({ baselineShiftMps: stepColorBaseline(s.color.scaleMps, base, 1) })
          }
        />
        <i className="quick-sep" aria-hidden="true" />
        <Stepper
          label="Gan."
          value={`${s.color.gainDb > 0 ? '+' : ''}${s.color.gainDb} dB`}
          less="Menos ganancia color"
          more="Más ganancia color"
          onLess={() => s.setColor({ gainDb: Math.max(-20, s.color.gainDb - 2) })}
          onMore={() => s.setColor({ gainDb: Math.min(20, s.color.gainDb + 2) })}
        />
        <i className="quick-sep" aria-hidden="true" />
        <Stepper
          label="Prof."
          value={`${s.settings.depthCm} cm`}
          less="Menos profundidad"
          more="Más profundidad"
          onLess={() => s.setSettings({ depthCm: s.settings.depthCm - 1 })}
          onMore={() => s.setSettings({ depthCm: s.settings.depthCm + 1 })}
        />
        <i className="quick-sep" aria-hidden="true" />
        <button
          type="button"
          className={`quick-auto${s.settings.zoom > 1.05 ? ' on' : ''}`}
          aria-pressed={s.settings.zoom > 1.05}
          data-tip="Zoom HD focalizado (centra el foco y aumenta resolución para medir válvulas/PISA)"
          onClick={() => s.setSettings({ zoom: toggleHdZoom(s.settings.zoom) })}
        >
          {s.settings.zoom > 1.05 ? `HD ${s.settings.zoom.toFixed(1)}×` : 'Zoom 2×'}
        </button>
        {s.modality === 'cmm' && (
          <>
            <i className="quick-sep" aria-hidden="true" />
            <button
              type="button"
              className={`quick-auto quick-mono${sweepSpeed !== 50 ? ' on' : ''}`}
              aria-label={`Velocidad de barrido ${sweepSpeed} mm/s`}
              data-tip="Velocidad de barrido del modo M color (clic: 25 / 50 / 100 mm/s)"
              onClick={() => s.setSpectral({ sweepSpeedMmPerS: cycleSweepSpeed(sweepSpeed) })}
            >
              {sweepSpeed} mm/s
            </button>
          </>
        )}
      </div>
    );
  }

  if (group === 'spectral') {
    const mod = s.modality as SpectralModality;
    const isCw = mod === 'cw';
    const base = s.spectral.baselineShiftMps;
    const scaleText = isCw
      ? `±${s.spectral.scaleMps.toFixed(1)} m/s`
      : `±${Math.round(s.spectral.scaleMps * 100)}`;
    const baseText = isCw
      ? `${base >= 0 ? '+' : '−'}${Math.abs(base).toFixed(1)}`
      : `${base >= 0 ? '+' : '−'}${Math.round(Math.abs(base) * 100)}`;
    return (
      <div className="quick-bar" role="group" aria-label="Ajustes rápidos de Doppler espectral">
        <Stepper
          label="Escala"
          value={scaleText}
          less="Bajar escala espectral"
          more="Subir escala espectral"
          onLess={() =>
            s.setSpectral(
              stepSpectralScale(mod, s.spectral.scaleMps, base, limits.spectral.maxScaleMps, -1),
            )
          }
          onMore={() =>
            s.setSpectral(
              stepSpectralScale(mod, s.spectral.scaleMps, base, limits.spectral.maxScaleMps, 1),
            )
          }
        />
        <i className="quick-sep" aria-hidden="true" />
        <Stepper
          label="Base"
          value={baseText}
          less="Desplazar ventana hacia negativo (más margen abajo)"
          more="Desplazar ventana hacia positivo (más margen arriba)"
          onLess={() =>
            s.setSpectral({
              baselineShiftMps: stepSpectralBaseline(mod, s.spectral.scaleMps, base, -1),
            })
          }
          onMore={() =>
            s.setSpectral({
              baselineShiftMps: stepSpectralBaseline(mod, s.spectral.scaleMps, base, 1),
            })
          }
        />
        <i className="quick-sep" aria-hidden="true" />
        <Stepper
          label="Gan."
          value={`${s.spectral.gainDb > 0 ? '+' : ''}${s.spectral.gainDb} dB`}
          less="Menos ganancia espectral"
          more="Más ganancia espectral"
          onLess={() => s.setSpectral({ gainDb: Math.max(-20, s.spectral.gainDb - 2) })}
          onMore={() => s.setSpectral({ gainDb: Math.min(20, s.spectral.gainDb + 2) })}
        />
        <i className="quick-sep" aria-hidden="true" />
        <button
          type="button"
          className={`quick-auto${s.spectral.invert ? ' on' : ''}`}
          aria-pressed={s.spectral.invert}
          data-tip="Invertir polaridad del espectro"
          onClick={() => s.setSpectral({ invert: !s.spectral.invert })}
        >
          Inv
        </button>
        <i className="quick-sep" aria-hidden="true" />
        <button
          type="button"
          className={`quick-auto quick-mono${sweepSpeed !== 50 ? ' on' : ''}`}
          aria-label={`Velocidad de barrido ${sweepSpeed} mm/s`}
          data-tip="Velocidad de barrido (clic: 25 mm/s para FA/respiración, 50 estándar, 100 mm/s para TD/PHT)"
          onClick={() => s.setSpectral({ sweepSpeedMmPerS: cycleSweepSpeed(sweepSpeed) })}
        >
          {sweepSpeed} mm/s
        </button>
        <button
          type="button"
          className={`quick-auto quick-icon${s.spectral.audioOn ? ' on' : ''}`}
          aria-pressed={s.spectral.audioOn}
          aria-label={s.spectral.audioOn ? 'Silenciar audio Doppler' : 'Activar audio Doppler'}
          data-tip={
            s.spectral.audioOn
              ? 'Audio Doppler activo (clic para silenciar)'
              : 'Activar sonido Doppler en vivo (flujo laminar, soplo turbulento y clics)'
          }
          onClick={() => s.setSpectral({ audioOn: !s.spectral.audioOn })}
        >
          {s.spectral.audioOn ? <IconVolume size={13} /> : <IconVolumeOff size={13} />}
        </button>
        {mod === 'pw' && (
          <button
            type="button"
            className={`quick-auto${respHeld ? ' on' : ''}`}
            aria-label={`Respiración: ${respLabel}`}
            data-tip="Respiración del paciente (clic: Libre / Insp / Esp / Apnea — útil para variación respiratoria mitral/tricuspídea)"
            onClick={() => s.setPatient({ respiration: cycleRespiration(s.patient.respiration) })}
          >
            Resp: {respLabel}
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="quick-bar" role="group" aria-label="Profundidad, ganancia y adquisición rápida">
      <Stepper
        label="Profundidad"
        value={`${s.settings.depthCm} cm`}
        less="Menos profundidad"
        more="Más profundidad"
        onLess={() => s.setSettings({ depthCm: s.settings.depthCm - 1 })}
        onMore={() => s.setSettings({ depthCm: s.settings.depthCm + 1 })}
      />
      <i className="quick-sep" aria-hidden="true" />
      <Stepper
        label="Ganancia"
        value={`${s.settings.gainDb > 0 ? '+' : ''}${s.settings.gainDb} dB`}
        less="Menos ganancia"
        more="Más ganancia"
        onLess={() => s.setSettings({ gainDb: s.settings.gainDb - 2 })}
        onMore={() => s.setSettings({ gainDb: s.settings.gainDb + 2 })}
      />
      <i className="quick-sep" aria-hidden="true" />
      <button
        type="button"
        className={`quick-auto${s.settings.zoom > 1.05 ? ' on' : ''}`}
        aria-pressed={s.settings.zoom > 1.05}
        data-tip="Zoom HD focalizado (centra el foco a 2× y aumenta densidad de líneas)"
        onClick={() => s.setSettings({ zoom: toggleHdZoom(s.settings.zoom) })}
      >
        {s.settings.zoom > 1.05 ? `HD ${s.settings.zoom.toFixed(1)}×` : 'Zoom 2×'}
      </button>
      <button
        type="button"
        className="quick-auto"
        aria-label="Auto-optimizar ganancia y TGC"
        data-tip="Auto-Opt (iScan)"
        onClick={() => s.autoOptimize()}
      >
        Auto-Opt
      </button>
      {s.modality === 'm-mode' && (
        <>
          <i className="quick-sep" aria-hidden="true" />
          <button
            type="button"
            className={`quick-auto quick-mono${sweepSpeed !== 50 ? ' on' : ''}`}
            aria-label={`Velocidad de barrido ${sweepSpeed} mm/s`}
            data-tip="Velocidad de barrido del modo M (clic: 25 / 50 / 100 mm/s)"
            onClick={() => s.setSpectral({ sweepSpeedMmPerS: cycleSweepSpeed(sweepSpeed) })}
          >
            {sweepSpeed} mm/s
          </button>
        </>
      )}
      <i className="quick-sep" aria-hidden="true" />
      <button
        type="button"
        className={`quick-auto${respHeld ? ' on' : ''}`}
        aria-label={`Respiración: ${respLabel}`}
        data-tip="Respiración del paciente (clic: Libre / Insp / Esp / Apnea — útil para colapso de VCI / sniff y ventana pulmonar)"
        onClick={() => s.setPatient({ respiration: cycleRespiration(s.patient.respiration) })}
      >
        Resp: {respLabel}
      </button>
    </div>
  );
}

function Stepper(props: {
  label: string;
  value: string;
  less: string;
  more: string;
  onLess: () => void;
  onMore: () => void;
}) {
  return (
    <span className="quick-stepper">
      <span className="quick-label">{props.label}</span>
      <button
        type="button"
        className="icon-btn"
        aria-label={props.less}
        data-tip={props.less}
        onClick={props.onLess}
      >
        <IconMinus size={13} />
      </button>
      <span className="quick-value" aria-live="polite">
        {props.value}
      </span>
      <button
        type="button"
        className="icon-btn"
        aria-label={props.more}
        data-tip={props.more}
        onClick={props.onMore}
      >
        <IconPlus size={13} />
      </button>
    </span>
  );
}
