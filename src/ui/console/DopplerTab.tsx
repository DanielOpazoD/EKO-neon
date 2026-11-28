import { examClosed } from '@/app/modePolicy';
import { useSimStore } from '@/app/store';
import { acquisitionLimits } from '@/simulator/core/acquisitionInput';
import { SPECTRAL_SCALE_RANGE, type SpectralModality } from '@/simulator/doppler/spectral/spectrum';
import {
  COLOR_PALETTE_DEFAULT,
  COLOR_PALETTE_OPTIONS,
  type ColorPaletteId,
} from '@/simulator/renderer/postprocess/colorMap';
import {
  STRIP_TINT_DEFAULT,
  STRIP_TINT_OPTIONS,
  type StripTintMapChoice,
} from '@/simulator/renderer/postprocess/tintMap';
import {
  hasGate,
  isSpectralModality,
  MODALITIES,
} from '@/simulator/renderer/modality';
import { useShallow } from 'zustand/shallow';
import { Section, Segmented, Slider, Toggle } from '../controls';
import { useTip } from './useTip';
import { wallFilterControl, wallFilterRequestFor, wallFilterText } from './wallFilterControl';

/** The colour wall filter's slider (`wallFilterControl`): shows the effective cutoff and sets the request. */
function WallFilterSlider(props: {
  requestMps: number;
  scaleMps: number;
  onChange: (v: number) => void;
  title?: string;
}) {
  const c = wallFilterControl(props.requestMps, props.scaleMps);
  return (
    <>
      <Slider
        label="Filtro de pared"
        value={c.cutMps}
        min={c.min}
        max={c.max}
        step={c.step}
        format={wallFilterText}
        onChange={(v) => props.onChange(wallFilterRequestFor(v, props.requestMps, props.scaleMps))}
        title={props.title}
      />
      {c.limited && (
        <p className="small" aria-live="polite">
          Límite de escala: {wallFilterText(c.cutMps)}
        </p>
      )}
    </>
  );
}

/**
 * Streamlined contextual Doppler & M-mode controls rendered inside «Ajustes» whenever modality !== '2d'.
 * Keeps only the 3–5 clinical knobs used during acquisition up front; everything secondary lives in «Avanzado».
 */
export function DopplerTab() {
  const s = useSimStore(
    useShallow((st) => ({
      color: st.color,
      colorBoxCustom: st.colorBoxCustom,
      cursorThetaRad: st.cursorThetaRad,
      gateDepthCm: st.gateDepthCm,
      modality: st.modality,
      resetColorBox: st.resetColorBox,
      setColor: st.setColor,
      setCursor: st.setCursor,
      setModality: st.setModality,
      setSpectral: st.setSpectral,
      settings: st.settings,
      spectral: st.spectral,
      closed: examClosed(st),
    })),
  );
  const tip = useTip();
  const mod = s.modality;
  const limits = acquisitionLimits(s);
  const prfLimitedSpectral =
    isSpectralModality(mod) &&
    mod !== 'cw' &&
    s.spectral.scaleMps >= limits.spectral.maxScaleMps - 0.015;

  if (mod === 'color') {
    return (
      <Section
        title="Color Doppler"
        tip={tip('Arrastra la caja para moverla; Shift+arrastrar la redimensiona.')}
      >
        <Slider
          label="Escala (Nyquist)"
          value={s.color.scaleMps}
          min={0.15}
          max={Math.min(1.2, limits.color.maxScaleMps)}
          step={0.01}
          unit=" m/s"
          format={(v) => `${v.toFixed(2)} m/s`}
          onChange={(v) => {
            const maxShift = Math.round(v * 80) / 100;
            const clampedBase = Math.max(-maxShift, Math.min(maxShift, s.color.baselineShiftMps));
            s.setColor({ scaleMps: v, baselineShiftMps: clampedBase });
          }}
        />
        <Slider
          label="Línea base"
          value={s.color.baselineShiftMps}
          min={-Math.round(s.color.scaleMps * 80) / 100}
          max={Math.round(s.color.scaleMps * 80) / 100}
          step={0.02}
          unit=" m/s"
          format={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)} m/s`}
          onChange={(v) => s.setColor({ baselineShiftMps: v })}
          title={tip('Desplaza la línea base en dirección del chorro (maniobra PISA).')}
        />
        <Slider
          label="Ganancia color"
          value={s.color.gainDb}
          min={-20}
          max={20}
          unit=" dB"
          onChange={(v) => s.setColor({ gainDb: v })}
        />
        <Toggle
          label="Invertir mapa"
          value={s.color.invert}
          onChange={(v) => s.setColor({ invert: v })}
        />
        {s.colorBoxCustom && (
          <div className="row end">
            <button
              type="button"
              className="ghost"
              onClick={() => s.resetColorBox()}
              disabled={s.closed}
            >
              Centrar caja
            </button>
          </div>
        )}
        <details className="adv">
          <summary>Avanzado</summary>
          <div className="row">
            <label>Paleta</label>
            <Segmented
              ariaLabel="Paleta Doppler color"
              value={s.color.palette ?? COLOR_PALETTE_DEFAULT}
              onChange={(v) => s.setColor({ palette: v as ColorPaletteId })}
              options={COLOR_PALETTE_OPTIONS.map((o) => ({ id: o.id, label: o.label }))}
            />
          </div>
          <Toggle
            label="Mapa de varianza"
            value={s.color.showVariance}
            onChange={(v) => s.setColor({ showVariance: v })}
          />
          <Slider
            label="Persistencia"
            value={s.color.persistence}
            min={0}
            max={0.9}
            step={0.05}
            onChange={(v) => s.setColor({ persistence: v })}
          />
          <Slider
            label="Suavizado"
            value={s.color.smoothing}
            min={0}
            max={1}
            step={0.05}
            format={(v) => `${Math.round(v * 100)}%`}
            onChange={(v) => s.setColor({ smoothing: v })}
          />
          <WallFilterSlider
            requestMps={s.color.wallFilterMps}
            scaleMps={limits.color.scaleMps}
            onChange={(v) => s.setColor({ wallFilterMps: v })}
          />
          <div className="row end">
            <button
              type="button"
              className="ghost"
              disabled={s.closed}
              onClick={() => s.setModality('cmm')}
              data-tip="Abrir Modo M Color (CMM) para velocidad de propagación (Vp)"
            >
              Abrir Modo M Color
            </button>
          </div>
        </details>
      </Section>
    );
  }

  if (isSpectralModality(mod)) {
    return (
      <Section
        title={MODALITIES[mod].name}
        tip={tip(`Clic o arrastrar sobre la imagen 2D mueve el cursor${mod !== 'cw' ? ' y el gate' : ''}.`)}
      >
        <Slider
          label="Escala"
          value={s.spectral.scaleMps}
          min={
            mod === 'cw'
              ? SPECTRAL_SCALE_RANGE.cw.min
              : Math.min(
                  SPECTRAL_SCALE_RANGE[mod as SpectralModality].min,
                  limits.spectral.maxScaleMps,
                )
          }
          max={
            mod === 'cw'
              ? SPECTRAL_SCALE_RANGE.cw.max
              : Math.min(
                  SPECTRAL_SCALE_RANGE[mod as SpectralModality].max,
                  limits.spectral.maxScaleMps,
                )
          }
          step={mod === 'tdi' ? 0.02 : 0.05}
          format={(v) => (mod === 'cw' ? `±${v.toFixed(2)} m/s` : `±${Math.round(v * 100)} cm/s`)}
          onChange={(v) => {
            const maxShift = Math.round(v * 80) / 100;
            const clampedBase = Math.max(
              -maxShift,
              Math.min(maxShift, s.spectral.baselineShiftMps),
            );
            s.setSpectral({ scaleMps: v, baselineShiftMps: clampedBase });
          }}
        />
        {prfLimitedSpectral && (
          <p className="small" role="status">
            Límite PRF por profundidad (vN máx ±{Math.round(limits.spectral.maxScaleMps * 100)} cm/s)
          </p>
        )}
        <Slider
          label="Línea base"
          value={s.spectral.baselineShiftMps}
          min={-Math.max(0.02, Math.round(s.spectral.scaleMps * 80) / 100)}
          max={Math.max(0.02, Math.round(s.spectral.scaleMps * 80) / 100)}
          step={mod === 'tdi' ? 0.01 : mod === 'pw' ? 0.05 : 0.1}
          format={(v) => (mod === 'cw' ? `${v.toFixed(2)} m/s` : `${Math.round(v * 100)} cm/s`)}
          onChange={(v) => s.setSpectral({ baselineShiftMps: v })}
        />
        <Slider
          label="Ganancia"
          value={s.spectral.gainDb}
          min={-20}
          max={20}
          unit=" dB"
          onChange={(v) => s.setSpectral({ gainDb: v })}
        />
        <Slider
          label="Rechazo"
          value={s.spectral.reject ?? 0}
          min={0}
          max={0.5}
          step={0.02}
          format={(v) => `${Math.round(v * 100)}%`}
          onChange={(v) => s.setSpectral({ reject: v })}
        />
        <Toggle
          label="Invertir espectro"
          value={s.spectral.invert}
          onChange={(v) => s.setSpectral({ invert: v })}
        />
        <Toggle
          label="Audio Doppler"
          value={s.spectral.audioOn}
          onChange={(v) => s.setSpectral({ audioOn: v })}
        />
        {s.spectral.audioOn && (
          <Slider
            label="Volumen"
            value={s.spectral.volume}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => s.setSpectral({ volume: v })}
          />
        )}
        <details className="adv">
          <summary>Avanzado</summary>
          {hasGate(mod) && (
            <>
              <Slider
                label="Tamaño de gate"
                value={s.spectral.gateLengthCm}
                min={0.1}
                max={1.5}
                step={0.05}
                format={(v) => `${(v * 10).toFixed(0)} mm`}
                onChange={(v) => s.setSpectral({ gateLengthCm: v })}
              />
              <Slider
                label="Profundidad gate"
                value={s.gateDepthCm}
                min={1}
                max={s.settings.depthCm}
                step={0.1}
                format={(v) => `${v.toFixed(1)} cm`}
                onChange={(v) => s.setCursor(s.cursorThetaRad, v)}
              />
            </>
          )}
          <Slider
            label="Ángulo del cursor"
            value={(s.cursorThetaRad * 180) / Math.PI}
            min={-s.settings.sectorDeg / 2}
            max={s.settings.sectorDeg / 2}
            unit="°"
            onChange={(v) => s.setCursor((v * Math.PI) / 180)}
          />
          <Slider
            label="Filtro de pared"
            value={s.spectral.wallFilterMps}
            min={0}
            max={0.4}
            step={0.01}
            format={(v) => `${Math.round(v * 100)} cm/s`}
            onChange={(v) => s.setSpectral({ wallFilterMps: v })}
          />
          <div className="row">
            <label>Barrido</label>
            <Segmented
              ariaLabel="Velocidad de barrido"
              value={String(s.spectral.sweepSpeedMmPerS)}
              onChange={(v) => s.setSpectral({ sweepSpeedMmPerS: Number(v) })}
              options={[
                { id: '25', label: '25' },
                { id: '50', label: '50' },
                { id: '100', label: '100' },
              ]}
            />
          </div>
          <div className="row">
            <label>Tinte de tira</label>
            <Segmented
              ariaLabel="Tinte independiente de la tira espectral"
              value={s.spectral.stripTintMap ?? STRIP_TINT_DEFAULT}
              onChange={(v) => s.setSpectral({ stripTintMap: v as StripTintMapChoice })}
              options={STRIP_TINT_OPTIONS.map((o) => ({ id: o.id, label: o.label }))}
            />
          </div>
        </details>
      </Section>
    );
  }

  if (mod === 'm-mode') {
    return (
      <Section title="Modo M">
        <div className="row">
          <label>Barrido</label>
          <Segmented
            ariaLabel="Velocidad de barrido"
            value={String(s.spectral.sweepSpeedMmPerS)}
            onChange={(v) => s.setSpectral({ sweepSpeedMmPerS: Number(v) })}
            options={[
              { id: '25', label: '25' },
              { id: '50', label: '50' },
              { id: '100', label: '100' },
            ]}
          />
        </div>
        <Slider
          label="Ángulo del cursor"
          value={(s.cursorThetaRad * 180) / Math.PI}
          min={-s.settings.sectorDeg / 2}
          max={s.settings.sectorDeg / 2}
          unit="°"
          onChange={(v) => s.setCursor((v * Math.PI) / 180)}
        />
        <div className="row end">
          <button
            type="button"
            className="ghost"
            disabled={s.closed}
            onClick={() => s.setModality('cmm')}
            data-tip="Activar Modo M Color (CMM) para velocidad de propagación (Vp)"
          >
            Modo M Color (CMM)
          </button>
        </div>
        <details className="adv">
          <summary>Avanzado</summary>
          <div className="row">
            <label>Tinte de tira</label>
            <Segmented
              ariaLabel="Tinte independiente de la tira Modo M"
              value={s.spectral.stripTintMap ?? STRIP_TINT_DEFAULT}
              onChange={(v) => s.setSpectral({ stripTintMap: v as StripTintMapChoice })}
              options={STRIP_TINT_OPTIONS.map((o) => ({ id: o.id, label: o.label }))}
            />
          </div>
        </details>
      </Section>
    );
  }

  if (mod === 'cmm') {
    return (
      <Section
        title="Modo M Color"
        tip={tip('Desplaza la línea base para medir la velocidad de propagación (Vp) del llenado mitral.')}
      >
        <Slider
          label="Escala (Nyquist)"
          value={s.color.scaleMps}
          min={0.15}
          max={Math.min(1.2, limits.color.maxScaleMps)}
          step={0.01}
          unit=" m/s"
          format={(v) => `${v.toFixed(2)} m/s`}
          onChange={(v) => {
            const maxShift = Math.round(v * 80) / 100;
            const clampedBase = Math.max(-maxShift, Math.min(maxShift, s.color.baselineShiftMps));
            s.setColor({ scaleMps: v, baselineShiftMps: clampedBase });
          }}
        />
        <Slider
          label="Línea base"
          value={s.color.baselineShiftMps}
          min={-Math.round(s.color.scaleMps * 80) / 100}
          max={Math.round(s.color.scaleMps * 80) / 100}
          step={0.02}
          unit=" m/s"
          format={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)} m/s`}
          onChange={(v) => s.setColor({ baselineShiftMps: v })}
        />
        <Slider
          label="Ganancia color"
          value={s.color.gainDb}
          min={-20}
          max={20}
          unit=" dB"
          onChange={(v) => s.setColor({ gainDb: v })}
        />
        <div className="row">
          <label>Barrido</label>
          <Segmented
            ariaLabel="Velocidad de barrido"
            value={String(s.spectral.sweepSpeedMmPerS)}
            onChange={(v) => s.setSpectral({ sweepSpeedMmPerS: Number(v) })}
            options={[
              { id: '25', label: '25' },
              { id: '50', label: '50' },
              { id: '100', label: '100' },
            ]}
          />
        </div>
        <details className="adv">
          <summary>Avanzado</summary>
          <Slider
            label="Ángulo del cursor"
            value={(s.cursorThetaRad * 180) / Math.PI}
            min={-s.settings.sectorDeg / 2}
            max={s.settings.sectorDeg / 2}
            unit="°"
            onChange={(v) => s.setCursor((v * Math.PI) / 180)}
          />
          <div className="row">
            <label>Paleta</label>
            <Segmented
              ariaLabel="Paleta Modo M Color"
              value={s.color.palette ?? COLOR_PALETTE_DEFAULT}
              onChange={(v) => s.setColor({ palette: v as ColorPaletteId })}
              options={COLOR_PALETTE_OPTIONS.map((o) => ({ id: o.id, label: o.label }))}
            />
          </div>
          <div className="row">
            <label>Tinte de tira</label>
            <Segmented
              ariaLabel="Tinte independiente de la tira Modo M Color"
              value={s.spectral.stripTintMap ?? STRIP_TINT_DEFAULT}
              onChange={(v) => s.setSpectral({ stripTintMap: v as StripTintMapChoice })}
              options={STRIP_TINT_OPTIONS.map((o) => ({ id: o.id, label: o.label }))}
            />
          </div>
          <Toggle
            label="Mapa de varianza"
            value={s.color.showVariance}
            onChange={(v) => s.setColor({ showVariance: v })}
          />
          <Toggle
            label="Invertir mapa"
            value={s.color.invert}
            onChange={(v) => s.setColor({ invert: v })}
          />
          <WallFilterSlider
            requestMps={s.color.wallFilterMps}
            scaleMps={limits.color.scaleMps}
            onChange={(v) => s.setColor({ wallFilterMps: v })}
          />
        </details>
      </Section>
    );
  }

  return null;
}
