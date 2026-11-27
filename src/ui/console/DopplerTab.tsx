import { examClosed, modePolicy } from '@/app/modePolicy';
import { useSimStore } from '@/app/store';
import { useEffect, useState } from 'react';
import { acquisitionLimits } from '@/simulator/core/acquisitionInput';
import { SPECTRAL_SCALE_RANGE, type SpectralModality } from '@/simulator/doppler/spectral/spectrum';
import type { DopplerTargetId } from '@/simulator/measurements/autoDopplerTarget';
import {
  COLOR_PALETTE_DEFAULT,
  COLOR_PALETTE_OPTIONS,
  type ColorPaletteId,
} from '@/simulator/renderer/postprocess/colorMap';
import {
  TINT_DEFAULT,
  TINT_MAP_OPTIONS,
  type TintMapId,
} from '@/simulator/renderer/postprocess/tintMap';
import {
  hasGate,
  isSpectralModality,
  isStripModality,
  MODALITIES,
} from '@/simulator/renderer/modality';
import { useShallow } from 'zustand/shallow';
import { Section, Segmented, Slider, Toggle } from '../controls';
import { IconDoppler } from '../icons';
import { runAutoDopplerTarget } from '../display/measurementTools';
import { useTip } from './useTip';
import { wallFilterControl, wallFilterRequestFor, wallFilterText } from './wallFilterControl';

/** The colour wall filter's slider (`wallFilterControl`): it shows the cutoff that filters and sets the request. */
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
          Corte pedido {wallFilterText(props.requestMps)}: la escala lo limita a {wallFilterText(c.cutMps)}.
        </p>
      )}
    </>
  );
}

/** The contextual Doppler family, or an empty state with quick activators while in 2D. */
export function DopplerTab() {
  const s = useSimStore(
    useShallow((st) => ({
      color: st.color,
      colorBoxCustom: st.colorBoxCustom,
      frozen: st.frozen,
      cursorThetaRad: st.cursorThetaRad,
      gateDepthCm: st.gateDepthCm,
      modality: st.modality,
      mode: st.mode,
      resetColorBox: st.resetColorBox,
      setColor: st.setColor,
      setCursor: st.setCursor,
      setModality: st.setModality,
      setSettings: st.setSettings,
      setSpectral: st.setSpectral,
      setUi: st.setUi,
      settings: st.settings,
      spectral: st.spectral,
      // a finished exam keeps the image it was finished on (`examClosed`)
      closed: examClosed(st),
    })),
  );
  const tip = useTip();
  const [gateHint, setGateHint] = useState<string | null>(null);
  const mod = s.modality;
  const limits = acquisitionLimits(s);
  const presetsEnabled = modePolicy(s.mode).presetsEnabled;

  useEffect(() => {
    setGateHint(null);
  }, [mod]);

  const snapTarget = async (target: DopplerTargetId) => {
    setGateHint(null);
    const res = await runAutoDopplerTarget(target);
    setGateHint(res.ok ? res.summary : res.message);
  };

  if (mod === '2d') {
    return (
      <div className="empty">
        <IconDoppler size={22} />
        <p>Activa una modalidad Doppler para ajustarla aquí.</p>
        <div className="quick">
          <button onClick={() => s.setModality('color')} disabled={s.closed}>Color</button>
          <button onClick={() => s.setModality('pw')} disabled={s.closed}>PW</button>
          <button onClick={() => s.setModality('cw')} disabled={s.closed}>CW</button>
          <button onClick={() => s.setModality('tdi')} disabled={s.closed}>TDI</button>
          <button onClick={() => s.setModality('m-mode')} disabled={s.closed}>M</button>
          <button onClick={() => s.setModality('cmm')} disabled={s.closed}>CMM</button>
        </div>
      </div>
    );
  }

  return (
    <>
      {s.frozen && (
        <p className="small" role="status">
          {mod === 'color' || mod === 'cmm'
            ? `${mod === 'cmm' ? 'Tira congelada' : 'Imagen congelada'}: Paleta, Varianza y Línea base responden al instante; los demás ajustes se aplicarán al reanudar.`
            : isSpectralModality(mod)
              ? 'Tira congelada: Rechazo y Tinte responden al instante; los demás ajustes se aplicarán al reanudar.'
              : `${isStripModality(mod) ? 'Tira congelada' : 'Imagen congelada'}: los ajustes de adquisición se aplicarán al reanudar.`}
        </p>
      )}
      {mod === 'color' && (
        <Section
          title="Color Doppler"
          tip={tip(
            'Arrastra la caja para moverla; Shift+arrastrar la redimensiona; clic fuera crea otra.',
          )}
        >
          <Slider
            label="Ganancia color"
            value={s.color.gainDb}
            min={-20}
            max={20}
            unit=" dB"
            onChange={(v) => s.setColor({ gainDb: v })}
            title={tip('Exceso de ganancia produce blooming sobre el tejido.')}
          />
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
            title={tip(
              'La profundidad y la frecuencia limitan la PRF y Nyquist. Una escala baja aumenta el aliasing.',
            )}
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
            title={tip(
              'Desplaza la línea base en dirección del chorro para ajustar la velocidad de aliasing (maniobra PISA).',
            )}
          />
          <p className="small" aria-live="polite">
            PRF {(limits.color.prfHz / 1000).toFixed(2)} kHz · límite{' '}
            {(limits.color.maxPrfHz / 1000).toFixed(2)} kHz
          </p>
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
            title={tip(
              'Filtro espacio-temporal adaptativo: reduce el parpadeo y suaviza el flujo laminar preservando el mosaico turbulento.',
            )}
          />
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
          <Toggle
            label="Invertir mapa"
            value={s.color.invert}
            onChange={(v) => s.setColor({ invert: v })}
          />
          <details className="adv">
            <summary>Ajustes finos</summary>
            <WallFilterSlider
              requestMps={s.color.wallFilterMps}
              scaleMps={limits.color.scaleMps}
              onChange={(v) => s.setColor({ wallFilterMps: v })}
              title={tip(
                'Velocidad de corte a −3 dB. Con 8 pulsos va de 0,11 a 0,32 × Nyquist: para ver flujo más lento, baja la escala.',
              )}
            />
            {s.colorBoxCustom && (
              <button
                className="ghost"
                onClick={() => s.resetColorBox()}
                disabled={s.closed}
                data-tip={tip('Encuadra de nuevo la caja de color sobre las estructuras de la vista actual.')}
              >
                Ajustar caja a la vista
              </button>
            )}
          </details>
        </Section>
      )}
      {isSpectralModality(mod) && (
        <Section
          title={MODALITIES[mod].name}
          tip={tip(
            `Clic o arrastrar sobre la imagen 2D mueve el cursor${mod !== 'cw' ? ' y el gate' : ''}.`,
          )}
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
            title={tip(
              mod !== 'cw'
                ? 'La profundidad del gate y la frecuencia limitan Nyquist. Si la velocidad lo supera, el espectro se pliega; desplazar la línea base no aumenta la PRF.'
                : 'CW no aliasa: integra todo el trayecto del haz.',
            )}
          />
          {mod !== 'cw' && (
            <p className="small" aria-live="polite">
              PRF {(limits.spectral.prfHz / 1000).toFixed(2)} /{' '}
              {(limits.spectral.maxPrfHz / 1000).toFixed(2)} kHz · vN máx ±
              {Math.round(limits.spectral.maxScaleMps * 100)} cm/s
              {s.spectral.scaleMps >= limits.spectral.maxScaleMps - 0.015
                ? ' (límite por profundidad)'
                : ''}
            </p>
          )}
          <Slider
            label="Ganancia espectral"
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
            title={tip(
              'Postprocesado de rechazo: limpia el piso de ruido térmico sin alterar el trazo ni la velocidad pico (funciona incluso congelado).',
            )}
          />
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
                title={tip('Gate grande = ensanchamiento espectral.')}
              />
              <Slider
                label="Profundidad del gate"
                value={s.gateDepthCm}
                min={1}
                max={s.settings.depthCm}
                step={0.1}
                format={(v) => `${v.toFixed(1)} cm`}
                onChange={(v) => s.setCursor(s.cursorThetaRad, v)}
              />
              {mod === 'pw' && presetsEnabled && !s.closed && (
                <div className="doppler-quick-row">
                  <button
                    type="button"
                    className="btn-primary-soft"
                    onClick={() => void snapTarget('lvot')}
                    data-tip={tip('Detecta el TSVI en A5C/A3C y sitúa el gate PW ~0,5 cm proximal a la válvula aórtica.')}
                  >
                    Gate TSVI
                  </button>
                  <button
                    type="button"
                    className="ghost"
                    onClick={() => void snapTarget('mitral-inflow')}
                    data-tip={tip('Detecta las puntas mitrales en A4C y sitúa el gate PW para flujo E/A.')}
                  >
                    Gate Mitral
                  </button>
                </div>
              )}
              {mod === 'tdi' && presetsEnabled && !s.closed && (
                <div className="doppler-quick-row">
                  <button
                    type="button"
                    className="btn-primary-soft"
                    onClick={() => void snapTarget('tdi-septal')}
                    data-tip={tip('Sitúa el gate TDI en el anillo mitral septal en A4C para medir e′ septal.')}
                  >
                    Anillo septal
                  </button>
                  <button
                    type="button"
                    className="ghost"
                    onClick={() => void snapTarget('tdi-lateral')}
                    data-tip={tip('Sitúa el gate TDI en el anillo mitral lateral en A4C para medir e′ lateral.')}
                  >
                    Anillo lateral
                  </button>
                </div>
              )}
            </>
          )}
          {mod === 'cw' && presetsEnabled && !s.closed && (
            <div className="doppler-quick-row">
              <button
                type="button"
                className="btn-primary-soft"
                onClick={() => void snapTarget('tr-cw')}
                data-tip={tip('Alinea el cursor CW a través de la válvula tricúspide para Vmax IT.')}
              >
                Cursor IT
              </button>
            </div>
          )}
          {gateHint && (
            <p className="small" role="status">
              {gateHint}
            </p>
          )}
          <Slider
            label="Ángulo del cursor"
            value={(s.cursorThetaRad * 180) / Math.PI}
            min={-s.settings.sectorDeg / 2}
            max={s.settings.sectorDeg / 2}
            unit="°"
            onChange={(v) => s.setCursor((v * Math.PI) / 180)}
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
          <Slider
            label="Volumen"
            value={s.spectral.volume}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => s.setSpectral({ volume: v })}
            disabled={!s.spectral.audioOn}
          />
          <details className="adv">
            <summary>Ajustes finos</summary>
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
              label="Filtro de pared"
              value={s.spectral.wallFilterMps}
              min={0}
              max={0.4}
              step={0.01}
              format={(v) => `${Math.round(v * 100)} cm/s`}
              onChange={(v) => s.setSpectral({ wallFilterMps: v })}
            />
            <div className="row">
              <label>Velocidad de barrido</label>
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
              <label>Tinte</label>
              <Segmented
                ariaLabel="Tinte espectral"
                value={s.settings.tintMap ?? TINT_DEFAULT}
                onChange={(v) => s.setSettings({ tintMap: v as TintMapId })}
                options={TINT_MAP_OPTIONS.map((o) => ({ id: o.id, label: o.label }))}
              />
            </div>
          </details>
        </Section>
      )}
      {mod === 'm-mode' && (
        <Section title="Modo M">
          <div className="row">
            <label>Velocidad de barrido</label>
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
          {presetsEnabled && !s.closed && (
            <div className="doppler-quick-row">
              <button
                type="button"
                className="btn-primary-soft"
                onClick={() => void snapTarget('tapse')}
                data-tip={tip('Alinea el cursor de Modo M sobre el anillo tricuspídeo lateral en A4C o vista enfocada en VD.')}
              >
                Cursor TAPSE
              </button>
            </div>
          )}
          {gateHint && (
            <p className="small" role="status">
              {gateHint}
            </p>
          )}
        </Section>
      )}
      {mod === 'cmm' && (
        <Section
          title="Modo M Color"
          tip={tip(
            'Desplaza la línea base para medir la velocidad de propagación (Vp) del llenado mitral.',
          )}
        >
          <div className="row">
            <label>Velocidad de barrido</label>
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
          <Slider
            label="Ganancia color"
            value={s.color.gainDb}
            min={-20}
            max={20}
            unit=" dB"
            onChange={(v) => s.setColor({ gainDb: v })}
          />
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
          <p className="small" aria-live="polite">
            PRF {(limits.color.prfHz / 1000).toFixed(2)} kHz · límite{' '}
            {(limits.color.maxPrfHz / 1000).toFixed(2)} kHz
          </p>
          <div className="row">
            <label>Paleta</label>
            <Segmented
              ariaLabel="Paleta Modo M Color"
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
          <Toggle
            label="Invertir mapa"
            value={s.color.invert}
            onChange={(v) => s.setColor({ invert: v })}
          />
          <details className="adv">
            <summary>Ajustes finos</summary>
            <WallFilterSlider
              requestMps={s.color.wallFilterMps}
              scaleMps={limits.color.scaleMps}
              onChange={(v) => s.setColor({ wallFilterMps: v })}
              title={tip(
                'Velocidad de corte a −3 dB del filtro de regresión (8 pulsos, mismo estimador Kasai que el color 2D). Va de 0,11 a 0,32 × Nyquist.',
              )}
            />
          </details>
        </Section>
      )}
    </>
  );
}
