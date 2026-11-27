import { useSimStore } from '@/app/store';
import { TINT_MAP_OPTIONS } from '@/simulator/renderer/postprocess/tintMap';
import { useShallow } from 'zustand/shallow';
import { Section, Segmented, Slider, Toggle } from '../controls';
import { formatMHz } from '../format';
import { useTip } from './useTip';

/** B-mode image controls: the frequent ones up front, the fine-tuning set under "Ajustes finos". */
export function ImageTab() {
  const s = useSimStore(
    useShallow((st) => ({
      autoOptimize: st.autoOptimize,
      frozen: st.frozen,
      resetTgc: st.resetTgc,
      setSettings: st.setSettings,
      setTgc: st.setTgc,
      settings: st.settings,
    })),
  );
  const tip = useTip();
  return (
    <>
      {s.frozen && (
        <p className="small" role="status">
          Imagen congelada: Zoom, Invertir izq/der y Tinte responden al instante; los demás ajustes
          se aplicarán al reanudar.
        </p>
      )}
      <Section title="Imagen 2D">
      <Slider
        label="Profundidad"
        tour="depth"
        hint="[ ]"
        value={s.settings.depthCm}
        min={6}
        max={30}
        unit=" cm"
        onChange={(v) => s.setSettings({ depthCm: v })}
        title={tip('Más profundidad = menos frame rate y resolución.')}
      />
      <Slider
        label="Ganancia"
        tour="gain"
        hint="− +"
        value={s.settings.gainDb}
        min={-30}
        max={30}
        unit=" dB"
        onChange={(v) => s.setSettings({ gainDb: v })}
        title={tip('Exceso: la sangre se aclara hacia el gris del miocardio y sube el ruido.')}
      />
      <Slider
        label="Frecuencia"
        value={s.settings.frequencyMHz}
        min={1.5}
        max={5}
        step={0.25}
        unit=" MHz"
        format={formatMHz}
        onChange={(v) => s.setSettings({ frequencyMHz: v })}
        title={tip('Mayor frecuencia = más resolución, menos penetración.')}
      />
      <Toggle
        label="Armónicos (THI)"
        value={s.settings.harmonics}
        onChange={(v) => s.setSettings({ harmonics: v })}
        title={tip('Reduce clutter y mejora bordes; algo menos de penetración.')}
      />
      <Slider
        label="Foco"
        value={s.settings.focusCm}
        min={2}
        max={s.settings.depthCm}
        step={0.5}
        unit=" cm"
        format={(v) => `${v.toFixed(1)} cm`}
        onChange={(v) => s.setSettings({ focusCm: v })}
        title={tip('Mejor resolución lateral cerca del foco.')}
      />
      <Slider
        label="Zoom HD"
        value={s.settings.zoom}
        min={1}
        max={2.5}
        step={0.1}
        format={(v) => `${v.toFixed(1)}×`}
        onChange={(v) => s.setSettings({ zoom: v })}
        title={tip('Write-Zoom focalizado: centra la zona focal y aumenta la densidad de líneas.')}
      />
      <div className="row">
        <label>Zoom rápido</label>
        <Segmented
          ariaLabel="Zoom rápido HD"
          value={
            s.settings.zoom >= 1.85 ? '2' : s.settings.zoom >= 1.25 ? '1.5' : '1'
          }
          onChange={(v) => s.setSettings({ zoom: Number(v) })}
          options={[
            { id: '1', label: '1×' },
            { id: '1.5', label: '1,5×' },
            { id: '2', label: '2× HD' },
          ]}
        />
      </div>
      <Toggle
        label="Invertir izq/der"
        value={s.settings.invertLR}
        onChange={(v) => s.setSettings({ invertLR: v })}
      />
      <div className="tgc-head">
        <h4 className="subhead">TGC</h4>
        <div className="tgc-actions">
          <button
            type="button"
            className="tgc-btn tgc-btn-primary"
            onClick={() => s.autoOptimize()}
            title={tip('Ajusta automáticamente ganancia y TGC por banda de profundidad (iScan).')}
          >
            Auto-Opt
          </button>
          <button
            type="button"
            className="tgc-btn"
            onClick={() => s.resetTgc()}
            title={tip('Restablece todas las bandas TGC a 0 dB.')}
          >
            Reset
          </button>
        </div>
      </div>
      <div className="tgc" aria-label="TGC por profundidad">
        {s.settings.tgcDb.map((db, i) => (
          <input
            key={i}
            type="range"
            aria-label={`TGC banda ${i + 1}`}
            min={-15}
            max={15}
            step={1}
            value={db}
            onChange={(e) => s.setTgc(i, Number(e.target.value))}
          />
        ))}
      </div>
      <details className="adv">
        <summary>Ajustes finos</summary>
        <Slider
          label="Rango dinámico"
          value={s.settings.dynamicRangeDb}
          min={30}
          max={90}
          unit=" dB"
          onChange={(v) => s.setSettings({ dynamicRangeDb: v })}
          title={tip('Menor RD = imagen más contrastada.')}
        />
        <Slider
          label="Sector"
          value={s.settings.sectorDeg}
          min={30}
          max={100}
          unit="°"
          onChange={(v) => s.setSettings({ sectorDeg: v })}
          title={tip('Sector más ancho = menos frame rate.')}
        />
        <div className="row">
          <label>Densidad de líneas</label>
          <Segmented
            ariaLabel="Densidad de líneas"
            value={s.settings.lineDensity}
            onChange={(v) => s.setSettings({ lineDensity: v })}
            options={[
              { id: 'low', label: 'Baja' },
              { id: 'medium', label: 'Media' },
              { id: 'high', label: 'Alta' },
            ]}
          />
        </div>
        <Slider
          label="Persistencia"
          value={s.settings.persistence}
          min={0}
          max={0.9}
          step={0.05}
          onChange={(v) => s.setSettings({ persistence: v })}
        />
        <Slider
          label="Realce de bordes"
          value={s.settings.edgeEnhance}
          min={0}
          max={1}
          step={0.1}
          onChange={(v) => s.setSettings({ edgeEnhance: v })}
        />
        <Slider
          label="Reducción speckle (SRI)"
          value={s.settings.sri ?? 0}
          min={0}
          max={1}
          step={0.05}
          onChange={(v) => s.setSettings({ sri: v })}
          title={tip(
            'Suavizado anisotrópico orientado a paredes (tipo XRES/SRI) sin borrar bordes endocárdicos.',
          )}
        />
        <Slider
          label="Compounding espacial"
          value={s.settings.spatialCompound ?? 0}
          min={0}
          max={1}
          step={0.05}
          onChange={(v) => s.setSettings({ spatialCompound: v })}
          title={tip(
            'Compounding multi-ángulo (tipo SonoCT): reduce el dropout de paredes laterales y suaviza sombras.',
          )}
        />
        <div className="row">
          <label>Mapa de grises</label>
          <Segmented
            ariaLabel="Mapa de grises"
            value={s.settings.grayMap}
            onChange={(v) => s.setSettings({ grayMap: v })}
            options={[
              {
                id: 'clinical',
                label: 'Clín',
                title:
                  'Curva exponencial en dB que expande los grises altos, medida en imágenes clínicas',
              },
              { id: 'linear', label: 'Lin' },
              { id: 's-curve', label: 'S' },
              { id: 'high-contrast', label: 'Alto' },
            ]}
          />
        </div>
        <div className="row">
          <label>Tinte (Chroma)</label>
          <Segmented
            ariaLabel="Tinte clínico"
            value={s.settings.tintMap ?? 'gray'}
            onChange={(v) => s.setSettings({ tintMap: v })}
            options={[...TINT_MAP_OPTIONS]}
          />
        </div>
      </details>
    </Section>
    </>
  );
}
