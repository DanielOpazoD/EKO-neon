import { useSimStore } from '@/app/store';
import { TINT_MAP_OPTIONS } from '@/simulator/renderer/postprocess/tintMap';
import { useShallow } from 'zustand/shallow';
import { Section, Segmented, Slider, Toggle } from '../controls';
import { formatMHz } from '../format';
import { activeBmodePresetId, BMODE_IMAGE_PRESETS } from './imagePresets';
import { useTip } from './useTip';

/** Streamlined 2D image controls inside «Ajustes»: essential knobs up front, everything else under «Avanzado». */
export function ImageTab() {
  const s = useSimStore(
    useShallow((st) => ({
      autoOptimize: st.autoOptimize,
      resetTgc: st.resetTgc,
      setSettings: st.setSettings,
      setTgc: st.setTgc,
      settings: st.settings,
    })),
  );
  const tip = useTip();
  const activePreset = activeBmodePresetId(s.settings);

  return (
    <Section title="Imagen 2D">
      <div className="preset-grid image-preset-grid" role="group" aria-label="Perfil de textura 2D">
        {BMODE_IMAGE_PRESETS.map((preset) => (
          <button
            key={preset.id}
            type="button"
            className={activePreset === preset.id ? 'active' : ''}
            aria-pressed={activePreset === preset.id}
            data-tip={tip(preset.description)}
            onClick={() => s.setSettings(preset.patch)}
          >
            <span>{preset.label}</span>
          </button>
        ))}
      </div>

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
      <div className="row">
        <label>Zoom HD</label>
        <Segmented
          ariaLabel="Zoom HD"
          value={s.settings.zoom >= 1.85 ? '2' : s.settings.zoom >= 1.25 ? '1.5' : '1'}
          onChange={(v) => s.setSettings({ zoom: Number(v) })}
          options={[
            { id: '1', label: '1×' },
            { id: '1.5', label: '1,5×' },
            { id: '2', label: '2×' },
          ]}
        />
      </div>

      <div className="tgc-head">
        <h4 className="subhead">TGC</h4>
        <div className="tgc-actions">
          <button
            type="button"
            className="tgc-btn tgc-btn-primary"
            onClick={() => s.autoOptimize()}
            data-tip={tip('Ajusta automáticamente ganancia y TGC por profundidad (iScan).')}
          >
            Auto-Opt
          </button>
          <button
            type="button"
            className="tgc-btn"
            onClick={() => s.resetTgc()}
            data-tip={tip('Restablece todas las bandas TGC a 0 dB.')}
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
        <summary>Avanzado</summary>
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
          title={tip('Reduce clutter y mejora bordes.')}
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
        />
        <Slider
          label="Zoom fino"
          value={s.settings.zoom}
          min={1}
          max={2.5}
          step={0.1}
          format={(v) => `${v.toFixed(1)}×`}
          onChange={(v) => s.setSettings({ zoom: v })}
        />
        <Toggle
          label="Invertir izq/der"
          value={s.settings.invertLR}
          onChange={(v) => s.setSettings({ invertLR: v })}
        />
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
        />
        <Slider
          label="Compounding espacial"
          value={s.settings.spatialCompound ?? 0}
          min={0}
          max={1}
          step={0.05}
          onChange={(v) => s.setSettings({ spatialCompound: v })}
        />
        <div className="row">
          <label>Mapa de grises</label>
          <Segmented
            ariaLabel="Mapa de grises"
            value={s.settings.grayMap}
            onChange={(v) => s.setSettings({ grayMap: v })}
            options={[
              { id: 'clinical', label: 'Clín' },
              { id: 'linear', label: 'Lin' },
              { id: 's-curve', label: 'S' },
              { id: 'high-contrast', label: 'Alto' },
            ]}
          />
        </div>
        <div className="row">
          <label>Tinte</label>
          <Segmented
            ariaLabel="Tinte clínico"
            value={s.settings.tintMap ?? 'gray'}
            onChange={(v) => s.setSettings({ tintMap: v })}
            options={[...TINT_MAP_OPTIONS]}
          />
        </div>
      </details>
    </Section>
  );
}
