import { useSimStore } from '@/app/store';
import { modePolicy } from '@/app/modePolicy';
import { InfoTip } from './controls';

type Preset = { id: string; label: string; title: string };

/** The 8 core clinical windows arranged in a compact 4×2 grid. */
const CORE_VIEWS: Preset[] = [
  { id: 'plax', label: 'PLAX', title: 'Paraesternal eje largo' },
  { id: 'psax-av', label: 'PSAX', title: 'Paraesternal eje corto, válvula aórtica' },
  { id: 'a4c', label: 'A4C', title: 'Apical cuatro cámaras' },
  { id: 'a5c', label: 'A5C', title: 'Apical cinco cámaras' },
  { id: 'a2c', label: 'A2C', title: 'Apical dos cámaras' },
  { id: 'a3c', label: 'A3C', title: 'Apical tres cámaras (eje largo)' },
  { id: 'rv-focused', label: 'VD', title: 'Apical enfocada en ventrículo derecho' },
  { id: 'subcostal-ivc', label: 'VCI', title: 'Subcostal vena cava inferior' },
];

/** Less-frequent complementary windows folded under «Más vistas». */
const EXTRA_VIEWS: Preset[] = [
  { id: 'psax-mv', label: 'PSAX MV', title: 'Paraesternal eje corto, nivel mitral' },
  { id: 'psax-pm', label: 'PSAX PM', title: 'Paraesternal eje corto, músculos papilares' },
  { id: 'psax-apex', label: 'PSAX ápex', title: 'Paraesternal eje corto, nivel apical' },
  { id: 'subcostal-4c', label: 'SC 4C', title: 'Subcostal cuatro cámaras' },
  { id: 'suprasternal-arch', label: 'SSN', title: 'Supraesternal arco aórtico' },
];

const EXTRA_IDS = new Set(EXTRA_VIEWS.map((v) => v.id));

function PresetButton({ id, label, title }: Preset) {
  const disabled = useSimStore((s) => !modePolicy(s.mode).presetsEnabled);
  const active = useSimStore((s) => s.targetViewId === id);
  const start = useSimStore((s) => s.startPresetView);
  return (
    <button
      type="button"
      className={active ? 'active' : ''}
      disabled={disabled}
      data-tip={disabled ? undefined : title}
      onClick={() => start(id)}
      aria-pressed={active}
    >
      {label}
    </button>
  );
}

export function PresetViews() {
  const mode = useSimStore((s) => s.mode);
  const targetViewId = useSimStore((s) => s.targetViewId);
  const disabled = !modePolicy(mode).presetsEnabled;
  const extraActive = targetViewId !== null && EXTRA_IDS.has(targetViewId);

  return (
    <div className="section">
      <h4>
        Vistas
        {!disabled && (
          <InfoTip text="Mueve la sonda de forma continua hasta la ventana elegida; luego puedes afinarla manualmente." />
        )}
      </h4>
      {disabled && <div className="protocol-cap">No disponibles en modo examen</div>}
      <div className="preset-grid core">
        {CORE_VIEWS.map((p) => (
          <PresetButton key={p.id} {...p} />
        ))}
      </div>
      <details className="adv" open={extraActive || undefined}>
        <summary>Más vistas ({EXTRA_VIEWS.length})</summary>
        <div className="preset-grid secondary" style={{ marginTop: 6 }}>
          {EXTRA_VIEWS.map((p) => (
            <PresetButton key={p.id} {...p} />
          ))}
        </div>
      </details>
    </div>
  );
}
