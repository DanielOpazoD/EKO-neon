import { EXAM_CASE_NAME, showsCaseIdentity } from '@/app/modePolicy';
import { useSimStore } from '@/app/store';
import { listCases } from '@/cases';
import { VIEW_TARGETS } from '@/simulator/windows/viewDefinitions';
import { useShallow } from 'zustand/shallow';
import { CaseCard } from '../CaseCard';
import { Section, Segmented, Slider } from '../controls';
import { PresetViews } from '../PresetViews';
import { useTip } from './useTip';

/** Streamlined «Vistas» tab: core canonical windows, patient context, and collapsible manual probe sliders. */
export function AcquireTab() {
  const s = useSimStore(
    useShallow((st) => ({
      caseId: st.caseId,
      examFinished: st.examFinished,
      loadCase: st.loadCase,
      mode: st.mode,
      patient: st.patient,
      probe: st.probe,
      resetProbe: st.resetProbe,
      setPatient: st.setPatient,
      setProbe: st.setProbe,
      setTargetView: st.setTargetView,
      targetViewId: st.targetViewId,
    })),
  );
  const tip = useTip();

  return (
    <>
      <PresetViews />
      {s.mode === 'guided' && (
        <Section title="Vista objetivo">
          <select
            aria-label="Vista objetivo"
            value={s.targetViewId ?? ''}
            onChange={(e) => s.setTargetView(e.target.value || null)}
          >
            <option value="">— libre —</option>
            {VIEW_TARGETS.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name}
              </option>
            ))}
          </select>
        </Section>
      )}
      <Section title="Paciente">
        <div className="row stack">
          <label>Caso</label>
          <select
            aria-label="Caso"
            value={s.caseId}
            onChange={(e) => s.loadCase(e.target.value)}
            disabled={s.mode === 'exam'}
          >
            {showsCaseIdentity(s.mode, s.examFinished) ? (
              listCases().map((c) => (
                <option key={c.id} value={c.id}>
                  {c.title}
                </option>
              ))
            ) : (
              // the titles name the diagnoses: during an exam the selector names none (decision 234)
              <option value={s.caseId}>{EXAM_CASE_NAME}</option>
            )}
          </select>
        </div>
        <div className="row">
          <label>Posición</label>
          <select
            aria-label="Posición del paciente"
            value={s.patient.position}
            onChange={(e) =>
              s.setPatient({ position: e.target.value as typeof s.patient.position })
            }
          >
            <option value="left-lateral">Decúbito lateral izq.</option>
            <option value="supine">Supino</option>
            <option value="subcostal-supine">Supino (subcostal)</option>
          </select>
        </div>
        <div className="row">
          <label>Respiración</label>
          <Segmented
            ariaLabel="Respiración"
            value={s.patient.respiration}
            onChange={(v) => s.setPatient({ respiration: v })}
            options={[
              { id: 'expiration', label: 'Esp' },
              { id: 'breath-hold', label: 'Apnea' },
              { id: 'inspiration', label: 'Insp' },
              { id: 'free-breathing', label: 'Libre' },
            ]}
          />
        </div>
        <details className="adv" open={s.mode === 'exam' || undefined}>
          <summary>Historia clínica</summary>
          <CaseCard />
        </details>
      </Section>

      <div className="section" data-tour="sonda">
        <details className="adv" style={{ marginTop: 0, borderTop: 0, paddingTop: 0 }}>
          <summary>Ajuste manual de sonda</summary>
          <div style={{ marginTop: 6 }}>
            <Slider
              label="Rotación"
              hint="Q E"
              value={s.probe.rotationDeg}
              min={-180}
              max={180}
              unit="°"
              onChange={(v) => s.setProbe({ rotationDeg: v })}
              title={tip('Gira la sonda sobre su eje largo (Q/E).')}
            />
            <Slider
              label="Tilt (abanico)"
              hint="⌥↑↓"
              value={s.probe.tiltDeg}
              min={-70}
              max={70}
              unit="°"
              onChange={(v) => s.setProbe({ tiltDeg: v })}
              title={tip('Inclina el plano hacia adelante/atrás (Alt+↑↓).')}
            />
            <Slider
              label="Rock"
              hint="⌥←→"
              value={s.probe.rockDeg}
              min={-60}
              max={60}
              unit="°"
              onChange={(v) => s.setProbe({ rockDeg: v })}
              title={tip('Angula dentro del plano (Alt+←→).')}
            />
            <Slider
              label="Presión"
              hint="W S"
              value={s.probe.pressure}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) => s.setProbe({ pressure: v })}
              title={tip('Poca presión = mal acoplamiento (dropout).')}
            />
            <div className="row end">
              <button type="button" className="ghost" onClick={() => s.resetProbe()}>
                Reiniciar sonda
              </button>
            </div>
          </div>
        </details>
      </div>
    </>
  );
}
