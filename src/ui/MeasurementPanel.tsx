import { useSimStore } from '@/app/store';
import { examClosed, modePolicy, showsCaseIdentity } from '@/app/modePolicy';
import { useShallow } from 'zustand/shallow';
import { loadCaseById } from '@/cases';
import { MEASUREMENT_SPECS, type MeasurementSpec } from '@/simulator/measurements/protocol';
import { modalityLabel } from '@/simulator/renderer/modality';
import { phaseTag } from './console/measureContext';

const GROUP_LABEL: Record<MeasurementSpec['group'], string> = {
  lv: 'Ventrículo izquierdo',
  'lvot-av': 'TSVI y válvula aórtica',
  diastole: 'Función diastólica',
  valves: 'Insuficiencias valvulares',
  right: 'Corazón derecho',
  atria: 'Aurículas',
};

/**
 * Measurement protocol panel (spec 16, 28): the case's required measurements first with progress,
 * then the rest of the catalogue by anatomical group with modality badges and completion counts.
 */
export function MeasurementPanel() {
  const s = useSimStore(
    useShallow((st) => ({
      activeMeasurementId: st.activeMeasurementId,
      caseId: st.caseId,
      measurements: st.measurements,
      modality: st.modality,
      mode: st.mode,
      examFinished: st.examFinished,
      setActiveMeasurement: st.setActiveMeasurement,
    })),
  );
  const caseDef = loadCaseById(s.caseId);
  // the measurements a case requires name its diagnosis: during an exam they stay in their groups (decision 234)
  const requiredIds = showsCaseIdentity(s.mode, s.examFinished)
    ? caseDef.requiredMeasurements.map((r) => r.measurementId)
    : [];
  const latestOf = (id: string) => {
    for (let i = s.measurements.length - 1; i >= 0; i--)
      if (s.measurements[i]!.measurementId === id) return s.measurements[i]!;
    return null;
  };
  // the technique grade names the view the image shows («Vista A2C: … se mide en A4C»): not during an exam (decision 154)
  const gradeTechnique = modePolicy(s.mode).showViewFeedback;
  const closed = examClosed(s);

  const row = (spec: MeasurementSpec) => {
    const m = latestOf(spec.id);
    const level =
      m?.technique && gradeTechnique
        ? m.technique.findings.some((f) => f.level === 'invalid')
          ? 'invalid'
          : m.technique.findings.some((f) => f.level === 'warn')
            ? 'warn'
            : 'ok'
        : null;
    const isActive = s.activeMeasurementId === spec.id;
    const modMatch = spec.modalities.includes(s.modality);
    const primaryMod = modalityLabel(spec.modalities[0] ?? '2d');
    const phase = phaseTag(spec.phase);
    const valueText = m
      ? `${m.value.toFixed(spec.kind === 'time' || spec.kind === 'volume' ? 0 : spec.kind === 'vti' || spec.units === 'cm/s' ? 1 : 2)} ${spec.units}`
      : '—';
    return (
      <div
        key={spec.id}
        className={`protocol-row ${isActive ? 'active' : ''}${m ? ' done' : ''}`}
        data-measurement={spec.id}
      >
        <div className="protocol-left">
          <button
            className={isActive ? 'active' : ''}
            onClick={() => s.setActiveMeasurement(isActive ? null : spec.id)}
            data-tip={spec.instruction}
            aria-label={`Medir ${spec.label}`}
            disabled={closed}
          >
            {spec.shortLabel}
          </button>
          <span
            className={`mod-chip${modMatch ? ' match' : ''}`}
            data-tip={`Modalidad: ${spec.modalities.map(modalityLabel).join(' / ')}${phase ? ` · Fase: ${phase}` : ''}`}
          >
            {primaryMod}
          </span>
        </div>
        <span className="protocol-value">
          <span className={m ? 'protocol-num' : 'protocol-empty'}>{valueText}</span>
          {level && (
            <span
              className={`pill ${level === 'ok' ? 'ok' : level === 'warn' ? 'warn' : 'bad'}`}
              data-tip={
                m?.technique?.findings
                  .filter((f) => f.level !== 'ok')
                  .map((f) => f.message)
                  .join(' ') || 'Técnica adecuada'
              }
            >
              {level === 'ok' ? 'ok' : level === 'warn' ? 'revisar' : 'inválida'}
            </span>
          )}
        </span>
      </div>
    );
  };

  const required = MEASUREMENT_SPECS.filter((m) => requiredIds.includes(m.id));
  const requiredDone = required.filter((spec) => latestOf(spec.id) !== null).length;
  const groups = (['lv', 'lvot-av', 'diastole', 'valves', 'right', 'atria'] as const).map((g) => {
    const items = MEASUREMENT_SPECS.filter((m) => m.group === g && !requiredIds.includes(m.id));
    const done = items.filter((spec) => latestOf(spec.id) !== null).length;
    return { g, items, done };
  });

  return (
    <div className="protocol">
      {required.length > 0 && (
        <>
          <div className="protocol-cap-row">
            <span className="protocol-cap">Requeridas por el caso</span>
            <span className={`protocol-progress${requiredDone === required.length ? ' complete' : ''}`}>
              {requiredDone}/{required.length}
            </span>
          </div>
          {required.map(row)}
        </>
      )}
      {groups.map(({ g, items, done }) =>
        items.length ? (
          <details key={g} className="protocol-group">
            <summary className="small protocol-group-head">
              <span>{GROUP_LABEL[g]}</span>
              {done > 0 && (
                <span className="protocol-group-count">
                  {done}/{items.length}
                </span>
              )}
            </summary>
            {items.map(row)}
          </details>
        ) : null,
      )}
    </div>
  );
}
