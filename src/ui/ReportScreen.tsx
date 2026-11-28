import { useEffect, useRef, useState } from 'react';
import { useShallow } from 'zustand/shallow';
import { examSummaryOf, impressionCorrected, showsScoredTruth } from '@/app/examSummary';
import { EXAM_CASE_NAME, examClosed, showsCaseIdentity } from '@/app/modePolicy';
import { useSimStore } from '@/app/store';
import { formatCaptureCaption } from '@/app/studyCapture';
import { loadCaseById } from '@/cases';
import { formatClinical } from '@/clinical/reference-values';
import {
  expectedFindings,
  FINDINGS,
  getFinding,
  scoreImpression,
  type FindingDomain,
} from '@/education/impression';
import { buildEducationalReport } from '@/education/report';
import { scoreAcquisition } from '@/education/scoring/scoring';
import { modalityLabel } from '@/simulator/renderer/modality';
import { viewLabel } from '@/simulator/windows/viewLabels';
import { IconPrinter } from './icons';

const DOMAINS: { id: FindingDomain; label: string }[] = [
  { id: 'global', label: 'Global' },
  { id: 'lv', label: 'Ventrículo izquierdo' },
  { id: 'valves', label: 'Válvulas y TSVI' },
  { id: 'right', label: 'Corazón derecho' },
  { id: 'atria', label: 'Aurículas' },
  { id: 'pericardium', label: 'Pericardio' },
  { id: 'rhythm-diastole', label: 'Ritmo y diástole' },
];

function FinishExam({ onFinish }: { onFinish: () => void }) {
  const [asking, setAsking] = useState(false);
  const blankImpression = useSimStore((s) => s.impressionSelection.length === 0);
  if (!asking) {
    return (
      <button type="button" className="btn-primary-soft" onClick={() => setAsking(true)}>
        Finalizar examen y ver puntuación
      </button>
    );
  }
  return (
    <div className="finish-exam-confirm" role="alertdialog" aria-labelledby="finish-exam-q">
      <p id="finish-exam-q">
        ¿Finalizar el examen? No podrás adquirir, medir ni cambiar la impresión.
        {blankImpression && ' Sin impresión marcada (0/20 %).'}
      </p>
      <button type="button" className="btn-primary-soft" onClick={onFinish}>
        Sí, finalizar
      </button>{' '}
      <button type="button" className="ghost" onClick={() => setAsking(false)} autoFocus>
        Seguir
      </button>
    </div>
  );
}

/**
 * Concise, scannable clinical report:
 * - Top: 1-line patient header + Copy text / Print PDF actions.
 * - Body: derived KPI strip, clean 4-column measurements table, saved stills, and conclusion (when present).
 * - Bottom: collapsible «Evaluación docente e impresión» (collapsed in practice, open in exam mode).
 */
export function ReportScreen() {
  const s = useSimStore(
    useShallow((st) => ({
      captures: st.captures,
      caseId: st.caseId,
      closed: examClosed(st),
      examFinished: st.examFinished,
      examResult: st.examResult,
      finishExam: st.finishExam,
      checkImpression: st.checkImpression,
      impressionChecked: st.impressionChecked,
      impressionSelection: st.impressionSelection,
      measurements: st.measurements,
      mode: st.mode,
      redoImpression: st.redoImpression,
      removeCapture: st.removeCapture,
      setUi: st.setUi,
      toggleFinding: st.toggleFinding,
      truth: st.truth,
      viewProgress: st.viewProgress,
    })),
  );
  const [copied, setCopied] = useState<string | null>(null);
  const copyTimer = useRef<number | null>(null);
  useEffect(() => () => window.clearTimeout(copyTimer.current ?? undefined), []);

  const revealed = showsCaseIdentity(s.mode, s.examFinished);
  const report = buildEducationalReport(s.measurements, s.truth, !revealed);
  const caseDef = loadCaseById(s.caseId);
  const expected = s.truth ? expectedFindings(s.truth) : [];
  const impression = s.impressionSelection.length
    ? scoreImpression(s.impressionSelection, expected)
    : null;
  const summary = s.mode === 'exam' ? s.examResult : examSummaryOf(s);
  const acquisition = summary?.acquisition ?? scoreAcquisition(caseDef, s.viewProgress);
  const corrected = impressionCorrected(s);
  const isExam = s.mode === 'exam';
  const hasStudyData = report.rows.length > 0 || s.captures.length > 0;

  const rhythmText = !revealed
    ? 'Ritmo a informar'
    : caseDef.rhythm.type === 'atrial-fibrillation'
      ? 'FA'
      : 'Sinusal';
  const demoLine = [
    `${caseDef.demographics.ageYears} años`,
    caseDef.demographics.sexForReference === 'female' ? 'M' : 'H',
    s.truth ? `SC ${s.truth.bsaM2.toFixed(2).replace('.', ',')} m²` : null,
    `${rhythmText} ${caseDef.rhythm.heartRateBpm} lpm`,
  ]
    .filter(Boolean)
    .join(' · ');

  const copyText = async () => {
    const lines: string[] = [
      `Informe ETT — ${revealed ? caseDef.title : EXAM_CASE_NAME}`,
      demoLine,
      '',
    ];
    if (report.derived.length) {
      lines.push('Cálculos:');
      for (const d of report.derived) lines.push(`- ${d.label}: ${d.value}`);
      lines.push('');
    }
    if (report.rows.length) {
      lines.push('Mediciones:');
      for (const r of report.rows) {
        const flag = r.rangeFlag && r.rangeFlag !== 'normal' ? ` (${r.rangeFlag === 'high' ? 'alto' : 'bajo'})` : '';
        lines.push(`- ${r.label}: ${r.value}${flag}`);
      }
    }
    if (s.impressionSelection.length) {
      lines.push('', 'Conclusión:');
      for (const id of s.impressionSelection) lines.push(`- ${getFinding(id)?.label ?? id}`);
    }
    try {
      await navigator.clipboard.writeText(lines.join('\n'));
      setCopied('Copiado');
    } catch {
      setCopied('No se pudo copiar');
    }
    window.clearTimeout(copyTimer.current ?? undefined);
    copyTimer.current = window.setTimeout(() => setCopied(null), 2200);
  };

  return (
    <div className="screen report-clean">
      <div className="cr-top-row no-print">
        <div className="cr-top-actions" role="toolbar" aria-label="Acciones del informe">
          <button
            type="button"
            className="ghost"
            onClick={() => s.setUi({ screen: 'simulator', consoleTab: 'medir' })}
          >
            ← Simulador
          </button>
          <button
            type="button"
            className="ghost"
            disabled={!hasStudyData && !s.impressionSelection.length}
            onClick={() => void copyText()}
          >
            {copied ?? 'Copiar texto'}
          </button>
          <button
            type="button"
            className="btn-primary-soft cr-print-btn"
            onClick={() => window.print()}
            data-tip="Imprimir o guardar como PDF limpio"
          >
            <IconPrinter size={14} />
            PDF
          </button>
        </div>
        {isExam && !s.examFinished && <FinishExam onFinish={() => s.finishExam()} />}
      </div>

      <article className="clinical-report" aria-label="Informe ecocardiográfico">
        <header className="cr-head cr-head-compact">
          <div className="cr-title-row">
            <h2>{revealed ? caseDef.title : EXAM_CASE_NAME}</h2>
            <span className="small">{new Date().toLocaleDateString('es')}</span>
          </div>
          <p className="cr-demo-line">{demoLine}</p>
        </header>

        {!hasStudyData ? (
          <section className="cr-section cr-empty-box">
            <p className="small">Todavía no has tomado mediciones ni capturas en este caso.</p>
            <button
              type="button"
              className="btn-primary-soft no-print"
              onClick={() => s.setUi({ screen: 'simulator', consoleTab: 'medir' })}
            >
              Ir a medir
            </button>
          </section>
        ) : (
          <>
            {report.derived.length > 0 && (
              <section className="cr-section">
                <div className="cr-kpi-strip">
                  {report.derived.map((d) => (
                    <div
                      key={d.id}
                      className="cr-kpi"
                      data-derived={d.id}
                      data-tip={`${d.formula} (${d.inputs})`}
                    >
                      <span className="cr-kpi-label">{d.label}</span>
                      <span className="cr-kpi-val">{d.value}</span>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {report.rows.length > 0 && (
              <section className="cr-section">
                <h3>Mediciones</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Parámetro</th>
                      <th>Valor</th>
                      <th>Referencia</th>
                      <th>{revealed ? 'Origen' : 'Modo'}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.rows.map((r) => (
                      <tr key={r.id} data-technique={r.technique?.level ?? 'free'}>
                        <td>{r.label}</td>
                        <td className="cr-val-cell">{r.value}</td>
                        <td data-range-flag={r.rangeFlag ?? 'none'}>
                          {r.range ? (
                            <>
                              {r.range}{' '}
                              {r.rangeFlag === 'normal' ? (
                                <span className="pill ok">normal</span>
                              ) : (
                                <span className="pill warn">
                                  {r.rangeFlag === 'high' ? 'alto' : 'bajo'}
                                </span>
                              )}
                            </>
                          ) : (
                            '—'
                          )}
                        </td>
                        <td className="small">
                          {modalityLabel(r.modality)}
                          {r.view ? ` · ${viewLabel(r.view)}` : ''}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            )}

            {s.captures.length > 0 && (
              <section className="cr-section cr-captures" aria-label="Imágenes guardadas">
                <h3>Imágenes ({s.captures.length})</h3>
                <div className="cr-capture-grid">
                  {s.captures.map((cap, idx) => (
                    <figure key={cap.id} className="cr-capture-fig">
                      <img
                        src={cap.dataUrl}
                        alt={`Imagen ${idx + 1}: ${formatCaptureCaption(cap, revealed)}`}
                      />
                      <figcaption>
                        <span>
                          #{idx + 1} · {formatCaptureCaption(cap, revealed)}
                        </span>
                        {!s.closed && (
                          <button
                            type="button"
                            className="capture-del no-print"
                            aria-label={`Eliminar imagen ${idx + 1}`}
                            onClick={() => s.removeCapture(cap.id)}
                          >
                            ×
                          </button>
                        )}
                      </figcaption>
                    </figure>
                  ))}
                </div>
              </section>
            )}
          </>
        )}

        {s.impressionSelection.length > 0 && (
          <section className="cr-section cr-conclusion">
            <h3>Conclusión</h3>
            <ol>
              {s.impressionSelection.map((id) => (
                <li key={id}>{getFinding(id)?.label ?? id}</li>
              ))}
            </ol>
          </section>
        )}
      </article>

      <details className="cr-eval-details no-print" open={isExam || undefined}>
        <summary className="cr-eval-summary">
          <span>{isExam ? 'Impresión y evaluación del examen' : 'Evaluación docente e impresión'}</span>
          {summary && <span className="cr-eval-badge">{summary.total}/100</span>}
        </summary>
        <section className="cr-eval hide-in-print" aria-label="Evaluación docente">
          <h3>Impresión estructurada</h3>
          <div
            className="impression-form"
            data-impression-score={corrected ? (impression?.score ?? '') : ''}
          >
            {DOMAINS.map((d) => (
              <fieldset key={d.id}>
                <legend className="small">{d.label}</legend>
                {FINDINGS.filter((f) => f.domain === d.id).map((f) => {
                  const checked = s.impressionSelection.includes(f.id);
                  const isExpected = expected.includes(f.id);
                  const cls =
                    corrected && s.impressionSelection.length
                      ? checked && isExpected
                        ? 'ok'
                        : checked && !isExpected
                          ? 'bad'
                          : !checked && isExpected
                            ? 'missed'
                            : ''
                      : '';
                  return (
                    <label key={f.id} className={`finding ${cls}`} data-finding={f.id}>
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={corrected}
                        onChange={() => s.toggleFinding(f.id)}
                      />
                      {f.label}
                    </label>
                  );
                })}
              </fieldset>
            ))}
          </div>

          {!isExam && (
            <p className="impression-actions">
              {corrected ? (
                <button type="button" className="ghost" onClick={() => s.redoImpression()}>
                  Rehacer
                </button>
              ) : (
                <button
                  type="button"
                  className="btn-primary-soft"
                  onClick={() => s.checkImpression()}
                  disabled={!s.impressionSelection.length}
                >
                  Corregir
                </button>
              )}
            </p>
          )}

          {impression && corrected && (
            <p data-impression-result="1">
              Impresión: <b>{impression.score}/100</b> · correctos {impression.correct.length} ·
              omitidos {impression.missed.length} · sobrantes {impression.wrong.length}
            </p>
          )}
          {impression && !revealed && (
            <p className="small">
              {s.impressionSelection.length} hallazgos registrados; se evalúan al finalizar el
              examen.
            </p>
          )}

          <h3>Vistas requeridas</h3>
          {!revealed ? (
            <p className="small" data-exam-views="hidden">
              Se muestran y puntúan al finalizar el examen.
            </p>
          ) : (
            <div className="cr-view-pills">
              {acquisition.perView.map((v) => (
                <span key={v.viewId} className={`pill ${v.ok ? 'ok' : 'warn'}`}>
                  {v.ok ? '✓ ' : ''}
                  {viewLabel(v.viewId)} {v.achieved}/{v.required}
                </span>
              ))}
            </div>
          )}

          {summary && (
            <>
              <h3>
                Puntuación: {summary.total}/100{' '}
                <span className="small">
                  (Adq. {summary.acquisition.total} · Med. {summary.measurements.total}
                  {summary.impression !== null ? ` · Imp. ${summary.impression}` : ''})
                </span>
              </h3>
              <table>
                <thead>
                  <tr>
                    <th>Requerida</th>
                    <th>Medida</th>
                    <th>Modelo</th>
                    <th>Error</th>
                    <th>Pts</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.measurements.rows.map((r) => (
                    <tr key={r.measurementId}>
                      <td>{r.label}</td>
                      <td>{r.measured === null ? '—' : `${r.measured.toFixed(2)} ${r.units}`}</td>
                      <td>
                        {showsScoredTruth(r, corrected) ? `${r.truth.toFixed(2)} ${r.units}` : '—'}
                      </td>
                      <td>{r.errorPct === null ? '—' : `${r.errorPct.toFixed(0)} %`}</td>
                      <td>{r.points}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          {!isExam && corrected && s.truth && (
            <>
              <h3>Referencia del modelo</h3>
              <ul className="small">
                <li>
                  VI: DTD {formatClinical(s.truth.lv.eddCm, 'linearCm')} · FEVI{' '}
                  {formatClinical(s.truth.lv.efPct, 'percent')} · VS{' '}
                  {formatClinical(s.truth.lv.strokeVolumeMl, 'volumeMl')}
                </li>
                <li>
                  Aorta: TSVI {formatClinical(s.truth.lvot.diameterCm, 'linearCm')} · VTI{' '}
                  {formatClinical(s.truth.lvot.vtiCm, 'vtiCm')} · Vmáx VAo{' '}
                  {formatClinical(s.truth.aorticValve.vmaxMps, 'velocityMps')} · AVA{' '}
                  {formatClinical(s.truth.aorticValve.continuityAvaCm2, 'areaCm2')}
                </li>
                <li>
                  Diástole y VD: E {formatClinical(s.truth.mitral.ePeakMps, 'velocityMps')} · E/e′{' '}
                  {s.truth.mitral.eOverEPrimeAvg.toFixed(1)} · TAPSE{' '}
                  {formatClinical(s.truth.rightHeart.tapseCm, 'linearCm')} · PSVD{' '}
                  {s.truth.rightHeart.rvspMmHg
                    ? formatClinical(s.truth.rightHeart.rvspMmHg, 'gradientMmHg')
                    : '—'}
                </li>
              </ul>
            </>
          )}

          {(isExam || corrected) && report.impression.length > 0 && (
            <>
              <h3>Hallazgos del modelo</h3>
              <ul className="small" data-model-impression>
                {report.impression.map((l, i) => (
                  <li key={i}>{l}</li>
                ))}
              </ul>
            </>
          )}
        </section>
      </details>

      <p className="small cr-disclaimer">
        Simulador docente con casos sintéticos · no usar para diagnóstico clínico real.
      </p>
    </div>
  );
}
