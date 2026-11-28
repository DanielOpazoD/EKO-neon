import { useSimStore } from '@/app/store';
import { SHORTCUTS } from '@/app/shortcuts';
import { GUIDELINE_REFERENCES } from '@/clinical/guidelines/references';

/** Concise 1-page clinical guideline reference list with collapsible keyboard shortcuts. */
export function ReferencesScreen() {
  const setUi = useSimStore((s) => s.setUi);

  return (
    <div className="screen report-clean">
      <div className="cr-top-row">
        <div className="cr-top-actions">
          <button
            type="button"
            className="ghost"
            onClick={() => setUi({ screen: 'simulator' })}
          >
            ← Simulador
          </button>
        </div>
        <span className="small">{GUIDELINE_REFERENCES.length} guías clínicas</span>
      </div>

      <article className="clinical-report" aria-label="Referencias clínicas">
        <header className="cr-head cr-head-compact">
          <h2>Referencias clínicas</h2>
        </header>
        <section className="cr-section">
          <table>
            <thead>
              <tr>
                <th>Guía</th>
                <th>Sociedad</th>
                <th>Año</th>
              </tr>
            </thead>
            <tbody>
              {GUIDELINE_REFERENCES.map((r) => (
                <tr key={r.id}>
                  <td>
                    <a href={r.url} target="_blank" rel="noreferrer">
                      {r.title}
                    </a>
                  </td>
                  <td className="small">{r.society}</td>
                  <td className="nowrap small">{r.year}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </article>

      <details className="cr-eval-details">
        <summary className="cr-eval-summary">
          <span>Atajos de teclado</span>
          <span className="cr-eval-badge">{SHORTCUTS.length}</span>
        </summary>
        <section style={{ marginTop: 8 }}>
          <table>
            <tbody>
              {SHORTCUTS.map((s) => (
                <tr key={s.keys}>
                  <td className="nowrap">
                    <code>{s.keys}</code>
                  </td>
                  <td className="small">{s.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </details>
    </div>
  );
}
