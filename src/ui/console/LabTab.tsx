import { ArtifactLab } from '../ArtifactLab';
import { Section } from '../controls';

/** Artifact laboratory — a teaching tool, hidden in exam mode. */
export function LabTab() {
  return (
    <Section
      title="Laboratorio de artefactos"
      tip="Ajusta cada artefacto y observa su causa y su remedio. Parte de los valores del caso."
    >
      <ArtifactLab />
    </Section>
  );
}
