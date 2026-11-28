import type {
  EchoSourceId,
  PointExplanation,
  ProbePointInfo,
  QualityTier,
  SimInput,
  SimOutput,
} from '@/simulator/core/protocol';
import type { LassoPoint, RegionSummary } from './reviewRegion';
import type { ImagingModality } from '@/simulator/renderer/types';
import type { SectorMapping } from '@/simulator/renderer/scanConvert';
import { Structure, Tissue } from '@/simulator/anatomy/tissue';
import { acquisitionLimits } from '@/simulator/core/acquisitionInput';
import {
  colorWallFilterCutoffMps,
  colorWallFilterRangeMps,
  wallFilterCutoffNyquist,
} from '@/simulator/doppler/color/kasaiEstimator';

/**
 * Review mode (decision 134): the exchange format between whoever looks at the image and whoever changes
 * the model. A report carries the exact simulator input (so the frame can be reproduced offline with the
 * same seed), the frame's identity (phase, view analysis) and numbered markers placed on the image, each
 * with what the model holds there and a free note. Pure functions: no store, no DOM.
 */
export type ReviewCategory =
  'anatomia' | 'movimiento' | 'textura' | 'doppler' | 'vista' | 'interfaz' | 'otro';

export const REVIEW_CATEGORIES: { id: ReviewCategory; label: string }[] = [
  { id: 'anatomia', label: 'Anatomía / forma' },
  { id: 'movimiento', label: 'Movimiento / ciclo' },
  { id: 'textura', label: 'Textura / brillo' },
  { id: 'doppler', label: 'Doppler / tira' },
  { id: 'vista', label: 'Vista / plano' },
  { id: 'interfaz', label: 'Interfaz' },
  { id: 'otro', label: 'Otro' },
];

export interface ReviewMarker {
  id: string;
  /** 1-based number shown on the image. */
  n: number;
  /** Where it was placed: on the image (sector or strip) or on the 3D model of the navigator. */
  space: 'image' | 'model';
  /** Display pixel where it was placed and the sector mapping of that moment (reprojected on resize/zoom); image only. */
  x: number;
  y: number;
  captureSector: SectorMapping | null;
  /** Torso-frame point (cm) and the navigator surface hit (mesh group id, skin, skeleton); model markers only. */
  torso: { x: number; y: number; z: number } | null;
  group: string | null;
  /** Polar position in the sector; null when the click fell on a strip. */
  rCm: number | null;
  thetaRad: number | null;
  /** Strip coordinates when the click fell on a spectral or M-mode strip. */
  strip: { kind: 'spectral' | 'm-mode'; column: number; value: number } | null;
  frameId: number;
  phase: number;
  modality: ImagingModality;
  /** Structure id from the frame's own structure map at the click (0 on strips). */
  structure: number;
  /** The worker's classification of the point (arrives asynchronously; null on strips). */
  point: ProbePointInfo | null;
  note: string;
  category: ReviewCategory;
  /** Primary marker this one is linked to as a secondary point of the same problem (decision 136); null for a primary. */
  parentId: string | null;
  /** A hand-drawn outline (decision 340), polar: the marker sits at its centroid. Absent or null for a click. */
  lasso?: LassoPoint[] | null;
  /** What the outlined region holds (decision 340): the piece of the clicked structure, or the lasso's cells. */
  region?: RegionSummary | null;
  /** Where the echo at the marker comes from (decision 340), from the worker's CPU tracer; null until it arrives. */
  explain?: PointExplanation | null;
}

export interface ReviewReport {
  format: 'echotwin-review';
  /**
   * 2 since `input.color.wallFilterMps` is the −3 dB cutoff asked for (bounded where it filters); a version 1 report is
   * read with the request that replays the filter main's 2D colour gave its label (`legacyColorWallFilterMps`).
   */
  version: 2;
  createdAt: string;
  author: 'usuario' | 'claude';
  app: { url: string };
  caseId: string;
  caseTitle: string;
  mode: string;
  workerMode: string;
  /** The exact simulator input of the frame: replaying it reproduces the image (deterministic seed). */
  input: SimInput;
  frame: {
    frameId: number;
    phase: number;
    timeS: number;
    beatIndex: number;
    heartRateBpm: number;
    view: SimOutput['view'];
    backend: string;
    /** Tier the frame was drawn with: the input's quality choice resolved by the core (decision 154). */
    tier?: QualityTier;
  } | null;
  note: string;
  markers: ReviewMarker[];
}

export interface ReviewSnapshot {
  caseId: string;
  caseTitle: string;
  mode: string;
  workerMode: string;
  input: SimInput;
  hud: SimOutput | null;
  note: string;
  markers: ReviewMarker[];
  author?: ReviewReport['author'];
  url?: string;
}

export const STRUCTURE_LABELS: Record<number, string> = {
  [Structure.None]: 'fuera del modelo',
  [Structure.LvCavity]: 'cavidad del VI',
  [Structure.LvWallSeptal]: 'pared septal del VI',
  [Structure.LvWallLateral]: 'pared lateral del VI',
  [Structure.LvWallAnterior]: 'pared anterior del VI',
  [Structure.LvWallInferior]: 'pared inferior o inferolateral del VI',
  [Structure.LvApex]: 'ápex del VI',
  [Structure.RvCavity]: 'cavidad del VD',
  [Structure.RvWall]: 'pared libre del VD',
  [Structure.LaCavity]: 'aurícula izquierda',
  [Structure.LaWall]: 'pared de la AI',
  [Structure.RaCavity]: 'aurícula derecha',
  [Structure.RaWall]: 'pared de la AD',
  [Structure.MitralAnterior]: 'velo mitral anterior',
  [Structure.MitralPosterior]: 'velo mitral posterior',
  [Structure.AorticValve]: 'válvula aórtica',
  [Structure.TricuspidValve]: 'válvula tricúspide',
  [Structure.AorticRoot]: 'raíz aórtica',
  [Structure.Lvot]: 'TSVI',
  [Structure.Rvot]: 'TSVD',
  [Structure.PapillaryMuscle]: 'músculo papilar',
  [Structure.Pericardium]: 'pericardio',
  [Structure.PericardialEffusion]: 'derrame pericárdico',
  [Structure.DescendingAorta]: 'aorta descendente',
  [Structure.Ivc]: 'vena cava inferior',
  [Structure.ChestWall]: 'pared torácica',
  [Structure.Rib]: 'costilla',
  [Structure.Sternum]: 'esternón',
  [Structure.Lung]: 'pulmón',
  [Structure.Liver]: 'hígado',
  [Structure.Spine]: 'columna',
  [Structure.InteratrialSeptum]: 'tabique interauricular',
  [Structure.PulmonaryValve]: 'válvula pulmonar',
  [Structure.MitralAnnulus]: 'anillo mitral',
  [Structure.TricuspidAnnulus]: 'anillo tricúspide',
  [Structure.Chordae]: 'cuerdas tendinosas',
  [Structure.ModeratorBand]: 'banda moderadora',
  [Structure.LaAppendage]: 'orejuela izquierda',
  [Structure.RaAppendage]: 'orejuela derecha',
  [Structure.AortomitralCurtain]: 'cortina aortomitral',
  [Structure.Esophagus]: 'esófago',
  [Structure.PosteriorMediastinum]: 'mediastino posterior',
  [Structure.SuperiorMediastinum]: 'mediastino superior',
  [Structure.PericardialFat]: 'grasa pericárdica',
  [Structure.PulmonaryVein]: 'vena pulmonar',
  [Structure.CoronarySinus]: 'seno coronario',
  [Structure.PulmonaryArtery]: 'arteria pulmonar',
  [Structure.RvPapillary]: 'papilar del VD',
  [Structure.Svc]: 'vena cava superior',
  [Structure.HepaticVein]: 'vena hepática',
  [Structure.Diaphragm]: 'diafragma',
  [Structure.EpicardialFat]: 'grasa epicárdica',
  [Structure.Stomach]: 'estómago',
  [Structure.AscendingAorta]: 'aorta ascendente',
  [Structure.AorticArch]: 'arco aórtico',
  [Structure.BrachiocephalicArtery]: 'tronco braquiocefálico',
  [Structure.LeftCommonCarotid]: 'carótida común izquierda',
  [Structure.LeftSubclavian]: 'subclavia izquierda',
};

export const TISSUE_LABELS: Record<number, string> = {
  [Tissue.None]: '—',
  [Tissue.Blood]: 'sangre',
  [Tissue.Myocardium]: 'miocardio',
  [Tissue.Valve]: 'válvula',
  [Tissue.Pericardium]: 'pericardio',
  [Tissue.Fat]: 'grasa',
  [Tissue.Muscle]: 'músculo',
  [Tissue.Bone]: 'hueso',
  [Tissue.Cartilage]: 'cartílago',
  [Tissue.Lung]: 'pulmón',
  [Tissue.Fluid]: 'líquido',
  [Tissue.VesselWall]: 'pared vascular',
  [Tissue.Calcium]: 'calcio',
  [Tissue.Skin]: 'piel',
  [Tissue.Liver]: 'hígado',
  [Tissue.Spine]: 'columna',
  [Tissue.Fibrous]: 'tejido fibroso',
  [Tissue.Chordae]: 'cuerdas',
};

/** Navigator surfaces a 3D marker can land on (heartMesh group ids plus the body layers). */
export const MESH_GROUP_LABELS: Record<string, string> = {
  'lv-myocardium': 'miocardio del VI',
  'rv-myocardium': 'miocardio del VD',
  'lv-cavity': 'cavidad del VI',
  'rv-cavity': 'cavidad del VD',
  'left-atrium': 'aurícula izquierda',
  'right-atrium': 'aurícula derecha',
  atria: 'aurículas',
  valves: 'válvulas',
  'aortic-root': 'raíz aórtica',
  'pulmonary-artery': 'arteria pulmonar',
  'venae-cavae': 'venas cavas',
  'pulmonary-veins': 'venas pulmonares',
  'coronary-arteries': 'arterias coronarias',
  'coronary-veins': 'venas cardíacas',
  'great-vessels': 'grandes vasos',
  'thoracic-aorta': 'arco y aorta torácica',
  skin: 'piel',
  skeleton: 'hueso',
  ghost: 'corazón esquemático',
};

export function groupLabel(id: string | null): string {
  return id === null ? 'superficie' : (MESH_GROUP_LABELS[id] ?? id);
}

/** Numbers primaries 1, 2, 3… in order and each primary's secondaries 1, 2, 3… (shown as a, b, c). */
export function renumberMarkers(ms: ReviewMarker[]): ReviewMarker[] {
  let p = 0;
  const kids = new Map<string, number>();
  return ms.map((m) => {
    if (!m.parentId) return { ...m, n: ++p };
    const c = (kids.get(m.parentId) ?? 0) + 1;
    kids.set(m.parentId, c);
    return { ...m, n: c };
  });
}

/** "3" for a primary, "3b" for the second secondary of marker 3. */
export function markerLabel(m: ReviewMarker, all: ReviewMarker[]): string {
  if (!m.parentId) return String(m.n);
  const parent = all.find((x) => x.id === m.parentId);
  return `${parent ? parent.n : '?'}${String.fromCharCode(96 + Math.min(26, Math.max(1, m.n)))}`;
}

/** Distance between two markers (cm): in the torso when both are placed there, else in the image plane. */
export function markerDistanceCm(
  a: ReviewMarker,
  b: ReviewMarker,
): { cm: number; where: 'torso' | 'imagen' } | null {
  const pa = a.point?.torso ?? a.torso;
  const pb = b.point?.torso ?? b.torso;
  if (pa && pb) return { cm: Math.hypot(pa.x - pb.x, pa.y - pb.y, pa.z - pb.z), where: 'torso' };
  if (a.rCm !== null && a.thetaRad !== null && b.rCm !== null && b.thetaRad !== null)
    return {
      cm: Math.sqrt(
        Math.max(
          0,
          a.rCm * a.rCm + b.rCm * b.rCm - 2 * a.rCm * b.rCm * Math.cos(a.thetaRad - b.thetaRad),
        ),
      ),
      where: 'imagen',
    };
  return null;
}

export function structureLabel(id: number): string {
  return STRUCTURE_LABELS[id] ?? `estructura ${id}`;
}

export function tissueLabel(id: number): string {
  return TISSUE_LABELS[id] ?? `tejido ${id}`;
}

const deg = (rad: number): string => `${((rad * 180) / Math.PI).toFixed(0)}°`;
const f1 = (v: number): string => v.toFixed(1);
const f2 = (v: number): string => v.toFixed(2);

export const ECHO_SOURCE_LABELS: Record<EchoSourceId, string> = {
  clutter: 'ruido de campo cercano',
  chestWall: 'reverberación de la pared torácica',
  sideLobe: 'lóbulos laterales',
  beamWidth: 'ancho del haz',
};
const pct = (v: number): string => `${Math.round(v * 100)} %`;

/** What the region of a marker holds, in one line (decision 340). */
export function regionLine(m: ReviewMarker): string | null {
  const g = m.region;
  if (!g) return null;
  const parts = [
    `${m.lasso ? 'lazo' : 'zona resaltada'} ${g.areaCm2.toFixed(1)} cm²`,
    g.structures
      .slice(0, 4)
      .map((x) => `${pct(x.share)} ${structureLabel(x.id)}`)
      .join(', '),
  ];
  if (g.grey) parts.push(`gris mediana ${g.grey.median} (p10 ${g.grey.p10}, p90 ${g.grey.p90})`);
  return parts.join(' · ');
}

/** Where the echo at a marker comes from, in one line (decision 340). */
export function explainLine(m: ReviewMarker): string | null {
  const e = m.explain;
  if (!e) return null;
  const parts: string[] = [];
  if (e.behindPleura)
    parts.push('tras la pleura: lo dibujado es la reverberación pleural, no tejido');
  // behind the pleura what no switchable source supplies is the pleura's reverberation, not the tissue's own echo
  const own = e.behindPleura ? 'reverberación pleural' : 'propio del tejido';
  const src = e.sources.filter((x) => x.share >= 0.05);
  parts.push(
    src.length
      ? `${src.map((x) => `${pct(x.share)} ${ECHO_SOURCE_LABELS[x.id]}`).join(', ')}; ${own} ${pct(e.ownShare)}`
      : `${own} (ninguna fuente conmutable aporta un 5 %)`,
  );
  if (e.shadow)
    parts.push(
      `mayor pérdida en el camino: ${structureLabel(e.shadow.structure)} (${tissueLabel(e.shadow.tissue)}) a ${f1(e.shadow.atCm)} cm, ${e.shadow.dropDb >= 60 ? 'no pasa nada (más de 60 dB)' : `−${f1(e.shadow.dropDb)} dB`}`,
    );
  if (e.mirrorLevel > 0)
    parts.push(`espejo de la consola activo (${f2(e.mirrorLevel)}), no medido aquí`);
  return `eco aquí: ${parts.join(' · ')}`;
}

/** Where a marker sits, in the words the model uses: structure, tissue, polar position and local coordinates. */
export function markerPlace(m: ReviewMarker): string {
  if (m.space === 'model') {
    const parts = [`3D · ${groupLabel(m.group)}`];
    const p = m.point;
    if (p) parts.push(structureLabel(p.structure), tissueLabel(p.tissue));
    if (m.torso) parts.push(`torso (${f1(m.torso.x)}, ${f1(m.torso.y)}, ${f1(m.torso.z)})`);
    if (p) {
      parts.push(`corazón (${f1(p.heart.x)}, ${f1(p.heart.y)}, ${f1(p.heart.z)})`);
      if (p.levelFrac !== null && p.azRad !== null)
        parts.push(`nivel ${f2(p.levelFrac)} · acimut ${deg(p.azRad)}`);
      if (p.rootT !== null && p.rootR !== null)
        parts.push(`raíz t ${f2(p.rootT)} r ${f2(p.rootR)}`);
      parts.push(
        `${Math.abs(p.offPlaneCm) < 0.3 ? 'en el plano de imagen' : `a ${f1(Math.abs(p.offPlaneCm))} cm del plano de imagen`}`,
      );
    }
    return parts.join(' · ');
  }
  if (m.strip)
    return `tira ${m.strip.kind === 'spectral' ? 'espectral' : 'modo M'} · columna ${m.strip.column.toFixed(0)} · ${m.strip.kind === 'spectral' ? `${f2(m.strip.value)} m/s` : `${f1(m.strip.value)} cm`}`;
  const parts: string[] = [];
  const p = m.point;
  parts.push(
    m.lasso
      ? `lazo a mano de ${m.lasso.length} puntos; centro: ${structureLabel(p ? p.structure : m.structure)}`
      : structureLabel(p ? p.structure : m.structure),
  );
  if (p) parts.push(tissueLabel(p.tissue));
  if (m.rCm !== null && m.thetaRad !== null) parts.push(`${f1(m.rCm)} cm · ${deg(m.thetaRad)}`);
  if (p) {
    parts.push(`corazón (${f1(p.heart.x)}, ${f1(p.heart.y)}, ${f1(p.heart.z)})`);
    if (p.levelFrac !== null && p.azRad !== null)
      parts.push(`nivel ${f2(p.levelFrac)} · acimut ${deg(p.azRad)}`);
    if (p.rootT !== null && p.rootR !== null) parts.push(`raíz t ${f2(p.rootT)} r ${f2(p.rootR)}`);
    parts.push(`sdf ${f2(p.sdfCm)}`);
  }
  return parts.join(' · ');
}

export function buildReviewReport(s: ReviewSnapshot): ReviewReport {
  const hud = s.hud;
  return {
    format: 'echotwin-review',
    version: 2,
    createdAt: new Date().toISOString(),
    author: s.author ?? 'usuario',
    app: { url: s.url ?? '' },
    caseId: s.caseId,
    caseTitle: s.caseTitle,
    mode: s.mode,
    workerMode: s.workerMode,
    input: s.input,
    frame: hud
      ? {
          frameId: hud.frameId,
          phase: hud.phase,
          timeS: hud.timeS,
          beatIndex: hud.beatIndex,
          heartRateBpm: hud.heartRateBpm,
          view: hud.view,
          backend: String(hud.stats['backend'] ?? ''),
          ...(hud.stats['tier'] ? { tier: hud.stats['tier'] as QualityTier } : {}),
        }
      : null,
    note: s.note,
    markers: s.markers,
  };
}

const PATIENT_POSITION: Record<string, string> = {
  'left-lateral': 'decúbito lateral izquierdo',
  supine: 'supino',
  'subcostal-supine': 'supino (subcostal)',
};
const RESPIRATION: Record<string, string> = {
  expiration: 'espiración',
  inspiration: 'inspiración',
  'breath-hold': 'apnea',
  'free-breathing': 'respiración libre',
};

function phaseName(phase: number): string {
  if (phase < 0.05 || phase >= 0.95) return 'telediástole';
  if (phase < 0.4) return 'sístole';
  if (phase < 0.5) return 'telesístole';
  if (phase < 0.75) return 'diástole precoz';
  return 'diástole tardía';
}

/** The report as the text pasted into a conversation: readable lines first, the JSON (exact state) after. */
export function reportToMarkdown(r: ReviewReport): string {
  const i = r.input;
  const s = i.settings;
  const lines: string[] = [];
  lines.push(`## Informe de revisión EchoTwin — ${r.createdAt.slice(0, 16).replace('T', ' ')}`);
  lines.push('');
  lines.push(`Caso: ${r.caseId} (${r.caseTitle}) · modo ${r.mode} · núcleo ${r.workerMode}`);
  lines.push(
    `Paciente: ${PATIENT_POSITION[i.patient.position] ?? i.patient.position}, ${RESPIRATION[i.patient.respiration] ?? i.patient.respiration}`,
  );
  lines.push(
    `Sonda: u ${f1(i.probe.u)} v ${f1(i.probe.v)} cm · rotación ${f1(i.probe.rotationDeg)}° · tilt ${f1(i.probe.tiltDeg)}° · rock ${f1(i.probe.rockDeg)}° · presión ${f2(i.probe.pressure)}`,
  );
  const v = r.frame?.view;
  if (v)
    lines.push(
      `Vista reconocida: ${v.bestViewName} (${v.bestViewId ?? '—'}, puntaje ${Math.round(v.score)}, acortamiento ${v.foreshorteningDeg.toFixed(0)}°, plano ${v.planeAngleDeg.toFixed(0)}°, desplazamiento ${v.offsetCm.toFixed(1)} cm)`,
    );
  lines.push(
    `Modalidad ${i.modality.toUpperCase()} · profundidad ${s.depthCm} cm · sector ${s.sectorDeg}° · ganancia ${s.gainDb} dB · ${s.frequencyMHz} MHz${s.harmonics ? ' THI' : ''} · foco ${s.focusCm} cm · zoom ${s.zoom} · rango ${s.dynamicRangeDb} dB · mapa ${s.grayMap} · persistencia ${s.persistence} · calidad ${i.quality}${r.frame?.tier && r.frame.tier !== i.quality ? ` (${r.frame.tier})` : ''} · backend ${i.rendererBackend}${r.frame?.backend ? ` (${r.frame.backend})` : ''}${s.invertLR ? ' · izq/der invertido' : ''}`,
  );
  // the colour's Nyquist as acquired (the strip's is set by the depth) and the wall filter's cutoff there: the request
  // the report keeps, bounded as the simulator bounds it
  const colorScale = acquisitionLimits(i).color.scaleMps;
  const wallFilter = `filtro de pared ${f1(100 * colorWallFilterCutoffMps(i.color.wallFilterMps, colorScale))} cm/s`;
  if (i.modality === 'color')
    lines.push(
      `Color: escala ${f2(i.color.scaleMps)} m/s · caja r ${f1(i.color.boxRMinCm)}–${f1(i.color.boxRMaxCm)} cm, θ ${deg(i.color.boxThetaMinRad)}…${deg(i.color.boxThetaMaxRad)} · ${wallFilter}`,
    );
  // the colour M-mode has no gate and no spectral scale: its line used to give PW's
  if (i.modality === 'cmm')
    lines.push(
      `Cursor ${deg(i.cursorThetaRad)} · escala color ${f2(colorScale)} m/s · ${wallFilter} · barrido ${i.spectral.sweepSpeedMmPerS} mm/s`,
    );
  else if (i.modality !== '2d' && i.modality !== 'color')
    lines.push(
      `Cursor ${deg(i.cursorThetaRad)} · compuerta ${f1(i.gateDepthCm)} cm (${f1(i.spectral.gateLengthCm)} cm) · escala ${f2(i.spectral.scaleMps)} m/s · barrido ${i.spectral.sweepSpeedMmPerS} mm/s`,
    );
  if (i.artifactOverrides)
    lines.push(`Laboratorio de artefactos: ${JSON.stringify(i.artifactOverrides)}`);
  if (r.frame)
    lines.push(
      `Cuadro ${r.frame.frameId} · fase ${f2(r.frame.phase)} (${phaseName(r.frame.phase)}) · latido ${r.frame.beatIndex} · FC ${Math.round(r.frame.heartRateBpm)} · ${i.frozen ? `congelado, cine ${i.cineOffset}` : 'en vivo'}`,
    );
  lines.push('');
  if (r.note.trim()) {
    lines.push(`Nota general: ${r.note.trim()}`);
    lines.push('');
  }
  if (r.markers.length) {
    lines.push('Marcadores:');
    for (const m of r.markers) {
      if (m.parentId) continue;
      const cat = REVIEW_CATEGORIES.find((c) => c.id === m.category)?.label ?? m.category;
      lines.push(
        `${m.n}. [${cat}] ${markerPlace(m)} · fase ${f2(m.phase)}${m.note.trim() ? ` — «${m.note.trim()}»` : ''}`,
      );
      for (const d of [regionLine(m), explainLine(m)]) if (d) lines.push(`   ${d}`);
      // secondary points of the same problem, with their distance to the primary (decision 136)
      for (const c of r.markers) {
        if (c.parentId !== m.id) continue;
        const d = markerDistanceCm(m, c);
        lines.push(
          `  ${markerLabel(c, r.markers)}. ${markerPlace(c)} · fase ${f2(c.phase)}${d ? ` · a ${f1(d.cm)} cm de ${m.n} (${d.where})` : ''}${c.note.trim() ? ` — «${c.note.trim()}»` : ''}`,
        );
        for (const t of [regionLine(c), explainLine(c)]) if (t) lines.push(`     ${t}`);
      }
    }
    lines.push('');
  }
  lines.push('```json');
  lines.push(JSON.stringify(r));
  lines.push('```');
  return lines.join('\n');
}

/** Main's default colour scale (m/s), the Nyquist a learner comes back to after any other. */
const MAIN_DEFAULT_COLOR_SCALE_MPS = 0.68;

/**
 * The −3 dB cutoff (m/s) main's 2D colour had for a wall filter label at a Nyquist scale. Until version 2
 * `wallFilterMps` was a label that set the shares of the packet regression by ramps in WF/Nyquist (the mean over
 * 0–0.03 Nyquist, the trend over 0.02–0.06, the quadratic over 0.055–0.115; kasaiEstimator.ts before c6a213d), whose
 * −3 dB point lay well above it: the default 0.06 at 0.68 m/s cut at 0.205 m/s.
 */
function mainColorWallFilterCutoffMps(labelMps: number, scaleMps: number): number {
  const cut = labelMps / scaleMps;
  const ramp = (start: number, span: number) => Math.min(1, Math.max(0, (cut - start) / span));
  return (
    wallFilterCutoffNyquist({ mean: ramp(0, 0.03), trend: ramp(0.02, 0.04), quad: ramp(0.055, 0.06) }) *
    scaleMps
  );
}

/**
 * The colour wall filter a version 1 report asks for in version 2. Replayed as a request, main's default label 0.06
 * gave the 7.6 cm/s mean removal, under which the moving myocardium flashed (A4C, CPU, 44 frames: 9317 coloured
 * myocardial samples, 70 in main and in this migration). But the label was relative to the scale and the request is
 * absolute: no one request filters as main did at every scale. Of the requests that replay the report's image at the
 * scale its replay acquires with (`acquisitionLimits`), it is the nearest to main's cutoff for the label at main's
 * default 0.68 m/s, the scale the learner comes back to:
 * - main's cutoff at the report's scale (`mainColorWallFilterCutoffMps`) inside the range the packet allows there: that
 *   cutoff, the only request that replays it. From 0.06 Nyquist up (mean and trend whole, the default among them) it
 *   gives back main's very shares.
 * - at the steepest filter, where the ramps saturate (from 0.115 Nyquist): every request from it up replays the image,
 *   and the request is the larger of the two cutoffs. A 2D colour at 0.35 m/s with 0.06 (the task colour-low-scale) cut
 *   at 11.2 cm/s; asked as 11.2, it went on filtering at 11.2 once the scale was back at 0.68 (A4C, CPU, 44 frames:
 *   7073 coloured myocardial samples, 70 in main). It now asks for 20.5: 11.2 at 0.35, 20.5 at 0.68 (70 samples).
 * - under the mean removal, the lowest filter (main's partial mean removal is bounded to it): every request up to it
 *   but off replays the image, and the request is the smaller of that bound and the cutoff at 0.68 (or the cutoff at
 *   the report's scale, if main's filter cut nothing at 0.68). A filter that cut nothing at the report's scale (a mean
 *   removal under 29 %) stays off.
 * Over main's controls (labels 0.01–0.30 m/s, scales 0.15–1.2 m/s), against the cutoff at the report's scale alone, no
 * image at that scale changes and none at 0.68 filters farther from main's; that cutoff missed it by up to 14.2 cm/s
 * (0.08 at 0.15 m/s asked for 4.8, where main cut at 21.8 at 0.68). What remains is what one absolute request cannot
 * do: up to 5.0 cm/s at 0.68 inside the range (0.03 at 0.27 m/s asks for 8.7, main cut at 13.7), and 7.6 where the
 * replay stays off (0.01 from 1.14 m/s). A colour M-mode at 20 cm acquires at 0.551 m/s, where 0.06 cuts at 17.6
 * cm/s, just inside the 17.7 the packet allows: the 2D colour at 0.68 after it filters at 17.6 where main's cut at
 * 20.5 (445 coloured myocardial samples, 70 in main).
 * One rule for every version 1, colour M-mode included. Main's strip cut at the label itself, a hard cut on the true
 * velocity, but the request is one for the strip and for the 2D colour of the rest of the session: a colour M-mode
 * report that kept its label left the 2D colour after it on the mean removal (7.5 cm/s at 0.676 m/s: 9339 coloured
 * myocardial samples). Migrated at 16 cm, the 2D colour filters where main's did, at 20.4 cm/s (74 samples, as in
 * main), and the strip paints with that filter's response (A4C through the crux: 11.0 % of the strip coloured, 17.7 %
 * in main). That difference is the strip's physics, not the migration: main's strip cut at 6 cm/s where its 2D colour,
 * on the same line and with the same label, cut at 20.4.
 */
export function legacyColorWallFilterMps(input: SimInput): number {
  const label = input.color.wallFilterMps;
  if (!(label > 0)) return 0;
  const scale = Math.max(1e-3, acquisitionLimits(input).color.scaleMps);
  const cut = mainColorWallFilterCutoffMps(label, scale);
  const { min, max } = colorWallFilterRangeMps(scale);
  const atDefault = mainColorWallFilterCutoffMps(label, MAIN_DEFAULT_COLOR_SCALE_MPS);
  // the steepest filter: every request from it up replays the image
  if (cut >= max) return Math.max(cut, atDefault);
  // the mean removal: every request up to it but off replays the image
  if (cut > 0 && cut <= min && atDefault > 0) return Math.min(min, atDefault);
  // inside the range the only request that replays it, and off what cut nothing
  return cut;
}

/**
 * Reads a report back from raw JSON or from a pasted markdown that carries it in a ```json block. A version 1 report is
 * migrated to version 2: its colour wall filter becomes the request that replays the filter main's 2D colour gave its
 * label (`legacyColorWallFilterMps`).
 */
export function parseReviewReport(text: string): ReviewReport | null {
  const candidates: string[] = [text.trim()];
  const fenced = /```json\s*([\s\S]*?)```/g;
  let m: RegExpExecArray | null;
  while ((m = fenced.exec(text)) !== null) candidates.push(m[1]!.trim());
  for (const c of candidates) {
    try {
      const obj = JSON.parse(c) as Partial<ReviewReport>;
      if (
        obj &&
        obj.format === 'echotwin-review' &&
        obj.input &&
        typeof obj.caseId === 'string' &&
        Array.isArray(obj.markers)
      ) {
        // reports written before markers could sit on the 3D model carry image markers only
        const markers = (obj.markers as Partial<ReviewMarker>[]).map((mk) => ({
          ...mk,
          space: mk.space ?? 'image',
          torso: mk.torso ?? null,
          group: mk.group ?? null,
          parentId: mk.parentId ?? null,
          lasso: mk.lasso ?? null,
          region: mk.region ?? null,
          explain: mk.explain ?? null,
        }));
        // version 1, or no version at all: the wall filter was a label
        const legacy = ((obj.version as number | undefined) ?? 1) < 2;
        const input = legacy
          ? {
              ...obj.input,
              color: { ...obj.input.color, wallFilterMps: legacyColorWallFilterMps(obj.input) },
            }
          : obj.input;
        return { ...obj, input, markers, note: obj.note ?? '', version: 2 } as ReviewReport;
      }
    } catch {
      /* not this candidate */
    }
  }
  return null;
}

/** The report narrowed to one primary marker and its secondary points (decision 340): what «copiar este» puts on the clipboard. */
export function markerReport(r: ReviewReport, id: string): ReviewReport {
  const primary = r.markers.find((m) => m.id === id);
  const root = primary?.parentId ?? id;
  return { ...r, note: '', markers: r.markers.filter((m) => m.id === root || m.parentId === root) };
}
