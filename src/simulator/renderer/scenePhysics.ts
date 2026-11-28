import type { CaseDefinition } from '@/cases/schema';
import type { ArtifactSettings } from './postprocess/consolePipeline';
import type { AcquisitionSettings, ScenePhysics } from './types';

/**
 * The case-configurable artifacts (spec 12) at their intensity. Side lobe, mirror and beam width act in the console and in
 * the renderer's lateral response; the near-field clutter raises the physics' clutter term. The geometric artifacts (rib,
 * lung and calcium shadows) come from the anatomy and are not here.
 */
export interface ArtifactLevels extends ArtifactSettings {
  clutter: number;
}

export function caseArtifactLevels(c: CaseDefinition): ArtifactLevels {
  const level = (type: string) =>
    c.artifacts
      .filter((x) => x.enabled && x.type === type)
      .reduce((m, x) => Math.max(m, x.intensity), 0);
  return {
    sideLobe: level('side-lobe'),
    mirror: level('mirror'),
    beamWidth: level('beam-width'),
    clutter: level('near-field-clutter'),
  };
}

export interface ScenePhysicsOptions {
  /** The artifact lab's levels in place of the case's. */
  artifacts?: ArtifactLevels;
  /** Scatterer realization; the case seed by default. */
  seed?: number;
  /** The frame being formed (the flowing blood's speckle changes with it, decision 163); 0 by default. */
  bloodFrame?: number;
}

const clamp01 = (v: number): number => Math.max(0, Math.min(1, v));

/**
 * Derive spontaneous echo contrast ("smoke" / SEC) intensity in the LA/LAA and mid-apical LV strictly
 * from chamber anatomy and hemodynamics (no case-ID checks):
 *  - LA stasis rises when significant LA dilation (>65 mL) coincides with mitral inflow obstruction
 *    (MVA < 1.8 cm²) or atrial fibrillation.
 *  - LV stasis rises when severe LV dilation (EDV > 165 mL) coincides with severe systolic dysfunction
 *    (EF < 34 %).
 */
export function computeChamberStasis(c: CaseDefinition): { secLa: number; secLv: number } {
  const laDil = clamp01((c.anatomy.la.volumeMl - 65) / 40);
  const mvArea = c.hemodynamics.mvEffectiveAreaCm2;
  const msObstr = mvArea !== undefined && mvArea < 1.8 ? clamp01((1.8 - mvArea) / 0.8) : 0;
  const afStasis = c.rhythm.type === 'atrial-fibrillation' ? 0.72 : 0;
  const secLa = clamp01(laDil * Math.max(msObstr, afStasis));

  const edv = Math.max(1, c.physiology.edvMl);
  const efPct = Math.max(0, ((edv - c.physiology.esvMl) / edv) * 100);
  const lvDil = clamp01((edv - 165) / 90);
  const lvHypo = clamp01((34 - efPct) / 14);
  const secLv = clamp01(lvDil * lvHypo * 0.85);

  return { secLa, secLv };
}

/**
 * The physics of a case's scene, the one the app renders (decision 238). The clinical calibration, the goldens, the
 * backend comparison and the offline tools each wrote their own, and none carried the case's near-field clutter: the
 * normal case was calibrated against CAMUS with a clutter of 0.10 while the app showed it with 0.28.
 */
export function scenePhysicsFor(
  c: CaseDefinition,
  settings: Pick<AcquisitionSettings, 'frequencyMHz' | 'harmonics' | 'spatialCompound'>,
  opts: ScenePhysicsOptions = {},
): ScenePhysics {
  const a = opts.artifacts ?? caseArtifactLevels(c);
  const w = c.acousticWindow;
  const stasis = computeChamberStasis(c);
  // Trans-thoracic phase aberration across heterogeneous adipose/intercostal layers broadens the far-field
  // beam waist and raises near-sidelobe skirts; Tissue Harmonic Imaging suppresses it (formed past the body wall).
  const aberration =
    Math.min(1, 0.45 * w.chestWallAttenuation + 0.65 * w.obesityAttenuation) *
    (settings.harmonics ? 0.42 : 1);
  return {
    frequencyMHz: settings.frequencyMHz,
    harmonics: settings.harmonics,
    clutterLevel: Math.min(1, w.clutterLevel + 0.6 * a.clutter) + w.emphysemaScatter * 0.5,
    windowAttenuation: Math.min(1, w.chestWallAttenuation + 0.25 * w.obesityAttenuation),
    seed: opts.seed ?? c.seed,
    beamWidth: Math.min(1, a.beamWidth + 0.22 * aberration),
    sideLobe: Math.min(1, a.sideLobe + 0.24 * aberration),
    bloodFrame: opts.bloodFrame ?? 0,
    spatialCompound: Math.max(0, Math.min(1, settings.spatialCompound ?? 0)),
    secLa: stasis.secLa,
    secLv: stasis.secLv,
  };
}

/** The console's share of the artifact levels. */
export function consoleArtifacts(a: ArtifactLevels): ArtifactSettings {
  return { sideLobe: a.sideLobe, mirror: a.mirror, beamWidth: a.beamWidth };
}

/** A switchable source of echo the CPU tracer can form a frame without (decision 340). */
export type EchoSourceId = 'clutter' | 'chestWall' | 'sideLobe' | 'beamWidth';

/**
 * The scene physics without one switchable source of echo (decision 340): what review mode's point explanation forms
 * the wedge with to measure that source's share. Null when the source is already off.
 */
export function physicsWithout(physics: ScenePhysics, source: EchoSourceId): ScenePhysics | null {
  const off: Record<EchoSourceId, Partial<ScenePhysics>> = {
    clutter: { clutterLevel: 0 },
    chestWall: { windowAttenuation: 0 },
    sideLobe: { sideLobe: 0 },
    beamWidth: { beamWidth: 0 },
  };
  const next = { ...physics, ...off[source] };
  const same = (Object.keys(off[source]) as (keyof ScenePhysics)[]).every(
    (k) => (physics[k] ?? 0) === (next[k] ?? 0),
  );
  return same ? null : next;
}
