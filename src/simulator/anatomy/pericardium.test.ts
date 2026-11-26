import assert from 'node:assert/strict';
import test from 'node:test';
import { loadCaseById } from '@/cases';
import { buildCaseModels, REST_PATIENT, type CaseModels } from './caseModels';
import { classifyHeart, computeHeartPose, heartAnchors, type HeartPose } from './heartModel';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import { makeSample, Structure } from './tissue';
import { smax } from './sdf';
import {
  OBLIQUE_SINUS_BEHIND,
  OBLIQUE_SINUS_EFFUSION_CM,
  OBLIQUE_SINUS_FOOTPRINT_FROM,
  OBLIQUE_SINUS_FOOTPRINT_TO,
  PARIETAL_AZ_BINS,
  PARIETAL_COLUMN_BLEND_CM,
  PARIETAL_POLAR_BINS,
  PERICARDIUM_CM,
  parietalSacDistance,
  parietalSdf,
} from './classify/pericardium';
import { GLSL_GENERATED } from '@/simulator/renderer/gpu/glslGenerated';
import { GLSL_HEART_CHAMBERS } from '@/simulator/renderer/gpu/glslHeartChambers';
import { PARAM_OFFSET } from '@/simulator/renderer/gpu/paramLayout';

/**
 * The pericardium (classify/pericardium.ts): over an effusion the parietal layer is a still sac the heart beats and
 * swings inside, and over the right ventricle the visceral layer wraps the free wall as thick as it is drawn.
 */

const models = new Map<string, CaseModels>();
function modelsOf(id: string): CaseModels {
  let m = models.get(id);
  if (!m) {
    m = buildCaseModels(loadCaseById(id), REST_PATIENT);
    models.set(id, m);
  }
  return m;
}
/** The phase of the beat where the ventricle is most contracted (end-systole). */
function endSystole(m: CaseModels): number {
  let best = 0,
    most = -1;
  for (let i = 0; i <= 400; i++) {
    const c = cycleStateAt(m.tables, i / 400).contraction;
    if (c > most) {
      most = c;
      best = i / 400;
    }
  }
  return best;
}

const s = makeSample();
const SAC: ReadonlySet<number> = new Set([
  Structure.EpicardialFat,
  Structure.Pericardium,
  Structure.PericardialEffusion,
]);
/** Structures met along a heart-frame ray (fixed in the torso: the swing is the heart's), in runs [structure, from, to]. */
function runsAlong(
  m: CaseModels,
  hp: HeartPose,
  o: readonly number[],
  d: readonly number[],
  length: number,
  step: number,
): [number, number, number][] {
  const runs: [number, number, number][] = [];
  for (let t = 0; t <= length; t += step) {
    const hit = classifyHeart(m.heart, hp, o[0]! + d[0]! * t, o[1]! + d[1]! * t, o[2]! + d[2]! * t, s);
    const st = hit ? s.structure : -1;
    const last = runs[runs.length - 1];
    if (last && last[0] === st) last[2] = t + step;
    else runs.push([st, t, t + step]);
  }
  return runs;
}

test('effusion: the parietal pericardium stays still over the beat and the fluid between it and the heart changes', () => {
  const m = modelsOf('pericardial-effusion-tamponade');
  const L = m.heart.lv.lengthCm;
  const es = endSystole(m);
  const phases = [0, 0.1, 0.2, 0.25, 0.3, 0.4, es, 0.7, 0.75, 0.8, 0.9];
  const rays: [string, number[], number[]][] = [
    ['inferior', [0, 0, 0.5 * L], [0, -1, 0]],
    ['lateral', [0, 0, 0.5 * L], [1, 0, 0]],
    ['anterior', [0, 0, 0.5 * L], [0, 1, 0]],
    ['right ventricular', [0, 0, 0.5 * L], [-1, 0, 0]],
    ['apical', [0, 0, 0.6 * L], [0, 0, 1]],
  ];
  const poses = phases.map((ph) => computeHeartPose(m.heart, cycleStateAt(m.tables, ph)));
  const fluid = new Map<string, number[]>();
  for (const [name, o, d] of rays) {
    const faces: number[] = [];
    const thick: number[] = [];
    for (const [i, ph] of phases.entries()) {
      const runs = runsAlong(m, poses[i]!, o, d, 10, 0.005);
      const lastSac = [...runs].reverse().find((r) => SAC.has(r[0]));
      assert.ok(lastSac, `${name}: no sac at phase ${ph}`);
      faces.push(lastSac[2]);
      thick.push(
        runs.filter((r) => r[0] === Structure.PericardialEffusion).reduce((a, r) => a + r[2] - r[1], 0),
      );
    }
    // the sac's outer face travelled 0.11-1.12 cm on these rays when it was rebuilt from the beating heart
    const travel = Math.max(...faces) - Math.min(...faces);
    assert.ok(travel < 0.2, `${name}: the parietal pericardium travels ${travel.toFixed(3)} cm over the beat`);
    fluid.set(name, thick);
  }
  const inferior = fluid.get('inferior')!,
    lateral = fluid.get('lateral')!;
  // measured at end-diastole, the fluid is the case's effusion (2.2 cm)
  for (const [name, t] of [
    ['inferior', inferior],
    ['lateral', lateral],
  ] as const)
    assert.ok(
      Math.abs(t[0]! - m.caseDef.anatomy.pericardium.effusionCm) < 0.05,
      `${name}: ${t[0]!.toFixed(3)} cm of fluid at end-diastole`,
    );
  // the echo-free space widens as the ventricle contracts away from the still sac: it was 2.20 cm in every phase
  const esIndex = phases.indexOf(es);
  assert.ok(
    inferior[esIndex]! - inferior[0]! > 0.15,
    `inferior fluid ${inferior[0]!.toFixed(3)} → ${inferior[esIndex]!.toFixed(3)} cm from end-diastole to end-systole`,
  );
  // and the heart swings toward the lateral sac and away from it (it moved 0.005 cm before)
  const swing = Math.max(...lateral) - Math.min(...lateral);
  assert.ok(swing > 0.5, `lateral fluid changes ${swing.toFixed(3)} cm over the beat`);
});

test('right ventricle: the visceral pericardium wraps the free wall through the beat, as thick as the wall is drawn', () => {
  // past the last sample of right ventricular wall, after any epicardial fat, comes the pericardium; at end-systole the
  // wall (35 % thicker) used to stand out of the sac on 15 of these 20 rays in the normal case and on all 20 with
  // pulmonary hypertension, and near the outflow tract, which descends with the base, it did with the systolic
  // thickness alone
  for (const id of ['normal-excellent-window', 'pulmonary-hypertension-rv']) {
    const m = modelsOf(id);
    const A = heartAnchors(m.heart);
    const es = endSystole(m);
    for (const ph of [0, 0.5 * es, 0.75 * es, es]) {
      const hp = computeHeartPose(m.heart, cycleStateAt(m.tables, ph));
      let rays = 0;
      for (const zf of [0.3, 0.45, 0.6, 0.75])
        for (const u of [0.25, 0.4, 0.5, 0.6, 0.75]) {
          const az = A.rvAzA + (A.rvAzP - A.rvAzA) * u;
          const z = hp.zAnn + zf * hp.lengthNow;
          const runs = runsAlong(m, hp, [0, 0, z], [Math.cos(az), Math.sin(az), 0], 9, 0.004);
          let k = -1;
          runs.forEach((r, i) => {
            if (r[0] === Structure.RvWall) k = i;
          });
          assert.ok(k >= 0, `${id} phase ${ph.toFixed(2)}: no free wall at level ${zf}, u ${u}`);
          rays++;
          let q = k + 1;
          while (runs[q]?.[0] === Structure.EpicardialFat) q++;
          const band = runs[q];
          assert.ok(
            band && band[0] === Structure.Pericardium && band[2] - band[1] >= 0.05,
            `${id} phase ${ph.toFixed(2)} (contraction ${hp.state.contraction.toFixed(2)}), level ${zf}, u ${u}: ` +
              `the free wall meets ${band ? band[0] : 'nothing'}`,
          );
        }
      assert.equal(rays, 20);
    }
  }
});

test('the sac is built once per case with an effusion, around the centre the shader reads it from', () => {
  const tamponade = heartAnchors(modelsOf('pericardial-effusion-tamponade').heart).parietal;
  assert.ok(tamponade, 'the tamponade case has a sac');
  const m = modelsOf('pericardial-effusion-tamponade').heart;
  assert.deepEqual([tamponade.cx, tamponade.cy, tamponade.cz], [m.boundCenter.x, m.boundCenter.y, m.boundCenter.z]);
  assert.equal(tamponade.r.length, PARIETAL_AZ_BINS * PARIETAL_POLAR_BINS);
  // the shader's block holds the whole table
  const base = PARAM_OFFSET['SAC_R']!;
  const next = Math.min(...Object.values(PARAM_OFFSET).filter((v) => v > base));
  assert.equal(next - base, tamponade.r.length);
  // the pole rows are one direction each
  for (const j of [0, PARIETAL_POLAR_BINS - 1])
    for (let i = 1; i < PARIETAL_AZ_BINS; i++)
      assert.equal(tamponade.r[j * PARIETAL_AZ_BINS + i], tamponade.r[j * PARIETAL_AZ_BINS]);
  // every direction reaches past the heart and stays inside the bounding sphere
  for (const r of tamponade.r) assert.ok(r > 3 && r < m.boundRadius, `radius ${r}`);
  assert.equal(heartAnchors(modelsOf('normal-excellent-window').heart).parietal, null);
});

test('GLSL mirror: parietalSacDistance and parietalSdf evaluate as the TypeScript ones', () => {
  const sac = heartAnchors(modelsOf('pericardial-effusion-tamponade').heart).parietal!;
  const body = /float parietalSacDistance\(float x, float y, float z\) \{([\s\S]*?)\n\}/.exec(GLSL_HEART_CHAMBERS);
  assert.ok(body, 'parietalSacDistance missing from GLSL_HEART_CHAMBERS');
  // the scalar GLSL is JavaScript once its types and built-ins are spelled out; the table reads as the shader's
  // float texture does
  const js = body[1]!
    .replace(/\b(float|int)\s+(?=[A-Za-z_])/g, 'let ')
    .replace(/\b(float|int)\(/g, '(')
    .replace(/(?<![.\w])(sqrt|sin|floor|min|max)\(/g, 'Math.$1(')
    .replace(/(?<![.\w])atan\(/g, 'Math.atan2(');
  const glsl = new Function(
    'x',
    'y',
    'z',
    'P',
    'BOUND_CX',
    'BOUND_CY',
    'BOUND_CZ',
    'SAC_R_BASE',
    'SAC_AZ_BINS',
    'SAC_POLAR_BINS',
    'PI',
    'TWO_PI',
    js,
  ) as (...a: unknown[]) => number;
  const P = (i: number): number => Math.fround(sac.r[i]!);
  const f32 = { ...sac, r: Float64Array.from(sac.r, (v) => Math.fround(v)) };
  let seed = 3;
  const rnd = (): number => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296) * 2 - 1;
  for (let k = 0; k < 4000; k++) {
    const x = sac.cx + 9 * rnd(),
      y = sac.cy + 9 * rnd(),
      z = sac.cz + 9 * rnd();
    const g = glsl(x, y, z, P, sac.cx, sac.cy, sac.cz, 0, PARIETAL_AZ_BINS, PARIETAL_POLAR_BINS, Math.PI, 2 * Math.PI);
    const t = parietalSacDistance(f32, x, y, z);
    assert.ok(Math.abs(g - t) < 1e-9, `(${x}, ${y}, ${z}): GLSL ${g} TS ${t}`);
  }
  // on the axis through the centre and at the seam of the azimuths
  for (const [x, y, z] of [
    [sac.cx, sac.cy, sac.cz + 5],
    [sac.cx, sac.cy, sac.cz - 5],
    [sac.cx + 5, sac.cy - 1e-9, sac.cz],
    [sac.cx + 5, sac.cy, sac.cz],
  ] as const) {
    const g = glsl(x, y, z, P, sac.cx, sac.cy, sac.cz, 0, PARIETAL_AZ_BINS, PARIETAL_POLAR_BINS, Math.PI, 2 * Math.PI);
    assert.ok(Number.isFinite(g) && Math.abs(g - parietalSacDistance(f32, x, y, z)) < 1e-9);
  }

  const sdf = /float parietalSdf\(float eff, float dFull, float behind, float footprint, float colDist\) \{([\s\S]*?)\n\}/.exec(
    GLSL_GENERATED,
  );
  assert.ok(sdf, 'parietalSdf missing from GLSL_GENERATED');
  const sdfJs = sdf[1]!
    .replace(/\bfloat\s+/g, 'let ')
    .replace(/(?<![.\w])(min|max)\(/g, 'Math.$1(');
  const glslSdf = new Function(
    'eff',
    'dFull',
    'behind',
    'footprint',
    'colDist',
    'smax',
    'OBLIQUE_SINUS_EFFUSION_CM',
    'OBLIQUE_SINUS_BEHIND',
    'OBLIQUE_SINUS_FOOTPRINT_FROM',
    'OBLIQUE_SINUS_FOOTPRINT_TO',
    'PERICARDIUM_CM',
    'PARIETAL_COLUMN_BLEND_CM',
    sdfJs,
  ) as (...a: unknown[]) => number;
  for (let k = 0; k < 2000; k++) {
    const a = [2.5 * (rnd() + 1), 3 * rnd(), 2 * rnd(), 2 * (rnd() + 1), 3 * (rnd() + 1)] as const;
    const g = glslSdf(
      ...a,
      smax,
      OBLIQUE_SINUS_EFFUSION_CM,
      OBLIQUE_SINUS_BEHIND,
      OBLIQUE_SINUS_FOOTPRINT_FROM,
      OBLIQUE_SINUS_FOOTPRINT_TO,
      PERICARDIUM_CM,
      PARIETAL_COLUMN_BLEND_CM,
    );
    assert.ok(Math.abs(g - parietalSdf(...a)) < 1e-12, `parietalSdf(${a.join(', ')})`);
  }
});
