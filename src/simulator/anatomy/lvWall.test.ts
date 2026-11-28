import assert from 'node:assert/strict';
import test from 'node:test';
import { CASE_INPUTS, loadCaseById } from '@/cases';
import { buildCaseModels, REST_PATIENT, type CaseModels } from './caseModels';
import { classifyHeart, computeHeartPose, heartAnchors, type HeartPose } from './heartModel';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import { makeSample, Structure } from './tissue';
import { measureModel } from './measureModel';
import { getMeasurementSpec } from '@/simulator/measurements/protocol';
import {
  lvTrabecularRelief,
  meanWallThicknessCm,
  papillaryFluting,
  septalShare,
  wallThicknessAt,
  WALL_MODULATION_GAIN,
} from './lvWall';
import { rvTrabecularDepth } from './rv';
import { mitralAnnulusDistance, mitralHalfThickness } from './mitralValve';
import { cuspHalfThickness } from './aorticValve';
import { RV_GROOVE_ANTERIOR_RAD, RV_GROOVE_INFERIOR_RAD } from './lvSegments';
import { GLSL_GENERATED } from '@/simulator/renderer/gpu/glslGenerated';
import { GLSL_HEART_CHAMBERS } from '@/simulator/renderer/gpu/glslHeartChambers';

/**
 * The wall thickness the cases declare is the one their standard calipers read: the septum (ivsd) is the wall between
 * the RV insertions, the free wall (lvpwd) the rest. Until the septal share followed the insertions it was centred on the
 * A4C septum, and HOCM (2.1 / 1.1 cm) drew 1.88 cm of septum and 1.36 cm of posterior wall on its PLAX caliper line.
 */

const deg = (d: number): number => (d * Math.PI) / 180;
const septal = (rvAzA: number, rvAzP: number) => (rvAzA + rvAzP) / 2;
const wrap = (a: number): number => a - 2 * Math.PI * Math.floor(a / (2 * Math.PI) + 0.5);

const models = new Map<string, CaseModels & { ed: HeartPose }>();
function modelsOf(id: string): CaseModels & { ed: HeartPose } {
  let m = models.get(id);
  if (!m) {
    const built = buildCaseModels(loadCaseById(id), REST_PATIENT);
    m = { ...built, ed: computeHeartPose(built.heart, cycleStateAt(built.tables, 0)) };
    models.set(id, m);
  }
  return m;
}

test('septalShare: 1 between the RV insertions, 0 on the free wall, a smooth step at each insertion', () => {
  const a = RV_GROOVE_ANTERIOR_RAD,
    p = RV_GROOVE_INFERIOR_RAD;
  let integral = 0,
    prev = septalShare(-Math.PI, a, p);
  const n = 7200;
  for (let i = 0; i <= n; i++) {
    const az = -Math.PI + (2 * Math.PI * i) / n;
    const s = septalShare(az, a, p);
    assert.ok(s >= 0 && s <= 1, `share ${s} at ${az}`);
    const inside = Math.abs(wrap(az - septal(a, p))) - (p - a) / 2; // < 0 on the septum
    if (inside < -0.2 - 1e-9) assert.equal(s, 1, `septum at ${((az * 180) / Math.PI).toFixed(1)}°`);
    if (inside > 0.2 + 1e-9) assert.equal(s, 0, `free wall at ${((az * 180) / Math.PI).toFixed(1)}°`);
    // continuous: the smoothstep's slope is at most 1.5 / 0.4 rad⁻¹
    assert.ok(Math.abs(s - prev) <= (1.5 / 0.4) * ((2 * Math.PI) / n) + 1e-12, `jump at ${az}`);
    prev = s;
    if (i < n) integral += s * ((2 * Math.PI) / n);
  }
  assert.ok(Math.abs(septalShare(a, a, p) - 0.5) < 1e-12 && Math.abs(septalShare(p, a, p) - 0.5) < 1e-12);
  // the share integrates to the septal arc, the mean the shell volume carries
  assert.ok(Math.abs(integral - (p - a)) < 1e-6, `∫share = ${integral} vs arc ${p - a}`);
  assert.ok(Math.abs(meanWallThicknessCm(2.1, 1.1) - (1.1 + (integral / (2 * Math.PI)) * 1.0)) < 1e-6);
});

test('wallThicknessAt: the declared ivsd over the whole septum and lvpwd over the free wall, within the modulation', () => {
  // over the basal half the axial taper is 1: what remains is the low-frequency modulation, at most ±4 % (a third of the
  // 12 % the protocol grants SIVd and PPd); the old A4C-centred share left the PLAX septum at 0.75 of ivsd
  const bound = 0.04 + 1e-9;
  for (const input of CASE_INPUTS) {
    const { heart } = modelsOf(input.id);
    const { ivsd, lvpwd } = heart.lv;
    // the insertions the segments (and the GPU) are bounded by
    const { rvAzA: a, rvAzP: p } = heartAnchors(heart);
    for (let i = 0; i < 360; i++) {
      const az = wrap(deg(i));
      const inside = Math.abs(wrap(az - septal(a, p))) - (p - a) / 2;
      if (Math.abs(inside) < 0.2) continue; // the transitions at the insertions
      const declared = inside < 0 ? ivsd : lvpwd;
      for (const lf of [0.2, 0.3, 0.4, 0.5]) {
        const t = wallThicknessAt(heart, 1, az, lf, 1);
        assert.ok(
          Math.abs(t / declared - 1) <= bound,
          `${input.id}: ${t.toFixed(3)} cm at ${i}° (level ${lf}) against the declared ${declared} cm`,
        );
      }
    }
  }
});

test('PLAX, PSAX and A4C caliper lines read the declared IVSd and PWd at end-diastole in the 12 cases', () => {
  const SEPT = [Structure.LvWallSeptal, Structure.LvWallAnterior];
  const FREE = [Structure.LvWallLateral, Structure.LvWallInferior];
  const s = makeSample();
  /** run of the wall first met along d from c0 (the caliper from endocardium to epicardium) */
  const wallRun = (m: CaseModels & { ed: HeartPose }, c0: number[], d: number[], structs: Structure[]) => {
    const step = 0.005;
    const at = (t: number): boolean =>
      classifyHeart(m.heart, m.ed, c0[0]! + d[0]! * t, c0[1]! + d[1]! * t, c0[2]! + d[2]! * t, s) &&
      structs.includes(s.structure);
    let t = 0;
    while (t < 7 && !at(t)) t += step;
    const from = t;
    while (t < 7 && at(t + step)) t += step;
    return t - from + step;
  };
  // the PLAX's anteroseptal direction (viewDefinitions.ts); the parasternal short axes cut the same walls along it
  const u = [-0.5, 0.866, 0];
  const back = [0.5, -0.866, 0];
  // in the protocol's views the drawing takes at most half of the 12 % tolerance and leaves the other half to the
  // caliper; the A4C, where the protocol takes no wall thickness, within the tolerance itself
  const PROTOCOL_VIEW = 0.06,
    A4C = 0.12;
  let hocmRatio = NaN;
  for (const input of CASE_INPUTS) {
    const m = modelsOf(input.id);
    const { ivsdCm: S, lvpwdCm: P } = m.caseDef.anatomy.lv;
    const rows: [string, number, number, number][] = [];
    // the protocol's level (mitral tips) and a more apical one; PLAX through the axis, A4C in its plane y = −0.5
    for (const z of [2.0, 3.2])
      rows.push(
        [`PLAX septum at z ${z}`, wallRun(m, [0, 0, z], u, SEPT), S, PROTOCOL_VIEW],
        [`PLAX posterior wall at z ${z}`, wallRun(m, [0, 0, z], back, FREE), P, PROTOCOL_VIEW],
        [`A4C septum at z ${z}`, wallRun(m, [0, -0.5, z], [-1, 0, 0], SEPT), S, A4C],
        [`A4C lateral wall at z ${z}`, wallRun(m, [0, -0.5, z], [1, 0, 0], FREE), P, A4C],
      );
    // the papillary short axis, aimed at 4.6 cm of the reference 8.6 cm ventricle (viewDefinitions.ts)
    const zPm = (4.6 / 8.6) * m.heart.lv.lengthCm;
    rows.push(
      ['PSAX-PM septum', wallRun(m, [0, 0, zPm], u, SEPT), S, PROTOCOL_VIEW],
      ['PSAX-PM posterior wall', wallRun(m, [0, 0, zPm], back, FREE), P, PROTOCOL_VIEW],
    );
    for (const [name, value, truth, tol] of rows) {
      const err = value / truth - 1;
      assert.ok(
        Math.abs(err) <= tol,
        `${input.id} ${name}: ${value.toFixed(2)} cm drawn, ${truth} declared (${(err * 100).toFixed(1)} %)`,
      );
    }
    if (input.id === 'hocm-sam') hocmRatio = rows[0]![1] / rows[1]![1];
  }
  // asymmetric septal hypertrophy: SIV/PP 2.1 / 1.1 = 1.91, as the case's impression states (it read 1.38)
  assert.ok(Math.abs(hocmRatio / (2.1 / 1.1) - 1) < 0.08, `HOCM SIV/PP ${hocmRatio.toFixed(2)}`);
});

test('measureModel reports SIVd and PPd on the PLAX caliper line against the declared values', () => {
  for (const id of ['hocm-sam', 'normal-excellent-window']) {
    const c = loadCaseById(id);
    const rows = measureModel(c, REST_PATIENT, 2000).rows;
    for (const [measurementId, truth] of [
      ['ivsd', c.anatomy.lv.ivsdCm],
      ['lvpwd', c.anatomy.lv.lvpwdCm],
    ] as const) {
      const r = rows.find((x) => x.id === `${measurementId}-plax`);
      assert.ok(r, `missing row ${measurementId}-plax`);
      // the tolerance the student's caliper is graded with: the case's when it requires the measurement
      const tol =
        (c.requiredMeasurements.find((q) => q.measurementId === measurementId)?.tolerancePct ??
          getMeasurementSpec(measurementId)!.tolerancePct) / 100;
      assert.ok(Math.abs(r.lo - truth * (1 - tol)) < 1e-9 && Math.abs(r.hi - truth * (1 + tol)) < 1e-9);
      assert.equal(r.verdict, 'ok', `${id} ${r.id} = ${r.value.toFixed(2)} cm, declared ${truth}`);
      assert.ok(Math.abs(r.value / truth - 1) <= 0.06, `${id} ${r.id} = ${r.value.toFixed(2)} cm`);
    }
  }
});

test('GLSL mirror: septalShare evaluates as the TypeScript one and wallThicknessAt uses it and the same modulation', () => {
  const src = /float septalShare\(float az, float rvAzA, float rvAzP\) \{([\s\S]*?)\n\}/.exec(GLSL_GENERATED);
  assert.ok(src, 'septalShare missing from GLSL_GENERATED');
  // the scalar GLSL is JavaScript once its types and built-ins are spelled out
  const js = src[1]!
    .replace(/\bfloat\s+/g, 'let ')
    .replace(/(?<![.\w])(abs|floor|min|max)\(/g, 'Math.$1(')
    .replace(/\bPI\b/g, 'Math.PI');
  const glsl = new Function('az', 'rvAzA', 'rvAzP', js) as (a: number, b: number, c: number) => number;
  for (const [a, p] of [
    [RV_GROOVE_ANTERIOR_RAD, RV_GROOVE_INFERIOR_RAD],
    [1.3, 3.9],
  ] as const)
    for (let i = -400; i <= 400; i++) {
      const az = (i / 400) * Math.PI * 1.05;
      assert.ok(Math.abs(glsl(az, a, p) - septalShare(az, a, p)) < 1e-12, `az ${az}`);
    }
  const wall = /float wallThicknessAt\([^)]*\) \{([\s\S]*?)\n\}/.exec(GLSL_HEART_CHAMBERS);
  assert.ok(wall, 'wallThicknessAt missing from GLSL_HEART_CHAMBERS');
  const body = wall[1]!;
  assert.ok(body.includes('septalShare(az, RV_AZA, RV_AZP)'), 'GLSL septum not bounded by the RV insertions');
  assert.ok(!/0\.5 - 0\.5 \* cos\(az\)/.test(body), 'GLSL keeps the A4C-centred share');
  assert.ok(body.includes(`1.0 + ${WALL_MODULATION_GAIN.toFixed(8)} * (lat(`), 'GLSL modulation gain differs');
});

test('Phase B micro-anatomy: basal caliper protection, LV/RV trabeculae, rough-zone mitral, Arantius nodule & fibrous trigones', () => {
  // 1) LV trabeculae remain strictly 0 across the entire basal/mid PLAX caliper zone (levelFrac <= 0.45)
  assert.equal(lvTrabecularRelief(0.8, -4.0, 0.25, 0.3), 0);
  assert.equal(lvTrabecularRelief(0.8, -4.0, 0.45, 0.3), 0);
  // In mid-apical zone, free/lateral wall relief exceeds mid-septal relief at the same ridge phase (Δaz = π)
  const freeApical = Math.abs(lvTrabecularRelief(0.4, -1.2, 0.78, 0.25));
  const septApical = Math.abs(lvTrabecularRelief(0.4 - Math.PI, -1.2, 0.78, 0.25));
  assert.ok(freeApical > septApical && freeApical > 0.02);

  // 2) Papillary longitudinal fluting is bounded (<= 0.5 mm) and zero-mean across symmetries
  assert.ok(Math.abs(papillaryFluting(0.8, -1.1, 3.2)) <= 0.045 + 1e-9);

  // 3) RV coarse trabecular depth is 0 in basal inflow/outflow and positive on apical ridges
  assert.equal(rvTrabecularDepth(2.0, 8.0, 0.5), 0);
  const maxRvTrab = Math.max(...[0, 0.15, 0.3, 0.45, 0.6].map((a) => rvTrabecularDepth(6.4, 8.0, a)));
  assert.ok(maxRvTrab > 0.1, `expected apical RV trabecular ridge > 0.1 cm, got ${maxRvTrab}`);

  // 4) Mitral leaflet rough/coaptation zone (frac ~ 0.9) is > 1.5x thicker than mid-belly clear zone (frac = 0.35)
  const clearHalf = mitralHalfThickness(0.14, 0.35, 1);
  const roughHalf = mitralHalfThickness(0.14, 0.92, 1);
  assert.ok(roughHalf > clearHalf * 1.5, `expected roughHalf ${roughHalf} > 1.5 * clearHalf ${clearHalf}`);

  // 5) Aortic cusp nodule of Arantius thickens central free edge (aq = 0, frac = 1) vs lateral free edge (aq = 0.5, frac = 1)
  const arantiusHalf = cuspHalfThickness(0.1, 1.0, 1, 0);
  const lunulaHalf = cuspHalfThickness(0.1, 1.0, 1, 0.5);
  assert.ok(arantiusHalf > lunulaHalf + 0.018, `expected Arantius nodule > lunula (${arantiusHalf} vs ${lunulaHalf})`);

  // 6) Mitral annulus swells focally at the anterolateral & posteromedial fibrous trigones (±uc, D)
  const { ed } = modelsOf('normal-excellent-window');
  const mv = ed.valves.mitral;
  const uc = Math.sqrt(mv.R * mv.R - mv.D * mv.D);
  const trigX = mv.cx + mv.ux * mv.D - mv.uy * uc;
  const trigY = mv.cy + mv.uy * mv.D + mv.ux * uc;
  const trigZ = mv.cz + mv.saddle * ((uc * uc) / (uc * uc + mv.D * mv.D)) + mv.lift;
  const postX = mv.cx - mv.ux * mv.R;
  const postY = mv.cy - mv.uy * mv.R;
  const postZ = mv.cz;
  const dTrig = mitralAnnulusDistance(trigX, trigY, trigZ, mv, 0.11);
  const dPost = mitralAnnulusDistance(postX, postY, postZ, mv, 0.11);
  assert.ok(dTrig < dPost - 0.03, `expected thicker fibrous trigone (${dTrig} vs posterior ${dPost})`);
});
