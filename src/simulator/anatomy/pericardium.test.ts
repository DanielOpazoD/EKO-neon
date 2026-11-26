import assert from 'node:assert/strict';
import test from 'node:test';
import { loadCaseById } from '@/cases';
import { buildCaseModels, REST_PATIENT, type CaseModels } from './caseModels';
import { classifyHeart, computeHeartPose, heartAnchors, type HeartPose } from './heartModel';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import { makeSample, Structure } from './tissue';
import { ctx } from './classify/context';
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
  posteriorColumnDistance,
} from './classify/pericardium';
import { RAA_TIP_FROM_AV } from './atrialAppendages';
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

test('effusion: no heart tissue reaches the parietal layer over the beat, oblique sinus included, in every posture and breathing of the app, away from the posterior column', (t) => {
  // The sac is the heart's envelope pushed out by the effusion and still in the torso, but over the oblique sinus, where
  // the parietal layer rests on the heart: there its face lies between the full sac and the same thinned over the left
  // atrium's footprint, where the sinus' film over the heart of the moment puts it (parietalSdf). Probed along rays from
  // the sac's centre, fixed in the torso as the sac is: two per bin of its table in azimuth and one in polar angle, off
  // the bins' edges (where the slope of the bilinear read changes cell, and with it the distance by up to 0.4 mm near the
  // poles); every 2° over the right atrium and its appendage, from the atrium's centre to the appendage's tip seen from
  // the centre and 15° around, where the envelope turns fastest and the swing of early diastole carries the right atrium
  // toward the sac; and every 2° over the sinus, the directions of the table's rays where the full and the thinned faces
  // part, a cell of the table around. In eight phases (end-diastole, the swing's extremes 0.25 and 0.75, end-systole when
  // the atria are largest, 0.65 as the heart swings toward the right, and the eighths between):
  // - past the full sac's face, where no heart may be, no heart tissue in its first 0.25 cm, every 0.05 cm (on the
  //   table's rays); and where that face is the layer's, still and unbent by the column, fluid 0.05 cm inside it (on
  //   every ray);
  // - over the sinus, every 0.05 cm from 0.3 cm inside the thinned face to 0.3 cm past the first face the run leaves the
  //   sac by, that face where the classifier drew it in the phase (ctx.parietal): no heart tissue next to a point the sac
  //   leaves, no visceral layer or fat past the face, fluid 0.05 cm inside every face the run leaves the sac by, and the
  //   sinus' film, fluid 0.2 cm inside it (it is 0.3 cm; the margin is the run's step), but where the full sac sets
  //   the face.
  // Laid on the thinned sac alone, the sinus' face ran through the heart: the coronary sinus and the visceral layer over it
  // stood up to 0.57 cm past it in the resting posture, and supine in inspiration the left atrium's wall, its cavity and
  // the fat up to 0.91 cm, a ray from the atrium crossing the face twice with a centimetre of fluid beyond the heart
  // between the crossings; with the atrium's body 2.0 cm short in the envelope the interatrial septum reached the face
  // supine in inspiration. The sweep of the still face alone, from 0.3 cm inside the sac's radius, passed both: the
  // thinned face lay up to 1.9 cm deeper. With the appendage's term of the envelope 2.0 cm short the visceral layer met
  // the parietal without fluid from phase 0.62 to 0.79, and 2.2 cm short the atrial wall stood 0.15 cm past it at
  // 0.67-0.75: probed past the face alone, on the table's rays in three phases, both passed. Within 2·PERICARDIUM_CM of
  // the descending aorta and the vertebral body the parietal layer is held off them (decision 273) and meets the visceral
  // layer without fluid: not probed. Nor is fluid asked for where the column bends the layer. The great vessels leave the
  // sac.
  const LEAVE_THE_SAC: ReadonlySet<number> = new Set([
    Structure.PericardialEffusion,
    Structure.AscendingAorta,
    Structure.AorticArch,
    Structure.BrachiocephalicArtery,
    Structure.LeftCommonCarotid,
    Structure.LeftSubclavian,
    Structure.PulmonaryArtery,
    Structure.PulmonaryVein,
    Structure.Svc,
    Structure.Ivc,
    Structure.HepaticVein,
    Structure.DescendingAorta,
  ]);
  const SECTOR_STEP = Math.PI / 90,
    SECTOR_MARGIN = Math.PI / 12;
  const caseDef = loadCaseById('pericardial-effusion-tamponade');
  let past = 0,
    inside = 0,
    overSinus = 0,
    sinusFaces = 0;
  // free breathing and breath-hold keep the expiratory heart (thoraxModel.ts): six hearts in all
  for (const position of ['left-lateral', 'supine', 'subcostal-supine'] as const)
    for (const respiration of ['expiration', 'inspiration'] as const) {
      const m = buildCaseModels(caseDef, { position, respiration, headElevationDeg: 0 });
      const A = heartAnchors(m.heart);
      const sac = A.parietal!;
      const la = A.laCenter,
        lr = A.laR;
      const eff = m.heart.anatomy.pericardium.effusionCm;
      // at a point fixed in the torso, NaN by the column: the full sac held off the column (decision 273), past which the
      // face never lies wherever the heart is, and the face with the heart far from it, the still sac thinned over the
      // atrium (parietalSdf), where it stays while the heart is; `held` false leaves the column out
      const column = (x: number, y: number, z: number): number =>
        posteriorColumnDistance(
          x,
          y,
          z,
          A.colU.x,
          A.colU.y,
          A.colU.z,
          A.colAorta.x,
          A.colAorta.y,
          A.colAorta.z,
          A.colSpine.x,
          A.colSpine.y,
          A.colSpine.z,
        );
      const fullSac = (x: number, y: number, z: number, held = true): number => {
        const colDist = column(x, y, z);
        if (colDist < 2 * PERICARDIUM_CM) return NaN;
        const col = held ? 2 * PERICARDIUM_CM - colDist : -Infinity;
        return smax(parietalSacDistance(sac, x, y, z), col, PARIETAL_COLUMN_BLEND_CM);
      };
      const stillFace = (x: number, y: number, z: number, held = true): number => {
        const colDist = column(x, y, z);
        if (colDist < 2 * PERICARDIUM_CM) return NaN;
        return parietalSdf(
          eff,
          parietalSacDistance(sac, x, y, z),
          Infinity,
          (la.y - y) / lr.y,
          ((x - la.x) / lr.x) ** 2 + ((z - la.z) / lr.z) ** 2,
          held ? colDist : Infinity,
        );
      };
      // the still points probed in every phase: [x, y, z, par] past the face, [x, y, z] inside it; and over the sinus the
      // runs of [x, y, z, full sac, column bends it] along each ray, walked in every phase
      const pastPoints: number[] = [],
        insidePoints: number[] = [],
        sinusRuns: number[][] = [];
      const sinusDirections: [number, number][] = [];
      const probe = (phi: number, th: number, probePast: boolean): void => {
        const dx = Math.sin(phi) * Math.cos(th),
          dy = Math.sin(phi) * Math.sin(th),
          dz = Math.cos(phi);
        const full = (t: number): number => fullSac(sac.cx + dx * t, sac.cy + dy * t, sac.cz + dz * t);
        // the radii of the table's cell the ray runs through bound the full sac's face
        const fi = (((th + 2 * Math.PI) % (2 * Math.PI)) * PARIETAL_AZ_BINS) / (2 * Math.PI);
        const i0 = Math.floor(fi) % PARIETAL_AZ_BINS,
          i1 = (i0 + 1) % PARIETAL_AZ_BINS,
          j0 = Math.min(PARIETAL_POLAR_BINS - 2, Math.floor((phi * (PARIETAL_POLAR_BINS - 1)) / Math.PI));
        const corners = [j0, j0 + 1].flatMap((j) => [i0, i1].map((i) => sac.r[j * PARIETAL_AZ_BINS + i]!));
        const tOut = Math.max(...corners) + 0.6;
        // past the full sac is past the face wherever the heart is
        if (probePast)
          for (let t = Math.min(...corners) - 0.3; t < tOut; t += 0.05) {
            const par = full(t);
            if (par >= 0 && par < 0.25) pastPoints.push(sac.cx + dx * t, sac.cy + dy * t, sac.cz + dz * t, par);
          }
        // the run over the sinus, where the still faces part: from 0.3 cm inside the thinned face, which lies up to the
        // effusion less the sinus' film inside the full one, to 0.25 cm past the full face
        const run: number[] = [];
        let parts = false;
        for (let t = Math.min(...corners) - eff + Math.min(eff, OBLIQUE_SINUS_EFFUSION_CM) - 0.6; t < tOut; t += 0.05) {
          const x = sac.cx + dx * t,
            y = sac.cy + dy * t,
            z = sac.cz + dz * t;
          const lo = fullSac(x, y, z),
            hi = stillFace(x, y, z);
          if (Number.isNaN(lo)) continue;
          if (!run.length && hi < -0.3) continue;
          const bent = lo !== fullSac(x, y, z, false) || hi !== stillFace(x, y, z, false);
          run.push(x, y, z, lo, bent ? 1 : 0);
          parts ||= lo < hi;
          if (lo >= 0.25) break;
        }
        if (parts) {
          sinusRuns.push(run);
          sinusDirections.push([phi, th]);
        }
        // the outermost point 0.05 cm inside the full sac's face: stepped in from beyond it, then halved to 5e-5 cm
        let t = tOut;
        while (t > 0 && !(full(t) < -0.05)) t -= 0.05;
        if (t <= 0 || Number.isNaN(full(t + 0.05))) return; // the face here is the column's
        let lo = t,
          hi = t + 0.05;
        for (let k = 0; k < 10; k++) {
          const mid = (lo + hi) / 2;
          if (full(mid) < -0.05) lo = mid;
          else hi = mid;
        }
        const x = sac.cx + dx * lo,
          y = sac.cy + dy * lo,
          z = sac.cz + dz * lo;
        // where that is the face, still and not bent by the column (over the sinus the runs ask for the fluid)
        if (stillFace(x, y, z) === parietalSacDistance(sac, x, y, z)) insidePoints.push(x, y, z);
      };
      // the table's rays
      for (let j = 0; j < PARIETAL_POLAR_BINS - 1; j++)
        for (let i = 0; i < 2 * PARIETAL_AZ_BINS; i++)
          probe((Math.PI * (j + 0.5)) / (PARIETAL_POLAR_BINS - 1), (Math.PI * (i + 0.5)) / PARIETAL_AZ_BINS, true);
      // the sinus' sector: the table's rays where the still faces part, a cell around, the azimuths taken around their mean
      assert.ok(sinusDirections.length > 0, `${position}, ${respiration}: no ray over the oblique sinus`);
      const tableSinus = [...sinusDirections];
      const thMean = Math.atan2(
        tableSinus.reduce((a, [, th]) => a + Math.sin(th), 0),
        tableSinus.reduce((a, [, th]) => a + Math.cos(th), 0),
      );
      const sinusPhi = tableSinus.map(([phi]) => phi),
        sinusTh = tableSinus.map(([, th]) => thMean + Math.atan2(Math.sin(th - thMean), Math.cos(th - thMean)));
      const cellPhi = Math.PI / (PARIETAL_POLAR_BINS - 1),
        cellTh = (2 * Math.PI) / PARIETAL_AZ_BINS;
      for (
        let phi = Math.max(SECTOR_STEP, Math.min(...sinusPhi) - cellPhi);
        phi < Math.min(Math.PI - SECTOR_STEP, Math.max(...sinusPhi) + cellPhi);
        phi += SECTOR_STEP
      )
        for (let th = Math.min(...sinusTh) - cellTh; th < Math.max(...sinusTh) + cellTh; th += SECTOR_STEP)
          probe(phi, th, false);
      // the right atrium's sector: its centre, its appendage's base and the appendage's tip seen from the centre, the
      // azimuths taken around the tip's
      const tip = {
        x: A.avCenter.x + RAA_TIP_FROM_AV[0],
        y: A.avCenter.y + RAA_TIP_FROM_AV[1],
        z: A.avCenter.z + RAA_TIP_FROM_AV[2],
      };
      const thTip = Math.atan2(tip.y - sac.cy, tip.x - sac.cx);
      const phis: number[] = [],
        ths: number[] = [];
      for (const p of [A.raCenter, A.raaBase, tip]) {
        const th = Math.atan2(p.y - sac.cy, p.x - sac.cx);
        phis.push(Math.atan2(Math.hypot(p.x - sac.cx, p.y - sac.cy), p.z - sac.cz));
        ths.push(thTip + Math.atan2(Math.sin(th - thTip), Math.cos(th - thTip)));
      }
      for (let phi = Math.min(...phis) - SECTOR_MARGIN; phi < Math.max(...phis) + SECTOR_MARGIN; phi += SECTOR_STEP)
        for (let th = Math.min(...ths) - SECTOR_MARGIN; th < Math.max(...ths) + SECTOR_MARGIN; th += SECTOR_STEP)
          probe(phi, th, false);
      for (const ph of [0, 0.125, 0.25, 0.375, endSystole(m), 0.65, 0.75, 0.875]) {
        const hp = computeHeartPose(m.heart, cycleStateAt(m.tables, ph));
        const where = (x: number, y: number, z: number): string =>
          `${position}, ${respiration}, phase ${ph.toFixed(3)}, (${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)})`;
        for (let k = 0; k < pastPoints.length; k += 4) {
          const x = pastPoints[k]!,
            y = pastPoints[k + 1]!,
            z = pastPoints[k + 2]!,
            par = pastPoints[k + 3]!;
          past++;
          if (!classifyHeart(m.heart, hp, x, y, z, s) || LEAVE_THE_SAC.has(s.structure)) continue;
          // the parietal layer itself is pericardium: only a visceral layer past it counts
          if (!(s.structure === Structure.Pericardium && par < PERICARDIUM_CM))
            assert.fail(`${where(x, y, z)}: structure ${s.structure} ${par.toFixed(3)} cm past the parietal layer`);
        }
        for (let k = 0; k < insidePoints.length; k += 3) {
          const x = insidePoints[k]!,
            y = insidePoints[k + 1]!,
            z = insidePoints[k + 2]!;
          inside++;
          if (!classifyHeart(m.heart, hp, x, y, z, s)) assert.fail(`${where(x, y, z)}: nothing inside the sac`);
          if (!LEAVE_THE_SAC.has(s.structure))
            assert.fail(`${where(x, y, z)}: structure ${s.structure} 0.05 cm inside the parietal layer, not fluid`);
        }
        // over the sinus, outward along each run, the face as the classifier drew it in this phase, to 0.3 cm past where
        // the run first leaves the sac: tissue crossing the face lies against it. The still faces are not star-shaped from
        // the centre where the thinning fades, so a run may leave the sac and enter it again with nothing of the heart
        // about; what may not be is heart tissue next to a point the sac leaves.
        for (const run of sinusRuns) {
          // the last sample but the vessels that leave the sac: heart tissue, inside the face, past it
          let last: 'heart' | 'in' | 'past' | '' = '',
            heart = Structure.None,
            after = -1;
          // since the run last entered the sac, the structures at its outermost points 0.05 and 0.2 cm inside the face
          // (−1 where not asked for), and where; and whether the full sac sets the face, at the last point inside it
          let in05 = -1,
            in20 = -1,
            at05 = '',
            at20 = '',
            fullFace = false;
          for (let k = 0; k < run.length && after < 6; k += 5) {
            const x = run[k]!,
              y = run[k + 1]!,
              z = run[k + 2]!,
              full = run[k + 3]!,
              bent = run[k + 4]! === 1;
            overSinus++;
            if (after >= 0) after++;
            ctx.parietal = NaN;
            const hit = classifyHeart(m.heart, hp, x, y, z, s);
            const st = hit ? s.structure : Structure.None;
            if (hit && st !== Structure.PericardialEffusion && LEAVE_THE_SAC.has(st)) {
              last = '';
              continue;
            }
            if (hit && !SAC.has(st)) {
              if (last === 'past' || full >= 0)
                assert.fail(`${where(x, y, z)}: structure ${st} past the parietal layer over the oblique sinus`);
              last = 'heart';
              heart = st;
              continue;
            }
            // the pericardium's block drew or left the point (outside the bounding sphere it does not run)
            const par = Number.isNaN(ctx.parietal) ? Infinity : ctx.parietal;
            if (par >= 0) {
              if (last === 'heart')
                assert.fail(
                  `${where(x, y, z)}: structure ${heart} against the parietal layer over the oblique sinus`,
                );
              // the parietal layer itself is pericardium: a visceral layer or fat past it is not
              if (st === Structure.EpicardialFat || (st === Structure.Pericardium && par >= PERICARDIUM_CM))
                assert.fail(
                  `${where(x, y, z)}: structure ${st} ${par.toFixed(3)} cm past the parietal layer over the oblique sinus`,
                );
              if (last === 'in') {
                sinusFaces++;
                if (after < 0) after = 0;
                if (in05 >= 0 && !LEAVE_THE_SAC.has(in05))
                  assert.fail(
                    `${at05}: structure ${in05} 0.05 cm inside the parietal layer over the oblique sinus, not fluid`,
                  );
                // the film is 0.3 cm, asked for 0.2 cm in: a step of the run beyond the parietal layer's distance
                if (!fullFace && in20 >= 0 && !LEAVE_THE_SAC.has(in20))
                  assert.fail(
                    `${at20}: structure ${in20} 0.2 cm inside the parietal layer over the oblique sinus, not its film`,
                  );
              }
              last = 'past';
            } else {
              if (last !== 'in') in05 = in20 = -1;
              if (par < -0.05) {
                in05 = bent ? -1 : st;
                at05 = where(x, y, z);
              }
              if (par < -0.2) {
                in20 = bent ? -1 : st;
                at20 = where(x, y, z);
              }
              // where the full sac sets the face the heart may come nearer it than the sinus' film
              fullFace = full >= par;
              last = 'in';
            }
          }
        }
      }
    }
  t.diagnostic(
    `${past} samples past the full sac's face, ${inside} inside the still one; ${overSinus} over the sinus, ` +
      `${sinusFaces} faces`,
  );
  assert.ok(past > 400_000, `${past} samples past the parietal layer`);
  assert.ok(inside > 100_000, `${inside} samples inside it`);
  assert.ok(sinusFaces > 10_000, `${sinusFaces} faces over the sinus, ${overSinus} samples`);
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

  const sdf =
    /float parietalSdf\(float eff, float dFull, float dEpi, float behind, float footprint, float colDist\) \{([\s\S]*?)\n\}/.exec(
      GLSL_GENERATED,
    );
  assert.ok(sdf, 'parietalSdf missing from GLSL_GENERATED');
  const sdfJs = sdf[1]!
    .replace(/\bfloat\s+/g, 'let ')
    .replace(/(?<![.\w])(min|max)\(/g, 'Math.$1(');
  const glslSdf = new Function(
    'eff',
    'dFull',
    'dEpi',
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
    // dEpi over the range where the heart sets the face over the sinus, between the full and the thinned sac
    const a = [2.5 * (rnd() + 1), 3 * rnd(), 3 * rnd(), 2 * rnd(), 2 * (rnd() + 1), 3 * (rnd() + 1)] as const;
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
