import assert from 'node:assert/strict';
import test from 'node:test';
import { CASE_INPUTS, loadCaseById } from '@/cases';
import { buildCaseModels, REST_PATIENT, type CaseModels } from './caseModels';
import {
  classifyHeart,
  computeHeartPose,
  heartAnchors,
  lvEpicardialRadiusAt,
  type HeartPose,
} from './heartModel';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import { makeSample, Structure } from './tissue';
import {
  lvRegionalMotion,
  regionalMeanFraction,
  WALL_MOTION_BORDER_DEG,
  WALL_MOTION_BORDER_LEVEL,
} from './lvGeometry';
import { aha17FromCode, lvSegmentCode, RV_GROOVE_ANTERIOR_RAD, RV_GROOVE_INFERIOR_RAD } from './lvSegments';
import { ellipseFactor, lvProfileG, lvShapeFor } from './lvShape';
import { rvRadii } from './rv';
import { sampleTissueVelocity } from '@/simulator/doppler/flow-primitives/regurgitantJets';
import { GLSL_HEART, GLSL_HEART_CHAMBERS } from '@/simulator/renderer/gpu/glslHeart';
import { allocPacked, packScene, PARAM_OFFSET } from '@/simulator/renderer/gpu/paramLayout';
import { baseInput } from '@/simulator/core/baseInput';
import { scenePhysicsFor } from '@/simulator/renderer/scenePhysics';
import { polarSpecFor } from '@/simulator/renderer/types';
import { beamFrameFromPose, poseFromControl } from '@/simulator/probe/pose';
import { canonicalControl, VIEW_TARGETS } from '@/simulator/windows/viewTargets';

/**
 * Regional wall motion is one continuous field over the AHA segments (lvRegionalMotion): it moves the endocardium, sets
 * the thickening, places the RV crescent and scales the tissue velocity. Read per segment it was constant by pieces: in the
 * inferior infarction (inferior-rwma) at end-systole the endocardium stepped 7.6 mm and the epicardium 5.1 mm within 0.5°
 * at the inferior RV insertion, 5.6 and 4.0 mm within 0.25 % of the length at the mid-apical border; the akinetic mid
 * inferior segment (amplitude 0.1) thickened 17.6 % and its epicardium moved 1.1 mm outward.
 */

const SEPTAL = (RV_GROOVE_ANTERIOR_RAD + RV_GROOVE_INFERIOR_RAD) / 2;
/** heart-frame azimuth of an AHA angle (degrees; septal centre 180°, lateral near 0°, anterior near 90°) */
const azOf = (ahaDeg: number): number => SEPTAL + ((ahaDeg - 180) * Math.PI) / 180;

const LV_WALL = new Set([
  Structure.LvWallSeptal,
  Structure.LvWallAnterior,
  Structure.LvWallInferior,
  Structure.LvWallLateral,
  Structure.LvApex,
]);

interface Rwma extends CaseModels {
  ed: HeartPose;
  es: HeartPose;
  mid: HeartPose;
  esPhase: number;
  midPhase: number;
}
let rwmaCache: Rwma | null = null;
function rwma(): Rwma {
  if (!rwmaCache) {
    const built = buildCaseModels(loadCaseById('inferior-rwma'), REST_PATIENT);
    const { tables } = built;
    const esPhase = tables.endSystoleS / tables.rrS;
    const t = tables.timings;
    const midPhase = (t.ejectionStartS + 0.45 * (t.ejectionEndS - t.ejectionStartS)) / tables.rrS;
    rwmaCache = {
      ...built,
      ed: computeHeartPose(built.heart, cycleStateAt(tables, 0)),
      es: computeHeartPose(built.heart, cycleStateAt(tables, esPhase)),
      mid: computeHeartPose(built.heart, cycleStateAt(tables, midPhase)),
      esPhase,
      midPhase,
    };
  }
  return rwmaCache;
}

const s = makeSample();
/** the first run of LV wall along the radial ray at azimuth az and height z: [endocardium, epicardium] (cm, ±1e-5) */
function wallRun(m: CaseModels, pose: HeartPose, az: number, z: number): [number, number] {
  const c = Math.cos(az),
    sn = Math.sin(az);
  const wall = (r: number) =>
    classifyHeart(m.heart, pose, c * r, sn * r, z, s) && LV_WALL.has(s.structure);
  const step = 0.05;
  let r = 0.3;
  while (r < 6 && !wall(r)) r += step;
  let lo = r - step,
    hi = r;
  for (let i = 0; i < 13; i++) {
    const h = (lo + hi) / 2;
    if (wall(h)) hi = h;
    else lo = h;
  }
  const endo = hi;
  while (r < 8 && wall(r)) r += step;
  lo = r - step;
  hi = r;
  for (let i = 0; i < 13; i++) {
    const h = (lo + hi) / 2;
    if (wall(h)) lo = h;
    else hi = h;
  }
  return [endo, lo];
}
const zAt = (pose: HeartPose, levelFrac: number): number => pose.zAnn + levelFrac * pose.lengthNow;

/** amplitudes that exercise every segment: normal, hypokinetic, akinetic, dyskinetic and (clamped) hyperkinetic */
function mixedAmplitudes(): Float32Array {
  const amp = new Float32Array(18).fill(1);
  const values = [1, 0.1, -0.4, 0.6, 1.2, 0, 0.45, 1, 0.25, -0.1, 0.8, 0.05, 0.3, 1.1, -0.5, 0.7, 0.15];
  for (let id = 1; id <= 17; id++) amp[id] = values[id - 1]!;
  return amp;
}

test('lvRegionalMotion: each segment its own amplitude inside it, smooth borders between neighbours, 1 in a normal heart', () => {
  const amp = mixedAmplitudes();
  const clamped = (id: number) => Math.min(1, amp[id]!);
  let lo = Infinity,
    hi = -Infinity;
  for (let id = 1; id <= 17; id++) {
    lo = Math.min(lo, clamped(id));
    hi = Math.max(hi, clamped(id));
  }
  const W = WALL_MOTION_BORDER_DEG,
    H = WALL_MOTION_BORDER_LEVEL;
  // away from every border the field is the amplitude of the segment lvSegmentCode puts there (the cap reads level 1)
  const ringBorders = (levelFrac: number) =>
    levelFrac >= 2 / 3 ? [45, 135, 225, 315] : [0, 60, 120, 180, 240, 300];
  const levelBorders = [1 / 3, 2 / 3];
  let checked = 0;
  for (let i = 0; i < 720; i++) {
    const deg = i * 0.5;
    for (let k = 0; k <= 100; k++) {
      const lf = k / 100;
      const nearRing = ringBorders(lf).some((b) => {
        const d = Math.abs(deg - b);
        return Math.min(d, 360 - d) <= W + 1e-9;
      });
      const nearLevel = levelBorders.some((b) => Math.abs(lf - b) <= H + 1e-9) || (lf > 1 - H && lf < 1);
      if (nearRing || nearLevel) continue;
      const id = aha17FromCode(
        lvSegmentCode(azOf(deg), lf, RV_GROOVE_ANTERIOR_RAD, RV_GROOVE_INFERIOR_RAD),
      );
      assert.equal(lvRegionalMotion(amp, azOf(deg), lf), clamped(id), `AHA ${deg}° level ${lf}`);
      checked++;
    }
  }
  assert.ok(checked > 20000, `${checked} interior points`);
  // continuous with a bounded slope: a smoothstep across 2·W degrees (rings) or 2·H of the length (levels; the cap's is
  // one-sided over H), at most 1.5 times the mean slope of the amplitude jump
  const jump = hi - lo;
  const dDeg = 0.05,
    dLevel = 0.0005;
  const maxPerDeg = (jump * 1.5) / (2 * W),
    maxPerLevel = (jump * 1.5) / H;
  for (let k = 0; k <= 200; k++) {
    const lf = k / 200;
    let prev = lvRegionalMotion(amp, azOf(0), lf);
    for (let deg = dDeg; deg <= 360 + 1e-9; deg += dDeg) {
      const v = lvRegionalMotion(amp, azOf(deg), lf);
      assert.ok(v >= lo - 1e-12 && v <= hi + 1e-12, `${v} outside [${lo}, ${hi}]`);
      assert.ok(
        Math.abs(v - prev) <= maxPerDeg * dDeg + 1e-9,
        `ring step ${Math.abs(v - prev).toFixed(5)} at AHA ${deg.toFixed(2)}° level ${lf}`,
      );
      prev = v;
    }
  }
  for (let deg = 0; deg < 360; deg += 1.25) {
    let prev = lvRegionalMotion(amp, azOf(deg), 0);
    for (let lf = dLevel; lf <= 1 + 1e-9; lf += dLevel) {
      const v = lvRegionalMotion(amp, azOf(deg), Math.min(1, lf));
      assert.ok(
        Math.abs(v - prev) <= maxPerLevel * dLevel + 1e-9,
        `level step ${Math.abs(v - prev).toFixed(5)} at AHA ${deg}° level ${lf.toFixed(4)}`,
      );
      prev = v;
    }
    // the tissue beyond the cavity reads level 1: the cap's amplitude alone
    assert.equal(lvRegionalMotion(amp, azOf(deg), 1), clamped(17));
  }
  // a heart without regional abnormality: exactly 1, the old normal path
  const normal = new Float32Array(18).fill(1);
  for (let deg = 0; deg < 360; deg += 0.7)
    for (let lf = 0; lf <= 1; lf += 0.013) assert.equal(lvRegionalMotion(normal, azOf(deg), lf), 1);
});

test('regionalMeanFraction: the area-weighted share of the excursion withheld, over the field the classifier draws', () => {
  const sh = lvShapeFor(0.55);
  assert.equal(regionalMeanFraction(new Float32Array(18).fill(1), sh), 0);
  // hyperkinesia is not modelled: amplitudes above 1 withhold nothing
  assert.equal(regionalMeanFraction(new Float32Array(18).fill(1.2), sh), 0);
  // a uniform field withholds 1 − amplitude everywhere
  for (const a of [0.3, 0, -0.4])
    assert.ok(Math.abs(regionalMeanFraction(new Float32Array(18).fill(a), sh) - (1 - Math.fround(a))) < 1e-12);
  // the inferior infarction: the mean of 1 − motion over the cavity's cross-sections (area ∝ g²·ellipseFactor²), by a
  // four times finer integration than the model's
  const { heart } = rwma();
  const amp = heart.segAmp,
    shape = heart.lv.shape;
  let num = 0,
    den = 0;
  for (let i = 0; i < 600; i++) {
    const zeta = (i + 0.5) / 600;
    const g2 = lvProfileG(shape, zeta) ** 2;
    for (let j = 0; j < 1440; j++) {
      const az = -Math.PI + (2 * Math.PI * (j + 0.5)) / 1440;
      const w = g2 * ellipseFactor(shape.ratio, az) ** 2;
      num += (1 - lvRegionalMotion(amp, az, zeta)) * w;
      den += w;
    }
  }
  assert.ok(
    Math.abs(heart.regionalMeanFrac - num / den) < 1e-3,
    `regionalMeanFrac ${heart.regionalMeanFrac.toFixed(4)} vs ${(num / den).toFixed(4)}`,
  );
});

test('inferior infarction: endocardium and epicardium continuous around the ventricle and along its axis', () => {
  const m = rwma();
  for (const [name, pose] of [
    ['end-systole', m.es],
    ['mid-ejection', m.mid],
  ] as const) {
    // short axis: the old field stepped by the whole excursion at AHA 240° (inferoseptal | inferior) and 0° (inferolateral
    // | anterolateral), 7.6 mm (endocardium) and 5.1 mm (epicardium) between neighbours 0.5° apart
    for (const lf of [0.25, 0.5, 0.75]) {
      const z = zAt(pose, lf);
      let prev = wallRun(m, pose, azOf(0), z);
      let worstEndo = 0,
        worstEpi = 0;
      for (let deg = 0.5; deg <= 360; deg += 0.5) {
        const cur = wallRun(m, pose, azOf(deg), z);
        worstEndo = Math.max(worstEndo, Math.abs(cur[0] - prev[0]));
        worstEpi = Math.max(worstEpi, Math.abs(cur[1] - prev[1]));
        prev = cur;
      }
      assert.ok(
        worstEndo < 0.04 && worstEpi < 0.04,
        `${name} level ${lf}: radial step endo ${(worstEndo * 10).toFixed(2)} mm, epi ${(worstEpi * 10).toFixed(2)} mm per 0.5°`,
      );
    }
    // long axis of the infarcted walls and their borders: the old field stepped 5.6 mm at the mid | apical border
    for (const deg of [240, 270, 300, 330, 0]) {
      const az = azOf(deg);
      let prev = wallRun(m, pose, az, zAt(pose, 0.1));
      let worst = 0;
      for (let lf = 0.1025; lf <= 0.85; lf += 0.0025) {
        const cur = wallRun(m, pose, az, zAt(pose, lf));
        worst = Math.max(worst, Math.abs(cur[0] - prev[0]), Math.abs(cur[1] - prev[1]));
        prev = cur;
      }
      assert.ok(worst < 0.05, `${name} AHA ${deg}°: radial step ${(worst * 10).toFixed(2)} mm per 0.25 % of length`);
    }
  }
});

test('inferior infarction: a wall thickens by the share of the excursion it makes; akinetic walls neither thicken nor move out', () => {
  const m = rwma();
  /** mean thickness at ED and ES, endocardial and epicardial displacement over a patch inside a segment */
  const patch = (deg0: number, deg1: number, lf0: number, lf1: number) => {
    let tEd = 0,
      tEs = 0,
      endoIn = 0,
      epiOut = 0,
      n = 0;
    for (let i = 0; i <= 4; i++)
      for (let j = 0; j <= 4; j++) {
        const az = azOf(deg0 + ((deg1 - deg0) * i) / 4);
        const lf = lf0 + ((lf1 - lf0) * j) / 4;
        const a = wallRun(m, m.ed, az, zAt(m.ed, lf));
        const b = wallRun(m, m.es, az, zAt(m.es, lf));
        tEd += a[1] - a[0];
        tEs += b[1] - b[0];
        endoIn += a[0] - b[0];
        epiOut += b[1] - a[1];
        n++;
      }
    return { thickening: tEs / tEd - 1, endoIn: endoIn / n, epiOut: epiOut / n };
  };
  // the remote normal walls (mid inferoseptal and anterolateral)
  const n9 = patch(200, 220, 0.45, 0.55),
    n12 = patch(20, 40, 0.45, 0.55);
  const normal = (n9.thickening + n12.thickening) / 2;
  assert.ok(normal > 0.35, `normal thickening ${(normal * 100).toFixed(1)} %`);
  const segs = [
    { id: 4, p: patch(260, 280, 0.12, 0.22) },
    { id: 5, p: patch(320, 340, 0.12, 0.22) },
    { id: 10, p: patch(260, 280, 0.45, 0.55) },
    { id: 11, p: patch(320, 340, 0.45, 0.55) },
    { id: 15, p: patch(260, 280, 0.78, 0.84) },
  ];
  for (const { id, p } of segs) {
    const amp = m.heart.segAmp[id]!;
    // incompressible wall: the thickening follows the excursion (it kept 0.35 + 0.65·amplitude of it: 16-25 %)
    assert.ok(
      Math.abs(p.thickening - amp * normal) < 0.015,
      `segment ${id} (amplitude ${amp.toFixed(2)}): thickening ${(p.thickening * 100).toFixed(1)} % against ${(amp * normal * 100).toFixed(1)} %`,
    );
    if (amp <= 0.1)
      assert.ok(p.epiOut < 0.01, `akinetic segment ${id}: epicardium ${(p.epiOut * 10).toFixed(2)} mm outward`);
  }
  // the akinetic segments of the finding: no thickening to speak of (it was 17.6 % in the mid inferior wall)
  for (const id of [4, 10]) {
    const p = segs.find((x) => x.id === id)!.p;
    assert.ok(p.thickening < 0.05, `segment ${id}: thickening ${(p.thickening * 100).toFixed(1)} %`);
  }
  // the mid inferior endocardium makes a tenth of the normal excursion
  const seg10 = segs.find((x) => x.id === 10)!.p;
  const normalIn = (n9.endoIn + n12.endoIn) / 2;
  assert.ok(
    Math.abs(seg10.endoIn - 0.1 * normalIn) < 0.03,
    `segment 10 endocardium ${(seg10.endoIn * 10).toFixed(2)} mm against ${(normalIn).toFixed(2)} mm normal`,
  );
  // its tissue keeps its material coordinates (the speckle stays with it): the mid-wall at ED and at ES reads the same
  // material radius; with the global radial scale it read one 13 mm further out at ES
  const az = azOf(270);
  const matRadius = (pose: HeartPose) => {
    const z = zAt(pose, 0.5);
    const [endo, epi] = wallRun(m, pose, az, z);
    const r = (endo + epi) / 2;
    assert.ok(classifyHeart(m.heart, pose, Math.cos(az) * r, Math.sin(az) * r, z, s) && LV_WALL.has(s.structure));
    return Math.hypot(s.mx, s.my);
  };
  const drift = matRadius(m.es) - matRadius(m.ed);
  assert.ok(Math.abs(drift) < 0.1, `mid-wall material radius drifts ${(drift * 10).toFixed(2)} mm`);
});

test('inferior infarction: no gap between the blood and the akinetic basal wall', () => {
  // the regional offset moved the inflow column of the wall but not that of the cavity: at end-systole 1-5 mm of
  // epicardial fat lay between the blood and the basal inferior and inferolateral myocardium
  const m = rwma();
  const pose = m.es;
  const inside = new Set([Structure.LvCavity, Structure.PapillaryMuscle, Structure.Chordae]);
  for (const deg of [260, 270, 285, 300, 315, 330, 345])
    for (const lf of [0.08, 0.12, 0.16, 0.2]) {
      const az = azOf(deg),
        z = zAt(pose, lf);
      let r = 0.5;
      const at = (rr: number) => {
        classifyHeart(m.heart, pose, Math.cos(az) * rr, Math.sin(az) * rr, z, s);
        return s.structure;
      };
      while (r < 5 && inside.has(at(r))) r += 0.002;
      const leftBlood = r;
      while (r < 5 && !LV_WALL.has(at(r))) r += 0.002;
      assert.ok(
        r - leftBlood < 0.01,
        `AHA ${deg}° level ${lf}: ${((r - leftBlood) * 10).toFixed(2)} mm between the blood and the wall`,
      );
    }
});

test('inferior infarction: the drawn cavity keeps the volumes and the ejection fraction of the case', () => {
  const m = rwma();
  const cav = new Set([Structure.LvCavity, Structure.PapillaryMuscle]);
  const volume = (pose: HeartPose) => {
    const h = 0.1;
    let n = 0;
    for (let x = -4.5 + h / 2; x < 4.5; x += h)
      for (let y = -4.5 + h / 2; y < 4.5; y += h)
        for (let z = -1 + h / 2; z < m.heart.lv.lengthCm + 0.6; z += h)
          if (classifyHeart(m.heart, pose, x, y, z, s) && cav.has(s.structure)) n++;
    return n * h * h * h;
  };
  const edv = volume(m.ed),
    esv = volume(m.es);
  const { edvMl, esvMl } = m.caseDef.physiology;
  const ef = (edv - esv) / edv,
    efTruth = (edvMl - esvMl) / edvMl;
  // the case grades Simpson volumes within 15 % (ED) and 18 % (ES); the drawing takes a fraction of that, as the normal
  // heart's does (118 / 46 mL drawn for 120 / 45)
  assert.ok(Math.abs(edv / edvMl - 1) < 0.03, `EDV ${edv.toFixed(1)} mL drawn, ${edvMl} declared`);
  assert.ok(Math.abs(esv / esvMl - 1) < 0.03, `ESV ${esv.toFixed(1)} mL drawn, ${esvMl} declared`);
  assert.ok(
    Math.abs(ef - efTruth) < 0.015,
    `EF ${(ef * 100).toFixed(1)} % drawn, ${(efTruth * 100).toFixed(1)} % declared`,
  );
});

test('the RV crescent, the epicardial radius of the tools and the tissue velocity follow the same field', () => {
  const m = rwma();
  const A = heartAnchors(m.heart);
  const hp = m.es;
  const res = new Float64Array(4);
  let worstRv = 0,
    worstEpi = 0;
  for (let deg = 150; deg <= 360; deg += 2.5)
    for (const lf of [0.4, 0.5, 0.6]) {
      const az = azOf(deg),
        z = zAt(hp, lf);
      const [, epi] = wallRun(m, hp, az, z);
      rvRadii(m.heart, A, hp.prof, hp.thickK, hp.zAnn, hp.lengthNow, hp.tvZ, hp.state.contraction, hp.septalShiftCm, hp.rvCollapse, az, z, res);
      // the crescent rests on the epicardium the classifier draws (0.05 cm gap), its border band included
      if (res[1]! > 0.02 && res[1]! < 0.98) worstRv = Math.max(worstRv, Math.abs(res[0]! - 0.05 - epi));
      // the measurement tools' epicardium (it missed the regional offset: 7.4 mm short in the akinetic walls)
      worstEpi = Math.max(worstEpi, Math.abs(lvEpicardialRadiusAt(m.heart, hp, az, z) - epi));
    }
  assert.ok(worstRv < 0.03, `RV inner boundary ${(worstRv * 10).toFixed(2)} mm off the LV epicardium`);
  assert.ok(worstEpi < 0.03, `lvEpicardialRadiusAt ${(worstEpi * 10).toFixed(2)} mm off the drawn epicardium`);
  // TDI and Color clutter: the radial velocity scales with the same field, continuous across the borders (it stepped
  // 2 cm/s at AHA 240°) and a tenth of the normal one in the mid inferior wall
  const z = zAt(m.mid, 0.5),
    r = 2.2;
  const radial = (deg: number) => {
    const az = azOf(deg);
    const v = sampleTissueVelocity(m.heart, m.tables, m.midPhase, Math.cos(az) * r, Math.sin(az) * r, z);
    return v.vx * Math.cos(az) + v.vy * Math.sin(az);
  };
  let prev = radial(0),
    worst = 0;
  for (let deg = 0.5; deg <= 360; deg += 0.5) {
    const v = radial(deg);
    worst = Math.max(worst, Math.abs(v - prev));
    prev = v;
  }
  assert.ok(worst < 0.002, `tissue radial velocity steps ${(worst * 100).toFixed(2)} cm/s per 0.5°`);
  const vNormal = radial(30);
  assert.ok(vNormal < -0.01, `normal wall moves inward in ejection: ${vNormal.toFixed(4)} m/s`);
  assert.ok(Math.abs(radial(270) / vNormal - 0.1) < 0.01, `mid inferior / normal ${(radial(270) / vNormal).toFixed(3)}`);
});

test('only the cases that declare regional motion evaluate the field', () => {
  for (const input of CASE_INPUTS) {
    const { heart } = buildCaseModels(loadCaseById(input.id), REST_PATIENT);
    const declared = input.anatomy.wallMotion?.some((w) => Math.min(1, w.amplitude) < 1) ?? false;
    if (declared) assert.ok(heart.regionalMeanFrac > 0.1, `${input.id}: ${heart.regionalMeanFrac}`);
    // 0 exactly: the classifier, the RV crescent and the shaders then skip the field and draw what they drew
    else assert.equal(heart.regionalMeanFrac, 0, input.id);
  }
});

test('GLSL mirror: the shaders evaluate the same field and use it for the cavity, the wall, the thickening and the RV', () => {
  // the scalar GLSL is JavaScript once its types and built-ins are spelled out
  const fns = ['wallMotionSegment', 'wallMotionRing', 'lvRegionalMotion', 'lvMotionAt'];
  let js = '';
  for (const name of fns) {
    const f = new RegExp(`float ${name}\\(([^)]*)\\) \\{([\\s\\S]*?)\\n\\}`).exec(GLSL_HEART_CHAMBERS);
    assert.ok(f, `${name} missing from GLSL_HEART_CHAMBERS`);
    const params = f[1]!
      .split(',')
      .map((p) => p.trim().split(/\s+/)[1])
      .join(', ');
    js += `function ${name}(${params}) {${f[2]}\n}\n`;
  }
  js = js
    .replace(/\bfloat\s+(?=\w+\s*=)/g, 'let ')
    .replace(/(?<![.\w])(abs|floor|min|max)\(/g, 'Math.$1(');
  const glsl = new Function(
    'P',
    'int',
    'smoothstep',
    'SEG_AMP_BASE',
    'PI',
    'RV_AZA',
    'RV_AZP',
    `${js}\nreturn { lvRegionalMotion, lvMotionAt };`,
  );
  const smoothstep = (e0: number, e1: number, x: number) => {
    const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
    return t * t * (3 - 2 * t);
  };
  const base = 5;
  for (const amp of [mixedAmplitudes(), rwma().heart.segAmp, new Float32Array(18).fill(1)]) {
    const flag = regionalMeanFraction(amp, lvShapeFor(0.5));
    const params = new Float64Array(base + 18);
    params.set(amp, base);
    params[base] = flag;
    const g = glsl(
      (i: number) => params[i]!,
      Math.trunc,
      smoothstep,
      base,
      Math.PI,
      RV_GROOVE_ANTERIOR_RAD,
      RV_GROOVE_INFERIOR_RAD,
    ) as { lvRegionalMotion: (a: number, l: number) => number; lvMotionAt: (a: number, l: number) => number };
    for (let i = -400; i <= 400; i++) {
      const az = (i / 400) * Math.PI * 1.02;
      for (let k = 0; k <= 120; k++) {
        const lf = k / 120;
        const cpu = lvRegionalMotion(amp, az, lf);
        assert.ok(Math.abs(g.lvRegionalMotion(az, lf) - cpu) < 1e-12, `az ${az} level ${lf}`);
        assert.ok(Math.abs(g.lvMotionAt(az, lf) - (flag > 0 ? cpu : 1)) < 1e-12, `flag, az ${az} level ${lf}`);
      }
    }
  }
  const body = (src: string, sig: RegExp): string => {
    const f = sig.exec(src);
    assert.ok(f, `${sig} missing`);
    return f[1]!;
  };
  const offset = body(GLSL_HEART_CHAMBERS, /float regionalCavityOffset\(float az, float levelFrac, float motion\) \{([\s\S]*?)\n\}/);
  assert.ok(
    offset.includes('(1.0 - motion) * (LV_RMAX_ED - LV_RMAX) * lvProfileG(levelFrac) * ellipseFactor(LV_RATIO, az)'),
    'GLSL regionalCavityOffset differs',
  );
  const wall = body(GLSL_HEART_CHAMBERS, /float wallThicknessAt\(float az, float levelFrac, float motion\) \{([\s\S]*?)\n\}/);
  assert.ok(wall.includes('max(0.6, 1.0 + (LV_THICK_K - 1.0) * motion)'), 'GLSL thickening is not the excursion share');
  const rv = body(GLSL_HEART_CHAMBERS, /vec4 rvRadii\([^)]*\) \{([\s\S]*?)\n\}/);
  assert.ok(rv.includes('lvMotionAt(az, levelFrac)') && rv.includes('regionalCavityOffset(az, levelFrac, motion)'));
  for (const line of [
    'float motion = lvMotionAt(az, levelFrac);',
    'float regional = regionalCavityOffset(az, levelFrac, motion);',
    'float rsWall = rs + (1.0 - motion) * (1.0 - rs);',
    'float dEllR = smin(dProf - regional, dInflow, 0.3);',
    'float tFull = wallThicknessAt(az, levelFrac, motion);',
    'vec3(x / rsWall, y / rsWall, (z - LV_LEN) / ls)',
  ])
    assert.ok(GLSL_HEART.includes(line), `GLSL LV block missing: ${line}`);
  assert.ok(!/SEG_AMP_BASE \+ (seg|ahaSegment)/.test(GLSL_HEART), 'a shader still reads the amplitude per segment');
  // the packed flag: SEG_AMP[0] is the heart's regionalMeanFrac, the amplitudes follow at their AHA ids
  for (const caseId of ['inferior-rwma', 'normal-excellent']) {
    const c = loadCaseById(caseId);
    const { heart, thorax, tables } = buildCaseModels(c, REST_PATIENT);
    const settings = baseInput().settings;
    const view = VIEW_TARGETS.find((v) => v.id === 'a4c')!;
    const beam = beamFrameFromPose(poseFromControl(thorax, canonicalControl(view, heart, thorax)), 1);
    const packed = allocPacked();
    packScene(
      {
        heart,
        heartPose: computeHeartPose(heart, cycleStateAt(tables, 0.3)),
        thorax,
        physics: scenePhysicsFor(c, settings),
      },
      beam,
      polarSpecFor(settings, 'low'),
      packed,
    );
    const o = PARAM_OFFSET['SEG_AMP']!;
    assert.equal(packed.data[o], Math.fround(heart.regionalMeanFrac), caseId);
    for (let id = 1; id <= 17; id++) assert.equal(packed.data[o + id], heart.segAmp[id], `${caseId} SEG_AMP[${id}]`);
  }
});
