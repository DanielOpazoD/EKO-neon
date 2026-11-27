import assert from 'node:assert/strict';
import test from 'node:test';
import { CASE_INPUTS, loadCaseById } from '@/cases';
import { AORTIC_STENOSIS_RULES as AS } from '@/clinical/reference-values';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import { anchorsCached } from './anchors';
import {
  AV_CONTRACTION_COEFFICIENT,
  aorticOpeningForEoa,
  aorticOrificeAreaCm2,
  aorticRootProfile,
  buildAorticValve,
  rootRadiusAt,
} from './aorticValve';
import { buildCaseModels, REST_PATIENT } from './caseModels';
import { classifyHeart } from './classify';
import { heartDirToTorso, ROOT_EXCURSION } from './heartFrame';
import type { HeartModel } from './heartModel';
import { computeHeartPose, type HeartPose } from './heartPose';
import { makeSample, Tissue } from './tissue';
import { canonicalBeam, getViewTarget } from '@/simulator/windows/viewTargets';
import { cross, dot, normalize, v3 } from '@/core/vec3';

/**
 * One aortic valve for the 2D and the Doppler: at the peak of ejection the cusps open as far as the case's effective
 * orifice area (EOA, the area of the colour jet, the CW and the continuity equation) needs, through EOA = Cc·GOA. While
 * the opening was typed per case (maxOpeningFraction) the drawn orifice ignored the EOA: on this planimetry the severe
 * stenosis (0.3) drew 0.16 cm² for an EOA of 0.78 cm², the moderate one (0.5) 0.60 cm² for 1.05 (severe by planimetry,
 * moderate by continuity) and the artifact case (0.6) 1.10 cm² for 1.15. They now trace 0.96, 1.31 and 1.43 cm².
 */

/** Grade of an aortic valve area on the ASE/EACVI 2017 thresholds the report grades with. */
const grade = (areaCm2: number): string =>
  areaCm2 <= AS.severeAva.value
    ? 'severe'
    : areaCm2 <= AS.moderateAva.value.hi
      ? 'moderate'
      : 'mild or none';

interface Peak {
  heart: HeartModel;
  pose: HeartPose;
  eoa: number;
}
const peaks = new Map<string, Peak>();
/** The case at the sample of its beat where the aortic flow, and with it the cycle's opening, peaks. */
function peakOf(id: string): Peak {
  let p = peaks.get(id);
  if (!p) {
    const c = loadCaseById(id);
    const { heart, tables } = buildCaseModels(c, REST_PATIENT);
    const flow = tables.aorticFlowMlps;
    let k = 0;
    for (let i = 1; i < flow.length; i++) if (flow[i]! > flow[k]!) k = i;
    const state = cycleStateAt(tables, (k + 0.5) / flow.length);
    assert.ok(state.avOpen > 0.9999, `${id}: the cycle's opening peaks with the flow (${state.avOpen})`);
    p = { heart, pose: computeHeartPose(heart, state), eoa: c.hemodynamics.avEffectiveAreaCm2 };
    peaks.set(id, p);
  }
  return p;
}

/**
 * Planimetry of the drawn valve in the short axis of the root, through `classifyHeart`: rays from the root's axis to the
 * first valve sample (cusp or coaptation surface; the root wall when none), at `nPhi` azimuths and every 0.25 mm of
 * height. `slice` is the smallest lumen of a single cross-section, the one a sonographer traces at the tips; `projected`
 * the narrowest lumen of each azimuth over all heights, the orifice projected on the cross-section.
 */
function planimetry(heart: HeartModel, pose: HeartPose, nPhi = 96): { slice: number; projected: number } {
  const A = anchorsCached(heart);
  const ax = A.avAxis,
    e1 = A.avE1,
    e2 = A.avE2;
  // root coordinates as `rootCoordinates` reads them; the classifier undoes the tamponade's swing
  const cx = A.avCenter.x + pose.swingX,
    cy = A.avCenter.y,
    cz = A.avCenter.z + pose.zAnn * ROOT_EXCURSION;
  const s = makeSample();
  const valveAt = (t: number, r: number, phi: number): boolean => {
    const u = r * Math.cos(phi),
      v = r * Math.sin(phi);
    return (
      classifyHeart(
        heart,
        pose,
        cx + ax.x * t + e1.x * u + e2.x * v,
        cy + ax.y * t + e1.y * u + e2.y * v,
        cz + ax.z * t + e1.z * u + e2.z * v,
        s,
      ) && s.tissue === Tissue.Valve
    );
  };
  const V = pose.valves;
  const dPhi = (2 * Math.PI) / nPhi;
  const narrowest = new Float64Array(nPhi).fill(Infinity);
  let slice = Infinity;
  for (let t = 0.2; t <= V.aortic.hComm + 0.1; t += 0.025) {
    let area = 0;
    for (let i = 0; i < nPhi; i++) {
      const phi = (i + 0.5) * dPhi;
      const wall = rootRadiusAt(V.root, t, phi);
      let r = 0;
      while (r < wall && !valveAt(t, r, phi)) r += 0.01;
      if (r >= wall) r = wall;
      else {
        let lo = Math.max(0, r - 0.01),
          hi = r;
        for (let k = 0; k < 6; k++) {
          const mid = 0.5 * (lo + hi);
          if (valveAt(t, mid, phi)) hi = mid;
          else lo = mid;
        }
        r = 0.5 * (lo + hi);
      }
      area += 0.5 * r * r * dPhi;
      narrowest[i] = Math.min(narrowest[i]!, r);
    }
    slice = Math.min(slice, area);
  }
  let projected = 0;
  for (let i = 0; i < nPhi; i++) projected += 0.5 * narrowest[i]! * narrowest[i]! * dPhi;
  return { slice, projected };
}

test('aorticOpeningForEoa inverts the orifice: exact, monotone, and fully open beyond what the cusps can leave', () => {
  const root = aorticRootProfile(loadCaseById('aortic-stenosis-severe').anatomy);
  const thickness = 0.25;
  const fullOpen = aorticOrificeAreaCm2(buildAorticValve(root.count, 1, thickness), root);
  // the orifice widens with the opening, from nothing (closed cusps meet at the axis)
  let prev = -1;
  for (let o = 0; o <= 1.0001; o += 0.05) {
    const goa = aorticOrificeAreaCm2(buildAorticValve(root.count, o, thickness), root);
    assert.ok(goa >= prev, `orifice ${goa} at opening ${o.toFixed(2)} below ${prev}`);
    prev = goa;
  }
  assert.equal(aorticOrificeAreaCm2(buildAorticValve(root.count, 0, thickness), root), 0);
  let prevOpen = 0;
  for (let eoa = 0.3; eoa <= 3; eoa += 0.15) {
    const o = aorticOpeningForEoa(root, thickness, eoa);
    assert.ok(o >= prevOpen && o <= 1, `opening ${o} for EOA ${eoa.toFixed(2)}`);
    prevOpen = o;
    const goa = aorticOrificeAreaCm2(buildAorticValve(root.count, o, thickness), root);
    if (o < 1)
      assert.ok(
        Math.abs(AV_CONTRACTION_COEFFICIENT * goa - eoa) < 1e-4,
        `EOA ${eoa.toFixed(2)}: Cc·GOA = ${(AV_CONTRACTION_COEFFICIENT * goa).toFixed(5)}`,
      );
    // cusps whose full opening leaves less than EOA / Cc open fully
    else assert.ok(AV_CONTRACTION_COEFFICIENT * fullOpen <= eoa + 1e-9, `fully open for EOA ${eoa.toFixed(2)}`);
  }
});

test('the cusps leave at least the effective orifice of the Doppler at the peak of ejection, contracted by Cc (12 cases)', () => {
  const rows: string[] = [];
  const failures: string[] = [];
  for (const input of CASE_INPUTS) {
    const c = loadCaseById(input.id);
    const { heart, pose, eoa } = peakOf(input.id);
    const V = pose.valves;
    // the opening was solved on the root that is drawn
    assert.deepEqual(V.root, aorticRootProfile(c.anatomy), `${c.id}: root profile`);
    const goa = aorticOrificeAreaCm2(V.aortic, V.root);
    const cc = eoa / goa;
    const row =
      `${c.id}: opening ${V.aortic.open.toFixed(3)}, GOA ${goa.toFixed(3)} cm², ` +
      `EOA ${eoa.toFixed(2)} cm², Cc ${cc.toFixed(3)}`;
    rows.push(row);
    // a jet is never wider than the hole it leaves by (EOA = Cc·GOA with Cc ≤ 1); restricted cusps contract it by Cc, and
    // fully open ones, lying along the flow, by less
    if (!(cc > 0 && cc <= 1)) failures.push(`${row}: EOA > GOA`);
    else if (heart.aorticOpening < 1 && Math.abs(cc - AV_CONTRACTION_COEFFICIENT) > 1e-3)
      failures.push(`${row}: Cc ≠ ${AV_CONTRACTION_COEFFICIENT}`);
    else if (heart.aorticOpening === 1 && cc < AV_CONTRACTION_COEFFICIENT)
      failures.push(`${row}: fully open with a stronger contraction than Cc`);
  }
  assert.deepEqual(failures, [], rows.join('\n'));
});

test('short-axis planimetry of the drawn stenotic valves grades like the continuity equation', () => {
  const stenoses = CASE_INPUTS.map((c) => c.id).filter(
    (id) => loadCaseById(id).hemodynamics.avEffectiveAreaCm2 <= AS.moderateAva.value.hi,
  );
  assert.deepEqual(stenoses, ['aortic-stenosis-moderate', 'aortic-stenosis-severe', 'artifact-challenge']);
  for (const id of stenoses) {
    const { heart, pose, eoa } = peakOf(id);
    const { slice, projected } = planimetry(heart, pose);
    const where = `${id}: planimetry ${slice.toFixed(3)} cm² (projected ${projected.toFixed(3)}), EOA ${eoa.toFixed(2)} cm²`;
    // the trace reads the hole, which the jet contracts: above the EOA, within the contraction of a domed or a flat
    // valve (Cc 0.71–0.90 in Gilon et al. 2002), and in the same grade
    assert.ok(slice >= eoa, `${where}: the image's orifice is smaller than its jet`);
    assert.ok(eoa / slice >= 0.71 && eoa / slice <= 0.9, `${where}: Cc ${(eoa / slice).toFixed(3)}`);
    assert.equal(grade(slice), grade(eoa), where);
    // the orifice drawn is the one the opening was solved for (the single slice reads a few % more: the free edges rise
    // toward the commissures, so no plane holds them all)
    const goa = aorticOrificeAreaCm2(pose.valves.aortic, pose.valves.root);
    assert.ok(Math.abs(projected / goa - 1) < 0.02, `${where}: solved for ${goa.toFixed(3)} cm²`);
    assert.ok(slice >= projected && slice <= projected * 1.08, where);
  }
});

test('canonical PSAX-AV stays perpendicular to the aortic root so stenotic planimetry grades like EOA', () => {
  const view = getViewTarget('psax-av')!;
  const s = makeSample();
  for (const id of ['aortic-stenosis-severe', 'aortic-stenosis-moderate', 'artifact-challenge'] as const) {
    const c = loadCaseById(id);
    const { heart, thorax } = buildCaseModels(c, REST_PATIENT);
    const { pose, eoa } = peakOf(id);
    const A = anchorsCached(heart);
    const bf = canonicalBeam(view, heart, thorax);
    const avAxisT = heartDirToTorso(heart.frame, A.avAxis);
    const obliqDeg = (Math.acos(Math.min(1, Math.abs(dot(bf.normal, avAxisT)))) * 180) / Math.PI;
    // Before removing the 12° inflow tilt, canonical PSAX-AV was 12° oblique and inflated severe AS to 1.17 cm².
    assert.ok(obliqDeg < 3, `${id}: canonical PSAX-AV obliquity ${obliqDeg.toFixed(1)}° (expected < 3°)`);
    const nH = normalize(
      v3(
        dot(bf.normal, heart.frame.ex),
        dot(bf.normal, heart.frame.ey),
        dot(bf.normal, heart.frame.ez),
      ),
    );
    const uH = normalize(cross(Math.abs(nH.y) < 0.9 ? v3(0, 1, 0) : v3(1, 0, 0), nH));
    const vH = cross(nH, uH);
    const cx = A.avCenter.x + pose.swingX,
      cy = A.avCenter.y,
      cz = A.avCenter.z + pose.zAnn * ROOT_EXCURSION;
    const nPhi = 96,
      dPhi = (2 * Math.PI) / nPhi;
    let minArea = Infinity;
    for (let t = 0.2; t <= pose.valves.aortic.hComm + 0.1; t += 0.025) {
      let area = 0;
      for (let i = 0; i < nPhi; i++) {
        const phi = (i + 0.5) * dPhi;
        const cphi = Math.cos(phi),
          sphi = Math.sin(phi);
        let r = 0;
        for (; r < 2.0; r += 0.01) {
          const px = cx + A.avAxis.x * t + (uH.x * cphi + vH.x * sphi) * r;
          const py = cy + A.avAxis.y * t + (uH.y * cphi + vH.y * sphi) * r;
          const pz = cz + A.avAxis.z * t + (uH.z * cphi + vH.z * sphi) * r;
          if (classifyHeart(heart, pose, px, py, pz, s) && s.tissue === Tissue.Valve) break;
        }
        area += 0.5 * r * r * dPhi;
      }
      if (area < minArea) minArea = area;
    }
    assert.equal(
      grade(minArea),
      grade(eoa),
      `${id}: canonical PSAX-AV planimetry ${minArea.toFixed(3)} cm² graded ${grade(minArea)}, expected ${grade(eoa)} (EOA ${eoa.toFixed(2)})`,
    );
  }
});
