import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { loadCaseById } from '@/cases';
import { buildCaseModels } from './caseModels';
import { computeHeartPose } from './heartModel';
import { heartToTorso } from './heartFrame';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import { buildHeartMeshes } from './heartMesh';

describe('3D navigator heart meshes, AO, physiological groups & coronaries', () => {
  const caseDef = loadCaseById('normal');
  const patient = {
    position: 'left-lateral' as const,
    respiration: 'expiration' as const,
    headElevationDeg: 0,
  };
  const { heart, thorax, tables } = buildCaseModels(caseDef, patient);
  const pose = computeHeartPose(heart, cycleStateAt(tables, 0));

  it('extracts separated physiological chambers, vessels, AO, groove masks, and epicardial coronaries', () => {
    const groups = buildHeartMeshes(heart, pose, { stepCm: 0.35, thorax });
    const byId = new Map(groups.map((g) => [g.id, g]));

    for (const id of [
      'lv-myocardium',
      'left-atrium',
      'right-atrium',
      'aortic-root',
      'pulmonary-artery',
      'venae-cavae',
      'thoracic-aorta',
      'coronary-arteries',
      'coronary-veins',
    ]) {
      const g = byId.get(id);
      assert.ok(g, `expected group ${id}`);
      assert.ok(g.indices.length > 0, `expected non-empty indices for ${id}`);
      assert.ok(g.ao && g.ao.length * 3 === g.positions.length, `expected per-vertex AO for ${id}`);
      for (let i = 0; i < g.ao.length; i++) {
        assert.ok(g.ao[i]! >= 0.5 && g.ao[i]! <= 1.1, `AO out of range for ${id}: ${g.ao[i]}`);
      }
    }

    // Unified biventricular myocardium: no detached separate rv-myocardium overhang shell
    assert.equal(byId.has('rv-myocardium'), false);

    // Distinct physiological color coding: arterial aorta vs venous cavae & PA, LA vs RA
    assert.notEqual(byId.get('aortic-root')!.color, byId.get('pulmonary-artery')!.color);
    assert.notEqual(byId.get('aortic-root')!.color, byId.get('venae-cavae')!.color);
    assert.notEqual(byId.get('left-atrium')!.color, byId.get('right-atrium')!.color);

    // Unified ventricles still tag LV walls with AHA 17 segments and leave RV vertices as 0
    const lv = byId.get('lv-myocardium')!;
    assert.ok(lv.segments && lv.segments.length * 3 === lv.positions.length);
    assert.ok(lv.segments.some((c) => c > 0), 'expected LV AHA segments > 0');
    assert.ok(lv.segments.some((c) => c === 0), 'expected RV vertices tagged as segment 0');

    // Inferior trim keeps IVC and descending aorta cleanly tucked to the cardiac silhouette
    for (const vesselId of ['venae-cavae', 'thoracic-aorta']) {
      const g = byId.get(vesselId)!;
      let minTorsoY = Infinity;
      for (let i = 0; i < g.positions.length; i += 3) {
        const p = heartToTorso(heart.frame, {
          x: g.positions[i]!,
          y: g.positions[i + 1]!,
          z: g.positions[i + 2]!,
        });
        if (p.y < minTorsoY) minTorsoY = p.y;
      }
      assert.ok(
        minTorsoY >= heart.frame.origin.y - 6.2,
        `${vesselId} should not dangle below cardiac silhouette (minTorsoY=${minTorsoY.toFixed(2)})`,
      );
    }
  });

  it('supports omitting thoracic-aorta on secondary phases for fast 16-phase extraction', () => {
    const secondary = buildHeartMeshes(heart, pose, {
      stepCm: 0.35,
      thorax,
      includeAorta: false,
    });
    assert.equal(
      secondary.some((g) => g.id === 'thoracic-aorta'),
      false,
    );
    assert.equal(
      secondary.some((g) => g.id === 'coronary-arteries'),
      true,
    );
  });
});
