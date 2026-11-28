import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { listCases, loadCaseById } from '@/cases';
import { buildCaseModels } from '@/simulator/anatomy/caseModels';
import { computeHeartPose } from '@/simulator/anatomy/heartModel';
import { heartToTorso } from '@/simulator/anatomy/heartFrame';
import { buildHeartMeshes } from '@/simulator/anatomy/heartMesh';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import {
  computeAnteriorRibClearanceCm,
  visualRibInnerZ,
  visualSkinZ,
} from './thoraxClearance';

describe('3D navigator rib-cage vs heart zero-collision clearance', () => {
  it('keeps the beating 3D heart strictly behind the anterior ribs across all cases in left-lateral decubitus', () => {
    const patient = {
      position: 'left-lateral' as const,
      respiration: 'expiration' as const,
      headElevationDeg: 0,
    };

    for (const meta of listCases()) {
      const caseDef = loadCaseById(meta.id);
      const { heart, thorax, tables } = buildCaseModels(caseDef, patient);
      const edPose = computeHeartPose(heart, cycleStateAt(tables, 0));
      const sysPose = computeHeartPose(heart, cycleStateAt(tables, 0.28));
      const clearanceCm = computeAnteriorRibClearanceCm(heart, thorax, [edPose, sysPose]);

      assert.ok(
        clearanceCm >= 0 && clearanceCm <= 2.4,
        `${meta.id}: unexpected clearance ${clearanceCm}`,
      );

      for (const pose of [edPose, sysPose]) {
        const groups = buildHeartMeshes(heart, pose, {
          stepCm: 0.4,
          thorax,
          includeAorta: false,
          includeCoronaries: true,
        });
        for (const g of groups) {
          if (g.id === 'lv-cavity' || g.id === 'rv-cavity' || g.id === 'valves') continue;
          for (let i = 0; i < g.positions.length; i += 3) {
            const p = heartToTorso(heart.frame, {
              x: g.positions[i]!,
              y: g.positions[i + 1]!,
              z: g.positions[i + 2]!,
            });
            const innerZ = visualRibInnerZ(thorax, p.x, p.y, clearanceCm);
            const outerSkinZ = visualSkinZ(thorax, p.x, p.y, clearanceCm);
            assert.ok(
              p.z < innerZ - 0.15,
              `${meta.id} (${g.id}) collides with anterior rib at (${p.x.toFixed(1)}, ${p.y.toFixed(1)}): heartZ=${p.z.toFixed(2)} vs ribInnerZ=${innerZ.toFixed(2)}`,
            );
            assert.ok(
              innerZ < outerSkinZ,
              `${meta.id}: rib inner face must remain beneath outer skin`,
            );
          }
        }
      }
    }
  });
});
