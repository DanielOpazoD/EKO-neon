import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { loadCaseById } from '@/cases';
import { buildCaseModels, REST_PATIENT } from './caseModels';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import { computeHeartPose } from './heartModel';
import { sceneClassifier } from './sceneClassifier';
import { Structure, Tissue, makeSample } from './tissue';
import {
  mediastinumDistance,
  posteriorMediastinumDistance,
  superiorMediastinumDistance,
} from './thoraxModel';
import { VIEW_TARGETS, canonicalBeam } from '@/simulator/windows/viewTargets';

const REGIONS = [
  Structure.PosteriorMediastinum,
  Structure.SuperiorMediastinum,
  Structure.PericardialFat,
] as const;

/** Structure counts over a view's canonical sector (90°, 18 cm), as the tracer classifies it. */
function sectorCounts(caseId: string, viewId: string): Map<Structure, number> {
  const def = loadCaseById(caseId);
  const { heart, thorax, tables } = buildCaseModels(def, REST_PATIENT);
  const pose = computeHeartPose(heart, cycleStateAt(tables, 0.3));
  const classify = sceneClassifier({ heart, heartPose: pose, thorax });
  const view = VIEW_TARGETS.find((v) => v.id === viewId)!;
  const b = canonicalBeam(view, heart, thorax);
  const q = makeSample();
  const counts = new Map<Structure, number>();
  for (let i = 0; i < 80; i++)
    for (let j = 1; j <= 100; j++) {
      const th = ((-45 + (90 * i) / 79) * Math.PI) / 180;
      const r = (18 * j) / 100;
      const f = Math.cos(th),
        l = Math.sin(th);
      const px = b.origin.x + r * (b.forward.x * f + b.lateral.x * l);
      const py = b.origin.y + r * (b.forward.y * f + b.lateral.y * l);
      const pz = b.origin.z + r * (b.forward.z * f + b.lateral.z * l);
      if (!classify(px, py, pz, q)) continue;
      if ((REGIONS as readonly Structure[]).includes(q.structure)) {
        // named by region, the tissue is the same soft tissue it was: the image does not change
        assert.equal(q.tissue, Tissue.Fat);
        const zc = pz - thorax.columnShiftCm;
        const inside = mediastinumDistance(px, py, zc) <= 0;
        assert.equal(q.structure === Structure.PericardialFat, !inside, `region of (${px}, ${py}, ${pz})`);
      }
      counts.set(q.structure, (counts.get(q.structure) ?? 0) + 1);
    }
  return counts;
}

describe('the thorax tissue no organ fills is named by region', () => {
  it('the mediastinum is its posterior and superior parts, exactly', () => {
    for (let k = 0; k < 4000; k++) {
      const x = -6 + 12 * ((k * 0.618034) % 1);
      const y = -4 + 14 * ((k * 0.414214) % 1);
      const z = -20 + 18 * ((k * 0.732051) % 1);
      assert.equal(
        mediastinumDistance(x, y, z),
        Math.min(posteriorMediastinumDistance(x, y, z), superiorMediastinumDistance(x, y, z)),
      );
    }
  });

  it('the short axis at the valve shows pericardial fat in front of the heart and both parts of the mediastinum', () => {
    const c = sectorCounts('normal-excellent-window', 'psax-av');
    for (const s of REGIONS) assert.ok((c.get(s) ?? 0) > 20, `structure ${s}: ${c.get(s) ?? 0} samples`);
  });

  it('the subcostal inferior vena cava view keeps the posterior mediastinum around the descending aorta', () => {
    const c = sectorCounts('normal-excellent-window', 'subcostal-ivc');
    assert.ok((c.get(Structure.PosteriorMediastinum) ?? 0) > 200);
  });
});
