import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { Structure, Tissue } from '@/simulator/anatomy/tissue';
import { baseInput } from '@/simulator/core/baseInput';
import { SimulatorCore } from '@/simulator/core/simulatorCore';
import { scenePhysicsFor } from '@/simulator/renderer/scenePhysics';
import { polarSpecFor } from '@/simulator/renderer/types';
import { enumDefinesGlsl, GLSL_COMMON } from './glslCommon';
import { GLSL_HEART } from './glslHeart';
import {
  GLSL_PASS_A_MAIN,
  GLSL_PASS_A_SIDE_MAIN,
  GLSL_PASS_B_MAIN,
  GLSL_PASS_C_MAIN,
  GLSL_PASS_D_MAIN,
  GLSL_PASS_L_MAIN,
  GLSL_PASS_P_MAIN,
} from './glslPasses';
import { GLSL_THORAX } from './glslThorax';
import {
  allocPacked,
  packScene,
  PARAM_COUNT,
  PARAM_OFFSET,
  PARAM_TEXELS,
  paramDefinesGlsl,
} from './paramLayout';

describe('CPU <-> GPU paramLayout & GLSL contract', () => {
  it('packs all scalar and array parameter offsets tightly within PARAM_TEXELS without duplicates', () => {
    assert.ok(PARAM_COUNT > 0);
    assert.ok(PARAM_TEXELS * 4 >= PARAM_COUNT);
    assert.ok(PARAM_TEXELS * 4 - PARAM_COUNT < 4);

    const defines = paramDefinesGlsl();
    const definedNames = new Set<string>();
    for (const line of defines.split('\n')) {
      const m = /^#define\s+([A-Z0-9_]+)\s+(?:P\((\d+)\)|(\d+))$/.exec(line.trim());
      assert.ok(m, `Unexpected define syntax in paramDefinesGlsl(): ${line}`);
      const name = m[1]!;
      const offset = Number(m[2] ?? m[3]);
      assert.ok(!definedNames.has(name), `Duplicate GLSL macro ${name}`);
      definedNames.add(name);
      assert.ok(offset >= 0 && offset < PARAM_COUNT, `Out-of-range offset ${offset} for ${name}`);
    }

    // Every *_BASE symbol referenced in the GPU shaders must exist in paramDefinesGlsl()
    const shaderSources = [
      GLSL_HEART,
      GLSL_THORAX,
      GLSL_PASS_A_MAIN,
      GLSL_PASS_A_SIDE_MAIN,
      GLSL_PASS_B_MAIN,
      GLSL_PASS_L_MAIN,
      GLSL_PASS_P_MAIN,
      GLSL_PASS_C_MAIN,
      GLSL_PASS_D_MAIN,
    ].join('\n');
    const allDefined = new Set<string>(definedNames);
    const preambleAndShaders = `${GLSL_COMMON}\n${shaderSources}`;
    for (const m of preambleAndShaders.matchAll(/^\s*#define\s+([A-Z][A-Z0-9_]*)\b/gm)) {
      allDefined.add(m[1]!);
    }
    for (const m of preambleAndShaders.matchAll(/\bconst\s+\w+\s+([A-Z][A-Z0-9_]*)\b/g)) {
      allDefined.add(m[1]!);
    }
    const referencedBases = new Set(shaderSources.match(/\b[A-Z][A-Z0-9_]*_BASE\b/g) ?? []);
    assert.ok(referencedBases.size > 5, 'Expected array *_BASE references in GLSL sources');
    for (const base of referencedBases) {
      assert.ok(allDefined.has(base), `GLSL shader references undefined symbol ${base}`);
    }
  });

  it('keeps GLSL Tissue and Structure enum defines synced with TypeScript const enums', () => {
    const glslEnums = enumDefinesGlsl();
    const map = new Map<string, number>();
    for (const line of glslEnums.split('\n')) {
      const m = /^#define\s+([A-Z0-9_]+)\s+(\d+)$/.exec(line.trim());
      if (m) map.set(m[1]!, Number(m[2]));
    }
    assert.equal(map.get('T_BLOOD'), Tissue.Blood);
    assert.equal(map.get('T_MYO'), Tissue.Myocardium);
    assert.equal(map.get('T_VALVE'), Tissue.Valve);
    assert.equal(map.get('T_CALC'), Tissue.Calcium);
    assert.equal(map.get('T_LUNG'), Tissue.Lung);
    assert.equal(map.get('T_BONE'), Tissue.Bone);
    assert.equal(map.get('T_PERI'), Tissue.Pericardium);

    assert.equal(map.get('S_LV_CAV'), Structure.LvCavity);
    assert.equal(map.get('S_LV_SEPT'), Structure.LvWallSeptal);
    assert.equal(map.get('S_LA_CAV'), Structure.LaCavity);
    assert.equal(map.get('S_RV_CAV'), Structure.RvCavity);
    assert.equal(map.get('S_AO_ROOT'), Structure.AorticRoot);
    assert.equal(map.get('S_MV_ANT'), Structure.MitralAnterior);
    assert.equal(map.get('S_AV'), Structure.AorticValve);
  });

  it('packs canonical normal and severe AS scenes with 100% finite floats', () => {
    for (const caseId of ['normal-excellent-window', 'aortic-stenosis-severe']) {
      const caseDef = loadCaseById(caseId);
      const inp = baseInput();
      inp.rendererBackend = 'procedural';
      inp.quality = 'low';
      const core = new SimulatorCore(caseDef, inp);
      core.step(0.05);
      const beam = core.lastBeam!;
      assert.ok(beam, `Expected beam after step for ${caseId}`);

      const packed = allocPacked();
      const spec = polarSpecFor(inp.settings, 'low');
      const scene = {
        heart: core.models.heart,
        heartPose: core.heartPoseNow(),
        thorax: core.models.thorax,
        physics: scenePhysicsFor(caseDef, inp.settings),
        settings: inp.settings,
      };
      packScene(scene, beam, spec, packed);

      assert.equal(packed.data.length, PARAM_TEXELS * 4);
      for (let i = 0; i < PARAM_COUNT; i++) {
        assert.ok(
          Number.isFinite(packed.data[i]),
          `Non-finite packed float at index ${i} for ${caseId}`,
        );
      }
      assert.ok(Math.abs(packed.data[PARAM_OFFSET['B_OX']!]! - beam.origin.x) < 1e-5);
      core.dispose();
    }
  });
});
