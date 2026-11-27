import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { buildCaseModels, REST_PATIENT } from '@/simulator/anatomy/caseModels';
import { computeHeartPose } from '@/simulator/anatomy/heartModel';
import { Structure } from '@/simulator/anatomy/tissue';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import { baseInput } from '@/simulator/core/baseInput';
import { beamFrameFromPose, poseFromControl } from '@/simulator/probe/pose';
import {
  GLSL_PASS_A_MAIN,
  GLSL_PASS_A_SIDE_MAIN,
  GLSL_PASS_B_MAIN,
} from '@/simulator/renderer/gpu/glslPasses';
import { allocPacked, packScene, PARAM_OFFSET } from '@/simulator/renderer/gpu/paramLayout';
import {
  COLOR_TISSUE_PRIO_HI,
  COLOR_TISSUE_PRIO_LO,
} from '@/simulator/renderer/postprocess/colorMap';
import { ProceduralSliceRenderer } from '@/simulator/renderer/procedural/sliceRenderer';
import { computeChamberStasis, scenePhysicsFor } from '@/simulator/renderer/scenePhysics';
import { allocPolarFrame, polarSpecFor } from '@/simulator/renderer/types';
import { canonicalControl, getViewTarget } from '@/simulator/windows/viewTargets';

function meanAmpForStructure(
  amp: Float32Array,
  structure: Uint8Array,
  target: Structure,
): { mean: number; count: number } {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < structure.length; i++) {
    if (structure[i] === target) {
      sum += amp[i]!;
      count++;
    }
  }
  return { mean: count > 0 ? sum / count : 0, count };
}

describe('PR C: Spontaneous Echo Contrast (SEC) & Color write-priority physics', () => {
  it('derives chamber stasis purely from hemodynamics/anatomy (zero in normal, active in MS, AF and HFrEF)', () => {
    const normal = computeChamberStasis(loadCaseById('normal-excellent-window'));
    assert.equal(normal.secLa, 0);
    assert.equal(normal.secLv, 0);

    const ms = computeChamberStasis(loadCaseById('mitral-stenosis-severe'));
    assert.ok(ms.secLa > 0.55, `Expected marked LA SEC in severe MS, got ${ms.secLa}`);
    assert.equal(ms.secLv, 0);

    const af = computeChamberStasis(loadCaseById('af-diastolic'));
    assert.ok(af.secLa > 0.35, `Expected LA SEC in AF with dilated LA, got ${af.secLa}`);
    assert.equal(af.secLv, 0);

    const hfref = computeChamberStasis(loadCaseById('hfref-severe-mr'));
    assert.ok(hfref.secLv > 0.35, `Expected apical LV SEC in severe HFrEF, got ${hfref.secLv}`);
  });

  it('raises LA cavity backscatter smoothly in severe MS without altering myocardial reflectivity', () => {
    const c = loadCaseById('mitral-stenosis-severe');
    const settings = baseInput().settings;
    const { heart, thorax, tables } = buildCaseModels(c, REST_PATIENT);
    const pose = computeHeartPose(heart, cycleStateAt(tables, 0.25));
    const beam = beamFrameFromPose(
      poseFromControl(thorax, canonicalControl(getViewTarget('a4c'), heart, thorax)),
      1,
    );
    const spec = polarSpecFor(settings, 'low');
    const renderer = new ProceduralSliceRenderer();

    const physWithSec = scenePhysicsFor(c, settings, { bloodFrame: 0 });
    const physNoSec = { ...physWithSec, secLa: 0, secLv: 0 };

    const outSec = allocPolarFrame(spec);
    const outPlain = allocPolarFrame(spec);
    renderer.render({ heart, heartPose: pose, thorax, physics: physWithSec }, beam, spec, 0.25, outSec);
    renderer.render({ heart, heartPose: pose, thorax, physics: physNoSec }, beam, spec, 0.25, outPlain);

    const laSec = meanAmpForStructure(outSec.amplitude, outSec.structure, Structure.LaCavity);
    const laPlain = meanAmpForStructure(outPlain.amplitude, outPlain.structure, Structure.LaCavity);
    const myoSec = meanAmpForStructure(outSec.amplitude, outSec.structure, Structure.LvWallSeptal);
    const myoPlain = meanAmpForStructure(outPlain.amplitude, outPlain.structure, Structure.LvWallSeptal);

    assert.ok(laSec.count > 100, `Expected LA cavity samples in A4C, got ${laSec.count}`);
    assert.ok(
      laSec.mean > laPlain.mean * 1.15,
      `Expected SEC smoke to raise LA backscatter (with=${laSec.mean}, without=${laPlain.mean})`,
    );
    // Smoke stays clearly darker than myocardium so endocardial borders remain sharp
    assert.ok(
      laSec.mean < myoSec.mean * 0.55,
      `Expected SEC smoke (${laSec.mean}) to remain well below myocardium (${myoSec.mean})`,
    );
    // Septal myocardium unaffected except minor PSF edge overlap
    assert.ok(
      Math.abs(myoSec.mean - myoPlain.mean) / Math.max(1e-6, myoPlain.mean) < 0.03,
      `Myocardium drifted unexpectedly: ${myoSec.mean} vs ${myoPlain.mean}`,
    );
  });

  it('raises LV cavity backscatter in severely hypokinetic dilated HFrEF LV', () => {
    const c = loadCaseById('hfref-severe-mr');
    const settings = baseInput().settings;
    const { heart, thorax, tables } = buildCaseModels(c, REST_PATIENT);
    const pose = computeHeartPose(heart, cycleStateAt(tables, 0.0));
    const beam = beamFrameFromPose(
      poseFromControl(thorax, canonicalControl(getViewTarget('a4c'), heart, thorax)),
      1,
    );
    const spec = polarSpecFor(settings, 'low');
    const renderer = new ProceduralSliceRenderer();

    const physWithSec = scenePhysicsFor(c, settings, { bloodFrame: 2 });
    const physNoSec = { ...physWithSec, secLa: 0, secLv: 0 };

    const outSec = allocPolarFrame(spec);
    const outPlain = allocPolarFrame(spec);
    renderer.render({ heart, heartPose: pose, thorax, physics: physWithSec }, beam, spec, 0.0, outSec);
    renderer.render({ heart, heartPose: pose, thorax, physics: physNoSec }, beam, spec, 0.0, outPlain);

    const lvSec = meanAmpForStructure(outSec.amplitude, outSec.structure, Structure.LvCavity);
    const lvPlain = meanAmpForStructure(outPlain.amplitude, outPlain.structure, Structure.LvCavity);
    assert.ok(lvSec.count > 200);
    assert.ok(
      lvSec.mean > lvPlain.mean * 1.08,
      `Expected HFrEF LV apical SEC to raise LV cavity backscatter (${lvSec.mean} vs ${lvPlain.mean})`,
    );
  });

  it('packs SEC_LA and SEC_LV scalars cleanly into GPU paramLayout and mirrors SEC in lightweight Pass B', () => {
    const msCase = loadCaseById('mitral-stenosis-severe');
    const settings = baseInput().settings;
    const { heart, thorax, tables } = buildCaseModels(msCase, REST_PATIENT);
    const pose = computeHeartPose(heart, cycleStateAt(tables, 0.2));
    const beam = beamFrameFromPose(
      poseFromControl(thorax, canonicalControl(getViewTarget('a4c'), heart, thorax)),
      1,
    );
    const phys = scenePhysicsFor(msCase, settings);
    const packed = allocPacked();
    packScene({ heart, heartPose: pose, thorax, physics: phys }, beam, polarSpecFor(settings, 'low'), packed);

    assert.ok(Math.abs(packed.data[PARAM_OFFSET['SEC_LA']!]! - (phys.secLa ?? 0)) < 1e-6);
    assert.ok(Math.abs(packed.data[PARAM_OFFSET['SEC_LV']!]! - (phys.secLv ?? 0)) < 1e-6);

    assert.ok(GLSL_PASS_B_MAIN.includes('SEC_LA') && GLSL_PASS_B_MAIN.includes('SEC_LV'));
    assert.ok(GLSL_PASS_B_MAIN.includes('SEC_GAIN') && GLSL_PASS_B_MAIN.includes('SEC_FREQ'));
    assert.ok(GLSL_PASS_B_MAIN.includes('SEC_LAA_BOOST'));
    // Pass A and Pass A Side remain free of SEC code to preserve SwiftShader compile budget
    assert.ok(!GLSL_PASS_A_MAIN.includes('SEC_LA') && !GLSL_PASS_A_SIDE_MAIN.includes('SEC_LA'));
  });

  it('keeps tighter Color Doppler tissue write-priority window (full color on dark blood, clean valve rejection)', () => {
    assert.ok(COLOR_TISSUE_PRIO_LO >= 75 && COLOR_TISSUE_PRIO_LO <= 92);
    assert.ok(COLOR_TISSUE_PRIO_HI >= 145 && COLOR_TISSUE_PRIO_HI <= 165);
    const prio = (lum: number) => {
      const u = Math.min(
        1,
        Math.max(0, (lum - COLOR_TISSUE_PRIO_LO) / (COLOR_TISSUE_PRIO_HI - COLOR_TISSUE_PRIO_LO)),
      );
      return 1 - u * u * (3 - 2 * u);
    };
    assert.equal(prio(70), 1); // dark blood pool retains 100% color priority
    assert.ok(prio(122) < 0.5); // bright valve leaflet suppresses color overlap cleanly
    assert.equal(prio(160), 0); // calcified cusp punches completely through color jet
  });
});
