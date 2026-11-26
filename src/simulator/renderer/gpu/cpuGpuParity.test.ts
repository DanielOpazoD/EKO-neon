import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  COLOR_BLEND,
  COLOR_MAP,
  COLOR_PALETTE_IDS,
  COLOR_PALETTES,
  colorMap,
  colorMapDefinesGlsl,
  type ColorPaletteId,
} from '../postprocess/colorMap';
import { GLSL_GENERATED } from './glslGenerated';
import { GLSL_HEART } from './glslHeart';
import { noiseLatticeTexels } from './glHelpers';
import {
  glslMirrors,
  GlslUndefined,
  MIRROR_TOLERANCE,
  MIRROR_UNINFORMATIVE_SHARE,
  mirrorDeviation,
  mirrorSamples,
  type GlslBindings,
  type GlslBlockRun,
  type GlslMirror,
  type GlslValue,
} from './shaderMirrors';
import { packScene, PARAM_TEXELS, type PackedScene } from './paramLayout';
import { GLSL_PRESENT_FRAG } from './glslImage';
import {
  dopplerDefinesGlsl,
  GLSL_DOPPLER_AUTOCORR_FRAG,
  GLSL_DOPPLER_SHADE_FRAG,
  GLSL_DOPPLER_SPATIAL_FRAG,
} from './glslDoppler';
import {
  COLOR_THRESHOLD,
  DOPPLER_SHADOW_TRANSMISSION,
} from '@/simulator/doppler/color/colorDoppler';
import {
  CLUTTER_BLOOD_RATIO_POW,
  CLUTTER_JITTER_RAD,
  CLUTTER_SIGMA_V_MPS,
  COLOR_PACKET,
  NOISE_COHERENCE,
  NOISE_POWER,
  NOISE_POWER_GATE,
  TRANSIT_JITTER_RAD,
} from '@/simulator/doppler/color/kasaiEstimator';
import {
  colorPowerConfidence,
  encodeColorBreakthroughPower,
  ST_HYSTERESIS_RATIO,
  ST_POW_ALPHA_MIN,
  ST_SPATIAL_LAMINAR_MIX,
  ST_SPATIAL_TURB_MIX,
} from '@/simulator/doppler/color/spatiotemporalFilter';
import {
  DOPPLER_AUTOCORR_UNIFORMS,
  DOPPLER_SHADE_UNIFORMS,
  DOPPLER_SPATIAL_UNIFORMS,
} from './deferredDoppler';
import { loadCaseById } from '@/cases';
import { fastAtan2 } from '@/core/noise';
import { classifyAorticExtension } from '@/simulator/anatomy/aorticExtension';
import { buildCaseModels, REST_PATIENT, type CaseModels } from '@/simulator/anatomy/caseModels';
import { classifyChambers, FAR_FROM_HEART_CM } from '@/simulator/anatomy/classify';
import { classifyAorticRoot } from '@/simulator/anatomy/classify/aorticRoot';
import { classifyAtria } from '@/simulator/anatomy/classify/atria';
import { ctx, type ClassifyCtx } from '@/simulator/anatomy/classify/context';
import { classifyLeftVentricle } from '@/simulator/anatomy/classify/leftVentricle';
import { parietalSacDistance } from '@/simulator/anatomy/classify/pericardium';
import { classifyRightVentricle } from '@/simulator/anatomy/classify/rightVentricle';
import {
  pulmonaryTrunkDistance,
  pulmonaryTrunkEpiDistance,
  trunkFrontDistance,
} from '@/simulator/anatomy/greatVessels';
import {
  classifyHeart,
  computeHeartPose,
  heartAnchors,
  type HeartPose,
} from '@/simulator/anatomy/heartModel';
import { lvRegionalMotion, regionalCavityOffset } from '@/simulator/anatomy/lvGeometry';
import {
  lvCavityRadius,
  lvCavitySdf,
  lvProfileDG,
  lvProfileDGExact,
  lvProfileG,
  lvProfileGExact,
  lvRadialOffsetFactor,
} from '@/simulator/anatomy/lvShape';
import { wallThicknessAt } from '@/simulator/anatomy/lvWall';
import { rvCrescent, rvRadii } from '@/simulator/anatomy/rv';
import { makeSample, Structure, type TissueSample } from '@/simulator/anatomy/tissue';
import { skirtOffsetAt, skirtWarpScale, tvInflowSdf } from '@/simulator/anatomy/valveSkirt';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import { baseInput } from '@/simulator/core/baseInput';
import { beamFrameFromPose, poseFromControl } from '@/simulator/probe/pose';
import { speckleLattices } from '@/simulator/renderer/procedural/sliceRenderer';
import { scenePhysicsFor } from '@/simulator/renderer/scenePhysics';
import { polarSpecFor } from '@/simulator/renderer/types';
import { canonicalControl, getViewTarget } from '@/simulator/windows/viewTargets';

/**
 * CPU ↔ GPU parity guard suite.
 *
 * Catches silent divergence between CPU TypeScript physics/postprocess modules and their WebGL2
 * GLSL counterparts — including, numerically, every function and classifier block the shaders mirror by hand
 * (`glslGenerated.ts` and the other annotated mirrors, see `shaderMirrors.ts`).
 */

/**
 * GLSL mirrors evaluated in Node (shaderMirrors.ts). Each shader function preceded by `// src/<path>.ts: <export>` is
 * translated to JavaScript with its program's constants and compared with that export within MIRROR_TOLERANCE (float32
 * resolution); a vector argument is a point the TypeScript takes as its components.
 * - Those that read nothing of the frame run on MIRROR_SAMPLES arguments. Measured when written: 58 mirrors, 54 equal to
 *   the last bit on every sample and four within 1.4e-10 of max(1, |y|), all through the shader's PI printed with 15
 *   digits (rvAzProfile, where pow(x, 0.7) magnifies the last bits of sin(PI·u) near 0).
 * - Those that read the frame's textures run on FRAME_SAMPLES arguments in each frame, their texelFetch bound to the data
 *   the CPU uploads (`Frame.bindings`): P(i) to the array packScene packs for the case and the phase, the noise lattice
 *   to noiseLatticeTexels' bytes, through which `lat` and its smoothstep weights are translated and compared with
 *   latticeNoise3. Their TypeScript is called with the models the scene was packed from (FRAME_ORIGINALS). tvOffsetAt
 *   and tvInflowSdf read the tricuspid annulus's azimuth with atan where valveSkirt.ts reads it with fastAtan2 (0.005 rad
 *   apart at most): each is compared with its atan bound to fastAtan2 (FRAME_MIRROR_BINDINGS), and the other mirrors and
 *   runs are given the CPU's. Measured when written: 16 mirrors over 9 frames, within 3.1e-10 (rvRadii).
 * - The classifier, classifyHeart, block by block in its order (CLASSIFY_HEART_BLOCKS, which the first test holds to
 *   classify.ts's blocks and glslHeart.ts's sections):
 *   - the aortic extension: in the browser only (its arch reads boolean vectors); the runs are given the CPU's distance
 *     to it, and the two lines that pass it on are held as written (HELD_AS_WRITTEN);
 *   - the point after the swing, the bounding sphere and the posterior column (classifyChambers): run;
 *   - the aortic root's coordinates (rootCoordinates): run, its atan bound to fastAtan2 as root.ts reads it;
 *   - the valves: in the browser only (CLASSIFY_HEART_BLOCKS says why);
 *   - the azimuth, `float az = atan(y, x)`: held as written; the runs after it are given the CPU's, fastAtan2's;
 *   - the left ventricle, the aortic root with the LVOT and the aortomitral curtain, the atria with their appendages,
 *     septum, veins, venae cavae and coronary sinus, the right ventricle with its outflow tract, and the pericardium
 *     (classifyLeftVentricle, classifyAorticRoot, classifyAtria, classifyRightVentricle, classifyPericardium): run.
 *   Each run is given what the GLSL runs before it left (glslClassifier), on 108 000 points of the frames (over the
 *   bounding sphere, about the ventricles, the atria, the interatrial septum, the aortic root, the curtain and the
 *   tricuspid and mitral hinges, where the thin bands turn), and compared with the CPU's block on the CPU's context:
 *   what it labels, with the sample's distance, normal, material point and transmural depth, and what it leaves for the
 *   blocks after it. The pericardium runs on the points the CPU's sac draws or leaves, given all it reads by the GLSL
 *   runs before it (PERICARDIUM_INPUTS): the atria's outer faces and the packing of RA_*, RAA_* and LA_* they read are
 *   the GPU's, run in Node. Measured when written: within 2.1e-13 at every point, the pericardium's within 3.4e-14; the
 *   atria's run, which reads PV_COURSE as a constant array, met classify/atria.ts on 234 288 points with no difference
 *   before it entered this file.
 * Mutants of the GPU's wiring, the whole suite run on each (review 3, third round of its fix). Of the 16 the review of
 * the second round found alive (the atria's outer faces the sac reads and the right atrium's packing, the aortic root
 * with the LVOT and the curtain, tvOffsetAt and tvInflowSdf, the lattices of channels 0-2) and 15 more of those kinds
 * (the left atrium's centre and radii and the sleeve's radii, held as text before; the atria's scales; the appendage's
 * and the left atrium's packing; the curtain's distance and the root's material point; the inflow's bulge, the annulus's
 * tilts and the extension's warp; the CPU's own speckle and wall lattices), this suite fails 30 of the 31, where the
 * second round's failed 4 of the 26 that apply to its code (the lines it held as text, and the left atrium's radius as
 * packed). The one that passes reseeds a lattice in NOISE_LATTICE_SEED_XOR, which both sides read now: the speckle
 * changes on both alike. The translator's three faults put back fail shaderMirrors.test.ts, and a block of either
 * classifier without a state, or with its annotation gone, fails the first test. The first two rounds' 97 named mutants
 * fail as before: 88, the 8 that pass equivalent or acting on 0.01 % of the sac, and one no longer applies (the wall
 * lattice's seed, now one table for both sides). Of 797 generic ones (each literal and float constant ×1.01, each ±
 * flipped) of the GLSL the third round newly runs (the root's and the atria's runs, tvOffsetAt, tvInflowSdf), 703 fail
 * this file. Of the 94 that pass, 28 change nothing at any point of these frames (normals setSample normalises, guards
 * on the root's −99 and on a vein's side ±1, the descent's threshold the frames' TVZ clear, the posterior prism's
 * shortcut, which the blend after it ignores, the bulge's sign at the septal zone's azimuth 0, a branch at the annulus's
 * centre); the other 66 move a threshold, a blend or a wall by 1 % (a threshold at 0 by 0.1 mm), or change a term read
 * only through a max or a clip that another term holds there (the hold beside the root, the sections of the septal
 * corner and of the fossa, the membranous septum's clip): 18 000 more points about the septum and the root killed 2 of
 * the 68 that passed before them.
 *
 * Mirrors outside the translation stay with the whole-image comparison in the browser
 * (window.__echotwin.compareBackends, npm run test:gpu):
 * - the shader functions with an original that no annotation names: ahaSegment (lvGeometry.ts), skinZ, leftLungBorderX,
 *   ribCenterY (thoraxModel.ts); rootRadiusAt, mvHingeHeight, mvInflowTaper and tvWarpScale (valveSkirt.ts
 *   skirtWarpScale, which tvInflowSdf's mirror binds) run inside the classifier's runs;
 * - the classifier's valves and the aortic extension (above).
 * The Doppler programs hold scalar mirrors not annotated yet: wallFilterResponse and colorPowerConf (colorPowerConfidence
 * at COLOR_THRESHOLD) agree with their TypeScript; aliasVelocity agrees with clinical/formulas only for a positive
 * Nyquist velocity (for a non-positive one the TypeScript returns v and the shader NaN), and takes its baseline as a
 * third parameter where the TypeScript defaults it.
 */
const MIRROR_SAMPLES = 4096;
const FRAME_SAMPLES = 2048;
/** Annotated mirrors when written: 54 scalar, 20 that read the frame or take vectors, and seven runs of the classifier. */
const MIRROR_FLOOR = 81;
/**
 * The blocks of classifyHeart, by the CPU function each is (classify.ts calls them in this order) and the header of its
 * section in the GLSL (glslHeart.ts), with what this file does with the GPU's: 'run', its statements annotated and run
 * in Node against the CPU's block, or why it is left to the whole-image comparison in the browser. A block of either
 * classifier without an entry here fails the first test. Between the runs the GPU's wiring is the aortic extension's
 * distance and the azimuth, held as written (HELD_AS_WRITTEN).
 */
const CLASSIFY_HEART_BLOCKS: Record<string, { section?: string; state: 'run' | { browser: string } }> = {
  classifyAorticExtension: {
    state: { browser: 'its arch reads boolean vectors (any(lessThan(…))), which the translation does not hold' },
  },
  classifyChambers: { state: 'run' },
  rootCoordinates: { section: 'aortic root coordinates', state: 'run' },
  classifyValves: {
    section: 'valves',
    state: {
      browser:
        'not annotated: its tricuspid skirt reads the azimuth with atan where valveSkirt.ts skirtDistance uses fastAtan2, ' +
        'its mitral leaflets read theirs exactly on both sides, and a binding of atan replaces it for the whole run',
    },
  },
  classifyLeftVentricle: { section: 'LV cavity & wall', state: 'run' },
  classifyAorticRoot: { section: 'aortic root / LVOT', state: 'run' },
  classifyAtria: { section: 'atria', state: 'run' },
  classifyRightVentricle: { section: 'RV', state: 'run' },
  classifyPericardium: { section: 'pericardium & effusion', state: 'run' },
};
/** The statements of classifyHeart annotated as mirrors, by the TypeScript function each mirrors, in its order. */
const CLASSIFIER_RUNS = Object.keys(CLASSIFY_HEART_BLOCKS).filter((b) => CLASSIFY_HEART_BLOCKS[b]!.state === 'run');

/** Every GLSL string the shader modules export: a mirror annotated in any of them enters by itself. */
let sourcesCache: string[] | undefined;
async function shaderSources(): Promise<string[]> {
  if (sourcesCache) return sourcesCache;
  const here = import.meta.dirname;
  const sources: string[] = [];
  for (const file of fs.readdirSync(here).filter((f) => /^glsl\w*\.ts$/.test(f)).sort()) {
    const mod = (await import(pathToFileURL(path.join(here, file)).href)) as Record<string, unknown>;
    for (const v of Object.values(mod)) if (typeof v === 'string' && !sources.includes(v)) sources.push(v);
  }
  return (sourcesCache = sources);
}

const label = (m: GlslMirror): string => `${m.glslName || '?'} (GLSL) ↔ ${m.tsPath}: ${m.tsName}`;
const show = (v: unknown): string =>
  JSON.stringify(v, (_k, x: unknown) => (typeof x === 'number' && !Number.isInteger(x) ? +x.toPrecision(6) : x));
/** Whether every component of a value is NaN: an original that is undefined there itself. */
const allNaN = (v: GlslValue): boolean => (Array.isArray(v) ? v.every(Number.isNaN) : Number.isNaN(v as number));

/**
 * The largest deviation of a mirror from its original over samples, beyond MIRROR_TOLERANCE, or the samples that told
 * nothing (NaN on both sides, or GLSL undefined where the original is NaN) beyond MIRROR_UNINFORMATIVE_SHARE: what is
 * wrong with it, or undefined. A mirror that meets what GLSL leaves undefined (GlslUndefined) where its original gives a
 * number diverges: the GPU need not give that number.
 */
function divergence(
  samples: GlslValue[][],
  original: (...a: GlslValue[]) => GlslValue,
  mirror: (...a: GlslValue[]) => GlslValue,
): { problem?: string; deviation: number } {
  let worst = { deviation: 0, at: [] as GlslValue[], ts: 0 as GlslValue, glsl: 0 as GlslValue };
  let uninformative = 0;
  for (const x of samples) {
    let ts: GlslValue, glsl: GlslValue;
    try {
      ts = original(...x);
    } catch (e) {
      return { problem: `the TypeScript throws at ${show(x)}: ${(e as Error).message}`, deviation: Infinity };
    }
    try {
      // a function with out parameters gives its value and then theirs: their components in a row, as the original's
      glsl = mirror(...x);
      if (Array.isArray(glsl)) glsl = glsl.flat();
    } catch (e) {
      if (e instanceof GlslUndefined && allNaN(ts)) {
        uninformative++;
        continue;
      }
      const what = e instanceof GlslUndefined ? (e as Error).message : `throws: ${(e as Error).message}`;
      return { problem: `at ${show(x)} TypeScript ${show(ts)}, GLSL ${what}`, deviation: Infinity };
    }
    const deviation = mirrorDeviation(ts, glsl);
    if (Number.isNaN(deviation)) uninformative++;
    else if (deviation > worst.deviation) worst = { deviation, at: x, ts, glsl };
  }
  if (worst.deviation > MIRROR_TOLERANCE)
    return { problem: `at ${show(worst.at)} TypeScript ${show(worst.ts)}, GLSL ${show(worst.glsl)}`, deviation: worst.deviation };
  if (uninformative > MIRROR_UNINFORMATIVE_SHARE * samples.length)
    return { problem: `${uninformative} of ${samples.length} samples tell nothing (NaN or undefined on both sides)`, deviation: worst.deviation };
  return { deviation: worst.deviation };
}

/** The annotations however they are laid out, read more loosely than the translator does: each must be a mirror. */
const LAX_ANNOTATION = /\/\/\s*(src\/\S*?\.ts)\s*:\s*(\w+)/g;

test('every annotated GLSL function and block is a mirror, and the translation reads every one', async () => {
  const sources = await shaderSources();
  const mirrors = glslMirrors(sources);
  const read = new Set(mirrors.map((m) => `${m.tsPath}: ${m.tsName}`));
  const annotated = new Set(sources.flatMap((s) => [...s.matchAll(LAX_ANNOTATION)].map((m) => `${m[1]}: ${m[2]}`)));
  assert.deepEqual([...annotated].filter((a) => !read.has(a)), [], 'annotations no mirror was made from');
  assert.ok(mirrors.length >= MIRROR_FLOOR, `${mirrors.length} mirrors, fewer than the ${MIRROR_FLOOR} annotated before`);
  // every function of glslGenerated.ts is one of them: a definition without its `// src/….ts: name` line is unchecked
  const generated = [...GLSL_GENERATED.matchAll(/^\w+\s+(\w+)\s*\(/gm)].map((m) => m[1]!);
  assert.ok(generated.length > 0, 'no function definition found in GLSL_GENERATED');
  assert.deepEqual(
    generated.filter((name) => !mirrors.some((m) => m.glslName === name)),
    [],
    'GLSL_GENERATED functions without the annotation naming their TypeScript',
  );
  const untranslated = mirrors.filter((m) => m.reason).map((m) => `${label(m)}: ${m.reason}`);
  assert.deepEqual(untranslated, [], `GLSL mirrors the translation cannot evaluate:\n${untranslated.join('\n')}`);
  // and every annotated block is one this file runs against its TypeScript
  const blocks = mirrors.filter((m) => m.kind === 'block').map((m) => m.tsName);
  assert.deepEqual(blocks.filter((n) => !CLASSIFIER_RUNS.includes(n)), [], 'annotated statements of a GLSL function no test runs');
  assert.deepEqual(CLASSIFIER_RUNS.filter((n) => !blocks.includes(n)), [], 'runs of the classifier this file compares, no longer annotated');
  // and every block of classifyHeart has a state here: the CPU's as classify.ts calls them, the GPU's by its sections
  const cpu = fs
    .readFileSync(path.resolve(import.meta.dirname, '../../anatomy/classify.ts'), 'utf8')
    .replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '');
  const cpuBlocks = [...new Set([...cpu.matchAll(/(?<!function )\b(classify[A-Z]\w*|rootCoordinates)\(/g)].map((m) => m[1]!))];
  const glslHeart = GLSL_HEART.slice(GLSL_HEART.indexOf('bool classifyHeart('));
  const sections = [...glslHeart.matchAll(/^[ \t]*\/\/ -{10} (.+?) -{10}[ \t]*$/gm)].map((m) => m[1]!);
  assert.ok(cpuBlocks.length >= 9 && sections.length >= 7, `${cpuBlocks.length} blocks read in classify.ts, ${sections.length} GLSL sections`);
  assert.deepEqual(cpuBlocks.filter((b) => !CLASSIFY_HEART_BLOCKS[b]), [], 'blocks of classify.ts without a state in CLASSIFY_HEART_BLOCKS');
  assert.deepEqual(Object.keys(CLASSIFY_HEART_BLOCKS).filter((b) => !cpuBlocks.includes(b)), [], 'blocks of CLASSIFY_HEART_BLOCKS classify.ts no longer calls');
  const named = Object.values(CLASSIFY_HEART_BLOCKS).map((b) => b.section);
  assert.deepEqual(sections.filter((h) => !named.includes(h)), [], 'sections of the GLSL classifyHeart without a state in CLASSIFY_HEART_BLOCKS');
  assert.deepEqual(named.filter((h) => h && !sections.includes(h)), [], 'sections of CLASSIFY_HEART_BLOCKS the GLSL classifyHeart no longer has');
});

test('GLSL mirrors of TypeScript functions evaluate as their originals on sampled inputs', async (t) => {
  const mirrors = glslMirrors(await shaderSources()).filter((m) => m.kind === 'function' && m.glsl);
  const problems: string[] = [];
  let worst = { deviation: 0, label: '' };
  for (const m of mirrors) {
    const file = path.resolve(import.meta.dirname, '../../../..', m.tsPath);
    const mod = fs.existsSync(file) ? ((await import(pathToFileURL(file).href)) as Record<string, unknown>) : {};
    const original = mod[m.tsName];
    // a vector argument is a point or a vector the TypeScript takes as its components
    const arity = m.paramTypes.reduce((n, t) => n + Number(/^i?vec(\d)$/.exec(t)?.[1] ?? 1), 0);
    if (typeof original !== 'function') problems.push(`${label(m)}: no such exported function`);
    else if (original.length !== arity)
      problems.push(`${label(m)}: ${original.length} parameters in TypeScript, ${arity} components in GLSL`);
    else {
      const ts = (...x: GlslValue[]) => (original as (...a: GlslValue[]) => GlslValue)(...x.flat());
      const d = divergence(mirrorSamples(m, MIRROR_SAMPLES), ts, m.glsl!);
      if (d.problem) problems.push(`${label(m)}: ${d.problem}`);
      else if (d.deviation > worst.deviation) worst = { deviation: d.deviation, label: label(m) };
    }
  }
  t.diagnostic(
    `${mirrors.length} GLSL mirrors × ${MIRROR_SAMPLES} samples; largest deviation within tolerance ` +
      `${worst.deviation.toExponential(1)} of max(1, |y|) (${worst.label})`,
  );
  assert.deepEqual(problems, [], `GLSL mirrors that diverge from their TypeScript:\n${problems.join('\n')}`);
});

/** One frame: a case's models, its pose at a phase and the parameter texture packed from them. */
interface Frame extends CaseModels {
  label: string;
  A: ReturnType<typeof heartAnchors>;
  pose: HeartPose;
  data: Float64Array;
  /** The scene's seed, and the noise lattice's texels as uploadNoiseLattice3D uploads them for it. */
  seed: number;
  texels: Uint8Array;
  /**
   * What the shader reads of textures, as the CPU uploads it, and what the CPU computes another way:
   * - P(i), whose texel of uParams is data[i] (packScene's layout, four floats a texel), read from the array directly;
   * - uNoise, texelFetch of the RGBA8 texels uploadNoiseLattice3D uploads (noiseLatticeTexels: each byte / 255), so
   *   that `lat` and its smoothstep weights are the shader's own and channel 3 meets heart.wallNoise in the originals;
   * - the profile g(ζ) and g′(ζ) as the CPU reads them, from the 512-bin table of the analytic profile (lvShape.ts
   *   lvProfileG, lvProfileDG) that the shader evaluates (lvProfileGExact, lvProfileDGExact): the functions built on the
   *   profile are compared on their own formula; the table's own error is the CPU's approximation, outside this
   *   comparison (when written, up to 0.03 of g at the top of the dome and in the last bin before the apex and 7e-4 in
   *   the body, and 0.75 of g′ at the annulus, where g′ steps from the dome's 0 to the slope of the neck);
   * - the tricuspid annulus's offset and inflow column at the azimuth the CPU reads with fastAtan2 (valveSkirt.ts
   *   skirtOffsetAt, tvInflowSdf), which the shader's tvOffsetAt and tvInflowSdf read with atan: each is a mirror of its
   *   own, compared with the CPU's at the CPU's azimuth (FRAME_MIRROR_BINDINGS).
   */
  bindings: GlslBindings;
}
/**
 * The four cases whose GPU wiring the pericardium and the regional motion changed, each at end-diastole, at its most
 * contracted phase and, when the heart swings (tamponade), at its largest swing.
 */
const FRAME_CASES = [
  'pericardial-effusion-tamponade',
  'inferior-rwma',
  'pulmonary-hypertension-rv',
  'normal-excellent-window',
];
let framesCache: Frame[] | undefined;
function frames(): Frame[] {
  if (framesCache) return framesCache;
  const settings = baseInput().settings;
  const out: Frame[] = [];
  for (const caseId of FRAME_CASES) {
    const c = loadCaseById(caseId);
    const models = buildCaseModels(c, REST_PATIENT);
    const { heart, thorax, tables } = models;
    const A = heartAnchors(heart);
    const beam = beamFrameFromPose(poseFromControl(thorax, canonicalControl(getViewTarget('a4c'), heart, thorax)), 1);
    const physics = scenePhysicsFor(c, settings);
    const texels = noiseLatticeTexels(physics.seed);
    const uNoise = (coord: GlslValue, lod: GlslValue): number[] => {
      const [x, y, z] = coord as number[];
      if (lod !== 0 || !(x! >= 0 && x! < 128 && y! >= 0 && y! < 128 && z! >= 0 && z! < 128))
        throw new RangeError(`uNoise read at ${show(coord)}, level ${lod}`);
      const i = ((z! * 128 + y!) * 128 + x!) * 4;
      return [texels[i]! / 255, texels[i + 1]! / 255, texels[i + 2]! / 255, texels[i + 3]! / 255];
    };
    let contracted = 0,
      swung = 0,
      most = -1,
      widest = 0;
    for (let i = 0; i < 200; i++) {
      const pose = computeHeartPose(heart, cycleStateAt(tables, i / 200));
      if (pose.state.contraction > most) [contracted, most] = [i / 200, pose.state.contraction];
      if (Math.abs(pose.swingX) > widest) [swung, widest] = [i / 200, Math.abs(pose.swingX)];
    }
    for (const phase of new Set([0, contracted, swung])) {
      const pose = computeHeartPose(heart, cycleStateAt(tables, phase));
      // packed in double precision: the values the texture receives before its float32 rounding, which MIRROR_TOLERANCE
      // leaves out as it does for the scalar mirrors
      const packed: PackedScene = { data: new Float64Array(PARAM_TEXELS * 4) as unknown as Float32Array };
      packScene({ heart, heartPose: pose, thorax, physics }, beam, polarSpecFor(settings, 'low'), packed);
      const data = packed.data as unknown as Float64Array;
      const tv = pose.valves.tv;
      out.push({
        ...models,
        label: `${caseId} at phase ${phase.toFixed(3)}`,
        A,
        pose,
        data,
        seed: physics.seed,
        texels,
        bindings: {
          P: (i) => data[i as number]!,
          uNoise,
          lvProfileG: (zeta) => lvProfileG(heart.lv.shape, zeta as number),
          lvProfileDG: (zeta) => lvProfileDG(heart.lv.shape, zeta as number),
          tvOffsetAt: (q) => skirtOffsetAt(tv, (q as number[])[0]!, (q as number[])[1]!),
          tvInflowSdf: (p) => tvInflowSdf((p as number[])[0]!, (p as number[])[1]!, (p as number[])[2]!, tv, pose.tvZ),
        },
      });
    }
  }
  return (framesCache = out);
}

/** atan as the CPU reads azimuths: fastAtan2 (core/noise.ts), within 0.005 rad of the GPU's exact atan. */
const cpuAtan = (y: GlslValue, x?: GlslValue): number =>
  x === undefined ? Math.atan(y as number) : fastAtan2(y as number, x as number);

type Original = (...a: never[]) => GlslValue;
/**
 * The TypeScript originals of the mirrors that read the frame, as functions of the shader's arguments; what else the
 * TypeScript takes comes from the case's models and the pose the frame was packed from (null: the frame has no such
 * structure, as no sac without an effusion).
 */
const FRAME_ORIGINALS: Record<string, (f: Frame) => Original | null> = {
  lvProfileG: ({ heart }) => (zeta: number) => lvProfileGExact(heart.lv.shape, zeta),
  lvProfileDG: ({ heart }) => (zeta: number) => lvProfileDGExact(heart.lv.shape, zeta),
  lvCavityRadius: ({ heart, pose }) => (az: number, z: number) => lvCavityRadius(heart.lv.shape, pose.prof, az, z),
  // the distance and then the normal the shader writes to its out parameter
  lvCavitySdf: ({ heart, pose }) => {
    const n = new Float64Array(3);
    return (xs: number, y: number, z: number) => [lvCavitySdf(pose.prof, heart.lv.shape.ratio, xs, y, z, n), ...n];
  },
  lvRadialOffsetFactor: ({ heart, pose }) => (az: number, z: number) =>
    lvRadialOffsetFactor(heart.lv.shape, pose.prof, az, z),
  lvRegionalMotion: ({ heart }) => (az: number, levelFrac: number) => lvRegionalMotion(heart.segAmp, az, levelFrac),
  regionalCavityOffset: ({ heart, pose }) => (az: number, levelFrac: number, motion: number) =>
    regionalCavityOffset(heart.lv, pose.rMax, az, levelFrac, motion),
  wallThicknessAt: ({ heart, pose }) => (az: number, levelFrac: number, motion: number) =>
    wallThicknessAt(heart, pose.thickK, az, levelFrac, motion),
  rvRadii: ({ heart, A, pose }) => {
    const res = new Float64Array(4);
    return (az: number, z: number, contraction: number, tvZ: number, rvCollapse: number) => {
      rvRadii(heart, A, pose.prof, pose.thickK, pose.zAnn, pose.lengthNow, tvZ, contraction, pose.septalShiftCm, rvCollapse, az, z, res);
      return [...res];
    };
  },
  rvCrescent: ({ heart, A, pose }) => {
    const res = new Float64Array(3);
    return ([x, y, z]: number[], az: number) => {
      rvCrescent(heart, pose, A, x!, y!, z!, az, res);
      return [...res];
    };
  },
  pulmonaryTrunkDist: ({ A, pose }) => ([x, y, z]: number[]) => pulmonaryTrunkDistance(A, pose.pvZ, x!, y!, z!),
  pulmonaryTrunkEpiDist: ({ A }) => ([x, y, z]: number[]) => pulmonaryTrunkEpiDistance(A, x!, y!, z!),
  trunkFrontDist: ({ heart, A }) => ([x, y, z]: number[], dPaEpi: number) =>
    trunkFrontDistance(heart, A, x!, y!, z!, dPaEpi),
  parietalSacDistance: ({ A }) =>
    A.parietal ? (x: number, y: number, z: number) => parietalSacDistance(A.parietal!, x, y, z) : null,
  tvOffsetAt: ({ pose }) => ([x, y]: number[]) => skirtOffsetAt(pose.valves.tv, x!, y!),
  tvInflowSdf: ({ pose }) => ([x, y, z]: number[]) => tvInflowSdf(x!, y!, z!, pose.valves.tv, pose.tvZ),
};
/**
 * What a mirror of the frame reads as the CPU computes it, besides Frame.bindings: the tricuspid annulus's azimuth, which
 * valveSkirt.ts reads with fastAtan2 and the shader with atan (bound for the mirror's whole translation), and the warp of
 * the annulus's extension, which the CPU reads exactly (skirtWarpScale, Math.atan2) where that binding would not.
 */
const FRAME_MIRROR_BINDINGS: Record<string, (f: Frame) => GlslBindings> = {
  tvOffsetAt: () => ({ atan: cpuAtan }),
  tvInflowSdf: ({ pose }) => ({
    atan: cpuAtan,
    tvWarpScale: (d) => skirtWarpScale(pose.valves.tv, (d as number[])[0]!, (d as number[])[1]!),
  }),
};

test('GLSL mirrors that read the packed scene evaluate as their originals in four cases over the beat', async (t) => {
  const mirrors = glslMirrors(await shaderSources()).filter((m) => m.kind === 'function' && m.bind && !m.glsl);
  const problems = mirrors.filter((m) => !FRAME_ORIGINALS[m.glslName]).map((m) => `${label(m)}: no FRAME_ORIGINALS entry`);
  let worst = { deviation: 0, label: '' },
    runs = 0;
  for (const f of frames()) {
    for (const m of mirrors) {
      const original = FRAME_ORIGINALS[m.glslName]?.(f);
      if (!original) continue;
      // the mirror's own GLSL, the functions it calls bound as the CPU computes them
      const { [m.glslName]: _own, ...bindings } = f.bindings;
      const samples = mirrorSamples(m, FRAME_SAMPLES, m.parameterReads.map((i) => f.data[i]!));
      const glsl = m.bind!({ ...bindings, ...FRAME_MIRROR_BINDINGS[m.glslName]?.(f) });
      const d = divergence(samples, original as (...a: GlslValue[]) => GlslValue, glsl);
      runs++;
      if (d.problem) problems.push(`${label(m)} in ${f.label}: ${d.problem}`);
      else if (d.deviation > worst.deviation) worst = { deviation: d.deviation, label: `${label(m)} in ${f.label}` };
    }
  }
  t.diagnostic(
    `${mirrors.length} mirrors over ${frames().length} frames (${runs} runs × ${FRAME_SAMPLES} samples); largest ` +
      `deviation within tolerance ${worst.deviation.toExponential(1)} (${worst.label})`,
  );
  assert.ok(runs >= mirrors.length * FRAME_CASES.length, `only ${runs} runs`);
  assert.deepEqual(problems, [], `GLSL mirrors that diverge from their TypeScript:\n${problems.join('\n')}`);
});

test('the noise lattices the GPU reads are the CPU speckle lattices and the heart wall lattice', () => {
  // each channel of the uploaded texels against the lattice the CPU reads for it, byte for byte (the mirrors above meet
  // channel 3 only where their samples read it): channels 0-2 the slice renderer's speckle lattices A, B and C, channel 3
  // the heart's wall lattice. Another seed or channel order would speckle the B-mode or texture the walls otherwise
  for (const f of frames().filter((x, i, all) => all.findIndex((y) => y.heart === x.heart) === i)) {
    const { latA, latB, latC } = speckleLattices(f.seed);
    const cpu = [latA, latB, latC, f.heart.wallNoise];
    const differ = cpu.map((lattice, k) => {
      let n = 0;
      for (let i = 0; i < lattice.length; i++) if (f.texels[i * 4 + k] !== lattice[i]) n++;
      return n;
    });
    assert.equal(f.texels.length, latA.length * 4);
    assert.deepEqual(differ, [0, 0, 0, 0], `${f.label}: texels of channels 0-3 that are not the CPU's lattice`);
  }
});

/** Names of the structures the classifier's runs label, for the counts and the messages (Structure is a const enum). */
const STRUCTURE_NAMES = new Map<number, string>([
  [Structure.LvCavity, 'LvCavity'],
  [Structure.LvWallSeptal, 'LvWallSeptal'],
  [Structure.LvWallLateral, 'LvWallLateral'],
  [Structure.LvWallAnterior, 'LvWallAnterior'],
  [Structure.LvWallInferior, 'LvWallInferior'],
  [Structure.LvApex, 'LvApex'],
  [Structure.LaCavity, 'LaCavity'],
  [Structure.LaWall, 'LaWall'],
  [Structure.LaAppendage, 'LaAppendage'],
  [Structure.PapillaryMuscle, 'PapillaryMuscle'],
  [Structure.AorticRoot, 'AorticRoot'],
  [Structure.Lvot, 'Lvot'],
  [Structure.RvCavity, 'RvCavity'],
  [Structure.RvWall, 'RvWall'],
  [Structure.RaCavity, 'RaCavity'],
  [Structure.RaWall, 'RaWall'],
  [Structure.RaAppendage, 'RaAppendage'],
  [Structure.InteratrialSeptum, 'InteratrialSeptum'],
  [Structure.PulmonaryVein, 'PulmonaryVein'],
  [Structure.Svc, 'Svc'],
  [Structure.Ivc, 'Ivc'],
  [Structure.HepaticVein, 'HepaticVein'],
  [Structure.CoronarySinus, 'CoronarySinus'],
  [Structure.Rvot, 'Rvot'],
  [Structure.PulmonaryArtery, 'PulmonaryArtery'],
  [Structure.ModeratorBand, 'ModeratorBand'],
  [Structure.RvPapillary, 'RvPapillary'],
]);
const structureName = (s: number): string => STRUCTURE_NAMES.get(s) ?? String(s);
/** A GLSL `Sample` as the translation leaves it. */
type GlslSample = Record<'tissue' | 'structure' | 'segment' | 'sdf' | 'extra' | 'transmural', number> & Record<'n' | 'm', number[]>;
/**
 * How far the GLSL sample of a labelled point is from the CPU's over setSample's fields (Infinity when it labels the
 * point otherwise), and what is wrong beyond MIRROR_TOLERANCE.
 */
function sampleDifference(cpu: TissueSample, s: GlslSample): { deviation: number; problem?: string } {
  if (s.structure !== cpu.structure || s.tissue !== cpu.tissue || s.segment !== cpu.segment)
    return {
      deviation: Infinity,
      problem: `CPU structure ${cpu.structure} tissue ${cpu.tissue} segment ${cpu.segment}, GLSL ${s.structure} ${s.tissue} ${s.segment}`,
    };
  const cpuValues = [cpu.sdf, cpu.nx, cpu.ny, cpu.nz, cpu.mx, cpu.my, cpu.mz, cpu.extraReflect, cpu.transmural];
  const glslValues = [s.sdf, ...s.n, ...s.m, s.extra, s.transmural];
  const deviation = mirrorDeviation(cpuValues, glslValues);
  return deviation <= MIRROR_TOLERANCE
    ? { deviation }
    : { deviation, problem: `${structureName(cpu.structure)}: CPU sdf, n, m, extra, transmural ${show(cpuValues)}, GLSL ${show(glslValues)}` };
}
/** How far a value a run leaves is from the CPU's: a bool 0 or Infinity, numbers as mirrorDeviation (NaN: Infinity). */
const valueDeviation = (cpu: unknown, glsl: unknown): number => {
  if (typeof cpu === 'boolean') return cpu === glsl ? 0 : Infinity;
  const d = mirrorDeviation(cpu as GlslValue, glsl as GlslValue);
  return Number.isNaN(d) ? Infinity : d;
};
/**
 * What the CPU's context holds, after classify.ts set it and a block ran on it, of what a GLSL run leaves for the blocks
 * after it (the root and LVOT leave them nothing).
 */
const RUN_OUTPUTS: Record<string, Record<string, (c: ClassifyCtx) => unknown>> = {
  classifyChambers: {
    p: (c) => [c.x, c.y, c.z],
    x: (c) => c.x,
    y: (c) => c.y,
    z: (c) => c.z,
    colDist: (c) => c.colDist,
    zAnn: (c) => c.hp.zAnn,
  },
  rootCoordinates: {
    rootT: (c) => c.rootT,
    rootRr: (c) => c.rootRr,
    rootR: (c) => c.rootR,
    rootPhi: (c) => c.rootPhi,
    rootQ: (c) => [c.rootQx, c.rootQy, c.rootQz],
    inRootLumen: (c) => c.inRootLumen,
    inOutflowLumen: (c) => c.inOutflowLumen,
  },
  classifyLeftVentricle: {
    levelFrac: (c) => c.levelFrac,
    n0: (c) => [c.nx0, c.ny0, c.nz0],
    dEllR: (c) => c.dEllR,
    wallT: (c) => c.wallT,
    crestLoss: (c) => c.crestLoss,
    zHinge: (c) => c.zHinge,
    inAnnularRegion: (c) => c.inAnnularRegion,
  },
  classifyAorticRoot: {},
  // the atria's outer faces, which the pericardium reads, and the left atrium's centre and radii (LA_*)
  classifyAtria: {
    laEpi: (c) => c.laEpi,
    raSleeve: (c) => c.raSleeve,
    raBody: (c) => c.raBody,
    raaEpi: (c) => c.raaEpi,
    raColumn: (c) => c.raColumn,
    la: (c) => [c.A.laCenter.x, c.A.laCenter.y, c.A.laCenter.z],
    lr: (c) => [c.A.laR.x, c.A.laR.y, c.A.laR.z],
  },
  classifyRightVentricle: { dCavRv: (c) => c.rvSdf[0] },
};
/** The CPU's blocks the runs after classifyChambers and rootCoordinates are compared with, each on the CPU's context. */
const CPU_BLOCKS: Record<string, (c: ClassifyCtx) => boolean> = {
  classifyLeftVentricle,
  classifyAorticRoot,
  classifyAtria,
  classifyRightVentricle,
};
type Runs = Record<string, (inputs: Record<string, unknown>) => GlslBlockRun>;
/**
 * The GLSL runs of the classifier, bound to a frame's textures, by the TypeScript function each mirrors. The root's
 * reads its azimuth around the root axis with atan where root.ts uses fastAtan2: that run's atan is the CPU's.
 */
let mirrorsCache: GlslMirror[] | undefined;
async function classifierRuns(f: Frame): Promise<Runs> {
  const mirrors = (mirrorsCache ??= glslMirrors(await shaderSources()));
  return Object.fromEntries(
    CLASSIFIER_RUNS.map((n) => {
      const m = mirrors.find((x) => x.kind === 'block' && x.tsName === n);
      assert.ok(m?.bindBlock, m?.reason ?? `no mirror of ${n} in classifyHeart`);
      return [n, m.bindBlock(n === 'rootCoordinates' ? { ...f.bindings, atan: cpuAtan } : f.bindings)];
    }),
  );
}
/** The runs before the sac, in classifyHeart's order. */
const BEFORE_SAC = CLASSIFIER_RUNS.filter((n) => n !== 'classifyPericardium');
/**
 * The GPU's classifier on a point up to the sac, as its GLSL runs: each run given what the runs before it left, from
 * the point before the swing and the distance of the aortic extension, whose GLSL is not run (the CPU's: what
 * classifyHeart holds in vascularDistance), with the azimuth as the CPU reads it, fastAtan2 of the point after the swing
 * (the GPU's line is held as written). What each run left, and the run that returned (that labelled the point or, the
 * chambers, stopped it outside the bounding sphere), when one did.
 */
function glslClassifier(runs: Runs, p0: number[], vascularDistance: number): { left: Record<string, GlslBlockRun>; stop?: string } {
  const known: Record<string, unknown> = { p0, vascularDistance };
  const left: Record<string, GlslBlockRun> = {};
  for (const name of BEFORE_SAC) {
    if (name === 'classifyLeftVentricle') known.az = fastAtan2(known.y as number, known.x as number);
    // a sample of its own for each run (a runner reads its inputs from what it is given, and writes only the sample)
    known.s = {};
    const run = runs[name]!(known);
    left[name] = run;
    if (run.returned !== undefined) return { left, stop: name };
    Object.assign(known, run.values);
  }
  return { left };
}
/**
 * Points of the frame before the swing: over the bounding sphere, around the ventricles' axis, at the tricuspid and
 * mitral hinges and the aortomitral curtain, about the atria, the interatrial septum and the aortic root, and just
 * outside the sphere.
 */
function classifierPoints(f: Frame): [number, number, number][] {
  const { boundCenter: bc, boundRadius: R } = f.heart;
  const sx = f.pose.swingX;
  let seed = 11;
  const rnd = (): number => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;
  const points: [number, number, number][] = [];
  for (let k = 0; k < 2500; k++) points.push([bc.x + sx + R * (2 * rnd() - 1), bc.y + R * (2 * rnd() - 1), bc.z + R * (2 * rnd() - 1)]);
  // the ventricles: within 7.5 cm of the long axis, from above the annulus to beyond the apex
  const z0 = f.pose.zAnn - 2,
    z1 = f.heart.lv.lengthCm + 1;
  for (let k = 0; k < 3500; k++) {
    const r = 7.5 * Math.sqrt(rnd()),
      a = 2 * Math.PI * rnd();
    points.push([sx + r * Math.cos(a), r * Math.sin(a), z0 + (z1 - z0) * rnd()]);
  }
  // the hinges, where thin bands of the ventricles turn: the tricuspid ring, from inside it to beyond the reach of the
  // free wall's hinge, about the annulus plane (decision 367); the mitral outline in the annular plane (decision 327)
  const { tv, mitral: mv } = f.pose.valves;
  const tvPlane = f.A.tvCenter.z + f.pose.tvZ;
  for (let k = 0; k < 1000; k++) {
    const r = tv.R - 0.6 + 2.2 * rnd(),
      a = 2 * Math.PI * rnd();
    points.push([sx + tv.cx + r * Math.cos(a), tv.cy + r * Math.sin(a), tvPlane - 1.2 + 1.6 * rnd()]);
  }
  for (let k = 0; k < 600; k++) {
    const r = (mv.R + 0.3) * Math.sqrt(rnd()),
      a = 2 * Math.PI * rnd();
    points.push([sx + mv.cx + r * Math.cos(a), mv.cy + r * Math.sin(a), f.pose.zAnn - 0.6 * rnd()]);
  }
  // the aortomitral curtain, in front of the mitral outline's straight anterior edge and basal of the annulus (decisions
  // 327 and 328), where the root's block draws it
  for (let k = 0; k < 600; k++) {
    const along = mv.D - 0.5 + 2 * rnd(),
      across = 3 * rnd() - 1.5;
    points.push([sx + mv.cx + mv.ux * along - mv.uy * across, mv.cy + mv.uy * along + mv.ux * across, f.pose.zAnn - 1.3 * rnd()]);
  }
  // the atria, their appendages, veins and walls: boxes 1.6 times their radii about the left and right atria
  for (const [c, r] of [
    [f.A.laCenter, f.A.laR],
    [f.A.raCenter, f.A.raR],
  ] as const)
    for (let k = 0; k < 1500; k++)
      points.push([sx + c.x + 1.6 * r.x * (2 * rnd() - 1), c.y + 1.6 * r.y * (2 * rnd() - 1), c.z + 1.6 * r.z * (2 * rnd() - 1)]);
  // the interatrial septum with its fossa and the left atrium's septal corner down to the annulus (decisions 318, 327),
  // and the root's surroundings, where the atria give way to it (the membranous septum, the hold, decisions 222, 341)
  const { laCenter: la, laR: lr, iasX, avCenter: av } = f.A;
  for (let k = 0; k < 1000; k++)
    points.push([sx + iasX - 1 + 2.5 * rnd(), la.y + 1.2 * lr.y * (2 * rnd() - 1), f.pose.zAnn - 3.5 + 4 * rnd()]);
  for (let k = 0; k < 1000; k++) points.push([sx + av.x + 2.5 * (2 * rnd() - 1), av.y + 2.5 * (2 * rnd() - 1), av.z + 2.5 * (2 * rnd() - 1)]);
  for (let k = 0; k < 100; k++) {
    const u = [2 * rnd() - 1, 2 * rnd() - 1, 2 * rnd() - 1];
    const l = Math.hypot(...u) || 1;
    const r = R * (1.0005 + 0.05 * rnd());
    points.push([bc.x + sx + (r * u[0]!) / l, bc.y + (r * u[1]!) / l, bc.z + (r * u[2]!) / l]);
  }
  return points;
}

/** Points of each kind the classifier's points reach in each block's branches, at least (a third or less of when written). */
const MIN_LABELLED: [string, number][] = [
  ['classifyLeftVentricle LvCavity', 300],
  ['classifyLeftVentricle PapillaryMuscle', 20],
  ['classifyLeftVentricle LvWallInferior', 100],
  ['classifyLeftVentricle LvWallSeptal', 100],
  ['classifyLeftVentricle LvApex', 20],
  ['classifyLeftVentricle LaCavity', 20],
  ['classifyLeftVentricle leaves', 5000],
  ['classifyAorticRoot AorticRoot', 1000],
  ['classifyAorticRoot Lvot', 400],
  // the aortomitral curtain
  ['classifyAorticRoot LvWallSeptal', 40],
  ['classifyAorticRoot leaves', 25000],
  ['classifyAtria LaCavity', 2000],
  ['classifyAtria LaWall', 500],
  ['classifyAtria LaAppendage', 30],
  ['classifyAtria RaCavity', 2000],
  ['classifyAtria RaWall', 400],
  ['classifyAtria RaAppendage', 100],
  // the vestibule past the tricuspid annulus
  ['classifyAtria RvCavity', 130],
  ['classifyAtria InteratrialSeptum', 200],
  ['classifyAtria PulmonaryVein', 200],
  ['classifyAtria Svc', 300],
  ['classifyAtria Ivc', 500],
  ['classifyAtria CoronarySinus', 90],
  ['classifyAtria leaves', 18000],
  ['classifyRightVentricle RvCavity', 300],
  ['classifyRightVentricle RvWall', 100],
  ['classifyRightVentricle Rvot', 30],
  ['classifyRightVentricle PulmonaryArtery', 30],
  ['classifyRightVentricle leaves', 5000],
  ['rootCoordinates along the root', 200],
];

test('the GLSL classifier before the sac runs as classify.ts and its blocks do: the swing, the root, the LV, the LVOT, the atria and the RV', async (t) => {
  const problems: string[] = [];
  const labelled = new Map<string, number>();
  let inside = 0,
    outside = 0,
    swing = 0,
    worst = 0;
  const out = makeSample(),
    aorta = makeSample();
  const cpuOut: Record<string, TissueSample> = Object.fromEntries(Object.keys(CPU_BLOCKS).map((n) => [n, makeSample()]));
  const compare = (at: string, name: string, cpuHit: boolean, cpuSample: TissueSample, run: GlslBlockRun) => {
    // a CPU block returns false to let the next one try; a GLSL one that returns false ends classifyHeart with a miss,
    // which only the bounding sphere's test may do (classifyChambers, outside it)
    if (run.returned === false) return problems.push(`${at}: the GLSL's ${name} ends the classifier with a miss, the CPU's ${cpuHit ? 'labels it' : 'goes on'}`);
    const glslHit = run.returned === true;
    if (glslHit !== cpuHit) return problems.push(`${at}: the CPU's ${name} ${cpuHit ? 'labels it' : 'leaves it'}, the GLSL's ${glslHit ? 'labels it' : 'leaves it'}`);
    if (cpuHit) {
      const kind = `${name} ${structureName(cpuSample.structure)}`;
      labelled.set(kind, (labelled.get(kind) ?? 0) + 1);
      const d = sampleDifference(cpuSample, run.values.s as GlslSample);
      if (d.problem) problems.push(`${at}: ${name}: ${d.problem}`);
      else worst = Math.max(worst, d.deviation);
      return;
    }
    labelled.set(`${name} leaves`, (labelled.get(`${name} leaves`) ?? 0) + 1);
    for (const [v, cpu] of Object.entries(RUN_OUTPUTS[name]!)) {
      const d = valueDeviation(cpu(ctx), run.values[v]);
      if (d > MIRROR_TOLERANCE) problems.push(`${at}: ${name} leaves ${v} ${show(run.values[v])}, the CPU ${show(cpu(ctx))}`);
      else worst = Math.max(worst, d);
    }
  };
  for (const f of frames()) {
    const runs = await classifierRuns(f);
    const { boundCenter: bc, boundRadius: R } = f.heart;
    swing = Math.max(swing, Math.abs(f.pose.swingX));
    for (const [x0, y, z] of classifierPoints(f)) {
      const at = `${f.label}, (${x0.toFixed(3)}, ${y.toFixed(3)}, ${z.toFixed(3)}) before the swing`;
      aorta.sdf = FAR_FROM_HEART_CM;
      classifyAorticExtension(f.pose.aorta, x0, y, z, aorta);
      const glsl = glslClassifier(runs, [x0, y, z], aorta.sdf);
      const pre = glsl.left.classifyChambers!;
      if ((x0 - f.pose.swingX - bc.x) ** 2 + (y - bc.y) ** 2 + (z - bc.z) ** 2 > R * R) {
        outside++;
        if (pre.returned !== false || (pre.values.s as GlslSample).sdf !== aorta.sdf)
          problems.push(`${at}: outside the bounding sphere the GLSL does not stop`);
        continue;
      }
      inside++;
      if (pre.returned !== undefined) {
        problems.push(`${at}: inside the bounding sphere the GLSL stops`);
        continue;
      }
      // the CPU's context of the point as classify.ts sets it before its blocks (the swing, the column, the root's
      // coordinates); then each block on it in classify.ts's order, on the points the blocks before it leave on both
      // sides, against the GLSL run given what the runs before it left
      classifyChambers(f.heart, f.pose, x0, y, z, out);
      compare(at, 'classifyChambers', false, out, pre);
      compare(at, 'rootCoordinates', false, out, glsl.left.rootCoordinates!);
      if ((ctx.rootT as number) > -90) labelled.set('rootCoordinates along the root', (labelled.get('rootCoordinates along the root') ?? 0) + 1);
      ctx.aaOuter = aorta.sdf;
      for (const name of ['classifyLeftVentricle', 'classifyAorticRoot', 'classifyAtria']) {
        const run = glsl.left[name];
        if (!run) break;
        ctx.out = cpuOut[name]!;
        const hit = CPU_BLOCKS[name]!(ctx);
        compare(at, name, hit, cpuOut[name]!, run);
        if (hit || run.returned !== undefined) break;
      }
      // the right ventricle on every point, whichever block before it labels the point: its run reads only the point
      // and the azimuth, and so does rightVentricle.ts
      const rv = glsl.left.classifyRightVentricle ?? runs.classifyRightVentricle!({ ...pre.values, az: fastAtan2(pre.values.y as number, pre.values.x as number), s: {} });
      ctx.out = cpuOut.classifyRightVentricle!;
      compare(at, 'classifyRightVentricle', classifyRightVentricle(ctx), cpuOut.classifyRightVentricle!, rv);
      ctx.aaOuter = FAR_FROM_HEART_CM;
    }
  }
  t.diagnostic(
    `${inside} points inside the bounding spheres of ${frames().length} frames and ${outside} outside, largest ` +
      `deviation ${worst.toExponential(1)}; ${[...labelled].sort().map(([k, n]) => `${k}: ${n}`).join(', ')}`,
  );
  assert.deepEqual(problems.slice(0, 8), [], `${problems.length} differences between the GLSL classifier and the CPU's`);
  // the points reached the branches of each block
  for (const [kind, least] of MIN_LABELLED)
    assert.ok((labelled.get(kind) ?? 0) >= least, `${labelled.get(kind) ?? 0} points of ${kind}, fewer than ${least}`);
  assert.ok(outside >= 300, `${outside} points outside the bounding spheres`);
  assert.ok(swing > 0.2, `the largest swing is ${swing} cm: the point after the swing goes untested`);
});
/**
 * What the GLSL pericardium reads, by the run of the GPU's classifier that leaves it (glslClassifier): the GPU's own
 * wiring up to the sac, run in Node, but for the point before the swing and the aortic extension's distance, which
 * classifyHeart holds from its start. The block may read nothing else: a variable of classifyHeart it reads that is not
 * here (the RV block's dRvU, say, which classify/pericardium.ts never receives) is a value the CPU's pericardium is not
 * drawn from, and fails the test.
 */
const PERICARDIUM_INPUTS: Record<string, string> = {
  // the point before the tamponade swing, where the still sac is read, and the aortic extension's outer wall
  p0: 'classifyHeart',
  vascularDistance: 'classifyHeart',
  // the point after the swing, and the posterior column
  p: 'classifyChambers',
  x: 'classifyChambers',
  y: 'classifyChambers',
  z: 'classifyChambers',
  colDist: 'classifyChambers',
  // the aortic root's coordinates
  rootT: 'rootCoordinates',
  rootRr: 'rootCoordinates',
  rootR: 'rootCoordinates',
  // the left ventricle's epicardium and the normal of its cavity
  dEllR: 'classifyLeftVentricle',
  wallT: 'classifyLeftVentricle',
  crestLoss: 'classifyLeftVentricle',
  zHinge: 'classifyLeftVentricle',
  n0: 'classifyLeftVentricle',
  // the atria's outer faces, and the left atrium's centre and radii
  la: 'classifyAtria',
  lr: 'classifyAtria',
  laEpi: 'classifyAtria',
  raSleeve: 'classifyAtria',
  raBody: 'classifyAtria',
  raaEpi: 'classifyAtria',
  raColumn: 'classifyAtria',
  // the cavity with its outflow tract that the right ventricle's free wall is drawn around
  dCavRv: 'classifyRightVentricle',
};
/** The structures only classify/pericardium.ts labels. */
const SAC = new Map<number, string>([
  [Structure.EpicardialFat, 'epicardial fat'],
  [Structure.Pericardium, 'pericardium'],
  [Structure.PericardialEffusion, 'effusion'],
]);

test('the GLSL pericardium draws what classify/pericardium.ts draws from what the GPU classifier before it leaves', async (t) => {
  const block = glslMirrors(await shaderSources()).find((m) => m.kind === 'block' && m.tsName === 'classifyPericardium');
  assert.ok(block?.bindBlock, block?.reason ?? 'no mirror of classifyPericardium in classifyHeart');
  assert.deepEqual(
    block.inputs.filter((n) => n !== 's' && !(n in PERICARDIUM_INPUTS)),
    [],
    'variables of classifyHeart the GLSL pericardium reads, which the CPU pericardium does not receive',
  );
  const out = makeSample(),
    aorta = makeSample();
  const counts = new Map<string, number>();
  const problems: string[] = [];
  let swing = 0,
    worst = 0;
  for (const f of frames()) {
    const runs = await classifierRuns(f);
    const { boundCenter: bc, boundRadius: R } = f.heart;
    swing = Math.max(swing, Math.abs(f.pose.swingX));
    let seed = 7;
    const rnd = (): number => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296) * 2 - 1;
    // points of the bounding sphere the classifier reads after the swing, given before it; around each one the CPU's
    // pericardium labels or leaves within 1 cm of the sac or of the root's wall, three more within 3 mm, where the
    // terms of the sac meet (the root's wall, the oblique sinus, the free wall's epicardium)
    const points: [number, number, number, boolean][] = [];
    for (let k = 0; k < 4000; k++) points.push([bc.x + f.pose.swingX + R * rnd(), bc.y + R * rnd(), bc.z + R * rnd(), true]);
    // and around the ascending aorta where it leaves the sac, whose wall is there the nearest surface
    const { avCenter: c, avAxis: ax, avE1: e1, avE2: e2, ascR } = f.A;
    for (let k = 0; k < 400; k++) {
      const t = 1.5 + 1.5 * rnd(),
        r = ascR + 0.6 + 0.6 * rnd(),
        phi = Math.PI * rnd();
      const [u, v] = [r * Math.cos(phi), r * Math.sin(phi)];
      points.push([
        c.x + ax.x * t + e1.x * u + e2.x * v + f.pose.swingX,
        c.y + ax.y * t + e1.y * u + e2.y * v,
        c.z + ax.z * t + e1.z * u + e2.z * v,
        true,
      ]);
    }
    while (points.length) {
      const [x0, y, z, seedPoint] = points.pop()!;
      if ((x0 - f.pose.swingX - bc.x) ** 2 + (y - bc.y) ** 2 + (z - bc.z) ** 2 > R * R) continue;
      const hit = classifyHeart(f.heart, f.pose, x0, y, z, out);
      // the CPU's pericardium drew it, or the point fell through it
      if (hit && !SAC.has(out.structure)) continue;
      if (seedPoint && (hit || out.sdf < 1))
        for (let j = 0; j < 3; j++) points.push([x0 + 0.3 * rnd(), y + 0.3 * rnd(), z + 0.3 * rnd(), false]);
      aorta.sdf = FAR_FROM_HEART_CM;
      classifyAorticExtension(f.pose.aorta, x0, y, z, aorta);
      const at = `${f.label}, (${x0.toFixed(3)}, ${y.toFixed(3)}, ${z.toFixed(3)}) before the swing`;
      const glsl = glslClassifier(runs, [x0, y, z], aorta.sdf);
      if (glsl.stop) {
        problems.push(`${at}: the GLSL's ${glsl.stop} ${glsl.left[glsl.stop]!.returned ? 'labels it' : 'stops'} before the sac`);
        continue;
      }
      const held: Record<string, unknown> = { p0: [x0, y, z], vascularDistance: aorta.sdf };
      const inputs: Record<string, unknown> = { s: {} };
      for (const n of block.inputs) {
        if (n === 's') continue;
        const from = PERICARDIUM_INPUTS[n]!;
        const values = from === 'classifyHeart' ? held : glsl.left[from]!.values;
        assert.ok(n in values, `the GLSL pericardium reads ${n}, which ${from} does not leave`);
        inputs[n] = values[n];
      }
      const s = inputs.s as GlslSample;
      const glslHit = runs.classifyPericardium!(inputs).returned === true;
      const kind = hit ? SAC.get(out.structure)! : 'outside the sac';
      counts.set(kind, (counts.get(kind) ?? 0) + 1);
      if (glslHit !== hit) problems.push(`${at}: the CPU ${hit ? 'labels it' : 'leaves it'}, the GLSL ${glslHit ? 'labels it' : 'leaves it'}`);
      else if (hit) {
        const d = sampleDifference(out, s);
        if (d.problem) problems.push(`${at}: ${d.problem}`);
        else worst = Math.max(worst, d.deviation);
      } else {
        // the sample as setSample leaves it, on a miss its distance to outside the sac
        const d = mirrorDeviation(out.sdf, s.sdf);
        if (!(d <= MIRROR_TOLERANCE)) problems.push(`${at}: CPU sdf ${out.sdf}, GLSL ${s.sdf} (outside the sac)`);
        else worst = Math.max(worst, d);
      }
    }
  }
  t.diagnostic(
    `${[...counts.values()].reduce((a, b) => a + b, 0)} points over ${frames().length} frames ` +
      `(${[...counts].map(([k, n]) => `${k}: ${n}`).join(', ')}); largest deviation ${worst.toExponential(1)}`,
  );
  assert.deepEqual(problems.slice(0, 8), [], `${problems.length} points where the GLSL pericardium differs`);
  // the points reached every branch: fluid, the parietal and visceral layers, the fat, and outside the sac
  for (const [kind, least] of [['effusion', 1000], ['pericardium', 300], ['epicardial fat', 150], ['outside the sac', 10000]] as const)
    assert.ok((counts.get(kind) ?? 0) >= least, `${counts.get(kind) ?? 0} points of ${kind}, fewer than ${least}`);
  assert.ok(swing > 0.2, `the largest swing is ${swing} cm: the sac's reading before the swing goes untested`);
});

/**
 * The GPU's wiring between the runs that no run executes, held as written: the distance the aortic extension leaves
 * (FAR_FROM_HEART_CM where it misses), which the runs are given from the CPU's extension, and the azimuth, which the runs
 * after it are given as the CPU reads it (exactly on the GPU, with fastAtan2 on the CPU, 0.005 rad apart at most).
 */
const HELD_AS_WRITTEN = ['s.sdf = FAR_FROM_HEART_CM;', 'float vascularDistance = s.sdf;', 'float az = atan(y, x);'];
test('the GPU wiring no run executes is held as written', () => {
  assert.deepEqual(
    HELD_AS_WRITTEN.filter((line) => GLSL_HEART.split(line).length !== 2),
    [],
    'lines of classifyHeart held as written that changed or now repeat',
  );
});
test('CPU colorMap matches GLSL_PRESENT_FRAG defines, uColorPalette, and formula across all 4 palettes', () => {
  const defs = colorMapDefinesGlsl();
  for (const [k, v] of Object.entries(COLOR_MAP)) {
    const token = `#define CM_${k} ${Number.isInteger(v) ? `${v}.0` : `${v}`}`;
    assert.ok(defs.includes(token), `missing define ${token}`);
    assert.ok(GLSL_PRESENT_FRAG.includes(token), `GLSL_PRESENT_FRAG missing ${token}`);
  }
  assert.ok(GLSL_PRESENT_FRAG.includes(`#define COLOR_BLEND ${COLOR_BLEND}`));
  assert.ok(GLSL_PRESENT_FRAG.includes('uniform int uColorPalette;'), 'GLSL_PRESENT_FRAG missing uColorPalette');
  assert.ok(GLSL_PRESENT_FRAG.includes(`mix(${ST_POW_ALPHA_MIN}, 1.0`));
  for (let i = 0; i < COLOR_PALETTE_IDS.length; i++) {
    assert.ok(GLSL_PRESENT_FRAG.includes(`paletteId == ${i}`), `GLSL_PRESENT_FRAG missing palette branch ${i}`);
  }

  // Mirror of GLSL_PRESENT_FRAG colorMap() using the exact injected COLOR_PALETTES parameters
  const glslColorMapRef = (
    v: number,
    scale: number,
    variance: number,
    showVariance: boolean,
    baselineShift: number,
    palette: ColorPaletteId,
  ): [number, number, number] => {
    const M = COLOR_PALETTES[palette];
    const posSpan = Math.max(1e-3, scale + baselineShift);
    const negSpan = Math.max(1e-3, scale - baselineShift);
    const t = v >= 0 ? Math.min(1, v / posSpan) : -Math.min(1, -v / negSpan);
    const a = Math.abs(t);
    const dominant = M.DOMINANT_BASE + M.DOMINANT_RANGE * Math.min(1, a * M.DOMINANT_GAIN);
    const hotTip = Math.min(1, Math.max(0, (a - 0.72) / 0.28));
    let r: number;
    let g: number;
    let b: number;
    if (t >= 0) {
      r = dominant;
      g = Math.min(
        255,
        M.GREEN_BASE_TOWARD + M.GREEN_RANGE_TOWARD * Math.max(0, a - M.GREEN_KNEE) * M.GREEN_GAIN,
      );
      b = Math.min(255, M.QUIET + M.QUIET_BOOST_TOWARD * hotTip);
    } else {
      r = Math.min(255, M.QUIET + M.QUIET_BOOST_AWAY * hotTip);
      g = Math.min(
        255,
        M.GREEN_BASE_AWAY + M.GREEN_RANGE_AWAY * Math.max(0, a - M.GREEN_KNEE) * M.GREEN_GAIN,
      );
      b = dominant;
    }
    if (showVariance && variance > M.VARIANCE_MIN) {
      const m = Math.min(1, (variance - M.VARIANCE_MIN) * M.VARIANCE_GAIN);
      g = g * (1 - m) + M.VARIANCE_GREEN * m;
      r = r * (1 - m * M.VARIANCE_FADE);
      b = b * (1 - m * M.VARIANCE_FADE);
    }
    return [r, g, b];
  };

  const cpuOut: [number, number, number] = [0, 0, 0];
  let maxErr = 0;
  for (const palette of COLOR_PALETTE_IDS) {
    for (const baseline of [-0.25, 0, 0.3]) {
      const scale = 0.68;
      for (let iv = 0; iv <= 24; iv++) {
        const v = baseline - scale + (2 * scale * iv) / 24;
        for (let ivar = 0; ivar <= 6; ivar++) {
          const variance = ivar / 6;
          for (const showVar of [false, true]) {
            colorMap(v, scale, variance, showVar, cpuOut, baseline, palette);
            const glOut = glslColorMapRef(v, scale, variance, showVar, baseline, palette);
            for (let c = 0; c < 3; c++) {
              maxErr = Math.max(maxErr, Math.abs(cpuOut[c]! - glOut[c]!));
            }
          }
        }
      }
    }
  }
  assert.ok(maxErr < 1e-6, `CPU vs GLSL colorMap diverged across 4 palettes: maxErr=${maxErr}`);
});

test('Doppler GLSL passes stay synchronized with TypeScript physical constants and tissue write priority', () => {
  const defs = dopplerDefinesGlsl();
  const requiredDefines = [
    `#define DOPPLER_SHADOW_TRANSMISSION ${DOPPLER_SHADOW_TRANSMISSION}`,
    `#define COLOR_THRESHOLD ${COLOR_THRESHOLD}`,
    `#define COLOR_PACKET ${COLOR_PACKET}`,
    `#define CLUTTER_BLOOD_RATIO_POW ${CLUTTER_BLOOD_RATIO_POW}`,
    `#define CLUTTER_SIGMA_V_MPS ${CLUTTER_SIGMA_V_MPS}`,
    `#define CLUTTER_JITTER_RAD ${CLUTTER_JITTER_RAD}`,
    `#define NOISE_POWER ${NOISE_POWER}`,
    `#define TRANSIT_JITTER_RAD ${TRANSIT_JITTER_RAD}`,
    `#define NOISE_COHERENCE ${NOISE_COHERENCE}`,
    `#define NOISE_POWER_GATE ${NOISE_POWER_GATE}`,
    `#define ST_HYSTERESIS_RATIO ${ST_HYSTERESIS_RATIO}`,
    `#define ST_SPATIAL_LAMINAR_MIX ${ST_SPATIAL_LAMINAR_MIX}`,
    `#define ST_SPATIAL_TURB_MIX ${ST_SPATIAL_TURB_MIX}`,
  ];
  for (const def of requiredDefines) {
    assert.ok(defs.includes(def), `missing Doppler GLSL define: ${def}`);
  }

  // Tissue write priority bounds (98..172) and breakthrough blend structure must match CPU overlayColorField
  assert.ok(
    GLSL_PRESENT_FRAG.includes('smoothstep(98.0, 172.0, lum)'),
    'GLSL_PRESENT_FRAG tissue write priority bounds must match CPU overlayColorField (98..172)',
  );
  assert.ok(
    GLSL_PRESENT_FRAG.includes('mix(tissuePrio, 1.0, breakthrough)'),
    'GLSL_PRESENT_FRAG must relax tissue write priority via breakthrough matching CPU overlayColorField',
  );
  assert.ok(
    GLSL_DOPPLER_AUTOCORR_FRAG.includes('KASAI_LIN_BASIS') &&
      GLSL_DOPPLER_AUTOCORR_FRAG.includes('KASAI_QUAD_BASIS'),
    'Pass D1 must use shared 3-basis Legendre KASAI_LIN_BASIS and KASAI_QUAD_BASIS',
  );
  assert.ok(
    GLSL_DOPPLER_SHADE_FRAG.includes('ST_HYSTERESIS_RATIO'),
    'Pass D2 must use shared ST_HYSTERESIS_RATIO',
  );
  assert.ok(
    GLSL_DOPPLER_SPATIAL_FRAG.includes('ST_SPATIAL_LAMINAR_MIX') &&
      GLSL_DOPPLER_SPATIAL_FRAG.includes('ST_SPATIAL_TURB_MIX') &&
      GLSL_DOPPLER_SPATIAL_FRAG.includes('uniform float uGainDb;'),
    'Pass D3 must use shared ST_SPATIAL_* constants and uGainDb breakthrough encoding',
  );
});

test('Deferred Doppler GLSL uniform declarations match TypeScript uniform location tables exactly', () => {
  const extractUniforms = (glsl: string): string[] =>
    [...glsl.matchAll(/^\s*uniform\s+\w+\s+(\w+)\s*;/gm)].map((m) => m[1]!).sort();

  const pairs: Array<[string, string, readonly string[]]> = [
    ['Pass D1 (autocorr)', GLSL_DOPPLER_AUTOCORR_FRAG, DOPPLER_AUTOCORR_UNIFORMS],
    ['Pass D2 (shade)', GLSL_DOPPLER_SHADE_FRAG, DOPPLER_SHADE_UNIFORMS],
    ['Pass D3 (spatial)', GLSL_DOPPLER_SPATIAL_FRAG, DOPPLER_SPATIAL_UNIFORMS],
  ];

  for (const [label, glsl, expected] of pairs) {
    assert.deepEqual(
      extractUniforms(glsl),
      [...expected].sort(),
      `${label} GLSL uniforms diverged from deferredDoppler.ts uniform table`,
    );
  }
});

test('CPU breakthrough power encoding and GPU display-alpha encoding decode identically in Present pass', () => {
  const smoothstep = (e0: number, e1: number, x: number) => {
    const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
    return t * t * (3 - 2 * t);
  };
  const gpuEncodeDisplayAlpha = (powVal: number, isClutter: boolean, gainDb: number) => {
    const bt = isClutter
      ? smoothstep(0.25, 0.9, powVal)
      : smoothstep(4.0, 12.0, gainDb) * 0.45;
    if (bt > 0) return 1.0 + bt;
    return 0.5 + 0.5 * colorPowerConfidence(powVal, COLOR_THRESHOLD);
  };

  for (const isClutter of [false, true]) {
    for (const gainDb of [-4, 0, 4, 8, 12]) {
      for (const powVal of [0.25, 0.4, 0.65, 0.9, 1.0]) {
        const cpuEncoded = encodeColorBreakthroughPower(powVal, isClutter, gainDb);
        // Mirror glPresentUploads.ts CPU->GPU upload transform
        const uploadedAlpha =
          cpuEncoded > 1 ? cpuEncoded : 0.5 + 0.5 * colorPowerConfidence(cpuEncoded, COLOR_THRESHOLD);
        const gpuAlpha = gpuEncodeDisplayAlpha(powVal, isClutter, gainDb);
        assert.ok(
          Math.abs(uploadedAlpha - gpuAlpha) < 1e-6,
          `CPU vs GPU alpha mismatch (clutter=${isClutter}, gain=${gainDb}, pow=${powVal}): cpu=${uploadedAlpha} gpu=${gpuAlpha}`,
        );
        // Verify decoded breakthrough & powAlpha in overlayColorField vs GLSL_PRESENT_FRAG
        const cpuBt = Math.max(0, Math.min(1, cpuEncoded - 1));
        const gpuBt = Math.max(0, Math.min(1, gpuAlpha - 1));
        assert.ok(Math.abs(cpuBt - gpuBt) < 1e-6);
        const cpuPowAlpha =
          ST_POW_ALPHA_MIN + (1 - ST_POW_ALPHA_MIN) * colorPowerConfidence(cpuEncoded, COLOR_THRESHOLD);
        const gpuPowAlpha =
          ST_POW_ALPHA_MIN +
          (1 - ST_POW_ALPHA_MIN) * Math.max(0, Math.min(1, (gpuAlpha - 0.5) * 2));
        assert.ok(Math.abs(cpuPowAlpha - gpuPowAlpha) < 1e-6);
      }
    }
  }
});
