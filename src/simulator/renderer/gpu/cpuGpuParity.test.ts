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
import {
  glslMirrors,
  MIRROR_TOLERANCE,
  mirrorDeviation,
  mirrorSamples,
  type GlslScalar,
} from './shaderMirrors';
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

/**
 * CPU ↔ GPU parity guard suite.
 *
 * Catches silent divergence between CPU TypeScript physics/postprocess modules and their WebGL2
 * GLSL counterparts — including, numerically, every scalar function the shaders mirror by hand
 * (`glslGenerated.ts` and the other annotated mirrors, see `shaderMirrors.ts`).
 */

/**
 * GLSL mirrors evaluated in Node. Each shader function preceded by a line `// src/<path>.ts: <export>` is translated to
 * JavaScript with its program's constants (shaderMirrors.ts) and compared with that export on MIRROR_SAMPLES arguments,
 * within MIRROR_TOLERANCE (float32 resolution). Measured when written: 54 mirrors, 50 equal to the last bit on every
 * sample and four within 1.4e-10 of max(1, |y|), all through the shader's PI printed with 15 digits (rvAzProfile, where
 * pow(x, 0.7) magnifies the last bits of sin(PI·u) near 0). Of 705 mutants of their GLSL (each literal and constant
 * ×1.01, each binary ± flipped) 682 fail this test, against 17 before it, all septalShare's, in lvWall.test.ts; the
 * other 23 change nothing (rounding guards, clamps above a plateau, early returns where the ramps are already 0) or act
 * only within sub-millimetre bands.
 *
 * Mirrors outside the scalar subset stay with the whole-image comparison in the browser
 * (window.__echotwin.compareBackends, npm run test:gpu):
 * - vec3 arguments: sdEllipsoid, sdCapsule, sdRoundCone and sdTorusZ (glslCommon.ts), which sdf.ts takes as triples;
 * - the per-frame parameter texture P(i) and its #defines, where the TypeScript reads a model object: rootRadiusAt
 *   (aorticValve.ts), mvHingeHeight and mvInflowTaper (mitralValve.ts hingeHeight, inflowTaper), lvProfileG and
 *   lvCavityRadius (lvShape.ts), ahaSegment (lvGeometry.ts), skinZ, leftLungBorderX, ribCenterY (thoraxModel.ts);
 *   pairing them needs the scene packed by paramLayout.ts;
 * - the noise texture: wallThicknessAt (lvWall.ts; lvWall.test.ts checks its septal share and modulation gain);
 * - out/inout parameters and structs: the classifier blocks (skirtDistance, mitralDistance, classifyHeart…).
 * The Doppler programs hold scalar mirrors not annotated yet: wallFilterResponse and colorPowerConf (colorPowerConfidence
 * at COLOR_THRESHOLD) agree with their TypeScript; aliasVelocity agrees with clinical/formulas only for a positive
 * Nyquist velocity (for a non-positive one the TypeScript returns v and the shader NaN), and takes its baseline as a
 * third parameter where the TypeScript defaults it.
 */
const MIRROR_SAMPLES = 4096;

test('GLSL mirrors of TypeScript scalar functions evaluate as their originals on sampled inputs', async (t) => {
  // every GLSL string the shader modules export: a mirror annotated in any of them enters by itself
  const here = import.meta.dirname;
  const sources: string[] = [];
  for (const file of fs.readdirSync(here).filter((f) => /^glsl\w*\.ts$/.test(f)).sort()) {
    const mod = (await import(pathToFileURL(path.join(here, file)).href)) as Record<string, unknown>;
    for (const v of Object.values(mod)) if (typeof v === 'string' && !sources.includes(v)) sources.push(v);
  }
  const mirrors = glslMirrors(sources);
  // every function of glslGenerated.ts is one of them: a definition without its `// src/….ts: name` line is unchecked
  const generated = [...GLSL_GENERATED.matchAll(/^\w+\s+(\w+)\s*\(/gm)].map((m) => m[1]!);
  const unannotated = generated.filter((name) => !mirrors.some((m) => m.glslName === name));
  assert.ok(generated.length > 0, 'no function definition found in GLSL_GENERATED');
  assert.deepEqual(unannotated, [], 'GLSL_GENERATED functions without the annotation naming their TypeScript');

  const problems: string[] = [];
  let worst = { deviation: 0, label: '' };
  for (const m of mirrors) {
    const label = `${m.glslName} (GLSL) ↔ ${m.tsPath}: ${m.tsName}`;
    const file = path.resolve(here, '../../../..', m.tsPath);
    const mod = fs.existsSync(file) ? ((await import(pathToFileURL(file).href)) as Record<string, unknown>) : {};
    const original = mod[m.tsName];
    if (typeof original !== 'function') problems.push(`${label}: no such exported function`);
    else if (original.length !== m.paramTypes.length)
      problems.push(`${label}: ${original.length} parameters in TypeScript, ${m.paramTypes.length} in GLSL`);
    else if (!m.glsl) problems.push(`${label}: not translated, ${m.reason}`);
    else {
      let largest = { deviation: 0, at: [] as GlslScalar[], ts: 0 as GlslScalar, glsl: 0 as GlslScalar };
      for (const x of mirrorSamples(m, MIRROR_SAMPLES)) {
        let ts: GlslScalar, glsl: GlslScalar;
        try {
          ts = (original as (...a: GlslScalar[]) => GlslScalar)(...x);
          glsl = m.glsl(...x);
        } catch (e) {
          problems.push(`${label}: throws at (${x.join(', ')}): ${(e as Error).message}`);
          break;
        }
        const deviation = mirrorDeviation(ts, glsl);
        if (deviation > largest.deviation) largest = { deviation, at: x, ts, glsl };
      }
      if (largest.deviation > MIRROR_TOLERANCE)
        problems.push(`${label}: at (${largest.at.join(', ')}) TypeScript ${largest.ts}, GLSL ${largest.glsl}`);
      else if (largest.deviation > worst.deviation) worst = { deviation: largest.deviation, label };
    }
  }
  t.diagnostic(
    `${mirrors.length} GLSL mirrors × ${MIRROR_SAMPLES} samples; largest deviation within tolerance ` +
      `${worst.deviation.toExponential(1)} of max(1, |y|) (${worst.label})`,
  );
  assert.deepEqual(problems, [], `GLSL mirrors that diverge from their TypeScript:\n${problems.join('\n')}`);
});

test('the mirror translator evaluates scalar GLSL as written and names what it cannot translate', () => {
  const [scalar, vector, outParam, frameParam, fetched] = glslMirrors([
    `
const float K = 0.25;
#define FRAME_GAIN P(3)
// src/a.ts: scalar
float scalar(float x, int n, bool b) {
  float y = b ? x : -x;
  for (int j = 0; j < 2; j++) y += K * float(n + j);
  return smoothstep(0.0, 1.0, y);
}
// src/a.ts: vector
float vector(float x) { vec2 v = vec2(x, 1.0); return v.x; }
// src/a.ts: outParam
float outParam(float x, out float y) { y = x; return x; }
// src/a.ts: frameParam
float frameParam(float x) { return x * FRAME_GAIN; }
// src/a.ts: fetched
float fetched(float x) { return texelFetch(x); }
`,
  ]);
  assert.equal(scalar!.glsl!(-1, 2, true), 0.15625); // y = −1 + 0.25·2 + 0.25·3 = 0.25 → 0.25²·(3 − 0.5)
  assert.equal(scalar!.glsl!(-1, 2, false), 1);
  assert.deepEqual(scalar!.paramTypes, ['float', 'int', 'bool']);
  const sample = mirrorSamples(scalar!, 64);
  assert.ok(sample.every(([, n, b]) => Number.isInteger(n) && typeof b === 'boolean'));
  assert.match(vector!.reason!, /vectors/);
  assert.match(outParam!.reason!, /out float y/);
  assert.match(frameParam!.reason!, /FRAME_GAIN, P\(3\)/);
  assert.match(fetched!.reason!, /texelFetch is neither a scalar GLSL built-in/);
  assert.equal(mirrorDeviation(1, 1 + 2 ** -21), 2 ** -21);
  assert.equal(mirrorDeviation(NaN, NaN), 0);
  assert.equal(mirrorDeviation(1, NaN), Infinity);
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
