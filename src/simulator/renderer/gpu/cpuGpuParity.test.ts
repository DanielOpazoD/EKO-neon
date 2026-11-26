import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
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
 * GLSL counterparts — including every scalar function manually retained in `glslGenerated.ts`.
 */

test('GLSL_GENERATED audit: every transpiled function exists in its declared TypeScript source file', () => {
  const entries = [...GLSL_GENERATED.matchAll(/^\/\/ (src\/[^\s:]+\.ts): ([A-Za-z0-9_]+)$/gm)];
  assert.ok(
    entries.length >= 40,
    `expected >= 40 annotated scalar functions in GLSL_GENERATED, found ${entries.length}`,
  );

  const fileCache = new Map<string, string>();
  const missing: string[] = [];
  const missingGlslDef: string[] = [];

  const arityMismatches: string[] = [];
  const countParams = (paramList: string): number => {
    const trimmed = paramList.trim();
    if (!trimmed || trimmed === 'void') return 0;
    let depth = 0;
    let count = 1;
    for (let i = 0; i < trimmed.length; i++) {
      const ch = trimmed[i]!;
      if (ch === '(' || ch === '<' || ch === '{' || ch === '[') depth++;
      else if (ch === ')' || ch === '>' || ch === '}' || ch === ']') depth--;
      else if (ch === ',' && depth === 0) count++;
    }
    // Ignore trailing comma in multi-line TS parameter lists
    if (trimmed.endsWith(',')) count--;
    return count;
  };

  for (const m of entries) {
    const relPath = m[1]!;
    const fnName = m[2]!;
    let src = fileCache.get(relPath);
    if (src === undefined) {
      const absPath = path.resolve(process.cwd(), relPath);
      assert.ok(fs.existsSync(absPath), `GLSL_GENERATED references missing TS file: ${relPath}`);
      src = fs.readFileSync(absPath, 'utf8');
      fileCache.set(relPath, src);
    }
    const tsDeclRe = new RegExp(`\\bfunction\\s+${fnName}\\s*\\(([\\s\\S]*?)\\)`);
    const tsConstRe = new RegExp(`\\bconst\\s+${fnName}\\s*=`);
    const tsMatch = src.match(tsDeclRe);
    if (!tsMatch && !tsConstRe.test(src)) {
      missing.push(`${relPath}:${fnName}`);
    }
    const glslDeclRe = new RegExp(`\\b(?:float|int|vec[234]|bool)\\s+${fnName}\\s*\\(([^)]*)\\)`);
    const glslMatch = GLSL_GENERATED.match(glslDeclRe);
    if (!glslMatch) {
      missingGlslDef.push(fnName);
    } else if (tsMatch) {
      const tsArity = countParams(tsMatch[1]!);
      const glslArity = countParams(glslMatch[1]!);
      if (tsArity !== glslArity) {
        arityMismatches.push(`${relPath}:${fnName} (TS=${tsArity}, GLSL=${glslArity})`);
      }
    }
  }

  assert.deepEqual(
    missing,
    [],
    `TypeScript functions referenced by GLSL_GENERATED no longer exist: ${missing.join(', ')}`,
  );
  assert.deepEqual(
    missingGlslDef,
    [],
    `GLSL definitions missing for annotated entries: ${missingGlslDef.join(', ')}`,
  );
  assert.deepEqual(
    arityMismatches,
    [],
    `Parameter arity drift between TypeScript and GLSL_GENERATED: ${arityMismatches.join(', ')}`,
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
