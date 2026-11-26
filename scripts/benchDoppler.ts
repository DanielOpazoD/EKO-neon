/**
 * Reproducible CPU microbenchmark for all 5 Doppler hot paths (Color 2D, CMM, PW, CW, TDI).
 *
 * Usage:
 *   npm run bench:doppler
 *   npx tsx scripts/benchDoppler.ts --check
 */
import { performance } from 'node:perf_hooks';
import { loadCaseById } from '../src/cases/index';
import { buildCaseModels, REST_PATIENT } from '../src/simulator/anatomy/caseModels';
import { computeHeartPose } from '../src/simulator/anatomy/heartModel';
import { makeSample, Tissue } from '../src/simulator/anatomy/tissue';
import { cycleStateAt } from '../src/simulator/cardiac-cycle/cycleModel';
import { sampleSpectralColumn } from '../src/simulator/core/strip/spectralSampler';
import { paintMmodeStripColumn } from '../src/simulator/core/strip/stripPainter';
import type { StripCtx } from '../src/simulator/core/stripEngine';
import {
  allocColorField,
  computeColorField,
  DEFAULT_COLOR,
  overlayColorField,
} from '../src/simulator/doppler/color/colorDoppler';
import {
  buildFlowParams,
  sampleFlow,
  type FlowSample,
} from '../src/simulator/doppler/flow-primitives/flowField';
import { DEFAULT_SPECTRAL } from '../src/simulator/doppler/spectral/spectrum';
import { beamFrameFromPose, poseFromControl } from '../src/simulator/probe/pose';
import { softTissueTransmission } from '../src/simulator/renderer/acoustic/acoustics';
import { computeDopplerBoxBounds } from '../src/simulator/renderer/gpu/deferredDoppler';
import { buildScanLut, computeSectorMapping } from '../src/simulator/renderer/scanConvert';
import {
  allocPolarFrame,
  DEFAULT_ACQUISITION,
  type PolarFrameSpec,
} from '../src/simulator/renderer/types';
import { canonicalControl, VIEW_TARGETS } from '../src/simulator/windows/viewTargets';

const checkMode = process.argv.includes('--check');

const c = loadCaseById('normal-excellent-window');
const { heart, thorax, tables } = buildCaseModels(c, REST_PATIENT);
const flow = buildFlowParams(c, heart, tables, thorax);
const vA4C = VIEW_TARGETS.find((v) => v.id === 'a4c')!;
const vA5C = VIEW_TARGETS.find((v) => v.id === 'a5c')!;
const ctrlA4C = canonicalControl(vA4C, heart, thorax);
const ctrlA5C = canonicalControl(vA5C, heart, thorax);
const beamA4C = beamFrameFromPose(poseFromControl(thorax, ctrlA4C), 1);
const beamA5C = beamFrameFromPose(poseFromControl(thorax, ctrlA5C), 1);

const spec: PolarFrameSpec = {
  lines: 112,
  samples: 256,
  sectorRad: Math.PI / 2,
  depthCm: 16,
  focusCm: 8,
  elevationSamples: 1,
};
const frame = allocPolarFrame(spec);
frame.tissue.fill(Tissue.Blood);
for (let li = 0; li < spec.lines; li++) {
  for (let si = 0; si < spec.samples; si++) {
    frame.transmission[li * spec.samples + si] = softTissueTransmission(
      ((si + 0.5) / spec.samples) * spec.depthCm,
      2.5,
      true,
    );
  }
}

const mapping = computeSectorMapping(spec, 640, 480, false, 1);
const lut = buildScanLut(spec, mapping);
const rgba = new Uint8ClampedArray(640 * 480 * 4);
const field = allocColorField(spec.lines * spec.samples);
const fs: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };
const hf = heart.frame;
const dr = spec.depthCm / spec.samples;

// 1. Color Doppler CPU fallback (computeColorField + overlayColorField)
let sumComp = 0;
let sumOver = 0;
const nColor = 15;
for (let i = -2; i < nColor; i++) {
  const phase = (Math.max(0, i) / nColor) % 1;
  const hp = computeHeartPose(heart, cycleStateAt(tables, phase, flow.flowPeaks));
  const ox = beamA5C.origin.x - hf.origin.x;
  const oy = beamA5C.origin.y - hf.origin.y;
  const oz = beamA5C.origin.z - hf.origin.z;
  const ohx = ox * hf.ex.x + oy * hf.ex.y + oz * hf.ex.z;
  const ohy = ox * hf.ey.x + oy * hf.ey.y + oz * hf.ey.z;
  const ohz = ox * hf.ez.x + oy * hf.ez.y + oz * hf.ez.z;

  const t0 = performance.now();
  computeColorField(
    frame,
    DEFAULT_COLOR,
    null,
    (_idx, li, si, o) => {
      const theta = -spec.sectorRad / 2 + (spec.sectorRad * (li + 0.5)) / spec.lines;
      const ct = Math.cos(theta);
      const sn = Math.sin(theta);
      const dx = beamA5C.forward.x * ct + beamA5C.lateral.x * sn;
      const dy = beamA5C.forward.y * ct + beamA5C.lateral.y * sn;
      const dz = beamA5C.forward.z * ct + beamA5C.lateral.z * sn;
      const dhx = dx * hf.ex.x + dy * hf.ex.y + dz * hf.ex.z;
      const dhy = dx * hf.ey.x + dy * hf.ey.y + dz * hf.ey.z;
      const dhz = dx * hf.ez.x + dy * hf.ez.y + dz * hf.ez.z;
      const r = (si + 0.5) * dr;
      sampleFlow(flow, tables, hp, phase, ohx + dhx * r, ohy + dhy * r, ohz + dhz * r, fs);
      o.v = -(fs.vx * dhx + fs.vy * dhy + fs.vz * dhz);
      o.disp = fs.dispersion;
      o.present = fs.present;
    },
    field,
    { frequencyMHz: 2.5, harmonics: true },
    { realization: Math.max(0, i), seed: 42 },
  );
  const t1 = performance.now();
  overlayColorField(rgba, lut, field.vel, field.variance, DEFAULT_COLOR, field.power);
  const t2 = performance.now();
  if (i >= 0) {
    sumComp += t1 - t0;
    sumOver += t2 - t1;
  }
}

// 2. Color Doppler GPU-deferred CPU-side ROI sampling
const box = computeDopplerBoxBounds(spec, DEFAULT_COLOR);
const rawFlow = new Float32Array(box.boxW * box.boxH * 4);
let sumRaw = 0;
for (let i = -2; i < nColor; i++) {
  const phase = (Math.max(0, i) / nColor) % 1;
  const hp = computeHeartPose(heart, cycleStateAt(tables, phase, flow.flowPeaks));
  const ox = beamA5C.origin.x - hf.origin.x;
  const oy = beamA5C.origin.y - hf.origin.y;
  const oz = beamA5C.origin.z - hf.origin.z;
  const ohx = ox * hf.ex.x + oy * hf.ex.y + oz * hf.ex.z;
  const ohy = ox * hf.ey.x + oy * hf.ey.y + oz * hf.ey.z;
  const ohz = ox * hf.ez.x + oy * hf.ez.y + oz * hf.ez.z;
  const t0 = performance.now();
  for (let li = box.liMin; li <= box.liMax; li++) {
    const theta = -spec.sectorRad / 2 + (spec.sectorRad * (li + 0.5)) / spec.lines;
    const ct = Math.cos(theta);
    const sn = Math.sin(theta);
    const dx = beamA5C.forward.x * ct + beamA5C.lateral.x * sn;
    const dy = beamA5C.forward.y * ct + beamA5C.lateral.y * sn;
    const dz = beamA5C.forward.z * ct + beamA5C.lateral.z * sn;
    const dhx = dx * hf.ex.x + dy * hf.ex.y + dz * hf.ex.z;
    const dhy = dx * hf.ey.x + dy * hf.ey.y + dz * hf.ey.z;
    const dhz = dx * hf.ez.x + dy * hf.ez.y + dz * hf.ez.z;
    const rowOff = (li - box.liMin) * box.boxW * 4;
    for (let si = box.siMin; si <= box.siMax; si++) {
      const r = (si + 0.5) * dr;
      sampleFlow(flow, tables, hp, phase, ohx + dhx * r, ohy + dhy * r, ohz + dhz * r, fs);
      const o = rowOff + (si - box.siMin) * 4;
      rawFlow[o] = -(fs.vx * dhx + fs.vy * dhy + fs.vz * dhz);
      rawFlow[o + 1] = fs.dispersion;
      rawFlow[o + 2] = fs.present ? 1 : 0;
      rawFlow[o + 3] = 0;
    }
  }
  const t1 = performance.now();
  if (i >= 0) sumRaw += t1 - t0;
}

// 3. Spectral columns (PW, CW, TDI)
const ts = makeSample();
const physics = {
  frequencyMHz: 2.5,
  harmonics: true,
  depthCm: 16,
  gainDb: 0,
  dynamicRangeDb: 60,
  tgcDb: [0, 0, 0, 0, 0, 0, 0, 0],
  focusCm: 8,
  sectorDeg: 80,
  lineDensity: 'medium' as const,
  persistence: 0.3,
  edgeEnhance: 0,
  grayMap: 'clinical' as const,
  invertLR: false,
  zoom: 1,
};

const specMeans: Record<'pw' | 'cw' | 'tdi', number> = { pw: 0, cw: 0, tdi: 0 };
for (const mode of ['pw', 'cw', 'tdi'] as const) {
  const beam = mode === 'cw' ? beamA5C : beamA4C;
  const gateDepthCm = mode === 'tdi' ? 10.5 : 8.5;
  const spectral = {
    ...DEFAULT_SPECTRAL,
    scaleMps: mode === 'cw' ? 6.0 : mode === 'tdi' ? 0.2 : 1.0,
  };
  const sctx = {
    input: {
      probe: mode === 'cw' ? ctrlA5C : ctrlA4C,
      patient: REST_PATIENT,
      settings: DEFAULT_ACQUISITION,
      modality: mode,
      frozen: false,
      cineOffset: 0,
      color: DEFAULT_COLOR,
      spectral,
      cursorThetaRad: 0,
      gateDepthCm,
      quality: 'medium' as const,
      display: { width: 640, height: 480 },
      rendererBackend: 'procedural' as const,
      artifactOverrides: null,
    },
    caseDef: c,
    heart,
    tables,
    flow,
    timeS: 0,
    modelVersion: 1,
    tablesVersion: 1,
    phaseNow: 0,
    rrS: tables.rrS,
    scene: (p: number) => ({
      heart,
      thorax,
      heartPose: computeHeartPose(heart, cycleStateAt(tables, p, flow.flowPeaks)),
      physics,
    }),
    moment: (t: number) => {
      const p = (t % tables.rrS) / tables.rrS;
      return {
        phase: p,
        beatIndex: 0,
        tablesVersion: 1,
        tables,
        flow,
        scene: {
          heart,
          thorax,
          heartPose: computeHeartPose(heart, cycleStateAt(tables, p, flow.flowPeaks)),
          physics,
        },
      };
    },
    physics: () => physics,
  } as unknown as StripCtx;

  sampleSpectralColumn(beam, spec, 0.25, 0, 0, 0.25, sctx, fs, ts);
  const nCols = 8;
  const t0 = performance.now();
  for (let i = 0; i < nCols; i++) {
    const t = i * 0.011;
    sampleSpectralColumn(beam, spec, (t % tables.rrS) / tables.rrS, i, i, t, sctx, fs, ts);
  }
  specMeans[mode] = (performance.now() - t0) / nCols;
}

// 4. CMM column paint (3 cols / frame)
const mmAmp = new Uint8ClampedArray(800).fill(90);
const mmVel = new Float32Array(256).fill(0.45);
const mmVar = new Float32Array(256).fill(0.05);
const mmPow = new Float32Array(256).fill(1.0);
const mmRgba = new Uint8ClampedArray(640 * 278 * 4);
const rowMap = {
  rows: 278,
  mmodeIdx: new Int32Array(278),
  colorIdx: new Int32Array(278),
};
for (let y = 0; y < 278; y++) {
  rowMap.mmodeIdx[y] = Math.floor((y / 278) * 800);
  rowMap.colorIdx[y] = Math.floor((y / 278) * 256);
}
const c0 = performance.now();
for (let i = 0; i < 60; i++) {
  paintMmodeStripColumn(
    mmRgba,
    640,
    0,
    i % 640,
    rowMap,
    mmAmp,
    mmVel,
    mmVar,
    mmPow,
    DEFAULT_COLOR,
    'gray',
  );
}
const cmmPaint3ColsMs = (performance.now() - c0) / 20;

const metrics = {
  colorComputeMs: +(sumComp / nColor).toFixed(2),
  colorOverlayMs: +(sumOver / nColor).toFixed(2),
  colorRawFlowBoxMs: +(sumRaw / nColor).toFixed(2),
  pwColMs: +specMeans.pw.toFixed(2),
  cwColMs: +specMeans.cw.toFixed(2),
  tdiColMs: +specMeans.tdi.toFixed(2),
  cmmPaint3ColsMs: +cmmPaint3ColsMs.toFixed(3),
};

console.log('=== DOPPLER CPU HOT-PATH BENCHMARK (normal-excellent-window, 640x480) ===');
console.table(metrics);

if (checkMode) {
  // Generous container-safe regression ceilings (~1.5x baseline to avoid noisy false alarms while catching real blowups)
  const ceilings: Record<keyof typeof metrics, number> = {
    colorComputeMs: 85,
    colorOverlayMs: 14,
    colorRawFlowBoxMs: 55,
    pwColMs: 25,
    cwColMs: 45,
    tdiColMs: 6,
    cmmPaint3ColsMs: 2,
  };
  const violations: string[] = [];
  for (const [k, limit] of Object.entries(ceilings)) {
    const val = metrics[k as keyof typeof metrics];
    if (val > limit) violations.push(`${k}: ${val}ms > ceiling ${limit}ms`);
  }
  if (violations.length > 0) {
    console.error('BENCHMARK REGRESSION:', violations.join('; '));
    process.exit(1);
  }
  console.log('BENCHMARK CHECK PASSED');
}
