/**
 * Reproducible CPU microbenchmark for all 5 Doppler hot paths (Color 2D, CMM, PW, CW, TDI).
 *
 * Usage:
 *   npm run bench:doppler
 *   npx tsx scripts/benchDoppler.ts --check
 */
import { performance } from 'node:perf_hooks';
import { loadCaseById } from '../src/cases/index';
import { aseSweepDurationS } from '../src/clinical/formulas';
import { buildCaseModels, REST_PATIENT } from '../src/simulator/anatomy/caseModels';
import { computeHeartPose } from '../src/simulator/anatomy/heartModel';
import { makeSample, Tissue } from '../src/simulator/anatomy/tissue';
import { cycleStateAt } from '../src/simulator/cardiac-cycle/cycleModel';
import { buildRowMap, mmodeLineSamples, type MmodeLine } from '../src/simulator/core/mmodeStrip';
import { STRIP_MM_WIDTH } from '../src/simulator/core/strip/autoTrace';
import { writeCmmColumn } from '../src/simulator/core/strip/cmmColumn';
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
  WallFilterResponseTable,
  wallFilterPowerResponse,
  wallFilterShares,
} from '../src/simulator/doppler/color/kasaiEstimator';
import {
  buildFlowParams,
  sampleFlow,
  type FlowSample,
} from '../src/simulator/doppler/flow-primitives/flowField';
import { DEFAULT_SPECTRAL } from '../src/simulator/doppler/spectral/spectrum';
import { beamFrameFromPose, poseFromControl } from '../src/simulator/probe/pose';
import { softTissueTransmission } from '../src/simulator/renderer/acoustic/acoustics';
import { computeDopplerBoxBounds } from '../src/simulator/renderer/gpu/deferredDoppler';
import { pulsedAcquisition } from '../src/simulator/renderer/pulseTiming';
import { buildScanLut, computeSectorMapping } from '../src/simulator/renderer/scanConvert';
import {
  allocPolarFrame,
  DEFAULT_ACQUISITION,
  polarSpecFor,
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

// 4. Colour M-mode through the mitral inflow: the A4C cursor at 0 and the high tier's 320 samples over 16 cm, at the
// fastest sweep on a wide display (1600 columns at 100 mm/s: 27 a 60 Hz frame). The lines are the flow's axial velocity
// along the cursor at 128 phases of the beat, blood where the flow is (as the strip's cmmVelocity samples it): 118
// blood samples a line on median (0 to 259), as the app's strip has. Written as the strip writes them (writeCmmColumn:
// the wall filter's response, aliasing, variance and power), then painted 3 columns at a time into a strip of 278 rows
// (paintMmodeStripColumn). The paint case used to call paintMmodeStripColumn in an order its signature no longer had,
// and it returned before painting: both cases now check that they wrote and painted colour.
// The write is timed as the best of 7 runs of 180 frames, each beside a run of the reference: the wall filter's exact
// response alone (16 sin/cos a sample) for the blood samples of the same columns, which the strip computed per sample
// before the table. The best run is the one the scheduler, a slower core or another load disturbed least; and against
// a reference timed in the same process the table's gain holds where a ceiling in ms did not (0.12 ms, 2.5 times one
// run on a performance core, failed on efficiency cores and beside a test run)
const cmmSpec = polarSpecFor(DEFAULT_ACQUISITION, 'high');
const cmmCols = 1600;
const cmmPerFrame = Math.round(cmmCols / aseSweepDurationS(100, STRIP_MM_WIDTH) / 60);
const cmmColor = {
  ...DEFAULT_COLOR,
  scaleMps: pulsedAcquisition(cmmSpec.depthCm, DEFAULT_ACQUISITION.frequencyMHz, DEFAULT_COLOR.scaleMps)
    .scaleMps,
};
const cmmLines: MmodeLine[] = [];
{
  const d = beamA4C.forward;
  const dhx = d.x * hf.ex.x + d.y * hf.ex.y + d.z * hf.ex.z;
  const dhy = d.x * hf.ey.x + d.y * hf.ey.y + d.z * hf.ey.z;
  const dhz = d.x * hf.ez.x + d.y * hf.ez.y + d.z * hf.ez.z;
  const ox = beamA4C.origin.x - hf.origin.x;
  const oy = beamA4C.origin.y - hf.origin.y;
  const oz = beamA4C.origin.z - hf.origin.z;
  const ohx = ox * hf.ex.x + oy * hf.ex.y + oz * hf.ex.z;
  const ohy = ox * hf.ey.x + oy * hf.ey.y + oz * hf.ey.z;
  const ohz = ox * hf.ez.x + oy * hf.ez.y + oz * hf.ez.z;
  const cmmDr = cmmSpec.depthCm / cmmSpec.samples;
  for (let k = 0; k < 128; k++) {
    const phase = k / 128;
    const hp = computeHeartPose(heart, cycleStateAt(tables, phase, flow.flowPeaks));
    const velocity = new Float32Array(cmmSpec.samples).fill(NaN);
    const dispersion = new Float32Array(cmmSpec.samples);
    for (let si = 0; si < cmmSpec.samples; si++) {
      const r = (si + 0.5) * cmmDr;
      sampleFlow(flow, tables, hp, phase, ohx + dhx * r, ohy + dhy * r, ohz + dhz * r, fs);
      if (!fs.present) continue;
      velocity[si] = -(fs.vx * dhx + fs.vy * dhy + fs.vz * dhz);
      dispersion[si] = Math.min(1, fs.dispersion * 1.6);
    }
    const relTrans = new Float32Array(cmmSpec.samples).fill(1);
    cmmLines.push({ amp: new Float32Array(0), velocity, dispersion, relTrans });
  }
}
const cmmStrip = new Float32Array(cmmSpec.samples * cmmCols).fill(NaN);
const cmmVar = new Float32Array(cmmSpec.samples * cmmCols);
const cmmPow = new Float32Array(cmmSpec.samples * cmmCols);
const cmmWallFilter = new WallFilterResponseTable();
const writeCmmColumns = (first: number, n: number) => {
  for (let c = first; c < first + n; c++) {
    const col = c % cmmCols;
    writeCmmColumn(
      cmmLines[col % cmmLines.length]!,
      cmmSpec.samples,
      cmmColor,
      cmmWallFilter,
      col,
      cmmCols,
      cmmStrip,
      cmmVar,
      cmmPow,
    );
  }
};
const cmmShares = wallFilterShares(cmmColor.wallFilterMps, cmmColor.scaleMps);
let cmmExactPower = 0;
const exactResponseColumns = (first: number, n: number) => {
  for (let c = first; c < first + n; c++) {
    const v = cmmLines[(c % cmmCols) % cmmLines.length]!.velocity!;
    for (let si = 0; si < cmmSpec.samples; si++) {
      const x = v[si]!;
      if (x === x) cmmExactPower += wallFilterPowerResponse(x, cmmColor.scaleMps, cmmShares);
    }
  }
};
/** Mean ms a frame of `cmmPerFrame` columns, over `frames` frames. */
const cmmFrameMs = (run: (first: number, n: number) => void, frames: number) => {
  const t0 = performance.now();
  for (let f = 0; f < frames; f++) run(f * cmmPerFrame, cmmPerFrame);
  return (performance.now() - t0) / frames;
};
writeCmmColumns(0, cmmCols);
exactResponseColumns(0, 30 * cmmPerFrame);
let cmmWrite27ColsMs = Infinity;
let cmmExact27ColsMs = Infinity;
for (let rep = 0; rep < 7; rep++) {
  cmmWrite27ColsMs = Math.min(cmmWrite27ColsMs, cmmFrameMs(writeCmmColumns, 180));
  cmmExact27ColsMs = Math.min(cmmExact27ColsMs, cmmFrameMs(exactResponseColumns, 60));
}
// colour where the lines have blood: every column has been written
let cmmBlood = 0;
let cmmBloodColoured = 0;
for (let col = 0; col < cmmCols; col++) {
  const v = cmmLines[col % cmmLines.length]!.velocity!;
  for (let si = 0; si < cmmSpec.samples; si++) {
    if (!Number.isFinite(v[si]!)) continue;
    cmmBlood++;
    if (Number.isFinite(cmmStrip[si * cmmCols + col]!)) cmmBloodColoured++;
  }
}

const mmSamples = mmodeLineSamples(cmmSpec.depthCm);
const mmRows = 278;
const mmGrey = new Uint8ClampedArray(mmSamples * cmmCols).fill(90);
const mmRowMap = buildRowMap(mmRows, mmSamples);
const mmRgba = new Uint8ClampedArray(cmmCols * mmRows * 4);
const paintCmmColumn = (col: number) =>
  paintMmodeStripColumn(
    col,
    mmRgba,
    mmRowMap,
    mmGrey,
    cmmCols,
    mmSamples,
    'gray',
    cmmStrip,
    cmmColor,
    cmmVar,
    cmmPow,
  );
for (let col = 0; col < 60; col++) paintCmmColumn(col);
const c0 = performance.now();
for (let i = 0; i < 600; i++) paintCmmColumn(i % cmmCols);
const cmmPaint3ColsMs = (performance.now() - c0) / 200;
// colour in the painted columns: a pixel whose channels differ (the grey strip has them equal)
let cmmPaintedPx = 0;
let cmmColouredPx = 0;
for (let col = 0; col < 600; col++)
  for (let y = 0; y < mmRows; y++) {
    const o = (y * cmmCols + col) * 4;
    cmmPaintedPx++;
    if (mmRgba[o] !== mmRgba[o + 1] || mmRgba[o + 1] !== mmRgba[o + 2]) cmmColouredPx++;
  }

const metrics = {
  colorComputeMs: +(sumComp / nColor).toFixed(2),
  colorOverlayMs: +(sumOver / nColor).toFixed(2),
  colorRawFlowBoxMs: +(sumRaw / nColor).toFixed(2),
  pwColMs: +specMeans.pw.toFixed(2),
  cwColMs: +specMeans.cw.toFixed(2),
  tdiColMs: +specMeans.tdi.toFixed(2),
  cmmWrite27ColsMs: +cmmWrite27ColsMs.toFixed(3),
  cmmPaint3ColsMs: +cmmPaint3ColsMs.toFixed(3),
};

const cmmTableGain = cmmExact27ColsMs / cmmWrite27ColsMs;
const pct = (n: number, of: number) => `${((100 * n) / Math.max(1, of)).toFixed(1)} %`;

console.log('=== DOPPLER CPU HOT-PATH BENCHMARK (normal-excellent-window, 640x480) ===');
console.table(metrics);
console.log(
  `CMM colour write: ${cmmTableGain.toFixed(1)} times under the exact response alone (${cmmExact27ColsMs.toFixed(3)} ms a frame); colour on ${pct(cmmBloodColoured, cmmBlood)} of the blood samples, ${pct(cmmColouredPx, cmmPaintedPx)} of the painted strip`,
);

if (checkMode) {
  // Generous container-safe regression ceilings (~1.5x baseline to avoid noisy false alarms while catching real blowups)
  const ceilings: Record<keyof typeof metrics, number> = {
    colorComputeMs: 85,
    colorOverlayMs: 14,
    colorRawFlowBoxMs: 55,
    pwColMs: 25,
    cwColMs: 45,
    tdiColMs: 6,
    // a blowup only, as its neighbours: the best run takes 0.03 ms on a performance core, up to 0.1 on efficiency cores
    // or at a load average of 30, and 0.29 ms a frame with the exact response; its own check is the table's gain, below
    cmmWrite27ColsMs: 1,
    cmmPaint3ColsMs: 2,
  };
  const violations: string[] = [];
  for (const [k, limit] of Object.entries(ceilings)) {
    const val = metrics[k as keyof typeof metrics];
    if (val > limit) violations.push(`${k}: ${val}ms > ceiling ${limit}ms`);
  }
  // The table is what keeps the strip's colour write cheap: at least 3 times under the exact response, timed in this
  // same run. Best of 7, it measured 8.7–8.9 times on a quiet performance core and 6.9–13.6 on efficiency cores
  // (taskpolicy -c background), beside two npm test or at a load average of 30; a strip back to the response per sample
  // writes at 0.9 of it (0.287 against 0.270 ms), and one refilling the table for every column at 0.1
  if (!(cmmTableGain >= 3))
    violations.push(
      `cmmWrite27ColsMs: ${cmmWrite27ColsMs.toFixed(3)}ms, only ${cmmTableGain.toFixed(1)} times under the exact response's ${cmmExact27ColsMs.toFixed(3)}ms (at least 3)`,
    );
  // and the CMM cases measure something: colour written where there is blood, and painted; a reference on blood
  if (cmmBloodColoured === 0) violations.push(`cmmWrite27ColsMs: no colour on ${cmmBlood} blood samples`);
  if (cmmColouredPx === 0) violations.push(`cmmPaint3ColsMs: no colour on ${cmmPaintedPx} painted pixels`);
  if (!(cmmExactPower > 0)) violations.push('cmmWrite27ColsMs: its reference, the exact response, ran on no blood');
  if (violations.length > 0) {
    console.error('BENCHMARK REGRESSION:', violations.join('; '));
    process.exit(1);
  }
  console.log('BENCHMARK CHECK PASSED');
}
