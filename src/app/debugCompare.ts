import { loadCaseById } from '@/cases';
import { computeHeartPose, type HeartModel } from '@/simulator/anatomy/heartModel';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import type { PatientState } from '@/simulator/anatomy/thoraxModel';
import { buildCaseModels, REST_PATIENT } from '@/simulator/anatomy/caseModels';
import { scenePhysicsFor } from '@/simulator/renderer/scenePhysics';
import { ProceduralSliceRenderer } from '@/simulator/renderer/procedural/sliceRenderer';
import { createWebgl2Renderer } from '@/simulator/renderer/gpu/webgl2Renderer';
import { enumDefinesGlsl } from '@/simulator/renderer/gpu/glslCommon';
import {
  allocPolarFrame,
  DEFAULT_ACQUISITION,
  polarSpecFor,
  type AcquisitionSettings,
  type PolarFrameSpec,
  type Scene,
} from '@/simulator/renderer/types';
import { applyConsole, createConsoleState } from '@/simulator/renderer/postprocess/consolePipeline';
import {
  buildScanLut,
  computeSectorMapping,
  scanConvertLut,
} from '@/simulator/renderer/scanConvert';
import {
  allocColorField,
  colorSpectralWidthMps,
  computeColorField,
  DEFAULT_COLOR,
  overlayColorField,
  type ColorSettings,
} from '@/simulator/doppler/color/colorDoppler';
import {
  buildFlowParams,
  prepareClutterContext,
  sampleClutterAxialVelocity,
  sampleFlow,
  type FlowSample,
} from '@/simulator/doppler/flow-primitives/flowField';
import { colorWallFilterRangeMps } from '@/simulator/doppler/color/kasaiEstimator';
import { computeDopplerBoxBounds } from '@/simulator/renderer/gpu/deferredDoppler';
import { Tissue } from '@/simulator/anatomy/tissue';
import {
  beamFrameFromPose,
  poseFromControl,
  type BeamFrame,
  type ProbeControl,
} from '@/simulator/probe/pose';
import { canonicalControl, getViewTarget } from '@/simulator/windows/viewTargets';
import type { AnatomyChoice, ImageFormationChoice } from '@/simulator/core/protocol';

export interface BackendComparison {
  error?: string;
  lines: number;
  samples: number;
  structureAgreement: number;
  tissueAgreement: number;
  /**
   * Σ|a − b| / Σa over the samples both backends label alike: the parity of the acoustic chain. The boundary samples
   * that float precision labels differently on the two sides are counted by the structure/tissue agreement and left
   * out here — with them in, the ratio followed the composition of the frame (a bright pericardium in the near field
   * of the tamponade PLAX took it from 0.65 % to 1.0 % at decision 111 and past 1 % once the beam had its focusing
   * profile, decision 144) rather than any drift of the chain.
   */
  ampRelDiff: number;
  /** The same ratio over every sample, boundary flips included (information). */
  ampRelDiffAll: number;
  /** mean |Δtransmission| */
  transDiff: number;
  cpuMs: number;
  gpuMs: number;
  /** Most frequent structure disagreements as "cpu>gpu": count (diagnostic). */
  mismatches?: Record<string, number>;
  /**
   * Per CPU structure (by GLSL name, e.g. S_MV_ANT): how many samples it has and how many the GPU labels
   * differently. The frame-wide agreement hides a whole leaflet or annulus (< 0.5 % of the samples), so the
   * equivalence test bounds the disagreement of each structure separately.
   */
  perStructure?: Record<string, { samples: number; mismatched: number }>;
  /**
   * Per CPU LV segment code (decision 152, `lvSegments.ts`; 1–16 AHA, 17–20 the cap by quadrant): samples and how many
   * the GPU labels with another code. The segment boundaries are surfaces, so only boundary samples may differ.
   */
  perSegment?: Record<string, { samples: number; mismatched: number }>;
  /** Samples the GPU labels with a segment where the CPU has none, and the CPU's segment samples (decision 152). */
  segmentExtra?: number;
  segmentSamples?: number;
  /** A few mismatched samples with their heart-frame coordinates (diagnostic). */
  examples?: {
    line: number;
    sample: number;
    cpu: number;
    gpu: number;
    hx: number;
    hy: number;
    hz: number;
  }[];
}

type Tier = 'low' | 'medium' | 'high';

/** Canonical view of a case at a phase, with the scene the renderers take. */
function canonicalSetup(
  viewId: string,
  phase: number,
  caseId: string,
  tier: Tier,
  settings: AcquisitionSettings,
  probeOffsetV = 0,
  controlOverride?: ProbeControl,
  patient: PatientState = REST_PATIENT,
) {
  const c = loadCaseById(caseId);
  const { thorax, heart, tables } = buildCaseModels(c, patient);
  const canonical = canonicalControl(getViewTarget(viewId), heart, thorax);
  // an offset along the ribs' spacing puts a rib under the probe, which the presets avoid
  const ctrl = controlOverride ?? { ...canonical, v: canonical.v + probeOffsetV };
  const beam = beamFrameFromPose(poseFromControl(thorax, ctrl), 1);
  const spec = polarSpecFor(settings, tier);
  const sceneAt = (ph: number): Scene => ({
    heart,
    heartPose: computeHeartPose(heart, cycleStateAt(tables, ph)),
    thorax,
    // the app's physics (decision 238), at a frame other than the first so the flowing blood's lattice shift reaches
    // the comparison (decision 163)
    physics: scenePhysicsFor(c, settings, { bloodFrame: 7 }),
  });
  return { c, heart, thorax, tables, beam, spec, scene: sceneAt(phase), sceneAt };
}

/**
 * Debug/QA hook (window.__echotwin.compareBackends): renders the same canonical view with the CPU
 * reference renderer and the WebGL2 port on the main thread and reports agreement metrics. Used by
 * e2e/gpu-equivalence.spec.ts.
 */
export function compareBackends(
  viewId: string,
  phase: number,
  caseId = 'normal-excellent-window',
  tier: Tier = 'medium',
  probeOffsetV = 0,
  controlOverride?: ProbeControl,
  patient: PatientState = REST_PATIENT,
): BackendComparison {
  const { heart, beam, spec, scene } = canonicalSetup(
    viewId,
    phase,
    caseId,
    tier,
    DEFAULT_ACQUISITION,
    probeOffsetV,
    controlOverride,
    patient,
  );
  return compareFrames(heart, beam, spec, scene, phase);
}

/**
 * The same comparison with the experimental choices of the Dev panel: the Rodero heart (decisions 352 and 353), the
 * shipped volume fetched under the app's base, bound to the case's heart, read by the CPU tracer over its classifier and
 * by the GPU port over its textures; and the convolution image (decision 354), point scatterers and the DAS kernels in
 * one look, formed by the CPU tracer and by the GPU port's variant of pass A.
 */
export async function compareExperimentalBackends(
  viewId: string,
  phase: number,
  anatomy: AnatomyChoice,
  image: ImageFormationChoice,
  caseId = 'normal-excellent-window',
  tier: Tier = 'medium',
): Promise<BackendComparison> {
  const { heart, beam, spec, scene } = canonicalSetup(
    viewId,
    phase,
    caseId,
    tier,
    DEFAULT_ACQUISITION,
    0,
    undefined,
    REST_PATIENT,
  );
  let s: Scene = scene;
  if (anatomy === 'rodero') {
    const lib = await import('@/simulator/core/roderoHeart');
    const vol = await lib.loadRoderoVolume(async (path) => {
      const r = await fetch(`${import.meta.env.BASE_URL}${path}`);
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return r.arrayBuffer();
    });
    const c = loadCaseById(caseId);
    const ed = computeHeartPose(heart, cycleStateAt(buildCaseModels(c, REST_PATIENT).tables, 0));
    const vh = lib.volumeHeart(vol, heart, ed);
    s = {
      ...s,
      heartClassifier: vh.classifier({
        apparatus: scene.heartPose,
        motion: { pose: scene.heartPose, endDiastole: ed },
      }),
      heartVolumeGpu: lib.volumeHeartGpu(vh, ed),
    };
  }
  if (image === 'cole') {
    const lib = await import('@/simulator/core/coleImage');
    s = { ...s, imageFormation: lib.COLE_FORMATION, scatterGpu: lib.COLE_GPU };
  }
  return compareFrames(heart, beam, spec, s, phase);
}

/** CPU and GPU frames of one scene, compared sample by sample. */
function compareFrames(
  heart: HeartModel,
  beam: BeamFrame,
  spec: PolarFrameSpec,
  scene: Scene,
  phase: number,
): BackendComparison {
  const empty: BackendComparison = {
    lines: spec.lines,
    samples: spec.samples,
    structureAgreement: 0,
    tissueAgreement: 0,
    ampRelDiff: 1,
    ampRelDiffAll: 1,
    transDiff: 1,
    cpuMs: 0,
    gpuMs: 0,
  };
  const cpu = new ProceduralSliceRenderer();
  const fa = allocPolarFrame(spec);
  const t0 = performance.now();
  cpu.render(scene, beam, spec, phase, fa);
  const cpuMs = performance.now() - t0;
  const g = createWebgl2Renderer(undefined, { allowSoftware: true }); // the check runs on SwiftShader in Playwright
  if (!g.renderer) return { ...empty, error: g.reason, cpuMs };
  const fb = allocPolarFrame(spec);
  g.renderer.render(scene, beam, spec, phase, fb); // warm-up (shader compile, textures)
  const t1 = performance.now();
  g.renderer.render(scene, beam, spec, phase, fb);
  const gpuMs = performance.now() - t1;
  g.renderer.dispose();
  const n = spec.lines * spec.samples;
  let sAgree = 0,
    tAgree = 0,
    ampDiff = 0,
    ampSum = 0,
    ampDiffSame = 0,
    ampSumSame = 0,
    trDiff = 0;
  const mm = new Map<string, number>();
  const per = new Map<number, { samples: number; mismatched: number }>();
  const perSeg = new Map<number, { samples: number; mismatched: number }>();
  let segExtra = 0,
    segSamples = 0;
  const examples: NonNullable<BackendComparison['examples']> = [];
  for (let i = 0; i < n; i++) {
    const sa = fa.structure[i]!;
    let ps = per.get(sa);
    if (!ps) per.set(sa, (ps = { samples: 0, mismatched: 0 }));
    ps.samples++;
    if (sa === fb.structure[i]) sAgree++;
    else {
      ps.mismatched++;
      const k = `${fa.structure[i]}>${fb.structure[i]}`;
      mm.set(k, (mm.get(k) ?? 0) + 1);
      if (examples.length < 40 && i % 3 === 0) {
        const li = Math.floor(i / spec.samples),
          si = i % spec.samples;
        const theta = -spec.sectorRad / 2 + (spec.sectorRad * (li + 0.5)) / spec.lines;
        const r = (si + 0.5) * (spec.depthCm / spec.samples);
        const ct = Math.cos(theta),
          sn = Math.sin(theta);
        const px = beam.origin.x + (beam.forward.x * ct + beam.lateral.x * sn) * r;
        const py = beam.origin.y + (beam.forward.y * ct + beam.lateral.y * sn) * r;
        const pz = beam.origin.z + (beam.forward.z * ct + beam.lateral.z * sn) * r;
        const hf = heart.frame;
        const dx = px - hf.origin.x,
          dy = py - hf.origin.y,
          dz = pz - hf.origin.z;
        examples.push({
          line: li,
          sample: si,
          cpu: fa.structure[i]!,
          gpu: fb.structure[i]!,
          hx: +(dx * hf.ex.x + dy * hf.ex.y + dz * hf.ex.z).toFixed(2),
          hy: +(dx * hf.ey.x + dy * hf.ey.y + dz * hf.ey.z).toFixed(2),
          hz: +(dx * hf.ez.x + dy * hf.ez.y + dz * hf.ez.z).toFixed(2),
        });
      }
    }
    const ca = fa.segment[i]!;
    if (ca === 0 && fb.segment[i]! > 0) segExtra++;
    if (ca > 0) {
      segSamples++;
      let e = perSeg.get(ca);
      if (!e) perSeg.set(ca, (e = { samples: 0, mismatched: 0 }));
      e.samples++;
      if (fb.segment[i] !== ca) e.mismatched++;
    }
    const sameTissue = fa.tissue[i] === fb.tissue[i];
    if (sameTissue) tAgree++;
    const d = Math.abs(fa.amplitude[i]! - fb.amplitude[i]!);
    ampDiff += d;
    ampSum += fa.amplitude[i]!;
    if (sameTissue && sa === fb.structure[i]) {
      ampDiffSame += d;
      ampSumSame += fa.amplitude[i]!;
    }
    trDiff += Math.abs(fa.transmission[i]! - fb.transmission[i]!);
  }
  const mismatches = Object.fromEntries([...mm.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8));
  const names = structureNames();
  const perStructure = Object.fromEntries(
    [...per.entries()].map(([id, v]) => [names.get(id) ?? `S_${id}`, v]),
  );
  return {
    lines: spec.lines,
    samples: spec.samples,
    structureAgreement: sAgree / n,
    tissueAgreement: tAgree / n,
    ampRelDiff: ampDiffSame / Math.max(ampSumSame, 1e-6),
    ampRelDiffAll: ampDiff / Math.max(ampSum, 1e-6),
    transDiff: trDiff / n,
    cpuMs,
    gpuMs,
    mismatches,
    examples,
    perStructure,
    perSegment: Object.fromEntries([...perSeg.entries()].map(([k, v]) => [String(k), v])),
    segmentExtra: segExtra,
    segmentSamples: segSamples,
  };
}

/** Structure id → GLSL name, from the same define list the shaders use (`S_LV_CAV 1` …). */
function structureNames(): Map<number, string> {
  const out = new Map<number, string>();
  for (const m of enumDefinesGlsl().matchAll(/#define (S_[A-Z0-9_]+) (\d+)/g))
    out.set(Number(m[2]), m[1]!);
  return out;
}

export interface ImageChainComparison {
  error?: string;
  lines: number;
  samples: number;
  /** Console path of each frame of the mixed chain compared with the all-CPU reference chain. */
  framePaths: string[];
  /** Mixed chain vs the CPU console on every frame, per frame (persistence carries across the switches). */
  displayMeanAbsDiff: number[];
  displayMaxDiff: number[];
  displayFracOver1: number[];
  /** Structure and tissue ids of the packed read-back vs the float read-back. */
  idsAgreement: number;
  /** LV segment codes of the display path's read-back vs the float path's (decision 152); samples with a segment. */
  segmentAgreement: number;
  segmentSamples: number;
  /** Largest relative error of the 8-bit transmission code where the transmission is ≥ 1e-4. */
  transMaxRelErr: number;
  /** GPU present pass vs CPU scan conversion + colour overlay of the same display (RGB channels, every pixel). */
  presentMaxDiff: number;
  /** Pixels differing by more than one level, as a fraction of the pixels inside the sector. */
  presentFracOver1: number;
  presentColorPixels: number;
  /** Pixels coloured on one side and grey on the other (a colour box edge placed differently). */
  presentColourFlips: number;
  gpuDisplayMs: number;
  cpuConsoleMs: number;
}

/**
 * Debug/QA hook (window.__echotwin.compareImageChain, decision 54): the GPU console and present pass must
 * reproduce the CPU console, scan conversion and colour overlay. Four frames form a mixed chain (GPU, GPU, CPU,
 * GPU) compared with the CPU console on every frame, so persistence is checked within the GPU and across both
 * switches. The present pass is compared on a mirrored sector with a synthetic colour field that has sign jumps
 * like aliasing, saturated values, variance and holes.
 */
export function compareImageChain(
  viewId: string,
  phase: number,
  caseId = 'normal-excellent-window',
  tier: Tier = 'medium',
  overrides: Partial<AcquisitionSettings> = {},
): ImageChainComparison {
  const settings: AcquisitionSettings = { ...DEFAULT_ACQUISITION, ...overrides };
  const { c, beam, spec, sceneAt } = canonicalSetup(viewId, phase, caseId, tier, settings);
  const n = spec.lines * spec.samples;
  const result: ImageChainComparison = {
    lines: spec.lines,
    samples: spec.samples,
    framePaths: [],
    displayMeanAbsDiff: [],
    displayMaxDiff: [],
    displayFracOver1: [],
    idsAgreement: 0,
    segmentAgreement: 0,
    segmentSamples: 0,
    transMaxRelErr: 1,
    presentMaxDiff: 255,
    presentFracOver1: 1,
    presentColorPixels: 0,
    presentColourFlips: 0,
    gpuDisplayMs: 0,
    cpuConsoleMs: 0,
  };
  const g = createWebgl2Renderer(undefined, { allowSoftware: true });
  if (!g.renderer) return { ...result, error: g.reason };
  const gpu = g.renderer;
  const refState = createConsoleState(c.seed);
  const mixState = createConsoleState(c.seed);
  const fFloat = allocPolarFrame(spec);
  const fGpu = allocPolarFrame(spec);
  const dispRef = new Uint8ClampedArray(n);
  const dispMix = new Uint8ClampedArray(n);
  const dispGpu = new Uint8ClampedArray(n); // last display formed on the GPU (input of the present comparison)
  for (const [k, path] of (['gpu', 'gpu', 'cpu', 'gpu'] as const).entries()) {
    const ph = (phase + 0.05 * k) % 1;
    const scene = sceneAt(ph);
    gpu.render(scene, beam, spec, ph, fFloat); // the float envelope both CPU consoles read
    const tc = performance.now();
    applyConsole(fFloat, settings, refState, dispRef);
    result.cpuConsoleMs = performance.now() - tc;
    if (path === 'gpu') {
      const tg = performance.now();
      gpu.renderDisplay(
        scene,
        beam,
        spec,
        ph,
        fGpu,
        undefined,
        { settings, state: mixState },
        dispMix,
      );
      result.gpuDisplayMs = performance.now() - tg;
      dispGpu.set(dispMix);
    } else {
      if (mixState.gpuHistory) gpu.restoreCpuHistory(mixState, spec);
      applyConsole(fFloat, settings, mixState, dispMix);
    }
    result.framePaths.push(path);
    let sum = 0,
      max = 0,
      over = 0;
    for (let i = 0; i < n; i++) {
      const d = Math.abs(dispRef[i]! - dispMix[i]!);
      sum += d;
      if (d > max) max = d;
      if (d > 1) over++;
    }
    result.displayMeanAbsDiff.push(sum / n);
    result.displayMaxDiff.push(max);
    result.displayFracOver1.push(over / n);
  }
  let ids = 0,
    trErr = 0,
    segSame = 0,
    segN = 0;
  for (let i = 0; i < n; i++) {
    if (fFloat.structure[i] === fGpu.structure[i] && fFloat.tissue[i] === fGpu.tissue[i]) ids++;
    if (fFloat.segment[i]! > 0 || fGpu.segment[i]! > 0) {
      segN++;
      if (fFloat.segment[i] === fGpu.segment[i]) segSame++;
    }
    const t = fFloat.transmission[i]!;
    if (t >= 1e-4) trErr = Math.max(trErr, Math.abs(fGpu.transmission[i]! - t) / t);
  }
  result.idsAgreement = ids / n;
  result.segmentAgreement = segN ? segSame / segN : 0;
  result.segmentSamples = segN;
  result.transMaxRelErr = trErr;
  // present: mirrored sector, synthetic colour field with aliasing and variance
  const W = 640,
    H = 520;
  const mapping = computeSectorMapping(spec, W, H, true, 1);
  const lut = buildScanLut(spec, mapping);
  const vel = new Float32Array(n).fill(NaN);
  const variance = new Float32Array(n);
  for (let li = Math.floor(spec.lines * 0.2); li < Math.floor(spec.lines * 0.8); li++)
    for (let si = Math.floor(spec.samples * 0.25); si < Math.floor(spec.samples * 0.7); si++) {
      if ((li + si) % 7 === 0) continue; // holes: samples without colour
      const i = li * spec.samples + si;
      vel[i] = ((((li * 7 + si * 3) % 41) - 20) / 20) * 0.8; // ±0.8 m/s against a 0.62 m/s scale
      variance[i] = (si % 11) / 10;
    }
  const colorSettings: ColorSettings = {
    ...DEFAULT_COLOR,
    boxRMinCm: 4,
    boxRMaxCm: 12.5,
    boxThetaMinRad: -0.35,
    boxThetaMaxRad: 0.28,
    showVariance: true,
  };
  const cpuRgba = new Uint8ClampedArray(W * H * 4);
  scanConvertLut(dispGpu, lut, cpuRgba, settings.tintMap);
  overlayColorField(cpuRgba, lut, vel, variance, colorSettings);
  const gpuRgba = new Uint8Array(W * H * 4);
  gpu.present({
    lut,
    width: W,
    height: H,
    color: { vel, variance, version: 1, settings: colorSettings },
    tintMap: settings.tintMap,
    readback: gpuRgba,
  });
  gpu.dispose();
  let pMax = 0,
    pOver = 0,
    colored = 0,
    flips = 0,
    inside = 0;
  for (let p = 0; p < W * H; p++) {
    const o = p * 4;
    if ((lut.idx[p] ?? -1) >= 0) inside++;
    let d = 0;
    for (let ch = 0; ch < 3; ch++) d = Math.max(d, Math.abs(cpuRgba[o + ch]! - gpuRgba[o + ch]!));
    if (d > pMax) pMax = d;
    if (d > 1) pOver++;
    const cpuColoured = cpuRgba[o]! !== cpuRgba[o + 1]! || cpuRgba[o + 1]! !== cpuRgba[o + 2]!;
    const gpuColoured = gpuRgba[o]! !== gpuRgba[o + 1]! || gpuRgba[o + 1]! !== gpuRgba[o + 2]!;
    if (cpuColoured) colored++;
    if (cpuColoured !== gpuColoured) flips++;
  }
  result.presentMaxDiff = pMax;
  result.presentFracOver1 = pOver / Math.max(1, inside);
  result.presentColorPixels = colored;
  result.presentColourFlips = flips;
  return result;
}

export interface DeferredDopplerComparison {
  error?: string;
  boxW: number;
  boxH: number;
  /** Deterministic mode (estimate = undefined): sample-by-sample CPU vs GPU agreement inside ROI box. */
  detValidAgreement: number;
  detColoredCount: number;
  detVelMeanDiff: number;
  detVelMaxDiff: number;
  detVarMaxDiff: number;
  /** Deterministic deferred present RGBA parity (CPU overlayColorField vs GPU deferred present). */
  detPresentMaxDiff: number;
  detPresentFracOver1: number;
  /**
   * Stochastic Kasai (8-pulse 3-basis Legendre): blood & clutter parity at the box's wall filter (the default 20 cm/s,
   * which keeps the myocardium out, unless `colorOverrides` sets one) vs the lowest cutoff the packet allows at that scale
   * (0.11 Nyquist, where the moving wall flashes).
   */
  kasaiWf: {
    cpuBloodFill: number;
    gpuBloodFill: number;
    cpuBloodMeanV: number;
    gpuBloodMeanV: number;
    cpuClutterFill: number;
    gpuClutterFill: number;
  };
  kasaiMinWf: {
    cpuBloodFill: number;
    gpuBloodFill: number;
    cpuBloodMeanV: number;
    gpuBloodMeanV: number;
    cpuClutterFill: number;
    gpuClutterFill: number;
  };
}

/**
 * Debug/QA hook (`window.__echotwin.compareDeferredDoppler`): runs the real WebGL2 deferred Color
 * Doppler pipeline (Pass D1 + D2 + D3 + Present) against CPU `computeColorField` + `overlayColorField`
 * on the exact same Polar G-Buffer and ROI raw-flow/clutter samples, verifying both deterministic
 * sample-by-sample parity and stochastic 3-basis Legendre Kasai clutter rejection/flash parity.
 */
export function compareDeferredDoppler(
  viewId = 'plax',
  phase = 0.22,
  caseId = 'normal-excellent-window',
  tier: Tier = 'low',
  colorOverrides: Partial<ColorSettings> = {},
): DeferredDopplerComparison {
  const emptyStats = {
    cpuBloodFill: 0,
    gpuBloodFill: 0,
    cpuBloodMeanV: 0,
    gpuBloodMeanV: 0,
    cpuClutterFill: 0,
    gpuClutterFill: 0,
  };
  const empty: DeferredDopplerComparison = {
    boxW: 0,
    boxH: 0,
    detValidAgreement: 0,
    detColoredCount: 0,
    detVelMeanDiff: 1,
    detVelMaxDiff: 1,
    detVarMaxDiff: 1,
    detPresentMaxDiff: 255,
    detPresentFracOver1: 1,
    kasaiWf: { ...emptyStats },
    kasaiMinWf: { ...emptyStats },
  };
  const g = createWebgl2Renderer(undefined, { allowSoftware: true });
  if (!g.renderer) return { ...empty, error: g.reason };
  const gpu = g.renderer;
  try {
    const settings: AcquisitionSettings = { ...DEFAULT_ACQUISITION };
    const { c, heart, thorax, tables, beam, spec, scene } = canonicalSetup(
      viewId,
      phase,
      caseId,
      tier,
      settings,
    );
    const flow = buildFlowParams(c, heart, tables, thorax);
    const n = spec.lines * spec.samples;
    const frame = allocPolarFrame(spec);
    const disp = new Uint8ClampedArray(n);
    const consoleState = createConsoleState(c.seed);
    // Populate CPU exact float transmission FIRST, then GPU Polar G-Buffer + packed display
    // (keeping packedFresh=true so zero-readback gpu.present() runs)
    const frameFloat = allocPolarFrame(spec);
    gpu.render(scene, beam, spec, phase, frameFloat);
    gpu.renderDisplay(scene, beam, spec, phase, frame, undefined, { settings, state: consoleState }, disp);
    frame.transmission.set(frameFloat.transmission);
    frame.tissue.set(frameFloat.tissue);
    frame.structure.set(frameFloat.structure);

    const baseColor: ColorSettings = {
      ...DEFAULT_COLOR,
      boxRMinCm: 3.5,
      boxRMaxCm: 12.0,
      boxThetaMinRad: -0.32,
      boxThetaMaxRad: 0.32,
      persistence: 0,
      smoothing: 0.45,
      gainDb: 0,
      ...colorOverrides,
    };
    const box = computeDopplerBoxBounds(spec, baseColor);
    const rawFlowBox = new Float32Array(box.boxW * box.boxH * 4);
    const boxDisp = new Float32Array(box.boxW * box.boxH);
    const boxVPerp = new Float32Array(box.boxW * box.boxH);
    const hf = heart.frame;
    const hp = scene.heartPose;
    const ox = beam.origin.x - hf.origin.x;
    const oy = beam.origin.y - hf.origin.y;
    const oz = beam.origin.z - hf.origin.z;
    const ohx = ox * hf.ex.x + oy * hf.ex.y + oz * hf.ex.z;
    const ohy = ox * hf.ey.x + oy * hf.ey.y + oz * hf.ey.z;
    const ohz = ox * hf.ez.x + oy * hf.ez.y + oz * hf.ez.z;
    const dr = spec.depthCm / spec.samples;
    const clutterCtx = prepareClutterContext(heart, tables, hp, phase, flow.flowPeaks);
    const fs: FlowSample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };

    for (let li = box.liMin; li <= box.liMax; li++) {
      const theta = -spec.sectorRad / 2 + (spec.sectorRad * (li + 0.5)) / spec.lines;
      const ct = Math.cos(theta);
      const sn = Math.sin(theta);
      const dx = beam.forward.x * ct + beam.lateral.x * sn;
      const dy = beam.forward.y * ct + beam.lateral.y * sn;
      const dz = beam.forward.z * ct + beam.lateral.z * sn;
      const dhx = dx * hf.ex.x + dy * hf.ex.y + dz * hf.ex.z;
      const dhy = dx * hf.ey.x + dy * hf.ey.y + dz * hf.ey.z;
      const dhz = dx * hf.ez.x + dy * hf.ez.y + dz * hf.ez.z;
      const rowIdx = (li - box.liMin) * box.boxW;
      for (let si = box.siMin; si <= box.siMax; si++) {
        const gIdx = rowIdx + (si - box.siMin);
        const o = gIdx * 4;
        const pIdx = li * spec.samples + si;
        const r = (si + 0.5) * dr;
        const hx = ohx + dhx * r;
        const hy = ohy + dhy * r;
        const hz = ohz + dhz * r;
        const tId = frame.tissue[pIdx]!;
        const isClutterTissue = tId === Tissue.Myocardium || tId === Tissue.Valve;
        sampleFlow(flow, tables, hp, phase, hx, hy, hz, fs);
        if (fs.present) {
          const vAxial = -(fs.vx * dhx + fs.vy * dhy + fs.vz * dhz);
          const vSq = fs.vx * fs.vx + fs.vy * fs.vy + fs.vz * fs.vz;
          const vPerp = Math.sqrt(Math.max(0, vSq - vAxial * vAxial));
          rawFlowBox[o] = vAxial;
          rawFlowBox[o + 1] = colorSpectralWidthMps(vAxial, fs.dispersion, vPerp, r);
          rawFlowBox[o + 2] = 1;
          boxDisp[gIdx] = fs.dispersion;
          boxVPerp[gIdx] = vPerp;
        }
        rawFlowBox[o + 3] = isClutterTissue
          ? sampleClutterAxialVelocity(
              clutterCtx,
              heart,
              hx,
              hy,
              hz,
              tId,
              frame.structure[pIdx] ?? -1,
              dhx,
              dhy,
              dhz,
            )
          : 0;
      }
    }

    const sampleFromBox = (
      _idx: number,
      li: number,
      si: number,
      out: { v: number; disp: number; present: number; vPerp?: number; vWall?: number },
    ) => {
      const gIdx = (li - box.liMin) * box.boxW + (si - box.siMin);
      const o = gIdx * 4;
      out.v = rawFlowBox[o]!;
      out.disp = boxDisp[gIdx]!;
      out.vPerp = boxVPerp[gIdx]!;
      out.present = rawFlowBox[o + 2]!;
      out.vWall = rawFlowBox[o + 3]!;
    };

    const acq = { frequencyMHz: settings.frequencyMHz, harmonics: settings.harmonics };
    const cpuDet = allocColorField(n);
    // Sample-by-sample CPU <-> GPU Kasai parity at the packet's lowest cutoff (maximum blood coverage),
    // using the exact same (realization, seed) now that CPU and GPU share the same 2x2 correlated Gaussian draw
    const detCfg: ColorSettings = {
      ...baseColor,
      wallFilterMps: colorWallFilterRangeMps(baseColor.scaleMps).min,
    };
    const detEst = { realization: 1, seed: 42 };
    computeColorField(frame, detCfg, null, sampleFromBox, cpuDet, acq, detEst);

    const shadedDet = gpu.shadeDoppler({
      spec,
      settings: detCfg,
      frequencyMHz: settings.frequencyMHz,
      harmonics: settings.harmonics,
      rawFlowBox,
      estimate: detEst,
      resetHistory: true,
    });
    if (!shadedDet) return { ...empty, error: 'shadeDoppler Kasai parity pass failed' };
    const gpuDetVel = new Float32Array(n);
    const gpuDetVar = new Float32Array(n);
    gpu.readbackDopplerBox(spec, detCfg.scaleMps, detCfg.baselineShiftMps, gpuDetVel, gpuDetVar);

    let agree = 0;
    let bothColored = 0;
    let velDiffSum = 0;
    let velMaxDiff = 0;
    let varMaxDiff = 0;
    const span = 2 * detCfg.scaleMps;
    for (let li = box.liMin; li <= box.liMax; li++) {
      for (let si = box.siMin; si <= box.siMax; si++) {
        const i = li * spec.samples + si;
        const vc = cpuDet.vel[i]!;
        const vg = gpuDetVel[i]!;
        const cOk = Number.isFinite(vc);
        const gOk = Number.isFinite(vg);
        if (cOk === gOk) agree++;
        if (cOk && gOk) {
          bothColored++;
          let dv = Math.abs(vc - vg);
          if (dv > detCfg.scaleMps) dv = Math.abs(dv - span);
          velDiffSum += dv;
          if (dv > velMaxDiff) velMaxDiff = dv;
          const dvar = Math.abs(cpuDet.variance[i]! - gpuDetVar[i]!);
          if (dvar > varMaxDiff) varMaxDiff = dvar;
        }
      }
    }

    // End-to-end zero-readback deferred present parity on deterministic color field
    const W = 480;
    const H = 360;
    const mapping = computeSectorMapping(spec, W, H, false, 1);
    const lut = buildScanLut(spec, mapping);
    const cpuRgba = new Uint8ClampedArray(W * H * 4);
    scanConvertLut(disp, lut, cpuRgba, settings.tintMap);
    overlayColorField(cpuRgba, lut, cpuDet.vel, cpuDet.variance, detCfg, cpuDet.power);
    const gpuRgba = new Uint8Array(W * H * 4);
    gpu.present({
      lut,
      width: W,
      height: H,
      color: { deferred: true, settings: detCfg },
      tintMap: settings.tintMap,
      readback: gpuRgba,
    });
    let detPresentMaxDiff = 0;
    let detPresentOver1 = 0;
    let inside = 0;
    for (let p = 0; p < W * H; p++) {
      if ((lut.idx[p] ?? -1) < 0) continue;
      inside++;
      const o = p * 4;
      const d = Math.max(
        Math.abs(cpuRgba[o]! - gpuRgba[o]!),
        Math.abs(cpuRgba[o + 1]! - gpuRgba[o + 1]!),
        Math.abs(cpuRgba[o + 2]! - gpuRgba[o + 2]!),
      );
      if (d > detPresentMaxDiff) detPresentMaxDiff = d;
      if (d > 1) detPresentOver1++;
    }

    // Stochastic 8-pulse Kasai + 3-basis Legendre clutter check: the box's wall filter vs the packet's lowest cutoff
    const runKasai = (wfMps: number) => {
      const cfg: ColorSettings = { ...baseColor, wallFilterMps: wfMps };
      const cpuOut = allocColorField(n);
      const est = { realization: 2, seed: 42 };
      computeColorField(frame, cfg, null, sampleFromBox, cpuOut, acq, est);
      gpu.shadeDoppler({
        spec,
        settings: cfg,
        frequencyMHz: settings.frequencyMHz,
        harmonics: settings.harmonics,
        estimate: est,
        resetHistory: true,
      });
      const gVel = new Float32Array(n);
      const gVar = new Float32Array(n);
      gpu.readbackDopplerBox(spec, cfg.scaleMps, cfg.baselineShiftMps, gVel, gVar);

      let bloodTot = 0,
        cpuBloodN = 0,
        gpuBloodN = 0,
        cpuBloodV = 0,
        gpuBloodV = 0;
      let clutTot = 0,
        cpuClutN = 0,
        gpuClutN = 0;
      for (let li = box.liMin; li <= box.liMax; li++) {
        for (let si = box.siMin; si <= box.siMax; si++) {
          const i = li * spec.samples + si;
          const tId = frame.tissue[i]!;
          const o = ((li - box.liMin) * box.boxW + (si - box.siMin)) * 4;
          if (tId === Tissue.Blood && rawFlowBox[o + 2]! > 0.5) {
            bloodTot++;
            if (Number.isFinite(cpuOut.vel[i])) {
              cpuBloodN++;
              cpuBloodV += cpuOut.vel[i]!;
            }
            if (Number.isFinite(gVel[i])) {
              gpuBloodN++;
              gpuBloodV += gVel[i]!;
            }
          } else if (tId === Tissue.Myocardium || tId === Tissue.Valve) {
            clutTot++;
            if (Number.isFinite(cpuOut.vel[i])) cpuClutN++;
            if (Number.isFinite(gVel[i])) gpuClutN++;
          }
        }
      }
      return {
        cpuBloodFill: bloodTot ? cpuBloodN / bloodTot : 0,
        gpuBloodFill: bloodTot ? gpuBloodN / bloodTot : 0,
        cpuBloodMeanV: cpuBloodN ? cpuBloodV / cpuBloodN : 0,
        gpuBloodMeanV: gpuBloodN ? gpuBloodV / gpuBloodN : 0,
        cpuClutterFill: clutTot ? cpuClutN / clutTot : 0,
        gpuClutterFill: clutTot ? gpuClutN / clutTot : 0,
      };
    };

    const kasaiWf = runKasai(baseColor.wallFilterMps);
    const kasaiMinWf = runKasai(colorWallFilterRangeMps(baseColor.scaleMps).min);

    return {
      boxW: box.boxW,
      boxH: box.boxH,
      detValidAgreement: box.totalTexels ? agree / box.totalTexels : 1,
      detColoredCount: bothColored,
      detVelMeanDiff: bothColored ? velDiffSum / bothColored : 0,
      detVelMaxDiff: velMaxDiff,
      detVarMaxDiff: varMaxDiff,
      detPresentMaxDiff,
      detPresentFracOver1: inside ? detPresentOver1 / inside : 0,
      kasaiWf,
      kasaiMinWf,
    };
  } finally {
    gpu.dispose();
  }
}
