import type { CaseDefinition } from '@/cases/schema';
import { buildCaseModels } from '@/simulator/anatomy/caseModels';
import {
  computeHeartPose,
  heartLandmarks,
  type HeartModel,
  type HeartPose,
} from '@/simulator/anatomy/heartModel';
import type { AssetFetcher } from '@/simulator/anatomy/roderoAsset';
import type { ThoraxModel } from '@/simulator/anatomy/thoraxModel';
import { CardiacClock } from '@/simulator/cardiac-cycle/clock';
import { cycleStateAt, type BeatTables } from '@/simulator/cardiac-cycle/cycleModel';
import { respiratoryDepth } from '@/simulator/cardiac-cycle/respiration';
import type { ColorField } from '@/simulator/doppler/color/colorDoppler';
import { buildDopplerPerfFrame, isDopplerModality } from '@/simulator/doppler/dopplerPerf';
import {
  buildFlowParams,
  type FlowFieldParams,
} from '@/simulator/doppler/flow-primitives/flowField';
import { spectralRange } from '@/simulator/doppler/spectral/spectrum';
import { computeGroundTruth, type StructuredEchoTruth } from '@/simulator/hemodynamics/groundTruth';
import {
  beamFrameFromPose,
  contactQuality,
  poseFromControl,
  type BeamFrame,
} from '@/simulator/probe/pose';
import { AtlasRenderer, AtlasRenderer as AtlasBackend } from '@/simulator/renderer/atlas/atlasRenderer';
import { acquisitionFrameRate, cadenceHz } from '@/simulator/renderer/frameRate';
import { createWebgl2Renderer, type Webgl2Renderer } from '@/simulator/renderer/gpu/webgl2Renderer';
import { isSpectralModality, isStripModality } from '@/simulator/renderer/modality';
import {
  applyConsole,
  createConsoleState,
  NO_ARTIFACTS,
  persistenceOverTime,
  type ArtifactSettings,
  type ConsoleState,
} from '@/simulator/renderer/postprocess/consolePipeline';
import { ProceduralSliceRenderer } from '@/simulator/renderer/procedural/sliceRenderer';
import type { SectorMapping } from '@/simulator/renderer/scanConvert';
import {
  caseArtifactLevels,
  consoleArtifacts,
  scenePhysicsFor,
  type ArtifactLevels,
} from '@/simulator/renderer/scenePhysics';
import { encodeTransmission } from '@/simulator/renderer/transmissionCode';
import {
  allocPolarFrame,
  polarSpecFor,
  type AcquisitionSettings,
  type PolarFrame,
  type PolarFrameSpec,
  type RendererBackend,
  type RenderHints,
  type Scene,
  type ScenePhysics,
} from '@/simulator/renderer/types';
import { analyzeView, type ViewAnalysis } from '@/simulator/view-recognition/viewQuality';
import { constrainAcquisition } from './acquisitionInput';
import { artifactOverridesKey, patientKey } from './acquisitionKeys';
import { buildChainedBeatTables, computePhaseMarks } from './beatChaining';
import { BeatHistory } from './beatHistory';
import { CineBuffer, type CineFrame } from './cineBuffer';
import { ColorFieldDriver } from './colorFieldDriver';
import { EcgRing } from './ecgRing';
import { ExperimentalLoaders } from './experimentalLoaders';
import { FrameCompositor } from './frameCompositor';
import { GpuRecovery, type GpuFactory } from './gpuRecovery';
import { MMODE_TRACE_SHARE } from './mmodeStrip';
import type {
  PhaseMarks,
  QualityTier,
  SimInput,
  SimOutput,
  SimRequest,
  SimResponse,
  StripInfo,
} from './protocol';
import { resolveQualityTier } from './quality';
import { routeSimRequest } from './requestRouter';
import { StripEngine, type StripCtx } from './stripEngine';

/**
 * Headless simulator (spec 32 data flow). Runs in a Web Worker in the app and directly in tests.
 * Owns: case models, cardiac clock, renderer backend, console state, cine ring, Doppler strips,
 * ECG ring and view analysis. The UI only sends inputs and draws outputs + overlays.
 */
export class SimulatorCore {
  readonly caseDef: CaseDefinition;
  readonly truth: StructuredEchoTruth;
  private thorax: ThoraxModel;
  private heart: HeartModel;
  private tables: BeatTables;
  /** Tables at the case's heart rate: the ground truth and, in atrial fibrillation, the orifice every beat fills through. */
  private nominalTables: BeatTables;
  private tablesBeat = -1;
  private tablesVersion = 0;
  private clock: CardiacClock;
  private flow: FlowFieldParams;
  private beatHistory = new BeatHistory<{
    tables: BeatTables;
    flow: FlowFieldParams;
    tablesVersion: number;
    heart: HeartModel;
    thorax: ThoraxModel;
  }>();
  private procedural = new ProceduralSliceRenderer();
  /** Lazy loaders for the experimental Rodero volume heart and Cole convolution image formation (decisions 352–354). */
  private experimental: ExperimentalLoaders;
  /** The CPU tracer of review mode's point explanation (decision 340), made on first use. */
  private explainTracer: ProceduralSliceRenderer | null = null;
  private atlas: AtlasRenderer;
  private backend: RendererBackend;
  /** WebGL2 port of the procedural renderer; null when unavailable (reason in `gpuReason`). */
  private gpu: Webgl2Renderer | null;
  private gpuReason: string;
  /** Brings the port back after its context was lost (suspend, GPU process restart) instead of keeping the CPU tracer. */
  private gpuRecovery: GpuRecovery;
  /** The context the `gpuContext` dev request lost, to restore it: WEBGL_lose_context is null on a lost context. */
  private devLostContext: WEBGL_lose_context | null = null;
  private tier: QualityTier = 'medium';
  private consoleState: ConsoleState;
  private artifactLevels: ArtifactLevels = { sideLobe: 0, mirror: 0, beamWidth: 0, clutter: 0 };
  private artifacts: ArtifactSettings = NO_ARTIFACTS;
  private caseArtifacts: ArtifactLevels = this.artifactLevels;
  private input: SimInput;
  private acquisitionModality: SimInput['modality'];
  private displayedFrame: CineFrame | undefined;
  private frame: PolarFrame | null = null;
  private display: Uint8ClampedArray | null = null;
  private colorDriver = new ColorFieldDriver();
  private colorFps = 0;
  private displayOnGpu = false;
  private dopplerOnGpu = false;
  private transCodeScratch = new Uint8Array(0);
  private cine = new CineBuffer();
  private frameId = 0;
  private timeS = 0;
  private frameAccumulator = 0;
  private frameIntervalS = 1 / 30;
  private lastFrameTimeS = NaN;
  private lastPersistence = 0;
  private lastAnalysis: ViewAnalysis | null = null;
  private analysisCounter = 0;
  /** The learner's session the frames are acquired in (`SimInput.sessionEpoch`). */
  private sessionEpoch = 0;
  private ecgRing = new EcgRing();
  private strips = new StripEngine();
  private poseCacheHeart: HeartModel | null = null;
  private poseCacheTablesVersion = -1;
  private poseCacheBins: Array<HeartPose | undefined> = new Array(256);
  private patientKey = '';
  private compositor = new FrameCompositor();
  private modelVersion = 0;
  private lastOutputBeam: BeamFrame | null = null;
  private lastSector: (SectorMapping & { x: number; y: number }) | null = null;
  private lastStrip: StripInfo | null = null;
  private timing = {
    renderFrameMs: 0,
    compositeMs: 0,
    analysisMs: 0,
    consoleMs: 0,
    cineMs: 0,
    presentMs: 0,
    overlayColorMs: 0,
  };
  private lastColorStepMs = 0;
  private prevBeam: BeamFrame | null = null;
  private stationaryFrames = 0;
  private frameBudgetMs = 25;
  private disposed = false;
  private artifactKey = 'none';

  constructor(caseDef: CaseDefinition, input: SimInput, options: SimulatorCoreOptions = {}) {
    this.caseDef = caseDef;
    // an experimental asset arriving (the Rodero heart, the Cole kernels) changes what the beam meets: the frames cached by
    // pose and the strip's caches by ray and bin were formed without it
    this.experimental = new ExperimentalLoaders(options.fetchAsset ?? fetchFromAppBase, () => {
      this.atlas.invalidate();
      this.strips.reset();
    });
    this.input = constrainAcquisition(input);
    this.acquisitionModality = input.modality;
    this.sessionEpoch = input.sessionEpoch ?? 0;
    const models = buildCaseModels(caseDef, input.patient);
    this.thorax = models.thorax;
    this.heart = models.heart;
    this.tables = models.tables;
    this.patientKey = patientKey(input.patient);
    this.artifactKey = artifactOverridesKey(input.artifactOverrides);
    this.nominalTables = this.tables;
    this.clock = new CardiacClock(caseDef.rhythm, caseDef.seed);
    this.truth = computeGroundTruth(caseDef, this.tables);
    this.syncBeatTables(false);
    this.consoleState = createConsoleState(caseDef.seed);
    this.caseArtifacts = caseArtifactLevels(caseDef);
    this.applyArtifactOverrides(input.artifactOverrides);
    const createGpu = options.createGpu ?? createWebgl2Renderer;
    this.gpuRecovery = new GpuRecovery(createGpu, options.now ?? (() => performance.now()));
    const g = createGpu();
    this.gpu = g.renderer;
    this.gpuReason = g.reason;
    this.atlas = new AtlasRenderer(this.gpu ?? this.procedural, caseDef.seed);
    this.backend = this.pickBackend(input.rendererBackend);
    this.flow = this.buildFlow();
    this.rememberBeat();
  }

  private applyArtifactOverrides(o: SimInput['artifactOverrides']): void {
    this.artifactLevels = o ?? this.caseArtifacts;
    this.artifacts = consoleArtifacts(this.artifactLevels);
  }

  private pickBackend(
    kind: SimInput['rendererBackend'],
    anatomy: SimInput['anatomy'] = this.input.anatomy,
  ): RendererBackend {
    if (kind === 'procedural' || (anatomy === 'rodero' && !this.gpu)) return this.procedural;
    if (kind === 'webgl2') return this.gpu ?? this.procedural;
    return this.atlas;
  }

  private syncBeatTables(rebuildFlow = true): void {
    const c = this.clock.current;
    const breathing = this.input.patient.respiration === 'free-breathing';
    if (
      (this.caseDef.rhythm.type !== 'atrial-fibrillation' && !breathing) ||
      c.beatIndex === this.tablesBeat
    )
      return;
    this.tables = buildChainedBeatTables(
      this.caseDef,
      c,
      this.timeS,
      breathing,
      this.tables,
      this.nominalTables,
    );
    this.tablesBeat = c.beatIndex;
    this.tablesVersion++;
    if (rebuildFlow)
      this.flow = buildFlowParams(this.caseDef, this.heart, this.tables, this.thorax);
  }

  private buildFlow(): FlowFieldParams {
    heartLandmarks(this.heart);
    computeHeartPose(this.heart, cycleStateAt(this.tables, 0));
    return buildFlowParams(this.caseDef, this.heart, this.tables, this.thorax);
  }

  private rememberBeat(): void {
    this.beatHistory.remember(this.clock.current, {
      tables: this.tables,
      flow: this.flow,
      tablesVersion: this.tablesVersion,
      heart: this.heart,
      thorax: this.thorax,
    });
  }

  private stripCtx(): StripCtx {
    const c = this.clock.current;
    return {
      input: this.input,
      caseDef: this.caseDef,
      heart: this.heart,
      tables: this.tables,
      flow: this.flow,
      timeS: this.timeS,
      modelVersion: this.modelVersion,
      tablesVersion: this.tablesVersion,
      phaseNow: c.phase,
      rrS: c.rrS,
      procedural: this.procedural,
      moment: (timeS) => {
        const beat = this.beatHistory.at(timeS),
          bc = beat.context;
        if (
          this.poseCacheHeart !== bc.heart ||
          this.poseCacheTablesVersion !== bc.tablesVersion
        ) {
          this.poseCacheHeart = bc.heart;
          this.poseCacheTablesVersion = bc.tablesVersion;
          this.poseCacheBins.fill(undefined);
        }
        const phaseBin = Math.floor((((beat.phase % 1) + 1) % 1) * 256) & 255;
        let heartPose = this.poseCacheBins[phaseBin];
        if (!heartPose) {
          const binPhase = (phaseBin + 0.5) / 256;
          heartPose = computeHeartPose(
            bc.heart,
            cycleStateAt(bc.tables, binPhase, bc.flow.flowPeaks),
          );
          this.poseCacheBins[phaseBin] = heartPose;
        }
        // The column's own instant, as the 2D would see it then. With the beat's static thorax the spectrum kept the
        // end-expiratory window all breath long (PW at 1.62 s: lung border −1.60 cm and diaphragm 0 against the 2D's
        // −0.45 and −0.50), so a lung that shadowed the 2D and M-mode in inspiration left the spectrum intact.
        return {
          phase: beat.phase,
          beatIndex: beat.beatIndex,
          tablesVersion: bc.tablesVersion,
          tables: bc.tables,
          flow: bc.flow,
          scene: this.sceneAt(bc.heart, heartPose, bc.thorax, timeS),
        };
      },
      scene: (ph) => this.scene(ph),
      physics: () => this.physics(),
    };
  }

  setInput(input: SimInput): void {
    const key = patientKey(input.patient);
    if (key !== this.patientKey) {
      const models = buildCaseModels(this.caseDef, input.patient);
      this.thorax = models.thorax;
      this.heart = models.heart;
      if (
        input.patient.respiration !== this.input.patient.respiration &&
        this.caseDef.rhythm.type !== 'atrial-fibrillation'
      ) {
        this.tables = this.nominalTables;
        this.tablesBeat = -1;
        this.tablesVersion++;
      }
      this.flow = this.buildFlow();
      this.atlas.invalidate();
      this.patientKey = key;
      this.modelVersion++;
    }
    const anatomyChanged = (input.anatomy ?? 'analytic') !== (this.input.anatomy ?? 'analytic');
    if (input.rendererBackend !== this.input.rendererBackend || anatomyChanged) {
      this.backend = this.pickBackend(input.rendererBackend, input.anatomy ?? 'analytic');
      if (anatomyChanged) {
        // another heart: a new acquisition, or the strip's caches keyed by ray (anatomy is not in the key) kept serving
        // CW columns traced through the other one
        this.atlas.invalidate();
        this.strips.reset();
      }
    }
    if ((input.imageFormation ?? 'lattice') !== (this.input.imageFormation ?? 'lattice'))
      this.atlas.invalidate();
    const nextArtifactKey = artifactOverridesKey(input.artifactOverrides);
    if (nextArtifactKey !== this.artifactKey) {
      this.artifactKey = nextArtifactKey;
      this.applyArtifactOverrides(input.artifactOverrides);
    }
    const sessionEpoch = input.sessionEpoch ?? 0;
    const newSession = sessionEpoch !== this.sessionEpoch;
    if (newSession) {
      // a new session (an exam started on the case in practice, decision 235) acquires from scratch: the view analysed
      // in the last one was still attached to the next frames and scored in the new one (A4C 97 the instant an exam
      // began), and its cine and strips were there to scroll back to and measure
      this.sessionEpoch = sessionEpoch;
      this.lastAnalysis = null;
      this.analysisCounter = 0;
      this.cine.clear();
    }
    if (newSession || (!input.frozen && input.modality !== this.acquisitionModality)) {
      this.strips.reset();
      this.colorDriver.resetHistory();
      this.gpu?.resetDopplerHistory();
      this.dopplerOnGpu = false;
      this.acquisitionModality = input.modality;
    }
    this.input = constrainAcquisition(input);
  }

  recycle(buffer: ArrayBuffer): void {
    this.compositor.recycle(buffer);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.gpu?.dispose();
    this.gpu = null;
    this.gpuRecovery.dispose();
    this.atlas.invalidate();
    this.compositor.clear();
  }

  get isDisposed(): boolean {
    return this.disposed;
  }

  /** Advance simulation by dt seconds. Returns an output when a new composite frame is ready. */
  step(dtS: number): SimOutput | null {
    if (this.disposed) return null;
    const tStepStart = performance.now();
    const inp = this.input;
    const dt = Math.min(0.1, Math.max(0, dtS));
    if (inp.frozen) return this.frozenOutput();
    this.strips.resetStepTiming();
    this.lastColorStepMs = 0;
    // A context lost since the last step is seen here, before the tier is resolved: the CPU tracer's first frame is formed
    // at medium («auto»), not at the calibrated tier (about 40 ms instead of 170 in Node; about a second in a browser, as
    // quality.ts says). The port comes back here too, so the frame of the return is at the calibrated tier, and when the
    // browser has restored the context by now the loss costs no CPU frame at all. Nothing to do on a machine without it.
    if (this.gpu?.contextLost) this.dropGpu('WebGL context lost: CPU tracer');
    if (!this.gpu) this.recoverGpu();
    const tier = resolveQualityTier(
      inp.quality,
      this.gpu !== null && this.backend !== this.procedural,
    );
    this.tier = tier;
    const spec = polarSpecFor(inp.settings, tier);
    const isStrip = isStripModality(inp.modality);
    const colorLines =
      inp.modality === 'color'
        ? Math.round(
            ((inp.color.boxThetaMaxRad - inp.color.boxThetaMinRad) / spec.sectorRad) * spec.lines,
          )
        : 0;
    const acquisitionFps = this.acquisitionFps();
    const fps = cadenceHz(spec, acquisitionFps, colorLines);
    this.colorFps = inp.modality === 'color' ? acquisitionFps : 0;
    const beam = beamFrameFromPose(
      poseFromControl(this.thorax, inp.probe),
      contactQuality(inp.probe.pressure),
    );
    const pre = this.clock.current;
    const previousTablesVersion = this.tablesVersion;
    this.rememberBeat();
    this.clock.advance(dt);
    this.timeS += dt;
    this.syncBeatTables();
    this.rememberBeat();
    if (this.tablesVersion !== previousTablesVersion) this.atlas.invalidate();
    this.ecgRing.advance(
      pre.timeInBeatS,
      pre.rrS,
      dt,
      this.timeS,
      this.cine.oldest?.timeS,
      this.caseDef.rhythm,
      this.clock.current.beatIndex,
      this.caseDef.seed,
    );
    if (isStrip)
      this.strips.advance(dt, beam, spec, (MMODE_TRACE_SHARE * 1000) / fps, this.stripCtx());
    this.frameAccumulator += dt;
    const frameInterval = 1 / fps;
    this.frameIntervalS = frameInterval;
    this.frameBudgetMs = 600 * frameInterval;
    let produced = false;
    let bmodeRenderMs = 0;
    if (this.frameAccumulator >= frameInterval * 0.85 || !this.frame) {
      this.frameAccumulator = Math.min(
        frameInterval,
        Math.max(0, this.frameAccumulator - frameInterval),
      );
      const gateBefore = this.strips.lastStepTiming.gateInfoMs;
      const t0 = performance.now();
      this.renderFrame(beam, spec);
      this.timing.renderFrameMs = performance.now() - t0;
      const gateInRender = this.strips.lastStepTiming.gateInfoMs - gateBefore;
      bmodeRenderMs = Math.max(0, this.timing.renderFrameMs - this.lastColorStepMs - gateInRender);
      produced = true;
    }
    // Independent Deferred Doppler Re-shading: if Doppler-only console settings changed between
    // B-mode acquisition ticks, re-shade Doppler directly on GPU against the resident Polar G-Buffer
    // and re-present without re-running B-mode anatomical raymarching.
    let dopplerReshaded = false;
    if (
      !produced &&
      !isStrip &&
      inp.modality === 'color' &&
      this.displayOnGpu &&
      this.dopplerOnGpu &&
      this.gpu?.gbuffer.ready &&
      this.frame
    ) {
      const acquiredPrev = this.cine.newest;
      const phaseUse = acquiredPrev ? acquiredPrev.phase : this.clock.current.phase;
      const beamUse = acquiredPrev ? acquiredPrev.beam : beam;
      const specUse = acquiredPrev ? acquiredPrev.spec : spec;
      const sceneUse = acquiredPrev ? acquiredPrev.scene : this.scene(phaseUse);
      const tReshade = performance.now();
      dopplerReshaded = this.colorDriver.reshadeDeferredGpu(
        {
          frame: this.frame,
          scene: sceneUse,
          // the acquired frame's own beat: a step that crosses a beat boundary without forming a frame has already
          // swapped the live tables and flow for the next beat's
          heart: sceneUse.heart,
          flow: acquiredPrev?.flow ?? this.flow,
          tables: acquiredPrev?.tables ?? this.tables,
          beam: beamUse,
          spec: specUse,
          phase: phaseUse,
          probe: inp.probe,
          color: inp.color,
          frequencyMHz: inp.settings.frequencyMHz,
          harmonics: inp.settings.harmonics,
          modelVersion: this.modelVersion,
          timeS: this.timeS,
          frameIntervalS: this.frameIntervalS,
          seed: this.caseDef.seed,
        },
        this.gpu,
      );
      if (dopplerReshaded) this.lastColorStepMs += performance.now() - tReshade;
      if (this.gpu.contextLost) this.dropGpu('WebGL context lost: CPU tracer');
    }
    if (!produced && !isStrip && !dopplerReshaded) return null;
    const t1 = performance.now();
    const acquired = this.cine.newest;
    const out = this.composite(acquired?.beam ?? beam, acquired?.spec ?? spec, false, acquired);
    this.timing.compositeMs = performance.now() - t1;
    if (out && isDopplerModality(inp.modality)) {
      const totalMs = performance.now() - tStepStart;
      const colorDopplerMs =
        this.lastColorStepMs + this.timing.overlayColorMs + this.strips.lastCmmColorMs;
      out.dopplerPerf = buildDopplerPerfFrame({
        frameId: out.frameId,
        modality: inp.modality,
        totalMs,
        spectral: this.strips.lastStepTiming,
        colorDopplerMs,
        bmodeRenderMs,
      });
    }
    return out;
  }

  private acquisitionFps(): number {
    const inp = this.input;
    return acquisitionFrameRate(
      inp.settings,
      inp.modality === 'color' ? inp.color : undefined,
      isSpectralModality(inp.modality),
    );
  }

  /**
   * The port's context was lost: the frames go to the CPU tracer, and the lost renderer to the recovery, which disposes it
   * once replaced or given up. Before, it was only dropped: never disposed (its listener stayed on the canvas) and never
   * rebuilt, so the session kept the CPU tracer, at medium quality, until the case was reloaded.
   */
  private dropGpu(reason: string): void {
    if (this.gpu) this.gpuRecovery.lost(this.gpu);
    this.gpu = null;
    this.gpuReason = this.gpuRecovery.reason ?? reason;
    this.switchBackendSource(this.procedural);
  }

  /** The port is back: a new atlas over it, the calibrated tier from this step on, and the HUD's warning gone. */
  private recoverGpu(): void {
    const renderer = this.gpuRecovery.poll();
    if (!renderer) {
      this.gpuReason = this.gpuRecovery.reason ?? this.gpuReason;
      return;
    }
    this.gpu = renderer;
    this.gpuReason = 'ok';
    this.switchBackendSource(renderer);
  }

  /**
   * The image source changes between the port and the CPU tracer: a fresh atlas over it (no frame cached from the other
   * one), and a colour field that starts again. The deferred Doppler's phasor history died with the lost context, and the
   * CPU field still held whatever it had before the GPU took over: shown as it was on the frames it is not recomputed
   * (every other one), and blended by persistence into the next.
   */
  private switchBackendSource(source: RendererBackend): void {
    this.atlas = new AtlasRenderer(source, this.caseDef.seed);
    this.backend = this.pickBackend(this.input.rendererBackend);
    this.displayOnGpu = false;
    this.dopplerOnGpu = false;
    this.colorDriver.resetHistory();
  }

  private physics(): ScenePhysics {
    return scenePhysicsFor(this.caseDef, this.input.settings, {
      artifacts: this.artifactLevels,
      bloodFrame: this.consoleState.frameIndex,
    });
  }

  /**
   * The thorax at `timeS`: in free breathing the lung border slides medially by up to 1.15 cm and the diaphragm descends
   * 0.5 cm with the depth of inspiration; otherwise `base`, the patient's static thorax. `base` is the thorax of the beat
   * the instant belongs to, since the strips acquire up to a step back.
   */
  private thoraxAtTime(timeS: number, base = this.thorax): ThoraxModel {
    if (this.input.patient.respiration !== 'free-breathing') return base;
    const d = respiratoryDepth(timeS);
    return {
      ...base,
      lungShiftCm: base.lungShiftCm + 1.15 * d,
      diaphragmRiseCm: base.diaphragmRiseCm - 0.5 * d,
    };
  }

  /** The scene of the frames now: the current beat's heart at `phase`. */
  private scene(phase: number): Scene {
    const state = cycleStateAt(this.tables, phase, this.flow.flowPeaks);
    return this.sceneAt(this.heart, computeHeartPose(this.heart, state), this.thorax, this.timeS);
  }

  /**
   * The scene a beat's heart in `heartPose` is acquired in at `timeS`: the one recipe of the 2D, colour and M-mode frames
   * (`scene`) and of every spectral column (`moment`), so the thorax breathes and the experimental anatomy and image
   * formation apply alike in all of them.
   */
  private sceneAt(
    heart: HeartModel,
    heartPose: HeartPose,
    thorax: ThoraxModel,
    timeS: number,
  ): Scene {
    const rodero = this.experimental.roderoHeart(this.input.anatomy, this.heart, this.tables);
    const cole = this.experimental.coleImage(this.input.imageFormation);
    return {
      heart,
      heartPose,
      thorax: this.thoraxAtTime(timeS, thorax),
      physics: this.physics(),
      ...(cole ? { imageFormation: cole.COLE_FORMATION, scatterGpu: cole.COLE_GPU } : {}),
      // The volume heart is bound to the current heart: a column of the beat before a change of patient (at most a step
      // back) samples that beat's analytic heart, as the frames do while the volume loads, instead of rebinding it.
      ...(rodero && rodero.heart === heart
        ? {
            heartClassifier: rodero.vh.classifier({
              apparatus: heartPose,
              motion: { pose: heartPose, endDiastole: rodero.endDiastole },
            }),
            heartVolumeGpu: rodero.gpu,
          }
        : {}),
    };
  }

  async coleReady(): Promise<void> {
    await this.experimental.coleReady(this.input.imageFormation);
  }

  async roderoReady(): Promise<void> {
    await this.experimental.roderoReady(this.input.anatomy, this.heart, this.tables);
  }

  private renderFrame(beam: BeamFrame, spec: PolarFrameSpec): void {
    const inp = this.input;
    if (
      !this.frame ||
      this.frame.spec.lines !== spec.lines ||
      this.frame.spec.samples !== spec.samples ||
      this.frame.spec.depthCm !== spec.depthCm ||
      this.frame.spec.sectorRad !== spec.sectorRad
    ) {
      this.frame = allocPolarFrame(spec);
      this.display = new Uint8ClampedArray(spec.lines * spec.samples);
      this.consoleState = createConsoleState(this.caseDef.seed);
      this.colorDriver.resize(spec.lines * spec.samples);
      this.lastFrameTimeS = NaN;
    }
    this.frame.spec = spec;
    const elapsed = Number.isFinite(this.lastFrameTimeS)
      ? this.timeS - this.lastFrameTimeS
      : this.frameIntervalS;
    this.lastFrameTimeS = this.timeS;
    this.lastPersistence = persistenceOverTime(
      inp.settings.persistence,
      elapsed,
      this.frameIntervalS,
    );
    const settings = { ...inp.settings, persistence: this.lastPersistence };
    const phase = this.clock.current.phase;
    const scene = this.scene(phase);
    const moved = this.prevBeam ? AtlasBackend.poseDistance(this.prevBeam, beam) : Infinity;
    this.stationaryFrames = moved < 0.03 ? this.stationaryFrames + 1 : 0;
    this.prevBeam = beam;
    const breathing = inp.patient.respiration === 'free-breathing';
    if (breathing) this.atlas.invalidate();
    const hints: RenderHints = {
      stationary: !breathing && this.stationaryFrames >= 6,
      budgetMs: this.frameBudgetMs,
      sceneAtPhase: (ph) => this.scene(ph),
    };
    const needSegments = (inp.needSegments ?? true) || (this.analysisCounter + 1) % 5 === 1;
    if (this.gpu?.contextLost) this.dropGpu('WebGL context lost: CPU tracer');
    let onGpu = this.formDisplay(scene, beam, spec, phase, hints, settings, needSegments);
    if (this.gpu?.contextLost) {
      this.dropGpu('WebGL context lost: CPU tracer');
      onGpu = this.formDisplay(scene, beam, spec, phase, hints, settings, needSegments);
    }
    this.displayOnGpu = onGpu;
    let colorVel: Float32Array | null = null;
    let colorVar: Float32Array | null = null;
    let colorPower: Float32Array | null = null;
    let colorTissue: Uint8Array | null = null;
    let colorTransCode: Uint8Array | null = null;
    if (inp.modality === 'color') {
      const tColor = performance.now();
      const colorCtx = {
        frame: this.frame,
        scene,
        heart: this.heart,
        flow: this.flow,
        tables: this.tables,
        beam,
        spec,
        phase,
        probe: inp.probe,
        color: inp.color,
        frequencyMHz: inp.settings.frequencyMHz,
        harmonics: inp.settings.harmonics,
        modelVersion: this.modelVersion,
        timeS: this.timeS,
        frameIntervalS: this.frameIntervalS,
        seed: this.caseDef.seed,
      };
      if (onGpu && this.gpu && this.gpu.gbuffer.ready) {
        const ok = this.colorDriver.stepDeferredGpu(colorCtx, this.gpu);
        if (this.gpu.contextLost) this.dropGpu('WebGL context lost: CPU tracer');
        if (ok && this.gpu && this.gpu.hasDeferredDoppler) {
          this.dopplerOnGpu = true;
          const n = spec.lines * spec.samples;
          if (this.transCodeScratch.length !== n) this.transCodeScratch = new Uint8Array(n);
          const tr = this.frame.transmission;
          for (let i = 0; i < n; i++) this.transCodeScratch[i] = encodeTransmission(tr[i] ?? 0);
          colorTissue = this.frame.tissue;
          colorTransCode = this.transCodeScratch;
        } else {
          this.dopplerOnGpu = false;
          const cOut = this.colorDriver.step(colorCtx);
          colorVel = cOut.vel;
          colorVar = cOut.variance;
          colorPower = cOut.power;
        }
      } else {
        this.dopplerOnGpu = false;
        const cOut = this.colorDriver.step(colorCtx);
        colorVel = cOut.vel;
        colorVar = cOut.variance;
        colorPower = cOut.power;
      }
      this.lastColorStepMs += performance.now() - tColor;
    } else {
      this.dopplerOnGpu = false;
    }
    this.analysisCounter++;
    if (this.analysisCounter % 5 === 1) {
      const ta = performance.now();
      this.lastAnalysis = {
        ...analyzeView({
          heart: this.heart,
          // The view's reference pose is the patient's at rest, the thorax the presets are solved against; the breath
          // reaches the analysis through the frame, whose transmission carries the lung sliding over the window. Given
          // the frame's thorax, rebuilt every frame in free breathing, each analysis solved the reference poses again
          // (A4C 29 ms on average, suprasternal 30 ms; 0.37 and 0.30 ms now, as in expiration) and the reference moved
          // with the breath: past half an inspiration the apical window left the axis, and the A4C preset, held still,
          // scored 68-97 instead of 87-97, up to 4.8° and 1.2 cm off its own reference.
          thorax: this.thorax,
          control: inp.probe,
          beam,
          frame: this.frame,
          display: this.display,
          settings: inp.settings,
          heartPose: scene.heartPose,
        }),
        // the next frames carry this analysis too, possibly at other poses: the score is this pose's
        probe: { ...inp.probe },
      };
      this.timing.analysisMs = performance.now() - ta;
    }
    const tCine = performance.now();
    this.cine.capture({
      acquisition: {
        modality: inp.modality,
        probe: inp.probe,
        patient: inp.patient,
        settings: inp.settings,
        color: inp.color,
        spectral: inp.spectral,
        cursorThetaRad: inp.cursorThetaRad,
        gateDepthCm: inp.gateDepthCm,
        modelVersion: this.modelVersion,
        tablesVersion: this.tablesVersion,
        phaseMarks: this.phaseMarks(),
        stripAcquisitionId: isStripModality(inp.modality) ? this.strips.historyId : null,
        colorTimeS:
          inp.modality === 'color' && Number.isFinite(this.colorDriver.lastTimeS)
            ? this.colorDriver.lastTimeS
            : null,
        caseId: this.caseDef.id,
        sessionEpoch: this.sessionEpoch,
      },
      scene,
      tables: this.tables,
      flow: this.flow,
      gate: isStripModality(inp.modality)
        ? this.strips.gateInfo(beam, spec, phase, this.frame.structure, this.stripCtx())
        : null,
      spectrumColumn: this.strips.spectrumColumn,
      spectralRange: this.strips.velocityRange ?? spectralRange(inp.spectral),
      frameId: this.frameId + 1,
      rrS: this.clock.current.rrS,
      structure: this.frame.structure,
      segment: this.frame.segment,
      display: this.display!,
      spec,
      phase,
      timeS: this.timeS,
      beatIndex: this.clock.current.beatIndex,
      colorVel,
      colorVar,
      colorPower,
      colorTissue,
      colorTransCode,
      beam,
      analysis: this.lastAnalysis,
    });
    this.timing.cineMs = performance.now() - tCine;
    this.frameId++;
  }

  private formDisplay(
    scene: Scene,
    beam: BeamFrame,
    spec: PolarFrameSpec,
    phase: number,
    hints: RenderHints,
    settings: AcquisitionSettings,
    needSegments: boolean,
  ): boolean {
    const frame = this.frame!;
    const display = this.display!;
    if (
      this.artifacts.mirror <= 0 &&
      this.backend.renderDisplay?.(
        scene,
        beam,
        spec,
        phase,
        frame,
        hints,
        { settings, state: this.consoleState, needSegments },
        display,
      ) === true
    ) {
      this.timing.consoleMs = 0;
      return true;
    }
    this.backend.render(scene, beam, spec, phase, frame, hints);
    const tc = performance.now();
    if (this.consoleState.gpuHistory) this.gpu?.restoreCpuHistory(this.consoleState, spec);
    applyConsole(frame, settings, this.consoleState, display, this.artifacts);
    this.timing.consoleMs = performance.now() - tc;
    return false;
  }

  /**
   * Dev hook (`gpuContext` request): lose the port's context, or restore the one it lost, through WEBGL_lose_context, as
   * the browser does on a suspend or a GPU process restart. A context lost this way comes back only on 'restore'; left
   * lost, it exercises the recovery's retries on a new canvas.
   */
  private forceGpuContext(action: 'lose' | 'restore'): SimResponse {
    if (action === 'lose') {
      this.devLostContext = this.gpu?.loseContextExtension() ?? null;
      this.devLostContext?.loseContext();
      return { kind: 'gpuContext', ok: this.devLostContext !== null };
    }
    const ext = this.devLostContext;
    this.devLostContext = null;
    ext?.restoreContext();
    return { kind: 'gpuContext', ok: ext !== null };
  }

  /** Cardiac phase landmarks as fractions of RR (for phase checks in the technique engine). */
  phaseMarks(): PhaseMarks {
    return computePhaseMarks(this.tables);
  }

  /** LV length at end-diastole (cm), the reference for Simpson foreshortening checks. */
  lvLengthCm(): number {
    return this.heart.lv.lengthCm;
  }

  /** Answer an on-demand request (auto-trace, canonical control, or the model at a point of the image). */
  request(req: SimRequest): SimResponse | null {
    if (req.kind === 'gpuContext') return this.forceGpuContext(req.action);
    return routeSimRequest(req, {
      input: this.input,
      displayedFrame: this.displayedFrame,
      display: this.display,
      frame: this.frame,
      heart: this.heart,
      thorax: this.thorax,
      tables: this.tables,
      experimental: this.experimental,
      mirrorLevel: this.artifactLevels.mirror,
      cine: this.cine,
      strips: this.strips,
      phaseMarks: () => this.phaseMarks(),
      getExplainTracer: () => {
        this.explainTracer ??= new ProceduralSliceRenderer();
        return this.explainTracer;
      },
    });
  }

  private frozenOutput(): SimOutput | null {
    const cf = this.cine.select(this.input.cineOffset);
    if (!cf) return null;
    // a frame acquired on the GPU deferred path keeps no colour field (zero readback): form it on its first frozen display
    if (cf.acquisition.modality === 'color' && !cf.colorVel && cf.colorTissue && cf.colorTransCode) {
      // the field the GPU was presenting belongs to the newest frame, and nothing re-shades it while frozen: freeze that
      // frame as it was seen; re-estimate any other from its own beat
      const asSeen =
        cf === this.cine.newest &&
        this.dopplerOnGpu &&
        this.gpu !== null &&
        this.colorDriver.readbackDeferredGpu(cf, this.gpu);
      if (!asSeen) this.colorDriver.reconstructCineColor(cf, this.caseDef.seed);
    }
    return this.composite(cf.beam, cf.spec, true, cf);
  }

  private composite(
    beam: BeamFrame,
    spec: PolarFrameSpec,
    frozen: boolean,
    cf?: CineFrame,
  ): SimOutput | null {
    const res = this.compositor.composite({
      input: this.input,
      beam,
      spec,
      frozen,
      cf,
      display: this.display,
      frame: this.frame,
      displayOnGpu: this.displayOnGpu,
      dopplerOnGpu: this.dopplerOnGpu,
      gpu: this.gpu,
      gpuReason: this.gpuReason,
      onGpuContextLost: (reason) => this.dropGpu(reason),
      colorCurrent: this.colorDriver.current,
      colorVersion: this.colorDriver.version,
      colorFps: this.colorFps,
      strips: this.strips,
      stripCtx: () => this.stripCtx(),
      lastAnalysis: this.lastAnalysis,
      clockCurrent: this.clock.current,
      timeS: this.timeS,
      frameId: this.frameId,
      frameIntervalS: this.frameIntervalS,
      acquisitionFps: this.acquisitionFps(),
      ecgRing: this.ecgRing,
      cine: this.cine,
      backend: this.backend,
      isProceduralBackend: this.backend === this.procedural,
      experimental: this.experimental,
      tier: this.tier,
      lastPersistence: this.lastPersistence,
      timing: this.timing,
    });
    if (!res) return null;
    this.lastOutputBeam = beam;
    this.lastSector = res.sector;
    this.lastStrip = res.strip;
    this.displayedFrame = res.displayedFrame;
    return res.output;
  }

  // ---- accessors for tests / devtools ----
  get models(): { heart: HeartModel; thorax: ThoraxModel; tables: BeatTables } {
    return { heart: this.heart, thorax: this.thorax, tables: this.tables };
  }
  get mmodeStrip(): StripEngine['mmodeStrip'] {
    return this.strips.mmodeStrip;
  }
  get lastView(): ViewAnalysis | null {
    return this.lastAnalysis;
  }
  get lastBeam(): BeamFrame | null {
    return this.lastOutputBeam;
  }
  get lastFrame(): PolarFrame | null {
    return this.frame;
  }
  get lastColorField(): { field: ColorField | null; version: number } {
    return { field: this.colorDriver.current, version: this.colorDriver.version };
  }
  get currentPhase(): number {
    return this.clock.current.phase;
  }
  get lastSectorMapping(): (SectorMapping & { x: number; y: number }) | null {
    return this.lastSector;
  }
  get lastStripInfo(): StripInfo | null {
    return this.lastStrip;
  }
  get spectralStrip(): StripEngine['spectralStrip'] {
    return this.strips.spectralStrip;
  }
  heartPoseNow(): HeartPose {
    return computeHeartPose(
      this.heart,
      cycleStateAt(this.tables, this.clock.current.phase, this.flow.flowPeaks),
    );
  }
}

/** Options of a simulator core: how it fetches the app's assets (decision 352) and builds its GPU port. */
export interface SimulatorCoreOptions {
  fetchAsset?: AssetFetcher;
  /** Builds the GPU port, at start and after a lost context; `createWebgl2Renderer` by default (tests simulate a GL). */
  createGpu?: GpuFactory;
  /** Wall clock (ms) the GPU recovery spaces its attempts by; `performance.now` by default. */
  now?: () => number;
}

/** The app's assets, fetched under its base URL (the worker's and the page's). */
const fetchFromAppBase: AssetFetcher = async (path) => {
  const r = await fetch(`${import.meta.env.BASE_URL}${path}`);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.arrayBuffer();
};
