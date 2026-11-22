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
import { overlayColorField, type ColorField } from '@/simulator/doppler/color/colorDoppler';
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
import {
  buildScanLut,
  computeSectorMapping,
  lutKey,
  scanConvertLut,
  type ScanLut,
  type SectorMapping,
} from '@/simulator/renderer/scanConvert';
import {
  caseArtifactLevels,
  consoleArtifacts,
  scenePhysicsFor,
  type ArtifactLevels,
} from '@/simulator/renderer/scenePhysics';
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
import { buildChainedBeatTables, computePhaseMarks } from './beatChaining';
import { BeatHistory } from './beatHistory';
import { CineBuffer, type CineFrame } from './cineBuffer';
import { ColorFieldDriver } from './colorFieldDriver';
import { EcgRing } from './ecgRing';
import { ExperimentalLoaders } from './experimentalLoaders';
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
  private cine = new CineBuffer();
  private frameId = 0;
  private timeS = 0;
  private frameAccumulator = 0;
  private frameIntervalS = 1 / 30;
  private lastFrameTimeS = NaN;
  private lastPersistence = 0;
  private lastAnalysis: ViewAnalysis | null = null;
  private analysisCounter = 0;
  private ecgRing = new EcgRing();
  private strips = new StripEngine();
  private patientKey = '';
  private rgbaPool: ArrayBuffer[] = [];
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
  };
  private lut: ScanLut | null = null;
  private prevBeam: BeamFrame | null = null;
  private stationaryFrames = 0;
  private frameBudgetMs = 25;
  private disposed = false;

  constructor(caseDef: CaseDefinition, input: SimInput, options: SimulatorCoreOptions = {}) {
    this.caseDef = caseDef;
    this.experimental = new ExperimentalLoaders(options.fetchAsset ?? fetchFromAppBase, () =>
      this.atlas.invalidate(),
    );
    this.input = constrainAcquisition(input);
    this.acquisitionModality = input.modality;
    const models = buildCaseModels(caseDef, input.patient);
    this.thorax = models.thorax;
    this.heart = models.heart;
    this.tables = models.tables;
    this.patientKey = JSON.stringify(input.patient);
    this.nominalTables = this.tables;
    this.clock = new CardiacClock(caseDef.rhythm, caseDef.seed);
    this.truth = computeGroundTruth(caseDef, this.tables);
    this.syncBeatTables(false);
    this.consoleState = createConsoleState(caseDef.seed);
    this.caseArtifacts = caseArtifactLevels(caseDef);
    this.applyArtifactOverrides(input.artifactOverrides);
    const g = createWebgl2Renderer();
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
        return {
          phase: beat.phase,
          beatIndex: beat.beatIndex,
          tablesVersion: bc.tablesVersion,
          tables: bc.tables,
          flow: bc.flow,
          scene: {
            heart: bc.heart,
            thorax: bc.thorax,
            heartPose: computeHeartPose(
              bc.heart,
              cycleStateAt(bc.tables, beat.phase, bc.flow.flowPeaks),
            ),
            physics: this.physics(),
          },
        };
      },
      scene: (ph) => this.scene(ph),
      physics: () => this.physics(),
    };
  }

  setInput(input: SimInput): void {
    const key = JSON.stringify(input.patient);
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
      if (anatomyChanged) this.atlas.invalidate();
    }
    if ((input.imageFormation ?? 'lattice') !== (this.input.imageFormation ?? 'lattice'))
      this.atlas.invalidate();
    if (JSON.stringify(input.artifactOverrides) !== JSON.stringify(this.input.artifactOverrides))
      this.applyArtifactOverrides(input.artifactOverrides);
    if (!input.frozen && input.modality !== this.acquisitionModality) {
      this.strips.reset();
      this.colorDriver.resetHistory();
      this.acquisitionModality = input.modality;
    }
    this.input = constrainAcquisition(input);
  }

  recycle(buffer: ArrayBuffer): void {
    if (buffer.byteLength > 0 && this.rgbaPool.length < 3) this.rgbaPool.push(buffer);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.gpu?.dispose();
    this.gpu = null;
    this.atlas.invalidate();
    this.rgbaPool.length = 0;
  }

  get isDisposed(): boolean {
    return this.disposed;
  }

  /** Advance simulation by dt seconds. Returns an output when a new composite frame is ready. */
  step(dtS: number): SimOutput | null {
    if (this.disposed) return null;
    const inp = this.input;
    const dt = Math.min(0.1, Math.max(0, dtS));
    if (inp.frozen) return this.frozenOutput();
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
    if (this.frameAccumulator >= frameInterval * 0.85 || !this.frame) {
      this.frameAccumulator = Math.min(
        frameInterval,
        Math.max(0, this.frameAccumulator - frameInterval),
      );
      const t0 = performance.now();
      this.renderFrame(beam, spec);
      this.timing.renderFrameMs = performance.now() - t0;
      produced = true;
    }
    if (!produced && !isStrip) return null;
    const t1 = performance.now();
    const acquired = this.cine.newest;
    const out = this.composite(acquired?.beam ?? beam, acquired?.spec ?? spec, false, acquired);
    this.timing.compositeMs = performance.now() - t1;
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

  private dropGpu(reason: string): void {
    this.gpu = null;
    this.gpuReason = reason;
    this.atlas = new AtlasRenderer(this.procedural, this.caseDef.seed);
    this.backend = this.pickBackend(this.input.rendererBackend);
    this.displayOnGpu = false;
  }

  private physics(): ScenePhysics {
    return scenePhysicsFor(this.caseDef, this.input.settings, {
      artifacts: this.artifactLevels,
      bloodFrame: this.consoleState.frameIndex,
    });
  }

  private scene(phase: number): Scene {
    const state = cycleStateAt(this.tables, phase, this.flow.flowPeaks);
    const heartPose = computeHeartPose(this.heart, state);
    const rodero = this.experimental.roderoHeart(this.input.anatomy, this.heart, this.tables);
    const cole = this.experimental.coleImage(this.input.imageFormation);
    return {
      heart: this.heart,
      heartPose,
      thorax: this.thorax,
      physics: this.physics(),
      ...(cole ? { imageFormation: cole.COLE_FORMATION, scatterGpu: cole.COLE_GPU } : {}),
      ...(rodero
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
    const hints: RenderHints = {
      stationary: this.stationaryFrames >= 6,
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
    if (inp.modality === 'color') {
      const cOut = this.colorDriver.step({
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
      });
      colorVel = cOut.vel;
      colorVar = cOut.variance;
    }
    this.analysisCounter++;
    if (this.analysisCounter % 5 === 1) {
      const ta = performance.now();
      this.lastAnalysis = analyzeView({
        heart: this.heart,
        thorax: this.thorax,
        control: inp.probe,
        beam,
        frame: this.frame,
        display: this.display,
        settings: inp.settings,
        heartPose: scene.heartPose,
      });
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
      },
      scene,
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
    return this.composite(cf.beam, cf.spec, true, cf);
  }

  private takeBuffer(size: number): ArrayBuffer {
    for (let i = 0; i < this.rgbaPool.length; i++) {
      const b = this.rgbaPool[i]!;
      if (b.byteLength === size) {
        this.rgbaPool.splice(i, 1);
        return b;
      }
    }
    return new ArrayBuffer(size);
  }

  private composite(
    beam: BeamFrame,
    spec: PolarFrameSpec,
    frozen: boolean,
    cf?: CineFrame,
  ): SimOutput | null {
    const inp = this.input;
    const W = Math.max(64, inp.display.width);
    const H = Math.max(64, inp.display.height);
    const modality = cf?.acquisition.modality ?? inp.modality;
    const colorSettings = cf?.acquisition.color ?? inp.color;
    const isStrip = isStripModality(modality);
    const sectorH = isStrip ? Math.round(H * 0.42) : H;
    const display = cf ? cf.display : this.display;
    const fspec = cf ? cf.spec : spec;
    if (!display) return null;
    const mapping = computeSectorMapping(
      fspec,
      W,
      sectorH,
      inp.settings.invertLR,
      inp.settings.zoom,
    );
    const key = lutKey(fspec, mapping);
    if (!this.lut || this.lut.key !== key) this.lut = buildScanLut(fspec, mapping);
    const colorCur = this.colorDriver.current;
    const colorVel = cf ? cf.colorVel : inp.modality === 'color' && colorCur ? colorCur.vel : null;
    const colorVar = cf
      ? cf.colorVar
      : inp.modality === 'color' && colorCur
        ? colorCur.variance
        : null;
    const tintMap = inp.settings.tintMap;
    let bitmap: ImageBitmap | null = null;
    const tp = performance.now();
    if (!frozen && !isStrip && this.displayOnGpu && this.gpu) {
      const color =
        colorVel && colorVar
          ? {
              vel: colorVel,
              variance: colorVar,
              version: this.colorDriver.version,
              settings: colorSettings,
            }
          : null;
      bitmap = this.gpu.present({ lut: this.lut, width: W, height: sectorH, color, tintMap });
    }
    this.timing.presentMs = bitmap ? performance.now() - tp : 0;
    const buffer = bitmap ? new ArrayBuffer(0) : this.takeBuffer(W * H * 4);
    const rgba = new Uint8ClampedArray(buffer);
    if (!bitmap) {
      const sectorRgba = new Uint8ClampedArray(buffer, 0, W * sectorH * 4);
      scanConvertLut(display, this.lut, sectorRgba, tintMap);
      if (colorVel && colorVar)
        overlayColorField(sectorRgba, this.lut, colorVel, colorVar, colorSettings);
    }
    if (isStrip) {
      rgba.fill(0, W * sectorH * 4);
      for (let i = W * sectorH * 4 + 3; i < rgba.length; i += 4) rgba[i] = 255;
    }
    let strip: StripInfo = {
      x: 0,
      y: sectorH,
      width: W,
      height: H - sectorH,
      secondsPerColumn: 0,
      headColumn: 0,
      columns: 0,
      topValue: 0,
      bottomValue: 0,
      kind: null,
    };
    if (isStrip && (!cf || cf.acquisition.stripAcquisitionId === this.strips.historyId))
      strip = this.strips.drawStrip(rgba, W, H, sectorH, tintMap);
    const view = cf ? cf.analysis : this.lastAnalysis;
    const c = this.clock.current;
    const structure = cf ? cf.structure : this.frame ? this.frame.structure : new Uint8Array(0);
    const segment = cf ? cf.segment : this.frame ? this.frame.segment : new Uint8Array(0);
    const gate = cf
      ? cf.gate
      : isStrip
        ? this.strips.gateInfo(beam, fspec, c.phase, structure, this.stripCtx())
        : null;
    const ecgTail = this.ecgRing.extractTail(frozen, this.timeS, this.cine.oldest?.timeS);
    this.lastOutputBeam = beam;
    const sector = { ...mapping, x: 0, y: 0 };
    this.lastSector = sector;
    this.lastStrip = strip;
    this.displayedFrame = cf ?? this.cine.newest;
    return {
      acquisition: cf?.acquisition,
      frameId: cf ? cf.frameId : this.frameId,
      width: W,
      height: H,
      rgba: buffer,
      bitmap,
      sector,
      strip,
      polar: {
        lines: fspec.lines,
        samples: fspec.samples,
        sectorRad: fspec.sectorRad,
        depthCm: fspec.depthCm,
      },
      structure: frozen || inp.needStructure !== false ? new Uint8Array(structure) : new Uint8Array(0),
      segment: frozen || inp.needSegments !== false ? new Uint8Array(segment) : new Uint8Array(0),
      gate,
      timeS: cf ? cf.timeS : this.timeS,
      phase: cf ? cf.phase : c.phase,
      beatIndex: cf ? cf.beatIndex : c.beatIndex,
      heartRateBpm: 60 / (cf ? cf.rrS : c.rrS),
      rrS: cf ? cf.rrS : c.rrS,
      simulatedFps: this.acquisitionFps(),
      cadenceHz: 1 / this.frameIntervalS,
      ecg: ecgTail,
      ecgHead: this.timeS,
      view,
      spectrumColumn: frozen && cf ? cf.spectrumColumn : this.strips.spectrumColumn,
      spectralRange:
        frozen && cf
          ? cf.spectralRange
          : (this.strips.velocityRange ?? spectralRange(inp.spectral)),
      frozen,
      cineLength: this.cine.length,
      cineOffset: cf ? inp.cineOffset : 0,
      cineFramePhase: cf ? cf.phase : c.phase,
      cineWindow: {
        startS: this.cine.oldest?.timeS ?? this.timeS,
        endS: this.cine.newest?.timeS ?? this.timeS,
        frameS: cf ? cf.timeS : this.timeS,
      },
      stats: {
        ...this.backend.stats(),
        backend: this.backend.id,
        anatomy: this.experimental.anatomyStatus(this.input.anatomy, this.backend === this.procedural),
        imagen: this.experimental.imageFormationStatus(this.input.imageFormation),
        gpu: this.gpuReason,
        tier: this.tier,
        lines: spec.lines,
        samples: spec.samples,
        renderFrameMs: Number(this.timing.renderFrameMs.toFixed(1)),
        consoleMs: Number(this.timing.consoleMs.toFixed(1)),
        persistence: Number(this.lastPersistence.toFixed(3)),
        analysisMs: Number(this.timing.analysisMs.toFixed(1)),
        compositeMs: Number(this.timing.compositeMs.toFixed(1)),
        cineMs: Number(this.timing.cineMs.toFixed(2)),
        console: this.displayOnGpu ? 'gpu' : 'cpu',
        present: bitmap ? 'gpu' : 'cpu',
        presentMs: Number(this.timing.presentMs.toFixed(2)),
      },
      colorFps: this.colorFps,
      probeBeam: {
        origin: [beam.origin.x, beam.origin.y, beam.origin.z],
        forward: [beam.forward.x, beam.forward.y, beam.forward.z],
        lateral: [beam.lateral.x, beam.lateral.y, beam.lateral.z],
        normal: [beam.normal.x, beam.normal.y, beam.normal.z],
      },
    };
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

/** Options of a simulator core: how it fetches the app's assets (decision 352). */
export interface SimulatorCoreOptions {
  fetchAsset?: AssetFetcher;
}

/** The app's assets, fetched under its base URL (the worker's and the page's). */
const fetchFromAppBase: AssetFetcher = async (path) => {
  const r = await fetch(`${import.meta.env.BASE_URL}${path}`);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.arrayBuffer();
};
