import type { ImagingModality } from '@/simulator/renderer/types';

/** Real-time frame budget target for 60 fps responsiveness (ms). */
export const DOPPLER_FRAME_BUDGET_MS = 16;

export type DopplerModality = 'pw' | 'cw' | 'tdi' | 'color' | 'cmm';

export function isDopplerModality(modality: ImagingModality): modality is DopplerModality {
  return (
    modality === 'pw' ||
    modality === 'cw' ||
    modality === 'tdi' ||
    modality === 'color' ||
    modality === 'cmm'
  );
}

export const SPECTRAL_STAGE_KEYS = [
  'momentSetup',
  'acousticAccess',
  'accelChirp',
  'quadrature',
  'valveClicks',
  'spectrumSynth',
  'spectrumPresent',
  'stripRaster',
  'gateInfo',
] as const;

export type SpectralStageKey = (typeof SPECTRAL_STAGE_KEYS)[number];

export const DOPPLER_STAGE_KEYS = [
  ...SPECTRAL_STAGE_KEYS,
  'colorDoppler',
  'bmodeRender',
  'compositeOther',
] as const;

export type DopplerStageKey = (typeof DOPPLER_STAGE_KEYS)[number];

export const DOPPLER_STAGE_LABELS: Readonly<Record<DopplerStageKey, string>> = {
  momentSetup: 'Pose/ciclo cardíaco',
  acousticAccess: 'Acceso acústico (atenuación)',
  accelChirp: 'Aceleración dv/dt (chirp)',
  quadrature: 'Cuadratura 3D y muestreo flujo',
  valveClicks: 'Clicks valvulares',
  spectrumSynth: 'Síntesis espectral (I/Q + FFT)',
  spectrumPresent: 'Presentación y compresión',
  stripRaster: 'Rasterización strip RGBA',
  gateInfo: 'Gate y ángulo beam-flujo',
  colorDoppler: 'Doppler Color / CMM',
  bmodeRender: 'B-mode dúplex',
  compositeOther: 'Composición / otros',
};

export type DopplerStageTimings = Record<DopplerStageKey, number>;

/** Mutable per-frame accumulator filled by StripEngine / spectralSampler / stripPainter. */
export interface SpectralFrameTiming {
  columns: number;
  samplesEvaluated: number;
  momentSetupMs: number;
  acousticAccessMs: number;
  accelChirpMs: number;
  quadratureMs: number;
  valveClicksMs: number;
  spectrumSynthMs: number;
  spectrumPresentMs: number;
  stripRasterMs: number;
  gateInfoMs: number;
}

export function createSpectralFrameTiming(): SpectralFrameTiming {
  return {
    columns: 0,
    samplesEvaluated: 0,
    momentSetupMs: 0,
    acousticAccessMs: 0,
    accelChirpMs: 0,
    quadratureMs: 0,
    valveClicksMs: 0,
    spectrumSynthMs: 0,
    spectrumPresentMs: 0,
    stripRasterMs: 0,
    gateInfoMs: 0,
  };
}

export function resetSpectralFrameTiming(t: SpectralFrameTiming): void {
  t.columns = 0;
  t.samplesEvaluated = 0;
  t.momentSetupMs = 0;
  t.acousticAccessMs = 0;
  t.accelChirpMs = 0;
  t.quadratureMs = 0;
  t.valveClicksMs = 0;
  t.spectrumSynthMs = 0;
  t.spectrumPresentMs = 0;
  t.stripRasterMs = 0;
  t.gateInfoMs = 0;
}

export interface BottleneckInfo {
  stage: DopplerStageKey;
  label: string;
  ms: number;
  pctOfTotal: number;
}

export interface DopplerPerfFrame {
  frameId: number;
  modality: DopplerModality;
  budgetMs: number;
  totalMs: number;
  spectralTotalMs: number;
  columns: number;
  samplesEvaluated: number;
  stages: DopplerStageTimings;
  bottleneck: BottleneckInfo;
  spectralBottleneck: BottleneckInfo | null;
  overBudget: boolean;
  headroomMs: number;
  timestampMs: number;
}

export interface BuildDopplerPerfParams {
  frameId: number;
  modality: DopplerModality;
  totalMs: number;
  spectral?: Readonly<SpectralFrameTiming>;
  colorDopplerMs?: number;
  bmodeRenderMs?: number;
  budgetMs?: number;
  timestampMs?: number;
}

function round3(v: number): number {
  return Math.round(Math.max(0, v) * 1000) / 1000;
}

/**
 * Assemble a complete per-frame Doppler performance snapshot and identify both the overall
 * bottleneck stage and the top bottleneck strictly within the spectral reconstruction pipeline.
 */
export function buildDopplerPerfFrame(params: BuildDopplerPerfParams): DopplerPerfFrame {
  const budgetMs = params.budgetMs ?? DOPPLER_FRAME_BUDGET_MS;
  const totalMs = round3(params.totalMs);
  const sp = params.spectral;

  const momentSetup = round3(sp?.momentSetupMs ?? 0);
  const acousticAccess = round3(sp?.acousticAccessMs ?? 0);
  const accelChirp = round3(sp?.accelChirpMs ?? 0);
  const quadrature = round3(sp?.quadratureMs ?? 0);
  const valveClicks = round3(sp?.valveClicksMs ?? 0);
  const spectrumSynth = round3(sp?.spectrumSynthMs ?? 0);
  const spectrumPresent = round3(sp?.spectrumPresentMs ?? 0);
  const stripRaster = round3(sp?.stripRasterMs ?? 0);
  const gateInfo = round3(sp?.gateInfoMs ?? 0);
  const colorDoppler = round3(params.colorDopplerMs ?? 0);
  const bmodeRender = round3(params.bmodeRenderMs ?? 0);

  const spectralTotalMs = round3(
    momentSetup +
      acousticAccess +
      accelChirp +
      quadrature +
      valveClicks +
      spectrumSynth +
      spectrumPresent +
      stripRaster +
      gateInfo,
  );
  const knownSum = spectralTotalMs + colorDoppler + bmodeRender;
  const compositeOther = round3(Math.max(0, totalMs - knownSum));

  const stages: DopplerStageTimings = {
    momentSetup,
    acousticAccess,
    accelChirp,
    quadrature,
    valveClicks,
    spectrumSynth,
    spectrumPresent,
    stripRaster,
    gateInfo,
    colorDoppler,
    bmodeRender,
    compositeOther,
  };

  let topStage: DopplerStageKey = 'compositeOther';
  let topMs = -1;
  for (const key of DOPPLER_STAGE_KEYS) {
    if (stages[key] > topMs) {
      topMs = stages[key];
      topStage = key;
    }
  }
  const denomTotal = Math.max(totalMs, knownSum, 1e-6);
  const bottleneck: BottleneckInfo = {
    stage: topStage,
    label: DOPPLER_STAGE_LABELS[topStage],
    ms: stages[topStage],
    pctOfTotal: Math.round((stages[topStage] / denomTotal) * 1000) / 10,
  };

  let spectralBottleneck: BottleneckInfo | null = null;
  if (spectralTotalMs > 0) {
    let spStage: SpectralStageKey = 'quadrature';
    let spMs = -1;
    for (const key of SPECTRAL_STAGE_KEYS) {
      if (stages[key] > spMs) {
        spMs = stages[key];
        spStage = key;
      }
    }
    spectralBottleneck = {
      stage: spStage,
      label: DOPPLER_STAGE_LABELS[spStage],
      ms: stages[spStage],
      pctOfTotal: Math.round((stages[spStage] / Math.max(spectralTotalMs, 1e-6)) * 1000) / 10,
    };
  }

  return {
    frameId: params.frameId,
    modality: params.modality,
    budgetMs,
    totalMs,
    spectralTotalMs,
    columns: sp?.columns ?? 0,
    samplesEvaluated: sp?.samplesEvaluated ?? 0,
    stages,
    bottleneck,
    spectralBottleneck,
    overBudget: totalMs > budgetMs,
    headroomMs: Math.round((budgetMs - totalMs) * 1000) / 1000,
    timestampMs: params.timestampMs ?? Date.now(),
  };
}

/** Format one concise single-line console diagnostic report for a Doppler frame. */
export function formatDopplerPerfLine(frame: DopplerPerfFrame): string {
  const mod = frame.modality.toUpperCase();
  const status = frame.overBudget
    ? `OVER (+${Math.abs(frame.headroomMs).toFixed(2)}ms)`
    : `OK (-${frame.headroomMs.toFixed(2)}ms)`;
  const colMeta =
    frame.columns > 0 ? ` cols=${frame.columns} smp=${frame.samplesEvaluated}` : '';
  const bn = frame.bottleneck;
  const specTag =
    frame.spectralTotalMs > 0
      ? ` | spectral=${frame.spectralTotalMs.toFixed(2)}ms` +
        (frame.spectralBottleneck && frame.spectralBottleneck.stage !== bn.stage
          ? ` (cuello esp.: ${frame.spectralBottleneck.stage} ${frame.spectralBottleneck.ms.toFixed(2)}ms)`
          : '')
      : '';

  const activeStages = DOPPLER_STAGE_KEYS.filter((k) => frame.stages[k] >= 0.05)
    .sort((a, b) => frame.stages[b] - frame.stages[a])
    .slice(0, 5)
    .map((k) => `${k}=${frame.stages[k].toFixed(2)}ms`)
    .join(' ');

  return (
    `[DopplerPerf] #${frame.frameId} [${mod}] ${frame.totalMs.toFixed(2)}ms / ${frame.budgetMs.toFixed(1)}ms ${status}${colMeta}` +
    `${specTag} | cuello=${bn.stage} ${bn.ms.toFixed(2)}ms (${bn.pctOfTotal.toFixed(1)}%)` +
    (activeStages ? ` | ${activeStages}` : '')
  );
}

export interface StageAggregate {
  stage: DopplerStageKey;
  label: string;
  meanMs: number;
  p95Ms: number;
  maxMs: number;
  sharePct: number;
  bottleneckHits: number;
}

export interface DopplerPerfSummary {
  frames: number;
  modality: DopplerModality | 'all';
  budgetMs: number;
  overBudgetFrames: number;
  overBudgetPct: number;
  meanMs: number;
  p95Ms: number;
  maxMs: number;
  meanSpectralMs: number;
  p95SpectralMs: number;
  maxSpectralMs: number;
  topBottleneck: StageAggregate | null;
  topSpectralBottleneck: StageAggregate | null;
  stages: StageAggregate[];
}

export interface DopplerPerfLogger {
  log: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
  table?: (data: unknown) => void;
}

export interface DopplerPerfMonitorOptions {
  budgetMs?: number;
  maxHistory?: number;
  enabled?: boolean;
  onlyOverBudget?: boolean;
  logger?: DopplerPerfLogger;
}

function percentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[idx]!;
}

/**
 * Console diagnostic monitor for Doppler frame processing times and spectral reconstruction
 * bottleneck identification against a configurable 16 ms frame budget.
 */
export class DopplerPerfMonitor {
  private budgetMs: number;
  private readonly maxHistory: number;
  private enabled: boolean;
  private onlyOverBudget: boolean;
  private readonly logger: DopplerPerfLogger;
  private frames: DopplerPerfFrame[] = [];

  constructor(options: DopplerPerfMonitorOptions = {}) {
    this.budgetMs = options.budgetMs ?? DOPPLER_FRAME_BUDGET_MS;
    this.maxHistory = Math.max(16, options.maxHistory ?? 240);
    this.enabled = options.enabled ?? true;
    this.onlyOverBudget = options.onlyOverBudget ?? false;
    this.logger = options.logger ?? console;
  }

  /** Record a Doppler frame and report it to the console if logging is active. */
  record(frame: DopplerPerfFrame): void {
    const calibrated: DopplerPerfFrame =
      frame.budgetMs === this.budgetMs
        ? frame
        : {
            ...frame,
            budgetMs: this.budgetMs,
            overBudget: frame.totalMs > this.budgetMs,
            headroomMs: Math.round((this.budgetMs - frame.totalMs) * 1000) / 1000,
          };
    this.frames.push(calibrated);
    if (this.frames.length > this.maxHistory) {
      this.frames.splice(0, this.frames.length - this.maxHistory);
    }
    if (!this.enabled) return;
    if (this.onlyOverBudget && !calibrated.overBudget) return;
    const line = formatDopplerPerfLine(calibrated);
    if (calibrated.overBudget) {
      this.logger.warn(line);
    } else {
      this.logger.log(line);
    }
  }

  /** Compute aggregate statistics and identify the dominant bottlenecks across recorded frames. */
  summary(modality?: DopplerModality): DopplerPerfSummary {
    const subset = modality ? this.frames.filter((f) => f.modality === modality) : this.frames;
    const n = subset.length;
    if (n === 0) {
      return {
        frames: 0,
        modality: modality ?? 'all',
        budgetMs: this.budgetMs,
        overBudgetFrames: 0,
        overBudgetPct: 0,
        meanMs: 0,
        p95Ms: 0,
        maxMs: 0,
        meanSpectralMs: 0,
        p95SpectralMs: 0,
        maxSpectralMs: 0,
        topBottleneck: null,
        topSpectralBottleneck: null,
        stages: [],
      };
    }

    const totals = subset.map((f) => f.totalMs).sort((a, b) => a - b);
    const spectrals = subset.map((f) => f.spectralTotalMs).sort((a, b) => a - b);
    let sumTotal = 0;
    let sumSpectral = 0;
    let overCount = 0;
    for (const f of subset) {
      sumTotal += f.totalMs;
      sumSpectral += f.spectralTotalMs;
      if (f.overBudget) overCount++;
    }
    const meanMs = round3(sumTotal / n);
    const meanSpectralMs = round3(sumSpectral / n);

    const stages: StageAggregate[] = DOPPLER_STAGE_KEYS.map((stage) => {
      const vals = subset.map((f) => f.stages[stage]).sort((a, b) => a - b);
      let s = 0;
      let hits = 0;
      for (const f of subset) {
        s += f.stages[stage];
        if (f.bottleneck.stage === stage) hits++;
      }
      const stMean = round3(s / n);
      return {
        stage,
        label: DOPPLER_STAGE_LABELS[stage],
        meanMs: stMean,
        p95Ms: round3(percentile(vals, 0.95)),
        maxMs: round3(vals[vals.length - 1] ?? 0),
        sharePct: sumTotal > 0 ? Math.round((s / sumTotal) * 1000) / 10 : 0,
        bottleneckHits: hits,
      };
    }).sort((a, b) => b.meanMs - a.meanMs);

    const spectralSet = new Set<DopplerStageKey>(SPECTRAL_STAGE_KEYS);
    const topBottleneck = stages.find((s) => s.maxMs > 0) ?? null;
    const topSpectralBottleneck =
      stages.find((s) => spectralSet.has(s.stage) && s.maxMs > 0) ?? null;

    return {
      frames: n,
      modality: modality ?? 'all',
      budgetMs: this.budgetMs,
      overBudgetFrames: overCount,
      overBudgetPct: Math.round((overCount / n) * 1000) / 10,
      meanMs,
      p95Ms: round3(percentile(totals, 0.95)),
      maxMs: round3(totals[totals.length - 1] ?? 0),
      meanSpectralMs,
      p95SpectralMs: round3(percentile(spectrals, 0.95)),
      maxSpectralMs: round3(spectrals[spectrals.length - 1] ?? 0),
      topBottleneck,
      topSpectralBottleneck,
      stages,
    };
  }

  /** Print a structured bottleneck summary report in the console and return the summary object. */
  report(modality?: DopplerModality): DopplerPerfSummary {
    const sum = this.summary(modality);
    if (sum.frames === 0) {
      this.logger.log(
        `[DopplerPerf] Sin frames Doppler registrados${modality ? ` para ${modality.toUpperCase()}` : ''}. Activa PW, CW, TDI o Color para capturar métricas.`,
      );
      return sum;
    }
    const status =
      sum.p95Ms > sum.budgetMs
        ? `ALERTA: p95 ${sum.p95Ms.toFixed(2)}ms supera el umbral de ${sum.budgetMs}ms`
        : `OK: p95 ${sum.p95Ms.toFixed(2)}ms bajo el umbral de ${sum.budgetMs}ms`;
    const header =
      `[DopplerPerf] Resumen (${String(sum.modality).toUpperCase()}, n=${sum.frames}) — ${status} | ` +
      `media=${sum.meanMs.toFixed(2)}ms, max=${sum.maxMs.toFixed(2)}ms, >${sum.budgetMs}ms=${sum.overBudgetFrames}/${sum.frames} (${sum.overBudgetPct}%)` +
      (sum.topSpectralBottleneck
        ? ` | cuello espectral #1: ${sum.topSpectralBottleneck.stage} (${sum.topSpectralBottleneck.meanMs.toFixed(2)}ms media, ${sum.topSpectralBottleneck.sharePct}%)`
        : '');

    if (sum.p95Ms > sum.budgetMs) {
      this.logger.warn(header);
    } else {
      this.logger.log(header);
    }

    const active = sum.stages.filter((s) => s.maxMs > 0);
    if (active.length > 0 && typeof this.logger.table === 'function') {
      this.logger.table(
        active.map((s) => ({
          etapa: s.stage,
          descripcion: s.label,
          mediaMs: s.meanMs,
          p95Ms: s.p95Ms,
          maxMs: s.maxMs,
          pesoPct: s.sharePct,
          vecesCuello: s.bottleneckHits,
        })),
      );
    }
    return sum;
  }

  enable(): void {
    this.enabled = true;
  }

  disable(): void {
    this.enabled = false;
  }

  setBudget(budgetMs: number): void {
    if (Number.isFinite(budgetMs) && budgetMs > 0) this.budgetMs = budgetMs;
  }

  setOnlyOverBudget(onlyOverBudget: boolean): void {
    this.onlyOverBudget = onlyOverBudget;
  }

  reset(): void {
    this.frames.length = 0;
  }

  history(modality?: DopplerModality): readonly DopplerPerfFrame[] {
    return modality ? this.frames.filter((f) => f.modality === modality) : [...this.frames];
  }

  status(): {
    enabled: boolean;
    onlyOverBudget: boolean;
    budgetMs: number;
    recordedFrames: number;
  } {
    return {
      enabled: this.enabled,
      onlyOverBudget: this.onlyOverBudget,
      budgetMs: this.budgetMs,
      recordedFrames: this.frames.length,
    };
  }
}

/**
 * Default main-thread singleton monitor attached to `frameBus` and exposed on `window.dopplerPerf`. It is silent: it
 * keeps every Doppler frame in its history, so `report()` and `summary()` work at any time, and writes a console line
 * per frame only after `enable()`. Logging by default printed a line for every Doppler frame (30 or more per second)
 * in the console of every build, the learner's included, burying the warnings that matter (worker failure, lost WebGL
 * context) and paying a formatted string per frame.
 */
export const dopplerPerfMonitor = new DopplerPerfMonitor({ enabled: false });
