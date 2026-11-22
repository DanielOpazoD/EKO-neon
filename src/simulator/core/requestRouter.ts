import { v3 } from '@/core/vec3';
import type { HeartModel } from '@/simulator/anatomy/heartModel';
import type { ThoraxModel } from '@/simulator/anatomy/thoraxModel';
import type { BeatTables } from '@/simulator/cardiac-cycle/cycleModel';
import { computeAutoOpt } from '@/simulator/renderer/postprocess/autoOpt';
import { ProceduralSliceRenderer } from '@/simulator/renderer/procedural/sliceRenderer';
import type { PolarFrame } from '@/simulator/renderer/types';
import { controlThroughCrux, getViewTarget } from '@/simulator/windows/viewTargets';
import type { CineBuffer, CineFrame } from './cineBuffer';
import type { ExperimentalLoaders } from './experimentalLoaders';
import { explainPointAt } from './explainPoint';
import { probePointAt, probeTorsoPointAt } from './probePoint';
import type { PhaseMarks, SimInput, SimRequest, SimResponse } from './protocol';
import type { StripEngine } from './stripEngine';

export interface RequestRouterCtx {
  input: SimInput;
  displayedFrame: CineFrame | undefined;
  display: Uint8ClampedArray | null;
  frame: PolarFrame | null;
  heart: HeartModel;
  thorax: ThoraxModel;
  tables: BeatTables;
  experimental: ExperimentalLoaders;
  mirrorLevel: number;
  cine: CineBuffer;
  strips: StripEngine;
  phaseMarks(): PhaseMarks;
  getExplainTracer(): ProceduralSliceRenderer;
}

/** Dispatch an on-demand SimRequest (probePoint, explainPoint, canonicalControl, autoOpt, cinePhaseJump, autoTrace). */
export function routeSimRequest(req: SimRequest, ctx: RequestRouterCtx): SimResponse | null {
  if (req.kind === 'probePoint') {
    const frame = ctx.displayedFrame;
    if (!frame) return null;
    const { scene, beam, phase } = frame;
    return {
      kind: 'probePoint',
      point: req.torso
        ? probeTorsoPointAt(scene.heart, scene.thorax, scene.heartPose, beam, req.torso, phase)
        : probePointAt(
            scene.heart,
            scene.thorax,
            scene.heartPose,
            beam,
            req.rCm ?? 0,
            req.thetaRad ?? 0,
            phase,
          ),
    };
  }
  if (req.kind === 'explainPoint') {
    const frame = ctx.displayedFrame;
    if (!frame) return null;
    return {
      kind: 'explainPoint',
      explain: explainPointAt(
        ctx.getExplainTracer(),
        frame.scene,
        frame.beam,
        frame.spec,
        req.rCm,
        req.thetaRad,
        ctx.mirrorLevel,
      ),
    };
  }
  if (req.kind === 'canonicalControl') {
    const view = getViewTarget(req.viewId);
    // The Rodero heart's four-chamber plane passes through its own crux (decision 360)
    const rodero =
      ctx.input.anatomy === 'rodero'
        ? ctx.experimental.roderoHeart(ctx.input.anatomy, ctx.heart, ctx.tables)
        : null;
    const c = rodero?.vh.crux();
    return {
      kind: 'canonicalControl',
      control: controlThroughCrux(view, ctx.heart, ctx.thorax, c ? v3(c[0], c[1], c[2]) : null),
    };
  }
  if (req.kind === 'autoOpt') {
    if (!ctx.display || !ctx.frame) return null;
    const opt = computeAutoOpt(ctx.display, ctx.frame.tissue, ctx.frame.spec, ctx.input.settings);
    return { kind: 'autoOpt', gainDb: opt.gainDb, tgcDb: opt.tgcDb };
  }
  if (req.kind === 'cinePhaseJump') {
    const targetPhase = req.target === 'es' ? ctx.phaseMarks().endSystole : 0;
    const found = ctx.cine.findNearestPhaseOffset(targetPhase);
    if (!found) return null;
    return { kind: 'cinePhaseJump', offset: found.offset, phase: found.phase };
  }
  if (
    ctx.input.frozen &&
    ctx.displayedFrame &&
    ctx.displayedFrame.acquisition.stripAcquisitionId !== ctx.strips.historyId
  )
    return null;
  return ctx.strips.request(req);
}
