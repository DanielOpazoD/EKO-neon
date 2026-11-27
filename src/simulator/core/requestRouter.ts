import { v3 } from '@/core/vec3';
import type { HeartModel } from '@/simulator/anatomy/heartModel';
import type { ThoraxModel } from '@/simulator/anatomy/thoraxModel';
import type { BeatTables } from '@/simulator/cardiac-cycle/cycleModel';
import { computeAutoChamberVolume } from '@/simulator/measurements/autoVolume';
import { findLvotGateFromPolar } from '@/simulator/measurements/autoLvotGate';
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
    const byId = req.frameId !== undefined ? ctx.cine.findFrameOffset(req.frameId) : null;
    const targetPhase = req.target === 'es' ? ctx.phaseMarks().endSystole : 0;
    const found = byId ?? ctx.cine.findNearestPhaseOffset(targetPhase);
    if (!found) return null;
    return { kind: 'cinePhaseJump', offset: found.offset, phase: found.phase };
  }
  if (req.kind === 'autoChamberVolume') {
    const refFrame = ctx.displayedFrame ?? ctx.cine.newest;
    const result = computeAutoChamberVolume(
      ctx.cine,
      refFrame,
      ctx.phaseMarks(),
      ctx.input.display,
      ctx.input.settings,
      req.chamber,
    );
    return { kind: 'autoChamberVolume', result };
  }
  if (req.kind === 'autoLvotGate') {
    const refFrame = ctx.displayedFrame ?? ctx.cine.newest;
    if (!refFrame) {
      return {
        kind: 'autoLvotGate',
        result: {
          ok: false,
          reason: 'no-frame',
          message: 'No hay un cuadro 2D disponible para localizar el TSVI.',
        },
      };
    }
    // Prefer a mid-systolic frame (~50% of ejection) at the same probe pose so the fixed PW gate
    // sits at the ASE 0.5 cm LVOT station during peak systolic ejection (when VTI velocity is recorded).
    const pm = ctx.phaseMarks();
    const midSysPhase = pm.ejectionStart + 0.5 * (pm.ejectionEnd - pm.ejectionStart);
    const samePose = (f: CineFrame) =>
      Math.abs(f.acquisition.probe.u - refFrame.acquisition.probe.u) < 0.2 &&
      Math.abs(f.acquisition.probe.v - refFrame.acquisition.probe.v) < 0.2 &&
      Math.abs(f.acquisition.probe.rotationDeg - refFrame.acquisition.probe.rotationDeg) < 3 &&
      Math.abs(f.acquisition.probe.tiltDeg - refFrame.acquisition.probe.tiltDeg) < 3 &&
      Math.abs(f.acquisition.probe.rockDeg - refFrame.acquisition.probe.rockDeg) < 3;
    const foundSys = ctx.cine.findNearestPhaseOffsetMatching(midSysPhase, samePose);
    const sysFrame = foundSys ? ctx.cine.select(foundSys.offset) : undefined;
    const distSys = sysFrame
      ? Math.min(
          Math.abs(sysFrame.phase - midSysPhase),
          1 - Math.abs(sysFrame.phase - midSysPhase),
        )
      : 1;
    const chosen = sysFrame && distSys <= 0.12 ? sysFrame : refFrame;
    const isSystolic =
      chosen.phase >= pm.ejectionStart && chosen.phase <= pm.ejectionEnd + 0.05;
    const result = findLvotGateFromPolar(chosen.structure, chosen.spec, {
      systolicFrame: isSystolic,
    });
    return { kind: 'autoLvotGate', result };
  }
  if (
    ctx.input.frozen &&
    ctx.displayedFrame &&
    ctx.displayedFrame.acquisition.stripAcquisitionId !== ctx.strips.historyId
  )
    return null;
  return ctx.strips.request(req, ctx.phaseMarks());
}
