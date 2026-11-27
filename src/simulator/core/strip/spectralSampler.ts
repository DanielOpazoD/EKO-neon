import { createRng } from '@/core/random';
import { smoothstep } from '@/core/vec3';
import { computeHeartPose } from '@/simulator/anatomy/heartModel';
import { sceneClassifier } from '@/simulator/anatomy/sceneClassifier';
import { Structure, Tissue, type TissueSample } from '@/simulator/anatomy/tissue';
import { cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import { relativeEchoAmplitude } from '@/simulator/doppler/acousticAccess';
import {
  sampleFlow,
  sampleTissueVelocity,
  type FlowSample,
} from '@/simulator/doppler/flow-primitives/flowField';
import { accumulatePulsedSpectrum } from '@/simulator/doppler/spectral/pulsedIq';
import {
  depthLimitedPulsedAcquisition,
  rayleighBackscatterGain,
  sampleVolumeWeight,
  sliceHalfWidthCm,
} from '@/simulator/doppler/spectral/scatteringModel';
import {
  accumulateSpectrum,
  presentSpectralColumn,
  SPECTRAL_BINS,
  type VelocitySample,
} from '@/simulator/doppler/spectral/spectrum';
import type { SpectralFrameTiming } from '@/simulator/doppler/dopplerPerf';
import { valveClickSignal, valveClickWeight } from '@/simulator/doppler/spectral/valveClicks';
import type { BeamFrame } from '@/simulator/probe/pose';
import { beamHalfWidthCm, focusingGain } from '@/simulator/renderer/acoustic/acoustics';
import { MODALITIES } from '@/simulator/renderer/modality';
import { DUPLEX_BMODE_SHARE } from '@/simulator/renderer/pulseTiming';
import type { PolarFrameSpec } from '@/simulator/renderer/types';
import type { GateInfo } from '../protocol';
import type { StripCtx } from '../stripEngine';

export interface GateFlowCache {
  key: string;
  best: number;
  bx: number;
  by: number;
  bz: number;
}

export interface SpectralSampleOptions {
  /** Optional pre-cached central-ray relative echo amplitude query for this pose and phase bin. */
  received?: (depthCm: number) => number;
  /** Optional reusable scratch buffer for the raw spectral column (SPECTRAL_BINS). */
  outColumn?: Float32Array;
  /** Optional reusable scratch buffer for the presented spectral column (SPECTRAL_BINS). */
  outDisplay?: Float32Array;
  /** Optional hook receiving the un-presented CW raw spectrum and valve-click ray geometry before presentation. */
  onRawCw?: (
    rawColumn: Float32Array,
    clickPoints: readonly number[],
    clickPowers: readonly number[],
    rayHeartDir: { x: number; y: number; z: number },
  ) => void;
}

/** Sample one spectral Doppler column (PW, CW or TDI) at a given cardiac phase and time instant. */
export function sampleSpectralColumn(
  beam: BeamFrame,
  spec: PolarFrameSpec,
  phase: number,
  col: number,
  stripHead: number,
  timeS: number,
  ctx: StripCtx,
  flowSample: FlowSample,
  tissueSample: TissueSample,
  timing?: SpectralFrameTiming,
  options?: SpectralSampleOptions,
): { column: Float32Array; display: Float32Array } {
  const inp = ctx.input;
  const hf = ctx.heart.frame;
  const f0MHz = inp.settings.frequencyMHz;
  const focusCm = spec.focusCm;
  const isTdi = inp.modality === 'tdi';
  const isCw = inp.modality === 'cw';
  const gateLenOrNull = isCw ? null : inp.spectral.gateLengthCm;
  const backscatterScale = rayleighBackscatterGain(f0MHz, isTdi);
  const theta = Math.max(-spec.sectorRad / 2, Math.min(spec.sectorRad / 2, inp.cursorThetaRad));
  const ct = Math.cos(theta),
    sn = Math.sin(theta);
  const dx = beam.forward.x * ct + beam.lateral.x * sn;
  const dy = beam.forward.y * ct + beam.lateral.y * sn;
  const dz = beam.forward.z * ct + beam.lateral.z * sn;
  const dhx = dx * hf.ex.x + dy * hf.ex.y + dz * hf.ex.z;
  const dhy = dx * hf.ey.x + dy * hf.ey.y + dz * hf.ey.z;
  const dhz = dx * hf.ez.x + dy * hf.ez.y + dz * hf.ez.z;
  const scene = ctx.scene(phase);
  const hp = scene.heartPose;
  const classifyScene = sceneClassifier(scene);
  const tAcoustic = timing ? performance.now() : 0;
  const received =
    options?.received ??
    relativeEchoAmplitude(scene, beam.origin, { x: dx, y: dy, z: dz }, beam.contact);
  if (timing) timing.acousticAccessMs += performance.now() - tAcoustic;
  const samples: VelocitySample[] = [];
  const lx = beam.lateral.x * ct - beam.forward.x * sn,
    ly = beam.lateral.y * ct - beam.forward.y * sn,
    lz = beam.lateral.z * ct - beam.forward.z * sn;
  const aliasing = MODALITIES[inp.modality].aliasing;

  // Estimate axial blood acceleration (dv/dt) at the PW gate center for intra-packet chirp physics.
  // Over ±4 ms cardiac geometry moves <0.4 mm, so reusing the column's already-computed HeartPose
  // while querying beat flow tables at pPrev/pNext preserves dv/dt without two redundant pose builds.
  let gateAccelMps2 = 0;
  const tAccel = timing && aliasing && !isTdi ? performance.now() : 0;
  if (aliasing && !isTdi) {
    const dtAcc = 0.004;
    const dPhase = dtAcc / ctx.tables.rrS;
    const pPrev = (((phase - dPhase) % 1) + 1) % 1;
    const pNext = (((phase + dPhase) % 1) + 1) % 1;
    const r0 = inp.gateDepthCm;
    const px0 = beam.origin.x + dx * r0;
    const py0 = beam.origin.y + dy * r0;
    const pz0 = beam.origin.z + dz * r0;
    const hx0 =
      (px0 - hf.origin.x) * hf.ex.x + (py0 - hf.origin.y) * hf.ex.y + (pz0 - hf.origin.z) * hf.ex.z;
    const hy0 =
      (px0 - hf.origin.x) * hf.ey.x + (py0 - hf.origin.y) * hf.ey.y + (pz0 - hf.origin.z) * hf.ey.z;
    const hz0 =
      (px0 - hf.origin.x) * hf.ez.x + (py0 - hf.origin.y) * hf.ez.y + (pz0 - hf.origin.z) * hf.ez.z;
    sampleFlow(ctx.flow, ctx.tables, hp, pPrev, hx0, hy0, hz0, flowSample);
    const vPrev = flowSample.present
      ? -(flowSample.vx * dhx + flowSample.vy * dhy + flowSample.vz * dhz)
      : 0;
    sampleFlow(ctx.flow, ctx.tables, hp, pNext, hx0, hy0, hz0, flowSample);
    const vNext = flowSample.present
      ? -(flowSample.vx * dhx + flowSample.vy * dhy + flowSample.vz * dhz)
      : 0;
    gateAccelMps2 = Math.max(-25, Math.min(25, (vNext - vPrev) / (2 * dtAcc)));
    if (timing) timing.accelChirpMs += performance.now() - tAccel;
  }

  const classify = (r: number, dr: number, du: number, dv: number, tapNorm: number): number => {
    const px = beam.origin.x + dx * r + lx * du + beam.normal.x * dv;
    const py = beam.origin.y + dy * r + ly * du + beam.normal.y * dv;
    const pz = beam.origin.z + dz * r + lz * du + beam.normal.z * dv;
    const hx =
      (px - hf.origin.x) * hf.ex.x + (py - hf.origin.y) * hf.ex.y + (pz - hf.origin.z) * hf.ex.z;
    const hy =
      (px - hf.origin.x) * hf.ey.x + (py - hf.origin.y) * hf.ey.y + (pz - hf.origin.z) * hf.ey.z;
    const hz =
      (px - hf.origin.x) * hf.ez.x + (py - hf.origin.y) * hf.ez.y + (pz - hf.origin.z) * hf.ez.z;
    const cls = classifyScene(px, py, pz, tissueSample);
    const inScene = cls !== 0;
    const beamWeight =
      sampleVolumeWeight(r, focusCm, dr, du, dv, gateLenOrNull) * backscatterScale * tapNorm;
    if (isTdi) {
      if (inScene && tissueSample.tissue === Tissue.Myocardium) {
        // Myocardial parenchyma scatters ~30–40 dB more strongly than blood, keeping deep basal
        // annuli (11–13 cm in dilated LV) well above thermal noise across [-8, +8] dB gain.
        const power = received(r) ** 2 * beamWeight * 12;
        if (power <= 0) return cls;
        const tv = sampleTissueVelocity(
          ctx.heart,
          ctx.tables,
          phase,
          hx,
          hy,
          hz,
          tissueSample.structure,
        );
        const axial = tv.vx * dhx + tv.vy * dhy + tv.vz * dhz;
        const vPerp = Math.sqrt(
          Math.max(0, tv.vx * tv.vx + tv.vy * tv.vy + tv.vz * tv.vz - axial * axial),
        );
        samples.push({
          v: -axial,
          weight: power,
          dispersion: 0.035,
          vPerp,
          depthCm: r,
          focusCm,
          gateLengthCm: gateLenOrNull,
          frequencyMHz: f0MHz,
        });
      }
      return cls;
    }
    if (!inScene || tissueSample.tissue !== Tissue.Blood) return cls;
    // Split-aperture CW probes have a broad, weakly focused depth sensitivity compared to PW,
    // preventing deep valve jets (11–14 cm) from being crushed relative to proximal LVOT blood.
    const cwDepthBalance = isCw ? 1 / Math.max(0.45, focusingGain(r, focusCm)) : 1;
    const power = received(r) ** 2 * beamWeight * cwDepthBalance;
    if (power <= 0) return cls;
    sampleFlow(ctx.flow, ctx.tables, hp, phase, hx, hy, hz, flowSample);
    if (!flowSample.present) {
      samples.push({ v: 0, weight: 0.3 * power, dispersion: 0.05, depthCm: r });
      return cls;
    }
    const axial = flowSample.vx * dhx + flowSample.vy * dhy + flowSample.vz * dhz;
    const vPerp = Math.sqrt(
      Math.max(
        0,
        flowSample.vx * flowSample.vx +
          flowSample.vy * flowSample.vy +
          flowSample.vz * flowSample.vz -
          axial * axial,
      ),
    );
    // High-velocity stenotic/regurgitant jets (both laminar vena contracta cores and turbulent
    // free jets) occupy only ~0.5–1.5 cm of a 16 cm CW ray; weight high-velocity jet backscatter
    // so narrow eccentric MR/AS vena contractas remain cleanly resolved above slow cavity blood.
    const cwJetBoost = isCw
      ? 1 +
        2.8 *
          smoothstep(1.2, 2.8, Math.abs(axial)) *
          (0.65 + 0.35 * smoothstep(0.12, 0.45, flowSample.dispersion))
      : 1;
    samples.push({
      v: -axial,
      weight: power * cwJetBoost,
      dispersion: flowSample.dispersion,
      vPerp,
      depthCm: r,
      focusCm,
      gateLengthCm: gateLenOrNull,
      frequencyMHz: f0MHz,
      accelMps2: aliasing && !isTdi ? gateAccelMps2 : undefined,
    });
    return cls;
  };

  // Heart-frame points where the sample volume can meet a valve's leaflets: along the CW line, or the PW gate
  const clickPoints: number[] = [];
  const clickPowers: number[] = [];
  const clickPoint = (r: number): void => {
    const px = beam.origin.x + dx * r,
      py = beam.origin.y + dy * r,
      pz = beam.origin.z + dz * r;
    const point = [
      (px - hf.origin.x) * hf.ex.x + (py - hf.origin.y) * hf.ex.y + (pz - hf.origin.z) * hf.ex.z,
      (px - hf.origin.x) * hf.ey.x + (py - hf.origin.y) * hf.ey.y + (pz - hf.origin.z) * hf.ey.z,
      (px - hf.origin.x) * hf.ez.x + (py - hf.origin.y) * hf.ez.y + (pz - hf.origin.z) * hf.ez.z,
    ];
    if (
      valveClickWeight(ctx.heart, hp, ctx.tables, phase * ctx.tables.rrS, point) === 0 ||
      received(r) === 0
    )
      return;
    clickPoints.push(...point);
    clickPowers.push(received(r) ** 2);
  };

  const tQuad = timing ? performance.now() : 0;
  if (isCw) {
    // CW integrates along the entire focused beam with jittered 0.24 cm axial stations and 5-tap
    // 3D cross-beam quadrature (center + lateral + elevation) so narrow vena contractas (AS, MR,
    // HOCM SAM) are resolved smoothly on every column without axial aliasing.
    const rng = createRng(ctx.caseDef.seed ^ (col * 104729));
    const drStep = 0.24;
    const rJitter = rng.next() * drStep;
    for (let r = 1.0 + rJitter; r < spec.depthCm; r += drStep) {
      if (received(r) === 0) break;
      const wl = beamHalfWidthCm(r, focusCm);
      const we = sliceHalfWidthCm(r);
      const centerCls = classify(r, 0, 0, 0, 0.4);
      if (centerCls === 2) {
        classify(r, 0, -0.55 * wl, 0, 0.16);
        classify(r, 0, 0.55 * wl, 0, 0.16);
        classify(r, 0, 0, -0.5 * we, 0.14);
        classify(r, 0, 0, 0.5 * we, 0.14);
        clickPoint(r);
      } else if (
        centerCls === 1 &&
        (tissueSample.tissue === Tissue.Blood ||
          tissueSample.structure === Structure.DescendingAorta ||
          tissueSample.structure === Structure.PosteriorMediastinum ||
          tissueSample.structure === Structure.SuperiorMediastinum ||
          tissueSample.structure === Structure.PericardialFat ||
          tissueSample.structure === Structure.Diaphragm ||
          (tissueSample.structure === Structure.ChestWall &&
            tissueSample.sdf > -Math.max(0.55 * wl, 0.5 * we) - 0.1))
      ) {
        classify(r, 0, -0.55 * wl, 0, 0.16);
        classify(r, 0, 0.55 * wl, 0, 0.16);
        classify(r, 0, 0, -0.5 * we, 0.14);
        classify(r, 0, 0, 0.5 * we, 0.14);
      }
    }
  } else {
    const g = inp.spectral.gateLengthCm;
    if (inp.modality === 'pw')
      for (const k of [-0.5, 0, 0.5]) clickPoint(inp.gateDepthCm + k * g);
    // Stratified 3D sample-volume quadrature (4 axial x 3 lateral x 2 elevation = 24 jittered taps)
    // scaled by true depth-dependent lateral beam width and elevational slice thickness.
    const rng = createRng(ctx.caseDef.seed ^ (col * 7919));
    const nAx = 4;
    const nLat = 3;
    const nEl = 2;
    const invN = 1 / (nAx * nLat * nEl);
    for (let ia = 0; ia < nAx; ia++) {
      const dr = ((ia + rng.next()) / nAx - 0.5) * 1.4 * g;
      const r = Math.max(0.25, inp.gateDepthCm + dr);
      const wl = beamHalfWidthCm(r, focusCm);
      const we = sliceHalfWidthCm(r);
      for (let il = 0; il < nLat; il++) {
        const du = ((il + rng.next()) / nLat - 0.5) * 2.0 * wl;
        for (let ie = 0; ie < nEl; ie++) {
          const dv = ((ie + rng.next()) / nEl - 0.5) * 1.8 * we;
          classify(r, dr, du, dv, invN);
        }
      }
    }
  }
  if (timing) {
    timing.quadratureMs += performance.now() - tQuad;
    timing.samplesEvaluated += samples.length;
    timing.columns += 1;
  }

  const column = options?.outColumn ?? new Float32Array(SPECTRAL_BINS);
  const display = options?.outDisplay ?? new Float32Array(SPECTRAL_BINS);
  const tClicks = timing ? performance.now() : 0;
  const clickSig = clickPoints.length
    ? valveClickSignal(
        ctx.heart,
        hp,
        ctx.tables,
        phase * ctx.tables.rrS,
        clickPoints,
        { x: dhx, y: dhy, z: dhz },
        clickPowers,
      )
    : { weight: 0, direction: 0 };
  if (timing) timing.valveClicksMs += performance.now() - tClicks;

  if (aliasing) {
    const { effectiveScaleMps } = depthLimitedPulsedAcquisition(
      inp.gateDepthCm,
      inp.spectral.gateLengthCm,
      f0MHz,
      inp.spectral.scaleMps,
    );
    const maxShift = 0.85 * Math.max(0.01, effectiveScaleMps);
    const clampedShift = Math.max(-maxShift, Math.min(maxShift, inp.spectral.baselineShiftMps));
    const effectiveSpectral =
      effectiveScaleMps < inp.spectral.scaleMps || clampedShift !== inp.spectral.baselineShiftMps
        ? { ...inp.spectral, scaleMps: effectiveScaleMps, baselineShiftMps: clampedShift }
        : inp.spectral;
    const tSynth = timing ? performance.now() : 0;
    accumulatePulsedSpectrum(
      samples,
      effectiveSpectral,
      f0MHz,
      ctx.caseDef.seed,
      timeS,
      column,
      inp.gateDepthCm,
      focusCm,
    );
    if (timing) timing.spectrumSynthMs += performance.now() - tSynth;
    const tPres = timing ? performance.now() : 0;
    presentSpectralColumn(
      column,
      effectiveSpectral,
      stripHead,
      ctx.caseDef.seed,
      f0MHz,
      1 - DUPLEX_BMODE_SHARE,
      clickSig.weight,
      display,
      timeS,
      true,
      clickSig.direction,
    );
    if (timing) timing.spectrumPresentMs += performance.now() - tPres;
  } else {
    const tSynth = timing ? performance.now() : 0;
    accumulateSpectrum(samples, inp.spectral, aliasing, column);
    if (timing) timing.spectrumSynthMs += performance.now() - tSynth;
    if (isCw && options?.onRawCw) {
      options.onRawCw(column, clickPoints, clickPowers, { x: dhx, y: dhy, z: dhz });
    }
    const tPres = timing ? performance.now() : 0;
    presentSpectralColumn(
      column,
      inp.spectral,
      stripHead,
      ctx.caseDef.seed,
      f0MHz,
      1 - DUPLEX_BMODE_SHARE,
      clickSig.weight,
      display,
      timeS,
      true,
      clickSig.direction,
    );
    if (timing) timing.spectrumPresentMs += performance.now() - tPres;
  }
  return { column, display };
}

/** Structures at the Doppler/M-mode cursor and the beam-flow angle at the PW/TDI gate (technique checks). */
export function computeGateInfo(
  beam: BeamFrame,
  spec: PolarFrameSpec,
  phase: number,
  structure: Uint8Array,
  ctx: StripCtx,
  gateFlow: GateFlowCache,
  flowSample: FlowSample,
  tissueSample: TissueSample,
): GateInfo {
  const inp = ctx.input;
  const theta = Math.max(-spec.sectorRad / 2, Math.min(spec.sectorRad / 2, inp.cursorThetaRad));
  const li = Math.min(
    spec.lines - 1,
    Math.max(0, Math.round(((theta + spec.sectorRad / 2) / spec.sectorRad) * spec.lines - 0.5)),
  );
  const lineStructures: number[] = [];
  if (structure.length === spec.lines * spec.samples) {
    for (let si = 0; si < spec.samples; si++) {
      const st = structure[li * spec.samples + si]!;
      if (st !== 0 && !lineStructures.includes(st)) lineStructures.push(st);
    }
  }
  const ct = Math.cos(theta),
    sn = Math.sin(theta);
  const r = inp.gateDepthCm;
  const dx = beam.forward.x * ct + beam.lateral.x * sn;
  const dy = beam.forward.y * ct + beam.lateral.y * sn;
  const dz = beam.forward.z * ct + beam.lateral.z * sn;
  const px = beam.origin.x + dx * r,
    py = beam.origin.y + dy * r,
    pz = beam.origin.z + dz * r;
  const hf = ctx.heart.frame;
  const hx =
    (px - hf.origin.x) * hf.ex.x + (py - hf.origin.y) * hf.ex.y + (pz - hf.origin.z) * hf.ex.z;
  const hy =
    (px - hf.origin.x) * hf.ey.x + (py - hf.origin.y) * hf.ey.y + (pz - hf.origin.z) * hf.ey.z;
  const hz =
    (px - hf.origin.x) * hf.ez.x + (py - hf.origin.y) * hf.ez.y + (pz - hf.origin.z) * hf.ez.z;
  const inScene = sceneClassifier(ctx.scene(phase))(px, py, pz, tissueSample) !== 0;
  let flowPresent = false;
  let flowAngleDeg: number | null = null;
  if (inp.modality === 'cw') {
    const dhx = dx * hf.ex.x + dy * hf.ex.y + dz * hf.ex.z;
    const dhy = dx * hf.ey.x + dy * hf.ey.y + dz * hf.ey.z;
    const dhz = dx * hf.ez.x + dy * hf.ez.y + dz * hf.ez.z;
    const cwKey = `cw|${beam.origin.x.toFixed(4)},${beam.origin.y.toFixed(4)},${beam.origin.z.toFixed(4)}|${dhx.toFixed(5)},${dhy.toFixed(5)},${dhz.toFixed(5)}|${spec.depthCm.toFixed(2)}|${ctx.modelVersion}|${ctx.tablesVersion}`;
    if (gateFlow.key !== cwKey) {
      let best = 0;
      let bx = 0,
        by = 0,
        bz = 0;
      const oxH =
        (beam.origin.x - hf.origin.x) * hf.ex.x +
        (beam.origin.y - hf.origin.y) * hf.ex.y +
        (beam.origin.z - hf.origin.z) * hf.ex.z;
      const oyH =
        (beam.origin.x - hf.origin.x) * hf.ey.x +
        (beam.origin.y - hf.origin.y) * hf.ey.y +
        (beam.origin.z - hf.origin.z) * hf.ey.z;
      const ozH =
        (beam.origin.x - hf.origin.x) * hf.ez.x +
        (beam.origin.y - hf.origin.y) * hf.ez.y +
        (beam.origin.z - hf.origin.z) * hf.ez.z;
      for (let k = 0; k < 12; k++) {
        const ph = k / 12;
        const sc = ctx.scene(ph);
        const cls = sceneClassifier(sc);
        const hp = sc.heartPose;
        for (let rr = 1.5; rr <= spec.depthCm - 0.5; rr += 0.4) {
          const rx = beam.origin.x + dx * rr;
          const ry = beam.origin.y + dy * rr;
          const rz = beam.origin.z + dz * rr;
          if (cls(rx, ry, rz, tissueSample) === 0 || tissueSample.tissue !== Tissue.Blood) continue;
          sampleFlow(
            ctx.flow,
            ctx.tables,
            hp,
            ph,
            oxH + dhx * rr,
            oyH + dhy * rr,
            ozH + dhz * rr,
            flowSample,
          );
          if (!flowSample.present) continue;
          const sp = Math.hypot(flowSample.vx, flowSample.vy, flowSample.vz);
          if (sp > best) {
            best = sp;
            bx = flowSample.vx;
            by = flowSample.vy;
            bz = flowSample.vz;
          }
        }
      }
      gateFlow.key = cwKey;
      gateFlow.best = best;
      gateFlow.bx = bx;
      gateFlow.by = by;
      gateFlow.bz = bz;
      // Restore tissueSample at gate depth for consistent structure/tissue fields
      sceneClassifier(ctx.scene(phase))(px, py, pz, tissueSample);
    }
    if (gateFlow.best > 0.05) {
      flowPresent = true;
      const cos = Math.abs(
        (gateFlow.bx * dhx + gateFlow.by * dhy + gateFlow.bz * dhz) / gateFlow.best,
      );
      flowAngleDeg = (Math.acos(Math.min(1, cos)) * 180) / Math.PI;
    }
  } else if (inScene && tissueSample.tissue === Tissue.Blood) {
    // Cache peak flow vector over the cycle for this heart-frame gate position.
    const gateKey = `${hx.toFixed(5)},${hy.toFixed(5)},${hz.toFixed(5)}|${ctx.modelVersion}|${ctx.tablesVersion}`;
    if (gateFlow.key !== gateKey) {
      let best = 0;
      let bx = 0,
        by = 0,
        bz = 0;
      for (let k = 0; k < 16; k++) {
        const ph = k / 16;
        sampleFlow(
          ctx.flow,
          ctx.tables,
          computeHeartPose(ctx.heart, cycleStateAt(ctx.tables, ph, ctx.flow.flowPeaks)),
          ph,
          hx,
          hy,
          hz,
          flowSample,
        );
        const sp = Math.hypot(flowSample.vx, flowSample.vy, flowSample.vz);
        if (flowSample.present && sp > best) {
          best = sp;
          bx = flowSample.vx;
          by = flowSample.vy;
          bz = flowSample.vz;
        }
      }
      gateFlow.key = gateKey;
      gateFlow.best = best;
      gateFlow.bx = bx;
      gateFlow.by = by;
      gateFlow.bz = bz;
    }
    if (gateFlow.best > 0.05) {
      flowPresent = true;
      const dhx = dx * hf.ex.x + dy * hf.ex.y + dz * hf.ex.z;
      const dhy = dx * hf.ey.x + dy * hf.ey.y + dz * hf.ey.z;
      const dhz = dx * hf.ez.x + dy * hf.ez.y + dz * hf.ez.z;
      const cos = Math.abs((gateFlow.bx * dhx + gateFlow.by * dhy + gateFlow.bz * dhz) / gateFlow.best);
      flowAngleDeg = (Math.acos(Math.min(1, cos)) * 180) / Math.PI;
    }
  } else if (inScene && tissueSample.tissue === Tissue.Myocardium && inp.modality === 'tdi') {
    flowPresent = true;
    const dhz = dx * hf.ez.x + dy * hf.ez.y + dz * hf.ez.z;
    flowAngleDeg = (Math.acos(Math.min(1, Math.abs(dhz))) * 180) / Math.PI;
  }
  return {
    thetaRad: theta,
    depthCm: r,
    structure: inScene ? tissueSample.structure : 0,
    tissue: inScene ? tissueSample.tissue : 0,
    flowPresent,
    flowAngleDeg,
    lineStructures,
  };
}
