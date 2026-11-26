import type { PatientState } from '@/simulator/anatomy/thoraxModel';
import type { ColorSettings } from '@/simulator/doppler/color/colorDoppler';
import type { ProbeControl } from '@/simulator/probe/pose';
import type { AcquisitionSettings, PolarFrameSpec } from '@/simulator/renderer/types';
import type { SimInput } from './protocol';

/**
 * Single source of truth for deterministic cache/invalidation keys and the formal
 * Acquisition vs. Postprocess boundary across SimClient, SimulatorCore, ColorFieldDriver,
 * StripEngine, FrameCompositor, and main-thread case-model memoisation.
 *
 * Replaces ad-hoc `JSON.stringify(...)` calls in per-frame hot paths and guarantees that
 * presentation-only controls (e.g. `showVariance`, `tintMap`, and future `palette` / `reject`)
 * never trigger redundant Kasai re-shading or strip resets.
 */

/** Deterministic key for patient positioning and respiratory state. */
export function patientKey(p: PatientState): string {
  return `${p.position}|${p.respiration}|${p.headElevationDeg}`;
}

/** Deterministic key for transducer probe pose and pressure. */
export function probeKey(p: ProbeControl): string {
  return `${p.u}|${p.v}|${p.rotationDeg}|${p.tiltDeg}|${p.rockDeg}|${p.pressure}`;
}

/** Deterministic key for artifact laboratory overrides. */
export function artifactOverridesKey(o: SimInput['artifactOverrides']): string {
  if (!o) return 'none';
  return `${o.sideLobe}|${o.mirror}|${o.beamWidth}|${o.clutter}`;
}

/** Deterministic key for B-mode console & acquisition settings. */
export function acquisitionSettingsKey(s: AcquisitionSettings): string {
  const tgc = s.tgcDb;
  const tgcStr = `${tgc[0] ?? 0},${tgc[1] ?? 0},${tgc[2] ?? 0},${tgc[3] ?? 0},${tgc[4] ?? 0},${tgc[5] ?? 0},${tgc[6] ?? 0},${tgc[7] ?? 0}`;
  return `${s.depthCm}|${s.sectorDeg}|${s.gainDb}|${tgcStr}|${s.dynamicRangeDb}|${s.frequencyMHz}|${s.harmonics ? 1 : 0}|${s.focusCm}|${s.lineDensity}|${s.persistence}|${s.edgeEnhance}|${s.sri ?? ''}|${s.spatialCompound ?? ''}|${s.grayMap}|${s.tintMap ?? ''}|${s.invertLR ? 1 : 0}|${s.zoom}|${s.depthCompensationDbPerCmMHz ?? ''}`;
}

/**
 * Color Doppler acquisition-only key (box geometry + autocorrelation/wall-filter/gain/persistence).
 * Presentation-only controls (such as `showVariance`) are deliberately excluded so toggling them
 * performs zero Kasai re-shading and preserves temporal history.
 */
export function colorAcquisitionKey(c: ColorSettings): string {
  return `${c.boxThetaMinRad}|${c.boxThetaMaxRad}|${c.boxRMinCm}|${c.boxRMaxCm}|${c.scaleMps}|${c.baselineShiftMps}|${c.wallFilterMps}|${c.gainDb}|${c.persistence}|${c.smoothing}|${c.invert ? 1 : 0}`;
}

/** Per-geometry key for GPU raw-flow ROI reuse (probe, ROI box, acoustic frequency/harmonics, polar grid). */
export function colorGeometryKey(
  probe: ProbeControl,
  color: ColorSettings,
  frequencyMHz: number,
  harmonics: boolean,
  modelVersion: number,
  spec: PolarFrameSpec,
): string {
  return `${probeKey(probe)}|${color.boxThetaMinRad}|${color.boxThetaMaxRad}|${color.boxRMinCm}|${color.boxRMaxCm}|${frequencyMHz}|${harmonics ? 1 : 0}|${modelVersion}|${spec.lines}|${spec.samples}|${spec.depthCm}|${spec.sectorRad}`;
}

/** Combined geometry + acquisition key for the CPU fallback Color Doppler driver. */
export function colorCpuAcquisitionKey(
  probe: ProbeControl,
  color: ColorSettings,
  frequencyMHz: number,
  harmonics: boolean,
  modelVersion: number,
  spec: PolarFrameSpec,
): string {
  return `${colorGeometryKey(probe, color, frequencyMHz, harmonics, modelVersion, spec)}|${colorAcquisitionKey(color)}`;
}

/**
 * Resolve Color Doppler settings while frozen: acquisition-time controls remain locked to the
 * captured frame (`acquired`, coherent with `reconstructCineColor`), while presentation-only
 * postprocess controls (`palette`, `showVariance`, and circularly re-wrapped `baselineShiftMps`)
 * respond immediately to live console changes (`live`).
 */
export function resolveFrozenColorSettings(
  acquired: ColorSettings,
  live: ColorSettings,
): ColorSettings {
  const maxShift = 0.85 * Math.max(0.01, acquired.scaleMps);
  const baselineShiftMps = Math.max(-maxShift, Math.min(maxShift, live.baselineShiftMps));
  return {
    ...acquired,
    palette: live.palette ?? acquired.palette,
    showVariance: live.showVariance,
    baselineShiftMps,
  };
}

/**
 * Strip history acquisition key: only controls that change the acquired strip signal reset the
 * strip buffer. Modality-irrelevant and pure-presentation controls (`palette`, `showVariance`,
 * `spectral.reject`) do not wipe strip history.
 */
export function stripAcquisitionKey(inp: SimInput): string {
  const base = `${inp.modality}|${inp.spectral.sweepSpeedMmPerS}|${inp.settings.depthCm}|${inp.settings.frequencyMHz}|${inp.cursorThetaRad}`;
  if (inp.modality === 'm-mode') return base;
  if (inp.modality === 'cmm') {
    const c = inp.color;
    return `${base}|${c.scaleMps}|${c.baselineShiftMps}|${c.invert ? 1 : 0}|${c.wallFilterMps}|${c.gainDb}`;
  }
  const sp = inp.spectral;
  return `${base}|${inp.gateDepthCm}|${sp.scaleMps}|${sp.baselineShiftMps}|${sp.invert ? 1 : 0}|${sp.wallFilterMps}|${sp.gainDb}|${sp.gateLengthCm}`;
}

/** Fast deterministic key covering every field of `SimInput` (replaces `JSON.stringify` in `SimClient.send`). */
export function simInputKey(inp: SimInput): string {
  const c = inp.color;
  const sp = inp.spectral;
  return [
    probeKey(inp.probe),
    patientKey(inp.patient),
    acquisitionSettingsKey(inp.settings),
    inp.modality,
    inp.frozen ? 1 : 0,
    inp.cineOffset,
    `${colorAcquisitionKey(c)}|${c.showVariance ? 1 : 0}|${c.palette ?? 'vivid'}`,
    `${sp.scaleMps}|${sp.baselineShiftMps}|${sp.wallFilterMps}|${sp.gainDb}|${sp.sweepSpeedMmPerS}|${sp.gateLengthCm}|${sp.invert ? 1 : 0}|${sp.audioOn ? 1 : 0}|${sp.volume}|${sp.reject ?? 0}`,
    inp.cursorThetaRad,
    inp.gateDepthCm,
    inp.quality,
    `${inp.display.width}x${inp.display.height}`,
    inp.rendererBackend,
    inp.anatomy ?? 'analytic',
    inp.imageFormation ?? 'lattice',
    artifactOverridesKey(inp.artifactOverrides),
    inp.needSegments === undefined ? 'u' : inp.needSegments ? 1 : 0,
    inp.needStructure === undefined ? 'u' : inp.needStructure ? 1 : 0,
  ].join('#');
}
