import type { BeamFrame } from '@/simulator/probe/pose';
import type { PolarFrameSpec, Scene } from '../types';
import { Tissue } from '@/simulator/anatomy/tissue';
import { latticeNoise3 } from '@/core/noise';
import { hash3 } from '@/core/random';
import type { LineKernels } from '../acoustic/psf';
import {
  ATTEN_NP_PER_DB,
  BEAM_ATTEN_MAX_LINES,
  beamAttenWindowLines,
  CLUTTER_AMP,
  CLUTTER_BASE,
  CLUTTER_DECAY_CM,
  CLUTTER_HAZE,
  CLUTTER_MAX_CM,
  CLUTTER_MOD_DEPTH_FREQ,
  CLUTTER_MOD_FREQ,
  CLUTTER_MOD_LINE_FREQ,
  GRAIN_THRESHOLD,
  SOFT_TISSUE_ATTEN_DB,
  TRANSMISSION_FLOOR,
} from '../acoustic/acoustics';

export interface LineContext {
  scene: Scene;
  beam: BeamFrame;
  spec: PolarFrameSpec;
  dr: number;
  fAtten: number;
  seed: number;
  harm: boolean;
  clutter: number;
  windowAttenuation: number;
  contact: number;
  fwdH: { x: number; y: number; z: number };
  latH: { x: number; y: number; z: number };
  nrmH: { x: number; y: number; z: number };
  /** Lattice shift of the flowing blood's scatterers in this frame (decision 163). */
  bloodShift: number;
  /** Multi-angle spatial compounding strength (0..1). */
  spatialCompound: number;
  /** Hemodynamic LA/LAA spontaneous echo contrast intensity (0..1). */
  secLa: number;
  /** Hemodynamic LV mid-apical spontaneous echo contrast intensity (0..1). */
  secLv: number;
  latA: Uint8Array;
  latB: Uint8Array;
  latC: Uint8Array;
  /** Set only for an M-mode line (decision 84). */
  line: {
    kernels: LineKernels;
    /** Beam axes (along, across in the scan plane, elevation) in torso and heart coordinates, 3 × 3 flattened. */
    torsoAxes: number[];
    heartAxes: number[];
    /** Lattice offset of flowing blood for this pulse: its speckle does not persist from one pulse to the next. */
    bloodCells: number;
    sigma: Float32Array;
    across: Float32Array;
    elevation: Float32Array;
    along: Float32Array;
    grainScale: Float32Array;
    grainCells: Float32Array;
  } | null;
}

/**
 * The chest wall's reverberation at depth `r` on line `li`, times the scene's clutter level (decisions 238 and 258): the
 * near-field term and the haze of the cavities. It is an artifact of the wall, so it appears at that depth whatever lies
 * there, the lung included (decision 270).
 */
export function chestWallClutter(ctx: LineContext, li: number, r: number): number {
  const { beam } = ctx;
  const ox = beam.origin.x,
    oy = beam.origin.y,
    oz = beam.origin.z;
  let cm = 0;
  if (r < CLUTTER_MAX_CM)
    cm =
      Math.exp(-r / CLUTTER_DECAY_CM) *
      (CLUTTER_BASE +
        CLUTTER_AMP *
          latticeNoise3(
            ox * CLUTTER_MOD_FREQ + li * CLUTTER_MOD_LINE_FREQ,
            oy * CLUTTER_MOD_FREQ + oz * CLUTTER_MOD_FREQ,
            r * CLUTTER_MOD_DEPTH_FREQ,
            ctx.latC,
          ));
  cm += CLUTTER_HAZE;
  return cm * ctx.clutter;
}

/**
 * The loss the chest wall's reverberation carries to a sample (decision 312): the wall's own loss, at most the soft-tissue
 * one, until the line reaches the heart (`wall` < 0), and past the heart's surface the soft-tissue loss times what it had
 * there over the soft-tissue loss there (`wall`).
 */
export function clutterLoss(transmission: number, soft: number, wall: number): number {
  return wall < 0 ? Math.min(transmission, soft) : soft * wall;
}

/**
 * The march of the beam (decisions 144 and 299). Each line marches its own pencil, and the transmission a line's echo pays
 * at a depth is the arithmetic mean of those pencil transmissions over the aperture window (O(1) via prefix sums).
 */
export function beamMarch(
  re: Float32Array,
  im: Float32Array,
  tr: Float32Array,
  ti: Uint8Array,
  atten: Float32Array,
  spec: PolarFrameSpec,
  contact: number,
  seed: number,
  prefix: Float32Array,
  looks = 1,
  clutter: {
    re: Float32Array;
    im: Float32Array;
    heart: Uint8Array;
    fAtten: number;
  } | null = null,
  spatialCompound = 0,
): void {
  const { lines, samples } = spec;
  const n = lines * samples;
  const dr = spec.depthCm / samples;
  const dTheta = spec.sectorRad / lines;
  // 1. each line's pencil: the transmission reaching each body-tissue sample, without the contact drop-out, kept in `tr`
  for (let li = 0; li < lines; li++) {
    let t = 1;
    for (let si = 0; si < samples; si++) {
      const i = li * samples + si;
      const tis = ti[i]!;
      if (tis === Tissue.None || tis === Tissue.Lung) continue;
      tr[i] = t;
      t *= Math.exp(-atten[i]!);
      if (t < TRANSMISSION_FLOOR) t = TRANSMISSION_FLOOR;
    }
  }
  // 2. the window mean of the pencils at each depth, over the body-tissue samples, replaces them in place
  const count = new Int32Array(lines + 1);
  for (let si = 0; si < samples; si++) {
    const r = (si + 0.5) * dr;
    const baseK = beamAttenWindowLines(r, spec.focusCm, dTheta);
    const K =
      spatialCompound > 0
        ? Math.min(BEAM_ATTEN_MAX_LINES, Math.floor(baseK * (1 + 0.55 * spatialCompound) + 0.5))
        : baseK;
    let acc = 0;
    let cnt = 0;
    prefix[si] = 0;
    count[0] = 0;
    for (let li = 0; li < lines; li++) {
      const i = li * samples + si;
      const t = ti[i]!;
      if (t !== Tissue.None && t !== Tissue.Lung) {
        acc += tr[i]!;
        cnt++;
      }
      prefix[(li + 1) * samples + si] = acc;
      count[li + 1] = cnt;
    }
    for (let li = 0; li < lines; li++) {
      const i = li * samples + si;
      const t = ti[i]!;
      if (t === Tissue.None || t === Tissue.Lung) continue;
      const a = Math.max(0, li - K),
        b = Math.min(lines, li + K + 1);
      const m = count[b]! - count[a]!;
      if (m > 0) tr[i] = (prefix[b * samples + si]! - prefix[a * samples + si]!) / m;
    }
  }
  // 3. the line's own contact, and the echoes scaled
  for (let li = 0; li < lines; li++) {
    const base = li * samples;
    const drop = hash3(li, 7, 0, seed) > contact ? 0.08 : 1;
    let transmission = drop;
    let lungScale = -1;
    let soft = drop;
    let wall = -1;
    const softStep = clutter
      ? Math.exp(-ATTEN_NP_PER_DB * SOFT_TISSUE_ATTEN_DB * clutter.fAtten * dr)
      : 1;
    for (let si = 0; si < samples; si++) {
      const i = base + si;
      const t = ti[i]!;
      if (t === Tissue.None) {
        tr[i] = transmission;
        continue;
      }
      if (t === Tissue.Lung) {
        if (lungScale < 0) {
          lungScale = transmission;
          tr[i] = transmission;
        } else tr[i] = 0;
        let loss = 0;
        if (clutter) {
          const softR = Math.exp(
            -ATTEN_NP_PER_DB * SOFT_TISSUE_ATTEN_DB * clutter.fAtten * (si + 0.5) * dr,
          );
          loss = wall < 0 ? Math.min(softR, lungScale) : drop * softR * wall;
        }
        for (let k = 0; k < looks; k++) {
          re[k * n + i] =
            re[k * n + i]! * lungScale + (clutter ? clutter.re[k * n + i]! * loss : 0);
          im[k * n + i] =
            im[k * n + i]! * lungScale + (clutter ? clutter.im[k * n + i]! * loss : 0);
        }
        continue;
      }
      transmission = Math.max(TRANSMISSION_FLOOR, drop * tr[i]!);
      tr[i] = transmission;
      let loss = 0;
      if (clutter) {
        if (clutter.heart[i] && wall < 0) wall = Math.min(1, transmission / soft);
        loss = clutterLoss(transmission, soft, wall);
        soft *= softStep;
        if (soft < TRANSMISSION_FLOOR) soft = TRANSMISSION_FLOOR;
      }
      for (let k = 0; k < looks; k++) {
        re[k * n + i] =
          re[k * n + i]! * transmission + (clutter ? clutter.re[k * n + i]! * loss : 0);
        im[k * n + i] =
          im[k * n + i]! * transmission + (clutter ? clutter.im[k * n + i]! * loss : 0);
      }
    }
  }
}

/** A grain field above its threshold, 0–1. */
const grainAbove = (g: number): number =>
  g > GRAIN_THRESHOLD ? (g - GRAIN_THRESHOLD) / (1 - GRAIN_THRESHOLD) : 0;

/**
 * Coefficient (0–1) of the parenchymal grains at a lattice coordinate (decision 145).
 */
export function grainCoef(
  x: number,
  y: number,
  z: number,
  latA: Uint8Array,
  latB: Uint8Array,
  latC: Uint8Array,
): number {
  return (
    0.25 * grainAbove(latticeNoise3(x, y, z, latA)) +
    0.5 * grainAbove(latticeNoise3(x, y, z, latB)) +
    0.25 * grainAbove(latticeNoise3(x, y, z, latC))
  );
}

export function torsoToHeartDir(
  f: {
    ex: { x: number; y: number; z: number };
    ey: { x: number; y: number; z: number };
    ez: { x: number; y: number; z: number };
  },
  d: { x: number; y: number; z: number },
) {
  return {
    x: d.x * f.ex.x + d.y * f.ex.y + d.z * f.ex.z,
    y: d.x * f.ey.x + d.y * f.ey.y + d.z * f.ey.z,
    z: d.x * f.ez.x + d.y * f.ez.y + d.z * f.ez.z,
  };
}
