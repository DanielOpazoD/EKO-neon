import type { PolarFrameSpec } from '../types';
import { latticeNoise3 } from '@/core/noise';
import {
  ATTEN_NP_PER_DB,
  LOOK_SHIFT,
  PHASOR_NORM,
  PLEURA_AMP,
  PLEURA_BASE,
  PLEURA_DEPTH_FREQ,
  PLEURA_DIFFUSE_FLOOR,
  PLEURA_LINE_FREQ,
  PLEURA_SLOPE_WINDOW_CM,
  PLEURA_Z,
  pleuralCoherence,
  pleuralIncidenceCos,
  pleuralReverberation,
  REVERB_MOD_AMP,
  REVERB_MOD_BASE,
  REVERB_MOD_DEPTH_FREQ,
  REVERB_MOD_LINE_FREQ,
  REVERB_MOD_Z,
  REVERB_PHASOR_IM_A,
  REVERB_PHASOR_IM_B,
  REVERB_PHASOR_LINE_FREQ,
  REVERB_PHASOR_RE_A_Z,
  REVERB_PHASOR_RE_B,
  SCATTER_FREQ,
  SCATTER_FREQ_RATIO,
  SOFT_TISSUE_ATTEN_DB,
  TRANSMISSION_FLOOR,
} from '../acoustic/acoustics';
import { chestWallClutter, type LineContext } from './beamMarch';

const [RRB_X, RRB_Y, RRB_Z] = REVERB_PHASOR_RE_B;
const [RIA_X, RIA_Y, RIA_Z] = REVERB_PHASOR_IM_A;
const [RIB_X, RIB_Y, RIB_Z] = REVERB_PHASOR_IM_B;

/** The pleural line at sample `idx`: coherent, the same in every look; its specular share follows the incidence. */
export function pleuraEcho(
  ctx: LineContext,
  li: number,
  idx: number,
  r: number,
  entryT: number,
  coherence: number,
  looks: number,
  re: Float32Array,
  im: Float32Array,
): void {
  const lk = ctx.line ? ctx.line.kernels : null;
  const nFrame = ctx.spec.lines * ctx.spec.samples;
  let pleura =
    entryT *
    (PLEURA_BASE +
      PLEURA_AMP *
        latticeNoise3(li * PLEURA_LINE_FREQ, r * PLEURA_DEPTH_FREQ, PLEURA_Z, ctx.latA)) *
    (PLEURA_DIFFUSE_FLOOR + (1 - PLEURA_DIFFUSE_FLOOR) * coherence);
  if (lk) pleura *= lk.single;
  for (let k = 0; k < looks; k++) {
    re[k * nFrame + idx] = pleura;
    im[k * nFrame + idx] = 0;
  }
}

/** Reverberation behind the pleura at sample `idx`: incoherent, a phasor tied to the line and the depth per look. */
export function lungEcho(
  ctx: LineContext,
  li: number,
  idx: number,
  r: number,
  entryR: number,
  entryT: number,
  coherence: number,
  looks: number,
  re: Float32Array,
  im: Float32Array,
  clutterEntryT: number,
  clutterScale: number,
  clut: { re: Float32Array; im: Float32Array } | null = null,
): void {
  const { latA, latB, latC } = ctx;
  const incAxial = ctx.line ? ctx.line.kernels.incoherentAxial : 1;
  const nFrame = ctx.spec.lines * ctx.spec.samples;
  const R = SCATTER_FREQ_RATIO;
  const nn =
    REVERB_MOD_BASE +
    REVERB_MOD_AMP *
      latticeNoise3(li * REVERB_MOD_LINE_FREQ, r * REVERB_MOD_DEPTH_FREQ, REVERB_MOD_Z, latC);
  const softT = Math.exp(-ATTEN_NP_PER_DB * SOFT_TISSUE_ATTEN_DB * ctx.fAtten * r);
  const cm = ctx.clutter > 0 ? chestWallClutter(ctx, li, r) : 0;
  const pleural = pleuralReverberation(r, entryR, entryT, nn, coherence, ctx.fAtten);
  const a = clut ? pleural : pleural + cm * Math.min(softT, clutterEntryT) * clutterScale;
  const px2 = li * REVERB_PHASOR_LINE_FREQ,
    pr = r * SCATTER_FREQ;
  for (let k = 0; k < looks; k++) {
    const sx = LOOK_SHIFT[k]![0],
      sy = LOOK_SHIFT[k]![1],
      sz = LOOK_SHIFT[k]![2];
    const zr =
      (latticeNoise3(px2 + sx, pr + sy, REVERB_PHASOR_RE_A_Z + sz, latA) +
        latticeNoise3(px2 + RRB_X + sx, pr * R + RRB_Y + sy, RRB_Z + sz, latB) -
        1) *
      PHASOR_NORM *
      incAxial;
    const zi =
      (latticeNoise3(px2 + RIA_X + sx, pr + RIA_Y + sy, RIA_Z + sz, latC) +
        latticeNoise3(px2 + RIB_X + sx, pr * R + RIB_Y + sy, RIB_Z + sz, latA) -
        1) *
      PHASOR_NORM *
      incAxial;
    re[k * nFrame + idx] = a * zr;
    im[k * nFrame + idx] = a * zi;
    if (clut) {
      clut.re[k * nFrame + idx] = cm * zr;
      clut.im[k * nFrame + idx] = cm * zi;
    }
  }
}

/**
 * The pleura and the reverberations of a frame (decision 221): how square the pleura stands to a line comes from the
 * entries of its neighbours, each taken within PLEURA_SLOPE_WINDOW_CM of its own.
 */
export function drawLungFrame(
  ctx: LineContext,
  spec: PolarFrameSpec,
  lungEntry: Int32Array,
  looks: number,
  re: Float32Array,
  im: Float32Array,
  tr: Float32Array,
  clutRe: Float32Array,
  clutIm: Float32Array,
): void {
  const { dr } = ctx;
  const lines = spec.lines,
    samples = spec.samples;
  const w = Math.floor(PLEURA_SLOPE_WINDOW_CM / dr + 0.5);
  const dTheta = spec.sectorRad / lines;
  const clut = { re: clutRe, im: clutIm };
  for (let li = 0; li < lines; li++) {
    const e = lungEntry[li]!;
    if (e < 0) continue;
    const near = (j: number): number => {
      const ej = lungEntry[j]!;
      return ej < 0 ? e + w : Math.min(e + w, Math.max(e - w, ej));
    };
    let dEntry = 0,
      span = 1;
    if (li > 0 && li < lines - 1) {
      dEntry = near(li + 1) - near(li - 1);
      span = 2;
    } else if (li > 0) dEntry = e - near(li - 1);
    else if (li < lines - 1) dEntry = near(li + 1) - e;
    const rE = (e + 0.5) * dr;
    const coherence = pleuralCoherence(pleuralIncidenceCos(dEntry * dr, span * rE * dTheta));
    const base = li * samples;
    pleuraEcho(ctx, li, base + e, rE, 1, coherence, looks, re, im);
    const tE = Math.max(tr[base + e]!, TRANSMISSION_FLOOR);
    for (let si = e + 1; si < samples; si++)
      lungEcho(
        ctx,
        li,
        base + si,
        (si + 0.5) * dr,
        rE,
        1,
        coherence,
        looks,
        re,
        im,
        tE,
        1 / tE,
        clut,
      );
  }
}
