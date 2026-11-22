import type { BeamFrame } from '@/simulator/probe/pose';
import type { HeartFrame } from '@/simulator/anatomy/heartFrame';
import { latticeNoise3 } from '@/core/noise';
import { formEnvelopeLine, LINE_LATTICE_PATH, type LineKernels } from '../acoustic/psf';
import {
  BLOOD_DECORRELATION_CELLS,
  GRAIN_OFFSET,
  PHASOR_IM_A,
  PHASOR_IM_B,
  PHASOR_NORM,
  PHASOR_RE_B,
  SCATTER_FREQ,
  SCATTER_FREQ_RATIO,
} from '../acoustic/acoustics';
import { grainCoef, torsoToHeartDir, type LineContext } from './beamMarch';

const [PRB_X, PRB_Y, PRB_Z] = PHASOR_RE_B;
const [PIA_X, PIA_Y, PIA_Z] = PHASOR_IM_A;
const [PIB_X, PIB_Y, PIB_Z] = PHASOR_IM_B;
const [GR_X, GR_Y, GR_Z] = GRAIN_OFFSET;
const [, PATH_Y, PATH_Z] = LINE_LATTICE_PATH;

/**
 * Reusable scratch buffers for M-mode single-line formation (decision 84).
 * Allocated only when the line sample count changes so steady-state M-mode
 * pulses incur zero typed-array allocations.
 */
export class MmodeLineBuffers {
  re = new Float32Array(0);
  im = new Float32Array(0);
  tmpRe = new Float32Array(0);
  tmpIm = new Float32Array(0);
  sigma = new Float32Array(0);
  across = new Float32Array(0);
  elevation = new Float32Array(0);
  along = new Float32Array(0);
  grainScale = new Float32Array(0);
  grainCells = new Float32Array(0);

  /** Resize buffers if needed, clear per-line accumulators, and build `LineContext['line']`. */
  prepare(
    samples: number,
    kernels: LineKernels,
    beam: BeamFrame,
    heartFrame: HeartFrame,
    theta: number,
    pulse: number,
  ): NonNullable<LineContext['line']> {
    if (this.re.length !== samples) {
      this.re = new Float32Array(samples);
      this.im = new Float32Array(samples);
      this.tmpRe = new Float32Array(samples);
      this.tmpIm = new Float32Array(samples);
      this.sigma = new Float32Array(samples);
      this.across = new Float32Array(samples);
      this.elevation = new Float32Array(samples);
      this.along = new Float32Array(samples);
      this.grainScale = new Float32Array(samples);
      this.grainCells = new Float32Array(samples);
    }
    this.sigma.fill(0);
    this.grainScale.fill(0);

    const ct = Math.cos(theta);
    const sn = Math.sin(theta);
    const lat = {
      x: beam.lateral.x * ct - beam.forward.x * sn,
      y: beam.lateral.y * ct - beam.forward.y * sn,
      z: beam.lateral.z * ct - beam.forward.z * sn,
    };
    const fwd = {
      x: beam.forward.x * ct + beam.lateral.x * sn,
      y: beam.forward.y * ct + beam.lateral.y * sn,
      z: beam.forward.z * ct + beam.lateral.z * sn,
    };
    const fH = torsoToHeartDir(heartFrame, fwd);
    const lH = torsoToHeartDir(heartFrame, lat);
    const nH = torsoToHeartDir(heartFrame, beam.normal);

    return {
      kernels,
      torsoAxes: [
        fwd.x,
        fwd.y,
        fwd.z,
        lat.x,
        lat.y,
        lat.z,
        beam.normal.x,
        beam.normal.y,
        beam.normal.z,
      ],
      heartAxes: [fH.x, fH.y, fH.z, lH.x, lH.y, lH.z, nH.x, nH.y, nH.z],
      // a new realization of the flowing blood in every pulse (decision 163)
      bloodCells: (pulse % 4096) * BLOOD_DECORRELATION_CELLS,
      sigma: this.sigma,
      across: this.across,
      elevation: this.elevation,
      along: this.along,
      grainScale: this.grainScale,
      grainCells: this.grainCells,
    };
  }
}

/**
 * Synthesizes per-structure moving speckle + parenchymal grains along one M-mode line and forms
 * its linear envelope.
 *
 * Each contiguous run of a heart structure and tissue along the line receives an axial coordinate
 * relative to the run's centre so its speckle travels with moving leaflets/walls rather than
 * streaking horizontally; static thorax tissue retains its material coordinate.
 */
export function synthesizeMmodeLine(
  bufs: MmodeLineBuffers,
  ctx: Pick<LineContext, 'latA' | 'latB' | 'latC'>,
  kernels: LineKernels,
  depthCm: number,
  samples: number,
  st: Uint8Array,
  ti: Uint8Array,
  amp: Float32Array,
): void {
  const { latA, latB, latC } = ctx;
  const R = SCATTER_FREQ_RATIO;
  const dr = depthCm / samples;
  const { re, im, sigma, across, elevation: elev, along, grainScale, grainCells } = bufs;

  for (let a = 0; a < samples; ) {
    let b = a + 1;
    while (b < samples && st[b] === st[a] && ti[b] === ti[a]) b++;
    const centre = (a + b) / 2;
    const offset = st[a]! * 7.31 + ti[a]! * 3.17;
    for (let i = a; i < b; i++) {
      const sg = sigma[i]!;
      if (sg === 0) continue;
      const uc = Number.isNaN(along[i]!) ? (i + 0.5 - centre) * dr : along[i]!;
      const u = uc * SCATTER_FREQ;
      const gs = grainScale[i]!;
      if (gs > 0) {
        // the grains of the run travel with it too; across the beam their cell is the beam width or the slice
        // thickness where those are coarser than the grain
        const gf = grainCells[i]!;
        const ug = uc * gf;
        re[i] =
          re[i]! +
          gs *
            grainCoef(
              ug + offset + GR_X,
              across[i]! * Math.min(1, gf / kernels.lateralCells[i]!) + ug * PATH_Y + GR_Y,
              elev[i]! * Math.min(1, gf / kernels.elevationCells[i]!) + ug * PATH_Z + GR_Z,
              latA,
              latB,
              latC,
            );
      }
      const qx = u + offset;
      const qy = across[i]! + u * PATH_Y;
      const qz = elev[i]! + u * PATH_Z;
      re[i] =
        re[i]! +
        sg *
          (latticeNoise3(qx, qy, qz, latA) +
            latticeNoise3(qx * R + PRB_X, qy * R + PRB_Y, qz * R + PRB_Z, latB) -
            1) *
          PHASOR_NORM;
      im[i] =
        im[i]! +
        sg *
          (latticeNoise3(qx + PIA_X, qy + PIA_Y, qz + PIA_Z, latC) +
            latticeNoise3(qx * R + PIB_X, qy * R + PIB_Y, qz * R + PIB_Z, latA) -
            1) *
          PHASOR_NORM;
    }
    a = b;
  }
  formEnvelopeLine(re, im, samples, kernels, amp, bufs.tmpRe, bufs.tmpIm);
}
