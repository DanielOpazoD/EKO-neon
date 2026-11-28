import type { BeamFrame } from '@/simulator/probe/pose';
import type { ImageFormation, PolarFrame, PolarFrameSpec, RendererBackend, Scene } from '../types';
import type { Phasor } from '../acoustic/scatterers';
import { sceneClassifier } from '@/simulator/anatomy/sceneClassifier';
import {
  makeSample,
  TISSUE_PROPS,
  Tissue,
  Structure,
  type TissueSample,
} from '@/simulator/anatomy/tissue';
import { latticeNoise3, sceneLattice } from '@/core/noise';
import { hash3 } from '@/core/random';
import { contactQuality } from '@/simulator/probe/pose';
import {
  buildLineKernels,
  buildPsfKernels,
  formEnvelope,
  lineKernelKey,
  psfKey,
  sideLobeLevelDb,
  sliceHalfWidthCm,
  type LineKernels,
  type PsfKernels,
} from '../acoustic/psf';
import {
  ATTEN_NP_PER_DB,
  attenuationFrequencyMHz,
  CHEST_WALL_ATTEN_DB,
  SOFT_TISSUE_ATTEN_DB,
  BLOOD_HARMONIC_SIGMA,
  bloodShiftCells,
  SEC_BASE_FLOOR,
  SEC_DRIFT_X,
  SEC_DRIFT_Y,
  SEC_FREQ,
  SEC_GAIN,
  SEC_LAA_BOOST,
  secNow,
  secOrificeKeep,
  CALCIUM_AMP,
  calciumAttenDb,
  CALCIUM_BASE,
  CALCIUM_FREQ,
  CALCIUM_GAIN,
  CALCIUM_OFFSET,
  CLUTTER_FREQ,
  CLUTTER_IM_A,
  CLUTTER_IM_B,
  CLUTTER_LINE_FREQ,
  CLUTTER_RE_A_X,
  CLUTTER_RE_B,
  HETERO_FREQ,
  HETERO_OFFSET,
  heteroDb,
  MYO_ANISO_FLOOR,
  MYO_ANISO_RADIAL_EPS,
  myoHelixGain,
  papillaryFibreGain,
  PHASOR_IM_A,
  PHASOR_IM_B,
  PHASOR_NORM,
  PHASOR_RE_B,
  RINGDOWN_CM,
  RINGDOWN_GAIN,
  focusingGain,
  membraneWeight,
  cordAlignment,
  cordWeight,
  COMPOUND_LOOKS,
  LOOK_SHIFT,
  GRAIN_GAIN,
  GRAIN_OFFSET,
  SCATTER_FREQ,
  SCATTER_FREQ_RATIO,
  SPECULAR_GAIN,
  SPECULAR_HARMONIC,
  SPEC_TAIL_DECAY_CM,
  interfaceEcho,
  interfaceTailSource,
  TRANSMISSION_FLOOR,
  ELEVATION_SIGMA_PER_HALF_WIDTH,
  STRONG_SCATTERER_FRACTION,
} from '../acoustic/acoustics';
import {
  beamMarch,
  chestWallClutter,
  clutterLoss,
  grainCoef,
  torsoToHeartDir,
  type LineContext,
} from './beamMarch';
import { drawLungFrame, lungEcho, pleuraEcho } from './lungEcho';
import { MmodeLineBuffers, synthesizeMmodeLine } from './mmodeScatter';

export { beamMarch, clutterLoss };

// lattice offsets as scalars: the inner loops read them per sample
const [PRB_X, PRB_Y, PRB_Z] = PHASOR_RE_B;
const [PIA_X, PIA_Y, PIA_Z] = PHASOR_IM_A;
const [PIB_X, PIB_Y, PIB_Z] = PHASOR_IM_B;
const [HET_X, HET_Y, HET_Z] = HETERO_OFFSET;
const [GR_X, GR_Y, GR_Z] = GRAIN_OFFSET;
const [CAL_X, CAL_Y, CAL_Z] = CALCIUM_OFFSET;
const [CRB_X, CRB_Y] = CLUTTER_RE_B;
const [CIA_X, CIA_Y, CIA_Z] = CLUTTER_IM_A;
const [CIB_X, CIB_Y, CIB_Z] = CLUTTER_IM_B;

/**
 * The speckle lattices A, B and C a scene's seed gives the slice renderer: the scene's lattices 0-2 (core/noise.ts
 * sceneLattice), which the GPU uploads as channels 0-2 of its noise texture (glHelpers.ts noiseLatticeTexels).
 */
export function speckleLattices(seed: number): Pick<LineContext, 'latA' | 'latB' | 'latC'> {
  return { latA: sceneLattice(seed, 0), latB: sceneLattice(seed, 1), latC: sceneLattice(seed, 2) };
}

/**
 * Procedural slice renderer: marches every scanline through the parametric thorax + heart model and forms
 * the image the way a scanner does (decision 52). Per sample it computes the incoherent tissue backscatter σ
 * (tissue reflectivity × myocardial anisotropy × heterogeneity), the coherent specular echo of an interface
 * crossing the sample (∝ |n·d|⁴) and a complex scatterer phasor anchored in tissue coordinates; the complex
 * signal σ·z + specular is multiplied by the two-way frequency-dependent transmission (shadowing), pleural
 * reverberation and near-field clutter are added, and the separable PSF + envelope detection of
 * `acoustic/psf.ts` produces the linear amplitude. No wave propagation, no RF carrier.
 *
 * Output amplitude is linear and *pre-console*: gain/TGC/compression/persistence are applied later
 * so every console control remains causal on this same frame.
 */
export type { ImageFormation };
export const LATTICE_FORMATION: ImageFormation = { looks: COMPOUND_LOOKS };

export class ProceduralSliceRenderer implements RendererBackend {
  readonly id = 'procedural' as const;
  /** The formation of the frame being formed: the scene's when it brings one (decision 354), else the tracer's own. */
  private active: ImageFormation;

  constructor(private readonly formation: ImageFormation = LATTICE_FORMATION) {
    this.active = formation;
  }
  private phasor: Phasor = { re: 0, im: 0 };
  private sample = makeSample();
  private sample2 = makeSample();
  private lastMs = 0;
  private lastSamples = 0;
  private re = new Float32Array(0);
  private im = new Float32Array(0);
  private tmpRe = new Float32Array(0);
  private tmpIm = new Float32Array(0);
  private tmpAmp = new Float32Array(0);
  /** Attenuation increment of each sample (Np), for the beam-averaged march of a frame (decision 144). */
  private atten = new Float32Array(0);
  private prefix = new Float32Array(0);
  /** The chest wall's reverberation of each sample and look before its loss, and whether the sample is heart (decision 312). */
  private clutRe = new Float32Array(0);
  private clutIm = new Float32Array(0);
  private heartFlag = new Uint8Array(0);
  /** Sample of each frame line where it enters lung, −1 where it does not (decision 221). */
  private lungEntry = new Int32Array(0);
  private mmode = new MmodeLineBuffers();
  private psf: PsfKernels | null = null;
  private linePsf: LineKernels | null = null;
  private acA = { sigma: 0, spec: 0 };
  private acB = { sigma: 0, spec: 0 };

  stats(): Record<string, number | string> {
    return { renderMs: Number(this.lastMs.toFixed(2)), samplesPerFrame: this.lastSamples };
  }
  dispose(): void {}

  /** PSF kernels for this frame geometry and probe settings (also uploaded by the GPU port). */
  kernels(scene: Scene, spec: PolarFrameSpec): PsfKernels {
    const f = scene.imageFormation ?? this.formation;
    if (f.kernels) return f.kernels(spec, scene.physics.frequencyMHz, scene.physics.harmonics);
    const { frequencyMHz, harmonics } = scene.physics;
    const bw = scene.physics.beamWidth ?? 0;
    const lobes = sideLobeLevelDb(scene.physics.sideLobe);
    if (!this.psf || this.psf.key !== psfKey(spec, frequencyMHz, harmonics, bw, lobes))
      this.psf = buildPsfKernels(spec, frequencyMHz, harmonics, bw, lobes);
    return this.psf;
  }

  render(
    scene: Scene,
    beam: BeamFrame,
    spec: PolarFrameSpec,
    _phase: number,
    out: PolarFrame,
  ): void {
    const t0 = performance.now();
    const n = spec.lines * spec.samples;
    this.active = scene.imageFormation ?? this.formation;
    const looks = this.active.looks;
    if (this.re.length !== n * looks) {
      // one complex signal per compounding look (decision 145)
      this.re = new Float32Array(n * looks);
      this.im = new Float32Array(n * looks);
      this.tmpRe = new Float32Array(n);
      this.tmpIm = new Float32Array(n);
      this.tmpAmp = new Float32Array(n);
      this.atten = new Float32Array(n);
      this.prefix = new Float32Array((spec.lines + 1) * spec.samples);
      this.clutRe = new Float32Array(n * looks);
      this.clutIm = new Float32Array(n * looks);
      this.heartFlag = new Uint8Array(n);
    }
    this.clutRe.fill(0);
    this.clutIm.fill(0);
    this.heartFlag.fill(0);
    const ctx = this.prepare(scene, beam, spec);
    if (this.lungEntry.length !== spec.lines) this.lungEntry = new Int32Array(spec.lines);
    this.lungEntry.fill(-1);
    for (let li = 0; li < spec.lines; li++) {
      const theta = -spec.sectorRad / 2 + (spec.sectorRad * (li + 0.5)) / spec.lines;
      this.renderLine(
        ctx,
        theta,
        li,
        li * spec.samples,
        this.re,
        this.im,
        out.structure,
        out.transmission,
        out.tissue,
        this.atten,
        out.segment,
        this.clutRe,
        this.clutIm,
        this.heartFlag,
      );
    }
    drawLungFrame(
      ctx,
      spec,
      this.lungEntry,
      this.active.looks,
      this.re,
      this.im,
      out.transmission,
      this.clutRe,
      this.clutIm,
    );
    const kernels = this.kernels(scene, spec);
    // the march of the beam, not of the pencil line (decision 144): the echoes scaled by its transmission
    beamMarch(
      this.re,
      this.im,
      out.transmission,
      out.tissue,
      this.atten,
      spec,
      ctx.contact,
      ctx.seed,
      this.prefix,
      looks,
      { re: this.clutRe, im: this.clutIm, heart: this.heartFlag, fAtten: ctx.fAtten },
      ctx.spatialCompound,
    );
    // the looks are detected one by one and their envelopes averaged (compounding, decision 145)
    out.amplitude.fill(0);
    for (let k = 0; k < looks; k++) {
      formEnvelope(
        this.re.subarray(k * n, (k + 1) * n),
        this.im.subarray(k * n, (k + 1) * n),
        spec.lines,
        spec.samples,
        kernels,
        this.tmpAmp,
        this.tmpRe,
        this.tmpIm,
      );
      const amp = out.amplitude,
        ta = this.tmpAmp,
        w = 1 / looks;
      for (let i = 0; i < n; i++) amp[i]! += ta[i]! * w;
    }
    this.lastMs = performance.now() - t0;
    this.lastSamples = n;
  }

  /** Kernels and level scales of an M-mode line of `samples` samples drawn under the frame geometry `frame` (decision 84). */
  lineKernels(scene: Scene, frame: PolarFrameSpec, samples: number): LineKernels {
    const { frequencyMHz, harmonics } = scene.physics;
    const bw = scene.physics.beamWidth ?? 0;
    if (
      !this.linePsf ||
      this.linePsf.key !== lineKernelKey(frame, samples, frequencyMHz, harmonics, bw)
    )
      this.linePsf = buildLineKernels(frame, samples, frequencyMHz, harmonics, bw);
    return this.linePsf;
  }

  /**
   * One M-mode line at angle theta (rad) through `samples` samples over the frame depth (decision 84). The line is finer
   * than a frame line so an echo moves continuously between columns, keeps the frame's levels (`LineKernels`), and draws
   * its scatterer phasor on a lattice aligned with the beam whose cell across the beam is the beam width: a line has no
   * lateral neighbours to average, and tissue sliding through it decorrelates over the beam and not over a 0.4 mm cell.
   * Blood is the exception: its scatterers flow a scatterer cell in a pulse interval or two (0.4 mm at 20 cm/s in 2 ms), so
   * its phasor is drawn anew for every `pulse` instead of streaking the cavity like still tissue. The slice-thickness side
   * planes are not sampled.
   */
  renderMmodeLine(
    scene: Scene,
    beam: BeamFrame,
    frame: PolarFrameSpec,
    samples: number,
    theta: number,
    pulse: number,
    amp: Float32Array,
    st: Uint8Array,
    tr: Float32Array,
    ti: Uint8Array,
  ): void {
    const spec: PolarFrameSpec = { ...frame, samples, elevationSamples: 1 };
    const k = this.lineKernels(scene, frame, samples);
    const ctx = this.prepare(scene, beam, spec);
    ctx.line = this.mmode.prepare(samples, k, beam, scene.heart.frame, theta, pulse);
    const li = Math.round(((theta + frame.sectorRad / 2) / frame.sectorRad) * frame.lines);
    this.renderLine(ctx, theta, li, 0, this.mmode.re, this.mmode.im, st, tr, ti);
    synthesizeMmodeLine(this.mmode, ctx, k, frame.depthCm, samples, st, ti, amp);
  }

  private prepare(scene: Scene, beam: BeamFrame, spec: PolarFrameSpec): LineContext {
    const { heart, physics } = scene;
    const f = physics.frequencyMHz;
    const harm = physics.harmonics;
    const hf = heart.frame;
    const sec = secNow(
      physics.secLa ?? 0,
      physics.secLv ?? 0,
      scene.heartPose.state.mitralFlowMlps,
      scene.heartPose.state.aorticFlowMlps,
    );
    return {
      scene,
      beam,
      spec,
      dr: spec.depthCm / spec.samples,
      fAtten: attenuationFrequencyMHz(f, harm),
      seed: physics.seed,
      harm,
      clutter:
        (physics.clutterLevel * 2.5 + physics.windowAttenuation * 0.8) *
        (harm ? 0.35 : 1) *
        Math.sqrt(2.5 / f),
      windowAttenuation: physics.windowAttenuation,
      contact: contactQuality(beam.contact),
      fwdH: torsoToHeartDir(hf, beam.forward),
      latH: torsoToHeartDir(hf, beam.lateral),
      nrmH: torsoToHeartDir(hf, beam.normal),
      bloodShift: bloodShiftCells(physics.bloodFrame ?? 0),
      spatialCompound: Math.max(0, Math.min(1, physics.spatialCompound ?? 0)),
      secLa: sec.secLa,
      secLv: sec.secLv,
      secShear: sec.shear,
      ...speckleLattices(physics.seed),
      line: null,
    };
  }

  private renderLine(
    ctx: LineContext,
    theta: number,
    li: number,
    base: number,
    re: Float32Array,
    im: Float32Array,
    st: Uint8Array,
    tr: Float32Array,
    ti: Uint8Array,
    atten: Float32Array | null = null,
    sg: Uint8Array | null = null,
    clutRe: Float32Array | null = null,
    clutIm: Float32Array | null = null,
    heartFlag: Uint8Array | null = null,
  ): void {
    const {
      beam,
      spec,
      dr,
      fAtten,
      seed,
      harm,
      clutter,
      contact,
      fwdH,
      latH,
      nrmH,
      latA,
      latB,
      latC,
      line,
    } = ctx;
    // M-mode line scales (decision 84); a frame line uses 1, which leaves its arithmetic unchanged
    const lk = line ? line.kernels : null;
    const inc = lk ? lk.incoherent : 1;
    const incAxial = lk ? lk.incoherentAxial : 1;
    const { heart } = ctx.scene;
    // the LV myocardium's backscatter over normal, as an amplitude (decision 267)
    const myoIb = Math.pow(10, heart.anatomy.lv.myocardialBackscatterDb / 20);
    const s = this.sample;
    const s2 = this.sample2;
    const acA = this.acA;
    const acB = this.acB;
    const samples = spec.samples;
    const nElev = spec.elevationSamples;
    const focus = spec.focusCm;
    // compounding looks of a frame (decision 145); an M-mode line has one
    const looks = line ? 1 : this.active.looks;
    // point scatterers anchored in tissue (decision 351): the wavenumber of the received echo and the line spacing
    const field = line ? undefined : this.active.scatterers;
    const kWave = field
      ? field.wavenumber(ctx.scene.physics.frequencyMHz, ctx.scene.physics.harmonics)
      : 0;
    const dTheta = spec.sectorRad / spec.lines;
    const ph = this.phasor;
    const nFrame = spec.lines * samples;
    const ox = beam.origin.x,
      oy = beam.origin.y,
      oz = beam.origin.z;
    const ct = Math.cos(theta),
      sn = Math.sin(theta);
    const dx = beam.forward.x * ct + beam.lateral.x * sn;
    const dy = beam.forward.y * ct + beam.lateral.y * sn;
    const dz = beam.forward.z * ct + beam.lateral.z * sn;
    const dhx = fwdH.x * ct + latH.x * sn,
      dhy = fwdH.y * ct + latH.y * sn,
      dhz = fwdH.z * ct + latH.z * sn;
    // Multi-angle spatial compounding (SonoCT-style): ±14°·sc virtual steered beams in the scan plane
    const sc = line ? 0 : ctx.spatialCompound;
    const scAlpha = sc * 0.24434609527920614;
    const scCos = Math.cos(scAlpha);
    const scSin = Math.sin(scAlpha);
    const pxT = -beam.forward.x * sn + beam.lateral.x * ct;
    const pyT = -beam.forward.y * sn + beam.lateral.y * ct;
    const pzT = -beam.forward.z * sn + beam.lateral.z * ct;
    const dxP = dx * scCos + pxT * scSin,
      dyP = dy * scCos + pyT * scSin,
      dzP = dz * scCos + pzT * scSin;
    const dxM = dx * scCos - pxT * scSin,
      dyM = dy * scCos - pyT * scSin,
      dzM = dz * scCos - pzT * scSin;
    const pxH = -fwdH.x * sn + latH.x * ct;
    const pyH = -fwdH.y * sn + latH.y * ct;
    const pzH = -fwdH.z * sn + latH.z * ct;
    const dhxP = dhx * scCos + pxH * scSin,
      dhyP = dhy * scCos + pyH * scSin,
      dhzP = dhz * scCos + pzH * scSin;
    const dhxM = dhx * scCos - pxH * scSin,
      dhyM = dhy * scCos - pyH * scSin,
      dhzM = dhz * scCos - pzH * scSin;
    const nX = beam.normal.x,
      nY = beam.normal.y,
      nZ = beam.normal.z;
    const R = SCATTER_FREQ_RATIO;
    const lineDrop = hash3(li, 7, 0, seed) > contact ? 0.08 : 1;
    let transmission = lineDrop;
    // what the line would transmit through average soft tissue: the chest-wall reverberation reaches each depth through
    // the wall, not through the tissue on the beam's axis (decision 263)
    let softTransmission = lineDrop;
    // the chest wall's reverberation at the heart's surface over the soft-tissue loss there: −1 until the line reaches it
    let clutterWall = -1;
    let lungEntryR = -1;
    let lungEntryT = 0;
    let dead = false;
    /** Classify a torso point into `q`: 0 outside the body, 1 thorax, 2 heart (anterior lung wins over the heart). */
    const classifyAt = sceneClassifier(ctx.scene);
    /** Incoherent backscatter σ and coherent specular echo of a classified sample, before attenuation. */
    const acoustic = (q: TissueSample, inH: boolean, a: { sigma: number; spec: number }): void => {
      const props = TISSUE_PROPS[q.tissue]!;
      const nd0 = Math.abs(
        inH ? q.nx * dhx + q.ny * dhy + q.nz * dhz : q.nx * dx + q.ny * dy + q.nz * dz,
      );
      // a cord's sample carries its axis: it reflects with the beam across it (decision 227)
      const nd = q.tissue === Tissue.Chordae ? cordAlignment(nd0) : nd0;
      let ndP = nd;
      let ndM = nd;
      if (sc > 0) {
        const ndP0 = Math.abs(
          inH ? q.nx * dhxP + q.ny * dhyP + q.nz * dhzP : q.nx * dxP + q.ny * dyP + q.nz * dzP,
        );
        const ndM0 = Math.abs(
          inH ? q.nx * dhxM + q.ny * dhyM + q.nz * dhzM : q.nx * dxM + q.ny * dyM + q.nz * dzM,
        );
        ndP = q.tissue === Tissue.Chordae ? cordAlignment(ndP0) : ndP0;
        ndM = q.tissue === Tissue.Chordae ? cordAlignment(ndM0) : ndM0;
      }
      let sigma = props.reflect;
      if (q.tissue === Tissue.Blood && harm) sigma *= BLOOD_HARMONIC_SIGMA;
      if (q.tissue === Tissue.Myocardium) {
        if (
          inH &&
          ((q.structure >= Structure.LvWallSeptal && q.structure <= Structure.LvApex) ||
            q.structure === Structure.RvWall)
        ) {
          const rr = Math.sqrt(q.mx * q.mx + q.my * q.my);
          const trm = q.transmural >= 0 ? q.transmural : 0.5;
          const dphi = rr > MYO_ANISO_RADIAL_EPS ? (dhy * q.mx - dhx * q.my) / rr : 0;
          let gMyo = myoHelixGain(dphi, dhz, trm);
          if (sc > 0) {
            const dphiP = rr > MYO_ANISO_RADIAL_EPS ? (dhyP * q.mx - dhxP * q.my) / rr : 0;
            const dphiM = rr > MYO_ANISO_RADIAL_EPS ? (dhyM * q.mx - dhxM * q.my) / rr : 0;
            gMyo =
              0.5 * gMyo +
              0.25 * myoHelixGain(dphiP, dhzP, trm) +
              0.25 * myoHelixGain(dphiM, dhzM, trm);
          }
          sigma *= gMyo;
          if (q.structure !== Structure.RvWall) sigma *= myoIb;
        } else if (
          inH &&
          (q.structure === Structure.PapillaryMuscle || q.structure === Structure.RvPapillary)
        ) {
          let gPap = papillaryFibreGain(dhz);
          if (sc > 0) {
            gPap =
              0.5 * gPap +
              0.25 * papillaryFibreGain(dhzP) +
              0.25 * papillaryFibreGain(dhzM);
          }
          sigma *= gPap;
          if (q.structure === Structure.PapillaryMuscle) sigma *= myoIb;
        } else if (
          !inH ||
          (q.structure !== Structure.LaWall &&
            q.structure !== Structure.RaWall &&
            q.structure !== Structure.InteratrialSeptum)
        ) {
          const nd2 = sc > 0 ? 0.5 * nd * nd + 0.25 * ndP * ndP + 0.25 * ndM * ndM : nd * nd;
          sigma *= MYO_ANISO_FLOOR + (1 - MYO_ANISO_FLOOR) * nd2;
        }
      }
      const het = heteroDb(q.tissue);
      if (het > 0)
        sigma *= Math.pow(
          10,
          ((latticeNoise3(
            q.mx * HETERO_FREQ + HET_X,
            q.my * HETERO_FREQ + HET_Y,
            q.mz * HETERO_FREQ + HET_Z,
            latC,
          ) -
            0.5) *
            het) /
            20,
        );
      if (q.extraReflect > 0)
        sigma +=
          q.extraReflect *
          CALCIUM_GAIN *
          (CALCIUM_BASE +
            CALCIUM_AMP *
              latticeNoise3(
                q.mx * CALCIUM_FREQ + CAL_X,
                q.my * CALCIUM_FREQ + CAL_Y,
                q.mz * CALCIUM_FREQ + CAL_Z,
                latB,
              ));
      let specular = 0;
      if (props.specular > 0) {
        const sScale = props.specular * SPECULAR_GAIN * (harm ? SPECULAR_HARMONIC : 1);
        const absSdf = Math.abs(q.sdf);
        const sp0 = sScale * interfaceEcho(absSdf, nd, dr);
        if (sc > 0) {
          const spP = sScale * interfaceEcho(absSdf, ndP, dr);
          const spM = sScale * interfaceEcho(absSdf, ndM, dr);
          specular = 0.5 * sp0 + 0.25 * spP + 0.25 * spM;
        } else {
          specular = sp0;
        }
      }
      a.sigma = sigma;
      a.spec = specular;
    };
    let tail0 = 0,
      tail1 = 0,
      tail2 = 0;
    const td1 = Math.exp(-dr / SPEC_TAIL_DECAY_CM),
      td2 = td1 * td1,
      td3 = td2 * td1;
    for (let si = 0; si < samples; si++) {
      const r = (si + 0.5) * dr;
      const idx = base + si;
      if (dead) {
        // an M-mode line: its pleura faces the beam (a frame's lines wait for their neighbours, decision 221)
        lungEcho(ctx, li, idx, r, lungEntryR, lungEntryT, 1, looks, re, im, lungEntryT, 1);
        st[idx] = Structure.Lung;
        tr[idx] = 0;
        ti[idx] = Tissue.Lung;
        if (sg) sg[idx] = 0;
        continue;
      }
      const px = ox + dx * r,
        py = oy + dy * r,
        pz = oz + dz * r;
      const kind = classifyAt(px, py, pz, s);
      if (kind === 0) {
        for (let k = 0; k < looks; k++) {
          re[k * nFrame + idx] = 0;
          im[k * nFrame + idx] = 0;
        }
        st[idx] = Structure.None;
        tr[idx] = transmission;
        ti[idx] = Tissue.None;
        if (sg) sg[idx] = 0;
        tail2 = tail1;
        tail1 = tail0;
        tail0 = 0;
        continue;
      }
      const inHeart = kind === 2;
      const tissue = s.tissue;
      const props = TISSUE_PROPS[tissue]!;
      st[idx] = s.structure;
      tr[idx] = transmission;
      ti[idx] = tissue;
      if (heartFlag) heartFlag[idx] = inHeart ? 1 : 0;
      // the LV segment of the tissue the beam crosses here (decision 152), from the same sample as the structure
      if (sg) sg[idx] = s.segment;
      if (tissue === Tissue.Lung) {
        // pleural line: a strong coherent reflector; everything behind it is reverberation
        lungEntryR = r;
        // a frame's lung echoes are scaled by the beam's transmission at the entry in the beam march; a line's by its own
        lungEntryT = line ? transmission : 1;
        if (!line) {
          // a frame line: its pleura and reverberations are drawn once every line's entry is known, since how square the
          // pleura stands to the beam comes from the neighbouring lines' entries (decision 221)
          this.lungEntry[li] = si;
          for (let sj = si; sj < samples; sj++) {
            const j = base + sj;
            for (let k = 0; k < looks; k++) {
              re[k * nFrame + j] = 0;
              im[k * nFrame + j] = 0;
            }
            if (sj === si) continue;
            st[j] = Structure.Lung;
            tr[j] = 0;
            ti[j] = Tissue.Lung;
            if (sg) sg[j] = 0;
          }
          break;
        }
        pleuraEcho(ctx, li, idx, r, lungEntryT, 1, looks, re, im);
        dead = true;
        continue;
      }
      acoustic(s, inHeart, acA);
      let sigma = acA.sigma;
      let specular = acA.spec;
      if (tissue === Tissue.Valve) {
        // a leaflet is a membrane thinner than the slice: it reads by the fraction of the slice it fills, full where it
        // stands across the plane and MEMBRANE_CM over the slice thickness where it lies in it; the side planes of the
        // high tier would miss it, so it takes no elevation average (decision 147)
        const nn = inHeart
          ? s.nx * nrmH.x + s.ny * nrmH.y + s.nz * nrmH.z
          : s.nx * nX + s.ny * nY + s.nz * nZ;
        const w = membraneWeight(nn, sliceHalfWidthCm(r));
        sigma *= w;
        specular *= w;
      } else if (tissue === Tissue.Chordae) {
        // a cord thinner than the slice reads by the fraction of the slice it fills, and takes no elevation average
        // either (decision 227)
        const w = cordWeight(s.nx * nrmH.x + s.ny * nrmH.y + s.nz * nrmH.z, sliceHalfWidthCm(r));
        sigma *= w;
        specular *= w;
      }
      const specCentral = specular;
      if (nElev > 1 && tissue !== Tissue.Valve && tissue !== Tissue.Chordae) {
        // slice thickness: the beam's elevational width grows away from the focus; backscatter and interface
        // echo are the weighted mean over the slice (¼ ½ ¼), which blurs obliquely cut structures
        const e = sliceHalfWidthCm(r);
        let accS = sigma * 0.5,
          accP = specular * 0.5,
          wsum = 0.5;
        for (let k = -1; k <= 1; k += 2) {
          const kk = classifyAt(px + nX * k * e, py + nY * k * e, pz + nZ * k * e, s2);
          if (kk > 0 && s2.tissue !== Tissue.Lung) {
            acoustic(s2, kk === 2, acB);
            accS += 0.25 * acB.sigma;
            accP += 0.25 * acB.spec;
            wsum += 0.25;
          }
        }
        sigma = accS / wsum;
        specular = accP / wsum;
      }
      // Spontaneous echo contrast ("smoke" / SEC) in low-flow stasis chambers (exact mirror of GLSL_PASS_B_MAIN)
      if (tissue === Tissue.Blood && (ctx.secLa > 0 || ctx.secLv > 0)) {
        const hf = heart.frame;
        const dpx = px - hf.origin.x;
        const dpy = py - hf.origin.y;
        const dpz = pz - hf.origin.z;
        const hx = dpx * hf.ex.x + dpy * hf.ex.y + dpz * hf.ex.z;
        const hy = dpx * hf.ey.x + dpy * hf.ey.y + dpz * hf.ey.z;
        const hz = dpx * hf.ez.x + dpy * hf.ez.y + dpz * hf.ez.z;
        let sec = 0;
        if (s.structure === Structure.LaCavity) {
          sec = ctx.secLa;
        } else if (s.structure === Structure.LaAppendage) {
          sec = Math.min(1, ctx.secLa * SEC_LAA_BOOST);
        } else if (s.structure === Structure.LvCavity && ctx.secLv > 0) {
          const lvApex = (hz - ctx.scene.heartPose.zAnn) / Math.max(1, ctx.scene.heartPose.lengthNow);
          const lvFrac = 0.2 + 0.8 * Math.max(0, Math.min(1, lvApex / 0.75));
          sec = ctx.secLv * lvFrac;
        }
        if (sec > 0) {
          // fresh blood the mitral inflow brings has not aggregated: cleared about the orifice (atrium) or the inflow axis
          const mv = ctx.scene.heartPose.valves.mitral;
          const dmx = hx - ctx.scene.heartPose.swingX - mv.cx;
          const dmy = hy - mv.cy;
          const dmz = s.structure === Structure.LvCavity ? 0 : hz - ctx.scene.heartPose.zAnn;
          const keep = secOrificeKeep(ctx.secShear, dmx * dmx + dmy * dmy + dmz * dmz);
          const sx = hx * SEC_FREQ + ctx.bloodShift * SEC_DRIFT_X;
          const sy = hy * SEC_FREQ - ctx.bloodShift * SEC_DRIFT_Y;
          const sz = hz * SEC_FREQ;
          const swirl =
            0.5 +
            0.5 *
              Math.sin(sx + 0.85 * Math.cos(sy * 1.15 + sz * 0.75)) *
              Math.cos(sy * 0.9 - 0.75 * Math.sin(sz * 1.1 - sx * 0.65));
          sigma += sec * keep * SEC_GAIN * (SEC_BASE_FLOOR + (1 - SEC_BASE_FLOOR) * swirl * swirl);
        }
      }
      // the beam's on-axis sensitivity at this depth: wide near the face, narrowest at the focus (decision 144)
      const fg = focusingGain(r, focus);
      sigma *= fg;
      specular *= fg;
      if (!line) {
        const tail = tail0 * td1 + tail1 * td2 + tail2 * td3;
        if (tail > 0) {
          const tScaled = tail * fg;
          sigma += 0.65 * tScaled;
          specular += 0.35 * tScaled;
        }
        const extraQ = Math.round(Math.max(0, Math.min(1, s.extraReflect)) * 255) / 255;
        tail2 = tail1;
        tail1 = tail0;
        tail0 = interfaceTailSource(specCentral, extraQ);
      }
      // complex scatterer phasor of the central plane, anchored in tissue coordinates (moves with the tissue). Across the
      // plane its lattice cell is the slice thickness, as for an M-mode line (decision 99): at the scatterer cell (0.4 mm)
      // a probe tilt that moved the plane by 0.31 mm at 9 cm left the myocardial speckle with a correlation of 0.71, for a
      // slice about 4 mm thick there
      const nhx = inHeart ? nrmH.x : nX,
        nhy = inHeart ? nrmH.y : nY,
        nhz = inHeart ? nrmH.z : nZ;
      const across =
        (SCATTER_FREQ - 1 / (2 * sliceHalfWidthCm(r))) * (s.mx * nhx + s.my * nhy + s.mz * nhz);
      // flowing blood is a new realization in every frame (decision 163)
      const qx =
          s.mx * SCATTER_FREQ - across * nhx + (tissue === Tissue.Blood ? ctx.bloodShift : 0),
        qy = s.my * SCATTER_FREQ - across * nhy,
        qz = s.mz * SCATTER_FREQ - across * nhz;
      // bright grains of the parenchyma: a sparse coherent component over the diffuse scatterers, the same in every look
      // (decision 145). The grain lattice is anchored in tissue coordinates at the tissue's grain frequency; across the
      // plane its cell is the slice thickness where that is coarser, as for the scatterer phasor, and the slice
      // integrates the grains it holds across (`grainCoef`). An M-mode line places its grains per structure run after
      // the march (`renderMmodeLine`), like its scatterers
      let grain = 0;
      const gf = props.grain;
      const parenchyma =
        tissue === Tissue.Myocardium || tissue === Tissue.Muscle || tissue === Tissue.Liver;
      // point scatterers draw parenchyma's bright grains with their strong scatterers (decision 356)
      if (parenchyma && !field) {
        if (line && lk) {
          line.grainScale[si] = sigma * GRAIN_GAIN * transmission * lk.smooth;
          line.grainCells[si] = gf;
        } else {
          const ga = gf - Math.min(gf, 1 / (2 * sliceHalfWidthCm(r)));
          const gm = (s.mx * nhx + s.my * nhy + s.mz * nhz) * ga;
          grain =
            sigma *
            GRAIN_GAIN *
            grainCoef(
              s.mx * gf - gm * nhx + GR_X,
              s.my * gf - gm * nhy + GR_Y,
              s.mz * gf - gm * nhz + GR_Z,
              latA,
              latB,
              latC,
            );
        }
      }
      // a frame forms one complex signal per compounding look: the scatterer and clutter phasors differ between looks,
      // the coherent echoes (specular, ring-down) do not (decision 145); an M-mode line keeps one look
      let sRe = 0,
        sIm = 0;
      if (line && lk) {
        // M-mode line (decision 84): the lattice axes follow the beam, with the beam width as the cell across it; the
        // coordinate along the beam is set per structure after the march (`renderMmodeLine`), so only σ is kept here
        const a = inHeart ? line.heartAxes : line.torsoAxes;
        line.along[si] = inHeart ? NaN : s.mx * a[0]! + s.my * a[1]! + s.mz * a[2]!;
        line.across[si] = (s.mx * a[3]! + s.my * a[4]! + s.mz * a[5]!) * lk.lateralCells[si]!;
        line.elevation[si] =
          (s.mx * a[6]! + s.my * a[7]! + s.mz * a[8]!) * lk.elevationCells[si]! +
          (tissue === Tissue.Blood ? line.bloodCells : 0);
        line.sigma[si] = sigma * inc * transmission;
        sRe = specular * lk.specular;
        sIm = 0;
      }
      let cm = 0;
      if (clutter > 0) {
        cm = chestWallClutter(ctx, li, r);
        // The reverberation of the chest wall lives in the wall: it pays the wall's own loss, at most the soft-tissue loss of
        // its depth (decision 263: it is not reinforced behind fluid or blood). Past the heart's surface it is an echo of
        // the wall that arrives late, not sound that crossed the heart, so it carries on with the soft-tissue loss from
        // what it had there (decision 312). It used to take the loss of whatever the line crossed, so the right ventricle
        // of the four-chamber view, reached through the apex and the septum, lost its haze: the band beside the septum
        // read 40 grey against 48-74 in CAMUS Good. A frame's line leaves it unscaled for the beam march, which applies
        // that loss with the beam's transmission (`clutterLoss`); an M-mode line applies it here with its own.
        if (line) {
          if (inHeart && clutterWall < 0)
            clutterWall = Math.min(1, transmission / softTransmission);
          cm *= clutterLoss(transmission, softTransmission, clutterWall) / transmission;
        }
      }
      const cx = ox * CLUTTER_FREQ + li * CLUTTER_LINE_FREQ,
        cy = oy * CLUTTER_FREQ + oz * CLUTTER_FREQ,
        cz = r * SCATTER_FREQ;
      const ringDown = r < RINGDOWN_CM ? RINGDOWN_GAIN * (1 - r / RINGDOWN_CM) : 0; // transducer ring-down
      if (line && lk) {
        if (cm > 0) {
          sRe +=
            cm *
            (latticeNoise3(cx + CLUTTER_RE_A_X, cy, cz, latA) +
              latticeNoise3(cx * R + CRB_X, cy + CRB_Y, cz * R, latB) -
              1) *
            PHASOR_NORM *
            incAxial;
          sIm +=
            cm *
            (latticeNoise3(cx + CIA_X, cy + CIA_Y, cz + CIA_Z, latC) +
              latticeNoise3(cx * R + CIB_X, cy + CIB_Y, cz * R + CIB_Z, latA) -
              1) *
            PHASOR_NORM *
            incAxial;
        }
        if (ringDown > 0) sRe += ringDown * lk.smooth;
        // an M-mode line scales its echoes by its own transmission (no lateral neighbours)
        re[idx] = sRe * transmission;
        im[idx] = sIm * transmission;
      } else {
        // a frame's echoes are scaled by the beam's transmission in the beam march
        for (let k = 0; k < looks; k++) {
          const sx = LOOK_SHIFT[k]![0],
            sy = LOOK_SHIFT[k]![1],
            sz = LOOK_SHIFT[k]![2];
          let zr: number, zi: number;
          if (field) {
            // point scatterers of the sample's neighbourhood on the probe's axes in the material frame (decision 351);
            // flowing blood is a new realization in every frame, shifted along x as the lattice is (decision 163)
            const fa = inHeart ? fwdH : beam.forward,
              la = inHeart ? latH : beam.lateral,
              na = inHeart ? nrmH : beam.normal;
            const mx = s.mx + (tissue === Tissue.Blood ? ctx.bloodShift / SCATTER_FREQ : 0);
            field.phasor(
              mx * fa.x + s.my * fa.y + s.mz * fa.z,
              mx * la.x + s.my * la.y + s.mz * la.z,
              mx * na.x + s.my * na.y + s.mz * na.z,
              ct,
              sn,
              r,
              dr,
              r * dTheta,
              sliceHalfWidthCm(r) * ELEVATION_SIGMA_PER_HALF_WIDTH,
              kWave,
              seed,
              parenchyma ? STRONG_SCATTERER_FRACTION : 0,
              ph,
            );
            zr = ph.re;
            zi = ph.im;
          } else {
            zr =
              (latticeNoise3(qx + sx, qy + sy, qz + sz, latA) +
                latticeNoise3(qx * R + PRB_X + sx, qy * R + PRB_Y + sy, qz * R + PRB_Z + sz, latB) -
                1) *
              PHASOR_NORM;
            zi =
              (latticeNoise3(qx + PIA_X + sx, qy + PIA_Y + sy, qz + PIA_Z + sz, latC) +
                latticeNoise3(qx * R + PIB_X + sx, qy * R + PIB_Y + sy, qz * R + PIB_Z + sz, latA) -
                1) *
              PHASOR_NORM;
          }
          re[k * nFrame + idx] = sigma * zr + specular + grain + ringDown;
          im[k * nFrame + idx] = sigma * zi;
          if (cm > 0 && clutRe && clutIm) {
            clutRe[k * nFrame + idx] =
              cm *
              (latticeNoise3(cx + CLUTTER_RE_A_X + sx, cy + sy, cz + sz, latA) +
                latticeNoise3(cx * R + CRB_X + sx, cy + CRB_Y + sy, cz * R + sz, latB) -
                1) *
              PHASOR_NORM;
            clutIm[k * nFrame + idx] =
              cm *
              (latticeNoise3(cx + CIA_X + sx, cy + CIA_Y + sy, cz + CIA_Z + sz, latC) +
                latticeNoise3(cx * R + CIB_X + sx, cy + CIB_Y + sy, cz * R + CIB_Z + sz, latA) -
                1) *
              PHASOR_NORM;
          }
        }
      }
      // two-way amplitude loss integrated over the sample's length (decision 89): 0.23 Np per dB·cm⁻¹·MHz⁻¹ of one-way
      // attenuation, so a layer loses the same whatever the sampling. Bone, calcium and spine used a fixed 1.2 Np per
      // sample, which made a rib's shadow depend on the quality tier (17 dB between low and high behind 4 mm of rib).
      // a calcified leaflet's calcium attenuates as calcium does (decision 251)
      const wallExtraDb =
        !inHeart && (tissue === Tissue.Fat || tissue === Tissue.Muscle)
          ? CHEST_WALL_ATTEN_DB * ctx.windowAttenuation
          : 0;
      const attenNp =
        ATTEN_NP_PER_DB *
        (props.attenuation + wallExtraDb + calciumAttenDb(s.extraReflect)) *
        fAtten *
        dr;
      if (atten) atten[idx] = attenNp;
      transmission *= Math.exp(-attenNp);
      if (transmission < TRANSMISSION_FLOOR) transmission = TRANSMISSION_FLOOR;
      softTransmission *= Math.exp(-ATTEN_NP_PER_DB * SOFT_TISSUE_ATTEN_DB * fAtten * dr);
      if (softTransmission < TRANSMISSION_FLOOR) softTransmission = TRANSMISSION_FLOOR;
    }
  }
}
