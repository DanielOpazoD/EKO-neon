import { aliasVelocity } from '@/clinical/formulas';
import { glslFloat } from '@/simulator/renderer/acoustic/acoustics';

/** Default clinical spatiotemporal smoothing strength (0 = off/raw, 1 = maximum clinical smoothing). */
export const DEFAULT_COLOR_SMOOTHING = 0.45;

/** Hysteresis floor fraction of COLOR_THRESHOLD for smooth temporal decay on momentary stochastic drop-outs. */
export const ST_HYSTERESIS_RATIO = 0.58;
/** Baseline decay retention multiplier when a previously colored blood sample momentarily dips below threshold. */
export const ST_DROPOUT_DECAY_BASE = 0.52;
/** Extra decay retention proportional to user persistence. */
export const ST_DROPOUT_DECAY_PERSIST = 0.34;
/** Attack persistence damping when new flow power rises sharply over history (fast systolic/E-wave onset). */
export const ST_ATTACK_PERSIST_SCALE = 0.55;
/** Extra temporal stability boost in steady laminar flow (low variance + small frame-to-frame phase delta). */
export const ST_LAMINAR_PERSIST_BOOST = 0.22;
/** Temporal persistence scale floor in high-turbulence or rapidly shifting flow (preserves lively mosaic & crisp onset). */
export const ST_TURB_PERSIST_SCALE = 0.45;
/** Frame-to-frame circular phase jump (rad) over which temporal persistence drops from steady to dynamic. */
export const ST_PHASE_JUMP_REF_RAD = 1.45;

/** Minimum 8-connected colored neighbor count required to survive isolated-speckle despeckling. */
export const ST_MIN_NEIGHBOR_COUNT = 2;
/** Minimum weighted neighbor support (out of 12.0 max for 3x3 binomial neighbors) at full smoothing. */
export const ST_MIN_NEIGHBOR_WEIGHT = 2.5;
/** Minimum 8-connected colored neighbors needed to close an isolated blood-pool pinhole at normal smoothing. */
export const ST_PINHOLE_MIN_COUNT = 5;
/** Minimum mean neighbor power required to close an isolated blood-pool pinhole. */
export const ST_PINHOLE_MIN_POWER = 0.34;

/** Maximum circular phasor spatial smoothing mix in pure laminar flow (at smoothing = 1.0). */
export const ST_SPATIAL_LAMINAR_MIX = 0.78;
/** Residual circular phasor spatial smoothing mix in turbulent mosaic flow (at smoothing = 1.0). */
export const ST_SPATIAL_TURB_MIX = 0.14;
/** Circular phase-similarity bilateral sharpness: weight = exp(-(1 - cos(dPhi)) * K). */
export const ST_PHASE_BILATERAL_K = 1.8;
/** Laminar variance cleanup strength: damps isolated stochastic variance spikes in coherent laminar flow. */
export const ST_LAMINAR_VAR_CLEANUP = 0.65;
/** Subtle turbulent variance consolidation gain when spectral variance and spatial incoherence agree. */
export const ST_TURB_VAR_BOOST = 0.18;
/** Minimum power-modulated overlay opacity at the hysteresis fade floor. */
export const ST_POW_ALPHA_MIN = 0.2;

/**
 * Shared GLSL constants for the CPU-locked spatiotemporal Color Doppler filter (Pass D2 + Pass D3).
 */
export function spatiotemporalDefinesGlsl(): string {
  return [
    `#define ST_HYSTERESIS_RATIO ${glslFloat(ST_HYSTERESIS_RATIO)}`,
    `#define ST_DROPOUT_DECAY_BASE ${glslFloat(ST_DROPOUT_DECAY_BASE)}`,
    `#define ST_DROPOUT_DECAY_PERSIST ${glslFloat(ST_DROPOUT_DECAY_PERSIST)}`,
    `#define ST_ATTACK_PERSIST_SCALE ${glslFloat(ST_ATTACK_PERSIST_SCALE)}`,
    `#define ST_LAMINAR_PERSIST_BOOST ${glslFloat(ST_LAMINAR_PERSIST_BOOST)}`,
    `#define ST_TURB_PERSIST_SCALE ${glslFloat(ST_TURB_PERSIST_SCALE)}`,
    `#define ST_PHASE_JUMP_REF_RAD ${glslFloat(ST_PHASE_JUMP_REF_RAD)}`,
    `#define ST_MIN_NEIGHBOR_COUNT ${ST_MIN_NEIGHBOR_COUNT}`,
    `#define ST_MIN_NEIGHBOR_WEIGHT ${glslFloat(ST_MIN_NEIGHBOR_WEIGHT)}`,
    `#define ST_PINHOLE_MIN_COUNT ${ST_PINHOLE_MIN_COUNT}`,
    `#define ST_PINHOLE_MIN_POWER ${glslFloat(ST_PINHOLE_MIN_POWER)}`,
    `#define ST_SPATIAL_LAMINAR_MIX ${glslFloat(ST_SPATIAL_LAMINAR_MIX)}`,
    `#define ST_SPATIAL_TURB_MIX ${glslFloat(ST_SPATIAL_TURB_MIX)}`,
    `#define ST_PHASE_BILATERAL_K ${glslFloat(ST_PHASE_BILATERAL_K)}`,
    `#define ST_LAMINAR_VAR_CLEANUP ${glslFloat(ST_LAMINAR_VAR_CLEANUP)}`,
    `#define ST_TURB_VAR_BOOST ${glslFloat(ST_TURB_VAR_BOOST)}`,
    `#define ST_POW_ALPHA_MIN ${glslFloat(ST_POW_ALPHA_MIN)}`,
  ].join('\n');
}

function clamp01(x: number): number {
  return x <= 0 ? 0 : x >= 1 ? 1 : x;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / Math.max(1e-6, edge1 - edge0));
  return t * t * (3 - 2 * t);
}

/**
 * Smooth power confidence in [0, 1] spanning from the temporal hysteresis decay floor up to strong
 * laminar/turbulent flow, so fading boundary pixels dissolve smoothly instead of popping on/off.
 */
export function colorPowerConfidence(power: number, colorThreshold: number): number {
  const lo = colorThreshold * ST_HYSTERESIS_RATIO;
  const hi = colorThreshold + 0.3;
  return smoothstep(lo, hi, power);
}

export interface TemporalStepInput {
  /** Whether the polar sample is unshadowed blood (or bloomed blood) on the current frame. */
  isBloodValid: boolean;
  /** Whether the current frame passed power/coherence/wall gating. */
  hasCurr: boolean;
  /** Current aliased velocity in m/s (ignored when !hasCurr). */
  vCurr: number;
  /** Current normalized spectral variance [0..1] (ignored when !hasCurr). */
  varCurr: number;
  /** Current gated signal power [0..1] (ignored when !hasCurr). */
  powCurr: number;
  /** Whether valid previous frame history exists under matching scale/baseline/inversion. */
  hasPrev: boolean;
  /** Previous aliased velocity in m/s. */
  vPrev: number;
  /** Previous normalized variance [0..1]. */
  varPrev: number;
  /** Previous power [0..1]. */
  powPrev: number;
  scaleMps: number;
  baselineShiftMps: number;
  persistence: number;
  smoothing: number;
  colorThreshold: number;
}

export interface TemporalStepOutput {
  valid: boolean;
  vel: number;
  variance: number;
  power: number;
}

/**
 * Aliasing-safe asymmetric Attack/Decay + Flow-Regime-Adaptive Circular Temporal Filter.
 *
 * - Immediately extinguishes samples that are no longer unshadowed blood (zero wall/valve ghosting).
 * - Uses fast attack when strong new flow arrives (crisp systolic ejection / E-wave onset).
 * - Stabilizes steady laminar flow (low variance, small circular phase delta) while keeping
 *   turbulent or rapidly reversing jets lively and responsive.
 * - Applies blood-pool temporal hysteresis decay on momentary 1-frame stochastic drop-outs so
 *   threshold-straddling boundary pixels fade smoothly instead of blinking on/off.
 * - Reduces to exact baseline circular persistence when `smoothing === 0`, and disables temporal
 *   blending strictly when `persistence === 0`.
 */
export function adaptiveTemporalStep(inp: TemporalStepInput, out: TemporalStepOutput): void {
  const pBase = Math.max(0, Math.min(0.95, inp.persistence));
  const sm = clamp01(inp.smoothing);
  const scale = Math.max(1e-3, inp.scaleMps);
  const baseline = inp.baselineShiftMps;

  // Non-blood or shadowed sample: cut immediately so moving valves/walls never carry color trails
  if (!inp.isBloodValid) {
    out.valid = false;
    out.vel = NaN;
    out.variance = 0;
    out.power = 0;
    return;
  }

  // Case A: Current frame has valid gated flow
  if (inp.hasCurr) {
    if (!inp.hasPrev || pBase <= 0) {
      out.valid = true;
      out.vel = inp.vCurr;
      out.variance = inp.varCurr;
      out.power = inp.powCurr;
      return;
    }

    const a1 = (Math.PI * (inp.vCurr - baseline)) / scale;
    const a0 = (Math.PI * (inp.vPrev - baseline)) / scale;
    const c1 = Math.cos(a1);
    const s1 = Math.sin(a1);
    const c0 = Math.cos(a0);
    const s0 = Math.sin(a0);

    // Aliasing-safe circular phase difference in [0, pi] with dead-zone for normal laminar motion
    const cosDelta = Math.max(-1, Math.min(1, c1 * c0 + s1 * s0));
    const phaseDelta = Math.acos(cosDelta);
    const phaseJump = smoothstep(0.45, ST_PHASE_JUMP_REF_RAD, phaseDelta);
    const steadyPhase = 1 - phaseJump;
    const maxVar = Math.max(inp.varCurr, inp.varPrev);
    const laminarCue = clamp01(1 - maxVar * 2.2) * steadyPhase;
    const turbCue = clamp01(maxVar * 1.6);

    // Fast attack when flow power rises briskly above prior frame
    const rise = clamp01((inp.powCurr - inp.powPrev - 0.08) / 0.35);
    const attackScale = 1 - (1 - ST_ATTACK_PERSIST_SCALE) * rise * sm;

    // Flow-regime modulation: more stable in steady laminar flow, more responsive in turbulence/reversals
    const dynamicCue = Math.max(turbCue, phaseJump);
    const pRegime =
      pBase +
      sm *
        (ST_LAMINAR_PERSIST_BOOST * laminarCue * (1 - pBase) -
          (1 - ST_TURB_PERSIST_SCALE) * dynamicCue * pBase);
    const pEff = Math.max(0, Math.min(0.92, pRegime * attackScale));

    const re = (1 - pEff) * inp.powCurr * c1 + pEff * inp.powPrev * c0;
    const im = (1 - pEff) * inp.powCurr * s1 + pEff * inp.powPrev * s0;
    const angle = Math.abs(re) + Math.abs(im) > 1e-12 ? Math.atan2(im, re) : a1;

    // Slightly stronger variance damping in steady laminar flow suppresses 1-frame green blips
    const pVar = Math.max(0, Math.min(0.92, pEff + 0.12 * sm * laminarCue));

    out.valid = true;
    out.vel = aliasVelocity(baseline + (scale * angle) / Math.PI, scale, baseline);
    out.variance = inp.varCurr * (1 - pVar) + inp.varPrev * pVar;
    out.power = inp.powCurr * (1 - pEff) + inp.powPrev * pEff;
    return;
  }

  // Case B: Momentary stochastic drop-out over valid blood -> smooth hysteresis decay instead of binary flicker
  if (inp.hasPrev && pBase > 0 && sm > 0) {
    const decay =
      (ST_DROPOUT_DECAY_BASE + ST_DROPOUT_DECAY_PERSIST * pBase) * (0.75 + 0.25 * sm);
    const decayedPow = inp.powPrev * decay;
    const hystFloor = inp.colorThreshold * ST_HYSTERESIS_RATIO;
    if (decayedPow >= hystFloor) {
      out.valid = true;
      out.vel = inp.vPrev;
      out.variance = inp.varPrev;
      out.power = decayedPow;
      return;
    }
  }

  out.valid = false;
  out.vel = NaN;
  out.variance = 0;
  out.power = 0;
}

/** Module-level scratch buffers for CPU spatial filtering (zero per-frame heap allocations). */
const spatialScratch = new Map<string, Float32Array>();

function getSpatialScratch(name: string, n: number): Float32Array {
  let b = spatialScratch.get(name);
  if (!b || b.length < n) {
    b = new Float32Array(Math.max(n, Math.ceil((b?.length ?? 0) * 1.5)));
    spatialScratch.set(name, b);
  }
  return b.subarray(0, n);
}

export interface SpatialFilterParams {
  /** Polar grid sample count per line (stride). */
  samples: number;
  liMin: number;
  liMax: number;
  siMin: number;
  siMax: number;
  scaleMps: number;
  baselineShiftMps: number;
  /** Smoothing strength in [0, 1]. At 0, copies src -> dst unchanged (still applying breakthrough encoding). */
  smoothing: number;
  /** Box-indexed mask (`nL * nS`): 1 where unshadowed blood (or bloomed blood), 0 otherwise. */
  bloodMask: Float32Array;
  /** Optional box-indexed Doppler class (`1.0` blood, `0.5` tissue/valve clutter, `0` invalid). */
  cellClass?: Float32Array;
  /** Color Doppler gain (dB) for extreme overgain tissue-write-priority breakthrough (> 4 dB). */
  gainDb?: number;
  srcVel: Float32Array;
  srcVar: Float32Array;
  srcPow: Float32Array;
  dstVel: Float32Array;
  dstVar: Float32Array;
  dstPow: Float32Array;
}

/**
 * Encode tissue-write-priority breakthrough in `power > 1.0` (`1.0 + breakthrough`) for surviving
 * tissue/valve clutter or extreme overgain (`gainDb > 4 dB`), leaving normal blood (`gainDb <= 4 dB`)
 * strictly in `[0, 1]` so standard tissue write priority is bit-for-bit unchanged.
 */
export function encodeColorBreakthroughPower(
  powVal: number,
  isClutter: boolean,
  gainDb: number,
): number {
  const bt = isClutter
    ? smoothstep(0.25, 0.9, powVal)
    : smoothstep(4.0, 12.0, gainDb) * 0.45;
  return bt > 0 ? 1 + bt : powVal;
}

/**
 * Coherence- & Variance-Adaptive Circular Spatial Filter for Color Doppler (CPU mirror of Pass D3).
 *
 * Inside the polar ROI box:
 *  1. Rejects isolated salt-and-pepper speckle pixels with insufficient same-class 8-neighbor support.
 *  2. Closes isolated single-sample pinholes strictly inside unshadowed blood (`class >= 0.75`, never myocardium).
 *  3. Estimates local spatial circular phase coherence `|Σ w·p·e^{iφ}| / Σ w·p` and local variance strictly
 *     within the same class (blood vs clutter) so wall flash never biases adjacent laminar blood flow.
 *  4. Encodes tissue-write-priority breakthrough (`power = 1 + breakthrough`) on surviving clutter / extreme overgain.
 */
export function applyColorSpatialFilter(p: SpatialFilterParams): void {
  const { samples, liMin, liMax, siMin, siMax } = p;
  const nL = Math.max(0, liMax - liMin + 1);
  const nS = Math.max(0, siMax - siMin + 1);
  if (nL === 0 || nS === 0) return;

  const sm = clamp01(p.smoothing);
  const gainDb = p.gainDb ?? 0;
  const clsArr = p.cellClass;
  if (sm <= 0) {
    for (let l = 0; l < nL; l++) {
      const li = liMin + l;
      const row = li * samples + siMin;
      const gRow = l * nS;
      for (let k = 0; k < nS; k++) {
        const idx = row + k;
        const pw = p.srcPow[idx]!;
        p.dstVel[idx] = p.srcVel[idx]!;
        p.dstVar[idx] = p.srcVar[idx]!;
        if (pw > 0) {
          const cls = clsArr ? clsArr[gRow + k]! : p.bloodMask[gRow + k]! > 0.5 ? 1 : 0;
          const isClutter = cls > 0.25 && cls < 0.75;
          p.dstPow[idx] = encodeColorBreakthroughPower(pw, isClutter, gainDb);
        } else {
          p.dstPow[idx] = 0;
        }
      }
    }
    return;
  }

  const scale = Math.max(1e-3, p.scaleMps);
  const baseline = p.baselineShiftMps;
  const kPhi = Math.PI / scale;
  const invPiScale = scale / Math.PI;
  const n = nL * nS;

  // Precompute baseline-centered unit phasors once per box sample so the 3x3 gather has zero trig calls
  const cosBuf = getSpatialScratch('cos', n);
  const sinBuf = getSpatialScratch('sin', n);
  const varBuf = getSpatialScratch('var', n);
  const powBuf = getSpatialScratch('pow', n);

  for (let l = 0; l < nL; l++) {
    const row = (liMin + l) * samples + siMin;
    const gRow = l * nS;
    for (let k = 0; k < nS; k++) {
      const idx = row + k;
      const g = gRow + k;
      const v = p.srcVel[idx]!;
      const pw = p.srcPow[idx]!;
      if (v === v && pw > 0) {
        const ph = (v - baseline) * kPhi;
        cosBuf[g] = Math.cos(ph);
        sinBuf[g] = Math.sin(ph);
        varBuf[g] = p.srcVar[idx]!;
        powBuf[g] = pw;
      } else {
        cosBuf[g] = 0;
        sinBuf[g] = 0;
        varBuf[g] = 0;
        powBuf[g] = 0;
      }
    }
  }

  const minPinholeCount = sm >= 0.5 ? ST_PINHOLE_MIN_COUNT : 6;
  const minNeighborW = ST_MIN_NEIGHBOR_WEIGHT * sm;

  for (let l = 0; l < nL; l++) {
    const row = (liMin + l) * samples + siMin;
    const gRow = l * nS;
    for (let k = 0; k < nS; k++) {
      const idx = row + k;
      const g = gRow + k;
      const powC = powBuf[g]!;
      const cosC = cosBuf[g]!;
      const sinC = sinBuf[g]!;
      const varC = varBuf[g]!;
      const centerClass = clsArr ? clsArr[g]! : p.bloodMask[g]! > 0.5 ? 1 : 0;
      const isClutter = centerClass > 0.25 && centerClass < 0.75;
      const targetClass = powC <= 0 ? 1 : centerClass;

      let nCount = 0;
      let nGeomW = 0;
      let nPowSum = 0;
      let nVarPowSum = 0;
      let nReSum = 0;
      let nImSum = 0;

      for (let dl = -1; dl <= 1; dl++) {
        const ll = l + dl;
        if (ll < 0 || ll >= nL) continue;
        const wl = dl === 0 ? 2 : 1;
        const nRow = ll * nS;
        for (let dk = -1; dk <= 1; dk++) {
          if (dl === 0 && dk === 0) continue;
          const kk = k + dk;
          if (kk < 0 || kk >= nS) continue;
          const h = nRow + kk;
          const pw = powBuf[h]!;
          if (pw <= 0) continue;
          if (clsArr) {
            const nCls = clsArr[h]!;
            if (Math.abs(nCls - targetClass) >= 0.25) continue;
          }
          const wGeom = wl * (dk === 0 ? 2 : 1);
          const wp = wGeom * pw;
          nCount++;
          nGeomW += wGeom;
          nPowSum += wp;
          nVarPowSum += wp * varBuf[h]!;
          nReSum += wp * cosBuf[h]!;
          nImSum += wp * sinBuf[h]!;
        }
      }

      // Case 1: Center uncolored -> safe blood-pool pinhole closure strictly on blood (class >= 0.75)
      if (powC <= 0) {
        if (
          centerClass >= 0.75 &&
          sm >= 0.25 &&
          nCount >= minPinholeCount &&
          nGeomW >= 7.0 &&
          nPowSum > 1e-6
        ) {
          const meanPow = nPowSum / nGeomW;
          if (meanPow >= ST_PINHOLE_MIN_POWER) {
            const angle =
              Math.abs(nReSum) + Math.abs(nImSum) > 1e-12 ? Math.atan2(nImSum, nReSum) : 0;
            const powFill = Math.min(1, meanPow * (0.65 + 0.25 * sm));
            p.dstVel[idx] = aliasVelocity(baseline + invPiScale * angle, scale, baseline);
            p.dstVar[idx] = nVarPowSum / nPowSum;
            p.dstPow[idx] = encodeColorBreakthroughPower(powFill, false, gainDb);
            continue;
          }
        }
        p.dstVel[idx] = NaN;
        p.dstVar[idx] = 0;
        p.dstPow[idx] = 0;
        continue;
      }

      // Case 2: Center colored -> isolated-speckle rejection
      const isolatedSingle = sm >= 0.2 && nCount === 0;
      const isolatedWeakPair =
        sm >= 0.4 &&
        (nCount < ST_MIN_NEIGHBOR_COUNT || nGeomW < minNeighborW) &&
        powC < 0.72;
      if (isolatedSingle || isolatedWeakPair) {
        p.dstVel[idx] = NaN;
        p.dstVar[idx] = 0;
        p.dstPow[idx] = 0;
        continue;
      }

      // Local circular phase coherence & variance regime classification
      const cWeight = 4 * powC;
      const allPowSum = nPowSum + cWeight;
      const allRe = nReSum + cWeight * cosC;
      const allIm = nImSum + cWeight * sinC;
      const spatialCoherence =
        allPowSum > 1e-6 ? Math.min(1, Math.hypot(allRe, allIm) / allPowSum) : 1;
      const varLocal = allPowSum > 1e-6 ? (nVarPowSum + cWeight * varC) / allPowSum : varC;
      const varLaminar = clamp01(1 - varLocal / 0.42);
      const cohLaminar = smoothstep(0.55, 0.92, spatialCoherence);
      const laminarFactor = varLaminar * cohLaminar;

      // Edge-preserving circular phase-similarity bilateral gather (strictly within the same class)
      const bCenterW = 2 * powC;
      let bRe = bCenterW * cosC;
      let bIm = bCenterW * sinC;
      let bVar = bCenterW * varC;
      let bW = bCenterW;
      let bPow = 2 * powC;
      let bGeomW = 2;

      for (let dl = -1; dl <= 1; dl++) {
        const ll = l + dl;
        if (ll < 0 || ll >= nL) continue;
        const wl = dl === 0 ? 2 : 1;
        const nRow = ll * nS;
        for (let dk = -1; dk <= 1; dk++) {
          if (dl === 0 && dk === 0) continue;
          const kk = k + dk;
          if (kk < 0 || kk >= nS) continue;
          const h = nRow + kk;
          const pw = powBuf[h]!;
          if (pw <= 0) continue;
          if (clsArr && Math.abs(clsArr[h]! - centerClass) >= 0.25) continue;
          const wGeom = wl * (dk === 0 ? 2 : 1);
          const cn = cosBuf[h]!;
          const sn = sinBuf[h]!;
          const cosDiff = Math.max(-1, Math.min(1, cn * cosC + sn * sinC));
          const wPhase = Math.exp(-(1 - cosDiff) * ST_PHASE_BILATERAL_K);
          const wBilat = wGeom * pw * wPhase;
          bRe += wBilat * cn;
          bIm += wBilat * sn;
          bVar += wBilat * varBuf[h]!;
          bW += wBilat;
          bPow += wGeom * wPhase * pw;
          bGeomW += wGeom * wPhase;
        }
      }

      const bMag = Math.hypot(bRe, bIm);
      const fRe = bMag > 1e-6 ? bRe / bMag : cosC;
      const fIm = bMag > 1e-6 ? bIm / bMag : sinC;

      const velMix =
        sm * (ST_SPATIAL_TURB_MIX + (ST_SPATIAL_LAMINAR_MIX - ST_SPATIAL_TURB_MIX) * laminarFactor);
      const outRe = (1 - velMix) * cosC + velMix * fRe;
      const outIm = (1 - velMix) * sinC + velMix * fIm;
      const outAngle =
        Math.abs(outRe) + Math.abs(outIm) > 1e-12 ? Math.atan2(outIm, outRe) : Math.atan2(sinC, cosC);

      // Laminar purity vs turbulent mosaic contrast enhancement on variance
      const varSmooth = bW > 1e-6 ? bVar / bW : varC;
      const varLamBase = varC + (varSmooth - varC) * Math.min(1, sm * 0.85);
      const varLaminarClean =
        varLamBase *
        (1 - sm * ST_LAMINAR_VAR_CLEANUP * laminarFactor * clamp01(1 - varSmooth / 0.25));

      const turbConfirm = smoothstep(0.18, 0.45, varLocal);
      const varTurb = clamp01(
        Math.max(varSmooth, varC * 0.88 + varSmooth * 0.12) +
          sm * ST_TURB_VAR_BOOST * turbConfirm * (1 - spatialCoherence),
      );
      const varOut = varTurb + (varLaminarClean - varTurb) * laminarFactor;

      const powSmooth = bGeomW > 1e-6 ? bPow / bGeomW : powC;
      const powMix = sm * (0.25 + 0.45 * laminarFactor);
      const powOut = clamp01(powC + (powSmooth - powC) * powMix);

      p.dstVel[idx] = aliasVelocity(baseline + invPiScale * outAngle, scale, baseline);
      p.dstVar[idx] = varOut;
      p.dstPow[idx] = encodeColorBreakthroughPower(powOut, isClutter, gainDb);
    }
  }
}
