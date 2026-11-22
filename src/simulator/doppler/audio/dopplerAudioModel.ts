import { SPEED_OF_SOUND_MPS } from '@/core/units';
import { SPECTRAL_BINS } from '../spectral/spectrum';

/** Number of quadrature bandpass channels per stereo direction (Left = toward, Right = away). */
export const AUDIO_BANDS = 24;

/** Safe audible frequency bounds (Hz) for WebAudio BiquadFilter stability and clinical tone. */
export const AUDIO_MIN_HZ = 85;
export const AUDIO_MAX_HZ = 11500;
/** Frequency (Hz) above which extreme CW jet shifts compress smoothly toward AUDIO_MAX_HZ. */
const AUDIO_COMPRESS_KNEE_HZ = 7500;

export interface DopplerAudioOptions {
  /** True for pulsed modes (PW/TDI) where frequencies fold across ±PRF/2; false for CW. */
  aliasing?: boolean;
  /** Physical pulse repetition frequency (Hz) for PW/TDI; bounds audible Nyquist pitch to PRF/2. */
  prfHz?: number;
  /** Wall-filter cutoff velocity (m/s) used to mute low-frequency baseline clutter hum. */
  wallFilterMps?: number;
}

export interface DopplerAudioBand {
  /** Bandpass centre frequency (Hz). */
  freqHz: number;
  /** Bandpass Q factor (higher = cleaner tonal laminar whoosh; lower = breathier turbulent hiss). */
  q: number;
  /** Target linear gain for this band (0..1). */
  gain: number;
}

export interface DopplerAudioFrame {
  /** Forward-channel bands (positive displayed velocity -> Left stereo channel). */
  toward: DopplerAudioBand[];
  /** Reverse-channel bands (negative displayed velocity -> Right stereo channel). */
  away: DopplerAudioBand[];
  /** Broadband click transient energy (0..1) when a valve leaflet crosses the sample volume. */
  clickLevel: number;
  /** Estimated spectral turbulence/bandwidth index (0 = narrow laminar, 1 = broadband turbulent). */
  turbulence: number;
  /** Aliasing fold intensity near the ±Nyquist boundary in PW/TDI (0..1). */
  aliasingIntensity: number;
}

/**
 * Maps a raw Doppler shift magnitude (Hz) into the comfortable audible range without folding CW
 * high-velocity jets into WebAudio Nyquist artifacts.
 */
export function mapAudibleDopplerHz(rawHz: number, maxHz = AUDIO_MAX_HZ): number {
  const absHz = Math.max(0, rawHz);
  if (absHz <= AUDIO_COMPRESS_KNEE_HZ) return Math.max(AUDIO_MIN_HZ, Math.min(maxHz, absHz));
  const excess = absHz - AUDIO_COMPRESS_KNEE_HZ;
  const headroom = Math.max(200, maxHz - AUDIO_COMPRESS_KNEE_HZ);
  const compressed = AUDIO_COMPRESS_KNEE_HZ + headroom * (1 - Math.exp(-excess / headroom));
  return Math.max(AUDIO_MIN_HZ, Math.min(maxHz, compressed));
}

/**
 * Pure spectral-to-audio quadrature filterbank mapper.
 * Integrates EVERY bin of the 128-bin spectral column (no skipped narrow laminar peaks), maps
 * positive velocities to the forward (left) channel and negative velocities to the reverse (right)
 * channel, enforces depth-dependent PW Nyquist ceiling (f <= PRF/2), and derives turbulence/Q and
 * valve-click transient cues directly from the spectral distribution.
 */
export function computeDopplerAudioFrame(
  column: ArrayLike<number> | null,
  vMin: number,
  vMax: number,
  f0Hz: number,
  opts: DopplerAudioOptions = {},
): DopplerAudioFrame {
  const span = vMax - vMin;
  const aliasing = opts.aliasing ?? false;
  const nyquistHz = aliasing && opts.prfHz && opts.prfHz > 0 ? opts.prfHz / 2 : AUDIO_MAX_HZ;
  const ceilingHz = Math.min(AUDIO_MAX_HZ, Math.max(AUDIO_MIN_HZ + 100, nyquistHz));
  const wfMps = Math.max(0, opts.wallFilterMps ?? 0);

  // Maximum displayed velocity magnitude determines the band frequency span on each channel.
  const vPeakSpan = Math.max(Math.abs(vMax), Math.abs(vMin), 0.1);
  const rawMaxHz =
    Number.isFinite(f0Hz) && f0Hz > 0 ? (2 * f0Hz * vPeakSpan) / SPEED_OF_SOUND_MPS : 3500;
  const topFreqHz = mapAudibleDopplerHz(rawMaxHz, ceilingHz);
  const logMin = Math.log(AUDIO_MIN_HZ);
  const logSpan = Math.max(0.2, Math.log(Math.max(AUDIO_MIN_HZ * 1.25, topFreqHz)) - logMin);

  // Quasi-logarithmic frequency centres give fine low-frequency resolution for TDI/venous/mitral
  // flows while covering high-pitched stenotic CW jets smoothly.
  const bandCenters = new Float64Array(AUDIO_BANDS);
  for (let i = 0; i < AUDIO_BANDS; i++) {
    const frac = (i + 0.5) / AUDIO_BANDS;
    bandCenters[i] = Math.exp(logMin + frac * logSpan);
  }

  const posPower = new Float64Array(AUDIO_BANDS);
  const negPower = new Float64Array(AUDIO_BANDS);

  let totalPower = 0;
  let activeBins = 0;
  let edgePower = 0;
  let binMax = 0;

  if (column && Number.isFinite(f0Hz) && f0Hz > 0 && span > 0) {
    const nBins = Math.min(SPECTRAL_BINS, column.length);
    for (let b = 0; b < nBins; b++) {
      const amp = column[b] ?? 0;
      if (amp > binMax) binMax = amp;
    }
    const noiseFloor = Math.max(0.06, binMax * 0.14);
    const scaleMps = span / 2;

    for (let b = 0; b < nBins; b++) {
      const amp = column[b] ?? 0;
      if (amp <= noiseFloor) continue;
      const v = vMax - ((b + 0.5) / nBins) * span;
      if (Math.abs(v) < wfMps * 0.85) continue;

      const excess = (amp - noiseFloor) / Math.max(1e-4, 1 - noiseFloor);
      const p = excess * excess;
      totalPower += p;
      activeBins++;

      // In PW/TDI, energy hugging the top/bottom Nyquist fold indicates active aliasing wrap.
      if (aliasing) {
        const edgeDist = Math.min(Math.abs(vMax - v), Math.abs(v - vMin));
        if (edgeDist < 0.18 * scaleMps) edgePower += p;
      }

      const rawFd = (2 * f0Hz * Math.abs(v)) / SPEED_OF_SOUND_MPS;
      const fHz = mapAudibleDopplerHz(rawFd, ceilingHz);
      const pos = ((Math.log(Math.max(AUDIO_MIN_HZ, fHz)) - logMin) / logSpan) * AUDIO_BANDS - 0.5;
      const i0 = Math.max(0, Math.min(AUDIO_BANDS - 1, Math.floor(pos)));
      const i1 = Math.min(AUDIO_BANDS - 1, i0 + 1);
      const frac = Math.max(0, Math.min(1, pos - i0));
      const target = v >= 0 ? posPower : negPower;
      target[i0] = target[i0]! + p * (1 - frac);
      target[i1] = target[i1]! + p * frac;
    }
  }

  // Bandwidth occupancy distinguishes laminar flow (narrow) from turbulent jets & valve clicks (wide).
  const occupancy = activeBins / SPECTRAL_BINS;
  const clickLevel = occupancy > 0.58 && binMax > 0.35 ? Math.min(1, (occupancy - 0.52) / 0.3) : 0;
  const turbulence = Math.max(0, Math.min(1, (occupancy - 0.06) / 0.35));
  const aliasingIntensity =
    aliasing && totalPower > 1e-5 ? Math.max(0, Math.min(1, edgePower / totalPower)) : 0;

  // Laminar flows get sharper, more musical bandpass Q; turbulent jets & aliased edges widen Q for
  // authentic harsh blood-scatterer hiss.
  const baseQ = 9.5 - 5.2 * turbulence - 1.8 * aliasingIntensity;
  const toward: DopplerAudioBand[] = new Array(AUDIO_BANDS);
  const away: DopplerAudioBand[] = new Array(AUDIO_BANDS);

  // Soft power compression preserves quiet diastolic waves while preventing stenotic jets clipping.
  const gainScale = 0.24 / Math.sqrt(Math.max(1, totalPower * 0.35 + 1));

  for (let i = 0; i < AUDIO_BANDS; i++) {
    const freqHz = bandCenters[i]!;
    // Slight Q taper at low frequencies so bandpass response stays warm and responsive.
    const q = Math.max(2.2, baseQ * (0.75 + 0.35 * (i / (AUDIO_BANDS - 1))));
    toward[i] = {
      freqHz,
      q,
      gain: Math.min(0.45, Math.sqrt(posPower[i]!) * gainScale),
    };
    away[i] = {
      freqHz,
      q,
      gain: Math.min(0.45, Math.sqrt(negPower[i]!) * gainScale),
    };
  }

  return {
    toward,
    away,
    clickLevel,
    turbulence,
    aliasingIntensity,
  };
}
