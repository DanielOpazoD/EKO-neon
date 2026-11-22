import {
  AUDIO_BANDS,
  computeDopplerAudioFrame,
  type DopplerAudioOptions,
} from './dopplerAudioModel';

interface FilterBandNode {
  filter: BiquadFilterNode;
  gain: GainNode;
}

/**
 * Quadrature stochastic blood-scatterer Doppler audio synthesizer.
 * Replaces harsh deterministic oscillators with two uncorrelated Gaussian white-noise streams
 * passed through stereo bandpass filterbanks (Left = toward transducer, Right = away), plus a
 * broadband valve-click transient channel and a soft-limiting dynamics compressor.
 * Driven directly from the active spectral column and depth-limited PRF/Nyquist ceiling.
 */
export class DopplerAudio {
  private ctx: AudioContext | null = null;
  private noiseSource: AudioBufferSourceNode | null = null;
  private towardBands: FilterBandNode[] = [];
  private awayBands: FilterBandNode[] = [];
  private clickFilter: BiquadFilterNode | null = null;
  private clickGain: GainNode | null = null;
  private master: GainNode | null = null;
  private enabled = false;

  start(): void {
    if (this.ctx) return;
    try {
      this.ctx = new AudioContext();
    } catch {
      return;
    }
    const ctx = this.ctx;
    if (ctx.state === 'suspended') void ctx.resume().catch(() => {});

    // 2-second stereo uncorrelated Gaussian white noise loop (blood scatterer stochastic basis).
    const length = Math.max(4096, Math.round(ctx.sampleRate * 2));
    const buffer = ctx.createBuffer(2, length, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const data = buffer.getChannelData(ch);
      // Deterministic LCG + Box-Muller for clean zero-mean unit-variance Gaussian noise.
      let state = (0x9e3779b9 ^ ((ch + 1) * 0x85ebca6b)) >>> 0;
      const nextU = () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return (state + 1) / 4294967297;
      };
      for (let i = 0; i < length; i += 2) {
        const u1 = Math.max(1e-7, nextU());
        const u2 = nextU();
        const r = Math.sqrt(-2 * Math.log(u1)) * 0.35;
        const a = 2 * Math.PI * u2;
        data[i] = r * Math.cos(a);
        if (i + 1 < length) data[i + 1] = r * Math.sin(a);
      }
    }

    const noise = ctx.createBufferSource();
    noise.buffer = buffer;
    noise.loop = true;

    const splitter = ctx.createChannelSplitter(2);
    const merger = ctx.createChannelMerger(2);
    noise.connect(splitter);

    const buildChannel = (channelIdx: number): FilterBandNode[] => {
      const list: FilterBandNode[] = [];
      for (let i = 0; i < AUDIO_BANDS; i++) {
        const filter = ctx.createBiquadFilter();
        filter.type = 'bandpass';
        filter.frequency.value = 300;
        filter.Q.value = 6;
        const gain = ctx.createGain();
        gain.gain.value = 0;
        splitter.connect(filter, channelIdx);
        filter.connect(gain);
        gain.connect(merger, 0, channelIdx);
        list.push({ filter, gain });
      }
      return list;
    };

    this.towardBands = buildChannel(0);
    this.awayBands = buildChannel(1);

    // Valve click channel: crisp mid-low broadband thump fed symmetrically to both ears.
    this.clickFilter = ctx.createBiquadFilter();
    this.clickFilter.type = 'bandpass';
    this.clickFilter.frequency.value = 480;
    this.clickFilter.Q.value = 1.4;
    this.clickGain = ctx.createGain();
    this.clickGain.gain.value = 0;
    splitter.connect(this.clickFilter, 0);
    this.clickFilter.connect(this.clickGain);
    this.clickGain.connect(merger, 0, 0);
    this.clickGain.connect(merger, 0, 1);

    // Subtle clinical speaker high-pass + soft compressor so stenotic jets never clip.
    const hp = ctx.createBiquadFilter();
    hp.type = 'highpass';
    hp.frequency.value = 65;

    const compressor = ctx.createDynamicsCompressor();
    compressor.threshold.value = -16;
    compressor.knee.value = 14;
    compressor.ratio.value = 4;
    compressor.attack.value = 0.004;
    compressor.release.value = 0.08;

    this.master = ctx.createGain();
    this.master.gain.value = 0.35;

    merger.connect(hp);
    hp.connect(compressor);
    compressor.connect(this.master);
    this.master.connect(ctx.destination);

    noise.start();
    this.noiseSource = noise;
    this.enabled = true;
  }

  setVolume(v: number): void {
    if (this.master) this.master.gain.value = Math.max(0, Math.min(1, v)) * 0.55;
  }

  /** Update filterbank from the active spectral column (index 0 = vMax) and acquisition timing. */
  update(
    column: Float32Array | null,
    vMin: number,
    vMax: number,
    f0Hz: number,
    opts?: DopplerAudioOptions,
  ): void {
    if (!this.ctx || !this.enabled) return;
    if (this.ctx.state === 'suspended') void this.ctx.resume().catch(() => {});
    const now = this.ctx.currentTime;
    const nyquistSafeHz = Math.max(200, this.ctx.sampleRate * 0.45);
    const frame = computeDopplerAudioFrame(column, vMin, vMax, f0Hz, opts);

    for (let i = 0; i < AUDIO_BANDS; i++) {
      const tb = this.towardBands[i];
      const td = frame.toward[i];
      if (tb && td) {
        tb.filter.frequency.setTargetAtTime(Math.min(nyquistSafeHz, td.freqHz), now, 0.015);
        tb.filter.Q.setTargetAtTime(td.q, now, 0.025);
        tb.gain.gain.setTargetAtTime(td.gain, now, 0.018);
      }
      const ab = this.awayBands[i];
      const ad = frame.away[i];
      if (ab && ad) {
        ab.filter.frequency.setTargetAtTime(Math.min(nyquistSafeHz, ad.freqHz), now, 0.015);
        ab.filter.Q.setTargetAtTime(ad.q, now, 0.025);
        ab.gain.gain.setTargetAtTime(ad.gain, now, 0.018);
      }
    }

    if (this.clickGain) {
      this.clickGain.gain.setTargetAtTime(frame.clickLevel * 0.28, now, 0.008);
    }
  }

  stop(): void {
    this.enabled = false;
    try {
      this.noiseSource?.stop();
    } catch {
      // already stopped
    }
    this.noiseSource = null;
    this.towardBands = [];
    this.awayBands = [];
    this.clickFilter = null;
    this.clickGain = null;
    this.master = null;
    void this.ctx?.close();
    this.ctx = null;
  }
}
