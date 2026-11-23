import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { buildCaseModels } from '@/simulator/anatomy/caseModels';
import { CardiacClock } from '@/simulator/cardiac-cycle/clock';
import { DEFAULT_SPECTRAL, SPECTRAL_BINS, spectralRange } from '@/simulator/doppler/spectral/spectrum';
import { buildChainedBeatTables, computePhaseMarks } from '../beatChaining';
import { autoTraceSpectralEnvelope } from './autoTrace';

describe('Extracted G1/G2 pure modules: autoTraceSpectralEnvelope & buildChainedBeatTables', () => {
  it('traces dominant spectral flow envelope and rejects isolated single-column spikes', () => {
    const spectral = { ...DEFAULT_SPECTRAL, scaleMps: 2.0, baselineShiftMps: 0, wallFilterMps: 0.1 };
    const { vMin, vMax } = spectralRange(spectral);
    const cols = 40;
    const strip = new Float32Array(cols * SPECTRAL_BINS);

    const binForVelocity = (v: number) =>
      Math.max(0, Math.min(SPECTRAL_BINS - 1, Math.round(((vMax - v) / (vMax - vMin)) * SPECTRAL_BINS)));

    // Paint a smooth positive systolic parabola (~1.2 m/s peak) across columns 8..28
    for (let x = 8; x <= 28; x++) {
      const u = (x - 8) / 20;
      const vPeak = 1.2 * Math.sin(Math.PI * u);
      if (vPeak <= 0.15) continue;
      const bEdge = binForVelocity(vPeak);
      const bBase = binForVelocity(0.12);
      for (let b = bEdge; b <= bBase; b++) strip[x * SPECTRAL_BINS + b] = 0.85;
    }

    // Inject an isolated single-column spike to 1.9 m/s at x = 18 (should be clamped by SPIKE_MPS)
    const spikeCol = 18;
    const bSpike = binForVelocity(1.9);
    const bBase = binForVelocity(0.12);
    for (let b = bSpike; b <= bBase; b++) strip[spikeCol * SPECTRAL_BINS + b] = 0.9;

    const traced = autoTraceSpectralEnvelope(strip, cols, cols, spectral, 8, 28);
    assert.equal(traced.kind, 'autoTrace');
    assert.equal(traced.velocitiesMps.length, 21);

    const peakVal = Math.max(...traced.velocitiesMps);
    assert.ok(
      peakVal > 0.9 && peakVal < 1.45,
      `Expected smooth peak ~1.2 m/s with 1.9 m/s spike rejected, got ${peakVal.toFixed(2)}`,
    );
    // Spike column (index 10) must be smoothed close to its neighbours, well below 1.7 m/s
    assert.ok(
      traced.velocitiesMps[10]! < 1.45,
      `Single-column spike was not suppressed: ${traced.velocitiesMps[10]}`,
    );
    assert.equal(traced.x0, 8.5, 'Auto-trace x0 should align with painted column centre');
  });

  it('traces ASE modal ridge in PW mode (excluding broadening fringe) and bridges single-column dropouts', () => {
    const spectral = { ...DEFAULT_SPECTRAL, scaleMps: 1.5, baselineShiftMps: 0, wallFilterMps: 0.08 };
    const { vMin, vMax } = spectralRange(spectral);
    const span = vMax - vMin;
    const cols = 40;
    const strip = new Float32Array(cols * SPECTRAL_BINS);

    // Paint a broadened laminar PW band centred at 0.80 m/s with wide low-intensity fringe (sigma 0.14 m/s)
    for (let x = 10; x <= 30; x++) {
      if (x === 20) continue; // Inject a 1-column mid-flow attenuation dropout
      for (let b = 0; b < SPECTRAL_BINS; b++) {
        const v = vMax - ((b + 0.5) / SPECTRAL_BINS) * span;
        const d = (v - 0.8) / 0.14;
        strip[x * SPECTRAL_BINS + b] = Math.exp(-0.5 * d * d);
      }
    }

    const cwOuter = autoTraceSpectralEnvelope(strip, cols, cols, spectral, 10, 30, false);
    const pwModal = autoTraceSpectralEnvelope(strip, cols, cols, spectral, 10, 30, true);

    // PW modal ridge tracks tightly near 0.80 m/s (+~0.14 m/s half-width), well below the CW outer skirt (+~0.20 m/s)
    assert.ok(
      pwModal.velocitiesMps[5]! < cwOuter.velocitiesMps[5]! - 0.04,
      `PW modal trace (${pwModal.velocitiesMps[5]!.toFixed(3)}) should sit inside CW outer fringe (${cwOuter.velocitiesMps[5]!.toFixed(3)})`,
    );
    assert.ok(
      Math.abs(pwModal.velocitiesMps[5]! - 0.94) < 0.04,
      `Expected PW modal edge near ~0.94 m/s, got ${pwModal.velocitiesMps[5]!.toFixed(3)}`,
    );
    // Single-column dropout at x=20 (index 10) must be bridged seamlessly in PW modal mode
    assert.ok(
      Math.abs(pwModal.velocitiesMps[10]! - pwModal.velocitiesMps[9]!) < 0.02,
      `Expected bridged dropout near ${pwModal.velocitiesMps[9]!.toFixed(3)}, got ${pwModal.velocitiesMps[10]!.toFixed(3)}`,
    );

    // Inject an isolated 1-column noise island at x=5 (flanked by 0s) and verify PW modal rejects it
    const islandStrip = new Float32Array(strip);
    for (let b = 0; b < SPECTRAL_BINS; b++) {
      const v = vMax - ((b + 0.5) / SPECTRAL_BINS) * span;
      if (Math.abs(v - 0.35) < 0.06) islandStrip[5 * SPECTRAL_BINS + b] = 0.85;
    }
    const withIsland = autoTraceSpectralEnvelope(islandStrip, cols, cols, spectral, 2, 30, true);
    assert.equal(withIsland.velocitiesMps[3], 0, 'Isolated 1-column PW noise island should be suppressed');

    // Extreme baseline shift pushing zero to the edge must remain finite and well-bounded
    const shifted = autoTraceSpectralEnvelope(
      strip,
      cols,
      cols,
      { ...spectral, baselineShiftMps: 1.5 },
      10,
      30,
      true,
    );
    assert.ok(shifted.velocitiesMps.every((v) => Number.isFinite(v)));
  });

  it('traces the true high-velocity outer jet in bimodal CW double-envelopes (LVOT + AS) while rejecting single-bin noise', () => {
    const spectral = { ...DEFAULT_SPECTRAL, scaleMps: 6.0, baselineShiftMps: 0, wallFilterMps: 0.2 };
    const { vMin, vMax } = spectralRange(spectral);
    const span = vMax - vMin;
    const cols = 30;
    const strip = new Float32Array(cols * SPECTRAL_BINS);

    for (let x = 6; x <= 24; x++) {
      for (let b = 0; b < SPECTRAL_BINS; b++) {
        const v = vMax - ((b + 0.5) / SPECTRAL_BINS) * span;
        // Bright proximal LVOT band (-0.8 m/s, peak 0.95) + spectral dip at -1.5 m/s + outer AS jet (-4.2 m/s, peak 0.65)
        const lvot = 0.95 * Math.exp(-0.5 * ((v + 0.8) / 0.22) ** 2);
        const avJet = 0.65 * Math.exp(-0.5 * ((v + 4.2) / 0.35) ** 2);
        strip[x * SPECTRAL_BINS + b] = Math.max(lvot, avJet);
      }
      // Inject an isolated single-bin noise pixel at -5.7 m/s (must NOT pull CW outer trace)
      const noiseBin = Math.min(SPECTRAL_BINS - 1, Math.floor(((vMax - -5.7) / span) * SPECTRAL_BINS));
      strip[x * SPECTRAL_BINS + noiseBin] = 0.9;
    }

    const cw = autoTraceSpectralEnvelope(strip, cols, cols, spectral, 6, 24, false);
    const mid = Math.abs(cw.velocitiesMps[9]!);
    assert.ok(
      mid >= 4.2 && mid <= 4.85,
      `CW bimodal outer-jet detector should trace the -4.2 m/s AS jet (not stop at -0.8 m/s LVOT or jump to -5.7 m/s single-bin noise), got ${mid.toFixed(2)} m/s`,
    );
  });

  it('chains free-breathing tamponade beats with respiratory mitral/tricuspid modulation and finite phase marks', () => {
    const caseDef = loadCaseById('pericardial-effusion-tamponade');
    const models = buildCaseModels(caseDef, {
      position: 'left-lateral',
      respiration: 'free-breathing',
      headElevationDeg: 0,
    });
    const clock = new CardiacClock(caseDef.rhythm, caseDef.seed);
    clock.advance(0.6);
    const chained = buildChainedBeatTables(
      caseDef,
      clock.current,
      1.2,
      true,
      models.tables,
      models.tables,
    );
    const marks = computePhaseMarks(chained);
    assert.ok(marks.ejectionStart > 0 && marks.ejectionEnd > marks.ejectionStart);
    assert.ok(marks.mitralOpen >= marks.ejectionEnd && marks.eEnd > marks.mitralOpen);
    assert.equal(marks.hasAWave, true);
    assert.ok(Number.isFinite(chained.endVolumeMl) && chained.endVolumeMl > 0);
  });
});
