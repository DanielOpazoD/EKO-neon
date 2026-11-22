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
