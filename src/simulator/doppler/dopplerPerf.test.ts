import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { loadCaseById } from '@/cases';
import { baseInput } from '@/simulator/core/baseInput';
import { SimulatorCore } from '@/simulator/core/simulatorCore';
import { DEFAULT_SPECTRAL } from '@/simulator/doppler/spectral/spectrum';
import { canonicalControl, getViewTarget } from '@/simulator/windows/viewTargets';
import {
  buildDopplerPerfFrame,
  createSpectralFrameTiming,
  DOPPLER_FRAME_BUDGET_MS,
  DopplerPerfMonitor,
  formatDopplerPerfLine,
  isDopplerModality,
} from './dopplerPerf';

describe('Doppler performance diagnostic utility', () => {
  it('identifies overall and spectral-only bottlenecks and 16 ms budget compliance', () => {
    const sp = createSpectralFrameTiming();
    sp.columns = 2;
    sp.samplesEvaluated = 44;
    sp.momentSetupMs = 0.25;
    sp.acousticAccessMs = 0.4;
    sp.accelChirpMs = 0.35;
    sp.quadratureMs = 2.1;
    sp.valveClicksMs = 0.12;
    sp.spectrumSynthMs = 5.4;
    sp.spectrumPresentMs = 0.45;
    sp.stripRasterMs = 1.1;
    sp.gateInfoMs = 0.18;

    const okFrame = buildDopplerPerfFrame({
      frameId: 10,
      modality: 'pw',
      totalMs: 11.2,
      spectral: sp,
      bmodeRenderMs: 0,
    });
    assert.equal(okFrame.budgetMs, DOPPLER_FRAME_BUDGET_MS);
    assert.equal(okFrame.overBudget, false);
    assert.ok(okFrame.headroomMs > 4.7 && okFrame.headroomMs < 4.9);
    assert.equal(okFrame.bottleneck.stage, 'spectrumSynth');
    assert.equal(okFrame.spectralBottleneck?.stage, 'spectrumSynth');
    assert.equal(okFrame.columns, 2);
    assert.equal(okFrame.samplesEvaluated, 44);
    assert.ok(okFrame.stages.compositeOther >= 0);

    // When duplex B-mode dominates the frame, overall bottleneck is bmodeRender while spectralBottleneck still points to quadrature
    const spCw = createSpectralFrameTiming();
    spCw.columns = 3;
    spCw.samplesEvaluated = 310;
    spCw.quadratureMs = 7.6;
    spCw.spectrumSynthMs = 1.9;
    spCw.stripRasterMs = 1.3;
    const overFrame = buildDopplerPerfFrame({
      frameId: 11,
      modality: 'cw',
      totalMs: 21.5,
      spectral: spCw,
      bmodeRenderMs: 9.8,
    });
    assert.equal(overFrame.overBudget, true);
    assert.equal(overFrame.headroomMs, -5.5);
    assert.equal(overFrame.bottleneck.stage, 'bmodeRender');
    assert.equal(overFrame.spectralBottleneck?.stage, 'quadrature');

    const lineOk = formatDopplerPerfLine(okFrame);
    assert.match(lineOk, /\[DopplerPerf\] #10 \[PW\]/);
    assert.match(lineOk, /OK/);
    assert.match(lineOk, /cuello=spectrumSynth/);

    const lineOver = formatDopplerPerfLine(overFrame);
    assert.match(lineOver, /\[CW\]/);
    assert.match(lineOver, /OVER \(\+5\.50ms\)/);
    assert.match(lineOver, /cuello esp\.: quadrature/);
  });

  it('logs console.log under 16ms, console.warn over 16ms, respects filters, and builds table reports', () => {
    const logs: string[] = [];
    const warns: string[] = [];
    const tables: unknown[] = [];
    const mon = new DopplerPerfMonitor({
      budgetMs: 16,
      maxHistory: 20,
      logger: {
        log: (m) => logs.push(String(m)),
        warn: (m) => warns.push(String(m)),
        table: (d) => tables.push(d),
      },
    });

    const sp1 = createSpectralFrameTiming();
    sp1.columns = 2;
    sp1.spectrumSynthMs = 4.2;
    sp1.quadratureMs = 1.8;
    sp1.stripRasterMs = 0.9;
    mon.record(buildDopplerPerfFrame({ frameId: 1, modality: 'pw', totalMs: 7.4, spectral: sp1 }));

    const sp2 = createSpectralFrameTiming();
    sp2.columns = 3;
    sp2.quadratureMs = 12.5;
    sp2.spectrumSynthMs = 3.2;
    sp2.stripRasterMs = 1.8;
    mon.record(buildDopplerPerfFrame({ frameId: 2, modality: 'cw', totalMs: 18.2, spectral: sp2 }));

    assert.equal(logs.length, 1);
    assert.equal(warns.length, 1);
    assert.match(logs[0]!, /OK/);
    assert.match(warns[0]!, /OVER/);

    // Filter to over-budget only: fast frame should still be stored in history, not logged
    mon.setOnlyOverBudget(true);
    mon.record(buildDopplerPerfFrame({ frameId: 3, modality: 'pw', totalMs: 6.1, spectral: sp1 }));
    assert.equal(logs.length, 1);
    assert.equal(mon.history().length, 3);

    const pwSummary = mon.summary('pw');
    assert.equal(pwSummary.frames, 2);
    assert.equal(pwSummary.overBudgetFrames, 0);
    assert.equal(pwSummary.topSpectralBottleneck?.stage, 'spectrumSynth');

    const allReport = mon.report();
    assert.equal(allReport.frames, 3);
    assert.equal(allReport.overBudgetFrames, 1);
    assert.equal(tables.length, 1);
    assert.ok(Array.isArray(tables[0]) && (tables[0] as unknown[]).length >= 3);

    mon.disable();
    mon.record(buildDopplerPerfFrame({ frameId: 4, modality: 'cw', totalMs: 22.0, spectral: sp2 }));
    assert.equal(warns.length, 2); // 1 from frame #2 + 1 from allReport (p95 > 16), NOT from disabled frame #4
  });

  it('populates real per-stage dopplerPerf timings in SimulatorCore on PW/CW and omits it in 2D B-mode', () => {
    assert.equal(isDopplerModality('pw'), true);
    assert.equal(isDopplerModality('cw'), true);
    assert.equal(isDopplerModality('tdi'), true);
    assert.equal(isDopplerModality('color'), true);
    assert.equal(isDopplerModality('cmm'), true);
    assert.equal(isDopplerModality('2d'), false);
    assert.equal(isDopplerModality('m-mode'), false);

    const caseDef = loadCaseById('normal-excellent-window');
    const core = new SimulatorCore(caseDef, baseInput());
    try {
      const a4c = canonicalControl(getViewTarget('a4c'), core.models.heart, core.models.thorax);

      // 1. 2D B-mode should not attach dopplerPerf
      core.setInput({
        ...baseInput(),
        probe: a4c,
        modality: '2d',
        quality: 'low',
        display: { width: 320, height: 240 },
      });
      const out2d = core.step(0.05);
      assert.ok(out2d);
      assert.equal(out2d.dopplerPerf, undefined);

      // 2. PW spectral Doppler should populate columns, spectral stages, and bottleneck
      core.setInput({
        ...baseInput(),
        probe: a4c,
        modality: 'pw',
        gateDepthCm: 8.5,
        spectral: { ...DEFAULT_SPECTRAL, scaleMps: 1.0, sweepSpeedMmPerS: 50 },
        quality: 'low',
        display: { width: 320, height: 240 },
      });
      const outPw = core.step(0.033);
      assert.ok(outPw?.dopplerPerf, 'Expected dopplerPerf on PW frame');
      assert.equal(outPw.dopplerPerf.modality, 'pw');
      assert.equal(outPw.dopplerPerf.budgetMs, 16);
      assert.ok(outPw.dopplerPerf.columns >= 1, `Expected >=1 spectral columns, got ${outPw.dopplerPerf.columns}`);
      assert.ok(outPw.dopplerPerf.totalMs > 0);
      assert.ok(outPw.dopplerPerf.spectralTotalMs > 0);
      assert.ok(outPw.dopplerPerf.spectralBottleneck !== null);

      // 3. CW spectral Doppler should also populate dopplerPerf with non-zero quadrature/synth stages
      const a5c = canonicalControl(getViewTarget('a5c'), core.models.heart, core.models.thorax);
      core.setInput({
        ...baseInput(),
        probe: a5c,
        modality: 'cw',
        spectral: { ...DEFAULT_SPECTRAL, scaleMps: 6.0, sweepSpeedMmPerS: 50 },
        quality: 'low',
        display: { width: 320, height: 240 },
      });
      const outCw = core.step(0.033);
      assert.ok(outCw?.dopplerPerf, 'Expected dopplerPerf on CW frame');
      assert.equal(outCw.dopplerPerf.modality, 'cw');
      assert.ok(outCw.dopplerPerf.columns >= 1);
      assert.ok(outCw.dopplerPerf.stages.quadrature > 0 || outCw.dopplerPerf.stages.spectrumSynth > 0);
    } finally {
      core.dispose();
    }
  });
});
