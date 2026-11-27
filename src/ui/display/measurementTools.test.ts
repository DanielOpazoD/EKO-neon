import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  aseSweepDurationS,
  dopplerShiftHz,
  dopplerVelocityFromShiftHz,
  vtiFromCmpsAndMs,
  vtiFromEnvelope,
} from '@/clinical/formulas';
import { summarizeEnvelope } from '@/simulator/measurements/vti';
import { useSimStore } from '@/app/store';
import type { SimOutput } from '@/simulator/core/protocol';
import {
  colorPisaAliasingVelocityMps,
  computeDecelerationTime,
  computeLinearCm,
  computeMModeExcursionCm,
  handleMeasurementToolClick,
  pressureHalfTimeFromDtMs,
  resampleManualVti,
  stripVelocityAtY,
  summarizeManualVti,
  toolActsAt,
} from './measurementTools';
import { Structure } from '@/simulator/anatomy/tissue';
import { computeSpectralTicks } from './overlayDraw';

describe('measurement tools on the live image', () => {
  /** A PW frame: the sector above y = 200, the spectral strip below. */
  const pwFrame = (frozen: boolean) =>
    ({
      frozen,
      sector: { height: 200 },
      strip: { kind: 'spectral', y: 210, height: 150 },
    }) as unknown as SimOutput;

  it('acts where its tool measures: the sector for calipers, its own strip for the others', () => {
    const hud = pwFrame(false);
    assert.equal(toolActsAt('caliper', { x: 50, y: 100 }, hud), true);
    assert.equal(toolActsAt('caliper', { x: 50, y: 300 }, hud), false);
    assert.equal(toolActsAt('velocity', { x: 50, y: 300 }, hud), true);
    assert.equal(toolActsAt('velocity', { x: 50, y: 100 }, hud), false, 'the sector of a PW frame');
    assert.equal(toolActsAt('tapse', { x: 50, y: 300 }, hud), false, 'not an M-mode strip');
    assert.equal(toolActsAt('time', { x: 50, y: 300 }, hud), true);
    assert.equal(toolActsAt('none', { x: 50, y: 300 }, hud), false);
  });

  it('freezes the live image on a click where the tool would measure, and measures nothing yet', () => {
    const st = useSimStore.getState();
    st.loadCase('normal-excellent-window');
    st.setActiveTool('velocity');
    const pend = { points: [] as { x: number; y: number }[] };
    // on the sector, where a strip tool does not measure: nothing happens, as before
    handleMeasurementToolClick({ x: 50, y: 100 }, 1, pwFrame(false), pend, () => {});
    assert.equal(useSimStore.getState().frozen, false);
    // on the strip: the image freezes; the point goes on the frozen frame, with the next click
    handleMeasurementToolClick({ x: 50, y: 300 }, 1, pwFrame(false), pend, () => {});
    assert.equal(useSimStore.getState().frozen, true);
    assert.equal(useSimStore.getState().measurements.length, 0);
    assert.equal(pend.points.length, 0);
    useSimStore.getState().setActiveTool('none');
  });

  it('records Color modality & PISA aliasing velocity on Color calipers and converts slope DT to PHT for PHT specs', () => {
    const st = useSimStore.getState();
    st.setMode('sandbox');
    st.loadCase('mvp-primary-mr');
    st.setModality('color');
    st.setColor({ scaleMps: 0.68, baselineShiftMps: -0.28 });
    st.setActiveMeasurement('mr-pisa-radius');
    const polar = { lines: 8, samples: 8, depthCm: 16, sectorRad: 1.4 };
    const structure = new Uint8Array(64).fill(Structure.LvCavity);
    const colorHud = {
      frozen: true,
      frameId: 10,
      phase: 0.2,
      timeS: 0.2,
      sector: {
        width: 400,
        height: 300,
        apexX: 200,
        apexY: 10,
        pxPerCm: 20,
        depthCm: 16,
        sectorRad: 1.4,
        invertLR: false,
      },
      polar,
      structure,
      strip: { kind: null, y: 300, height: 0, topValue: 0, bottomValue: 0, secondsPerColumn: 0.005 },
      view: { bestViewId: 'a4c', score: 92, components: { gain: 0.9 } },
      gate: null,
      acquisition: {
        modality: 'color',
        color: { scaleMps: 0.68, baselineShiftMps: -0.28 },
        phaseMarks: {
          ejectionStart: 0.08,
          ejectionEnd: 0.38,
          mitralOpen: 0.45,
          eEnd: 0.68,
          aStart: 0.82,
          aEnd: 0.96,
          hasAWave: true,
        },
      },
    } as unknown as SimOutput;
    const pend = { points: [] as { x: number; y: number }[] };
    handleMeasurementToolClick({ x: 200, y: 120 }, 1, colorHud, pend, () => {});
    handleMeasurementToolClick({ x: 200, y: 140 }, 1, colorHud, pend, () => {});
    const pisa = useSimStore.getState().measurements.at(-1)!;
    assert.equal(pisa.measurementId, 'mr-pisa-radius');
    assert.equal(pisa.modality, 'color', 'Color-only protocol measurement must preserve color modality');
    assert.ok(Math.abs(pisa.value - 1.0) < 1e-6);
    assert.ok(
      Math.abs((pisa.derived?.aliasingVelocityMps ?? 0) - 0.4) < 1e-6,
      `Expected v_a = 0.40 m/s, got ${pisa.derived?.aliasingVelocityMps}`,
    );
    assert.equal(
      pisa.technique?.findings.find((f) => f.code === 'modality')?.level,
      'ok',
      'Color caliper must pass Color modality technique check instead of collapsing to 2d',
    );
    assert.ok(Math.abs(colorPisaAliasingVelocityMps({ scaleMps: 0.68, baselineShiftMps: -0.28 }) - 0.4) < 1e-9);

    // PHT spec converts linear DT (200 ms) into Hatle PHT (58 ms) using the same slope tool
    assert.ok(Math.abs(pressureHalfTimeFromDtMs(200) - 58) < 1e-9);
    useSimStore.getState().setModality('cw');
    useSimStore.getState().setActiveMeasurement('ar-pht');
    const cwHud = {
      ...colorHud,
      strip: { kind: 'spectral', y: 300, height: 200, topValue: 4.0, bottomValue: -4.0, secondsPerColumn: 0.005 },
      view: { bestViewId: 'a5c', score: 90, components: { gain: 0.9 } },
      gate: { structure: Structure.Lvot, tissue: 1, flowPresent: true, flowAngleDeg: 10, lineStructures: [Structure.AorticValve, Structure.Lvot] },
      acquisition: { ...colorHud.acquisition, modality: 'cw' },
    } as unknown as SimOutput;
    const pendSlope = { points: [] as { x: number; y: number }[] };
    // topValue=+4, bottomValue=-4 across 200px => 0 m/s at y=400, +4 m/s at y=300, +2 m/s at y=350
    // Drop +4 -> +2 over 20 px (100 ms) => DT = 200 ms => PHT = 58 ms
    handleMeasurementToolClick({ x: 100, y: 300 }, 1, cwHud, pendSlope, () => {});
    handleMeasurementToolClick({ x: 120, y: 350 }, 1, cwHud, pendSlope, () => {});
    const pht = useSimStore.getState().measurements.at(-1)!;
    assert.equal(pht.measurementId, 'ar-pht');
    assert.ok(Math.abs(pht.value - 58) < 1e-6, `Expected PHT 58 ms, got ${pht.value}`);
    assert.ok(Math.abs((pht.derived?.dtMs ?? 0) - 200) < 1e-6);
    useSimStore.getState().clearMeasurements();
  });
});

describe('Display measurement tools math', () => {
  it('computes 2D caliper linear distance and M-mode TAPSE excursion in cm', () => {
    // 3-4-5 triangle at 25 px/cm => 50 px = 2.0 cm
    const dist = computeLinearCm({ x: 100, y: 120 }, { x: 130, y: 160 }, 25);
    assert.ok(Math.abs(dist - 2.0) < 1e-9);

    // M-mode strip 0..16 cm across 160 px => 22 px vertical excursion = 2.2 cm
    const tapse = computeMModeExcursionCm(
      { x: 50, y: 400 },
      { x: 90, y: 378 },
      { height: 160, topValue: 0, bottomValue: 16 },
    );
    assert.ok(Math.abs(tapse - 2.2) < 1e-9);
  });

  it('extrapolates mitral deceleration time to the zero baseline and rejects invalid slopes', () => {
    const strip = {
      y: 300,
      height: 200,
      topValue: 1.5,
      bottomValue: -0.5,
      secondsPerColumn: 0.005, // 5 ms/px
    };
    // Baseline (0 m/s) sits at y = 300 + (1.5 / 2.0) * 200 = 450
    // Peak 1.0 m/s sits at y = 300 + (0.5 / 2.0) * 200 = 350
    // Mid-slope 0.5 m/s sits at y = 300 + (1.0 / 2.0) * 200 = 400
    assert.ok(Math.abs(stripVelocityAtY(350, strip) - 1.0) < 1e-9);
    assert.ok(Math.abs(stripVelocityAtY(400, strip) - 0.5) < 1e-9);

    // Drop from 1.0 to 0.5 m/s over 20 px (100 ms) => full drop to 0 takes 200 ms at x = 140
    const dt = computeDecelerationTime({ x: 100, y: 350 }, { x: 120, y: 400 }, strip);
    assert.ok(dt !== null);
    assert.ok(Math.abs(dt.dtMs - 200) < 1e-6);
    assert.ok(Math.abs(dt.peakMps - 1.0) < 1e-6);
    assert.ok(Math.abs(dt.intercept.x - 140) < 1e-6);
    assert.ok(Math.abs(dt.intercept.y - 450) < 1e-6);

    // Clicking earlier or higher velocity on second point returns null
    assert.equal(computeDecelerationTime({ x: 120, y: 350 }, { x: 100, y: 400 }, strip), null);
    assert.equal(computeDecelerationTime({ x: 100, y: 400 }, { x: 120, y: 350 }, strip), null);
  });

  it('resamples unsorted manual VTI clicks into a triangular spectral envelope with exact VTI', () => {
    const strip = {
      y: 300,
      height: 200,
      topValue: 2.0,
      bottomValue: -2.0,
    };
    // y=400 => 0 m/s, y=350 => +1.0 m/s; triangle base 40 px (0.20 s at 5 ms/px), height 1.0 m/s
    // Theoretical integral = 0.5 * 0.20 s * 1.0 m/s = 0.10 m = 10.0 cm
    const { sorted, velocitiesMps } = resampleManualVti(
      [
        { x: 120, y: 350 },
        { x: 140, y: 400 },
        { x: 100, y: 400 },
      ],
      strip,
    );
    assert.deepEqual(
      sorted.map((p) => p.x),
      [100, 120, 140],
    );
    assert.equal(velocitiesMps.length, 41);
    const summary = summarizeEnvelope(velocitiesMps, 0.005);
    assert.ok(Math.abs(summary.vtiCm - 10.0) < 0.3, `Expected ~10 cm VTI, got ${summary.vtiCm}`);
    assert.ok(Math.abs(summary.vmaxMps - 1.0) < 1e-6);
  });

  it('integrates sub-pixel manual VTI polygons analytically and satisfies ASE cm/s vs ms scaling', () => {
    // ASE frequency-to-velocity round trip: 100 cm/s at 2.5 MHz, c = 1540 m/s
    const f0Hz = 2.5e6;
    const fdHz = dopplerShiftHz(f0Hz, 1.0, 1, 1540);
    assert.ok(Math.abs(dopplerVelocityFromShiftHz(fdHz, f0Hz, 1, 1540) - 1.0) < 1e-12);
    assert.equal(aseSweepDurationS(25), 4.0);
    assert.equal(aseSweepDurationS(50), 2.0);
    assert.equal(aseSweepDurationS(100), 1.0);

    // Clinical unit identity: ∫ v(cm/s)·dt(ms)/1000 === ∫ v(m/s)·dt(s)·100
    const vMps = [0, 0.5, 1.0, 0.5, 0];
    const vCmps = vMps.map((v) => v * 100);
    assert.ok(Math.abs(vtiFromCmpsAndMs(vCmps, 50) - vtiFromEnvelope(vMps, 0.05)) < 1e-12);

    // Non-integer x coordinates across 25 / 50 / 100 mm/s sweep speeds yield identical exact 15.0 cm VTI
    // Triangle of duration 300 ms and peak 100 cm/s => VTI = 0.5 * 100 * 300 / 1000 = 15.0 cm
    // Mean gradient of linear triangle from 0 to 1.0 m/s = (1/3) * 4 * 1^2 = 1.3333 mmHg
    for (const sweep of [25, 50, 100]) {
      const stripWidthPx = 400;
      const spc = aseSweepDurationS(sweep) / stripWidthPx;
      const dxPx = 0.3 / spc; // exact 300 ms span in fractional pixels
      const x0 = 50.37;
      const strip = {
        y: 200,
        height: 200,
        topValue: 1.0,
        bottomValue: -1.0,
        secondsPerColumn: spc,
      };
      // y=300 => 0 cm/s, y=200 => +100 cm/s
      const { summary } = summarizeManualVti(
        [
          { x: x0, y: 300 },
          { x: x0 + dxPx * 0.4, y: 200 },
          { x: x0 + dxPx, y: 300 },
        ],
        strip,
      );
      assert.ok(
        Math.abs(summary.vtiCm - 15.0) < 1e-9,
        `Expected exact 15.0 cm VTI at ${sweep} mm/s, got ${summary.vtiCm}`,
      );
      assert.ok(Math.abs(summary.vmaxMps - 1.0) < 1e-9);
      assert.ok(Math.abs(summary.meanGradientMmHg - 4 / 3) < 1e-9);
    }
  });

  it('computes ASE baseline-anchored velocity ticks in cm/s (PW/TDI) and m/s (CW)', () => {
    // Symmetric PW ±100 cm/s => +100 cm/s, 50, 0, -50, -100 with 0 dead-on frac 0.5
    const pwSym = computeSpectralTicks(1.0, -1.0, true);
    assert.deepEqual(
      pwSym.map((t) => t.label),
      ['100 cm/s', '50', '0', '-50', '-100'],
    );
    const zeroSym = pwSym.find((t) => Math.abs(t.valueMps) < 1e-6);
    assert.ok(zeroSym && Math.abs(zeroSym.frac - 0.5) < 1e-9);

    // Shifted PW baseline (+140 to -60 cm/s) => 0 stays locked exactly at frac = 1.4 / 2.0 = 0.7
    const pwShifted = computeSpectralTicks(1.4, -0.6, true);
    const zeroShifted = pwShifted.find((t) => Math.abs(t.valueMps) < 1e-6);
    assert.ok(zeroShifted && Math.abs(zeroShifted.frac - 0.7) < 1e-9);
    assert.ok(pwShifted[0]!.label.endsWith(' cm/s'));

    // Depth-limited PW Nyquist (±86 cm/s) => clean 40 cm/s or 50 cm/s ticks anchored at 0
    const pwDepth = computeSpectralTicks(0.86, -0.86, true);
    assert.ok(pwDepth.some((t) => Math.abs(t.valueMps) < 1e-6));
    assert.ok(pwDepth.length >= 3 && pwDepth.length <= 7);

    // CW ±6.0 m/s => clean 2 m/s steps in m/s anchored at 0
    const cw = computeSpectralTicks(6.0, -6.0, false);
    assert.deepEqual(
      cw.map((t) => t.label),
      ['6 m/s', '4', '2', '0', '-2', '-4', '-6'],
    );
  });
});
