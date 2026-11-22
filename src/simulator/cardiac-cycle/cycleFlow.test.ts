import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadCaseById } from '@/cases';
import { heartAnchors, computeHeartPose } from '@/simulator/anatomy/heartModel';
import { baseInput } from '@/simulator/core/baseInput';
import { SimulatorCore } from '@/simulator/core/simulatorCore';
import { buildFlowParams, sampleFlow } from '@/simulator/doppler/flow-primitives/flowField';
import {
  displaySpectrum,
  SPECTRAL_BINS,
  spectralDefaultsFor,
} from '@/simulator/doppler/spectral/spectrum';
import { valveClickSignal } from '@/simulator/doppler/spectral/valveClicks';
import { computeGroundTruth } from '@/simulator/hemodynamics/groundTruth';
import { beamFrameFromPose, poseFromControl } from '@/simulator/probe/pose';
import { controlThroughCrux, getViewTarget } from '@/simulator/windows/viewTargets';
import { cycleStateAt, valveEventTimes } from './cycleModel';
import { ecgSample } from './ecg';
import { eWaveShape } from './timing';

describe('Cardiac cycle model & Doppler hemodynamic physiology', () => {
  it('orders phase landmarks and valve events physiologically, and suppresses A-wave in atrial fibrillation', () => {
    const inp = baseInput();
    inp.rendererBackend = 'procedural';
    inp.quality = 'low';

    const normalCore = new SimulatorCore(loadCaseById('normal-excellent-window'), inp);
    const normalMarks = normalCore.phaseMarks();
    const normalValves = valveEventTimes(normalCore.models.tables);

    assert.ok(normalMarks.ejectionStart > 0 && normalMarks.ejectionStart < normalMarks.ejectionEnd);
    assert.ok(normalMarks.ejectionEnd < normalMarks.mitralOpen);
    assert.ok(normalMarks.mitralOpen < normalMarks.eEnd);
    assert.ok(normalMarks.eEnd <= normalMarks.aStart);
    assert.ok(normalMarks.aStart < normalMarks.aEnd);
    assert.equal(normalMarks.hasAWave, true);
    assert.ok(normalValves.aortic[0] < normalValves.aortic[1]);
    assert.ok(normalValves.mitral[0] > normalValves.aortic[1]);
    normalCore.dispose();

    const afCore = new SimulatorCore(loadCaseById('af-diastolic'), inp);
    const afMarks = afCore.phaseMarks();
    assert.equal(afMarks.hasAWave, false);
    afCore.dispose();
  });

  it('contracts LV volume from end-diastole to end-systole with distinct EF in normal vs HFrEF', () => {
    const inp = baseInput();
    inp.rendererBackend = 'procedural';
    inp.quality = 'low';

    const normalCore = new SimulatorCore(loadCaseById('normal-excellent-window'), inp);
    const normalMarks = normalCore.phaseMarks();
    const edNormal = cycleStateAt(normalCore.models.tables, 0);
    const esNormal = cycleStateAt(normalCore.models.tables, normalMarks.endSystole);
    assert.ok(edNormal.lvVolumeMl > esNormal.lvVolumeMl);
    const efNormal = ((edNormal.lvVolumeMl - esNormal.lvVolumeMl) / edNormal.lvVolumeMl) * 100;
    assert.ok(efNormal >= 52 && efNormal <= 74, `Unexpected normal EF: ${efNormal.toFixed(1)}%`);
    normalCore.dispose();

    const hfCore = new SimulatorCore(loadCaseById('hfref-severe-mr'), inp);
    const hfMarks = hfCore.phaseMarks();
    const edHf = cycleStateAt(hfCore.models.tables, 0);
    const esHf = cycleStateAt(hfCore.models.tables, hfMarks.endSystole);
    assert.ok(edHf.lvVolumeMl > esHf.lvVolumeMl);
    const efHf = ((edHf.lvVolumeMl - esHf.lvVolumeMl) / edHf.lvVolumeMl) * 100;
    assert.ok(efHf < 36, `Expected reduced EF (<36%) in HFrEF, got ${efHf.toFixed(1)}%`);
    assert.ok(edHf.lvVolumeMl > edNormal.lvVolumeMl, 'HFrEF LV EDV should be dilated vs normal');
    hfCore.dispose();
  });

  it('differentiates pathological hemodynamic jets across severe AS, primary MR, and obstructive HOCM', () => {
    const normalTruth = computeGroundTruth(loadCaseById('normal-excellent-window'));
    const asTruth = computeGroundTruth(loadCaseById('aortic-stenosis-severe'));
    const mrTruth = computeGroundTruth(loadCaseById('mvp-primary-mr'));
    const hocmTruth = computeGroundTruth(loadCaseById('hocm-sam'));

    // Severe AS: Vmax >= 4.0 m/s, mean gradient >= 40 mmHg, AVA <= 1.0 cm2
    assert.ok(normalTruth.aorticValve.vmaxMps < 1.8);
    assert.ok(
      asTruth.aorticValve.vmaxMps >= 4.0,
      `Expected severe AS Vmax >= 4.0, got ${asTruth.aorticValve.vmaxMps}`,
    );
    assert.ok(asTruth.aorticValve.meanGradientMmHg >= 40);
    assert.ok(asTruth.aorticValve.continuityAvaCm2 <= 1.0);

    // Primary severe MR: high-velocity regurgitant jet and large EROA
    assert.ok(mrTruth.regurgitation.mr !== null);
    assert.ok(mrTruth.regurgitation.mr.eroaCm2 >= 0.4);
    assert.ok(mrTruth.regurgitation.mr.regurgitantVolumeMl >= 60);

    // Obstructive HOCM: dynamic LVOT obstruction with peak gradient >= 30 mmHg and asymmetric septal hypertrophy
    assert.equal(hocmTruth.lvot.dynamicObstruction, true);
    assert.ok(hocmTruth.lvot.peakGradientMmHg >= 30);
    assert.ok(hocmTruth.lv.ivsdCm >= 1.5);
  });

  it('produces faithful healthy A4C transmitral E/A curves at the mitral leaflet tips without systolic bleed', () => {
    const c = loadCaseById('normal-excellent-window');
    const inp = baseInput();
    inp.rendererBackend = 'procedural';
    inp.quality = 'low';
    const core = new SimulatorCore(c, inp);
    const { heart, tables, thorax } = core.models;
    const flow = buildFlowParams(c, heart, tables, thorax);
    const a4cControl = controlThroughCrux(getViewTarget('a4c'), heart, thorax, null);
    const beam = beamFrameFromPose(poseFromControl(thorax, a4cControl), 1);
    const A = heartAnchors(heart);
    const hf = heart.frame;
    const tm = tables.timings;

    // 1. E-wave morphology: brisk normal AT (60-80 ms), asymmetric (AT < 0.45*DT), non-convex linear deceleration
    assert.ok(
      tm.eAccelS >= 0.06 && tm.eAccelS <= 0.08,
      `Normal mitral E acceleration time should be 60–80 ms, got ${(tm.eAccelS * 1000).toFixed(1)} ms`,
    );
    const midDecel = eWaveShape(tm.eAccelS + 0.5 * tm.eDecelS, tm.eAccelS, tm.eDecelS);
    assert.ok(
      midDecel <= 0.505 && midDecel >= 0.44,
      `E-wave mid-deceleration should be linear/faintly concave (0.44–0.505), got ${midDecel.toFixed(3)}`,
    );

    // 2. ECG R-peak aligns cleanly shortly after A-wave closure (no 50 ms dead gap)
    let rPeakS = 0;
    let rPeakV = -Infinity;
    for (let i = 0; i < 200; i++) {
      const t = (i / 200) * 0.1;
      const v = ecgSample(t, tables.rrS, c.rhythm, 0, c.seed);
      if (v > rPeakV) {
        rPeakV = v;
        rPeakS = t;
      }
    }
    const mvcToRMs = (tables.rrS - tm.aEndS + rPeakS) * 1000;
    assert.ok(
      mvcToRMs >= 10 && mvcToRMs <= 38,
      `Mitral closure should sit right at QRS onset (~10–38 ms before R peak), got ${mvcToRMs.toFixed(1)} ms`,
    );

    // 3. Place PW gate at the anatomical mitral leaflet tips in the A4C plane
    const hpDia = computeHeartPose(heart, cycleStateAt(tables, (tm.mitralOpenS + tm.eAccelS) / tables.rrS));
    const tipWorld = {
      x: hf.origin.x + hf.ex.x * A.mvCenter.x + hf.ey.x * A.mvCenter.y + hf.ez.x * (hpDia.zAnn + 1.2),
      y: hf.origin.y + hf.ex.y * A.mvCenter.x + hf.ey.y * A.mvCenter.y + hf.ez.y * (hpDia.zAnn + 1.2),
      z: hf.origin.z + hf.ex.z * A.mvCenter.x + hf.ey.z * A.mvCenter.y + hf.ez.z * (hpDia.zAnn + 1.2),
    };
    const vx = tipWorld.x - beam.origin.x;
    const vy = tipWorld.y - beam.origin.y;
    const vz = tipWorld.z - beam.origin.z;
    const fwd = vx * beam.forward.x + vy * beam.forward.y + vz * beam.forward.z;
    const lat = vx * beam.lateral.x + vy * beam.lateral.y + vz * beam.lateral.z;
    const rTips = Math.hypot(fwd, lat);
    const theta = Math.atan2(lat, fwd);
    const dx = beam.forward.x * Math.cos(theta) + beam.lateral.x * Math.sin(theta);
    const dy = beam.forward.y * Math.cos(theta) + beam.lateral.y * Math.sin(theta);
    const dz = beam.forward.z * Math.cos(theta) + beam.lateral.z * Math.sin(theta);
    const px = beam.origin.x + dx * rTips;
    const py = beam.origin.y + dy * rTips;
    const pz = beam.origin.z + dz * rTips;
    const hx = (px - hf.origin.x) * hf.ex.x + (py - hf.origin.y) * hf.ex.y + (pz - hf.origin.z) * hf.ex.z;
    const hy = (px - hf.origin.x) * hf.ey.x + (py - hf.origin.y) * hf.ey.y + (pz - hf.origin.z) * hf.ey.z;
    const hz = (px - hf.origin.x) * hf.ez.x + (py - hf.origin.y) * hf.ez.y + (pz - hf.origin.z) * hf.ez.z;
    const dhx = dx * hf.ex.x + dy * hf.ex.y + dz * hf.ex.z;
    const dhy = dx * hf.ey.x + dy * hf.ey.y + dz * hf.ey.z;
    const dhz = dx * hf.ez.x + dy * hf.ez.y + dz * hf.ez.z;

    const sample = { vx: 0, vy: 0, vz: 0, dispersion: 0, present: 0 };
    let peakE = 0;
    let peakA = 0;
    let maxSystolicAbs = 0;
    let maxDiastasisAbs = 0;
    for (let i = 0; i < 240; i++) {
      const ph = i / 240;
      const t = ph * tables.rrS;
      const hp = computeHeartPose(heart, cycleStateAt(tables, ph));
      sampleFlow(flow, tables, hp, ph, hx, hy, hz, sample);
      const vAxial = sample.present ? -(sample.vx * dhx + sample.vy * dhy + sample.vz * dhz) : 0;
      if (t >= tm.ejectionStartS && t <= tm.ejectionEndS) {
        maxSystolicAbs = Math.max(maxSystolicAbs, Math.abs(vAxial));
      } else if (t >= tm.mitralOpenS && t < tm.aStartS) {
        peakE = Math.max(peakE, vAxial);
        if (t > tm.eEndS + 0.01 && t < tm.aStartS - 0.01) {
          maxDiastasisAbs = Math.max(maxDiastasisAbs, Math.abs(vAxial));
        }
      } else if (t >= tm.aStartS) {
        peakA = Math.max(peakA, vAxial);
      }
    }

    const expectedE = c.physiology.ePeakMps;
    const expectedA = c.physiology.aPeakMps;
    assert.ok(
      Math.abs(peakE - expectedE) / expectedE < 0.08,
      `A4C mitral tips E peak (${peakE.toFixed(3)} m/s) should match case truth (${expectedE} m/s) within 8%`,
    );
    assert.ok(
      Math.abs(peakA - expectedA) / expectedA < 0.08,
      `A4C mitral tips A peak (${peakA.toFixed(3)} m/s) should match case truth (${expectedA} m/s) within 8%`,
    );
    const eaRatio = peakE / peakA;
    assert.ok(
      eaRatio >= 1.35 && eaRatio <= 1.6,
      `Healthy A4C transmitral E/A ratio should stay ~1.45, got ${eaRatio.toFixed(2)}`,
    );
    assert.ok(
      maxSystolicAbs < 0.06,
      `A4C mitral tips gate should stay clean below wall filter in systole, got ${maxSystolicAbs.toFixed(3)} m/s`,
    );
    assert.ok(
      maxDiastasisAbs < 0.02,
      `Healthy 65 bpm transmitral diastasis should return to baseline, got ${maxDiastasisAbs.toFixed(3)} m/s`,
    );

    // 4. Displayed valve clicks stay crisp near baseline without saturating the ±scale edges
    const sPw = spectralDefaultsFor('pw');
    const clickDisplay = new Float32Array(SPECTRAL_BINS);
    displaySpectrum(new Float32Array(SPECTRAL_BINS), sPw, 101, 1.0, 0.48, clickDisplay, 2.5, 0.72);
    const edgeBrightness = Math.max(clickDisplay[2]!, clickDisplay[SPECTRAL_BINS - 3]!);
    const nearBaselineBrightness = Math.max(
      clickDisplay[Math.floor(SPECTRAL_BINS * 0.42)]!,
      clickDisplay[Math.floor(SPECTRAL_BINS * 0.58)]!,
    );
    assert.ok(
      nearBaselineBrightness > 0.55,
      `Valve click should remain clearly visible near baseline (${nearBaselineBrightness.toFixed(2)})`,
    );
    assert.ok(
      edgeBrightness < 0.2,
      `Valve click should fade before full ±Nyquist edge (${edgeBrightness.toFixed(2)})`,
    );

    // 5. 3D valve-click polarity at mitral tips in A4C: MVO snaps toward probe (+), MVC snaps toward LA (-)
    const mvEvents = valveEventTimes(tables).mitral;
    const hpOpen = computeHeartPose(heart, cycleStateAt(tables, mvEvents[0] / tables.rrS));
    const hpClose = computeHeartPose(heart, cycleStateAt(tables, mvEvents[1] / tables.rrS));
    const tipPts = [hx, hy, hz];
    const rayDir = { x: dhx, y: dhy, z: dhz };
    const mvoSig = valveClickSignal(heart, hpOpen, tables, mvEvents[0], tipPts, rayDir);
    const mvcSig = valveClickSignal(heart, hpClose, tables, mvEvents[1], tipPts, rayDir);
    assert.ok(
      mvoSig.weight > 0.25 && mvoSig.direction > 0.6,
      `A4C mitral opening click should project upward toward probe, got weight=${mvoSig.weight.toFixed(2)} dir=${mvoSig.direction.toFixed(2)}`,
    );
    assert.ok(
      mvcSig.weight > 0.25 && mvcSig.direction < -0.6,
      `A4C mitral closing click should project downward toward LA, got weight=${mvcSig.weight.toFixed(2)} dir=${mvcSig.direction.toFixed(2)}`,
    );

    core.dispose();
  });

  it('measures high-fidelity PW auto-trace values on healthy A4C mitral inflow and syncs depth-limited Nyquist on A5C LVOT', () => {
    const c = loadCaseById('normal-excellent-window');
    const runPw = (
      viewId: 'a4c' | 'a5c',
      gateCm: number,
      thetaDeg: number,
      scaleMps: number,
      shiftMps: number,
    ) => {
      const inp = baseInput();
      inp.rendererBackend = 'procedural';
      inp.quality = 'low';
      inp.modality = 'pw';
      inp.gateDepthCm = gateCm;
      inp.cursorThetaRad = (thetaDeg * Math.PI) / 180;
      inp.spectral = {
        ...inp.spectral,
        scaleMps,
        baselineShiftMps: shiftMps,
        sweepSpeedMmPerS: 50,
        gateLengthCm: 0.4,
        wallFilterMps: 0.08,
        gainDb: 0,
      };
      inp.display = { width: 640, height: 640 };
      const core = new SimulatorCore(c, inp);
      inp.probe = controlThroughCrux(getViewTarget(viewId), core.models.heart, core.models.thorax, null);
      core.setInput(inp);
      // Sweep ~1.75 s (< 2.0 s strip width) so beat 1 is complete and never wraps across the sweep head
      let frame = core.step(1 / 30);
      for (let i = 0; i < 52; i++) frame = core.step(1 / 30);
      assert.ok(frame !== null, 'Expected rendered frame after stepping PW strip');
      const res = core.request({ kind: 'autoTrace', x0: 0, x1: frame.strip.width - 1 });
      core.dispose();
      return { frame, res };
    };

    // 1. A4C Mitral PW auto-trace accuracy vs case physiology
    const a4c = runPw('a4c', 9.42, 3.7, 1.0, 0);
    assert.ok(a4c.res && a4c.res.kind === 'autoTrace');
    const v = a4c.res.velocitiesMps;
    const spc = a4c.res.secondsPerColumn;
    let start = -1;
    let end = -1;
    for (let i = 10; i < v.length - 10; i++) {
      if (start < 0 && v[i]! > 0.15) start = i;
      else if (start >= 0 && v[i]! <= 0.05) {
        let cont = false;
        for (let k = i; k < Math.min(v.length, i + 45); k++) if (v[k]! > 0.25) cont = true;
        if (!cont && i - start > 60) {
          end = i;
          break;
        }
      }
    }
    assert.ok(start >= 0 && end > start, 'Expected at least one complete mitral diastolic beat on strip');
    let ePeak = 0;
    let eIdx = start;
    for (let k = start; k < Math.min(end, start + Math.round(0.18 / spc)); k++) {
      if (v[k]! > ePeak) {
        ePeak = v[k]!;
        eIdx = k;
      }
    }
    let aPeak = 0;
    for (let k = eIdx + Math.round(0.18 / spc); k < end; k++) aPeak = Math.max(aPeak, v[k]!);
    let mvVtiCm = 0;
    for (let k = start; k < end; k++) mvVtiCm += Math.max(0, v[k]!) * spc * 100;
    let i85 = eIdx;
    let i25 = eIdx;
    for (let k = eIdx; k < Math.min(end, eIdx + Math.round(0.25 / spc)); k++) {
      if (v[k]! <= 0.85 * ePeak && i85 === eIdx) i85 = k;
      if (v[k]! <= 0.25 * ePeak) {
        i25 = k;
        break;
      }
    }
    const slope = (v[i85]! - v[i25]!) / Math.max(1e-6, (i25 - i85) * spc);
    const dtMs = (ePeak / Math.max(1e-6, slope)) * 1000;

    assert.ok(Math.abs(ePeak - 0.8) / 0.8 < 0.05, `Auto-traced E (${ePeak.toFixed(3)}) outside ±5% of 0.80 m/s`);
    assert.ok(Math.abs(aPeak - 0.55) / 0.55 < 0.06, `Auto-traced A (${aPeak.toFixed(3)}) outside ±6% of 0.55 m/s`);
    assert.ok(dtMs >= 165 && dtMs <= 200, `Auto-traced DT (${dtMs.toFixed(1)} ms) outside [165, 200] ms`);
    assert.ok(mvVtiCm >= 13.0 && mvVtiCm <= 15.0, `Auto-traced MV VTI (${mvVtiCm.toFixed(2)} cm) outside [13, 15] cm`);

    // 2. Depth-limited PW Nyquist synchronization on deep A5C LVOT gate
    const a5c = runPw('a5c', 11.2, 0, 1.5, -0.35);
    assert.ok(a5c.res && a5c.res.kind === 'autoTrace');
    const nyquist = a5c.frame.strip.nyquistMps!;
    assert.ok(nyquist < 1.0, `Expected depth-limited Nyquist < 1.0 m/s at 11.2 cm, got ${nyquist.toFixed(3)}`);
    assert.ok(
      Math.abs(a5c.frame.strip.topValue - (nyquist - 0.35)) < 1e-4 &&
        Math.abs(a5c.frame.strip.bottomValue - (-nyquist - 0.35)) < 1e-4,
      `Strip top/bottom [${a5c.frame.strip.bottomValue.toFixed(3)}, ${a5c.frame.strip.topValue.toFixed(3)}] must match depth-limited Nyquist ±${nyquist.toFixed(3)} - 0.35`,
    );
    let lvotPeak = 0;
    for (const vel of a5c.res.velocitiesMps) lvotPeak = Math.max(lvotPeak, Math.abs(Math.min(0, vel)));
    assert.ok(
      lvotPeak >= 0.95 && lvotPeak <= 1.2,
      `Baseline-shifted A5C LVOT PW peak should read unaliased ~1.04 m/s, got ${lvotPeak.toFixed(3)} m/s`,
    );
  });
});
