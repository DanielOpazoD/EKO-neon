import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { listCases, loadCaseById } from '@/cases';
import { computeGroundTruth } from '@/simulator/hemodynamics/groundTruth';
import { buildBeatTables, cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import { caseOutflow } from '@/simulator/cardiac-cycle/outflow';
import { applySriPolar } from '@/simulator/renderer/postprocess/sriFilter';
import { DEFAULT_COLOR, overlayColorField } from '@/simulator/doppler/color/colorDoppler';
import type { ScanLut } from '@/simulator/renderer/scanConvert';

describe('EchoTwin TTE core physiology and clinical cases', () => {
  it('loads and validates all 12 clinical cases with finite ground-truth hemodynamics', () => {
    const catalog = listCases();
    assert.equal(catalog.length, 12, 'Expected 12 clinical cases');

    for (const entry of catalog) {
      const def = loadCaseById(entry.id);
      const truth = computeGroundTruth(def);
      assert.ok(Number.isFinite(truth.lv.efPct) && truth.lv.efPct > 5 && truth.lv.efPct < 90, `${entry.id} EF out of range`);
      assert.ok(Number.isFinite(truth.lv.edvMl) && truth.lv.edvMl > truth.lv.esvMl, `${entry.id} EDV <= ESV`);
      assert.ok(Number.isFinite(truth.lv.eddCm) && truth.lv.eddCm > 2 && truth.lv.eddCm < 10, `${entry.id} EDD out of range`);
      assert.ok(Number.isFinite(truth.aorticValve.vmaxMps) && truth.aorticValve.vmaxMps > 0.4, `${entry.id} AV Vmax invalid`);
      assert.ok(Number.isFinite(truth.wallMotion.wmsi) && truth.wallMotion.wmsi >= 1, `${entry.id} WMSI invalid`);
    }
  });

  it('reproduces clinical ground-truth signatures across key pathologies', () => {
    const normal = computeGroundTruth(loadCaseById('normal-excellent'));
    assert.ok(normal.lv.efPct >= 55 && normal.lv.efPct <= 72, `Normal EF=${normal.lv.efPct}`);
    assert.equal(normal.wallMotion.wmsi, 1);
    assert.ok(normal.aorticValve.effectiveAreaCm2 >= 2.5);

    const severeAs = computeGroundTruth(loadCaseById('aortic-stenosis-severe'));
    assert.ok(severeAs.aorticValve.effectiveAreaCm2 <= 1.0, `Severe AS AVA=${severeAs.aorticValve.effectiveAreaCm2}`);
    assert.ok(severeAs.aorticValve.vmaxMps >= 4.0, `Severe AS Vmax=${severeAs.aorticValve.vmaxMps}`);
    assert.ok(severeAs.aorticValve.meanGradientMmHg >= 40, `Severe AS mean grad=${severeAs.aorticValve.meanGradientMmHg}`);

    const hfref = computeGroundTruth(loadCaseById('hfref-severe-mr'));
    assert.ok(hfref.lv.efPct < 30, `HFrEF EF=${hfref.lv.efPct}`);
    assert.equal(hfref.wallMotion.wmsi, 2);
    assert.ok(hfref.regurgitation.mr !== null && hfref.regurgitation.mr.eroaCm2 >= 0.2);

    const tamp = computeGroundTruth(loadCaseById('pericardial-effusion-tamponade'));
    assert.ok(tamp.pericardium.effusionCm >= 1.5);
    assert.ok(tamp.pericardium.tamponade > 0);

    const af = computeGroundTruth(loadCaseById('af-diastolic'));
    assert.equal(af.mitral.aPeakMps, 0);
    assert.equal(af.mitral.eOverA, null);
  });

  it('builds periodic cardiac cycle beat tables with physiologic valve timing', () => {
    const def = loadCaseById('normal-excellent');
    const rr = 60 / def.rhythm.heartRateBpm;
    const tables = buildBeatTables(rr, def.physiology, def.rhythm, def.hemodynamics, {
      outflow: caseOutflow(def),
    });

    assert.ok(Math.abs(tables.volumeCorrectionMl) < 2.0, `Volume drift too large: ${tables.volumeCorrectionMl}`);
    assert.ok(tables.edvMl > tables.esvMl);

    let maxAvSys = 0;
    let maxMvDia = 0;
    for (let k = 0; k < 20; k++) {
      const s = cycleStateAt(tables, k / 20);
      assert.ok(Number.isFinite(s.lvVolumeMl) && Number.isFinite(s.contraction));
      if (s.timeInBeatS > tables.timings.ejectionStartS && s.timeInBeatS < tables.timings.ejectionEndS) {
        maxAvSys = Math.max(maxAvSys, s.avOpen);
      }
      if (s.timeInBeatS > tables.timings.mitralOpenS) {
        maxMvDia = Math.max(maxMvDia, s.mvOpen);
      }
    }
    assert.ok(maxAvSys > 0.7, `Expected systolic AV opening > 0.7, got ${maxAvSys}`);
    assert.ok(maxMvDia > 0.7, `Expected diastolic MV opening > 0.7, got ${maxMvDia}`);
  });

  it('applies structure-oriented SRI smoothing while preserving DC and sharp endocardial edges', () => {
    const lines = 24;
    const samples = 32;
    const n = lines * samples;

    // 1. Identity at strength = 0 and DC preservation on uniform field
    const uniform = new Float32Array(n).fill(0.42);
    const outZero = new Float32Array(n);
    const outUniform = new Float32Array(n);
    applySriPolar(uniform, outZero, lines, samples, 0);
    applySriPolar(uniform, outUniform, lines, samples, 0.7);
    for (let i = 0; i < n; i++) {
      assert.equal(outZero[i], uniform[i]);
      assert.ok(Math.abs(outUniform[i]! - uniform[i]!) < 1e-5, `DC drift at ${i}: ${outUniform[i]}`);
    }

    // 2. Step edge (blood 0.08 vs wall 0.72) with uncorrelated speckle inside the wall
    const src = new Float32Array(n);
    for (let li = 0; li < lines; li++) {
      for (let si = 0; si < samples; si++) {
        const isWall = si >= 16;
        const hash = Math.sin(li * 12.9898 + si * 78.233) * 43758.5453;
        const speckle = isWall ? ((hash - Math.floor(hash)) * 2 - 1) * 0.08 : 0;
        src[li * samples + si] = (isWall ? 0.72 : 0.08) + speckle;
      }
    }
    const filtered = new Float32Array(n);
    applySriPolar(src, filtered, lines, samples, 0.8);

    // Check speckle variance inside the wall drops significantly while step contrast stays crisp
    let rawVar = 0;
    let filtVar = 0;
    let count = 0;
    for (let li = 3; li < lines - 3; li++) {
      for (let si = 19; si < samples - 3; si++) {
        const idx = li * samples + si;
        rawVar += (src[idx]! - 0.72) ** 2;
        filtVar += (filtered[idx]! - 0.72) ** 2;
        count++;
      }
    }
    assert.ok(filtVar < rawVar * 0.75, `Expected SRI to reduce wall speckle variance: raw=${rawVar / count}, filt=${filtVar / count}`);

    // Sharp step contrast across si=15 -> si=16 must remain strong (> 0.55)
    const midLine = Math.floor(lines / 2);
    const edgeStep = filtered[midLine * samples + 16]! - filtered[midLine * samples + 15]!;
    assert.ok(edgeStep > 0.55, `Expected sharp endocardial step > 0.55, got ${edgeStep}`);

    for (let i = 0; i < n; i++) {
      assert.ok(Number.isFinite(filtered[i]) && filtered[i]! >= 0 && filtered[i]! <= 1);
    }
  });

  it('interpolates Color Doppler in complex Nyquist phase without cancelling aliasing and applies tissue priority', () => {
    // 2x2 polar grid; both output pixels sit midway between all 4 polar samples (wl=0.5, ws=0.5 -> 0.25 each)
    const lines = 2;
    const samples = 2;
    const lut: ScanLut = {
      key: 'test',
      lines,
      samples,
      idx: new Int32Array([0, 0]),
      inPix: new Int32Array([0, 1]),
      inIdx: new Int32Array([0, 0]),
      inWl: new Uint16Array([512, 512]),
      inWs: new Uint16Array([512, 512]),
      inDl: new Uint8Array([1, 1]),
      inDs: new Uint8Array([1, 1]),
      wl: new Float32Array([0.5, 0.5]),
      ws: new Float32Array([0.5, 0.5]),
      li: new Int16Array([0, 0]),
      si: new Int16Array([0, 0]),
      rCm: new Float32Array([6, 6]),
      theta: new Float32Array([0, 0]),
    };

    // Velocity pair straddling ±0.62 m/s Nyquist limit: +0.60 and -0.60 should interpolate near ±0.62 (bright yellow/cyan), NOT 0 m/s!
    const vel = new Float32Array([0.60, -0.60, 0.60, -0.60]);
    const variance = new Float32Array([0, 0, 0, 0]);
    const settings = {
      ...DEFAULT_COLOR,
      scaleMps: 0.62,
      boxRMinCm: 2,
      boxRMaxCm: 10,
      boxThetaMinRad: -0.5,
      boxThetaMaxRad: 0.5,
      showVariance: false,
    };

    // Pixel 0: dark blood (gray = 25), Pixel 1: bright valve/wall (gray = 230)
    const rgba = new Uint8ClampedArray([
      25, 25, 25, 255,
      230, 230, 230, 255,
    ]);
    overlayColorField(rgba, lut, vel, variance, settings);

    // On dark blood pixel: aliased peak velocity should produce high RGB intensity (not ~0 velocity near black/25)
    const bloodMaxCh = Math.max(rgba[0]!, rgba[1]!, rgba[2]!);
    assert.ok(bloodMaxCh > 150, `Expected aliased Nyquist interpolation > 150, got rgb=(${rgba[0]},${rgba[1]},${rgba[2]})`);

    // On bright tissue pixel: tissue priority must preserve underlying grayscale luminance better than blood
    const tissueMinCh = Math.min(rgba[4]!, rgba[5]!, rgba[6]!);
    assert.ok(tissueMinCh > 100, `Expected tissue priority to preserve bright wall structure, got rgb=(${rgba[4]},${rgba[5]},${rgba[6]})`);
  });
});

