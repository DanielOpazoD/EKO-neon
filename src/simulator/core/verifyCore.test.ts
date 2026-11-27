import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { CASE_INPUTS, listCases, loadCaseById } from '@/cases';
import { validateCase } from '@/cases/schema';
import { computeGroundTruth } from '@/simulator/hemodynamics/groundTruth';
import { buildBeatTables, cycleStateAt } from '@/simulator/cardiac-cycle/cycleModel';
import { caseOutflow } from '@/simulator/cardiac-cycle/outflow';
import { applySriPolar } from '@/simulator/renderer/postprocess/sriFilter';
import { DEFAULT_COLOR, overlayColorField } from '@/simulator/doppler/color/colorDoppler';
import {
  buildScanLut,
  computeSectorMapping,
  scanConvertLut,
  scanConvertLutReference,
  type ScanLut,
} from '@/simulator/renderer/scanConvert';
import {
  TINT_MAP_OPTIONS,
  tintGrayByte,
  tintPackedLut,
} from '@/simulator/renderer/postprocess/tintMap';
import { computeAutoOpt } from '@/simulator/renderer/postprocess/autoOpt';
import { Tissue } from '@/simulator/anatomy/tissue';
import { CineBuffer, type CineFrame } from '@/simulator/core/cineBuffer';
import { applyConsole, createConsoleState } from '@/simulator/renderer/postprocess/consolePipeline';
import { allocPolarFrame, DEFAULT_ACQUISITION } from '@/simulator/renderer/types';
import { DEFAULT_SPECTRAL } from '@/simulator/doppler/spectral/spectrum';

describe('EchoTwin TTE core physiology and clinical cases', () => {
  it('loads and validates all clinical cases with finite ground-truth hemodynamics and Zod schema parity', () => {
    const catalog = listCases();
    assert.equal(catalog.length, 13, 'Expected 13 clinical cases');

    for (const raw of CASE_INPUTS) {
      const validated = validateCase(raw);
      assert.equal(validated.ok, true, `Zod validation failed for ${raw.id}: ${validated.errors.join('; ')}`);
      const def = loadCaseById(raw.id);
      assert.deepEqual(def, validated.case, `normalizeCase drifted from Zod CaseDefinitionSchema for ${raw.id}`);
      const truth = computeGroundTruth(def);
      assert.ok(Number.isFinite(truth.lv.efPct) && truth.lv.efPct > 5 && truth.lv.efPct < 90, `${raw.id} EF out of range`);
      assert.ok(Number.isFinite(truth.lv.edvMl) && truth.lv.edvMl > truth.lv.esvMl, `${raw.id} EDV <= ESV`);
      assert.ok(Number.isFinite(truth.lv.eddCm) && truth.lv.eddCm > 2 && truth.lv.eddCm < 10, `${raw.id} EDD out of range`);
      assert.ok(Number.isFinite(truth.aorticValve.vmaxMps) && truth.aorticValve.vmaxMps > 0.4, `${raw.id} AV Vmax invalid`);
      assert.ok(Number.isFinite(truth.wallMotion.wmsi) && truth.wallMotion.wmsi >= 1, `${raw.id} WMSI invalid`);
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

  it('performs anti-ringing bicubic Catmull-Rom scan conversion with exact reference parity and chroma tint LUTs', () => {
    const spec = {
      lines: 48,
      samples: 64,
      elevationSamples: 1,
      sectorRad: (78 * Math.PI) / 180,
      depthCm: 16,
      focusCm: 8,
    };
    const mapping = computeSectorMapping(spec, 96, 80, false, 1);
    const lut = buildScanLut(spec, mapping);

    // Sharp high-contrast step edge (10 -> 240) across depth and azimuth to test anti-ringing clamp
    const polar = new Uint8ClampedArray(spec.lines * spec.samples);
    for (let li = 0; li < spec.lines; li++) {
      for (let si = 0; si < spec.samples; si++) {
        polar[li * spec.samples + si] = li >= 24 && si >= 32 ? 240 : 10;
      }
    }

    const rgbaFast = new Uint8ClampedArray(96 * 80 * 4);
    const rgbaRef = new Uint8ClampedArray(96 * 80 * 4);
    scanConvertLut(polar, lut, rgbaFast, 'gray');
    scanConvertLutReference(polar, lut, rgbaRef, 'gray');

    let insideCount = 0;
    for (let p = 0; p < 96 * 80; p++) {
      const o = p * 4;
      assert.equal(rgbaFast[o], rgbaRef[o], `Fast vs ref R mismatch at pixel ${p}`);
      assert.equal(rgbaFast[o + 1], rgbaRef[o + 1], `Fast vs ref G mismatch at pixel ${p}`);
      assert.equal(rgbaFast[o + 2], rgbaRef[o + 2], `Fast vs ref B mismatch at pixel ${p}`);
      if (lut.idx[p]! >= 0) {
        insideCount++;
        const v = rgbaFast[o]!;
        assert.ok(v >= 10 && v <= 240, `Catmull-Rom ringing escaped [10, 240]: got ${v} at pixel ${p}`);
      }
    }
    assert.ok(insideCount > 500, `Expected > 500 pixels inside sector, got ${insideCount}`);

    // Verify all 4 chroma tint maps preserve 0 -> black, 255 -> white, monotonicity, and packed LUT parity
    const rgb: [number, number, number] = [0, 0, 0];
    for (const opt of TINT_MAP_OPTIONS) {
      tintGrayByte(0, opt.id, rgb);
      assert.deepEqual(rgb, [0, 0, 0], `${opt.id} must map 0 to black`);
      tintGrayByte(255, opt.id, rgb);
      assert.deepEqual(rgb, [255, 255, 255], `${opt.id} must map 255 to white`);

      const packed = tintPackedLut(opt.id);
      const view = new Uint8Array(packed.buffer, packed.byteOffset, packed.byteLength);
      let prevLum = -1;
      for (let g = 0; g < 256; g++) {
        tintGrayByte(g, opt.id, rgb);
        assert.equal(view[g * 4], rgb[0], `${opt.id} packed R mismatch at ${g}`);
        assert.equal(view[g * 4 + 1], rgb[1], `${opt.id} packed G mismatch at ${g}`);
        assert.equal(view[g * 4 + 2], rgb[2], `${opt.id} packed B mismatch at ${g}`);
        const lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
        assert.ok(lum >= prevLum - 0.5, `${opt.id} luminance non-monotonic at ${g}: ${lum} < ${prevLum}`);
        prevLum = lum;
      }
    }
  });

  it('auto-optimizes gain and depth TGC for dark/attenuated frames while guarding washed-out cavities', () => {
    const lines = 40;
    const samples = 64;
    const n = lines * samples;
    const tissue = new Uint8Array(n).fill(Tissue.Myocardium);

    // 1. Dark & far-field attenuated frame (near p85 ~ 95, far p85 ~ 35)
    const dark = new Uint8ClampedArray(n);
    for (let li = 0; li < lines; li++) {
      for (let si = 0; si < samples; si++) {
        const depthFrac = si / (samples - 1);
        // 60% cavity (dark) + 40% myocardium wall (attenuating with depth)
        const isWall = li % 5 >= 3;
        dark[li * samples + si] = isWall ? Math.round(95 - 60 * depthFrac) : 14;
      }
    }
    const optDark = computeAutoOpt(
      dark,
      tissue,
      { lines, samples },
      { gainDb: 0, tgcDb: [0, 0, 0, 0, 0, 0, 0, 0], dynamicRangeDb: 70 },
    );
    assert.ok(optDark.gainDb > 0, `Expected Auto-Opt to boost gain on dark frame, got ${optDark.gainDb}`);
    assert.ok(
      optDark.tgcDb[7]! > optDark.tgcDb[0]!,
      `Expected Auto-Opt to tilt far TGC higher than near TGC: near=${optDark.tgcDb[0]}, far=${optDark.tgcDb[7]}`,
    );

    // 2. Over-gained frame with washed-out blood pool (p25 = 115 > 64)
    const bright = new Uint8ClampedArray(n);
    for (let li = 0; li < lines; li++) {
      for (let si = 0; si < samples; si++) {
        const isWall = li % 5 >= 3;
        bright[li * samples + si] = isWall ? 225 : 115;
      }
    }
    const optBright = computeAutoOpt(
      bright,
      tissue,
      { lines, samples },
      { gainDb: 10, tgcDb: [0, 0, 0, 0, 0, 0, 0, 0], dynamicRangeDb: 70 },
    );
    assert.ok(
      optBright.gainDb < 10,
      `Expected Auto-Opt to lower gain on washed-out frame, got ${optBright.gainDb}`,
    );
  });

  it('recycles CineBuffer typed arrays in-place across wrap-around and snaps accurately to ED/ES phases', () => {
    const buf = new CineBuffer(8);
    const spec = {
      lines: 8,
      samples: 10,
      elevationSamples: 1,
      sectorRad: 1.2,
      depthCm: 16,
      focusCm: 8,
    };
    const makeFrame = (id: number, phase: number, val: number): CineFrame => ({
      acquisition: {
        modality: '2d',
        probe: { u: 0, v: 0, rotationDeg: 0, tiltDeg: 0, rockDeg: 0, pressure: 0.5 },
        patient: { position: 'left-lateral', respiration: 'expiration', headElevationDeg: 0 },
        settings: { ...DEFAULT_ACQUISITION, tgcDb: [...DEFAULT_ACQUISITION.tgcDb] },
        color: { ...DEFAULT_COLOR },
        spectral: { ...DEFAULT_SPECTRAL },
        cursorThetaRad: 0,
        gateDepthCm: 8,
        modelVersion: 1,
        tablesVersion: 1,
        phaseMarks: {
          ejectionStart: 0.06,
          ejectionEnd: 0.36,
          endSystole: 0.35,
          mitralOpen: 0.44,
          eEnd: 0.65,
          aStart: 0.84,
          aEnd: 0.96,
          hasAWave: true,
        },
        stripAcquisitionId: null,
        colorTimeS: null,
      },
      formation: {} as CineFrame['formation'],
      scene: {} as CineFrame['scene'],
      tables: {} as CineFrame['tables'],
      flow: {} as CineFrame['flow'],
      gate: null,
      spectrumColumn: null,
      spectralRange: { vMin: -1.5, vMax: 1.5 },
      display: new Uint8ClampedArray(80).fill(val),
      structure: new Uint8Array(80).fill(1),
      segment: new Uint8Array(80).fill(2),
      spec,
      frameId: id,
      rrS: 0.92,
      phase,
      timeS: id * 0.05,
      beatIndex: Math.floor(id / 8),
      colorVel: null,
      colorVar: null,
      beam: {
        origin: { x: 0, y: 0, z: 0 },
        forward: { x: 0, y: 0, z: 1 },
        lateral: { x: 1, y: 0, z: 0 },
        normal: { x: 0, y: 1, z: 0 },
        contact: 1,
      },
      analysis: null,
    });

    for (let i = 0; i < 8; i++) buf.capture(makeFrame(i, i / 8, 10 + i));
    const slot0Before = buf.oldest!;
    const displayRefBefore = slot0Before.display;
    const structureRefBefore = slot0Before.structure;
    const segmentRefBefore = slot0Before.segment;

    // Capturing the 9th frame overwrites slot 0 and must reuse its TypedArrays in-place
    buf.capture(makeFrame(8, 0.02, 222));
    const newest = buf.newest!;
    assert.equal(newest.display, displayRefBefore, 'CineBuffer should reuse display Uint8ClampedArray');
    assert.equal(newest.structure, structureRefBefore, 'CineBuffer should reuse structure Uint8Array');
    assert.equal(newest.segment, segmentRefBefore, 'CineBuffer should reuse segment Uint8Array');
    assert.equal(newest.display[0], 222);

    // Phase snap across 2 full cardiac cycles prefers the latest beat's ED (0.0) and ES (0.35)
    const cine2 = new CineBuffer(40);
    for (let i = 0; i < 40; i++) {
      cine2.capture(makeFrame(i, (i % 20) / 20, i));
    }
    const ed = cine2.findNearestPhaseOffset(0);
    const es = cine2.findNearestPhaseOffset(0.35);
    assert.ok(ed !== null && ed.offset >= -19, `ED snap should prefer newest beat, got offset=${ed?.offset}`);
    assert.ok(ed!.phase <= 0.05 || ed!.phase >= 0.95, `ED phase should be ~0, got ${ed?.phase}`);
    assert.ok(es !== null && es.offset >= -19, `ES snap should prefer newest beat, got offset=${es?.offset}`);
    assert.ok(Math.abs(es!.phase - 0.35) <= 0.051, `ES phase should be ~0.35, got ${es?.phase}`);
  });

  it('reuses console persistence buffer in-place and maintains monotonic gray-map LUTs', () => {
    const spec = {
      lines: 16,
      samples: 32,
      elevationSamples: 1,
      sectorRad: 1.2,
      depthCm: 16,
      focusCm: 8,
    };
    const frame = allocPolarFrame(spec);
    for (let i = 0; i < frame.amplitude.length; i++) {
      frame.amplitude[i] = 0.005 + (i / frame.amplitude.length) * 0.9;
    }
    const state = createConsoleState(42);
    const out = new Uint8ClampedArray(spec.lines * spec.samples);

    applyConsole(frame, { ...DEFAULT_ACQUISITION, persistence: 0, sri: 0, edgeEnhance: 0, grayMap: 'clinical' }, state, out);
    const prevRef = state.prev;
    assert.ok(prevRef instanceof Float32Array);

    applyConsole(frame, { ...DEFAULT_ACQUISITION, persistence: 0, sri: 0, edgeEnhance: 0, grayMap: 'high-contrast' }, state, out);
    assert.equal(state.prev, prevRef, 'applyConsole should reuse state.prev in-place without reallocating');

    for (const gm of ['clinical', 's-curve', 'high-contrast', 'linear'] as const) {
      applyConsole(frame, { ...DEFAULT_ACQUISITION, persistence: 0, sri: 0, edgeEnhance: 0, grayMap: gm }, state, out);
      assert.ok(out[out.length - 1]! >= out[0]!, `Gray map ${gm} should preserve high > low amplitude`);
    }
  });
});



