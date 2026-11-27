import { beforeEach, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { useSimStore } from './store';
import { parseReviewReport } from './review';
import { buildInput } from './useSimulation';
import { acquisitionLimits } from '@/simulator/core/acquisitionInput';
import { baseInput } from '@/simulator/core/baseInput';
import { colorBoxForView, DEFAULT_COLOR } from '@/simulator/doppler/color/colorDoppler';
import {
  colorWallFilterCutoffMps,
  WF_MAX_NYQUIST,
  WF_MIN_NYQUIST,
} from '@/simulator/doppler/color/kasaiEstimator';
import { DEFAULT_ACQUISITION } from '@/simulator/renderer/types';

const get = () => useSimStore.getState();
/** The cutoff that filters with the store as it stands: the request bounded at the acquired Nyquist. */
const effective = () => colorWallFilterCutoffMps(get().color.wallFilterMps, acquisitionLimits(get()).color.scaleMps);
const near = (a: number, b: number, label: string) =>
  assert.ok(Math.abs(a - b) < 1e-12, `${label}: ${a} instead of ${b}`);

/**
 * The colour wall filter through the store's own actions (setColor, setSettings, setModality, loadCase, setMode,
 * loadReviewReport). The store keeps what the learner asked for; Nyquist bounds what filters. Bounded in the store
 * (constrainAcquisition), a scale lowered once lowered the filter for the rest of the session: the curriculum's
 * colour-low-scale (0.35 m/s) then colour-normal-scale (0.68 m/s) left the 20 cm/s default at 11.2 cm/s, and the
 * moving myocardium flashed in the very task that asks for no colour over tissue.
 */
describe('the colour wall filter the learner asks for survives the changes of Nyquist', () => {
  beforeEach(() => {
    get().setMode('sandbox');
    get().loadCase('normal-excellent-window');
    get().setModality('2d');
    get().setSettings({ ...DEFAULT_ACQUISITION, tgcDb: [...DEFAULT_ACQUISITION.tgcDb] });
    get().setColor({ ...DEFAULT_COLOR });
  });

  it('0.68 → 0.35 → 0.68 m/s gives the default 20 cm/s back (colour-low-scale, then colour-normal-scale)', () => {
    get().setModality('color');
    assert.equal(effective(), DEFAULT_COLOR.wallFilterMps);
    get().setColor({ scaleMps: 0.35 });
    assert.equal(get().color.wallFilterMps, DEFAULT_COLOR.wallFilterMps, 'the store keeps the request');
    near(effective(), WF_MAX_NYQUIST * 0.35, 'at 0.35 m/s it filters at 0.32 Nyquist, the most 8 pulses give');
    // what the simulator receives is the request too: the core bounds it at its own Nyquist
    assert.equal(buildInput({ width: 320, height: 240 }).color.wallFilterMps, DEFAULT_COLOR.wallFilterMps);
    get().setColor({ scaleMps: 0.68 });
    assert.equal(get().color.wallFilterMps, DEFAULT_COLOR.wallFilterMps);
    assert.equal(effective(), DEFAULT_COLOR.wallFilterMps);
  });

  it('the other way: 4 cm/s chosen at 0.30 m/s, through 0.68 and back to 0.30, is 4 cm/s again', () => {
    get().setModality('color');
    get().setColor({ scaleMps: 0.3 });
    get().setColor({ wallFilterMps: 0.04 });
    assert.equal(effective(), 0.04);
    get().setColor({ scaleMps: 0.68 });
    assert.equal(get().color.wallFilterMps, 0.04);
    near(effective(), WF_MIN_NYQUIST * 0.68, 'at 0.68 m/s the mean removal, the lowest that cancels still tissue');
    get().setColor({ scaleMps: 0.3 });
    assert.equal(effective(), 0.04);
  });

  it('a deep box, a higher frequency and colour M-mode at depth lower the cutoff, not the request: raising the scale gives it back', () => {
    // They lower Nyquist, and the store bounds the colour scale to it (constrainAcquisition, as in main): back at a
    // shallow box, the default frequency or 16 cm, the scale stays at the PRF's limit until the learner raises it, and
    // the cutoff that filters follows it (16.2 cm/s after the 22 cm box, 18.7 after 3.5 MHz). That ratchet is the
    // colour scale's, main's, and stays as it is. The wall filter keeps the request: raising the scale gives the
    // 20 cm/s back, where the request bounded in the store stayed at 16.2
    get().setModality('color');
    get().setSettings({ depthCm: 24 });
    get().setColor({ boxRMaxCm: 22 });
    const deepScale = acquisitionLimits(get()).color.scaleMps;
    assert.ok(deepScale < 0.68, `a 22 cm box limits the PRF: ${deepScale}`);
    near(effective(), WF_MAX_NYQUIST * deepScale, '22 cm box');
    get().setColor({ boxRMaxCm: 13 });
    get().setSettings({ depthCm: 16 });
    assert.equal(get().color.wallFilterMps, DEFAULT_COLOR.wallFilterMps, 'the request survives the deep box');
    get().setColor({ scaleMps: 0.68 });
    assert.equal(effective(), DEFAULT_COLOR.wallFilterMps);

    get().setSettings({ frequencyMHz: 3.5 });
    assert.ok(effective() < DEFAULT_COLOR.wallFilterMps, `3.5 MHz lowers Nyquist: ${effective()}`);
    get().setSettings({ frequencyMHz: DEFAULT_ACQUISITION.frequencyMHz });
    assert.equal(get().color.wallFilterMps, DEFAULT_COLOR.wallFilterMps, 'and 3.5 MHz');
    get().setColor({ scaleMps: 0.68 });
    assert.equal(effective(), DEFAULT_COLOR.wallFilterMps);

    get().setModality('cmm');
    get().setSettings({ depthCm: 20 });
    assert.ok(effective() < DEFAULT_COLOR.wallFilterMps, `colour M-mode at 20 cm: ${effective()}`);
    get().setSettings({ depthCm: 16 });
    get().setModality('color');
    assert.equal(get().color.wallFilterMps, DEFAULT_COLOR.wallFilterMps, 'and colour M-mode at 20 cm');
    get().setColor({ scaleMps: 0.68 });
    assert.equal(effective(), DEFAULT_COLOR.wallFilterMps);
  });

  it('a case load and the exam change none of it', () => {
    get().setModality('color');
    get().setColor({ scaleMps: 0.35 });
    get().loadCase('af-diastolic');
    assert.equal(get().color.wallFilterMps, DEFAULT_COLOR.wallFilterMps);
    get().setMode('exam', 'normal-excellent-window');
    get().setModality('color');
    assert.equal(get().color.wallFilterMps, DEFAULT_COLOR.wallFilterMps);
    near(effective(), WF_MAX_NYQUIST * 0.35, 'exam at 0.35 m/s');
    get().setColor({ scaleMps: 0.68 });
    assert.equal(effective(), DEFAULT_COLOR.wallFilterMps);
    get().setMode('sandbox');
    get().loadCase('normal-excellent-window');
    get().setColor({ scaleMps: 0.3, wallFilterMps: 0.04 });
    get().loadCase('hocm-sam');
    get().setColor({ scaleMps: 0.68 });
    get().setMode('exam', 'normal-excellent-window');
    get().setModality('color');
    get().setColor({ scaleMps: 0.3 });
    assert.equal(effective(), 0.04);
  });

  it('a version 1 review report loads with the filter its image had, and keeps it as the request', () => {
    // main's default label at 0.68 m/s, whose ramps cut at 0.2046 m/s; loaded as a request it was 7.6 cm/s
    const v1 = {
      format: 'echotwin-review',
      version: 1,
      createdAt: '2026-11-20T10:00:00.000Z',
      author: 'usuario',
      app: { url: '' },
      caseId: 'normal-excellent-window',
      caseTitle: 'Normal',
      mode: 'sandbox',
      workerMode: 'worker',
      input: baseInput({ modality: 'color', color: { ...DEFAULT_COLOR, wallFilterMps: 0.06 } }),
      frame: null,
      note: '',
      markers: [],
    };
    const r = parseReviewReport(JSON.stringify(v1));
    assert.ok(r);
    get().loadReviewReport(r);
    assert.equal(get().modality, 'color');
    assert.equal(get().color.wallFilterMps, r.input.color.wallFilterMps);
    assert.ok(Math.abs(effective() - 0.2046) < 1e-4, `the replay filters at ${effective()}`);
    get().setColor({ scaleMps: 0.35 });
    get().setColor({ scaleMps: 0.68 });
    assert.ok(Math.abs(effective() - 0.2046) < 1e-4, 'and keeps it through a change of scale');
  });

  it('a version 1 report made at 0.35 m/s filters as main did there, and as main did at 0.68 once the scale is back', () => {
    // At 0.35 m/s main's ramps saturate: the label 0.06 cut at the steepest filter, 11.2 cm/s. Asked as 11.2, the
    // replay went on filtering at 11.2 when the learner raised the scale to 0.68 (A4C, CPU, 44 frames: 7073 coloured
    // myocardial samples, 70 in main). Every request from 11.2 up filters so at 0.35: it asks for main's 20.5 at 0.68
    const v1 = {
      format: 'echotwin-review',
      version: 1,
      createdAt: '2026-11-20T10:00:00.000Z',
      author: 'usuario',
      app: { url: '' },
      caseId: 'normal-excellent-window',
      caseTitle: 'Normal',
      mode: 'sandbox',
      workerMode: 'worker',
      input: baseInput({ modality: 'color', color: { ...DEFAULT_COLOR, scaleMps: 0.35, wallFilterMps: 0.06 } }),
      frame: null,
      note: '',
      markers: [],
    };
    get().loadReviewReport(parseReviewReport(JSON.stringify(v1))!);
    assert.equal(acquisitionLimits(get()).color.scaleMps, 0.35);
    near(effective(), WF_MAX_NYQUIST * 0.35, 'at 0.35 m/s the steepest filter, as in main');
    get().setColor({ scaleMps: 0.68 });
    assert.ok(Math.abs(effective() - 0.2046) < 1e-4, `back at 0.68 the replay filters at ${effective()}`);
  });

  it('a version 1 colour M-mode report leaves the 2D colour after it on the filter main gave it', () => {
    // color.wallFilterMps is one for the strip and the 2D colour. The report kept as its label 0.06 asked for less than
    // the packet allows: the 2D colour after it filtered at the mean removal, 7.5 cm/s, for the rest of the session
    // (A4C, CPU, 44 frames: 9339 coloured myocardial samples). Migrated like any version 1, it is main's ramps' cutoff
    // at the strip's 0.676 m/s, 20.4 cm/s, where main's 2D colour filtered after it (74 samples, in main as here)
    const v1 = {
      format: 'echotwin-review',
      version: 1,
      createdAt: '2026-11-20T10:00:00.000Z',
      author: 'usuario',
      app: { url: '' },
      caseId: 'normal-excellent-window',
      caseTitle: 'Normal',
      mode: 'sandbox',
      workerMode: 'worker',
      input: baseInput({ modality: 'cmm', color: { ...DEFAULT_COLOR, wallFilterMps: 0.06 } }),
      frame: null,
      note: '',
      markers: [],
    };
    get().loadReviewReport(parseReviewReport(JSON.stringify(v1))!);
    assert.equal(get().modality, 'cmm');
    assert.ok(Math.abs(effective() - 0.204) < 1e-4, `the strip filters at ${effective()}`);
    get().setModality('color');
    assert.ok(Math.abs(effective() - 0.204) < 1e-4, `the 2D colour after it filters at ${effective()}`);
    get().loadCase('af-diastolic');
    get().setModality('color');
    assert.ok(Math.abs(effective() - 0.204) < 1e-4, `and after another case at ${effective()}`);
  });

  it('frames the Color ROI box for the active view while respecting manual user placement and exam privacy', () => {
    // Fresh session clears custom box override: entering Color on a target view frames its anatomical ROI
    get().loadCase('normal-excellent-window');
    get().setTargetView('subcostal-ivc');
    get().setModality('color');
    const ivcBox = colorBoxForView('subcostal-ivc', 16, (80 * Math.PI) / 180);
    assert.equal(get().color.boxRMinCm, ivcBox.boxRMinCm);
    assert.equal(get().color.boxRMaxCm, ivcBox.boxRMaxCm);
    assert.equal(get().color.boxThetaMinRad, ivcBox.boxThetaMinRad);
    assert.equal(get().color.boxThetaMaxRad, ivcBox.boxThetaMaxRad);
    assert.equal(get().color.scaleMps, DEFAULT_COLOR.scaleMps);
    assert.equal(get().colorBoxCustom, false);

    // Manually resizing the box sets colorBoxCustom and survives toggling 2D -> Color
    get().setColor({ boxRMinCm: 3.0, boxRMaxCm: 9.0, boxThetaMinRad: -0.2, boxThetaMaxRad: 0.2 });
    assert.equal(get().colorBoxCustom, true);
    get().setModality('2d');
    get().setTargetView('a4c');
    get().setModality('color');
    assert.equal(get().color.boxRMaxCm, 9.0, 'Manual Color ROI must not be overwritten on modality toggle');

    // Explicit resetColorBox snaps back to the active view's ROI
    get().resetColorBox();
    const a4cBox = colorBoxForView('a4c', 16, (80 * Math.PI) / 180);
    assert.equal(get().color.boxRMaxCm, a4cBox.boxRMaxCm);
    assert.equal(get().colorBoxCustom, false);

    // Exam mode never auto-frames the Color box to a view
    get().setMode('exam', 'normal-excellent-window');
    get().setColor({ boxRMinCm: 4.5, boxRMaxCm: 10.5 });
    get().resetColorBox();
    assert.equal(get().color.boxRMaxCm, DEFAULT_COLOR.boxRMaxCm, 'Exam mode reset uses neutral DEFAULT_COLOR');
  });
});
