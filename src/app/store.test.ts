import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { parsePrefs, pickPrefs, PERSISTED_PREF_KEYS } from './prefs';
import {
  START_PROBE,
  imageSegmentsOn,
  navigatorLayers,
  segmentLayerOn,
  useHudStore,
  useSimStore,
} from './store';
import { DEFAULT_ACQUISITION } from '@/simulator/renderer/types';
import { SPECTRAL_MODE_DEFAULTS } from '@/simulator/doppler/spectral/spectrum';
import type { SimOutput } from '@/simulator/core/protocol';

describe('UI preferences schema (prefs.ts)', () => {
  it('accepts valid keys, strips unknown keys, and drops wrong-typed entries without throwing', () => {
    assert.deepEqual(parsePrefs(null), {});
    assert.deepEqual(parsePrefs('corrupt'), {});
    assert.deepEqual(parsePrefs([1, 2, 3]), {});

    const parsed = parsePrefs({
      showTorso: false,
      consoleTab: 'doppler',
      segmentModel: 'LV_16',
      roomMode: true,
      // wrong types -> must be dropped individually
      showHud: 'yes',
      consoleTabInvalid: 'nope',
      navSkin: 123,
      // unknown / legacy keys -> must be dropped
      legacyFlag: true,
      guidanceOpen: true, // explicitly non-persisted
    });

    assert.equal(parsed.showTorso, false);
    assert.equal(parsed.consoleTab, 'doppler');
    assert.equal(parsed.segmentModel, 'LV_16');
    assert.equal(parsed.roomMode, true);
    assert.equal('showHud' in parsed, false);
    assert.equal('navSkin' in parsed, false);
    assert.equal('legacyFlag' in parsed, false);
    assert.equal('guidanceOpen' in parsed, false);
  });

  it('pickPrefs extracts only the persisted key set from UiPrefs', () => {
    const ui = useSimStore.getState().ui;
    const picked = pickPrefs(ui);
    assert.deepEqual(Object.keys(picked).sort(), [...PERSISTED_PREF_KEYS].sort());
  });
});

describe('SimStore clamping, cine controls, modality memory and exam policy', () => {
  beforeEach(() => {
    // Reset to clean sandbox defaults before every test
    const s = useSimStore.getState();
    s.setMode('sandbox');
    s.loadCase('normal-excellent-window');
    s.setModality('2d');
    s.setSettings({ ...DEFAULT_ACQUISITION, tgcDb: [...DEFAULT_ACQUISITION.tgcDb] });
    s.setUi({
      imageSegments: false,
      navSegments: false,
      devPanel: false,
      showPhysics: false,
      reviewMode: false,
    });
    useHudStore.getState().setHud(null);
  });

  it('clamps acquisition settings including SRI, spatialCompound, TGC and focus <= depth', () => {
    const s = useSimStore.getState();
    s.setSettings({
      depthCm: 100,
      sectorDeg: 10,
      gainDb: 99,
      dynamicRangeDb: 5,
      frequencyMHz: 9,
      focusCm: 50,
      persistence: 2,
      sri: 1.8,
      spatialCompound: -0.5,
      zoom: 10,
    });
    let st = useSimStore.getState().settings;
    assert.equal(st.depthCm, 30);
    assert.equal(st.sectorDeg, 30);
    assert.equal(st.gainDb, 30);
    assert.equal(st.dynamicRangeDb, 30);
    assert.equal(st.frequencyMHz, 5);
    assert.equal(st.focusCm, 30);
    assert.equal(st.persistence, 0.9);
    assert.equal(st.sri, 1);
    assert.equal(st.spatialCompound, 0);
    assert.equal(st.zoom, 2.5);

    // Reducing depth pulls focus inward
    useSimStore.getState().setSettings({ depthCm: 10 });
    st = useSimStore.getState().settings;
    assert.equal(st.depthCm, 10);
    assert.ok(st.focusCm <= 10);

    // TGC per-band clamp & reset
    useSimStore.getState().setTgc(2, 40);
    useSimStore.getState().setTgc(5, -40);
    assert.equal(useSimStore.getState().settings.tgcDb[2], 15);
    assert.equal(useSimStore.getState().settings.tgcDb[5], -15);
    useSimStore.getState().resetTgc();
    assert.deepEqual(useSimStore.getState().settings.tgcDb, [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('clamps probe position/angles and wraps rotationDeg into [-180, 180)', () => {
    useSimStore.getState().setProbe({
      u: 50,
      v: -50,
      rotationDeg: 270,
      tiltDeg: 120,
      rockDeg: -120,
      pressure: 3,
    });
    const p = useSimStore.getState().probe;
    assert.equal(p.u, 14);
    assert.equal(p.v, -12);
    assert.equal(p.rotationDeg, -90);
    assert.equal(p.tiltDeg, 70);
    assert.equal(p.rockDeg, -60);
    assert.equal(p.pressure, 1);

    useSimStore.getState().resetProbe();
    assert.deepEqual(useSimStore.getState().probe, START_PROBE);
  });

  it('handles freeze, cine offset bounds, stepCine wrap-around and playback speed', () => {
    useHudStore.getState().setHud({ cineLength: 10 } as SimOutput);
    useSimStore.getState().toggleFreeze();
    assert.equal(useSimStore.getState().frozen, true);
    assert.equal(useSimStore.getState().cineOffset, 0);

    useSimStore.getState().setCineOffset(-99);
    assert.equal(useSimStore.getState().cineOffset, -9);

    useSimStore.getState().setCineOffset(5);
    assert.equal(useSimStore.getState().cineOffset, 0);

    // Stepping at newest (0) wraps to oldest (-(length - 1) = -9), then increments toward 0
    useSimStore.getState().stepCine();
    assert.equal(useSimStore.getState().cineOffset, -9);
    useSimStore.getState().stepCine();
    assert.equal(useSimStore.getState().cineOffset, -8);

    useSimStore.getState().setCineSpeed(0.25);
    assert.equal(useSimStore.getState().cineSpeed, 0.25);

    // Unfreezing resets offset and playback
    useSimStore.getState().setCinePlaying(true);
    useSimStore.getState().toggleFreeze();
    assert.equal(useSimStore.getState().frozen, false);
    assert.equal(useSimStore.getState().cineOffset, 0);
    assert.equal(useSimStore.getState().cinePlaying, false);
  });

  it('preserves independent spectral velocity scales across PW, CW and TDI switches and enforces PW Nyquist limit', () => {
    useSimStore.getState().setModality('pw');
    // Requesting above physical PW Nyquist at gateDepthCm=9 / 2.5 MHz is physically constrained
    useSimStore.getState().setSpectral({ scaleMps: 2.5 });
    const nyqCapped = useSimStore.getState().spectral.scaleMps;
    assert.ok(nyqCapped > 0.8 && nyqCapped < 1.3, `Expected Nyquist-capped PW scale, got ${nyqCapped}`);

    // A valid sub-Nyquist PW scale + baseline shift is remembered independently of CW
    useSimStore.getState().setSpectral({ scaleMps: 0.9, baselineShiftMps: -0.2 });
    useSimStore.getState().setModality('cw');
    assert.equal(useSimStore.getState().spectral.scaleMps, SPECTRAL_MODE_DEFAULTS.cw.scaleMps);

    useSimStore.getState().setSpectral({ scaleMps: 5.5 });
    useSimStore.getState().setModality('pw');
    assert.equal(useSimStore.getState().spectral.scaleMps, 0.9);
    assert.equal(useSimStore.getState().spectral.baselineShiftMps, -0.2);

    useSimStore.getState().setModality('cw');
    assert.equal(useSimStore.getState().spectral.scaleMps, 5.5);
  });

  it('enforces exam mode policy: resets session and hides hints, segments, dev tools and heart anatomy', () => {
    useSimStore.getState().setUi({
      imageSegments: true,
      navSegments: true,
      devPanel: true,
      showPhysics: true,
      reviewMode: true,
    });
    useSimStore.getState().setProbe({ u: 8, v: -4 });
    useSimStore.getState().setArtifactLab({ sideLobe: 0.5, mirror: 0.2, beamWidth: 0.1, clutter: 0.1 });

    // In sandbox, overlays are enabled
    assert.equal(imageSegmentsOn(useSimStore.getState()), true);
    assert.equal(segmentLayerOn(useSimStore.getState()), true);
    assert.equal(navigatorLayers(useSimStore.getState()).heart, true);

    // Enter exam mode with fixed case
    useSimStore.getState().setMode('exam', 'aortic-stenosis-severe');
    const exam = useSimStore.getState();
    assert.equal(exam.mode, 'exam');
    assert.equal(exam.caseId, 'aortic-stenosis-severe');
    assert.deepEqual(exam.probe, START_PROBE);
    assert.equal(exam.artifactLab, null);
    assert.equal(exam.ui.devPanel, false);
    assert.equal(exam.ui.showPhysics, false);
    assert.equal(exam.ui.reviewMode, false);
    assert.equal(exam.ui.showHints, false);
    assert.equal(imageSegmentsOn(exam), false);
    assert.equal(segmentLayerOn(exam), false);
    const nav = navigatorLayers(exam);
    assert.equal(nav.skin, true);
    assert.equal(nav.skeleton, true);
    assert.equal(nav.heart, false);
    assert.equal(nav.cut, false);
  });
});
