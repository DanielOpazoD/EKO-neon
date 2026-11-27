import assert from 'node:assert/strict';
import { beforeEach, describe, it } from 'node:test';
import { Structure } from '@/simulator/anatomy/tissue';
import { computeStructureStats, placeLabels } from '@/ui/cutMap';
import { IMAGE_ANATOMY_LABELS } from '@/ui/display/rasterLayers';
import { parsePrefs, pickPrefs } from './prefs';
import {
  imageAnatomyLabelsOn,
  isClassModeOn,
  nextClassModeUi,
  useSimStore,
} from './store';

const get = () => useSimStore.getState();

describe('live teaching ergonomics (PR A)', () => {
  beforeEach(() => {
    get().setMode('sandbox');
    get().setUi({ imageAnatomyLabels: false, minimal: false, roomMode: false });
  });

  it('keeps 2D anatomy labels strictly disabled in exam mode even if enabled in prefs', () => {
    get().setUi({ imageAnatomyLabels: true });
    assert.equal(imageAnatomyLabelsOn(get()), true);

    get().setMode('exam', 'aortic-stenosis-severe');
    assert.equal(imageAnatomyLabelsOn(get()), false);

    get().setMode('guided');
    assert.equal(imageAnatomyLabelsOn(get()), true);
  });

  it('persists imageAnatomyLabels cleanly through prefs schema validation', () => {
    const parsed = parsePrefs({ imageAnatomyLabels: true, minimal: true, unknown: 42 });
    assert.equal(parsed.imageAnatomyLabels, true);
    assert.equal(parsePrefs({ imageAnatomyLabels: 'yes' }).imageAnatomyLabels, undefined);
    const picked = pickPrefs({ ...get().ui, imageAnatomyLabels: true });
    assert.equal(picked.imageAnatomyLabels, true);
  });

  it('toggles 1-click classroom / projector mode cleanly on and off', () => {
    assert.equal(isClassModeOn({ minimal: false, roomMode: false }), false);
    assert.equal(isClassModeOn({ minimal: true, roomMode: false }), false);
    assert.equal(isClassModeOn({ minimal: true, roomMode: true }), true);

    const onPatch = nextClassModeUi({ minimal: false, roomMode: false });
    assert.deepEqual(onPatch, { minimal: true, roomMode: true });

    const offPatch = nextClassModeUi({ minimal: true, roomMode: true });
    assert.deepEqual(offPatch, { minimal: false, roomMode: false });
  });

  it('computes structure centroids directly on a sector LUT and places collision-free 2D labels', () => {
    const W = 120;
    const H = 90;
    const lut = new Int32Array(W * H).fill(-1);
    const structure = new Uint8Array(W * H);
    // Left region: LV cavity, right region: LA cavity, bottom strip: descending aorta
    for (let y = 8; y < 44; y++) {
      for (let x = 8; x < 52; x++) {
        const idx = y * W + x;
        lut[idx] = idx;
        structure[idx] = Structure.LvCavity;
      }
      for (let x = 68; x < 112; x++) {
        const idx = y * W + x;
        lut[idx] = idx;
        structure[idx] = Structure.LaCavity;
      }
    }
    for (let y = 56; y < 84; y++) {
      for (let x = 40; x < 80; x++) {
        const idx = y * W + x;
        lut[idx] = idx;
        structure[idx] = Structure.DescendingAorta;
      }
    }
    const stats = computeStructureStats(lut, structure, W);
    assert.equal(stats.length, 3);
    const placed = placeLabels(
      stats,
      lut,
      structure,
      W,
      100,
      (t) => ({ w: t.length * 7 + 8, h: 16 }),
      new Set(),
      (id) => IMAGE_ANATOMY_LABELS[id],
    );
    const texts = placed.map((l) => l.text).sort();
    assert.deepEqual(texts, ['AI', 'Ao desc', 'VI']);
  });
});
