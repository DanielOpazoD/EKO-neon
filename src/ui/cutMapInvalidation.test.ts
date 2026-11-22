import test from 'node:test';
import assert from 'node:assert/strict';
import { useSimStore } from '@/app/store';
import { cutMapStoreChanged } from './cutMapInvalidation';

test('cutMapStoreChanged ignores unrelated store updates (gain, dynamic range, navSkin, fpsUi)', () => {
  const prev = useSimStore.getState();
  assert.equal(cutMapStoreChanged(prev, prev), false);

  const nextGain = {
    ...prev,
    fpsUi: prev.fpsUi + 1,
    settings: { ...prev.settings, gainDb: prev.settings.gainDb + 6, dynamicRangeDb: 55 },
  };
  assert.equal(cutMapStoreChanged(prev, nextGain), false);

  const nextNavSkin = {
    ...prev,
    ui: { ...prev.ui, navSkin: !prev.ui.navSkin, navHeart: !prev.ui.navHeart },
  };
  assert.equal(cutMapStoreChanged(prev, nextNavSkin), false);
});

test('cutMapStoreChanged triggers on every cut-map visual dependency', () => {
  const prev = useSimStore.getState();

  assert.equal(cutMapStoreChanged(prev, { ...prev, caseId: 'hfref' }), true);
  assert.equal(cutMapStoreChanged(prev, { ...prev, mode: 'exam' }), true);
  assert.equal(
    cutMapStoreChanged(prev, { ...prev, probe: { ...prev.probe, u: prev.probe.u + 1 } }),
    true,
  );
  assert.equal(
    cutMapStoreChanged(prev, {
      ...prev,
      settings: { ...prev.settings, invertLR: !prev.settings.invertLR },
    }),
    true,
  );
  assert.equal(
    cutMapStoreChanged(prev, {
      ...prev,
      settings: { ...prev.settings, zoom: 1.5 },
    }),
    true,
  );
  assert.equal(
    cutMapStoreChanged(prev, {
      ...prev,
      ui: { ...prev.ui, navLabels: !prev.ui.navLabels },
    }),
    true,
  );
  assert.equal(
    cutMapStoreChanged(prev, {
      ...prev,
      ui: { ...prev.ui, navSegments: !prev.ui.navSegments },
    }),
    true,
  );
  assert.equal(
    cutMapStoreChanged(prev, {
      ...prev,
      ui: {
        ...prev.ui,
        segmentModel: prev.ui.segmentModel === 'LV_AHA17' ? 'LV_16' : 'LV_AHA17',
      },
    }),
    true,
  );
  assert.equal(
    cutMapStoreChanged(prev, {
      ...prev,
      ui: { ...prev.ui, selectedSegment: 4 },
    }),
    true,
  );
});
