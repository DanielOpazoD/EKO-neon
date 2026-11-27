import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_COLOR } from '@/simulator/doppler/color/colorDoppler';
import type { SectorMapping } from '@/simulator/renderer/scanConvert';
import { computeColorBarGeometry, formatColorBarVelocity } from './colorBarOverlay';

const SECTOR: SectorMapping = {
  apexX: 320,
  apexY: 20,
  pxPerCm: 26,
  depthCm: 16,
  sectorRad: Math.PI / 2,
  invertLR: false,
  width: 640,
  height: 480,
};

test('computeColorBarGeometry reflects baseline shift, invert, variance width, and CMM height clamping', () => {
  const g0 = computeColorBarGeometry('color', DEFAULT_COLOR, SECTOR, 640, 480, 620, false);
  assert.equal(g0.bw, 12);
  assert.equal(g0.velW, 8);
  assert.equal(g0.topLabel, '+0.68');
  assert.equal(g0.botLabel, '−0.68');
  assert.equal(g0.topDirLabel, 'hacia');
  assert.equal(g0.botDirLabel, 'desde');
  assert.equal(g0.showZeroLabel, false);
  assert.ok(Math.abs(g0.yZero - (g0.by + g0.bh / 2)) < 1e-6);

  const shifted = {
    ...DEFAULT_COLOR,
    baselineShiftMps: 0.25,
    invert: true,
    showVariance: false,
  };
  const g1 = computeColorBarGeometry('color', shifted, SECTOR, 640, 480, 620, false);
  assert.equal(g1.bw, 9);
  assert.equal(g1.velW, 9);
  assert.equal(g1.topLabel, '+0.93');
  assert.equal(g1.botLabel, '−0.43');
  assert.equal(g1.topDirLabel, 'desde');
  assert.equal(g1.botDirLabel, 'hacia');
  assert.equal(g1.showZeroLabel, true);
  assert.ok(g1.yZero > g1.by + g1.bh / 2, 'positive baseline shift pushes zero line downward');

  const gCmm = computeColorBarGeometry('cmm', DEFAULT_COLOR, SECTOR, 640, 202, 620, false);
  assert.ok(gCmm.bh >= 58 && gCmm.bh <= 92);
  assert.equal(formatColorBarVelocity(0), '+0.00');
  assert.equal(formatColorBarVelocity(-0.43), '−0.43');
});

test('computeColorBarGeometry keeps compact CMM color bar clear of top-right HUD and inside sector', () => {
  const hudBottom = 104;
  // Compact CMM sector (H=440 -> sectorH=185) with 4-line HUD ending at y=104
  const compact = computeColorBarGeometry('cmm', DEFAULT_COLOR, SECTOR, 640, 185, 620, false, hudBottom);
  assert.ok(
    compact.by - 12 >= hudBottom,
    `Compact CMM bar top+V label (${compact.by - 12}) overlapped HUD bottom (${hudBottom})`,
  );
  assert.ok(
    compact.by + compact.bh + 14 <= 185,
    `Compact CMM bar m/s label (${compact.by + compact.bh + 14}) spilled below sectorH (185)`,
  );
  assert.ok(compact.bh >= 40, `Expected readable compact CMM bar height >= 40, got ${compact.bh}`);

  // Standard Color 2D with tall HUD notice still stays below HUD and inside sector
  const tallHud = computeColorBarGeometry('color', DEFAULT_COLOR, SECTOR, 640, 480, 620, false, 136);
  assert.ok(tallHud.by - 12 >= 136, `Color bar overlapped tall HUD (${tallHud.by - 12} < 136)`);
  assert.ok(tallHud.by + tallHud.bh + 18 <= 480);
});
