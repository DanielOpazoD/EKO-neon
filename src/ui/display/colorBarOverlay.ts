import type { ColorSettings } from '@/simulator/doppler/color/colorDoppler';
import type { ImagingModality } from '@/simulator/renderer/types';
import { colorMap } from '@/simulator/renderer/postprocess/colorMap';
import type { SectorMapping } from '@/simulator/renderer/scanConvert';
import { canvasFont } from '../canvasFonts';

export interface ColorBarGeometry {
  bx: number;
  by: number;
  bw: number;
  velW: number;
  bh: number;
  scale: number;
  baseline: number;
  vTop: number;
  vBot: number;
  yZero: number;
  showZeroLabel: boolean;
  topLabel: string;
  botLabel: string;
  topDirLabel: 'hacia' | 'desde';
  botDirLabel: 'hacia' | 'desde';
}

export function formatColorBarVelocity(v: number): string {
  return `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`;
}

/** Pure layout & scale computation for the Color Doppler / CMM velocity legend bar. */
export function computeColorBarGeometry(
  modality: ImagingModality,
  color: ColorSettings,
  sector: SectorMapping,
  width: number,
  sectorH: number,
  rightX: number,
  inward: boolean,
): ColorBarGeometry {
  const scale = Math.max(0.01, color.scaleMps);
  const baseline = color.baselineShiftMps;
  const vTop = baseline + scale;
  const vBot = baseline - scale;
  const bw = color.showVariance ? 12 : 9;
  const velW = color.showVariance ? 8 : 9;
  const bx = (inward ? rightX - 30 : width - 18) - bw;
  const bh = modality === 'cmm' ? Math.min(92, Math.max(58, sectorH - 180)) : 104;
  const by =
    modality === 'cmm'
      ? Math.max(sector.apexY + 26, Math.min(sectorH - bh - 18, Math.max(154, sectorH - bh - 26)))
      : Math.max(sector.apexY + 42, 128);
  const zeroFrac = Math.max(0, Math.min(1, vTop / (2 * scale)));
  const yZero = by + zeroFrac * bh;
  const showZeroLabel = Math.abs(baseline) >= 0.03 && yZero > by + 32 && yZero < by + bh - 32;

  return {
    bx,
    by,
    bw,
    velW,
    bh,
    scale,
    baseline,
    vTop,
    vBot,
    yZero,
    showZeroLabel,
    topLabel: formatColorBarVelocity(vTop),
    botLabel: formatColorBarVelocity(vBot),
    topDirLabel: color.invert ? 'desde' : 'hacia',
    botDirLabel: color.invert ? 'hacia' : 'desde',
  };
}

/** Draw the physical Color Doppler / CMM velocity & variance legend bar directly from `colorMap`. */
export function drawColorBar(
  ctx: CanvasRenderingContext2D,
  modality: ImagingModality,
  color: ColorSettings,
  sector: SectorMapping,
  width: number,
  sectorH: number,
  rightX: number,
  inward: boolean,
): void {
  const g = computeColorBarGeometry(modality, color, sector, width, sectorH, rightX, inward);
  const rgb: [number, number, number] = [0, 0, 0];

  for (let py = 0; py < g.bh; py++) {
    const frac = (py + 0.5) / g.bh;
    const v = g.vTop - frac * 2 * g.scale;
    if (Math.abs(v) < 0.025 * g.scale) {
      ctx.fillStyle = '#05070a';
      ctx.fillRect(g.bx, g.by + py, g.bw, 1);
    } else {
      colorMap(v, g.scale, 0, false, rgb, g.baseline, color.palette);
      ctx.fillStyle = `rgb(${Math.round(rgb[0])},${Math.round(rgb[1])},${Math.round(rgb[2])})`;
      ctx.fillRect(g.bx, g.by + py, g.velW, 1);
      if (color.showVariance) {
        colorMap(v, g.scale, 0.85, true, rgb, g.baseline, color.palette);
        ctx.fillStyle = `rgb(${Math.round(rgb[0])},${Math.round(rgb[1])},${Math.round(rgb[2])})`;
        ctx.fillRect(g.bx + g.velW, g.by + py, g.bw - g.velW, 1);
      }
    }
  }

  ctx.strokeStyle = 'rgba(185, 195, 208, 0.35)';
  ctx.strokeRect(g.bx - 0.5, g.by - 0.5, g.bw + 1, g.bh + 1);
  ctx.strokeStyle = '#d7e0ea';
  ctx.beginPath();
  ctx.moveTo(g.bx - 3, g.yZero);
  ctx.lineTo(g.bx + g.bw + 2, g.yZero);
  ctx.stroke();

  const prevBase = ctx.textBaseline;
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#cfd8e4';
  ctx.font = canvasFont(10, 'mono');
  ctx.textAlign = 'right';
  ctx.fillText(g.topLabel, g.bx - 6, g.by + 5);
  ctx.fillText(g.botLabel, g.bx - 6, g.by + g.bh - 5);
  if (g.showZeroLabel) {
    ctx.fillStyle = '#9aa8ba';
    ctx.fillText('0', g.bx - 6, g.yZero);
  }
  ctx.fillStyle = '#7f8a9a';
  ctx.font = canvasFont(9.5);
  ctx.fillText(g.topDirLabel, g.bx - 6, g.by + 20);
  ctx.fillText(g.botDirLabel, g.bx - 6, g.by + g.bh - 20);
  ctx.fillText('m/s', g.bx + g.bw, g.by + g.bh + 12);
  if (color.showVariance) {
    ctx.fillStyle = '#7ce0a8';
    ctx.font = canvasFont(8.5, 'mono');
    ctx.textAlign = 'center';
    ctx.fillText('V', g.bx + g.velW + (g.bw - g.velW) / 2, g.by - 7);
  }
  ctx.textBaseline = prevBase;
  ctx.textAlign = 'left';
  ctx.font = canvasFont(11);
}
