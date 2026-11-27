import { Structure } from '@/simulator/anatomy/tissue';
import type { PolarFrameSpec } from '@/simulator/renderer/types';

const AV_OUTLET_SET = new Set<number>([
  Structure.AorticValve,
  Structure.AorticRoot,
  Structure.AscendingAorta,
]);

/** Minimum LVOT polar area (cm²) required so stray edge pixels are rejected. */
const MIN_LVOT_AREA_CM2 = 0.35;
/** Minimum 2D alignment |cos θ| between ultrasound beam and LVOT outflow axis (<= ~46° in 2D slice; rejects PLAX). */
const MIN_BEAM_ALIGNMENT = 0.68;
/** ASE 2019 target distance proximal to the aortic annulus inside the LVOT on a mid-systolic frame (cm). */
const TARGET_DIST_SYSTOLE_CM = 0.50;
/** Extra apical offset when only a diastolic frame is available (~0.6 cm annular descent to mid-systole). */
const DIASTOLIC_DESCENT_COMP_CM = 0.60;
const MIN_DIST_PROXIMAL_CM = 0.25;
const MAX_DIST_PROXIMAL_CM = 1.30;

export type AutoLvotGateFailureReason =
  | 'no-frame'
  | 'lvot-not-visible'
  | 'misaligned-view';

export interface AutoLvotGateSuccess {
  ok: true;
  /** Polar sample-volume depth from transducer apex (cm). */
  rCm: number;
  /** Polar beam angle from sector centerline (rad). */
  thetaRad: number;
  /** Estimated 2D angle between the beam ray and the LVOT outflow axis (deg). */
  angle2dDeg: number;
  /** Distance from the selected gate to the nearest aortic valve/root boundary (cm). */
  distToAvCm: number;
  /** Visible LVOT cross-sectional area on the current 2D slice (cm²). */
  lvotAreaCm2: number;
}

export interface AutoLvotGateFailure {
  ok: false;
  reason: AutoLvotGateFailureReason;
  message: string;
}

export type AutoLvotGateResult = AutoLvotGateSuccess | AutoLvotGateFailure;

/**
 * Locate the optimal ASE PW Doppler gate position inside the LVOT (~0.5 cm proximal to the aortic
 * valve, mid-lumen) strictly from the rendered 2D polar `structure` map — zero cheating from 3D
 * anatomy anchors or ground truth. Rejects views without a visible LVOT (e.g. A4C, PSAX-PM) and
 * perpendicular views where the beam is not aligned with outflow (e.g. PLAX).
 */
export function findLvotGateFromPolar(
  structure: Uint8Array,
  spec: Pick<PolarFrameSpec, 'lines' | 'samples' | 'sectorRad' | 'depthCm'>,
  opts?: { systolicFrame?: boolean },
): AutoLvotGateResult {
  const { lines, samples, sectorRad, depthCm } = spec;
  const total = lines * samples;
  if (structure.length < total || lines < 8 || samples < 16) {
    return {
      ok: false,
      reason: 'no-frame',
      message: 'No hay un cuadro 2D disponible para localizar el TSVI.',
    };
  }

  // 1. Find the largest 4-connected component of Structure.Lvot on the polar grid.
  const visited = new Uint8Array(total);
  const queue = new Int32Array(total);
  let bestComp: number[] = [];
  for (let idx = 0; idx < total; idx++) {
    if (visited[idx] || structure[idx] !== Structure.Lvot) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = idx;
    visited[idx] = 1;
    const comp: number[] = [];
    while (head < tail) {
      const cur = queue[head++]!;
      comp.push(cur);
      const li = (cur / samples) | 0;
      const si = cur - li * samples;
      if (li > 0) {
        const n = cur - samples;
        if (!visited[n] && structure[n] === Structure.Lvot) {
          visited[n] = 1;
          queue[tail++] = n;
        }
      }
      if (li + 1 < lines) {
        const n = cur + samples;
        if (!visited[n] && structure[n] === Structure.Lvot) {
          visited[n] = 1;
          queue[tail++] = n;
        }
      }
      if (si > 0) {
        const n = cur - 1;
        if (!visited[n] && structure[n] === Structure.Lvot) {
          visited[n] = 1;
          queue[tail++] = n;
        }
      }
      if (si + 1 < samples) {
        const n = cur + 1;
        if (!visited[n] && structure[n] === Structure.Lvot) {
          visited[n] = 1;
          queue[tail++] = n;
        }
      }
    }
    if (comp.length > bestComp.length) bestComp = comp;
  }

  if (bestComp.length < 12) {
    return {
      ok: false,
      reason: 'lvot-not-visible',
      message: 'No se visualiza el TSVI en este corte. Adquiere una vista apical A5C o A3C.',
    };
  }

  const dr = depthCm / samples;
  const dth = sectorRad / lines;
  const sinTable = new Float64Array(lines);
  const cosTable = new Float64Array(lines);
  for (let li = 0; li < lines; li++) {
    const th = -0.5 * sectorRad + (li + 0.5) * dth;
    sinTable[li] = Math.sin(th);
    cosTable[li] = Math.cos(th);
  }

  interface LvotCell {
    li: number;
    si: number;
    r: number;
    th: number;
    x: number;
    y: number;
    area: number;
    wallClearance: number;
  }

  const cells: LvotCell[] = [];
  const outletPts: { x: number; y: number; weight: number }[] = [];
  const inletPts: { x: number; y: number }[] = [];
  let areaSum = 0;
  let cxSum = 0;
  let cySum = 0;

  const isWallNeighbor = (li: number, si: number): boolean => {
    if (li < 0 || li >= lines || si < 0 || si >= samples) return true;
    const st = structure[li * samples + si]!;
    return st !== Structure.Lvot && st !== Structure.LvCavity && !AV_OUTLET_SET.has(st);
  };

  for (const idx of bestComp) {
    const li = (idx / samples) | 0;
    const si = idx - li * samples;
    const r = (si + 0.5) * dr;
    const th = -0.5 * sectorRad + (li + 0.5) * dth;
    const x = r * sinTable[li]!;
    const y = r * cosTable[li]!;
    const cellArea = r * dr * dth;
    areaSum += cellArea;
    cxSum += x * cellArea;
    cySum += y * cellArea;

    // Count how many rings of LVOT/blood clearance separate this sample from non-blood walls
    let clear = 2;
    for (let dl = -2; dl <= 2; dl++) {
      for (let ds = -2; ds <= 2; ds++) {
        if (dl === 0 && ds === 0) continue;
        const step = Math.max(Math.abs(dl), Math.abs(ds));
        if (step < clear && isWallNeighbor(li + dl, si + ds)) {
          clear = step - 1;
        }
      }
    }
    cells.push({ li, si, r, th, x, y, area: cellArea, wallClearance: clear });

    // Collect aortic valve/root boundary pixels (distal) and LV cavity boundary pixels (proximal)
    for (let dl = -2; dl <= 2; dl++) {
      const nl = li + dl;
      if (nl < 0 || nl >= lines) continue;
      for (let ds = -2; ds <= 2; ds++) {
        const ns = si + ds;
        if (ns < 0 || ns >= samples) continue;
        const nst = structure[nl * samples + ns]!;
        if (nst === Structure.AorticValve || nst === Structure.AorticRoot) {
          const nr = (ns + 0.5) * dr;
          const weight = nst === Structure.AorticValve ? 2.5 : 1.0;
          outletPts.push({
            x: nr * sinTable[nl]!,
            y: nr * cosTable[nl]!,
            weight,
          });
        } else if (nst === Structure.LvCavity && Math.abs(dl) <= 1 && Math.abs(ds) <= 1) {
          const nr = (ns + 0.5) * dr;
          inletPts.push({
            x: nr * sinTable[nl]!,
            y: nr * cosTable[nl]!,
          });
        }
      }
    }
  }

  if (areaSum < MIN_LVOT_AREA_CM2) {
    return {
      ok: false,
      reason: 'lvot-not-visible',
      message: 'El TSVI es demasiado pequeño en este plano. Afinar A5C o A3C para abrir el tracto de salida.',
    };
  }

  const cx = cxSum / areaSum;
  const cy = cySum / areaSum;

  // 2. Estimate the 2D LVOT outflow axis from proximal LV-cavity inlet -> distal aortic-valve outlet
  //    (using the wider inlet-to-outlet baseline prevents lateral leaflet asymmetry from skewing the axis in dilated LVs).
  let avX = 0;
  let avY = 0;
  let avW = 0;
  for (const p of outletPts) {
    avX += p.x * p.weight;
    avY += p.y * p.weight;
    avW += p.weight;
  }

  // Fallback to scanning nearby AorticValve/AorticRoot pixels if LVOT was slightly separated by leaflet thickness.
  if (avW === 0) {
    for (let li = 0; li < lines; li++) {
      for (let si = 0; si < samples; si++) {
        const st = structure[li * samples + si]!;
        if (st !== Structure.AorticValve && st !== Structure.AorticRoot) continue;
        const r = (si + 0.5) * dr;
        const x = r * sinTable[li]!;
        const y = r * cosTable[li]!;
        if (Math.hypot(x - cx, y - cy) <= 3.0) {
          const weight = st === Structure.AorticValve ? 2.0 : 1.0;
          avX += x * weight;
          avY += y * weight;
          avW += weight;
        }
      }
    }
  }

  if (avW === 0) {
    return {
      ok: false,
      reason: 'lvot-not-visible',
      message: 'No se identifica el plano valvular aórtico junto al TSVI. Abre la vista A5C o A3C.',
    };
  }
  avX /= avW;
  avY /= avW;

  let inX = cx;
  let inY = cy;
  if (inletPts.length >= 4) {
    let sx = 0;
    let sy = 0;
    for (const p of inletPts) {
      sx += p.x;
      sy += p.y;
    }
    inX = sx / inletPts.length;
    inY = sy / inletPts.length;
  }

  const fx = avX - inX;
  const fy = avY - inY;
  const fLen = Math.hypot(fx, fy);
  if (fLen < 0.2) {
    return {
      ok: false,
      reason: 'lvot-not-visible',
      message: 'No se pudo orientar el eje del TSVI en este corte. Usa una vista A5C o A3C.',
    };
  }
  const ufx = fx / fLen;
  const ufy = fy / fLen;

  // 3. Check 2D beam–flow alignment (apical views A5C/A3C have LVOT roughly parallel to beam; PLAX is nearly perpendicular).
  const cDist = Math.hypot(cx, cy) || 1;
  const ubx = cx / cDist;
  const uby = cy / cDist;
  const absCos = Math.min(1, Math.abs(ufx * ubx + ufy * uby));
  const angle2dDeg = (Math.acos(absCos) * 180) / Math.PI;
  if (absCos < MIN_BEAM_ALIGNMENT) {
    return {
      ok: false,
      reason: 'misaligned-view',
      message: `El TSVI está casi perpendicular al haz (~${Math.round(angle2dDeg)}°). Pasa a vista apical A5C o A3C para Doppler pulsado.`,
    };
  }

  // 4. Select the mid-lumen LVOT point ~0.50 cm proximal to the systolic aortic valve interface.
  //    When only a diastolic frame is available, compensate apically so systolic annular descent keeps
  //    the fixed PW gate squarely inside the LVOT during ejection instead of crossing into the aortic root.
  const tx = -ufy;
  const ty = ufx;

  const avProj = avX * ufx + avY * ufy;
  let maxLvotProj = -Infinity;
  let minLvotProj = Infinity;
  let transCenterSum = 0;
  for (const c of cells) {
    const s = c.x * ufx + c.y * ufy;
    if (s > maxLvotProj) maxLvotProj = s;
    if (s < minLvotProj) minLvotProj = s;
    transCenterSum += (c.x * tx + c.y * ty) * c.area;
  }
  const valveAnchorProj = Math.min(avProj, maxLvotProj + 0.08);
  const lvotLenCm = Math.max(0.25, valveAnchorProj - minLvotProj);
  const systolicFrame = opts?.systolicFrame ?? true;
  const rawDist = systolicFrame
    ? Math.min(TARGET_DIST_SYSTOLE_CM, lvotLenCm * 0.45)
    : Math.min(TARGET_DIST_SYSTOLE_CM + DIASTOLIC_DESCENT_COMP_CM, lvotLenCm * 0.85);
  const desiredDist = Math.max(MIN_DIST_PROXIMAL_CM, Math.min(MAX_DIST_PROXIMAL_CM, rawDist));
  const targetProj = valveAnchorProj - desiredDist;

  // Compute local transverse center around the target axial band so the gate stays dead-center in the lumen.
  let bandTransSum = 0;
  let bandArea = 0;
  for (const c of cells) {
    const s = c.x * ufx + c.y * ufy;
    if (Math.abs(s - targetProj) <= 0.35) {
      bandTransSum += (c.x * tx + c.y * ty) * c.area;
      bandArea += c.area;
    }
  }
  const targetTrans = bandArea > 0 ? bandTransSum / bandArea : transCenterSum / areaSum;

  // Score every LVOT cell: prefer deep wall clearance, proximity to target axial station, and mid-lumen transverse centering.
  let bestCell = cells[0]!;
  let bestScore = Infinity;
  for (const c of cells) {
    const s = c.x * ufx + c.y * ufy;
    const t = c.x * tx + c.y * ty;
    const ds = s - targetProj;
    const dt = t - targetTrans;
    const wallPenalty = c.wallClearance >= 2 ? 0 : c.wallClearance === 1 ? 0.12 : 0.45;
    const score = ds * ds * 1.1 + dt * dt * 2.2 + wallPenalty;
    if (score < bestScore) {
      bestScore = score;
      bestCell = c;
    }
  }

  // If only a diastolic frame was available, apply residual systolic annular descent along -u_flow
  // so the fixed gate lands inside the LVOT during systolic ejection (never drifting into the aortic root).
  let gateX = bestCell.x;
  let gateY = bestCell.y;
  if (!systolicFrame) {
    const achievedDist = valveAnchorProj - (bestCell.x * ufx + bestCell.y * ufy);
    const neededShift = Math.max(0, TARGET_DIST_SYSTOLE_CM + DIASTOLIC_DESCENT_COMP_CM - achievedDist);
    const shift = Math.min(0.6, neededShift);
    gateX -= ufx * shift;
    gateY -= ufy * shift;
  }

  const distToAvCm = Math.max(0.05, valveAnchorProj - (gateX * ufx + gateY * ufy));
  const clampedTheta = Math.max(-sectorRad / 2, Math.min(sectorRad / 2, Math.atan2(gateX, gateY)));
  const clampedR = Math.max(1.0, Math.min(depthCm - 0.5, Math.hypot(gateX, gateY)));

  return {
    ok: true,
    rCm: clampedR,
    thetaRad: clampedTheta,
    angle2dDeg,
    distToAvCm,
    lvotAreaCm2: areaSum,
  };
}
