import { Structure } from '@/simulator/anatomy/tissue';
import type { ImagingModality, PolarFrameSpec } from '@/simulator/renderer/types';
import { findLvotGateFromPolar } from './autoLvotGate';

export type DopplerTargetId =
  | 'lvot'
  | 'av-cw'
  | 'mitral-inflow'
  | 'tdi-septal'
  | 'tdi-lateral'
  | 'tapse'
  | 'rv-s-prime'
  | 'tr-cw'
  | 'rvot-pw';

export interface AutoDopplerTargetSuccess {
  ok: true;
  target: DopplerTargetId;
  modality: ImagingModality;
  /** Polar sample-volume depth from transducer apex (cm). */
  rCm: number;
  /** Polar cursor angle from sector centerline (rad). */
  thetaRad: number;
  /** Optional clinical baseline shift (m/s) for aliasing headroom (e.g. -0.25 for apical LVOT, +0.25 for mitral E/A). */
  recommendedBaselineShiftMps?: number;
  /** Concise Spanish status summary for the operator. */
  summary: string;
}

export interface AutoDopplerTargetFailure {
  ok: false;
  target: DopplerTargetId;
  reason: 'no-frame' | 'structure-not-visible' | 'misaligned-view';
  message: string;
}

export type AutoDopplerTargetResult = AutoDopplerTargetSuccess | AutoDopplerTargetFailure;

interface PolarCell {
  li: number;
  si: number;
  r: number;
  th: number;
  x: number;
  y: number;
  area: number;
}

interface PolarLookup {
  lines: number;
  samples: number;
  sectorRad: number;
  depthCm: number;
  dr: number;
  dth: number;
  sinTable: Float64Array;
  cosTable: Float64Array;
}

function makeLookup(
  spec: Pick<PolarFrameSpec, 'lines' | 'samples' | 'sectorRad' | 'depthCm'>,
): PolarLookup {
  const { lines, samples, sectorRad, depthCm } = spec;
  const dr = depthCm / samples;
  const dth = sectorRad / lines;
  const sinTable = new Float64Array(lines);
  const cosTable = new Float64Array(lines);
  for (let li = 0; li < lines; li++) {
    const th = -0.5 * sectorRad + (li + 0.5) * dth;
    sinTable[li] = Math.sin(th);
    cosTable[li] = Math.cos(th);
  }
  return { lines, samples, sectorRad, depthCm, dr, dth, sinTable, cosTable };
}

function collectCells(
  structure: Uint8Array,
  lut: PolarLookup,
  match: (st: number) => boolean,
): PolarCell[] {
  const out: PolarCell[] = [];
  const { lines, samples, dr, dth, sectorRad, sinTable, cosTable } = lut;
  for (let li = 0; li < lines; li++) {
    const th = -0.5 * sectorRad + (li + 0.5) * dth;
    const sn = sinTable[li]!;
    const cs = cosTable[li]!;
    const base = li * samples;
    for (let si = 0; si < samples; si++) {
      if (!match(structure[base + si]!)) continue;
      const r = (si + 0.5) * dr;
      out.push({ li, si, r, th, x: r * sn, y: r * cs, area: r * dr * dth });
    }
  }
  return out;
}

function areaCentroid(cells: readonly PolarCell[]): { x: number; y: number; area: number } | null {
  if (!cells.length) return null;
  let sx = 0;
  let sy = 0;
  let sa = 0;
  for (const c of cells) {
    sx += c.x * c.area;
    sy += c.y * c.area;
    sa += c.area;
  }
  if (!(sa > 0)) return null;
  return { x: sx / sa, y: sy / sa, area: sa };
}

function hasNeighbor(
  structure: Uint8Array,
  lut: PolarLookup,
  li: number,
  si: number,
  radius: number,
  match: (st: number) => boolean,
): boolean {
  for (let dl = -radius; dl <= radius; dl++) {
    const nl = li + dl;
    if (nl < 0 || nl >= lut.lines) continue;
    const base = nl * lut.samples;
    for (let ds = -radius; ds <= radius; ds++) {
      const ns = si + ds;
      if (ns < 0 || ns >= lut.samples) continue;
      if (match(structure[base + ns]!)) return true;
    }
  }
  return false;
}

function sameStructClearance(
  structure: Uint8Array,
  lut: PolarLookup,
  li: number,
  si: number,
  target: number,
): number {
  let clear = 2;
  for (let dl = -2; dl <= 2; dl++) {
    for (let ds = -2; ds <= 2; ds++) {
      if (dl === 0 && ds === 0) continue;
      const step = Math.max(Math.abs(dl), Math.abs(ds));
      if (step >= clear) continue;
      const nl = li + dl;
      const ns = si + ds;
      if (nl < 0 || nl >= lut.lines || ns < 0 || ns >= lut.samples) {
        clear = step - 1;
      } else if (structure[nl * lut.samples + ns] !== target) {
        clear = step - 1;
      }
    }
  }
  return clear;
}

function clampPolar(lut: PolarLookup, x: number, y: number): { rCm: number; thetaRad: number } {
  const r = Math.hypot(x, y);
  const th = Math.atan2(x, y);
  return {
    rCm: Math.max(1.0, Math.min(lut.depthCm - 0.5, r)),
    thetaRad: Math.max(-lut.sectorRad / 2, Math.min(lut.sectorRad / 2, th)),
  };
}

/**
 * Pure 2D polar locator for the highest-frequency clinical Doppler & M-mode targets (LVOT PW,
 * mitral inflow PW, septal/lateral TDI e′, lateral tricuspid TAPSE M-mode, and TR CW).
 * Reads exclusively from the rendered 2D polar `structure` map — zero cheating from ground truth.
 */
export function findDopplerTargetFromPolar(
  structure: Uint8Array,
  spec: Pick<PolarFrameSpec, 'lines' | 'samples' | 'sectorRad' | 'depthCm'>,
  target: DopplerTargetId,
  opts?: { systolicFrame?: boolean },
): AutoDopplerTargetResult {
  const total = spec.lines * spec.samples;
  if (structure.length < total || spec.lines < 8 || spec.samples < 16) {
    return {
      ok: false,
      target,
      reason: 'no-frame',
      message: 'No hay un cuadro 2D disponible para posicionar el cursor.',
    };
  }

  if (target === 'lvot' || target === 'av-cw') {
    const res = findLvotGateFromPolar(structure, spec, opts);
    if (!res.ok) {
      return {
        ok: false,
        target,
        reason: res.reason === 'misaligned-view' ? 'misaligned-view' : 'structure-not-visible',
        message: res.message,
      };
    }
    if (target === 'lvot') {
      return {
        ok: true,
        target,
        modality: 'pw',
        rCm: res.rCm,
        thetaRad: res.thetaRad,
        recommendedBaselineShiftMps: -0.25,
        summary: `Gate PW en TSVI (${res.rCm.toFixed(1)} cm · ~${res.distToAvCm.toFixed(1)} cm proximal a VAo).`,
      };
    }
    // target === 'av-cw': aim the CW ray squarely through the open aortic valve orifice along LVOT -> AorticValve
    const lutAv = makeLookup(spec);
    const gx = res.rCm * Math.sin(res.thetaRad);
    const gy = res.rCm * Math.cos(res.thetaRad);
    const avValve = collectCells(structure, lutAv, (s) => s === Structure.AorticValve).filter(
      (c) => Math.hypot(c.x - gx, c.y - gy) <= 3.0,
    );
    const avRoot =
      avValve.length >= 2
        ? avValve
        : collectCells(structure, lutAv, (s) => s === Structure.AorticRoot).filter(
            (c) => Math.hypot(c.x - gx, c.y - gy) <= 3.2,
          );
    const avC = areaCentroid(avRoot);
    const aimX = avC ? 0.65 * avC.x + 0.35 * gx : gx;
    const aimY = avC ? 0.65 * avC.y + 0.35 * gy : gy;
    const p = clampPolar(lutAv, aimX, aimY);
    return {
      ok: true,
      target,
      modality: 'cw',
      rCm: p.rCm,
      thetaRad: p.thetaRad,
      recommendedBaselineShiftMps: -1.5,
      summary: `Cursor CW alineado con la válvula aórtica (${Math.round((p.thetaRad * 180) / Math.PI)}°) para VTI/Vmáx VAo.`,
    };
  }

  const lut = makeLookup(spec);

  // --- PSAX-AV RVOT / Pulmonary valve PW target (for Pulmonary Acceleration Time & RVOT VTI) ---
  if (target === 'rvot-pw') {
    const pvCells = collectCells(structure, lut, (s) => s === Structure.PulmonaryValve);
    const rvotCells = collectCells(structure, lut, (s) => s === Structure.Rvot);
    const paCells = collectCells(structure, lut, (s) => s === Structure.PulmonaryArtery);
    const aoCells = collectCells(structure, lut, (s) => s === Structure.AorticRoot);
    if (
      aoCells.length < 25 ||
      (pvCells.length === 0 && (rvotCells.length < 12 || paCells.length < 20))
    ) {
      return {
        ok: false,
        target,
        reason: 'structure-not-visible',
        message:
          'No se visualiza el TSVD y la válvula pulmonar en este corte. Usa eje corto de grandes vasos (PSAX Ao).',
      };
    }

    // Prefer the pulmonary valve orifice or the immediate RVOT-PA junction (ASE 2025 TAc pulmonar)
    let targetCell: PolarCell | null = null;
    if (pvCells.length >= 2) {
      const pvC = areaCentroid(pvCells)!;
      let bestD = Infinity;
      for (const c of pvCells) {
        const d = Math.hypot(c.x - pvC.x, c.y - pvC.y);
        if (d < bestD) {
          bestD = d;
          targetCell = c;
        }
      }
    } else {
      const junction = rvotCells.filter((c) =>
        hasNeighbor(structure, lut, c.li, c.si, 4, (s) => s === Structure.PulmonaryArtery),
      );
      const pool = junction.length >= 2 ? junction : paCells;
      const cMid = areaCentroid(pool);
      if (cMid) {
        let bestD = Infinity;
        for (const c of pool) {
          const d = Math.hypot(c.x - cMid.x, c.y - cMid.y);
          if (d < bestD) {
            bestD = d;
            targetCell = c;
          }
        }
      }
    }
    if (!targetCell) {
      return {
        ok: false,
        target,
        reason: 'structure-not-visible',
        message: 'No se pudo localizar el plano valvular pulmonar en el TSVD.',
      };
    }
    const p = clampPolar(lut, targetCell.x, targetCell.y);
    return {
      ok: true,
      target,
      modality: 'pw',
      rCm: p.rCm,
      thetaRad: p.thetaRad,
      recommendedBaselineShiftMps: -0.25,
      summary: `Gate PW en TSVD / válvula pulmonar (${p.rCm.toFixed(1)} cm) listo para TAc pulmonar.`,
    };
  }

  // --- Left-heart apical targets (Mitral inflow PW, TDI septal e′, TDI lateral e′) ---
  if (target === 'mitral-inflow' || target === 'tdi-septal' || target === 'tdi-lateral') {
    const lvCells = collectCells(structure, lut, (s) => s === Structure.LvCavity);
    const laCells = collectCells(structure, lut, (s) => s === Structure.LaCavity);
    const lvC = areaCentroid(lvCells);
    const laC = areaCentroid(laCells);
    if (!lvC || !laC || lvC.area < 3.0 || laC.area < 1.5) {
      return {
        ok: false,
        target,
        reason: 'structure-not-visible',
        message: 'Se requieren VI y AI visibles en plano apical (A4C) para este posicionamiento.',
      };
    }
    const dx = lvC.x - laC.x;
    const dy = lvC.y - laC.y;
    const dist = Math.hypot(dx, dy);
    const midX = 0.5 * (lvC.x + laC.x);
    const midY = 0.5 * (lvC.y + laC.y);
    const midLen = Math.hypot(midX, midY) || 1;
    // Apical beam–inflow alignment guard: LA→LV inflow must align closely (<~32°) toward the
    // transducer apex (rejects PLAX where beamCos ≈ 0.70 while accepting A4C/A2C/A3C ≥ 0.98).
    const beamCos = dist > 0 ? -(dx * midX + dy * midY) / (dist * midLen) : 0;
    if (!(dist > 1.5) || -dy / dist < 0.8 || beamCos < 0.85) {
      return {
        ok: false,
        target,
        reason: 'misaligned-view',
        message: 'Usa una vista apical (A4C) alineada con el eje longitudinal del VI.',
      };
    }

    // Estimate the mitral annular plane from leaflet/annulus pixels + LV–LA junction
    const mvCells = collectCells(
      structure,
      lut,
      (s) =>
        s === Structure.MitralAnterior ||
        s === Structure.MitralPosterior ||
        s === Structure.MitralAnnulus,
    );
    const mvC = areaCentroid(mvCells);
    const annX = mvC ? mvC.x : 0.55 * lvC.x + 0.45 * laC.x;
    const annY = mvC ? mvC.y : 0.55 * lvC.y + 0.45 * laC.y;
    const uInX = dx / dist;
    const uInY = dy / dist;

    if (target === 'mitral-inflow') {
      // Place PW sample volume at the mitral leaflet tips (~1.0 cm apical to annulus along LA->LV axis)
      const targetX = annX + uInX * 1.05;
      const targetY = annY + uInY * 1.05;
      let best = lvCells[0]!;
      let bestScore = Infinity;
      for (const c of lvCells) {
        const d2 = (c.x - targetX) ** 2 + (c.y - targetY) ** 2;
        const clear = sameStructClearance(structure, lut, c.li, c.si, Structure.LvCavity);
        const wallPen = clear >= 2 ? 0 : clear === 1 ? 0.12 : 0.5;
        const score = d2 + wallPen;
        if (score < bestScore) {
          bestScore = score;
          best = c;
        }
      }
      const p = clampPolar(lut, best.x, best.y);
      return {
        ok: true,
        target,
        modality: 'pw',
        rCm: p.rCm,
        thetaRad: p.thetaRad,
        recommendedBaselineShiftMps: 0.25,
        summary: `Gate PW en puntas mitrales (${p.rCm.toFixed(1)} cm) listo para E/A.`,
      };
    }

    // TDI septal or lateral mitral annulus: locate basal myocardium right beside the annular plane
    const wallStruct = target === 'tdi-septal' ? Structure.LvWallSeptal : Structure.LvWallLateral;
    const wallCells = collectCells(structure, lut, (s) => s === wallStruct);
    if (wallCells.length < 18) {
      return {
        ok: false,
        target,
        reason: 'structure-not-visible',
        message:
          target === 'tdi-septal'
            ? 'No se visualiza el septo basal en este corte. Usa vista A4C.'
            : 'No se visualiza la pared lateral basal en este corte. Usa vista A4C.',
      };
    }

    // Basal anchor: deepest portion of that wall closest to LA / MitralAnnulus / IAS
    const basalCells = wallCells.filter((c) =>
      hasNeighbor(
        structure,
        lut,
        c.li,
        c.si,
        5,
        (s) =>
          s === Structure.LaCavity ||
          s === Structure.MitralAnnulus ||
          s === Structure.InteratrialSeptum ||
          s === Structure.MitralAnterior ||
          s === Structure.MitralPosterior,
      ),
    );
    const pool = basalCells.length >= 6 ? basalCells : wallCells;
    // Anchor ~0.45 cm apical from the deepest basal edge and centered across mid-myocardial thickness
    // so physiological annular bobbing keeps the TDI gate inside the wall even in thin-walled HFrEF.
    let maxY = -Infinity;
    for (const c of pool) if (c.y > maxY) maxY = c.y;
    const desiredY = maxY - 0.45;
    const bandCells = wallCells.filter((c) => c.y >= maxY - 1.1 && c.y <= maxY - 0.1);
    const bandC = areaCentroid(bandCells.length >= 4 ? bandCells : pool);
    const midWallX = bandC ? bandC.x : pool[0]!.x;
    const candPool = bandCells.length >= 4 ? bandCells : pool;
    let best = candPool[0]!;
    let bestScore = Infinity;
    for (const c of candPool) {
      const clear = sameStructClearance(structure, lut, c.li, c.si, wallStruct);
      const dy2 = (c.y - desiredY) ** 2;
      const dx2 = (c.x - midWallX) ** 2;
      const wallPen = clear >= 2 ? 0 : clear === 1 ? 0.25 : 1.0;
      const score = dy2 * 2.0 + dx2 * 3.5 + wallPen;
      if (score < bestScore) {
        bestScore = score;
        best = c;
      }
    }
    const p = clampPolar(lut, best.x, best.y);
    return {
      ok: true,
      target,
      modality: 'tdi',
      rCm: p.rCm,
      thetaRad: p.thetaRad,
      recommendedBaselineShiftMps: 0,
      summary:
        target === 'tdi-septal'
          ? `Gate TDI en anillo mitral septal (${p.rCm.toFixed(1)} cm) listo para e′.`
          : `Gate TDI en anillo mitral lateral (${p.rCm.toFixed(1)} cm) listo para e′.`,
    };
  }

  // --- Right-heart targets (TAPSE M-mode, TR CW) ---
  const rvCells = collectCells(structure, lut, (s) => s === Structure.RvCavity);
  const raCells = collectCells(structure, lut, (s) => s === Structure.RaCavity);
  const rvC = areaCentroid(rvCells);
  const raC = areaCentroid(raCells);
  if (!rvC || !raC || rvC.area < 1.5 || raC.area < 1.0) {
    return {
      ok: false,
      target,
      reason: 'structure-not-visible',
      message: 'Se requieren VD y AD visibles (vista A4C o apical enfocada en VD).',
    };
  }
  const rx = raC.x - rvC.x;
  const ry = raC.y - rvC.y;
  const rDist = Math.hypot(rx, ry) || 1;
  // Beam alignment check along RV -> RA axis
  const midX = 0.5 * (rvC.x + raC.x);
  const midY = 0.5 * (rvC.y + raC.y);
  const midLen = Math.hypot(midX, midY) || 1;
  const align = Math.abs((rx * midX + ry * midY) / (rDist * midLen));

  if (target === 'tapse' || target === 'rv-s-prime') {
    if (ry / rDist < 0.65) {
      return {
        ok: false,
        target,
        reason: 'misaligned-view',
        message:
          target === 'tapse'
            ? 'TAPSE requiere vista apical A4C o enfocada en VD alineada con la pared libre.'
            : 'S′ VD requiere vista apical A4C o enfocada en VD alineada con el anillo tricuspídeo lateral.',
      };
    }
    const rvWallCells = collectCells(structure, lut, (s) => s === Structure.RvWall);
    const tvAnnCells = collectCells(structure, lut, (s) => s === Structure.TricuspidAnnulus);
    if (rvWallCells.length < 10 && tvAnnCells.length === 0) {
      return {
        ok: false,
        target,
        reason: 'structure-not-visible',
        message: 'No se identifica el anillo tricuspídeo lateral en este corte. Usa A4C o vista enfocada en VD.',
      };
    }
    // Find the basal junction of the RV free wall meeting RA / Tricuspid valve / Tricuspid annulus
    const basalRvWall = rvWallCells.filter((c) =>
      hasNeighbor(
        structure,
        lut,
        c.li,
        c.si,
        6,
        (s) =>
          s === Structure.RaCavity ||
          s === Structure.TricuspidAnnulus ||
          s === Structure.TricuspidValve ||
          s === Structure.RaWall,
      ),
    );
    const pool = tvAnnCells.length > 0 ? tvAnnCells : basalRvWall.length > 0 ? basalRvWall : rvWallCells;
    let best = pool[0]!;
    let bestScore = -Infinity;
    for (const c of pool) {
      // Deepest basal end of the lateral RV wall (closest to tricuspid annular hinge)
      const score = c.y - Math.abs(c.x - rvC.x) * 0.15;
      if (score > bestScore) {
        bestScore = score;
        best = c;
      }
    }
    if (target === 'rv-s-prime') {
      // Strictly target the lateral RV free wall (`RvWall`) inside the basal annular cuff
      // (never the septal tricuspid hinge adjacent to LvWallSeptal).
      const rvPool = basalRvWall.length >= 4 ? basalRvWall : rvWallCells;
      if (!rvPool.length) {
        return {
          ok: false,
          target,
          reason: 'structure-not-visible',
          message: 'No se identifica la pared libre basal del VD en este corte. Usa A4C o vista enfocada en VD.',
        };
      }
      let maxY = -Infinity;
      for (const c of rvPool) if (c.y > maxY) maxY = c.y;
      const desiredY = maxY - 0.35;
      const band = rvWallCells.filter((c) => c.y >= maxY - 0.9 && c.y <= maxY - 0.05);
      const candPool = band.length >= 3 ? band : rvPool;
      const bandC = areaCentroid(candPool);
      const midWallX = bandC ? bandC.x : candPool[0]!.x;
      let bestRv = candPool[0]!;
      let bestRvScore = Infinity;
      for (const c of candPool) {
        const clear = sameStructClearance(structure, lut, c.li, c.si, Structure.RvWall);
        const nearFat = hasNeighbor(
          structure,
          lut,
          c.li,
          c.si,
          1,
          (s) => s === Structure.EpicardialFat || s === Structure.Pericardium,
        );
        const wallPen = (clear >= 1 ? 0 : 0.35) + (nearFat ? 0.5 : 0);
        const score = (c.y - desiredY) ** 2 * 2.5 + (c.x - midWallX) ** 2 * 2.0 + wallPen;
        if (score < bestRvScore) {
          bestRvScore = score;
          bestRv = c;
        }
      }
      const p = clampPolar(lut, bestRv.x, bestRv.y);
      return {
        ok: true,
        target,
        modality: 'tdi',
        rCm: p.rCm,
        thetaRad: p.thetaRad,
        recommendedBaselineShiftMps: 0,
        summary: `Gate TDI en anillo tricuspídeo lateral (${p.rCm.toFixed(1)} cm) listo para S′ VD.`,
      };
    }
    const p = clampPolar(lut, best.x, best.y);
    return {
      ok: true,
      target,
      modality: 'm-mode',
      rCm: p.rCm,
      thetaRad: p.thetaRad,
      summary: `Cursor Modo M alineado con anillo tricuspídeo lateral (${p.rCm.toFixed(1)} cm) para TAPSE.`,
    };
  }

  // target === 'tr-cw'
  if (align < 0.62) {
    return {
      ok: false,
      target,
      reason: 'misaligned-view',
      message: 'El eje VD–AD no está alineado con el haz. Usa A4C, apical enfocada en VD o PSAX aórtico.',
    };
  }
  const tvCells = collectCells(structure, lut, (s) => s === Structure.TricuspidValve);
  const tvC = areaCentroid(tvCells);
  const tx = tvC ? tvC.x : midX;
  const ty = tvC ? tvC.y : midY;
  const p = clampPolar(lut, tx, ty);
  return {
    ok: true,
    target,
    modality: 'cw',
    rCm: p.rCm,
    thetaRad: p.thetaRad,
    recommendedBaselineShiftMps: -0.5,
    summary: `Cursor CW alineado a través de la válvula tricúspide (${Math.round((p.thetaRad * 180) / Math.PI)}°) para Vmax IT.`,
  };
}
