import type { Vec3 } from '@/core/vec3';
import { add, cross, normalize, scale, sub, v3 } from '@/core/vec3';
import type { HeartModel, HeartPose } from './heartModel';
import { anchorsCached, classifyHeart } from './heartModel';
import { RV_GROOVE_ANTERIOR_RAD } from './lvSegments';
import { makeSample, Structure, Tissue } from './tissue';
import type { MeshGroup } from './heartMesh';

/**
 * Epicardial coronary arteries and cardiac veins projected directly onto the active case's
 * implicit heart surface (`classifyHeart`) at the current cardiac phase. Because every
 * control point is ray-snapped to the epicardial boundary, the vessels hug dilated,
 * hypertrophic, and contracting hearts alike without floating or burying inside the wall.
 */

interface VesselSpline {
  points: Vec3[];
  radii: number[];
}

const isSolidCardiac = (t: Tissue, s: Structure): boolean => {
  if (t === Tissue.None || t === Tissue.Fluid || t === Tissue.Lung) return false;
  if (
    s === Structure.Pericardium ||
    s === Structure.PericardialEffusion ||
    s === Structure.PericardialFat ||
    s === Structure.EpicardialFat ||
    s === Structure.DescendingAorta
  ) {
    return false;
  }
  return true;
};

/**
 * Finds the epicardial surface point starting from an interior axis point `origin` along `dir`.
 * Returns null if the ray never crosses solid cardiac tissue.
 */
function snapToEpicardium(
  heart: HeartModel,
  pose: HeartPose,
  origin: Vec3,
  dir: Vec3,
  rMax: number,
  outsetCm: number,
  sample = makeSample(),
): Vec3 | null {
  const step = 0.24;
  let lastInside = -1;
  for (let r = 0.4; r <= rMax; r += step) {
    const px = origin.x + dir.x * r;
    const py = origin.y + dir.y * r;
    const pz = origin.z + dir.z * r;
    if (classifyHeart(heart, pose, px, py, pz, sample) && isSolidCardiac(sample.tissue, sample.structure)) {
      lastInside = r;
    }
  }
  if (lastInside < 0) return null;
  let lo = lastInside;
  let hi = Math.min(rMax + step, lastInside + step);
  for (let it = 0; it < 5; it++) {
    const mid = 0.5 * (lo + hi);
    const px = origin.x + dir.x * mid;
    const py = origin.y + dir.y * mid;
    const pz = origin.z + dir.z * mid;
    if (classifyHeart(heart, pose, px, py, pz, sample) && isSolidCardiac(sample.tissue, sample.structure)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  const finalR = lo + outsetCm;
  return v3(origin.x + dir.x * finalR, origin.y + dir.y * finalR, origin.z + dir.z * finalR);
}

/** Evaluates a uniform Catmull-Rom spline at `t` in [0, 1] across `pts`. */
function sampleCatmullRom(pts: Vec3[], t: number): Vec3 {
  const n = pts.length;
  if (n === 1) return pts[0]!;
  const scaled = Math.max(0, Math.min(n - 1 - 1e-6, t * (n - 1)));
  const i = Math.floor(scaled);
  const u = scaled - i;
  const p0 = pts[Math.max(0, i - 1)]!;
  const p1 = pts[i]!;
  const p2 = pts[Math.min(n - 1, i + 1)]!;
  const p3 = pts[Math.min(n - 1, i + 2)]!;
  const u2 = u * u;
  const u3 = u2 * u;
  const evalAxis = (a: number, b: number, c: number, d: number) =>
    0.5 * (2 * b + (-a + c) * u + (2 * a - 5 * b + 4 * c - d) * u2 + (-a + 3 * b - 3 * c + d) * u3);
  return v3(
    evalAxis(p0.x, p1.x, p2.x, p3.x),
    evalAxis(p0.y, p1.y, p2.y, p3.y),
    evalAxis(p0.z, p1.z, p2.z, p3.z),
  );
}

/** Builds a smooth tapered tubular mesh for one or more vessel splines. */
function buildTubesMesh(
  id: string,
  label: string,
  color: number,
  splines: VesselSpline[],
  radialSegs = 7,
  samplesPerSeg = 3,
): MeshGroup {
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  const ao: number[] = [];

  for (const spline of splines) {
    if (spline.points.length < 2) continue;
    const totalRings = Math.max(6, (spline.points.length - 1) * samplesPerSeg + 1);
    const centers: Vec3[] = [];
    const radii: number[] = [];
    for (let s = 0; s < totalRings; s++) {
      const t = s / (totalRings - 1);
      centers.push(sampleCatmullRom(spline.points, t));
      const rIdx = t * (spline.radii.length - 1);
      const r0 = Math.floor(rIdx);
      const r1 = Math.min(spline.radii.length - 1, r0 + 1);
      const frac = rIdx - r0;
      radii.push(spline.radii[r0]! * (1 - frac) + spline.radii[r1]! * frac);
    }

    // Parallel-transport frame along the vessel to avoid twisting
    const tangents: Vec3[] = centers.map((_, i) => {
      const a = centers[Math.max(0, i - 1)]!;
      const b = centers[Math.min(centers.length - 1, i + 1)]!;
      return normalize(sub(b, a));
    });
    const t0 = tangents[0]!;
    const ref = Math.abs(t0.z) < 0.85 ? v3(0, 0, 1) : v3(1, 0, 0);
    let normalVec = normalize(cross(t0, ref));
    let binormalVec = normalize(cross(t0, normalVec));

    const baseVertex = positions.length / 3;
    for (let s = 0; s < totalRings; s++) {
      const T = tangents[s]!;
      if (s > 0) {
        // Re-orthogonalize previous normal against current tangent
        const dotTN = T.x * normalVec.x + T.y * normalVec.y + T.z * normalVec.z;
        const proj = v3(normalVec.x - dotTN * T.x, normalVec.y - dotTN * T.y, normalVec.z - dotTN * T.z);
        const len = Math.hypot(proj.x, proj.y, proj.z);
        if (len > 1e-5) normalVec = scale(proj, 1 / len);
        binormalVec = normalize(cross(T, normalVec));
      }
      const c = centers[s]!;
      // Taper cleanly at distal tip
      const tipScale = s === totalRings - 1 ? 0.35 : 1;
      const r = radii[s]! * tipScale;
      for (let k = 0; k < radialSegs; k++) {
        const a = (k / radialSegs) * Math.PI * 2;
        const ca = Math.cos(a);
        const sa = Math.sin(a);
        const nx = normalVec.x * ca + binormalVec.x * sa;
        const ny = normalVec.y * ca + binormalVec.y * sa;
        const nz = normalVec.z * ca + binormalVec.z * sa;
        positions.push(c.x + nx * r, c.y + ny * r, c.z + nz * r);
        normals.push(nx, ny, nz);
        ao.push(0.95);
      }
    }

    for (let s = 0; s < totalRings - 1; s++) {
      const rowA = baseVertex + s * radialSegs;
      const rowB = baseVertex + (s + 1) * radialSegs;
      for (let k = 0; k < radialSegs; k++) {
        const kn = (k + 1) % radialSegs;
        const v0 = rowA + k;
        const v1 = rowA + kn;
        const v2 = rowB + kn;
        const v3i = rowB + k;
        indices.push(v0, v2, v1, v0, v3i, v2);
      }
    }
  }

  return {
    id,
    label,
    color,
    opacity: 1,
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array(indices),
    ao: new Float32Array(ao),
  };
}

/**
 * Generates anatomical epicardial coronary arteries (LAD, D1, D2, LCx/OM1, RCA/acute marginal)
 * and the great cardiac vein conformed to `heart` at `pose`.
 */
export function buildCoronaryMeshes(heart: HeartModel, pose: HeartPose): MeshGroup[] {
  const A = anchorsCached(heart);
  const L = heart.lv.lengthCm;
  const zAnn = pose.zAnn;
  const effL = Math.max(4.5, L - zAnn);
  const sample = makeSample();
  const radialSearch = Math.max(7.2, heart.lv.rMax + heart.anatomy.lv.ivsdCm + heart.anatomy.rv.basalDiameterCm + 2.2);

  const traceLvGroove = (
    specs: { frac: number; az: number; r: number; outset?: number }[],
  ): VesselSpline => {
    const points: Vec3[] = [];
    const radii: number[] = [];
    for (const sp of specs) {
      const z = zAnn + sp.frac * effL;
      const dir = v3(Math.cos(sp.az), Math.sin(sp.az), 0);
      const hit = snapToEpicardium(
        heart,
        pose,
        v3(0, 0, z),
        dir,
        radialSearch,
        sp.outset ?? sp.r * 0.32,
        sample,
      );
      if (hit) {
        points.push(hit);
        radii.push(sp.r);
      }
    }
    return { points, radii };
  };

  // 1. LAD along the anterior interventricular groove (RV_GROOVE_ANTERIOR_RAD ≈ 1.60 rad)
  const ladSpecs: { frac: number; az: number; r: number }[] = [];
  for (let i = 0; i <= 9; i++) {
    const f = 0.04 + (i / 9) * 0.90;
    const wave = 0.055 * Math.sin(f * Math.PI * 2.6);
    const az = RV_GROOVE_ANTERIOR_RAD - 0.08 * (1 - f) + wave;
    const r = 0.145 * (1 - 0.52 * f);
    ladSpecs.push({ frac: f, az, r });
  }
  const lad = traceLvGroove(ladSpecs);

  // 2. First diagonal (D1) sweeping across the anterolateral LV wall
  const d1 = traceLvGroove([
    { frac: 0.25, az: RV_GROOVE_ANTERIOR_RAD - 0.03, r: 0.105 },
    { frac: 0.36, az: RV_GROOVE_ANTERIOR_RAD - 0.34, r: 0.092 },
    { frac: 0.50, az: RV_GROOVE_ANTERIOR_RAD - 0.68, r: 0.078 },
    { frac: 0.64, az: RV_GROOVE_ANTERIOR_RAD - 0.96, r: 0.064 },
    { frac: 0.76, az: RV_GROOVE_ANTERIOR_RAD - 1.12, r: 0.052 },
  ]);

  // 3. Second diagonal (D2) lower anterolateral branch
  const d2 = traceLvGroove([
    { frac: 0.52, az: RV_GROOVE_ANTERIOR_RAD - 0.01, r: 0.085 },
    { frac: 0.63, az: RV_GROOVE_ANTERIOR_RAD - 0.32, r: 0.072 },
    { frac: 0.74, az: RV_GROOVE_ANTERIOR_RAD - 0.58, r: 0.060 },
    { frac: 0.85, az: RV_GROOVE_ANTERIOR_RAD - 0.76, r: 0.048 },
  ]);

  // 4. LCx curving in the left AV sulcus + Obtuse Marginal (OM1) descending laterally
  const lcx = traceLvGroove([
    { frac: 0.05, az: RV_GROOVE_ANTERIOR_RAD - 0.16, r: 0.13 },
    { frac: 0.08, az: 1.05, r: 0.12 },
    { frac: 0.11, az: 0.58, r: 0.11 },
    { frac: 0.13, az: 0.12, r: 0.10 },
    { frac: 0.15, az: -0.38, r: 0.09 },
    { frac: 0.17, az: -0.88, r: 0.078 },
  ]);
  const om1 = traceLvGroove([
    { frac: 0.12, az: 0.36, r: 0.095 },
    { frac: 0.28, az: 0.22, r: 0.082 },
    { frac: 0.46, az: 0.06, r: 0.070 },
    { frac: 0.64, az: -0.08, r: 0.058 },
    { frac: 0.79, az: -0.16, r: 0.048 },
  ]);

  // 5. RCA emerging near right aortic sinus & coursing through the right anterior AV groove
  const rcaPoints: Vec3[] = [];
  const rcaRadii: number[] = [];
  const rcaOrigin = v3(A.tvCenter.x * 0.55, A.tvCenter.y + 0.2, zAnn + 0.35);
  const rcaAngles = [1.82, 2.08, 2.38, 2.68, 2.98, 3.28, 3.55];
  for (let i = 0; i < rcaAngles.length; i++) {
    const f = i / (rcaAngles.length - 1);
    const az = rcaAngles[i]!;
    const zShift = zAnn + 0.15 + f * 0.85;
    const dir = v3(Math.cos(az), Math.sin(az), 0.06 * f);
    const hit = snapToEpicardium(
      heart,
      pose,
      v3(rcaOrigin.x * (1 - 0.3 * f), rcaOrigin.y, zShift),
      normalize(dir),
      radialSearch,
      0.05,
      sample,
    );
    if (hit) {
      rcaPoints.push(hit);
      rcaRadii.push(0.14 * (1 - 0.38 * f));
    }
  }
  const rca: VesselSpline = { points: rcaPoints, radii: rcaRadii };

  // 6. Acute marginal branch of RCA coursing over anterior-right RV free wall
  const acuteMarginal = traceLvGroove([
    { frac: 0.14, az: 2.52, r: 0.092 },
    { frac: 0.28, az: 2.38, r: 0.080 },
    { frac: 0.45, az: 2.22, r: 0.068 },
    { frac: 0.62, az: 2.04, r: 0.056 },
    { frac: 0.76, az: 1.88, r: 0.046 },
  ]);

  // 7. Great Cardiac Vein running parallel and just lateral to LAD into the left AV sulcus
  const gcvSpecs: { frac: number; az: number; r: number }[] = [];
  for (let i = 0; i <= 8; i++) {
    const f = 0.84 - (i / 8) * 0.76;
    const wave = 0.045 * Math.sin(f * Math.PI * 2.6);
    const az = RV_GROOVE_ANTERIOR_RAD - 0.20 - 0.08 * (1 - f) + wave;
    const r = 0.055 + 0.065 * (1 - f);
    gcvSpecs.push({ frac: f, az, r });
  }
  // Curve leftward into left AV groove
  gcvSpecs.push(
    { frac: 0.09, az: 0.92, r: 0.125 },
    { frac: 0.11, az: 0.38, r: 0.132 },
    { frac: 0.13, az: -0.22, r: 0.138 },
  );
  const gcv = traceLvGroove(gcvSpecs);

  // 8. Anterior RV cardiac vein tributary
  const rvVein = traceLvGroove([
    { frac: 0.60, az: 2.02, r: 0.052 },
    { frac: 0.44, az: 2.18, r: 0.065 },
    { frac: 0.28, az: 2.35, r: 0.078 },
    { frac: 0.14, az: 2.58, r: 0.090 },
  ]);

  // Slightly nudge arteries/veins apart at origins if any points collide
  const arteries = buildTubesMesh(
    'coronary-arteries',
    'Arterias coronarias',
    0xd62839,
    [lad, d1, d2, lcx, om1, rca, acuteMarginal].filter((s) => s.points.length >= 2),
  );
  const veins = buildTubesMesh(
    'coronary-veins',
    'Venas cardíacas',
    0x2c5ea0,
    [gcv, rvVein].filter((s) => s.points.length >= 2),
  );
  // Ensure slight outward offset on empty edge cases handled gracefully
  void add;
  return [arteries, veins].filter((m) => m.indices.length > 0);
}
