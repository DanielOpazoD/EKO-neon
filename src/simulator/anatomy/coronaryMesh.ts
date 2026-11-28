import type { Vec3 } from '@/core/vec3';
import { cross, normalize, scale, sub, v3 } from '@/core/vec3';
import type { HeartModel, HeartPose } from './heartModel';
import { anchorsCached } from './heartModel';
import { RV_GROOVE_ANTERIOR_RAD } from './lvSegments';
import type { MeshGroup } from './heartMesh';

/**
 * Epicardial coronary arteries and cardiac veins projected directly onto the ACTUAL
 * Taubin-smoothed rendered myocardial triangle meshes (`lv-myocardium` + `rv-myocardium`).
 *
 * Because every sampled ring center is raycast against the exact rendered surface triangles,
 * vessels sit flush in the epicardium with zero floating and zero sudden mesh clipping, use
 * realistic flattened cross-sections with underside micro-contact AO, and taper organically
 * down to fine terminal twigs.
 */

interface VesselSpec {
  /** Path points in heart frame: fractional base→apex `frac`, azimuth `az`, target radius `r` (cm). */
  nodes: { origin: Vec3; dir: Vec3; r: number }[];
  samplesPerSeg?: number;
}

interface SurfaceHit {
  point: Vec3;
  normal: Vec3;
  t: number;
}

interface TriSoup {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  /** Precomputed triangle bounding spheres [cx, cy, cz, r2] for fast ray rejection. */
  spheres: Float32Array;
}

function prepareTriSoup(meshes: MeshGroup[]): TriSoup[] {
  const soups: TriSoup[] = [];
  for (const m of meshes) {
    if (!m.indices.length) continue;
    const triCount = m.indices.length / 3;
    const spheres = new Float32Array(triCount * 4);
    const pos = m.positions;
    const idx = m.indices;
    for (let t = 0; t < triCount; t++) {
      const i0 = idx[t * 3]! * 3;
      const i1 = idx[t * 3 + 1]! * 3;
      const i2 = idx[t * 3 + 2]! * 3;
      const ax = pos[i0]!,
        ay = pos[i0 + 1]!,
        az = pos[i0 + 2]!;
      const bx = pos[i1]!,
        by = pos[i1 + 1]!,
        bz = pos[i1 + 2]!;
      const cx = pos[i2]!,
        cy = pos[i2 + 1]!,
        cz = pos[i2 + 2]!;
      const mx = (ax + bx + cx) / 3;
      const my = (ay + by + cy) / 3;
      const mz = (az + bz + cz) / 3;
      const d0 = (ax - mx) ** 2 + (ay - my) ** 2 + (az - mz) ** 2;
      const d1 = (bx - mx) ** 2 + (by - my) ** 2 + (bz - mz) ** 2;
      const d2 = (cx - mx) ** 2 + (cy - my) ** 2 + (cz - mz) ** 2;
      spheres[t * 4] = mx;
      spheres[t * 4 + 1] = my;
      spheres[t * 4 + 2] = mz;
      spheres[t * 4 + 3] = Math.max(d0, d1, d2) + 1e-4;
    }
    soups.push({ positions: pos, normals: m.normals, indices: idx, spheres });
  }
  return soups;
}

/**
 * Möller–Trumbore raycast returning the OUTERMOST intersection (`max t > 0`) on the
 * rendered myocardium meshes, along with the interpolated outward surface normal.
 */
function raycastOutermostSurface(origin: Vec3, dir: Vec3, soups: TriSoup[]): SurfaceHit | null {
  const ox = origin.x,
    oy = origin.y,
    oz = origin.z;
  const dx = dir.x,
    dy = dir.y,
    dz = dir.z;
  let bestT = -1;
  let bestNx = dx,
    bestNy = dy,
    bestNz = dz;

  for (const soup of soups) {
    const pos = soup.positions;
    const nrm = soup.normals;
    const idx = soup.indices;
    const sph = soup.spheres;
    const triCount = idx.length / 3;
    for (let tri = 0; tri < triCount; tri++) {
      // Fast bounding-sphere rejection
      const sOff = tri * 4;
      const toX = sph[sOff]! - ox;
      const toY = sph[sOff + 1]! - oy;
      const toZ = sph[sOff + 2]! - oz;
      const proj = toX * dx + toY * dy + toZ * dz;
      if (proj < 0) continue;
      const perp2 = toX * toX + toY * toY + toZ * toZ - proj * proj;
      if (perp2 > sph[sOff + 3]!) continue;

      const i0 = idx[tri * 3]! * 3;
      const i1 = idx[tri * 3 + 1]! * 3;
      const i2 = idx[tri * 3 + 2]! * 3;
      const v0x = pos[i0]!,
        v0y = pos[i0 + 1]!,
        v0z = pos[i0 + 2]!;
      const e1x = pos[i1]! - v0x,
        e1y = pos[i1 + 1]! - v0y,
        e1z = pos[i1 + 2]! - v0z;
      const e2x = pos[i2]! - v0x,
        e2y = pos[i2 + 1]! - v0y,
        e2z = pos[i2 + 2]! - v0z;

      const px = dy * e2z - dz * e2y;
      const py = dz * e2x - dx * e2z;
      const pz = dx * e2y - dy * e2x;
      const det = e1x * px + e1y * py + e1z * pz;
      if (Math.abs(det) < 1e-8) continue;
      const invDet = 1 / det;

      const tx = ox - v0x,
        ty = oy - v0y,
        tz = oz - v0z;
      const u = (tx * px + ty * py + tz * pz) * invDet;
      if (u < -1e-4 || u > 1.0001) continue;

      const qx = ty * e1z - tz * e1y;
      const qy = tz * e1x - tx * e1z;
      const qz = tx * e1y - ty * e1x;
      const v = (dx * qx + dy * qy + dz * qz) * invDet;
      if (v < -1e-4 || u + v > 1.0001) continue;

      const t = (e2x * qx + e2y * qy + e2z * qz) * invDet;
      if (t > 0.3 && t > bestT) {
        bestT = t;
        const w = 1 - u - v;
        let nx = w * nrm[i0]! + u * nrm[i1]! + v * nrm[i2]!;
        let ny = w * nrm[i0 + 1]! + u * nrm[i1 + 1]! + v * nrm[i2 + 1]!;
        let nz = w * nrm[i0 + 2]! + u * nrm[i1 + 2]! + v * nrm[i2 + 2]!;
        // Ensure normal points outward along the ray
        if (nx * dx + ny * dy + nz * dz < 0) {
          nx = -nx;
          ny = -ny;
          nz = -nz;
        }
        const nLen = Math.hypot(nx, ny, nz) || 1;
        bestNx = nx / nLen;
        bestNy = ny / nLen;
        bestNz = nz / nLen;
      }
    }
  }
  if (bestT < 0) return null;
  return {
    point: v3(ox + dx * bestT, oy + dy * bestT, oz + dz * bestT),
    normal: v3(bestNx, bestNy, bestNz),
    t: bestT,
  };
}

/** Evaluates a uniform Catmull-Rom spline at `t` in [0, 1]. */
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
  const ev = (a: number, b: number, c: number, d: number) =>
    0.5 * (2 * b + (-a + c) * u + (2 * a - 5 * b + 4 * c - d) * u2 + (-a + 3 * b - 3 * c + d) * u3);
  return v3(ev(p0.x, p1.x, p2.x, p3.x), ev(p0.y, p1.y, p2.y, p3.y), ev(p0.z, p1.z, p2.z, p3.z));
}

/**
 * Samples a vessel spec densely and snaps EVERY ring onto the actual rendered myocardial triangles,
 * building a slightly flattened, half-embedded vessel ribbon with contact-shadow AO on its underside
 * so it merges organically into the living epicardial surface.
 */
function buildEmbeddedVesselsMesh(
  id: string,
  label: string,
  color: number,
  specs: VesselSpec[],
  soups: TriSoup[],
  radialSegs = 8,
): MeshGroup {
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  const ao: number[] = [];

  for (const spec of specs) {
    if (spec.nodes.length < 2) continue;
    const origins = spec.nodes.map((n) => n.origin);
    const dirs = spec.nodes.map((n) => n.dir);
    const radiiNodes = spec.nodes.map((n) => n.r);
    const totalRings = Math.max(10, (spec.nodes.length - 1) * (spec.samplesPerSeg ?? 4) + 1);

    const hits: { p: Vec3; n: Vec3; r: number }[] = [];
    for (let s = 0; s < totalRings; s++) {
      const t = s / (totalRings - 1);
      const orig = sampleCatmullRom(origins, t);
      const dir = normalize(sampleCatmullRom(dirs, t));
      const rIdx = t * (radiiNodes.length - 1);
      const r0 = Math.floor(rIdx);
      const r1 = Math.min(radiiNodes.length - 1, r0 + 1);
      const frac = rIdx - r0;
      const r = radiiNodes[r0]! * (1 - frac) + radiiNodes[r1]! * frac;
      const hit = raycastOutermostSurface(orig, dir, soups);
      if (hit) hits.push({ p: hit.point, n: hit.normal, r });
    }
    if (hits.length < 3) continue;

    // Split into continuous epicardial chains if a ray ever jumps across an anatomical cleft (> 0.85 cm),
    // preventing floating bridge wires across base/infundibular gaps.
    const chains: { p: Vec3; n: Vec3; r: number }[][] = [];
    let current: { p: Vec3; n: Vec3; r: number }[] = [];
    for (const h of hits) {
      if (current.length > 0) {
        const prev = current[current.length - 1]!.p;
        if (Math.hypot(h.p.x - prev.x, h.p.y - prev.y, h.p.z - prev.z) > 0.85) {
          if (current.length >= 3) chains.push(current);
          current = [];
        }
      }
      current.push(h);
    }
    if (current.length >= 3) chains.push(current);

    for (const chain of chains) {
      // Gentle 1-pass longitudinal smoothing on surface normals & points to eliminate discrete facet jitter
      const smoothedP = chain.map((h) => ({ ...h.p }));
      const smoothedN = chain.map((h) => ({ ...h.n }));
      for (let i = 1; i < chain.length - 1; i++) {
        const a = chain[i - 1]!,
          b = chain[i]!,
          c = chain[i + 1]!;
        smoothedP[i] = v3(
          b.p.x * 0.6 + (a.p.x + c.p.x) * 0.2,
          b.p.y * 0.6 + (a.p.y + c.p.y) * 0.2,
          b.p.z * 0.6 + (a.p.z + c.p.z) * 0.2,
        );
        smoothedN[i] = normalize(
          v3(
            b.n.x * 0.6 + (a.n.x + c.n.x) * 0.2,
            b.n.y * 0.6 + (a.n.y + c.n.y) * 0.2,
            b.n.z * 0.6 + (a.n.z + c.n.z) * 0.2,
          ),
        );
      }

      const baseVertex = positions.length / 3;
      for (let s = 0; s < chain.length; s++) {
        const cPrev = smoothedP[Math.max(0, s - 1)]!;
        const cNext = smoothedP[Math.min(chain.length - 1, s + 1)]!;
        const T = normalize(sub(cNext, cPrev));
        const surfN = smoothedN[s]!;
        const sideRaw = cross(T, surfN);
        const sideLen = Math.hypot(sideRaw.x, sideRaw.y, sideRaw.z);
        const side = sideLen > 1e-5 ? scale(sideRaw, 1 / sideLen) : normalize(cross(T, v3(0, 0, 1)));
        const outN = normalize(cross(side, T));
        const sign = outN.x * surfN.x + outN.y * surfN.y + outN.z * surfN.z < 0 ? -1 : 1;
        const upN = scale(outN, sign);

        const endFade =
          s === chain.length - 1
            ? 0.2
            : s === chain.length - 2
              ? 0.56
              : s === 0
                ? 0.65
                : 1;
        const r = chain[s]!.r * endFade;
        // Flattened oval cross-section nestled cleanly INTO the epicardial sulcus
        const rWidth = r * 1.1;
        const rOut = r * 0.46;
        const pSurf = smoothedP[s]!;
        const cx = pSurf.x + upN.x * (r * 0.1);
        const cy = pSurf.y + upN.y * (r * 0.1);
        const cz = pSurf.z + upN.z * (r * 0.1);

        for (let k = 0; k < radialSegs; k++) {
          const ang = (k / radialSegs) * Math.PI * 2;
          const ca = Math.cos(ang);
          const sa = Math.sin(ang);
          const vx = cx + side.x * (ca * rWidth) + upN.x * (sa * rOut);
          const vy = cy + side.y * (ca * rWidth) + upN.y * (sa * rOut);
          const vz = cz + side.z * (ca * rWidth) + upN.z * (sa * rOut);
          const nx = side.x * (ca / Math.max(1e-4, rWidth)) + upN.x * (sa / Math.max(1e-4, rOut));
          const ny = side.y * (ca / Math.max(1e-4, rWidth)) + upN.y * (sa / Math.max(1e-4, rOut));
          const nz = side.z * (ca / Math.max(1e-4, rWidth)) + upN.z * (sa / Math.max(1e-4, rOut));
          const nLen = Math.hypot(nx, ny, nz) || 1;
          positions.push(vx, vy, vz);
          normals.push(nx / nLen, ny / nLen, nz / nLen);
          const contactAo = 0.62 + 0.34 * Math.max(0, sa * 0.5 + 0.5);
          ao.push(contactAo);
        }
      }

      for (let s = 0; s < chain.length - 1; s++) {
        const rowA = baseVertex + s * radialSegs;
        const rowB = baseVertex + (s + 1) * radialSegs;
        for (let k = 0; k < radialSegs; k++) {
          const kn = (k + 1) % radialSegs;
          indices.push(rowA + k, rowB + kn, rowA + kn, rowA + k, rowB + k, rowB + kn);
        }
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
 * Generates delicate, realistically tapered epicardial coronary arteries (LAD, D1, D2, LCx/OM1,
 * RCA/acute marginal) and companion cardiac veins flush-projected onto the actual smoothed
 * myocardial meshes (`lv-myocardium` + `rv-myocardium`).
 */
export function buildCoronaryMeshes(
  heart: HeartModel,
  pose: HeartPose,
  myocardiumMeshes: MeshGroup[],
): MeshGroup[] {
  const soups = prepareTriSoup(myocardiumMeshes);
  if (!soups.length) return [];

  const A = anchorsCached(heart);
  const L = heart.lv.lengthCm;
  const zAnn = pose.zAnn;
  const effL = Math.max(4.5, L - zAnn);
  const edv = Math.max(20, heart.physiology.edvMl);
  const esv = Math.max(5, heart.physiology.esvMl);
  const efScale = Math.max(0.25, Math.min(1.35, Math.max(0.12, (edv - esv) / edv) / 0.6));
  const contr = pose.state.contraction * efScale;

  // Match the systolic apical twist & shortening of `applyPhysiologicalVentricularBeat` so
  // epicardial coronaries twist and shorten in lockstep with the underlying myocardium.
  const lvNode = (frac: number, az: number, r: number) => {
    const twistRad = (((-3 + 14 * frac) * Math.PI) / 180) * contr;
    const zPos = zAnn + frac * effL - 0.85 * Math.pow(frac, 1.35) * contr;
    const twistedAz = az + twistRad;
    return {
      origin: v3(0, 0, zPos),
      dir: v3(Math.cos(twistedAz), Math.sin(twistedAz), 0),
      r,
    };
  };

  // 1. LAD — fine organic meandering path emerging below the PA trunk along the anterior IV groove
  const ladNodes: { origin: Vec3; dir: Vec3; r: number }[] = [];
  for (let i = 0; i <= 8; i++) {
    const f = 0.11 + (i / 8) * 0.80;
    const meander = 0.044 * Math.sin(f * Math.PI * 3.1) + 0.02 * Math.cos(f * Math.PI * 6.2);
    const az = RV_GROOVE_ANTERIOR_RAD - 0.04 * (1 - f) + meander;
    const r = 0.082 * (1 - 0.68 * f);
    ladNodes.push(lvNode(f, az, r));
  }

  // 2. First diagonal (D1) — acute organic bifurcation across anterolateral LV
  const d1Nodes = [
    lvNode(0.24, RV_GROOVE_ANTERIOR_RAD - 0.02, 0.064),
    lvNode(0.35, RV_GROOVE_ANTERIOR_RAD - 0.28, 0.054),
    lvNode(0.49, RV_GROOVE_ANTERIOR_RAD - 0.62, 0.044),
    lvNode(0.64, RV_GROOVE_ANTERIOR_RAD - 0.92, 0.034),
    lvNode(0.77, RV_GROOVE_ANTERIOR_RAD - 1.08, 0.024),
  ];

  // 3. Second diagonal (D2) — fine lower anterolateral twig
  const d2Nodes = [
    lvNode(0.48, RV_GROOVE_ANTERIOR_RAD + 0.02, 0.052),
    lvNode(0.60, RV_GROOVE_ANTERIOR_RAD - 0.26, 0.042),
    lvNode(0.73, RV_GROOVE_ANTERIOR_RAD - 0.52, 0.032),
    lvNode(0.84, RV_GROOVE_ANTERIOR_RAD - 0.68, 0.022),
  ];

  // 4. LCx & Obtuse Marginal (OM1) along lateral LV wall
  const lcxNodes = [
    lvNode(0.07, RV_GROOVE_ANTERIOR_RAD - 0.14, 0.078),
    lvNode(0.10, 1.04, 0.070),
    lvNode(0.13, 0.56, 0.062),
    lvNode(0.15, 0.08, 0.054),
    lvNode(0.17, -0.42, 0.044),
  ];
  const om1Nodes = [
    lvNode(0.14, 0.34, 0.056),
    lvNode(0.30, 0.20, 0.046),
    lvNode(0.48, 0.05, 0.036),
    lvNode(0.66, -0.08, 0.026),
    lvNode(0.79, -0.15, 0.020),
  ];

  // 5. RCA along right anterior AV margin + Acute Marginal branch over right anterior RV
  const rcaBaseX = A.tvCenter.x * 0.45;
  const rcaNodes = [1.88, 2.16, 2.46, 2.78, 3.1, 3.4].map((az, i, arr) => {
    const f = i / (arr.length - 1);
    return {
      origin: v3(rcaBaseX * (1 - 0.35 * f), A.tvCenter.y + 0.15, zAnn + 0.2 + f * 0.85),
      dir: normalize(v3(Math.cos(az), Math.sin(az), 0.05 * f)),
      r: 0.082 * (1 - 0.48 * f),
    };
  });
  const acuteMarginalNodes = [
    lvNode(0.16, 2.48, 0.056),
    lvNode(0.31, 2.34, 0.046),
    lvNode(0.48, 2.18, 0.036),
    lvNode(0.64, 2.02, 0.028),
    lvNode(0.76, 1.88, 0.020),
  ];

  // 6. Great Cardiac Vein — fine muted dark venous companion alongside LAD into left AV sulcus
  const gcvNodes: { origin: Vec3; dir: Vec3; r: number }[] = [];
  for (let i = 0; i <= 7; i++) {
    const f = 0.80 - (i / 7) * 0.71;
    const meander = 0.042 * Math.sin(f * Math.PI * 3.1);
    const az = RV_GROOVE_ANTERIOR_RAD - 0.14 - 0.05 * (1 - f) + meander;
    const r = 0.024 + 0.046 * (1 - f);
    gcvNodes.push(lvNode(f, az, r));
  }
  gcvNodes.push(lvNode(0.11, 0.86, 0.072), lvNode(0.14, 0.32, 0.076), lvNode(0.16, -0.22, 0.078));

  const arteries = buildEmbeddedVesselsMesh(
    'coronary-arteries',
    'Arterias coronarias',
    0x76141e,
    [
      { nodes: ladNodes, samplesPerSeg: 4 },
      { nodes: d1Nodes, samplesPerSeg: 4 },
      { nodes: d2Nodes, samplesPerSeg: 3 },
      { nodes: lcxNodes, samplesPerSeg: 3 },
      { nodes: om1Nodes, samplesPerSeg: 3 },
      { nodes: rcaNodes, samplesPerSeg: 3 },
      { nodes: acuteMarginalNodes, samplesPerSeg: 3 },
    ],
    soups,
  );
  const veins = buildEmbeddedVesselsMesh(
    'coronary-veins',
    'Venas cardíacas',
    0x273552,
    [{ nodes: gcvNodes, samplesPerSeg: 4 }],
    soups,
  );
  return [arteries, veins].filter((m) => m.indices.length > 0);
}
