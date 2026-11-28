import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import type { BeamFrame } from '@/simulator/probe/pose';

/** The transducer of the 3D navigator and the imaging sector it casts. */

/**
 * Phased-array cardiac transducer (GE M5S-D / 3S-RS proportions: 2.85 × 1.90 cm footprint).
 * Local frame: x lateral (marker side), y elevation, z ALONG THE BEAM into the patient
 * (the lens sits at z = 0 and the handle extends along −z).
 */
export function buildProbe(): { probe: THREE.Group; marker: THREE.Group } {
  const probe = new THREE.Group();
  const shell = new THREE.MeshPhysicalMaterial({
    color: 0xe1e6ec,
    roughness: 0.36,
    metalness: 0.05,
    clearcoat: 0.65,
    clearcoatRoughness: 0.24,
    envMapIntensity: 1.0,
  });
  const collar = new THREE.MeshStandardMaterial({
    color: 0x8b95a0,
    roughness: 0.42,
    metalness: 0.38,
  });
  const lens = new THREE.MeshPhysicalMaterial({
    color: 0x1a1f26,
    roughness: 0.18,
    clearcoat: 0.92,
    clearcoatRoughness: 0.12,
  });
  const rubber = new THREE.MeshStandardMaterial({ color: 0x262b32, roughness: 0.78 });

  // Head: 2.85 × 1.9 cm footprint, phased array running along x
  const head = new THREE.Mesh(new RoundedBoxGeometry(2.85, 1.9, 1.5, 4, 0.3), shell);
  head.position.z = -0.76;

  // Convex acoustic lens cap
  const capR = 2.4;
  const lensMesh = new THREE.Mesh(
    new THREE.SphereGeometry(capR, 32, 12, 0, Math.PI * 2, 0, Math.asin(1.36 / capR)),
    lens,
  );
  lensMesh.rotation.x = Math.PI / 2;
  lensMesh.scale.set(1, 1, 0.66);
  lensMesh.position.z = -capR + 0.04;

  // Shoulder transition into ergonomic barrel
  const shoulder = new THREE.Mesh(
    new THREE.LatheGeometry(
      [
        new THREE.Vector2(1.4, 0),
        new THREE.Vector2(1.34, 0.3),
        new THREE.Vector2(1.2, 0.7),
        new THREE.Vector2(1.04, 1.15),
      ],
      28,
    ),
    shell,
  );
  shoulder.rotation.x = -Math.PI / 2;
  shoulder.scale.set(1, 1, 0.72);
  shoulder.position.z = -1.5;

  // Ergonomic barrel with finger waist
  const rs = [1.04, 1.12, 1.1, 1.0, 0.93, 0.93, 1.0, 1.06, 1.02, 0.86];
  const profile = rs.map((r, i) => new THREE.Vector2(r, (i / (rs.length - 1)) * 7.4));
  profile.push(new THREE.Vector2(0.6, 7.5), new THREE.Vector2(0, 7.56));
  const handle = new THREE.Mesh(new THREE.LatheGeometry(profile, 32), shell);
  handle.rotation.x = -Math.PI / 2;
  handle.scale.set(1, 1, 0.74);
  handle.position.z = -2.65;

  // Tapered, ribbed strain relief
  const relief = new THREE.Group();
  for (let i = 0; i < 5; i++) {
    const t = i / 4;
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.62 - t * 0.28, 0.12 - t * 0.04, 8, 20),
      rubber,
    );
    ring.position.z = -(10.15 + t * 1.15);
    relief.add(ring);
  }
  const reliefCone = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.68, 1.35, 20), rubber);
  reliefCone.rotation.x = -Math.PI / 2;
  reliefCone.position.z = -10.75;
  relief.add(reliefCone);

  const band = new THREE.Mesh(new THREE.CylinderGeometry(0.9, 0.86, 0.32, 24), collar);
  band.rotation.x = Math.PI / 2;
  band.scale.set(1, 1, 0.78);
  band.position.z = -9.95;

  // Extended, naturally draped cord curving gracefully out of frame instead of ending bluntly mid-air
  const cable = new THREE.Mesh(
    new THREE.TubeGeometry(
      new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, 0, -11.35),
        new THREE.Vector3(0.3, 0.9, -14.2),
        new THREE.Vector3(1.6, 3.8, -18.5),
        new THREE.Vector3(4.5, 9.5, -24.5),
        new THREE.Vector3(9.5, 17.5, -31.0),
      ]),
      28,
      0.27,
      10,
      false,
    ),
    rubber,
  );

  // Index marker (+x, screen-right orientation): pickable handle + luminous cyan 360°-readable accent
  const marker = new THREE.Group();
  const markerMat = new THREE.MeshStandardMaterial({
    color: 0x38bdf8,
    emissive: 0x0c4a6e,
    emissiveIntensity: 0.75,
    roughness: 0.3,
  });
  const ridge = new THREE.Mesh(new RoundedBoxGeometry(0.32, 0.72, 1.15, 2, 0.12), markerMat);
  ridge.position.set(1.44, 0, -0.82);
  const dot = new THREE.Mesh(
    new THREE.SphereGeometry(0.3, 16, 12),
    new THREE.MeshStandardMaterial({
      color: 0x7dd3fc,
      emissive: 0x0284c7,
      emissiveIntensity: 0.9,
      roughness: 0.25,
    }),
  );
  dot.position.set(1.34, 0, -2.15);
  // Subtle wrap-around cyan collar stripe on the +x hemisphere of the barrel so orientation reads from straight-on views
  const stripe = new THREE.Mesh(
    new THREE.TorusGeometry(1.06, 0.085, 8, 24, Math.PI * 0.58),
    markerMat,
  );
  stripe.scale.set(1, 0.75, 1);
  stripe.rotation.z = -Math.PI * 0.29;
  stripe.position.z = -2.85;
  marker.add(ridge, dot, stripe);

  const seam = new THREE.Mesh(new THREE.TorusGeometry(1.38, 0.035, 8, 48), collar);
  seam.scale.set(1, 0.72, 1);
  seam.position.z = -1.5;

  probe.add(head, lensMesh, shoulder, handle, relief, band, cable, marker, seam);
  return { probe, marker };
}

/**
 * Updates the 3D ultrasound imaging fan with a vertex-colored radial falloff and
 * orientation-coded border lines (bright sky-cyan on the +lateral index-marker side,
 * cooler slate-cyan on the opposite side and far arc).
 */
export function updateFan(
  fan: THREE.Mesh,
  edges: THREE.LineSegments,
  beam: BeamFrame,
  depth: number,
  sectorDeg: number,
): void {
  const half = (sectorDeg * Math.PI) / 360;
  const o = beam.origin;
  const n = 20;
  const at = (a: number, r: number) => {
    const c = Math.cos(a),
      s = Math.sin(a);
    return [
      o.x + (beam.forward.x * c + beam.lateral.x * s) * r,
      o.y + (beam.forward.y * c + beam.lateral.y * s) * r,
      o.z + (beam.forward.z * c + beam.lateral.z * s) * r,
    ] as const;
  };

  // Multi-ring fan so vertex colors create a crisp near-to-far cyan beam gradient
  const rings = 4;
  const verts: number[] = [];
  const cols: number[] = [];
  const pushCol = (rFrac: number, aFrac: number) => {
    // Brighter cyan near the probe and along the +lateral marker edge, fading gently at max depth
    const markerBias = 0.5 + 0.5 * aFrac; // 0 on −lat, 1 on +lat (marker side)
    const bright = (1 - 0.58 * rFrac) * (0.72 + 0.28 * markerBias);
    cols.push(0.22 * bright, 0.74 * bright, 0.97 * bright);
  };

  for (let ri = 0; ri < rings; ri++) {
    const r0 = (depth * ri) / rings;
    const r1 = (depth * (ri + 1)) / rings;
    const f0 = ri / rings;
    const f1 = (ri + 1) / rings;
    for (let i = 0; i < n; i++) {
      const a0 = -half + (2 * half * i) / n;
      const a1 = -half + (2 * half * (i + 1)) / n;
      const af0 = -1 + (2 * i) / n;
      const af1 = -1 + (2 * (i + 1)) / n;
      const p00 = at(a0, r0);
      const p10 = at(a1, r0);
      const p01 = at(a0, r1);
      const p11 = at(a1, r1);
      if (ri === 0) {
        verts.push(o.x, o.y, o.z, ...p01, ...p11);
        pushCol(0, 0.5 * (af0 + af1));
        pushCol(f1, af0);
        pushCol(f1, af1);
      } else {
        verts.push(...p00, ...p01, ...p11, ...p00, ...p11, ...p10);
        pushCol(f0, af0);
        pushCol(f1, af0);
        pushCol(f1, af1);
        pushCol(f0, af0);
        pushCol(f1, af1);
        pushCol(f0, af1);
      }
    }
  }

  fan.geometry.dispose();
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  fan.geometry = g;

  // Color-coded fan borders: +lateral (marker side) bright electric sky-blue, −lateral subtle slate-cyan
  const ev: number[] = [];
  const ec: number[] = [];
  const l = at(-half, depth);
  const r = at(half, depth);
  const mid = at(0, depth);
  // −lateral border (dimmer slate-blue)
  ev.push(o.x, o.y, o.z, ...l);
  ec.push(0.45, 0.66, 0.82, 0.28, 0.44, 0.58);
  // +lateral border (bright index-marker cyan!)
  ev.push(o.x, o.y, o.z, ...r);
  ec.push(0.35, 0.85, 1.0, 0.16, 0.72, 0.98);
  // Subtle centerline beam ray
  ev.push(o.x, o.y, o.z, ...mid);
  ec.push(0.3, 0.72, 0.9, 0.12, 0.35, 0.48);
  // Far depth arc
  for (let i = 0; i < n; i++) {
    const p0 = at(-half + (2 * half * i) / n, depth);
    const p1 = at(-half + (2 * half * (i + 1)) / n, depth);
    const f = i / n;
    ev.push(...p0, ...p1);
    ec.push(0.28 + 0.12 * f, 0.54 + 0.22 * f, 0.72 + 0.24 * f, 0.28 + 0.12 * f, 0.54 + 0.22 * f, 0.72 + 0.24 * f);
  }
  edges.geometry.dispose();
  const eg = new THREE.BufferGeometry();
  eg.setAttribute('position', new THREE.Float32BufferAttribute(ev, 3));
  eg.setAttribute('color', new THREE.Float32BufferAttribute(ec, 3));
  edges.geometry = eg;
}
