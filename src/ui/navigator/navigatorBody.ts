import * as THREE from 'three';
import {
  ribCenterY,
  ribDepth,
  ribRadiusAt,
  skinNormal,
  skinZ,
  type ThoraxModel,
} from '@/simulator/anatomy/thoraxModel';
import { heartToTorso } from '@/simulator/anatomy/heartFrame';
import type { NavigatorModel, WindowMark } from '@/workers/heartMesh.worker';

/**
 * The body of the 3D navigator, built once per case from the navigator model the mesh worker sends:
 * translucent silhouette skin, sculpted anatomical sternum + flattened ribbon ribs with a true
 * subcostal arch (ribs 8–10 converging onto the 7th costal cartilage), deduplicated acoustic window
 * rings, and the schematic heart that stands in until the surfaces arrive.
 */

/** Superellipse cross-section point at angle θ (0 = front centre), scaled inward by `inset` cm. */
function crossSection(
  t: ThoraxModel,
  theta: number,
  y: number,
  inset: number,
): { x: number; z: number } {
  const n = t.n;
  const aw = t.aw - inset;
  const b = t.bDepth - inset;
  const s = Math.sin(theta),
    c = Math.cos(theta);
  const x = aw * Math.sign(s) * Math.pow(Math.abs(s), 2 / n);
  let z = -t.bDepth + b * Math.sign(c) * Math.pow(Math.abs(c), 2 / n);
  if (y > 8) z -= 0.12 * (y - 8) * (y - 8);
  if (y < -8) z -= 0.18 * (-8 - y);
  return { x, z };
}

/**
 * Translucent anterior & lateral thorax shell. Uses the exact `crossSection` surface so pointer
 * raycasting (`pickSkin`) stays 100% aligned with the acoustic skin model, while keeping the
 * neck & shoulders softly translucent so they never form a harsh opaque cut band at the top.
 */
export function buildSkin(t: ThoraxModel): THREE.Mesh {
  const nT = 64,
    nY = 44;
  const thMax = 1.45; // ~83° each side: anterior and lateral chest
  const ys = -19,
    ye = 12.6;
  const pos: number[] = [];
  const idx: number[] = [];
  for (let j = 0; j <= nY; j++) {
    const y = ys + ((ye - ys) * j) / nY;
    const taper = 1 - 0.06 * Math.max(0, (-y - 6) / 10) - 0.03 * Math.max(0, (y - 9) / 4);
    for (let i = 0; i <= nT; i++) {
      const th = -thMax + (2 * thMax * i) / nT;
      const p = crossSection(t, th, y, 0);
      pos.push(p.x * taper, y, p.z);
    }
  }
  for (let j = 0; j < nY; j++)
    for (let i = 0; i < nT; i++) {
      const a = j * (nT + 1) + i;
      idx.push(a, a + 1, a + nT + 1, a + 1, a + nT + 2, a + nT + 1);
    }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();

  // Fresnel silhouette skin: completely crystal-clear facing the camera over the heart (zero haze/veil),
  // softly visible along the torso flanks, shoulders, and neck contour.
  const skinUniforms = {
    uSkinCenterAlpha: { value: 0.0 },
    uSkinRimAlpha: { value: 0.36 },
  };
  const applyFresnelSkin = (m: THREE.MeshStandardMaterial) => {
    m.customProgramCacheKey = () => 'fresnel-torso-skin';
    m.onBeforeCompile = (shader) => {
      shader.uniforms['uSkinCenterAlpha'] = skinUniforms.uSkinCenterAlpha;
      shader.uniforms['uSkinRimAlpha'] = skinUniforms.uSkinRimAlpha;
      shader.fragmentShader = shader.fragmentShader
        .replace(
          '#include <common>',
          `#include <common>\nuniform float uSkinCenterAlpha;\nuniform float uSkinRimAlpha;`,
        )
        .replace(
          '#include <alphamap_fragment>',
          `#include <alphamap_fragment>
          float skinNdotV = abs(dot(normalize(vNormal), normalize(vViewPosition)));
          float skinRim = pow(1.0 - skinNdotV, 2.35);
          diffuseColor.a = mix(uSkinCenterAlpha, uSkinRimAlpha, skinRim);`,
        );
    };
  };

  const mat = new THREE.MeshStandardMaterial({
    color: 0xc2ab9a,
    roughness: 0.72,
    metalness: 0,
    transparent: true,
    opacity: 1,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  applyFresnelSkin(mat);
  const mesh = new THREE.Mesh(g, mat);
  mesh.renderOrder = 2;
  mesh.userData['skinUniforms'] = skinUniforms;

  const collarMat = new THREE.MeshStandardMaterial({
    color: 0xb69f8f,
    roughness: 0.75,
    metalness: 0,
    transparent: true,
    opacity: 1,
    depthWrite: false,
  });
  applyFresnelSkin(collarMat);
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(4.6, 5.8, 7.5, 28, 1, true), collarMat);
  neck.position.set(0, 15.8, -t.bDepth * 0.56);
  neck.userData['skinAccessory'] = true;
  mesh.add(neck);
  for (const side of [-1, 1]) {
    const sh = new THREE.Mesh(new THREE.SphereGeometry(5.2, 24, 16), collarMat);
    sh.scale.set(1.32, 0.78, 0.92);
    sh.position.set(side * (t.aw + 1.1), 10.2, -t.bDepth * 0.62);
    sh.userData['skinAccessory'] = true;
    mesh.add(sh);
  }
  return mesh;
}

/** Syncs Fresnel skin center & contour opacity cleanly between anatomy-visible and exam modes. */
export function updateSkinOpacity(skin: THREE.Mesh, centerOpacity: number, rimOpacity?: number): void {
  const u = skin.userData['skinUniforms'] as
    | { uSkinCenterAlpha: { value: number }; uSkinRimAlpha: { value: number } }
    | undefined;
  if (u) {
    u.uSkinCenterAlpha.value = centerOpacity;
    u.uSkinRimAlpha.value = rimOpacity ?? Math.min(0.75, centerOpacity + 0.36);
  }
}

/** Colour of each acoustic window's marks: the same family the console uses for the view presets. */
export const WINDOW_COLORS: Record<WindowMark['window'], number> = {
  parasternal: 0xf59e0b,
  apical: 0x0ea5e9,
  subcostal: 0x10b981,
  suprasternal: 0xec4899,
};

export const WINDOW_LABELS: Record<WindowMark['window'], string> = {
  parasternal: 'Paraesternal',
  apical: 'Apical',
  subcostal: 'Subcostal',
  suprasternal: 'Supraesternal',
};

export interface WindowBadgeAnchor {
  window: WindowMark['window'];
  label: string;
  colorHex: string;
  /** World-space torso position of the window centroid on the skin (cm). */
  anchor: THREE.Vector3;
  /** Outward skin normal at the anchor. */
  normal: THREE.Vector3;
}

export interface WindowMarksRig {
  group: THREE.Group;
  badges: WindowBadgeAnchor[];
  /** Highlights the acoustic window closest to the current probe tip. */
  setActiveProbe: (pos: { x: number; y: number; z: number }) => void;
}

/**
 * Clean, deduplicated target rings on the skin for each canonical acoustic window, plus
 * projected badge anchors rendered by `navigatorScene` with safe viewport clamping so labels
 * never collide with the probe or clip off the panel edges.
 */
export function buildWindowMarks(t: ThoraxModel, marks: WindowMark[]): WindowMarksRig {
  const g = new THREE.Group();
  const placed: {
    window: WindowMark['window'];
    u: number;
    v: number;
    pos: THREE.Vector3;
    ringMat: THREE.MeshBasicMaterial;
    haloMat: THREE.MeshBasicMaterial;
  }[] = [];

  for (const m of marks) {
    // Merge closely spaced sub-views of the same window (e.g. subcostal-4c & subcostal-ivc, plax & psax)
    if (placed.some((p) => p.window === m.window && Math.hypot(p.u - m.u, p.v - m.v) < 1.35))
      continue;
    const color = WINDOW_COLORS[m.window];
    const ringMat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.78,
      side: THREE.DoubleSide,
      depthWrite: false,
      toneMapped: false,
    });
    const haloMat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.18,
      side: THREE.DoubleSide,
      depthWrite: false,
      toneMapped: false,
    });
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.42, 0.52, 40), ringMat);
    const halo = new THREE.Mesh(new THREE.RingGeometry(0.56, 0.74, 40), haloMat);
    halo.renderOrder = 3;
    ring.add(halo);
    const n = skinNormal(t, m.u, m.v);
    const z = skinZ(t, m.u, m.v);
    const pos = new THREE.Vector3(m.u + n.x * 0.08, m.v + n.y * 0.08, z + n.z * 0.08);
    ring.position.copy(pos);
    ring.lookAt(m.u + n.x, m.v + n.y, z + n.z);
    ring.renderOrder = 3;
    g.add(ring);
    placed.push({ window: m.window, u: m.u, v: m.v, pos, ringMat, haloMat });
  }

  // One projected badge anchor per distinct window
  const byWindow = new Map<WindowMark['window'], { u: number; v: number; n: number }>();
  for (const p of placed) {
    const acc = byWindow.get(p.window) ?? { u: 0, v: 0, n: 0 };
    acc.u += p.u;
    acc.v += p.v;
    acc.n++;
    byWindow.set(p.window, acc);
  }
  const badges: WindowBadgeAnchor[] = [];
  for (const [w, acc] of byWindow) {
    const u = acc.u / acc.n;
    const v = acc.v / acc.n;
    const n = skinNormal(t, u, v);
    const z = skinZ(t, u, v);
    badges.push({
      window: w,
      label: WINDOW_LABELS[w],
      colorHex: `#${WINDOW_COLORS[w].toString(16).padStart(6, '0')}`,
      anchor: new THREE.Vector3(u + n.x * 0.25, v + n.y * 0.25, z + n.z * 0.25),
      normal: new THREE.Vector3(n.x, n.y, n.z),
    });
  }

  const setActiveProbe = (pos: { x: number; y: number; z: number }) => {
    let bestIdx = -1;
    let bestD = 3.2;
    for (let i = 0; i < placed.length; i++) {
      const p = placed[i]!.pos;
      const d = Math.hypot(p.x - pos.x, p.y - pos.y, p.z - pos.z);
      if (d < bestD) {
        bestD = d;
        bestIdx = i;
      }
    }
    for (let i = 0; i < placed.length; i++) {
      const active = i === bestIdx;
      placed[i]!.ringMat.opacity = active ? 0.98 : 0.56;
      placed[i]!.haloMat.opacity = active ? 0.42 : 0.12;
    }
  };

  return { group: g, badges, setActiveProbe };
}

/** A small text label that always faces the camera (canvas texture on a sprite, used by review markers). */
export function textSprite(text: string, color: number): THREE.Sprite {
  const canvas = document.createElement('canvas');
  const scale = 2;
  const font = `600 ${11 * scale}px system-ui, -apple-system, sans-serif`;
  const ctx = canvas.getContext('2d');
  const textW = ctx ? ((ctx.font = font), ctx.measureText(text).width) : 80;
  const dotR = 3 * scale;
  const dotX = 7 * scale + dotR;
  const textX = dotX + dotR + 4 * scale;
  const w = Math.ceil(textX + textW + 7 * scale);
  const h = 20 * scale;
  canvas.width = w;
  canvas.height = h;
  if (ctx) {
    ctx.font = font;
    const r = 5 * scale;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') {
      ctx.roundRect(0, 0, w, h, r);
    } else {
      ctx.rect(0, 0, w, h);
    }
    ctx.fill();
    ctx.lineWidth = 1.2 * scale;
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.42)';
    ctx.stroke();

    ctx.fillStyle = `#${color.toString(16).padStart(6, '0')}`;
    ctx.beginPath();
    ctx.arc(dotX, h / 2, dotR, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#f8fafc';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, textX, h / 2);
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: tex,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      toneMapped: false,
    }),
  );
  sprite.scale.set((canvas.width / canvas.height) * 1.1, 1.1, 1);
  return sprite;
}

/**
 * Builds a flattened anatomical ribbon-tube geometry along `curve`, oriented so its wider
 * vertical blade (`halfHeight`) lies tangent to the chest wall and its thinner profile
 * (`halfThickness`) points anteroposteriorly. Replaces circular "noodle" tubes with realistic ribs.
 */
function buildEllipticalTubeGeometry(
  curve: THREE.CatmullRomCurve3,
  tubularSegments: number,
  halfHeight: number,
  halfThickness: number,
  radialSegments = 10,
): THREE.BufferGeometry {
  const pts = curve.getPoints(tubularSegments);
  const positions: number[] = [];
  const indices: number[] = [];
  const up = new THREE.Vector3(0, 1, 0);
  const tangent = new THREE.Vector3();
  const radialOut = new THREE.Vector3();
  const bladeUp = new THREE.Vector3();

  for (let i = 0; i <= tubularSegments; i++) {
    const p = pts[i]!;
    const prev = pts[Math.max(0, i - 1)]!;
    const next = pts[Math.min(tubularSegments, i + 1)]!;
    tangent.subVectors(next, prev).normalize();
    // Vertical blade axis projected orthogonal to the rib tangent
    bladeUp.copy(up).addScaledVector(tangent, -up.dot(tangent));
    if (bladeUp.lengthSq() < 1e-4) bladeUp.set(0, 0, 1);
    bladeUp.normalize();
    // Outward chest-wall normal orthogonal to both tangent and bladeUp
    radialOut.crossVectors(tangent, bladeUp).normalize();
    // Gently taper ends so sternocostal / costal-arch joints sit neatly without blunt flat caps
    const endTaper =
      i === 0 || i === tubularSegments ? 0.62 : i === 1 || i === tubularSegments - 1 ? 0.9 : 1;
    const hh = halfHeight * endTaper;
    const ht = halfThickness * endTaper;

    for (let j = 0; j < radialSegments; j++) {
      const a = (j / radialSegments) * Math.PI * 2;
      const ca = Math.cos(a);
      const sa = Math.sin(a);
      positions.push(
        p.x + bladeUp.x * ca * hh + radialOut.x * sa * ht,
        p.y + bladeUp.y * ca * hh + radialOut.y * sa * ht,
        p.z + bladeUp.z * ca * hh + radialOut.z * sa * ht,
      );
    }
  }

  for (let i = 0; i < tubularSegments; i++) {
    const rowA = i * radialSegments;
    const rowB = (i + 1) * radialSegments;
    for (let j = 0; j < radialSegments; j++) {
      const jn = (j + 1) % radialSegments;
      indices.push(rowA + j, rowB + j, rowB + jn, rowA + j, rowB + jn, rowA + jn);
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geom.setIndex(indices);
  geom.computeVertexNormals();
  return geom;
}

/**
 * Sculpts the sternum (manubrium with suprasternal notch, contoured sternal body, and
 * tapered xiphoid process) extruded with soft bevels and curved gently along the anterior chest.
 */
function buildSculptedSternum(
  t: ThoraxModel,
  sternumMat: THREE.MeshPhysicalMaterial,
  xiphoidMat: THREE.MeshPhysicalMaterial,
): THREE.Group {
  const group = new THREE.Group();
  const sz = (y: number) => skinZ(t, 0, y) - t.chestWall * 0.34 - 0.38;

  const curveGeometryToChest = (geom: THREE.BufferGeometry, yOffset: number) => {
    const pos = geom.getAttribute('position') as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      const yLocal = pos.getY(i);
      const yWorld = yLocal + yOffset;
      const zLocal = pos.getZ(i);
      const xLocal = pos.getX(i);
      // Slight lateral curvature matching the anterior superellipse + vertical sternal angle
      const chestZ = sz(yWorld) - 0.03 * xLocal * xLocal;
      pos.setXYZ(i, xLocal, yWorld, chestZ + zLocal);
    }
    pos.needsUpdate = true;
    geom.computeVertexNormals();
  };

  // 1. Manubrium (y: 6.1 to 10.2) — shield profile with concave jugular (suprasternal) notch
  const manShape = new THREE.Shape();
  manShape.moveTo(-1.35, -2.05);
  manShape.quadraticCurveTo(-1.55, -0.6, -2.45, 1.05);
  manShape.quadraticCurveTo(-2.35, 1.85, -1.35, 2.02);
  // Concave jugular notch at top center
  manShape.quadraticCurveTo(0, 1.48, 1.35, 2.02);
  manShape.quadraticCurveTo(2.35, 1.85, 2.45, 1.05);
  manShape.quadraticCurveTo(1.55, -0.6, 1.35, -2.05);
  manShape.quadraticCurveTo(0, -2.18, -1.35, -2.05);

  const manGeom = new THREE.ExtrudeGeometry(manShape, {
    depth: 0.42,
    bevelEnabled: true,
    bevelThickness: 0.22,
    bevelSize: 0.18,
    bevelSegments: 4,
    curveSegments: 16,
  });
  manGeom.translate(0, 0, -0.21);
  curveGeometryToChest(manGeom, 8.2);
  const manMesh = new THREE.Mesh(manGeom, sternumMat);
  manMesh.renderOrder = 1;
  group.add(manMesh);

  // 2. Sternal body / gladiolus (y: -4.1 to 6.0) — gently waisted superiorly, widening at 4th–5th ribs, tapering distally
  const bodyShape = new THREE.Shape();
  bodyShape.moveTo(-1.25, 5.0);
  bodyShape.bezierCurveTo(-1.18, 2.8, -1.48, -0.4, -1.52, -2.2);
  bodyShape.quadraticCurveTo(-1.42, -4.1, -0.88, -5.05);
  bodyShape.quadraticCurveTo(0, -5.22, 0.88, -5.05);
  bodyShape.quadraticCurveTo(1.42, -4.1, 1.52, -2.2);
  bodyShape.bezierCurveTo(1.48, -0.4, 1.18, 2.8, 1.25, 5.0);
  bodyShape.quadraticCurveTo(0, 5.15, -1.25, 5.0);

  const bodyGeom = new THREE.ExtrudeGeometry(bodyShape, {
    depth: 0.36,
    bevelEnabled: true,
    bevelThickness: 0.2,
    bevelSize: 0.16,
    bevelSegments: 4,
    curveSegments: 16,
  });
  bodyGeom.translate(0, 0, -0.18);
  curveGeometryToChest(bodyGeom, 1.0);
  const bodyMesh = new THREE.Mesh(bodyGeom, sternumMat);
  bodyMesh.renderOrder = 1;
  group.add(bodyMesh);

  // 3. Xiphoid process (y: -6.5 to -4.15) — delicate tapered cartilaginous blade
  const xiphShape = new THREE.Shape();
  xiphShape.moveTo(-0.56, 1.15);
  xiphShape.quadraticCurveTo(-0.62, 0.1, -0.34, -0.95);
  xiphShape.quadraticCurveTo(0, -1.3, 0.34, -0.95);
  xiphShape.quadraticCurveTo(0.62, 0.1, 0.56, 1.15);
  xiphShape.closePath();

  const xiphGeom = new THREE.ExtrudeGeometry(xiphShape, {
    depth: 0.22,
    bevelEnabled: true,
    bevelThickness: 0.12,
    bevelSize: 0.1,
    bevelSegments: 3,
    curveSegments: 12,
  });
  xiphGeom.translate(0, 0, -0.11);
  curveGeometryToChest(xiphGeom, -5.3);
  const xiphMesh = new THREE.Mesh(xiphGeom, xiphoidMat);
  xiphMesh.renderOrder = 1;
  group.add(xiphMesh);

  return group;
}

/**
 * Anatomical rib cage: sculpted translucent sternum, flattened elliptical ribbon ribs (1–10),
 * clean sternocostal cartilages (1–7), a true converging subcostal arch (ribs 8–10 sweeping
 * upward to join the 7th costal cartilage), and doubly-curved S-shaped clavicles.
 */
export function buildSkeleton(t: ThoraxModel): THREE.Group {
  const g = new THREE.Group();
  const boneMat = new THREE.MeshPhysicalMaterial({
    color: 0xe6dfd0,
    roughness: 0.42,
    metalness: 0.03,
    clearcoat: 0.22,
    clearcoatRoughness: 0.35,
    transparent: true,
    opacity: 0.48,
    depthWrite: false,
  });
  const sternumMat = new THREE.MeshPhysicalMaterial({
    color: 0xe3dccb,
    roughness: 0.4,
    metalness: 0.03,
    clearcoat: 0.25,
    clearcoatRoughness: 0.32,
    transparent: true,
    opacity: 0.34,
    depthWrite: false,
  });
  const cartilageMat = new THREE.MeshPhysicalMaterial({
    color: 0xbdd1e6,
    roughness: 0.36,
    metalness: 0.02,
    clearcoat: 0.35,
    clearcoatRoughness: 0.28,
    transparent: true,
    opacity: 0.34,
    depthWrite: false,
  });
  g.userData['skeletonMats'] = { boneMat, sternumMat, cartilageMat };

  const depth = ribDepth(t);
  const thMax = 1.42;

  // Store 7th costal cartilage anchor points per side so false ribs 8–10 converge into a real costal arch
  const archTargetBySide = new Map<number, { x: number; y: number; z: number }[]>();

  for (let k = 1; k <= 10; k++) {
    for (const side of [-1, 1]) {
      const yFront = ribCenterY(t, k, 1.8);
      // Match sculpted sternum lateral border for true ribs 1–7
      const sternalHalfW = k === 1 ? 2.15 : k === 2 ? 1.32 : k <= 5 ? 1.48 : k === 6 ? 1.25 : 0.95;
      const cartSplitX = k <= 5 ? 5.4 : k === 6 ? 6.2 : k === 7 ? 7.2 : 7.8 + (k - 7) * 1.1;

      const bonePts: THREE.Vector3[] = [];
      const cartPts: THREE.Vector3[] = [];

      if (k <= 7) {
        // Bony lateral-to-anterior rib shaft
        for (let i = 0; i <= 36; i++) {
          const th = (thMax * i) / 36;
          const p = crossSection(t, side * th, yFront, depth);
          if (Math.abs(p.x) < cartSplitX - 0.5) continue;
          const y = yFront + 3.6 * Math.pow(th / thMax, 1.35);
          bonePts.push(new THREE.Vector3(p.x, y, p.z));
        }
        // Costal cartilage sweeping smoothly from bone junction into sternal facet
        // Ribs 6 & 7 angle gently superiorly as they approach the lower sternal body
        const sternalLift = k === 6 ? 0.75 : k === 7 ? 1.65 : k === 5 ? 0.25 : 0;
        const cSteps = 12;
        for (let i = 0; i <= cSteps; i++) {
          const u = i / cSteps; // 0 = sternal attachment, 1 = costochondral junction
          const xAbs = sternalHalfW + (cartSplitX - sternalHalfW) * u;
          const th = Math.asin(Math.min(0.98, xAbs / Math.max(1, t.aw - depth)));
          const p = crossSection(t, side * th, yFront, depth);
          const yCurve =
            yFront +
            3.6 * Math.pow(th / thMax, 1.35) +
            sternalLift * Math.pow(1 - u, 1.4);
          const pt = new THREE.Vector3(p.x, yCurve, p.z + 0.08 * (1 - u));
          cartPts.push(pt);
        }
        if (k === 7) archTargetBySide.set(side, cartPts.map((v) => ({ x: v.x, y: v.y, z: v.z })));
      } else {
        // False ribs 8–10: bony shaft laterally + cartilage curving UPWARD to fuse into the costal arch!
        for (let i = 0; i <= 36; i++) {
          const th = (thMax * i) / 36;
          const p = crossSection(t, side * th, yFront, depth);
          if (Math.abs(p.x) < cartSplitX - 0.4) continue;
          const y = yFront + 3.6 * Math.pow(th / thMax, 1.35);
          bonePts.push(new THREE.Vector3(p.x, y, p.z));
        }
        // Converge anterior cartilage smoothly along the chest curvature onto the 7th costal cartilage
        const arch7 = archTargetBySide.get(side);
        if (arch7 && bonePts.length > 0) {
          const startPt = bonePts[0]!;
          const targetIdx = Math.min(arch7.length - 1, 4 + (k - 8) * 3);
          const endRef = arch7[targetIdx]!;
          const cSteps = 6;
          for (let ci = 0; ci <= cSteps; ci++) {
            const u = ci / cSteps; // 0 = merge into 7th cartilage, 1 = bony costochondral junction
            // Smooth cubic blend so cartilage sweeps organically upward without an elbow kink
            const s = u * u * (3 - 2 * u);
            const x = endRef.x + (startPt.x - endRef.x) * u;
            const yBase = endRef.y - 0.14 * (k - 7) * (1 - u);
            const y = yBase + (startPt.y - yBase) * Math.pow(s, 0.82);
            const th = Math.asin(Math.min(0.98, Math.abs(x) / Math.max(1, t.aw - depth)));
            const surf = crossSection(t, side * th, y, depth);
            cartPts.push(new THREE.Vector3(x, y, surf.z + 0.04 * (1 - u)));
          }
        }
      }

      const baseR = ribRadiusAt(t, 5);
      const halfH = baseR * 1.12;
      const halfT = baseR * 0.52;
      if (bonePts.length >= 3) {
        const mesh = new THREE.Mesh(
          buildEllipticalTubeGeometry(new THREE.CatmullRomCurve3(bonePts), 36, halfH, halfT, 10),
          boneMat,
        );
        mesh.renderOrder = 1;
        g.add(mesh);
      }
      if (cartPts.length >= 3) {
        const mesh = new THREE.Mesh(
          buildEllipticalTubeGeometry(
            new THREE.CatmullRomCurve3(cartPts),
            14,
            halfH * 0.96,
            halfT * 0.96,
            10,
          ),
          cartilageMat,
        );
        mesh.renderOrder = 1;
        g.add(mesh);
      }
    }
  }

  // Sculpted anatomical sternum
  g.add(buildSculptedSternum(t, sternumMat, cartilageMat));

  // Doubly-curved S-shaped anatomical clavicles meeting the manubrium's clavicular notches
  const szTop = skinZ(t, 0, 10.2) - t.chestWall * 0.32 - 0.15;
  for (const side of [-1, 1]) {
    const pts = [
      new THREE.Vector3(side * 1.65, 10.15, szTop + 0.35),
      new THREE.Vector3(side * 5.2, 10.75, szTop + 0.55), // medial anterior convexity
      new THREE.Vector3(side * 9.8, 11.45, szTop - 1.15), // lateral anterior concavity
      new THREE.Vector3(side * 13.8, 11.95, szTop - 3.4),
    ];
    const clav = new THREE.Mesh(
      buildEllipticalTubeGeometry(new THREE.CatmullRomCurve3(pts), 22, 0.44, 0.36, 10),
      boneMat,
    );
    clav.renderOrder = 1;
    g.add(clav);
  }

  return g;
}

/**
 * Switches skeleton materials smoothly between translucent X-ray atlas glass (when heart anatomy is
 * visible) and solid bone landmarks (in Exam mode, where ribs & sternum are the sole guide).
 */
export function updateSkeletonStyle(skeleton: THREE.Group, xRay: boolean): void {
  const mats = skeleton.userData['skeletonMats'] as
    | {
        boneMat: THREE.MeshPhysicalMaterial;
        sternumMat: THREE.MeshPhysicalMaterial;
        cartilageMat: THREE.MeshPhysicalMaterial;
      }
    | undefined;
  if (!mats) return;
  mats.boneMat.opacity = xRay ? 0.48 : 0.88;
  mats.boneMat.depthWrite = !xRay;
  mats.sternumMat.opacity = xRay ? 0.32 : 0.9;
  mats.sternumMat.depthWrite = !xRay;
  mats.cartilageMat.opacity = xRay ? 0.34 : 0.82;
  mats.cartilageMat.depthWrite = !xRay;
}

export function buildHeartGhost(model: NavigatorModel): THREE.Group {
  const g = new THREE.Group();
  const f = model.frame;
  const basis = new THREE.Matrix4().makeBasis(
    new THREE.Vector3(f.ex.x, f.ex.y, f.ex.z),
    new THREE.Vector3(f.ey.x, f.ey.y, f.ey.z),
    new THREE.Vector3(f.ez.x, f.ez.y, f.ez.z),
  );
  for (const p of model.ghost) {
    const mat = new THREE.MeshStandardMaterial({
      color: p.color,
      transparent: true,
      opacity: p.opacity,
      roughness: 0.6,
      depthWrite: false,
    });
    if (p.kind === 'ellipsoid') {
      const m = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), mat);
      const c = heartToTorso(f, p.center);
      m.position.set(c.x, c.y, c.z);
      m.quaternion.setFromRotationMatrix(basis);
      m.scale.set(p.radii.x, p.radii.y, p.radii.z);
      g.add(m);
    } else if (p.end) {
      const a = heartToTorso(f, p.center),
        b = heartToTorso(f, p.end);
      const dir = new THREE.Vector3(b.x - a.x, b.y - a.y, b.z - a.z);
      const len = dir.length();
      const m = new THREE.Mesh(
        new THREE.CylinderGeometry(p.radii.x, p.radii.x * 0.9, len, 16),
        mat,
      );
      m.position.set((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2);
      m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
      g.add(m);
    }
  }
  const apex = heartToTorso(f, { x: 0, y: 0, z: model.lvLengthCm });
  const base = heartToTorso(f, { x: 0, y: 0, z: 0 });
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(base.x, base.y, base.z),
      new THREE.Vector3(apex.x, apex.y, apex.z),
    ]),
    new THREE.LineDashedMaterial({
      color: 0xffc857,
      dashSize: 0.5,
      gapSize: 0.3,
      transparent: true,
      opacity: 0.7,
    }),
  );
  line.computeLineDistances();
  g.add(line);
  g.renderOrder = 1;
  return g;
}
