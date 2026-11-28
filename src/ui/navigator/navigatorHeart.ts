import * as THREE from 'three';
import type { HeartFrame } from '@/simulator/anatomy/heartFrame';
import type { BeamFrame } from '@/simulator/probe/pose';
import type { MeshGroup } from '@/simulator/anatomy/heartMesh';
import type { MeshReply } from '@/workers/heartMesh.worker';
import { SEGMENT_RGB, segmentIdOf, type SegmentModelChoice } from '../segmentMap';

/**
 * The heart surfaces of the 3D navigator: meshes extracted from the same implicit model the beam samples,
 * one geometry per cardiac phase, shaded with per-vertex concavity/sulcus ambient occlusion, subtle
 * helical myocardial tissue variation, subepicardial groove fat warmth, and physiological artery/vein
 * PBR finishes. Clipped cleanly by the imaging plane with stencil caps when `layers.cut` is enabled.
 */

/** The navigator layers and segment state one frame of the heart depends on. */
export interface HeartFrameState {
  beam: BeamFrame;
  layers: {
    cut: boolean;
    heart: boolean;
    chambers: boolean;
    valves: boolean;
    vessels: boolean;
  };
  /** Whether the anatomy may be seen at all (never in the exam, decision 256). */
  anatomy: boolean;
  /** Cardiac phase on screen, 0–1. */
  phase: number;
  segments: {
    on: boolean;
    model: SegmentModelChoice;
    selected: number | null;
    hovered: number | null;
  };
}

export interface HeartMeshes {
  /** Surface of each group (`lv-myocardium`, `left-atrium`, …), once its first phase has arrived. */
  readonly meshByGroup: ReadonlyMap<string, THREE.Mesh>;
  /** Per-vertex LV segment codes of each phase's myocardium geometry. */
  readonly lvSegmentsByGeom: ReadonlyMap<THREE.BufferGeometry, Uint8Array>;
  /** Phases received so far (they arrive in order). */
  phasesReady: () => number;
  /** Adds the surfaces of one phase, placed in the torso with the heart frame. */
  addPhase: (reply: MeshReply, frame: HeartFrame) => void;
  /** Visibility, clipping, caps, the phase shown and the segment colours, for the frame about to be drawn. */
  update: (s: HeartFrameState) => void;
}

const LV_BASE = new THREE.Color(0x9b282f);
const EPICARDIAL_FAT = new THREE.Color(0xdcb878);
const FADED = new THREE.Color().setRGB(58 / 255, 63 / 255, 74 / 255, THREE.SRGBColorSpace);
const HOVER_LIFT = new THREE.Color(1, 1, 1);

/**
 * Computes sculptural per-vertex linear RGB colors from the group's base tissue color,
 * concavity + sulcus ambient occlusion, subtle helical myocardial fiber variation, and
 * narrow subepicardial adipose tinting inside the interventricular and AV grooves.
 */
function buildNaturalVertexColors(g: MeshGroup): Float32Array {
  const n = g.positions.length / 3;
  const out = new Float32Array(n * 3);
  const base = new THREE.Color(g.color);
  const tmp = new THREE.Color();
  const isMyo = g.id === 'lv-myocardium' || g.id === 'rv-myocardium';
  const isAtria = g.id === 'left-atrium' || g.id === 'right-atrium';
  const aoArr = g.ao;
  const grooveArr = g.groove;
  const pos = g.positions;

  for (let v = 0; v < n; v++) {
    const ao = aoArr ? aoArr[v]! : 1;
    tmp.copy(base);
    if (isMyo || isAtria) {
      const px = pos[v * 3]!;
      const py = pos[v * 3 + 1]!;
      const pz = pos[v * 3 + 2]!;
      // Continuous 3D helical fiber modulation (seam-free in heart-frame space)
      const az = Math.atan2(py, px);
      const helix = Math.sin(az * 6 + pz * 1.45) * 0.5 + Math.cos(az * 11 - pz * 2.1) * 0.5;
      const macro = Math.sin(px * 0.9 + py * 0.7 - pz * 0.5);
      const fiberLift = 1 + (isMyo ? 0.065 : 0.04) * helix + 0.035 * macro;
      tmp.multiplyScalar(fiberLift);
      // Narrow warm subepicardial adipose highlight tightly confined to deep sulcus crevices
      if (isMyo && grooveArr) {
        const gr = grooveArr[v]!;
        if (gr > 0.34) {
          const fatMix = Math.min(0.2, (gr - 0.34) * 0.3);
          tmp.lerp(EPICARDIAL_FAT, fatMix);
        }
      }
    }
    tmp.multiplyScalar(ao);
    out[v * 3] = tmp.r;
    out[v * 3 + 1] = tmp.g;
    out[v * 3 + 2] = tmp.b;
  }
  return out;
}

/** Tailored PBR material per anatomical tissue family (leverages scene RoomEnvironment IBL). */
function createGroupMaterial(g: MeshGroup): THREE.MeshPhysicalMaterial {
  const isMyo = g.id === 'lv-myocardium' || g.id === 'rv-myocardium';
  const isVessel =
    g.id === 'aortic-root' ||
    g.id === 'thoracic-aorta' ||
    g.id === 'pulmonary-artery' ||
    g.id === 'venae-cavae' ||
    g.id === 'pulmonary-veins' ||
    g.id === 'great-vessels';
  const isCoronary = g.id === 'coronary-arteries' || g.id === 'coronary-veins';
  const isValve = g.id === 'valves';
  const isAtria = g.id === 'left-atrium' || g.id === 'right-atrium' || g.id === 'atria';

  if (isMyo) {
    return new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.44,
      metalness: 0.02,
      clearcoat: 0.42,
      clearcoatRoughness: 0.26,
      sheen: 0.16,
      sheenRoughness: 0.58,
      sheenColor: new THREE.Color(0xd96868),
      envMapIntensity: 0.72,
      side: THREE.DoubleSide,
    });
  }
  if (isCoronary) {
    return new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.26,
      metalness: 0.04,
      clearcoat: 0.72,
      clearcoatRoughness: 0.16,
      envMapIntensity: 1.05,
      side: THREE.FrontSide,
    });
  }
  if (isVessel) {
    return new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.34,
      metalness: 0.03,
      clearcoat: 0.52,
      clearcoatRoughness: 0.22,
      envMapIntensity: 0.95,
      transparent: g.opacity < 1,
      opacity: g.opacity,
      depthWrite: true,
      side: THREE.DoubleSide,
    });
  }
  if (isAtria) {
    return new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.44,
      metalness: 0.02,
      clearcoat: 0.36,
      clearcoatRoughness: 0.3,
      sheen: 0.22,
      sheenRoughness: 0.6,
      sheenColor: new THREE.Color(0xdba2b2),
      envMapIntensity: 0.85,
      transparent: g.opacity < 1,
      opacity: g.opacity,
      depthWrite: true,
      side: THREE.DoubleSide,
    });
  }
  if (isValve) {
    return new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.28,
      metalness: 0.03,
      clearcoat: 0.5,
      clearcoatRoughness: 0.2,
      envMapIntensity: 1.0,
      side: THREE.DoubleSide,
    });
  }
  // Translucent blood cavities
  return new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    vertexColors: true,
    roughness: 0.35,
    metalness: 0.02,
    clearcoat: 0.22,
    clearcoatRoughness: 0.4,
    transparent: g.opacity < 1,
    opacity: g.opacity,
    depthWrite: g.opacity >= 1,
    side: THREE.DoubleSide,
  });
}

/** LV segments on the 3D heart (decision 152), modulated by AO so segments keep sculptural depth. */
function paintSegments(
  geom: THREE.BufferGeometry,
  codes: Uint8Array,
  ao: Float32Array | undefined,
  model: SegmentModelChoice,
  selected: number | null,
  hovered: number | null,
): void {
  const table = Array.from({ length: 21 }, (_, c) => {
    const id = segmentIdOf(c, model);
    if (id === 0) return LV_BASE;
    const rgb = SEGMENT_RGB[id]!;
    const col = new THREE.Color().setRGB(
      rgb[0] / 255,
      rgb[1] / 255,
      rgb[2] / 255,
      THREE.SRGBColorSpace,
    );
    if (id === hovered) return col.lerp(HOVER_LIFT, 0.35);
    return selected !== null && id !== selected ? col.lerp(FADED, 0.55) : col;
  });
  let attr = geom.getAttribute('color') as THREE.BufferAttribute | undefined;
  if (!attr || attr.count !== codes.length) {
    attr = new THREE.BufferAttribute(new Float32Array(codes.length * 3), 3);
    geom.setAttribute('color', attr);
  }
  const a = attr.array as Float32Array;
  for (let v = 0; v < codes.length; v++) {
    const col = table[codes[v]!] ?? LV_BASE;
    const occ = ao ? ao[v]! : 1;
    a[v * 3] = col.r * occ;
    a[v * 3 + 1] = col.g * occ;
    a[v * 3 + 2] = col.b * occ;
  }
  attr.needsUpdate = true;
}

/** Restores natural sculpted vertex colors on the LV myocardium when AHA segments are turned off. */
function restoreNaturalColors(geom: THREE.BufferGeometry, natural: Float32Array): void {
  let attr = geom.getAttribute('color') as THREE.BufferAttribute | undefined;
  if (!attr || attr.count * 3 !== natural.length) {
    attr = new THREE.BufferAttribute(new Float32Array(natural), 3);
    geom.setAttribute('color', attr);
  } else {
    (attr.array as Float32Array).set(natural);
    attr.needsUpdate = true;
  }
}

/**
 * The heart surfaces in `scene`, clipped by `cutPlane`. `phaseCount` is the number of phases the worker was asked for:
 * the heart beats only once all have arrived.
 */
export function createHeartMeshes(
  scene: THREE.Scene,
  cutPlane: THREE.Plane,
  phaseCount: number,
): HeartMeshes {
  const heartMeshes = new THREE.Group();
  scene.add(heartMeshes);
  const meshMaterials: THREE.MeshStandardMaterial[] = [];
  const meshByGroup = new Map<string, THREE.Mesh>();
  const capByGroup = new Map<string, { stencil: THREE.Group; cap: THREE.Mesh }>();
  let capOrder = 0;
  const geomByGroup = new Map<string, THREE.BufferGeometry[]>();
  const lvSegmentsByGeom = new Map<THREE.BufferGeometry, Uint8Array>();
  const lvAoByGeom = new Map<THREE.BufferGeometry, Float32Array>();
  const lvNaturalColorsByGeom = new Map<THREE.BufferGeometry, Float32Array>();
  let segmentKeyShown = '';
  let phasesReady = 0;

  const addPhase = (reply: MeshReply, f: HeartFrame) => {
    const basis = new THREE.Matrix4().makeBasis(
      new THREE.Vector3(f.ex.x, f.ex.y, f.ex.z),
      new THREE.Vector3(f.ey.x, f.ey.y, f.ey.z),
      new THREE.Vector3(f.ez.x, f.ez.y, f.ez.z),
    );
    const placement = new THREE.Matrix4()
      .makeTranslation(f.origin.x, f.origin.y, f.origin.z)
      .multiply(basis);
    for (const g of reply.groups) {
      if (!g.indices.length) continue;
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(g.positions, 3));
      geom.setIndex(new THREE.BufferAttribute(g.indices, 1));
      geom.setAttribute('normal', new THREE.BufferAttribute(g.normals, 3));
      const naturalColors = buildNaturalVertexColors(g);
      geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(naturalColors), 3));
      if (g.segments && g.segments.length * 3 === g.positions.length) {
        lvSegmentsByGeom.set(geom, g.segments);
        lvNaturalColorsByGeom.set(geom, naturalColors);
        if (g.ao) lvAoByGeom.set(geom, g.ao);
        segmentKeyShown = '';
      }
      let perPhase = geomByGroup.get(g.id);
      if (!perPhase) {
        perPhase = [];
        geomByGroup.set(g.id, perPhase);
      }
      perPhase[reply.index] = geom;
      if (!meshByGroup.has(g.id)) {
        const mat = createGroupMaterial(g);
        meshMaterials.push(mat);
        const mesh = new THREE.Mesh(geom, mat);
        mesh.applyMatrix4(placement);
        meshByGroup.set(g.id, mesh);
        heartMeshes.add(mesh);

        // Coronaries are thin epicardial vessels: clipped by cutPlane, no stencil cap needed
        if (g.id !== 'coronary-arteries' && g.id !== 'coronary-veins') {
          const stencil = createPlaneStencilGroup(geom, cutPlane, 10 + capOrder * 2);
          stencil.applyMatrix4(placement);
          const capMat = new THREE.MeshStandardMaterial({
            color: new THREE.Color(g.color).multiplyScalar(0.52),
            roughness: 0.72,
            metalness: 0.02,
            side: THREE.DoubleSide,
            stencilWrite: true,
            stencilRef: 0,
            stencilFunc: THREE.NotEqualStencilFunc,
            stencilFail: THREE.ReplaceStencilOp,
            stencilZFail: THREE.ReplaceStencilOp,
            stencilZPass: THREE.ReplaceStencilOp,
          });
          const cap = new THREE.Mesh(new THREE.PlaneGeometry(45, 45), capMat);
          cap.renderOrder = 11 + capOrder * 2;
          scene.add(stencil, cap);
          capByGroup.set(g.id, { stencil, cap });
          capOrder++;
        }
      }
    }
    phasesReady = Math.max(phasesReady, reply.index + 1);
  };

  const update = ({ beam, layers, anatomy, phase, segments }: HeartFrameState) => {
    const clip = layers.cut ? [cutPlane] : [];
    for (const m of meshMaterials) m.clippingPlanes = clip;
    const vis: Record<string, boolean> = {
      'lv-myocardium': layers.heart,
      'rv-myocardium': layers.heart,
      'coronary-arteries': layers.heart && !segments.on,
      'coronary-veins': layers.heart && !segments.on,
      'lv-cavity': layers.chambers,
      'rv-cavity': layers.chambers,
      'left-atrium': layers.chambers,
      'right-atrium': layers.chambers,
      atria: layers.chambers,
      valves: layers.valves,
      'aortic-root': layers.vessels,
      'pulmonary-artery': layers.vessels,
      'venae-cavae': layers.vessels,
      'pulmonary-veins': layers.vessels,
      'great-vessels': layers.vessels,
      'thoracic-aorta': layers.vessels,
    };
    // a group without a layer of its own shows with the anatomy, never in the exam
    for (const [id, mesh] of meshByGroup) mesh.visible = vis[id] ?? anatomy;
    for (const [id, c] of capByGroup) {
      const v = (vis[id] ?? anatomy) && layers.cut;
      c.stencil.visible = v;
      c.cap.visible = v;
      c.cap.position.set(beam.origin.x, beam.origin.y, beam.origin.z);
      c.cap.lookAt(
        beam.origin.x + beam.normal.x,
        beam.origin.y + beam.normal.y,
        beam.origin.z + beam.normal.z,
      );
    }
    if (phasesReady === phaseCount) {
      const idx = Math.round(phase * phaseCount) % phaseCount;
      for (const [id, mesh] of meshByGroup) {
        const list = geomByGroup.get(id);
        const geom = list?.[idx] ?? list?.[0];
        if (geom && mesh.geometry !== geom) {
          mesh.geometry = geom;
          const c = capByGroup.get(id);
          if (c) for (const sm of c.stencil.children) (sm as THREE.Mesh).geometry = geom;
        }
      }
    }
    const segKey = `${segments.on ? 1 : 0}|${segments.model}|${segments.selected ?? '-'}|${segments.hovered ?? '-'}`;
    if (segKey !== segmentKeyShown) {
      const lvMesh = meshByGroup.get('lv-myocardium');
      if (lvMesh) {
        if (segments.on) {
          for (const [geom, codes] of lvSegmentsByGeom)
            paintSegments(
              geom,
              codes,
              lvAoByGeom.get(geom),
              segments.model,
              segments.selected,
              segments.hovered,
            );
        } else {
          for (const [geom, natural] of lvNaturalColorsByGeom) restoreNaturalColors(geom, natural);
        }
        segmentKeyShown = segKey;
      }
    }
  };

  return { meshByGroup, lvSegmentsByGeom, phasesReady: () => phasesReady, addPhase, update };
}

/**
 * Stencil mask for one clipped mesh: back faces increment and front faces decrement, so pixels where
 * the solid interior projects end up non-zero. The cap plane then fills exactly that region (and the
 * replace ops reset the stencil so the next group's mask starts clean). Standard three.js capping.
 */
function createPlaneStencilGroup(
  geometry: THREE.BufferGeometry,
  plane: THREE.Plane,
  renderOrder: number,
): THREE.Group {
  const group = new THREE.Group();
  const baseMat = new THREE.MeshBasicMaterial({
    depthWrite: false,
    depthTest: false,
    colorWrite: false,
    stencilWrite: true,
    stencilFunc: THREE.AlwaysStencilFunc,
    clippingPlanes: [plane],
  });
  const backMat = baseMat.clone();
  backMat.side = THREE.BackSide;
  backMat.stencilFail = backMat.stencilZFail = backMat.stencilZPass = THREE.IncrementWrapStencilOp;
  const backMesh = new THREE.Mesh(geometry, backMat);
  backMesh.renderOrder = renderOrder;
  const frontMat = baseMat.clone();
  frontMat.side = THREE.FrontSide;
  frontMat.stencilFail =
    frontMat.stencilZFail =
    frontMat.stencilZPass =
      THREE.DecrementWrapStencilOp;
  const frontMesh = new THREE.Mesh(geometry, frontMat);
  frontMesh.renderOrder = renderOrder;
  group.add(backMesh, frontMesh);
  return group;
}
