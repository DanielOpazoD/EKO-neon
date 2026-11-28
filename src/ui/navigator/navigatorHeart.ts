import * as THREE from 'three';
import type { HeartFrame } from '@/simulator/anatomy/heartFrame';
import type { BeamFrame } from '@/simulator/probe/pose';
import type { MeshGroup } from '@/simulator/anatomy/heartMesh';
import type { MeshReply } from '@/workers/heartMesh.worker';
import { SEGMENT_RGB, segmentIdOf, type SegmentModelChoice } from '../segmentMap';

/**
 * The heart surfaces of the 3D navigator: meshes extracted from the same implicit model the beam samples,
 * one geometry per cardiac phase, rendered as solid organic tissue using an unparametrized 3D procedural
 * per-pixel muscle-fiber normal & roughness shader (Mikkelsen screen-space bump perturbation — zero UV
 * seams, zero plastic clearcoat/sheen film, zero semi-transparent double-wall ghosts).
 */

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

const LV_BASE = new THREE.Color(0x8b2329);
const FADED = new THREE.Color().setRGB(58 / 255, 63 / 255, 74 / 255, THREE.SRGBColorSpace);
const HOVER_LIFT = new THREE.Color(1, 1, 1);

/**
 * Computes natural per-vertex linear RGB colors modulated purely by geometric & sulcus ambient
 * occlusion (no washed-out cream blotches; fine organic tissue fibers are computed per pixel in GLSL).
 */
function buildNaturalVertexColors(g: MeshGroup): Float32Array {
  const n = g.positions.length / 3;
  const out = new Float32Array(n * 3);
  const base = new THREE.Color(g.color);
  const tmp = new THREE.Color();
  const aoArr = g.ao;
  for (let v = 0; v < n; v++) {
    const ao = aoArr ? aoArr[v]! : 1;
    tmp.copy(base).multiplyScalar(ao);
    out[v * 3] = tmp.r;
    out[v * 3 + 1] = tmp.g;
    out[v * 3 + 2] = tmp.b;
  }
  return out;
}

const ORGANIC_NOISE_GLSL = /* glsl */ `
varying vec3 vTissuePos;
uniform float uTissueBump;
uniform float uTissueFiber;
uniform float uTissueMode;

float hash31(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}

float vnoise3(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float n000 = hash31(i);
  float n100 = hash31(i + vec3(1.0, 0.0, 0.0));
  float n010 = hash31(i + vec3(0.0, 1.0, 0.0));
  float n110 = hash31(i + vec3(1.0, 1.0, 0.0));
  float n001 = hash31(i + vec3(0.0, 0.0, 1.0));
  float n101 = hash31(i + vec3(1.0, 0.0, 1.0));
  float n011 = hash31(i + vec3(0.0, 1.0, 1.0));
  float n111 = hash31(i + vec3(1.0, 1.0, 1.0));
  float nx00 = mix(n000, n100, f.x);
  float nx10 = mix(n010, n110, f.x);
  float nx01 = mix(n001, n101, f.x);
  float nx11 = mix(n011, n111, f.x);
  return mix(mix(nx00, nx10, f.y), mix(nx01, nx11, f.y), f.z);
}

float organicTissueField(vec3 p, float mode) {
  float macro = vnoise3(p * 1.1);
  float fine = vnoise3(p * 3.4 + vec3(macro * 1.4));
  if (mode < 0.5) {
    // Singularity-free 3D anisotropic helical myocardial fiber grain (no polar pinch at the apex)
    vec3 w = p + vec3(macro * 0.45, fine * 0.28, -macro * 0.35);
    float f1 = vnoise3(vec3(w.x * 2.8 + w.z * 1.9, w.y * 2.8 - w.z * 1.6, w.z * 1.1));
    float f2 = vnoise3(vec3(w.x * 5.2 - w.y * 2.4, w.y * 5.2 + w.z * 3.2, w.z * 2.1));
    return f1 * 0.45 + f2 * 0.25 + fine * 0.20 + macro * 0.10;
  }
  if (mode < 1.5) {
    // Atria: finer irregular pectinate/muscular grain
    return fine * 0.55 + macro * 0.45;
  }
  // Great vessels & coronaries: subtle smooth elastic-wall variation
  float ring = vnoise3(vec3(p.x * 2.0, p.y * 2.0, p.z * 4.5 + macro * 1.2));
  return fine * 0.45 + macro * 0.35 + ring * 0.20;
}
`;

/**
 * Injects seam-free 3D procedural myocardial/vascular texture, micro-roughness, and Mikkelsen
 * screen-space normal perturbation into a standard PBR material — giving genuine living tissue
 * relief without any glossy plastic clearcoat or milky sheen overlay.
 */
function applyOrganicTissueShader(
  mat: THREE.MeshStandardMaterial,
  mode: 0 | 1 | 2,
  bumpScale: number,
  fiberColorStrength: number,
): void {
  mat.customProgramCacheKey = () => `organic-tissue-${mode}`;
  mat.onBeforeCompile = (shader) => {
    shader.uniforms['uTissueBump'] = { value: bumpScale };
    shader.uniforms['uTissueFiber'] = { value: fiberColorStrength };
    shader.uniforms['uTissueMode'] = { value: mode };
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>\nvarying vec3 vTissuePos;`,
      )
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>\nvTissuePos = position;`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${ORGANIC_NOISE_GLSL}`)
      .replace(
        '#include <color_fragment>',
        `#include <color_fragment>
        float tissueH = organicTissueField(vTissuePos, uTissueMode);
        float tissueTint = mix(1.0 - uTissueFiber, 1.0 + uTissueFiber * 0.65, tissueH);
        diffuseColor.rgb *= tissueTint;`,
      )
      .replace(
        '#include <roughnessmap_fragment>',
        `#include <roughnessmap_fragment>
        roughnessFactor = clamp(roughnessFactor + (0.5 - tissueH) * 0.24, 0.36, 0.88);`,
      )
      .replace(
        '#include <normal_fragment_maps>',
        `#include <normal_fragment_maps>
        if (uTissueBump > 0.001) {
          vec3 dPosX = dFdx(vViewPosition);
          vec3 dPosY = dFdy(vViewPosition);
          float dHx = dFdx(tissueH);
          float dHy = dFdy(tissueH);
          vec3 r1 = cross(dPosY, normal);
          vec3 r2 = cross(normal, dPosX);
          float det = dot(dPosX, r1);
          vec3 surfGrad = sign(det) * (dHx * r1 + dHy * r2);
          normal = normalize(abs(det) * normal - uTissueBump * surfGrad);
        }`,
      );
  };
}

/** Natural anatomical PBR materials: solid opaque tissue with zero milky clearcoat/sheen film. */
function createGroupMaterial(g: MeshGroup): THREE.MeshStandardMaterial {
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
    const mat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.56,
      metalness: 0.0,
      envMapIntensity: 0.35,
      side: THREE.DoubleSide,
    });
    applyOrganicTissueShader(mat, 0, 0.16, 0.1);
    return mat;
  }
  if (isCoronary) {
    const mat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.5,
      metalness: 0.0,
      envMapIntensity: 0.35,
      side: THREE.FrontSide,
    });
    applyOrganicTissueShader(mat, 2, 0.06, 0.07);
    return mat;
  }
  if (isVessel) {
    const mat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.52,
      metalness: 0.0,
      envMapIntensity: 0.4,
      side: THREE.DoubleSide,
    });
    applyOrganicTissueShader(mat, 2, 0.1, 0.09);
    return mat;
  }
  if (isAtria) {
    const mat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.6,
      metalness: 0.0,
      envMapIntensity: 0.32,
      side: THREE.DoubleSide,
    });
    applyOrganicTissueShader(mat, 1, 0.16, 0.11);
    return mat;
  }
  if (isValve) {
    return new THREE.MeshStandardMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.44,
      metalness: 0.0,
      envMapIntensity: 0.45,
      side: THREE.DoubleSide,
    });
  }
  // Translucent blood cavities (only visible when sliced or when myocardium is hidden)
  return new THREE.MeshStandardMaterial({
    color: 0xffffff,
    vertexColors: true,
    roughness: 0.48,
    metalness: 0.0,
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

        if (g.id !== 'coronary-arteries' && g.id !== 'coronary-veins') {
          const stencil = createPlaneStencilGroup(geom, cutPlane, 10 + capOrder * 2);
          stencil.applyMatrix4(placement);
          const capMat = new THREE.MeshStandardMaterial({
            color: new THREE.Color(g.color).multiplyScalar(0.55),
            roughness: 0.72,
            metalness: 0.0,
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
    // Internal blood cavities only show when the heart is cut open or myocardium is hidden,
    // preventing translucent inner shells from bleeding through thin walls on the intact heart.
    const showInnerBlood = layers.chambers && (layers.cut || !layers.heart);
    const vis: Record<string, boolean> = {
      'lv-myocardium': layers.heart,
      'rv-myocardium': layers.heart,
      'coronary-arteries': layers.heart && !segments.on,
      'coronary-veins': layers.heart && !segments.on,
      'lv-cavity': showInnerBlood,
      'rv-cavity': showInnerBlood,
      'left-atrium': layers.chambers,
      'right-atrium': layers.chambers,
      atria: layers.chambers,
      valves: layers.valves && (layers.cut || !layers.heart),
      'aortic-root': layers.vessels,
      'pulmonary-artery': layers.vessels,
      'venae-cavae': layers.vessels,
      'pulmonary-veins': layers.vessels,
      'great-vessels': layers.vessels,
      'thoracic-aorta': layers.vessels,
    };
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
