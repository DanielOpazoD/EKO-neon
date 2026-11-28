import { loadCaseById } from '@/cases';
import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import {
  navigatorLayers,
  segmentLayerOn,
  useHudStore,
  useSegmentHover,
  useSimStore,
} from '@/app/store';
import { modePolicy } from '@/app/modePolicy';
import type { PatientState, ThoraxModel } from '@/simulator/anatomy/thoraxModel';
import type { HeartFrame } from '@/simulator/anatomy/heartFrame';
import type { MeshError, MeshReply, MeshRequest, NavigatorModel } from '@/workers/heartMesh.worker';
import { beamFrameFromPose, poseFromControl } from '@/simulator/probe/pose';
import {
  isSoftwareWebglContext,
  navigatorStoreChanged,
  shouldRenderNavigatorFrame,
} from './navigatorBudget';
import {
  buildHeartGhost,
  buildSkeleton,
  buildSkin,
  buildWindowMarks,
  updateSkeletonStyle,
  updateSkinOpacity,
  type WindowBadgeAnchor,
  type WindowMarksRig,
} from './navigatorBody';
import {
  createCameraRig,
  narrowPanelFovDeg,
  navigatorState,
  type NavigatorCameraState,
  type NavigatorZoom,
} from './navigatorCamera';
import { createHeartMeshes } from './navigatorHeart';
import { attachNavigatorInput } from './navigatorInput';
import { buildProbe, updateFan } from './navigatorProbe';
import { createReviewMarkers } from './navigatorReview';

export { narrowPanelFovDeg, type NavigatorZoom };

export interface NavigatorSceneOptions {
  caseId: string;
  patient: PatientState;
  /** What a drag on the torso does, read at each pointer event. */
  dragAction: { readonly current: 'probe' | 'camera' };
  /** Filled with the camera controls once the scene is up, for the component's buttons. */
  zoom: { current: NavigatorZoom | null };
  /** Where the camera stood when the last scene was torn down; the next one starts there. */
  camera?: { current: NavigatorCameraState | null };
}

/**
 * Builds the 3D navigator (torso, probe, fan, heart) in `el` and runs it until the returned cleanup:
 * WebGL2 renderer with studio PMREM environment lighting, sculpted X-ray rib cage, physiological
 * beating 3D heart meshes, safe-clamped acoustic window badges, and render-on-change budgeting.
 */
export function mountNavigatorScene(
  el: HTMLDivElement,
  {
    caseId,
    patient,
    dragAction: dragActionRef,
    zoom: zoomRef,
    camera: cameraStateRef,
  }: NavigatorSceneOptions,
): () => void {
  let thorax: ThoraxModel | null = null;
  let heartFrame: HeartFrame | null = null;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, stencil: true });
  const softwareGl = isSoftwareWebglContext(renderer.getContext());
  renderer.localClippingEnabled = true;
  renderer.setPixelRatio(softwareGl ? 1 : Math.min(2, window.devicePixelRatio || 1));
  renderer.setClearColor(0x0f1319, 0);
  el.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 1, 300);

  // Studio PBR reflection environment: gives clearcoat tissue, vascular walls, translucent ribs,
  // and the moulded probe shell authentic specular highlights and perceived 3D depth.
  let pmrem: THREE.PMREMGenerator | null = null;
  let envRT: THREE.WebGLRenderTarget | null = null;
  if (!softwareGl) {
    try {
      pmrem = new THREE.PMREMGenerator(renderer);
      const room = new RoomEnvironment();
      envRT = pmrem.fromScene(room, 0.04);
      scene.environment = envRT.texture;
      scene.environmentIntensity = 0.46;
      room.dispose();
    } catch {
      /* fallback cleanly to directional lights if PMREM fails on exotic drivers */
    }
  }

  // Projected HTML overlay layer for acoustic window pills: always crisp, clamped inside safe margins
  // away from Sonda|Cámara & RotationDial, and auto-faded when the probe rests directly on a window.
  const badgeLayer = document.createElement('div');
  badgeLayer.className = 'nav-win-layer';
  badgeLayer.setAttribute('aria-hidden', 'true');
  el.appendChild(badgeLayer);
  const badgeEls: { anchor: WindowBadgeAnchor; el: HTMLDivElement }[] = [];

  // render-on-change bookkeeping
  let dirty = true;
  let contextLost = false;
  let lastRenderAt = 0;
  let lastCostMs = 0;
  let shownPhaseIdx = -1;
  let shownPhasesReady = -1;

  // Hoisted scratch math objects so `tick()` allocates nothing per draw.
  const basisMat = new THREE.Matrix4();
  const basisLat = new THREE.Vector3();
  const basisNorm = new THREE.Vector3();
  const basisFwd = new THREE.Vector3();
  const axisOrigin = new THREE.Vector3();
  const axisTip = new THREE.Vector3();
  const projVec = new THREE.Vector3();

  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.02;

  // Warm medical studio key + cool anatomical fill + sculptural rim light
  scene.add(new THREE.HemisphereLight(0xeef4fa, 0x131822, 0.68));
  const key = new THREE.DirectionalLight(0xfff2e6, 1.18);
  key.position.set(-16, 22, 38);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xd4e8ff, 1.32);
  rim.position.set(-10, 14, -18);
  scene.add(rim);
  const fill = new THREE.DirectionalLight(0x9abbe8, 0.44);
  fill.position.set(24, -10, 26);
  scene.add(fill);

  // ---- body (built when the navigator model arrives) ----
  let skin: THREE.Mesh | null = null;
  let skeleton: THREE.Group | null = null;
  let ghost: THREE.Group | null = null;
  let windowMarks: WindowMarksRig | null = null;
  const cutPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);

  const rebuildBadges = (anchors: WindowBadgeAnchor[]) => {
    badgeLayer.textContent = '';
    badgeEls.length = 0;
    for (const a of anchors) {
      const div = document.createElement('div');
      div.className = 'nav-win-badge';
      const dot = document.createElement('i');
      dot.style.background = a.colorHex;
      const span = document.createElement('span');
      span.textContent = a.label;
      div.append(dot, span);
      badgeLayer.appendChild(div);
      badgeEls.push({ anchor: a, el: div });
    }
  };

  const onModel = (model: NavigatorModel) => {
    useSimStore.getState().setNavigatorReady(true);
    thorax = model.thorax;
    heartFrame = model.frame;
    skin = buildSkin(model.thorax);
    scene.add(skin);
    skeleton = buildSkeleton(model.thorax);
    scene.add(skeleton);
    ghost = buildHeartGhost(model);
    scene.add(ghost);
    windowMarks = buildWindowMarks(model.thorax, model.windows);
    scene.add(windowMarks.group);
    rebuildBadges(windowMarks.badges);
    dirty = true;
  };

  // 16 cardiac phases (~19 Hz geometric beat): fluid contraction while staying fast via thoracic-aorta reuse
  const MESH_PHASES = Array.from({ length: 16 }, (_, i) => i / 16);
  const heart = createHeartMeshes(scene, cutPlane, MESH_PHASES.length);
  const meshWorker = new Worker(new URL('../../workers/heartMesh.worker.ts', import.meta.url), {
    type: 'module',
  });
  meshWorker.onerror = (e) => {
    console.warn('heart mesh worker failed to load; keeping the schematic heart', e.message);
  };
  meshWorker.onmessage = (ev: MessageEvent<MeshReply | MeshError | NavigatorModel>) => {
    if ('kind' in ev.data) {
      onModel(ev.data);
      return;
    }
    if ('error' in ev.data) {
      console.warn('heart mesh extraction failed; keeping the schematic heart', ev.data.error);
      return;
    }
    if (!heartFrame) return;
    heart.addPhase(ev.data, heartFrame);
    if (ghost) ghost.visible = false;
    dirty = true;
  };
  meshWorker.postMessage({
    caseDef: loadCaseById(caseId),
    patient,
    stepCm: 0.26,
    phases: MESH_PHASES,
  } satisfies MeshRequest);

  // examination axes: beam axis, elevation normal and the in-plane lateral direction
  const axisGroup = new THREE.Group();
  const axisLine = (color: number): THREE.Line =>
    new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85 }),
    );
  const beamAxis = axisLine(0xffc857);
  const elevAxis = axisLine(0x7ce8a0);
  const latAxis = axisLine(0x5cc8ff);
  axisGroup.add(beamAxis, elevAxis, latAxis);
  scene.add(axisGroup);
  const review = createReviewMarkers(scene);

  // ---- probe & luminous vertex-colored acoustic fan ----
  const { probe, marker } = buildProbe();
  scene.add(probe);
  const fanMat = new THREE.MeshBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.34,
    side: THREE.DoubleSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const fan = new THREE.Mesh(new THREE.BufferGeometry(), fanMat);
  fan.renderOrder = 4;
  scene.add(fan);
  const fanEdges = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.92,
      depthWrite: false,
    }),
  );
  fanEdges.renderOrder = 5;
  scene.add(fanEdges);

  // ---- camera / interaction ----
  const rig = createCameraRig(
    camera,
    () => thorax,
    () => {
      dirty = true;
    },
    cameraStateRef?.current,
  );
  zoomRef.current = rig.zoom;
  const dom = renderer.domElement;
  const onContextLost = (e: Event) => {
    e.preventDefault();
    contextLost = true;
  };
  const onContextRestored = () => {
    contextLost = false;
    dirty = true;
  };
  dom.addEventListener('webglcontextlost', onContextLost, false);
  dom.addEventListener('webglcontextrestored', onContextRestored, false);

  const onVisibility = () => {
    if (!document.hidden) dirty = true;
  };
  document.addEventListener('visibilitychange', onVisibility);

  const detachInput = attachNavigatorInput({
    el,
    dom,
    camera,
    rig,
    dragAction: dragActionRef,
    marker,
    heart,
    cutPlane,
    body: () => ({ thorax, heartFrame, skin, skeleton, ghost }),
  });

  let lastW = 0;
  let lastH = 0;
  let forceImmediate = false;
  const resize = () => {
    const rawW = el.clientWidth;
    const rawH = el.clientHeight;
    if (lastW > 0 && (rawW <= 0 || rawH <= 0)) return;
    const w = rawW > 0 ? rawW : 300;
    const h = rawH > 0 ? rawH : 300;
    if (w === lastW && h === lastH) return;
    lastW = w;
    lastH = h;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.fov = narrowPanelFovDeg(camera.aspect);
    camera.updateProjectionMatrix();
    dirty = true;
    forceImmediate = true;
    if (!contextLost && !document.hidden && rawW > 0 && rawH > 0) {
      renderer.render(scene, camera);
    }
  };
  const ro = new ResizeObserver(resize);
  ro.observe(el);
  resize();

  let raf = 0;
  const unsubHover = useSegmentHover.subscribe(() => {
    dirty = true;
  });
  const unsubDirty = useSimStore.subscribe((next, prev) => {
    if (navigatorStoreChanged(prev, next)) dirty = true;
  });
  const setAxisEndpoints = (
    line: THREE.Line,
    dir: { x: number; y: number; z: number },
    len: number,
  ): void => {
    axisTip.set(
      axisOrigin.x + dir.x * len,
      axisOrigin.y + dir.y * len,
      axisOrigin.z + dir.z * len,
    );
    const attr = line.geometry.getAttribute('position') as THREE.BufferAttribute;
    attr.setXYZ(0, axisOrigin.x, axisOrigin.y, axisOrigin.z);
    attr.setXYZ(1, axisTip.x, axisTip.y, axisTip.z);
    attr.needsUpdate = true;
  };

  /** Projects acoustic window micro-badges with safe margins and probe-proximity auto-fade. */
  const syncWindowBadges = (
    visible: boolean,
    probePos: { x: number; y: number; z: number },
  ): void => {
    badgeLayer.style.display = visible ? 'block' : 'none';
    if (!visible || !badgeEls.length || lastW <= 0 || lastH <= 0) return;
    const padX = 6;
    const topSafeY = 44; // clears top-left Sonda|Cámara pill
    const dialSafeLeft = lastW - 96; // clears top-right RotationDial
    const dialSafeBottom = 94;

    for (const item of badgeEls) {
      const a = item.anchor;
      const dist = Math.hypot(
        a.anchor.x - probePos.x,
        a.anchor.y - probePos.y,
        a.anchor.z - probePos.z,
      );
      // Hide badge when probe sits directly over that window so it never blocks the handle or marker
      if (dist < 3.4) {
        item.el.style.opacity = '0';
        continue;
      }
      projVec.copy(a.anchor);
      projVec.y += a.window === 'subcostal' ? -1.2 : 0.9;
      projVec.project(camera);
      if (projVec.z < -1 || projVec.z > 1) {
        item.el.style.opacity = '0';
        continue;
      }
      const bw = item.el.offsetWidth || 72;
      const bh = item.el.offsetHeight || 18;
      let sx = ((projVec.x + 1) * 0.5) * lastW - bw * 0.5;
      let sy = ((1 - projVec.y) * 0.5) * lastH - bh * 0.5;

      // Clamp neatly inside panel borders (prevents right-edge "• Apic" clipping)
      sx = Math.max(padX, Math.min(lastW - bw - padX, sx));
      sy = Math.max(topSafeY, Math.min(lastH - bh - 8, sy));
      // Avoid overlapping the top-right RotationDial card
      if (sx + bw > dialSafeLeft && sy < dialSafeBottom) {
        if (dialSafeLeft - bw - 6 >= padX) sx = dialSafeLeft - bw - 6;
        else sy = dialSafeBottom + 4;
      }
      item.el.style.transform = `translate3d(${Math.round(sx)}px, ${Math.round(sy)}px, 0)`;
      item.el.style.opacity = dist < 5.0 ? '0.55' : '0.92';
    }
  };

  const tick = () => {
    raf = requestAnimationFrame(tick);
    if (contextLost) return;
    const st = navigatorState();
    const anatomy = modePolicy(st.mode).navigatorAnatomy;
    const hudPhaseNow = useHudStore.getState().hud?.phase ?? 0;
    const phaseIdxNow = Math.round(hudPhaseNow * MESH_PHASES.length) % MESH_PHASES.length;
    const phasesReady = heart.phasesReady();
    const now = performance.now();
    const decision = shouldRenderNavigatorFrame({
      hidden: document.hidden,
      width: el.clientWidth,
      height: el.clientHeight,
      hasThorax: thorax !== null,
      anatomy,
      dirty,
      phaseIdx: phaseIdxNow,
      shownPhaseIdx,
      phasesReady,
      shownPhasesReady,
      totalPhases: MESH_PHASES.length,
      nowMs: now,
      lastRenderAtMs: lastRenderAt,
      lastCostMs,
      softwareGl,
      forceImmediate,
    });
    dirty = decision.nextDirty;
    if (!decision.shouldRender) return;
    forceImmediate = false;

    lastRenderAt = now;
    if (!thorax) {
      renderer.render(scene, camera);
      lastCostMs = performance.now() - now;
      return;
    }
    shownPhaseIdx = phaseIdxNow;
    shownPhasesReady = phasesReady;
    const pose = poseFromControl(thorax, st.probe);
    const beam = beamFrameFromPose(pose);
    probe.position.set(pose.position.x, pose.position.y, pose.position.z);
    basisLat.set(beam.lateral.x, beam.lateral.y, beam.lateral.z);
    basisNorm.set(beam.normal.x, beam.normal.y, beam.normal.z);
    basisFwd.set(beam.forward.x, beam.forward.y, beam.forward.z);
    basisMat.makeBasis(basisLat, basisNorm, basisFwd);
    probe.quaternion.setFromRotationMatrix(basisMat);
    updateFan(fan, fanEdges, beam, st.settings.depthCm, st.settings.sectorDeg);
    cutPlane.normal.set(beam.normal.x, beam.normal.y, beam.normal.z);
    cutPlane.constant = -(
      beam.normal.x * beam.origin.x +
      beam.normal.y * beam.origin.y +
      beam.normal.z * beam.origin.z
    );

    const layers = navigatorLayers(st);
    const heartVisible = anatomy && (layers.heart || layers.chambers || layers.vessels);
    heart.update({
      beam,
      layers,
      anatomy,
      phase: hudPhaseNow,
      segments: {
        on: segmentLayerOn(st),
        model: st.ui.segmentModel,
        selected: st.ui.selectedSegment,
        hovered: useSegmentHover.getState().id,
      },
    });
    if (ghost) ghost.visible = anatomy && phasesReady === 0;
    axisGroup.visible = layers.axes;
    review.sync(st.reviewMarkers, st.reviewSelectedId);
    review.group.visible = st.ui.reviewMode;
    if (layers.axes) {
      axisOrigin.set(beam.origin.x, beam.origin.y, beam.origin.z);
      setAxisEndpoints(beamAxis, beam.forward, st.settings.depthCm);
      setAxisEndpoints(elevAxis, beam.normal, 4);
      setAxisEndpoints(latAxis, beam.lateral, 4);
    }
    if (skeleton) {
      skeleton.visible = layers.skeleton;
      updateSkeletonStyle(skeleton, heartVisible);
    }
    if (windowMarks) {
      windowMarks.group.visible = layers.windows;
      if (layers.windows) windowMarks.setActiveProbe(pose.position);
    }
    if (skin) {
      skin.visible = layers.skin;
      updateSkinOpacity(skin, heartVisible ? 0.14 : !anatomy ? 0.34 : 0.72);
    }
    const drawn = [
      skin?.visible && 'skin',
      skeleton?.visible && 'skeleton',
      [...heart.meshByGroup.values()].some((m) => m.visible) && 'heart',
      ghost?.visible && 'ghost',
      windowMarks?.group.visible && 'windows',
      axisGroup.visible && 'axes',
    ]
      .filter(Boolean)
      .join(' ');
    if (el.dataset['drawn'] !== drawn) el.dataset['drawn'] = drawn;
    renderer.render(scene, camera);
    syncWindowBadges(layers.windows && anatomy, pose.position);
    lastCostMs = performance.now() - now;
  };
  tick();
  return () => {
    cancelAnimationFrame(raf);
    document.removeEventListener('visibilitychange', onVisibility);
    dom.removeEventListener('webglcontextlost', onContextLost);
    dom.removeEventListener('webglcontextrestored', onContextRestored);
    unsubDirty();
    unsubHover();
    detachInput();
    meshWorker.terminate();
    ro.disconnect();
    if (cameraStateRef) cameraStateRef.current = rig.state();
    envRT?.dispose();
    pmrem?.dispose();
    renderer.dispose();
    renderer.forceContextLoss();
    if (badgeLayer.parentElement === el) el.removeChild(badgeLayer);
    el.removeChild(renderer.domElement);
  };
}
