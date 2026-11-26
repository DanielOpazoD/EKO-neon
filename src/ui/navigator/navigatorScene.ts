import { loadCaseById } from '@/cases';
import * as THREE from 'three';
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
import { buildHeartGhost, buildSkeleton, buildSkin, buildWindowMarks } from './navigatorBody';
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
 * Builds the 3D navigator (torso, probe, fan, heart) in `el` and runs it until the returned cleanup: renderer, scene,
 * mesh worker, pointer handling and the render-on-change loop. Framework-free, so the React component only mounts it.
 * The pieces live in their own modules: the body (`navigatorBody`), the probe and its fan (`navigatorProbe`), the heart
 * surfaces (`navigatorHeart`), the review markers (`navigatorReview`), the camera (`navigatorCamera`) and the pointer
 * (`navigatorInput`); this module wires them to the stores and draws when something they show has changed.
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
  // the models arrive from the mesh worker as data (NavigatorModel, audit B7): until then the scene has the
  // probe, the fan and the lights, and the pose-dependent parts wait
  let thorax: ThoraxModel | null = null;
  let heartFrame: HeartFrame | null = null;
  // transparent over the panel's own radial backdrop (decision 202): the scene reads as lit in a room, not on a card
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, stencil: true });
  const softwareGl = isSoftwareWebglContext(renderer.getContext());
  renderer.localClippingEnabled = true; // the heart is cut by the imaging plane (decision 57)
  renderer.setPixelRatio(softwareGl ? 1 : Math.min(2, window.devicePixelRatio || 1));
  renderer.setClearColor(0x0f1319, 0);
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 1, 300);
  // render-on-change bookkeeping for the navigator (see tick)
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
  // Filmic tone mapping and a rim light: the anatomy is read by its volume, and flat linear output made
  // the myocardial shells look like cut-outs. The geometry itself stays the one the beam cuts — importing
  // a sculpted heart mesh would look better and stop matching the image, which is the whole point.
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  // a warm key, a cool fill and a stronger rim from behind (decision 202): skin and shells get a soft edge light
  // instead of a flat front wash
  scene.add(new THREE.HemisphereLight(0xe8eef5, 0x12161d, 0.85));
  const key = new THREE.DirectionalLight(0xfff1e2, 1.25);
  key.position.set(-14, 24, 40);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xdcecff, 1.2); // behind and above: separates the shells from the background
  rim.position.set(-8, 12, -14);
  scene.add(rim);
  const fill = new THREE.DirectionalLight(0x8fb2e6, 0.42);
  fill.position.set(22, -8, 28);
  scene.add(fill);

  // ---- body (built when the navigator model arrives) ----
  let skin: THREE.Mesh | null = null;
  let skeleton: THREE.Group | null = null;
  let ghost: THREE.Group | null = null;
  let windowMarks: THREE.Group | null = null;
  // the imaging plane doubles as a clipping plane: the 3D heart is split exactly where the beam cuts
  const cutPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
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
    scene.add(windowMarks);
    dirty = true;
  };
  // 0.26 cm and ten phases: extraction costs 553 ms per phase at this step against 2171 ms at 0.20, so the
  // whole beat is ready in ~5.5 s of worker time with the first phase on screen in half a second. The
  // finer grid left fewer black facets (3.2% of folded vertices on the RV wall against 4.5% here), but a
  // heart that stands still beside a beating image gives the model away far more than half a point of
  // facets (decisions 67 and 68).
  const MESH_PHASES = Array.from({ length: 10 }, (_, i) => i / 10);
  // surfaces extracted from the same implicit model the beam samples, so navigator and image cannot disagree
  const heart = createHeartMeshes(scene, cutPlane, MESH_PHASES.length);
  // extraction samples the implicit model hundreds of thousands of times: off the UI thread, with the
  // ghost showing until the surfaces arrive
  const meshWorker = new Worker(new URL('../../workers/heartMesh.worker.ts', import.meta.url), {
    type: 'module',
  });
  meshWorker.onerror = (e) => {
    // the ghost stays; the image is unaffected (it does not use these meshes)
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
    if (!heartFrame) return; // the model always precedes the meshes
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
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 }),
    );
  const beamAxis = axisLine(0xffc857);
  const elevAxis = axisLine(0x7ce8a0);
  const latAxis = axisLine(0x5cc8ff);
  axisGroup.add(beamAxis, elevAxis, latAxis);
  scene.add(axisGroup);
  // review markers placed on the model (decision 135)
  const review = createReviewMarkers(scene);

  // ---- probe ----
  const { probe, marker } = buildProbe();
  scene.add(probe);
  const fanMat = new THREE.MeshBasicMaterial({
    color: 0x5cc8ff,
    transparent: true,
    opacity: 0.14,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const fan = new THREE.Mesh(new THREE.BufferGeometry(), fanMat);
  scene.add(fan);
  const fanEdges = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0x5cc8ff, transparent: true, opacity: 0.6 }),
  );
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

  const resize = () => {
    const w = el.clientWidth || 300,
      h = el.clientHeight || 300;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // a tall, narrow panel (the exam's rail, torso alone, decision 256) widens the vertical field so the chest keeps
    // the width it has in a panel half as tall instead of filling the view with two ribs
    camera.fov = narrowPanelFovDeg(camera.aspect);
    camera.updateProjectionMatrix();
    dirty = true;
  };
  const ro = new ResizeObserver(resize);
  ro.observe(el);
  resize();

  let raf = 0;
  // The navigator renders strictly when something it shows has changed, paced by render cost and
  // software-WebGL budget so it never starves the simulator or main thread.
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
    });
    dirty = decision.nextDirty;
    if (!decision.shouldRender) return;

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
    // clip the heart with the imaging plane, and point the axes along the probe
    cutPlane.normal.set(beam.normal.x, beam.normal.y, beam.normal.z);
    cutPlane.constant = -(
      beam.normal.x * beam.origin.x +
      beam.normal.y * beam.origin.y +
      beam.normal.z * beam.origin.z
    );
    // the saved layers, or in exam mode the skin and the ribs alone (decision 256)
    const layers = navigatorLayers(st);
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
    // the schematic heart stands in until the surfaces arrive, and only where the heart may be seen
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
    if (skeleton) skeleton.visible = layers.skeleton;
    if (windowMarks) windowMarks.visible = layers.windows;
    // the skin is a layer of its own: hiding the bones used to make it opaque, which hid the heart; in the exam the
    // ribs are the landmark, so the skin stays as translucent as over the heart (decision 256)
    if (skin) {
      skin.visible = layers.skin;
      (skin.material as THREE.MeshStandardMaterial).opacity =
        !anatomy || layers.heart || layers.chambers || layers.vessels ? 0.32 : 0.85;
    }
    // what the frame draws, read from the objects themselves, for the E2E check of the exam (decision 256)
    const drawn = [
      skin?.visible && 'skin',
      skeleton?.visible && 'skeleton',
      [...heart.meshByGroup.values()].some((m) => m.visible) && 'heart',
      ghost?.visible && 'ghost',
      windowMarks?.visible && 'windows',
      axisGroup.visible && 'axes',
    ]
      .filter(Boolean)
      .join(' ');
    if (el.dataset['drawn'] !== drawn) el.dataset['drawn'] = drawn;
    renderer.render(scene, camera);
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
    // the next scene starts where the learner left the camera
    if (cameraStateRef) cameraStateRef.current = rig.state();
    renderer.dispose();
    // dispose() keeps the WebGL context until the canvas is collected: each rebuild (breathing, position, case) left one
    // alive, and the browser, at its limit of contexts, takes the oldest away — possibly the one the scanner draws with
    renderer.forceContextLoss();
    el.removeChild(renderer.domElement);
  };
}
