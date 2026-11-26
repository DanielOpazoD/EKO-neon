import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { createCameraRig } from './navigatorCamera';

describe('3D navigator camera', () => {
  it('starts a rebuilt scene where the learner left the camera', () => {
    const first = new THREE.PerspectiveCamera(35, 1, 1, 300);
    const rig = createCameraRig(first, () => null, () => {});
    const overview = { ...rig.orbit };
    rig.zoom.zoomBy(0.6);
    rig.orbit.az += 0.9;
    first.up.set(0, 0, 1);
    rig.update();
    const left = rig.state();

    // breathing or position changed: a new scene, a new camera, the same framing
    const second = new THREE.PerspectiveCamera(35, 1, 1, 300);
    const rebuilt = createCameraRig(second, () => null, () => {}, left);
    assert.deepEqual({ ...rebuilt.orbit }, left.orbit);
    assert.ok(second.position.distanceTo(first.position) < 1e-9, 'same camera position');
    assert.deepEqual([second.up.x, second.up.y, second.up.z], [0, 0, 1]);

    // and a scene with nothing to carry over starts at the overview
    const fresh = createCameraRig(new THREE.PerspectiveCamera(35, 1, 1, 300), () => null, () => {});
    assert.deepEqual({ ...fresh.orbit }, overview);
    // the state is a copy: moving the new rig does not move the saved framing
    rebuilt.orbit.r = 80;
    assert.notEqual(left.orbit.r, 80);
  });
});
