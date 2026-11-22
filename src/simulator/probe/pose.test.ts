import assert from 'node:assert/strict';
import test from 'node:test';
import { loadCaseById } from '@/cases';
import { cross, dot, length, normalize, sub, v3 } from '@/core/vec3';
import { createThoraxModel } from '@/simulator/anatomy/thoraxModel';
import {
  beamFrameFromPose,
  contactQuality,
  controlAimingAt,
  nudgedControl,
  poseFromControl,
  probeCompressionCm,
  type ProbeControl,
} from './pose';

const normalCase = loadCaseById('normal-excellent');
const thorax = createThoraxModel(normalCase.bodyHabitus, normalCase.acousticWindow, {
  position: 'left-lateral',
  respiration: 'expiration',
  headElevationDeg: 15,
});

test('beamFrameFromPose produces an orthonormal right-handed basis', () => {
  const samples: ProbeControl[] = [
    { u: 2.5, v: 0.5, rotationDeg: 0, tiltDeg: 0, rockDeg: 0, pressure: 0.6 },
    { u: 4.0, v: -1.5, rotationDeg: 45, tiltDeg: -22, rockDeg: 18, pressure: 0.8 },
    { u: 9.2, v: -6.0, rotationDeg: -135, tiltDeg: 35, rockDeg: -25, pressure: 0.4 },
  ];

  for (const ctrl of samples) {
    const pose = poseFromControl(thorax, ctrl);
    const bf = beamFrameFromPose(pose, contactQuality(ctrl.pressure));

    assert.ok(Math.abs(length(bf.forward) - 1) < 1e-6);
    assert.ok(Math.abs(length(bf.lateral) - 1) < 1e-6);
    assert.ok(Math.abs(length(bf.normal) - 1) < 1e-6);

    assert.ok(Math.abs(dot(bf.forward, bf.lateral)) < 1e-6);
    assert.ok(Math.abs(dot(bf.forward, bf.normal)) < 1e-6);
    assert.ok(Math.abs(dot(bf.lateral, bf.normal)) < 1e-6);

    // local basis: x = lateral, y = normal, z = forward => lateral × normal = forward
    const handedness = dot(cross(bf.lateral, bf.normal), bf.forward);
    assert.ok(Math.abs(handedness - 1) < 1e-6, `expected right-handed basis, got ${handedness}`);
  }
});

test('probeCompressionCm and contactQuality clamp pressure cleanly', () => {
  assert.equal(probeCompressionCm(-1), 0.3);
  assert.equal(probeCompressionCm(0), 0.3);
  assert.equal(probeCompressionCm(0.5), 0.55);
  assert.equal(probeCompressionCm(1), 0.8);
  assert.equal(probeCompressionCm(5), 0.8);

  assert.equal(contactQuality(-0.5), 0);
  assert.equal(contactQuality(0), 0);
  assert.ok(Math.abs(contactQuality(0.175) - 0.5) < 1e-9);
  assert.equal(contactQuality(0.35), 1);
  assert.equal(contactQuality(0.9), 1);
  assert.equal(contactQuality(2), 1);
});

test('nudgedControl wraps rotation into [-180, 180) and clamps tilt to [-70, 70]', () => {
  const base: ProbeControl = {
    u: 3,
    v: 0,
    rotationDeg: 170,
    tiltDeg: 60,
    rockDeg: 5,
    pressure: 0.6,
  };
  const wrappedPos = nudgedControl(base, 25, 25);
  assert.equal(wrappedPos.rotationDeg, -165);
  assert.equal(wrappedPos.tiltDeg, 70);
  assert.equal(wrappedPos.rockDeg, 5);

  const wrappedNeg = nudgedControl({ ...base, rotationDeg: -175, tiltDeg: -65 }, -20, -20);
  assert.equal(wrappedNeg.rotationDeg, 165);
  assert.equal(wrappedNeg.tiltDeg, -70);
});

test('controlAimingAt aligns beam forward and in-plane lateral within 1 degree', () => {
  const target = normalCase.anatomy.heartPosition.baseCm;
  const screenRight = normalize(v3(-0.6, 0.75, 0.2));
  const ctrl = controlAimingAt(thorax, 3.2, 0.5, target, screenRight, 0.65);
  const bf = beamFrameFromPose(poseFromControl(thorax, ctrl));

  const desiredForward = normalize(sub(target, bf.origin));
  const fwdErrDeg =
    (Math.acos(Math.max(-1, Math.min(1, dot(bf.forward, desiredForward)))) * 180) / Math.PI;

  const projRight = normalize(
    sub(screenRight, {
      x: desiredForward.x * dot(screenRight, desiredForward),
      y: desiredForward.y * dot(screenRight, desiredForward),
      z: desiredForward.z * dot(screenRight, desiredForward),
    }),
  );
  const latErrDeg =
    (Math.acos(Math.max(-1, Math.min(1, dot(bf.lateral, projRight)))) * 180) / Math.PI;

  assert.ok(fwdErrDeg < 1.0, `forward aiming error too large: ${fwdErrDeg.toFixed(3)} deg`);
  assert.ok(latErrDeg < 1.0, `lateral aiming error too large: ${latErrDeg.toFixed(3)} deg`);
});
