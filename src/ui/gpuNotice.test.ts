import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { gpuNotice } from './gpuNotice';

// The texts of the core: why a frame was traced on the CPU (`FrameFormation.gpu`), and the port's state now
// (`GpuRecovery.status`). Scenarios that run the core over a simulated GPU are in gpuRecovery.test.ts.
const LOST = 'Contexto WebGL perdido';
const TRACED = 'Sin GPU · trazador CPU, calidad media';
const tipOver = (port: string) => `${LOST}: cuadro formado por el trazador CPU · ${port}`;

describe("the HUD's GPU notice", () => {
  it('says nothing over a frame the GPU formed while the port is there, nor before the first output', () => {
    assert.equal(gpuNotice({ gpu: 'ok', tier: 'high', gpuLive: 'ok' }), null);
    assert.equal(gpuNotice({ gpu: null, tier: null, gpuLive: null }), null);
  });

  it("over a frame the GPU formed, frozen with the port lost since, says the port's state and no tracer", () => {
    assert.deepEqual(gpuNotice({ gpu: 'ok', tier: 'high', gpuLive: 'GPU perdida (intento 1 de 5)' }), {
      text: 'GPU perdida (intento 1 de 5)',
    });
  });

  it('over a frame the CPU tracer formed, says so with the quality of that frame', () => {
    for (const [tier, label] of [
      ['low', 'baja'],
      ['medium', 'media'],
      ['high', 'alta'],
    ] as const)
      assert.equal(gpuNotice({ gpu: LOST, tier, gpuLive: 'ok' })?.text, `Sin GPU · trazador CPU, calidad ${label}`);
    assert.equal(gpuNotice({ gpu: LOST, gpuLive: 'ok' })?.text, 'Sin GPU · trazador CPU');
  });

  it('B1: over a CPU-traced frame with the GPU back, the tip says it is back, not that the tracer waits for it', () => {
    const notice = gpuNotice({ gpu: LOST, tier: 'medium', gpuLive: 'ok' });
    // before: the tip was the frame's reason at its capture, «trazador CPU hasta que vuelva la GPU (intento 1 de 5)»
    assert.deepEqual(notice, { text: TRACED, tip: tipOver('GPU de vuelta') });
    assert.doesNotMatch(String(notice?.tip), /hasta que vuelva/);
  });

  it('B2: over a CPU-traced frame, the tip follows the attempts and, given up, says to reload the case', () => {
    for (let k = 1; k <= 5; k++) {
      const port = `GPU perdida (intento ${k} de 5)`;
      assert.deepEqual(gpuNotice({ gpu: LOST, tier: 'medium', gpuLive: port }), { text: TRACED, tip: tipOver(port) });
    }
    const gaveUp = 'Sin GPU tras 5 intentos; recarga la página';
    assert.deepEqual(gpuNotice({ gpu: LOST, tier: 'medium', gpuLive: gaveUp }), { text: TRACED, tip: tipOver(gaveUp) });
  });

  it("without WebGL2 the tip is the reason alone: the port's state is that same reason, nothing to wait for", () => {
    const reason = 'WebGL2 unavailable';
    assert.deepEqual(gpuNotice({ gpu: reason, tier: 'medium', gpuLive: reason }), { text: TRACED, tip: reason });
  });
});
