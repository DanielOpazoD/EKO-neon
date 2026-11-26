import { afterEach, beforeEach, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { loadCaseById } from '@/cases';
import type { StructuredEchoTruth } from '@/simulator/hemodynamics/groundTruth';
import { baseInput } from './baseInput';
import { SimClient } from './client';
import type { MainToWorker, PhaseMarks, WorkerToMain } from './protocol';

/** A worker that only records what it is sent; the test plays its replies. */
class FakeWorker {
  static last: FakeWorker | null = null;
  onmessage: ((ev: { data: WorkerToMain }) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;
  readonly posted: MainToWorker[] = [];
  constructor() {
    FakeWorker.last = this;
  }
  postMessage(m: MainToWorker): void {
    this.posted.push(m);
  }
  terminate(): void {}
  reply(data: WorkerToMain): void {
    this.onmessage?.({ data });
  }
}

const TRUTH = {} as unknown as StructuredEchoTruth;
const MARKS = {} as unknown as PhaseMarks;
const scope = globalThis as unknown as { Worker?: unknown };

describe('SimClient announces only the latest case load', () => {
  let saved: unknown;
  beforeEach(() => {
    saved = scope.Worker;
    scope.Worker = FakeWorker;
  });
  afterEach(() => {
    scope.Worker = saved;
  });

  it('drops the ready of a load superseded before its core was announced', () => {
    const ready: string[] = [];
    const client = new SimClient({
      onFrame: () => {},
      onReady: (_t, id) => ready.push(id),
      onError: () => {},
    });
    assert.equal(client.mode, 'worker');
    const worker = FakeWorker.last!;
    // the learner picks B and, before B's core is ready, C (arrow keys on the case selector)
    void client.loadCase(loadCaseById('aortic-stenosis-severe'), baseInput());
    void client.loadCase(loadCaseById('hocm-sam'), baseInput());
    const loads = worker.posted.filter(
      (m): m is Extract<MainToWorker, { type: 'loadCase' }> => m.type === 'loadCase',
    );
    assert.equal(loads.length, 2);
    assert.notEqual(loads[0]!.generation, loads[1]!.generation);
    // the worker builds both cores in order and announces each one
    for (const m of loads)
      worker.reply({
        type: 'ready',
        truth: TRUTH,
        caseId: m.caseDef.id,
        phaseMarks: MARKS,
        lvLengthCm: 8,
        generation: m.generation,
      });
    assert.deepEqual(ready, ['hocm-sam'], 'only the core of the latest load is announced');
    client.dispose();
  });

  it('still announces a ready from a worker that does not echo the generation', () => {
    const ready: string[] = [];
    const client = new SimClient({
      onFrame: () => {},
      onReady: (_t, id) => ready.push(id),
      onError: () => {},
    });
    void client.loadCase(loadCaseById('hocm-sam'), baseInput());
    FakeWorker.last!.reply({
      type: 'ready',
      truth: TRUTH,
      caseId: 'hocm-sam',
      phaseMarks: MARKS,
      lvLengthCm: 8,
    });
    assert.deepEqual(ready, ['hocm-sam']);
    client.dispose();
  });
});
