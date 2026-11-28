import React from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './app/App';
import { useHudStore, useSimStore } from './app/store';
// the interface's own type, self-hosted (decision 198): Inter for text, JetBrains Mono for figures (latin + latin-ext only)
import './app/fonts.css';
import './app/styles.css';
import type {
  compareBackends,
  compareDeferredDoppler,
  compareExperimentalBackends,
  compareImageChain,
} from '@/app/debugCompare';
import { frameBus } from '@/app/frameBus';
import { dopplerPerfMonitor } from '@/simulator/doppler/dopplerPerf';

// Console-only Doppler frame performance diagnostics: `dopplerPerf.report()` summarises the frames already recorded,
// `dopplerPerf.enable()` / `disable()` switch the per-frame console line, which is off by default.
(window as unknown as { dopplerPerf?: typeof dopplerPerfMonitor }).dopplerPerf = dopplerPerfMonitor;

// Test/debug hook (used by Playwright E2E and the dev console). It exposes the stores, and with them the case's ground
// truth, so a shipped build must not carry it or a learner reads the answers of the blind exam from the console: it is
// installed only by the dev server and by the E2E build (`npm run build:e2e`, VITE_TEST_HOOKS=1), and
// tools/ci/bundle-budget.mjs fails if a production build contains it. The backend comparison pulls the CPU tracer, the
// WebGL2 port and its shaders onto the main thread, which the app never needs: it loads on first call (page.evaluate
// awaits the promise), keeping them out of the entry chunk.
if (import.meta.env.DEV || import.meta.env.VITE_TEST_HOOKS === '1') {
  // Yield one macrotask between back-to-back debug comparisons so Chromium's GPU process drains
  // `WEBGL_lose_context` IPC from the previous disposable Webgl2Renderer before allocating the next.
  const yieldGpu = () => new Promise<void>((r) => setTimeout(r, 0));
  (window as unknown as { __echotwin?: unknown }).__echotwin = {
    useSimStore,
    useHudStore,
    compareBackends: async (...args: Parameters<typeof compareBackends>) => {
      await yieldGpu();
      const m = await import('@/app/debugCompare');
      return m.compareBackends(...args);
    },
    compareExperimentalBackends: async (
      ...args: Parameters<typeof compareExperimentalBackends>
    ) => {
      await yieldGpu();
      const m = await import('@/app/debugCompare');
      return m.compareExperimentalBackends(...args);
    },
    compareImageChain: async (...args: Parameters<typeof compareImageChain>) => {
      await yieldGpu();
      const m = await import('@/app/debugCompare');
      return m.compareImageChain(...args);
    },
    compareDeferredDoppler: async (...args: Parameters<typeof compareDeferredDoppler>) => {
      await yieldGpu();
      const m = await import('@/app/debugCompare');
      return m.compareDeferredDoppler(...args);
    },
    dopplerPerf: dopplerPerfMonitor,
    frameBus,
  };
}

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
