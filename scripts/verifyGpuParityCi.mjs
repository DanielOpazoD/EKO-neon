#!/usr/bin/env node
/**
 * Automated real WebGL2 (headless Chromium + SwiftShader) CPU <-> GPU parity verification.
 * Mirrors the exact checks and thresholds of scripts/verifyGpuParity.sh without requiring
 * playwright-cli or adding Playwright to package.json dependencies.
 *
 * Usage:
 *   PLAYWRIGHT_MODULE=/tmp/pw-ci/node_modules/playwright node scripts/verifyGpuParityCi.mjs [url]
 */
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const URL = process.argv[2] || 'http://127.0.0.1:4173';

const browser = await chromium.launch({
  headless: true,
  args: [
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-webgl',
    '--enable-unsafe-swiftshader',
    '--ignore-gpu-blocklist',
    '--disable-gpu-watchdog',
    '--disable-dev-shm-usage',
  ],
});

const consoleErrors = [];
const pageErrors = [];

try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  page.on('pageerror', (err) => {
    pageErrors.push(err?.stack || String(err));
  });

  await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await page.waitForFunction(() => Boolean(window.__echotwin?.compareBackends), null, {
    timeout: 60_000,
  });

  const data = await page.evaluate(async () => {
    const b = await window.__echotwin.compareBackends('plax', 0.25, 'normal-excellent-window', 'low');
    const img = await window.__echotwin.compareImageChain('plax', 0.25, 'normal-excellent-window', 'low');
    const dop = await window.__echotwin.compareDeferredDoppler('plax', 0.22, 'normal-excellent-window', 'low');
    const dopMr = await window.__echotwin.compareDeferredDoppler('a4c', 0.22, 'mvp-primary-mr', 'low', {
      gainDb: 6,
      baselineShiftMps: 0.2,
    });
    const dopAr = await window.__echotwin.compareDeferredDoppler(
      'plax',
      0.56,
      'aortic-regurgitation-severe',
      'low',
    );
    const dopMs = await window.__echotwin.compareDeferredDoppler(
      'plax',
      0.65,
      'mitral-stenosis-severe',
      'low',
    );

    const { loadCaseById } = await import('/src/cases/index.ts');
    const { buildCaseModels, REST_PATIENT } = await import('/src/simulator/anatomy/caseModels.ts');
    const { cycleStateAt } = await import('/src/simulator/cardiac-cycle/cycleModel.ts');
    const { computeHeartPose } = await import('/src/simulator/anatomy/heartPose.ts');
    const PHASE_OF = {
      endSystole: (st) => st.contraction,
      swing: (st, heart) => Math.abs(computeHeartPose(heart, st).swingX),
      avOpen: (st) => st.avOpen,
    };
    const phaseOf = (caseId, at) => {
      const { heart, tables } = buildCaseModels(loadCaseById(caseId), REST_PATIENT);
      let phase = 0;
      let most = -Infinity;
      for (let i = 0; i <= 400; i++) {
        const v = PHASE_OF[at](cycleStateAt(tables, i / 400), heart);
        if (v > most) {
          most = v;
          phase = i / 400;
        }
      }
      return phase;
    };

    const wiring = [];
    for (const [caseId, view, at] of [
      ['pericardial-effusion-tamponade', 'psax-pm', 'endSystole'],
      ['pericardial-effusion-tamponade', 'subcostal-4c', 'endSystole'],
      ['pericardial-effusion-tamponade', 'a2c', 'endSystole'],
      ['pericardial-effusion-tamponade', 'psax-pm', 'swing'],
      ['pericardial-effusion-tamponade', 'subcostal-4c', 'swing'],
      ['pericardial-effusion-tamponade', 'a2c', 'swing'],
      ['inferior-rwma', 'psax-mv', 'endSystole'],
      ['inferior-rwma', 'psax-pm', 'endSystole'],
      ['inferior-rwma', 'a2c', 'endSystole'],
      ['aortic-stenosis-severe', 'psax-av', 'avOpen'],
      ['pulmonary-hypertension-rv', 'a4c', 'endSystole'],
      ['pulmonary-hypertension-rv', 'psax-pm', 'endSystole'],
    ]) {
      const phase = phaseOf(caseId, at);
      const r = await window.__echotwin.compareBackends(view, phase, caseId, 'low');
      wiring.push({
        caseId,
        view,
        at,
        phase,
        error: r.error,
        structureAgreement: r.structureAgreement,
        perStructure: r.perStructure,
      });
    }

    return { b, img, dop, dopMr, dopAr, dopMs, wiring };
  });

  const { b, img, dop, dopMr, dopAr, dopMs, wiring } = data;
  const failures = [];
  const check = (cond, msg) => {
    if (!cond) failures.push(msg);
  };

  // 1. B-mode raymarching CPU <-> GPU parity
  check(!b.error, `compareBackends error: ${b.error}`);
  check(b.structureAgreement >= 0.99, `structureAgreement ${b.structureAgreement} < 0.99`);
  check(b.tissueAgreement >= 0.99, `tissueAgreement ${b.tissueAgreement} < 0.99`);
  check(b.ampRelDiff <= 0.015, `ampRelDiff ${b.ampRelDiff} > 0.015`);
  check(b.transDiff <= 0.005, `transDiff ${b.transDiff} > 0.005`);

  // 2. Console + ScanConvert + Present parity
  check(!img.error, `compareImageChain error: ${img.error}`);
  check(img.idsAgreement >= 0.995, `idsAgreement ${img.idsAgreement} < 0.995`);
  check(Math.max(...img.displayMaxDiff) <= 2, `displayMaxDiff ${img.displayMaxDiff} > 2`);
  check(img.presentFracOver1 <= 0.005, `presentFracOver1 ${img.presentFracOver1} > 0.005`);

  // 3. Deferred Color Doppler (PLAX normal)
  check(!dop.error, `compareDeferredDoppler(plax) error: ${dop.error}`);
  check(dop.detValidAgreement >= 0.99, `plax detValidAgreement ${dop.detValidAgreement} < 0.99`);
  check(dop.detVelMaxDiff <= 0.015, `plax detVelMaxDiff ${dop.detVelMaxDiff} > 0.015`);
  check(dop.detVarMaxDiff <= 0.02, `plax detVarMaxDiff ${dop.detVarMaxDiff} > 0.02`);
  check(dop.detPresentFracOver1 <= 0.005, `plax detPresentFracOver1 ${dop.detPresentFracOver1} > 0.005`);
  check(
    Math.abs(dop.kasaiWf.cpuBloodFill - dop.kasaiWf.gpuBloodFill) <= 0.08,
    `plax default-WF blood fill drift cpu=${dop.kasaiWf.cpuBloodFill.toFixed(3)} gpu=${dop.kasaiWf.gpuBloodFill.toFixed(3)}`,
  );
  check(
    dop.kasaiWf.gpuClutterFill <= 0.06,
    `plax default-WF GPU clutter not suppressed: ${dop.kasaiWf.gpuClutterFill.toFixed(3)}`,
  );
  check(
    dop.kasaiMinWf.gpuClutterFill >= dop.kasaiWf.gpuClutterFill,
    `plax min-WF GPU clutter (${dop.kasaiMinWf.gpuClutterFill.toFixed(3)}) < default-WF (${dop.kasaiWf.gpuClutterFill.toFixed(3)})`,
  );

  // 4. Deferred Color Doppler across key valvular pathologies (MR, AR, MS)
  for (const [label, d] of [
    ['a4c-mr', dopMr],
    ['plax-ar', dopAr],
    ['plax-ms', dopMs],
  ]) {
    check(!d.error, `compareDeferredDoppler(${label}) error: ${d.error}`);
    check(d.detValidAgreement >= 0.99, `${label} detValidAgreement ${d.detValidAgreement} < 0.99`);
    check(d.detVelMaxDiff <= 0.02, `${label} detVelMaxDiff ${d.detVelMaxDiff} > 0.02`);
    check(d.detPresentFracOver1 <= 0.005, `${label} detPresentFracOver1 ${d.detPresentFracOver1} > 0.005`);
    check(
      Math.abs(d.kasaiWf.cpuBloodFill - d.kasaiWf.gpuBloodFill) <= 0.08,
      `${label} default-WF blood fill drift cpu=${d.kasaiWf.cpuBloodFill.toFixed(3)} gpu=${d.kasaiWf.gpuBloodFill.toFixed(3)}`,
    );
  }

  // 5. GPU wiring parity (pericardium, regional wall motion, aortic opening, PH)
  const WIRING_FRAME_MIN = 0.999;
  const WIRING_STRUCTURE_MAX = 0.01;
  const WIRING_STRUCTURE_MIN_SAMPLES = 200;
  for (const w of wiring) {
    const at = `${w.caseId} ${w.view} at ${w.at} (phase ${w.phase.toFixed(3)})`;
    check(!w.error, `compareBackends(${at}) error: ${w.error}`);
    check(
      w.structureAgreement >= WIRING_FRAME_MIN,
      `${at}: structureAgreement ${w.structureAgreement} < ${WIRING_FRAME_MIN}`,
    );
    for (const [name, s] of Object.entries(w.perStructure ?? {})) {
      if (s.samples < WIRING_STRUCTURE_MIN_SAMPLES) continue;
      const share = s.mismatched / s.samples;
      check(
        share <= WIRING_STRUCTURE_MAX,
        `${at}: ${name} ${s.mismatched} of ${s.samples} samples differ (${(100 * share).toFixed(2)} %)`,
      );
    }
  }

  // 6. Zero browser console errors or uncaught page errors
  check(
    consoleErrors.length === 0,
    `Browser console reported ${consoleErrors.length} error(s):\n${consoleErrors.join('\n')}`,
  );
  check(
    pageErrors.length === 0,
    `Browser reported ${pageErrors.length} uncaught page error(s):\n${pageErrors.join('\n')}`,
  );

  console.log('=== REAL WEBGL2 CPU <-> GPU PARITY SUMMARY ===');
  console.log('B-mode:', {
    structAgree: b.structureAgreement,
    tissueAgree: b.tissueAgreement,
    ampRelDiff: +b.ampRelDiff.toFixed(5),
    transDiff: +b.transDiff.toFixed(6),
  });
  console.log('ImageChain:', {
    idsAgree: img.idsAgreement,
    displayMaxDiff: img.displayMaxDiff,
    presentFracOver1: +img.presentFracOver1.toFixed(6),
  });
  for (const [label, d] of [
    ['PLAX normal', dop],
    ['A4C MR (+6dB, +0.20m/s)', dopMr],
    ['PLAX severe AR', dopAr],
    ['PLAX severe MS', dopMs],
  ]) {
    console.log(`Doppler ${label}:`, {
      detValidAgree: +d.detValidAgreement.toFixed(5),
      detVelMaxDiff: +d.detVelMaxDiff.toFixed(5),
      detPresentOver1: +d.detPresentFracOver1.toFixed(6),
      cpuBloodFill: +d.kasaiWf.cpuBloodFill.toFixed(3),
      gpuBloodFill: +d.kasaiWf.gpuBloodFill.toFixed(3),
      gpuClutterFill: +d.kasaiWf.gpuClutterFill.toFixed(3),
    });
  }
  console.log(`Wiring (${wiring.length} views): all >= ${WIRING_FRAME_MIN}`);
  for (const w of wiring) {
    console.log(`  ${w.caseId} ${w.view} ${w.at} @${w.phase.toFixed(3)}: ${w.structureAgreement?.toFixed(5)}`);
  }

  if (failures.length) {
    console.error('GPU PARITY CHECK FAILED:\n - ' + failures.join('\n - '));
    process.exitCode = 1;
  } else {
    console.log('GPU PARITY CHECK PASSED');
  }
} finally {
  await browser.close();
}
