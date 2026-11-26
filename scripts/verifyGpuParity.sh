#!/usr/bin/env bash
# Real WebGL2 (SwiftShader/GPU) CPU <-> GPU parity verification via headless Chromium.
# Requires the dev server running on http://localhost:3000 (or URL passed as $1).
set -euo pipefail

URL="${1:-http://localhost:3000}"
trap 'playwright-cli close >/dev/null 2>&1 || true' EXIT

playwright-cli open "$URL" >/dev/null
sleep 2

RAW=$(playwright-cli eval "async () => {
  const b = await window.__echotwin.compareBackends('plax', 0.25, 'normal-excellent-window', 'low');
  const img = await window.__echotwin.compareImageChain('plax', 0.25, 'normal-excellent-window', 'low');
  const dop = await window.__echotwin.compareDeferredDoppler('plax', 0.22, 'normal-excellent-window', 'low');
  const dopMr = await window.__echotwin.compareDeferredDoppler('a4c', 0.22, 'mvp-primary-mr', 'low', { gainDb: 6, baselineShiftMps: 0.2 });
  // the GPU wiring of the pericardium, the regional wall motion and the aortic opening, each at the phase of the beat
  // where it acts most (read from the sources the dev server serves, as compareBackends builds the case): end-systole,
  // the most contracted phase (the regional offset, the RV collapse of the tamponade); the largest swing of the
  // tamponade heart in its sac, where an error of the point read before or after the swing is largest (0.36 cm at 0.25,
  // against 0.15 cm at end-systole, 0.57); the aortic valve most open in severe stenosis (0.28, mid-ejection: the cusps
  // open to 0.58), where end-systole found it nearly closed (0.455: avOpen 0.14, the cusps at 0.08)
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
    let phase = 0, most = -Infinity;
    for (let i = 0; i <= 400; i++) {
      const v = PHASE_OF[at](cycleStateAt(tables, i / 400), heart);
      if (v > most) { most = v; phase = i / 400; }
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
    wiring.push({ caseId, view, at, phase, error: r.error, structureAgreement: r.structureAgreement, perStructure: r.perStructure });
  }
  return JSON.stringify({ b, img, dop, dopMr, wiring });
}")

CONSOLE_ERR=$(playwright-cli console error)

node -e '
const raw = process.argv[1];
const consoleErr = process.argv[2];
const start = raw.indexOf("\"{");
const end = raw.lastIndexOf("}\"");
if (start < 0 || end < 0) {
  console.error("Failed to parse Playwright JSON payload:\n", raw);
  process.exit(1);
}
const data = JSON.parse(JSON.parse(raw.slice(start, end + 2)));
const { b, img, dop, dopMr, wiring } = data;
const failures = [];
const check = (cond, msg) => { if (!cond) failures.push(msg); };

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

// 3. Deferred Color Doppler (PLAX normal: deterministic + Kasai 3-basis Legendre, default wall filter vs the lowest the packet allows; no apostrophe in this single-quoted program)
check(!dop.error, `compareDeferredDoppler(plax) error: ${dop.error}`);
check(dop.detValidAgreement >= 0.99, `plax detValidAgreement ${dop.detValidAgreement} < 0.99`);
check(dop.detVelMaxDiff <= 0.015, `plax detVelMaxDiff ${dop.detVelMaxDiff} > 0.015`);
check(dop.detVarMaxDiff <= 0.02, `plax detVarMaxDiff ${dop.detVarMaxDiff} > 0.02`);
check(dop.detPresentFracOver1 <= 0.005, `plax detPresentFracOver1 ${dop.detPresentFracOver1} > 0.005`);
check(Math.abs(dop.kasaiWf.cpuBloodFill - dop.kasaiWf.gpuBloodFill) <= 0.08, `plax default-WF blood fill drift cpu=${dop.kasaiWf.cpuBloodFill.toFixed(3)} gpu=${dop.kasaiWf.gpuBloodFill.toFixed(3)}`);
check(dop.kasaiWf.gpuClutterFill <= 0.06, `plax default-WF GPU clutter not suppressed: ${dop.kasaiWf.gpuClutterFill.toFixed(3)}`);
check(dop.kasaiMinWf.gpuClutterFill >= dop.kasaiWf.gpuClutterFill, `plax min-WF GPU clutter (${dop.kasaiMinWf.gpuClutterFill.toFixed(3)}) < default-WF (${dop.kasaiWf.gpuClutterFill.toFixed(3)})`);

// 4. Deferred Color Doppler (A4C severe MR with overgain +6 dB & baseline shift +0.20 m/s)
check(!dopMr.error, `compareDeferredDoppler(a4c-mr) error: ${dopMr.error}`);
check(dopMr.detValidAgreement >= 0.99, `a4c-mr detValidAgreement ${dopMr.detValidAgreement} < 0.99`);
check(dopMr.detVelMaxDiff <= 0.02, `a4c-mr detVelMaxDiff ${dopMr.detVelMaxDiff} > 0.02`);
check(dopMr.detPresentFracOver1 <= 0.005, `a4c-mr detPresentFracOver1 ${dopMr.detPresentFracOver1} > 0.005`);
check(Math.abs(dopMr.kasaiWf.cpuBloodFill - dopMr.kasaiWf.gpuBloodFill) <= 0.08, `a4c-mr default-WF blood fill drift cpu=${dopMr.kasaiWf.cpuBloodFill.toFixed(3)} gpu=${dopMr.kasaiWf.gpuBloodFill.toFixed(3)}`);

// 5. The GPU wiring of the pericardium, the regional motion and the aortic opening, at the phases above. A wiring error
// moves 0.2-3 % of a frame, which the 0.99 above lets through: with the three wiring mutations of review 3 injected, the
// frame-wide agreement fell from 0.9999 to 0.9717 (inferior-rwma PSAX-MV, the RV epicardium read from the crescent),
// 0.9881 (pulmonary-hypertension-rv PSAX-PM, the same), 0.9976 (inferior-rwma PSAX-PM, the regional offset inverted in
// rvRadii) and 0.9972 (tamponade A2C, the oblique sinus read after the swing), all of it in the structures where the
// wiring acts. So each frame is bounded as a whole and per structure (compareBackends perStructure: the samples the CPU
// labels so and how many of them the GPU labels otherwise). Calibrated in WebGL2, low quality, on the clean branch with
// the end-systole views: frame agreement 0.99993 to 1, the worst structure of 200 samples or more 0.17 % (the RV cavity
// in the subcostal view) and every other 0 %. The bounds stand between the clean branch and the mildest mutation: the
// frame at 0.999 (9e-4 under the clean worst, 1.4e-3 over 0.9976), a structure at 1 % (6 times the clean worst). The
// swing and aortic-opening phases were added after that measurement: their table is to be read once in the browser.
const WIRING_FRAME_MIN = 0.999;
const WIRING_STRUCTURE_MAX = 0.01;
const WIRING_STRUCTURE_MIN_SAMPLES = 200;
for (const w of wiring) {
  const at = `${w.caseId} ${w.view} at ${w.at} (phase ${w.phase.toFixed(3)})`;
  check(!w.error, `compareBackends(${at}) error: ${w.error}`);
  check(w.structureAgreement >= WIRING_FRAME_MIN, `${at}: structureAgreement ${w.structureAgreement} < ${WIRING_FRAME_MIN}`);
  for (const [name, s] of Object.entries(w.perStructure ?? {})) {
    if (s.samples < WIRING_STRUCTURE_MIN_SAMPLES) continue;
    const share = s.mismatched / s.samples;
    check(share <= WIRING_STRUCTURE_MAX, `${at}: ${name} ${s.mismatched} of ${s.samples} samples differ (${(100 * share).toFixed(2)} %)`);
  }
}

// 6. Zero browser console errors
check(consoleErr.includes("Errors: 0"), `Browser console reported errors:\n${consoleErr}`);

console.log("=== REAL WEBGL2 CPU <-> GPU PARITY SUMMARY ===");
console.log("B-mode:", { structAgree: b.structureAgreement, tissueAgree: b.tissueAgreement, ampRelDiff: +b.ampRelDiff.toFixed(5), transDiff: +b.transDiff.toFixed(6) });
console.log("ImageChain:", { idsAgree: img.idsAgreement, displayMaxDiff: img.displayMaxDiff, presentFracOver1: +img.presentFracOver1.toFixed(6) });
console.log("Doppler PLAX:", dop);
console.log("Doppler A4C MR (+6dB, +0.20m/s):", dopMr);
console.log("Wiring: frame agreement and the structures of >= " + WIRING_STRUCTURE_MIN_SAMPLES + " samples that differ most");
for (const w of wiring) {
  const worst = Object.entries(w.perStructure ?? {})
    .filter(([, s]) => s.samples >= WIRING_STRUCTURE_MIN_SAMPLES)
    .map(([name, s]) => [name, s.mismatched / s.samples, s.samples])
    .sort((a, c) => c[1] - a[1])
    .slice(0, 4)
    .map(([name, share, n]) => `${name} ${(100 * share).toFixed(2)} % of ${n}`);
  console.log(`  ${w.caseId} ${w.view} ${w.at} @${w.phase.toFixed(3)}: ${w.structureAgreement?.toFixed(5)} | ${worst.join(", ")}`);
}

if (failures.length) {
  console.error("GPU PARITY CHECK FAILED:\n - " + failures.join("\n - "));
  process.exit(1);
}
console.log("GPU PARITY CHECK PASSED");
' "$RAW" "$CONSOLE_ERR"
