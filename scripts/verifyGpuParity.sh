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
  return JSON.stringify({ b, img, dop, dopMr });
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
const { b, img, dop, dopMr } = data;
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

// 3. Deferred Color Doppler (PLAX normal: deterministic + Kasai 3-basis Legendre WF=6 vs WF=2)
check(!dop.error, `compareDeferredDoppler(plax) error: ${dop.error}`);
check(dop.detValidAgreement >= 0.99, `plax detValidAgreement ${dop.detValidAgreement} < 0.99`);
check(dop.detVelMaxDiff <= 0.015, `plax detVelMaxDiff ${dop.detVelMaxDiff} > 0.015`);
check(dop.detVarMaxDiff <= 0.02, `plax detVarMaxDiff ${dop.detVarMaxDiff} > 0.02`);
check(dop.detPresentFracOver1 <= 0.005, `plax detPresentFracOver1 ${dop.detPresentFracOver1} > 0.005`);
check(Math.abs(dop.kasaiWf6.cpuBloodFill - dop.kasaiWf6.gpuBloodFill) <= 0.08, `plax wf6 blood fill drift cpu=${dop.kasaiWf6.cpuBloodFill.toFixed(3)} gpu=${dop.kasaiWf6.gpuBloodFill.toFixed(3)}`);
check(dop.kasaiWf6.gpuClutterFill <= 0.06, `plax wf6 GPU clutter not suppressed: ${dop.kasaiWf6.gpuClutterFill.toFixed(3)}`);
check(dop.kasaiWf2.gpuClutterFill >= dop.kasaiWf6.gpuClutterFill, `plax wf2 GPU clutter (${dop.kasaiWf2.gpuClutterFill.toFixed(3)}) < wf6 (${dop.kasaiWf6.gpuClutterFill.toFixed(3)})`);

// 4. Deferred Color Doppler (A4C severe MR with overgain +6 dB & baseline shift +0.20 m/s)
check(!dopMr.error, `compareDeferredDoppler(a4c-mr) error: ${dopMr.error}`);
check(dopMr.detValidAgreement >= 0.99, `a4c-mr detValidAgreement ${dopMr.detValidAgreement} < 0.99`);
check(dopMr.detVelMaxDiff <= 0.02, `a4c-mr detVelMaxDiff ${dopMr.detVelMaxDiff} > 0.02`);
check(dopMr.detPresentFracOver1 <= 0.005, `a4c-mr detPresentFracOver1 ${dopMr.detPresentFracOver1} > 0.005`);
check(Math.abs(dopMr.kasaiWf6.cpuBloodFill - dopMr.kasaiWf6.gpuBloodFill) <= 0.08, `a4c-mr wf6 blood fill drift cpu=${dopMr.kasaiWf6.cpuBloodFill.toFixed(3)} gpu=${dopMr.kasaiWf6.gpuBloodFill.toFixed(3)}`);

// 5. Zero browser console errors
check(consoleErr.includes("Errors: 0"), `Browser console reported errors:\n${consoleErr}`);

console.log("=== REAL WEBGL2 CPU <-> GPU PARITY SUMMARY ===");
console.log("B-mode:", { structAgree: b.structureAgreement, tissueAgree: b.tissueAgreement, ampRelDiff: +b.ampRelDiff.toFixed(5), transDiff: +b.transDiff.toFixed(6) });
console.log("ImageChain:", { idsAgree: img.idsAgreement, displayMaxDiff: img.displayMaxDiff, presentFracOver1: +img.presentFracOver1.toFixed(6) });
console.log("Doppler PLAX:", dop);
console.log("Doppler A4C MR (+6dB, +0.20m/s):", dopMr);

if (failures.length) {
  console.error("GPU PARITY CHECK FAILED:\n - " + failures.join("\n - "));
  process.exit(1);
}
console.log("GPU PARITY CHECK PASSED");
' "$RAW" "$CONSOLE_ERR"
