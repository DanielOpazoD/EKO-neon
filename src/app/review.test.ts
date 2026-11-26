import assert from 'node:assert/strict';
import test from 'node:test';
import { acquisitionLimits } from '@/simulator/core/acquisitionInput';
import { baseInput } from '@/simulator/core/baseInput';
import type { SimInput } from '@/simulator/core/protocol';
import { DEFAULT_COLOR } from '@/simulator/doppler/color/colorDoppler';
import {
  colorWallFilterCutoffMps,
  COLOR_PACKET,
  KASAI_LIN_BASIS,
  KASAI_QUAD_BASIS,
  wallFilterShares,
  WF_MAX_NYQUIST,
  WF_MIN_NYQUIST,
} from '@/simulator/doppler/color/kasaiEstimator';
import { buildReviewReport, legacyColorWallFilterMps, parseReviewReport, reportToMarkdown } from './review';

type Shares = { mean: number; trend: number; quad: number };

/** The shares main set for a wall filter label: ramps in WF/Nyquist (kasaiEstimator.ts before c6a213d). */
function mainShares(wallFilterMps: number, scaleMps: number): Shares {
  const cut = Math.max(0, wallFilterMps / scaleMps);
  return {
    mean: Math.min(1, cut / 0.03),
    trend: Math.min(1, Math.max(0, (cut - 0.02) / 0.04)),
    quad: Math.min(1, Math.max(0, (cut - 0.055) / 0.06)),
  };
}

/** Power a unit tone at `frac`·Nyquist keeps through the packet regression with these shares, by projections. */
function toneKept(frac: number, s: Shares): number {
  const N = COLOR_PACKET;
  const xr = Array.from({ length: N }, (_, p) => Math.cos(Math.PI * frac * p));
  const xi = Array.from({ length: N }, (_, p) => Math.sin(Math.PI * frac * p));
  const bases: Array<[ArrayLike<number>, number]> = [
    [Array.from({ length: N }, () => 1 / Math.sqrt(N)), s.mean],
    [KASAI_LIN_BASIS, s.trend],
    [KASAI_QUAD_BASIS, s.quad],
  ];
  for (const [b, share] of bases) {
    let cr = 0;
    let ci = 0;
    for (let p = 0; p < N; p++) {
      cr += xr[p]! * b[p]!;
      ci += xi[p]! * b[p]!;
    }
    for (let p = 0; p < N; p++) {
      xr[p]! -= share * cr * b[p]!;
      xi[p]! -= share * ci * b[p]!;
    }
  }
  let pow = 0;
  for (let p = 0; p < N; p++) pow += xr[p]! ** 2 + xi[p]! ** 2;
  return pow / N;
}

/**
 * −3 dB cutoff (m/s) of main's filter for a label at a scale: first crossing of one half, scanned by 0.005 Nyquist then
 * bisected to the double's resolution. The response is a trigonometric polynomial of degree 7 in π·v/Nyquist, whose
 * shortest period is 2/7 Nyquist; scanned by 0.0005 it gave the very same cutoffs, three times slower.
 */
function mainCutoffMps(wallFilterMps: number, scaleMps: number): number {
  const s = mainShares(wallFilterMps, scaleMps);
  if (toneKept(0, s) >= 0.5) return 0;
  let lo = 0;
  let hi = 0.005;
  while (toneKept(hi, s) < 0.5) {
    lo = hi;
    hi += 0.005;
  }
  for (let i = 0; i < 60; i++) {
    const mid = 0.5 * (lo + hi);
    if (toneKept(mid, s) < 0.5) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi) * scaleMps;
}

/** A version 1 report as main wrote it: the console's input as it was, the wall filter a label. */
function v1Text(input: SimInput): string {
  return JSON.stringify({
    format: 'echotwin-review',
    version: 1,
    createdAt: '2026-11-20T10:00:00.000Z',
    author: 'usuario',
    app: { url: '' },
    caseId: 'normal-excellent-window',
    caseTitle: 'Normal, ventana excelente',
    mode: 'sandbox',
    workerMode: 'worker',
    input,
    frame: null,
    note: '',
    markers: [],
  });
}

const colorInput = (color: Partial<SimInput['color']>, over: Partial<SimInput> = {}): SimInput =>
  baseInput({ modality: 'color', color: { ...DEFAULT_COLOR, ...color }, ...over });

test('a version 1 report from main replays with the −3 dB cutoff and the very shares its image was formed with', () => {
  // main's default label, 0.06 m/s at 0.68 m/s: mean + trend + 55 % of the quadratic, −3 dB at 0.2046 m/s. Read as the
  // request it now is, it gave the 7.6 cm/s mean removal (A4C: 9317 coloured myocardial samples against main's 70)
  const r = parseReviewReport(v1Text(colorInput({ wallFilterMps: 0.06, scaleMps: 0.68 })));
  assert.ok(r);
  assert.equal(r.version, 2);
  const migrated = r.input.color.wallFilterMps;
  const old = mainCutoffMps(0.06, 0.68);
  assert.ok(Math.abs(old - 0.2046) < 1e-4, `main's default cut at ${old}`);
  // two bisections of the same response, the code's on closed-form tone powers and this one on projections: they agree
  // to the double's resolution, far under 1e-9 m/s
  assert.ok(Math.abs(migrated - old) < 1e-9, `migrated ${migrated}, main's cutoff ${old}`);
  // and it is the same filter: the shares solved for that cutoff are main's (1e-9 of a share moves the response by less)
  const replay = wallFilterShares(migrated, 0.68);
  const main = mainShares(0.06, 0.68);
  for (const k of ['mean', 'trend', 'quad'] as const)
    assert.ok(Math.abs(replay[k] - main[k]) < 1e-9, `${k}: replay ${replay[k]}, main ${main[k]}`);
  assert.equal(colorWallFilterCutoffMps(migrated, 0.68), migrated, 'within what the packet allows');
});

/** The cutoff (m/s) a request filters with at a Nyquist scale: bounded to what the packet allows there. */
const filtersAt = (requestMps: number, scaleMps: number) => colorWallFilterCutoffMps(requestMps, scaleMps);

test('a version 1 report at another scale filters there as main did, and asks for what filters nearest main at 0.68 m/s', () => {
  const migrate = (input: SimInput) => parseReviewReport(v1Text(input))!.input.color.wallFilterMps;
  const sharesAt = (request: number, scale: number, shares: Shares, label: string) => {
    const replay = wallFilterShares(request, scale);
    for (const k of ['mean', 'trend', 'quad'] as const)
      assert.ok(Math.abs(replay[k] - shares[k]) < 1e-9, `${label}, ${k}: ${replay[k]} instead of ${shares[k]}`);
  };
  // 0.04 at 1 m/s (a 10 cm box lets the PRF reach it): the mean and half the trend, −3 dB at 0.191 m/s, inside the
  // range the packet allows; the only request that filters there as main did
  const inside = colorInput({ wallFilterMps: 0.04, scaleMps: 1.0, boxRMaxCm: 10 });
  assert.equal(acquisitionLimits(inside).color.scaleMps, 1.0, 'a scale the PRF allows');
  const insideReq = migrate(inside);
  assert.ok(Math.abs(insideReq - mainCutoffMps(0.04, 1.0)) < 1e-9, `0.04 at 1: ${insideReq}`);
  assert.ok(Math.abs(insideReq - 0.1908) < 1e-4);
  sharesAt(insideReq, 1.0, { mean: 1, trend: 0.5, quad: 0 }, '0.04 at 1');

  // 0.06 at 0.35 m/s, the task colour-low-scale: main's ramps saturate at the steepest filter, 0.32 Nyquist. Every
  // request from 11.2 cm/s up filters so there; asked as 11.2, the replay went on filtering at 11.2 once the scale was
  // back at 0.68 (A4C, CPU, 44 frames: 7073 coloured myocardial samples, 70 in main). It asks for main's cutoff at 0.68
  const low = colorInput({ wallFilterMps: 0.06, scaleMps: 0.35 });
  assert.equal(acquisitionLimits(low).color.scaleMps, 0.35);
  const lowReq = migrate(low);
  assert.ok(Math.abs(mainCutoffMps(0.06, 0.35) - WF_MAX_NYQUIST * 0.35) < 1e-9, 'saturated at 0.35');
  assert.ok(Math.abs(filtersAt(lowReq, 0.35) - mainCutoffMps(0.06, 0.35)) < 1e-9, `at 0.35: ${lowReq}`);
  sharesAt(lowReq, 0.35, { mean: 1, trend: 1, quad: 1 }, '0.06 at 0.35');
  assert.ok(Math.abs(filtersAt(lowReq, 0.68) - mainCutoffMps(0.06, 0.68)) < 1e-9, `at 0.68: ${lowReq}`);
  assert.ok(Math.abs(filtersAt(lowReq, 0.68) - 0.2046) < 1e-4);

  // 0.02 at 1 m/s: main's partial mean removal cut at 10.2 cm/s, under the 11.2 of the whole one it is bounded to.
  // Every request up to 11.2 filters so there; the one nearest main at 0.68 is its 10.1 cm/s there
  const under = colorInput({ wallFilterMps: 0.02, scaleMps: 1.0, boxRMaxCm: 10 });
  const underReq = migrate(under);
  assert.ok(mainCutoffMps(0.02, 1.0) < WF_MIN_NYQUIST * 1.0, 'under the mean removal at 1 m/s');
  assert.ok(Math.abs(filtersAt(underReq, 1.0) - WF_MIN_NYQUIST * 1.0) < 1e-12, `at 1: ${filtersAt(underReq, 1.0)}`);
  sharesAt(underReq, 1.0, { mean: 1, trend: 0, quad: 0 }, '0.02 at 1');
  assert.ok(Math.abs(filtersAt(underReq, 0.68) - mainCutoffMps(0.02, 0.68)) < 1e-9, `at 0.68: ${underReq}`);
  // under main's 1 cm/s steps, a label whose filter cut nothing at 0.68 (0.005: a quarter of the mean) but cut at
  // 2.9 cm/s at 0.35 replays the mean removal there, not off
  const faint = migrate(colorInput({ wallFilterMps: 0.005, scaleMps: 0.35 }));
  assert.equal(mainCutoffMps(0.005, 0.68), 0);
  assert.ok(mainCutoffMps(0.005, 0.35) > 0);
  assert.ok(Math.abs(filtersAt(faint, 0.35) - WF_MIN_NYQUIST * 0.35) < 1e-12, `0.005 at 0.35: ${faint}`);

  // the scale is the one the replay acquires with: a box at 22 cm of a 24 cm image limits the PRF to 0.504 m/s, where
  // 0.06 saturates too (0.119 Nyquist); at the default 13 cm a scale of 1 m/s is acquired at 0.816, inside
  const deep = colorInput(
    { wallFilterMps: 0.06, scaleMps: 0.68, boxRMaxCm: 22 },
    { settings: { ...baseInput().settings, depthCm: 24 } },
  );
  const deepScale = acquisitionLimits(deep).color.scaleMps;
  assert.ok(deepScale < 0.68, `PRF-limited: ${deepScale}`);
  const deepReq = migrate(deep);
  assert.ok(Math.abs(filtersAt(deepReq, deepScale) - filtersAt(mainCutoffMps(0.06, deepScale), deepScale)) < 1e-9);
  assert.ok(Math.abs(filtersAt(deepReq, 0.68) - mainCutoffMps(0.06, 0.68)) < 1e-9, `deep box, at 0.68: ${deepReq}`);
  const fast = colorInput({ wallFilterMps: 0.04, scaleMps: 1.0 });
  const fastScale = acquisitionLimits(fast).color.scaleMps;
  assert.ok(fastScale < 1.0, `PRF-limited: ${fastScale}`);
  assert.ok(Math.abs(migrate(fast) - mainCutoffMps(0.04, fastScale)) < 1e-9, `acquired at ${fastScale}`);

  // a partial mean removal at 0.68 (0.01: a third of the mean) is asked as its −3 dB point and filters with the whole
  // mean removal, the lowest filter that cancels stationary tissue; a label of 0 stays off
  const partialCut = migrate(colorInput({ wallFilterMps: 0.01, scaleMps: 0.68 }));
  assert.ok(Math.abs(partialCut - mainCutoffMps(0.01, 0.68)) < 1e-9);
  assert.ok(Math.abs(filtersAt(partialCut, 0.68) - WF_MIN_NYQUIST * 0.68) < 1e-12);
  assert.equal(migrate(colorInput({ wallFilterMps: 0, scaleMps: 0.68 })), 0);
});

test('over main’s controls a version 1 report filters at its scale as main did, at 0.68 m/s as near main as that allows', () => {
  // main's labels 0.01–0.30 m/s by its slider's 1 cm/s steps, at colour scales 0.15–1.2 m/s (an 8 cm box lets the PRF
  // reach them all). The requests that filter at the report's scale as main did are found here by search: main's
  // cutoff there, off, and 1 mm/s steps up to 0.5 m/s, kept if they filter there alike. None may filter nearer main's
  // cutoff at 0.68 than the migrated request. It misses it by at most the mean removal's 7.6 cm/s, where main's filter
  // cut nothing at the report's scale (0.01 at 1.15 m/s) and the replay is off; the cutoff at the report's scale
  // missed it by up to 14.2 cm/s
  const seen = { inside: 0, steepest: 0, meanRemoval: 0, off: 0 };
  let worst = 0;
  for (let li = 1; li <= 30; li++) {
    const label = li / 100;
    const target = filtersAt(mainCutoffMps(label, 0.68), 0.68);
    const miss = (r: number) => Math.abs(filtersAt(r, 0.68) - target);
    for (let si = 15; si <= 120; si += 5) {
      const input = colorInput({ wallFilterMps: label, scaleMps: si / 100, boxRMaxCm: 8 });
      const scale = acquisitionLimits(input).color.scaleMps;
      assert.equal(scale, si / 100);
      const main = mainCutoffMps(label, scale);
      const image = filtersAt(main, scale);
      const request = legacyColorWallFilterMps(input);
      const where = `${label} at ${scale}: request ${request}`;
      const replay = filtersAt(request, scale);
      assert.ok(Math.abs(replay - image) < 1e-9, `${where} filters at ${replay}, main at ${image}`);
      for (let k = -1; k <= 500; k++) {
        const r = k < 0 ? main : k / 1000;
        if (Math.abs(filtersAt(r, scale) - image) < 1e-9)
          assert.ok(miss(request) <= miss(r) + 1e-9, `${where} misses at 0.68 by ${miss(request)}, ${r} by ${miss(r)}`);
      }
      worst = Math.max(worst, miss(request));
      if (main <= 0) seen.off++;
      else if (main >= WF_MAX_NYQUIST * scale - 1e-9) seen.steepest++;
      else if (main <= WF_MIN_NYQUIST * scale) seen.meanRemoval++;
      else seen.inside++;
    }
  }
  for (const [k, n] of Object.entries(seen)) assert.ok(n > 0, `no pair ${k}: ${JSON.stringify(seen)}`);
  assert.ok(worst <= WF_MIN_NYQUIST * 0.68 + 1e-9, `misses main at 0.68 by up to ${worst}`);
});

test('a version 1 colour M-mode report migrates by the same rule, at the Nyquist the strip acquires with', () => {
  // main's strip cut at the label itself, but the request is one for the strip and the 2D colour after it: kept as the
  // label 0.06, it left the 2D colour on the mean removal (7.5 cm/s) for the rest of the session. At 16 cm the strip
  // acquires at 0.676 m/s, where main's ramps cut at 0.2040 m/s: mean + trend + 56 % of the quadratic
  const cmm = baseInput({ modality: 'cmm', color: { ...DEFAULT_COLOR, wallFilterMps: 0.06 } });
  const scale = acquisitionLimits(cmm).color.scaleMps;
  assert.ok(scale < cmm.color.scaleMps, `the depth limits the strip's PRF: ${scale}`);
  const r = parseReviewReport(v1Text(cmm))!;
  assert.equal(r.version, 2);
  const migrated = r.input.color.wallFilterMps;
  assert.ok(Math.abs(migrated - mainCutoffMps(0.06, scale)) < 1e-9, `migrated ${migrated}`);
  assert.ok(Math.abs(migrated - 0.204) < 1e-4, `main's 2D cut that label at ${migrated}`);
  const replay = wallFilterShares(migrated, scale);
  const main = mainShares(0.06, scale);
  for (const k of ['mean', 'trend', 'quad'] as const)
    assert.ok(Math.abs(replay[k] - main[k]) < 1e-9, `${k}: replay ${replay[k]}, main ${main[k]}`);
});

test('a version 2 report reads back unchanged, from JSON and from the pasted markdown', () => {
  const input = colorInput({ wallFilterMps: 0.12, scaleMps: 0.35 });
  const report = buildReviewReport({
    caseId: 'normal-excellent-window',
    caseTitle: 'Normal, ventana excelente',
    mode: 'sandbox',
    workerMode: 'worker',
    input,
    hud: null,
    note: '',
    markers: [],
  });
  assert.equal(report.version, 2);
  const md = reportToMarkdown(report);
  for (const text of [JSON.stringify(report), md]) {
    const back = parseReviewReport(text);
    assert.ok(back);
    assert.equal(back.version, 2);
    assert.deepEqual(back.input, input);
  }
  // the readable line says the cutoff that filters: the 12 cm/s asked, bounded to 0.32 Nyquist at 0.35 m/s
  assert.ok(md.includes('filtro de pared 11.2 cm/s'), md.split('\n').find((l) => l.startsWith('Color:')));
});

test('the readable report of a colour M-mode says its colour scale and wall filter, and not PW’s scale', () => {
  // At 20 cm the strip acquires at 0.551 m/s, and the 30 cm/s asked filters at 0.32 of it, 17.7 cm/s. The line used to
  // read «Cursor 0° · compuerta 9.0 cm (0.4 cm) · escala 1.00 m/s · barrido 50 mm/s»: PW's gate and scale, which the
  // strip has nothing of, and neither its Nyquist nor its filter
  const input = baseInput({
    modality: 'cmm',
    settings: { ...baseInput().settings, depthCm: 20 },
    color: { ...DEFAULT_COLOR, wallFilterMps: 0.3 },
  });
  const scale = acquisitionLimits(input).color.scaleMps;
  assert.ok(Math.abs(scale - 0.551) < 1e-3, `the depth limits the strip's PRF: ${scale}`);
  const md = reportToMarkdown(
    buildReviewReport({
      caseId: 'normal-excellent-window',
      caseTitle: 'Normal, ventana excelente',
      mode: 'sandbox',
      workerMode: 'worker',
      input,
      hud: null,
      note: '',
      markers: [],
    }),
  );
  const line = md.split('\n').find((l) => l.startsWith('Cursor'));
  assert.equal(line, 'Cursor 0° · escala color 0.55 m/s · filtro de pared 17.7 cm/s · barrido 50 mm/s');
  assert.equal((100 * WF_MAX_NYQUIST * scale).toFixed(1), '17.7', 'the steepest filter at that Nyquist');
});
