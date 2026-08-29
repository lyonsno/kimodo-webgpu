#!/usr/bin/env node
/**
 * Headless WebGPU smoke test for Kimodo.
 *
 * Launches Chrome with WebGPU enabled, loads the app, waits for weight loading,
 * triggers a generation, and captures all console output.
 *
 * Usage:
 *   node tools/headless_smoke.mjs [--url http://localhost:5175] [--prompt "a person walks"]
 *
 * Requires:
 *   - Chrome installed at /Applications/Google Chrome.app
 *   - Vite dev server running (npm run dev)
 *   - Text embedding server running (python tools/embed_server.py --port 8098)
 */

import puppeteer from 'puppeteer-core';
import {
  validateRouteReceipt,
  assertAuthoritativeRouteReceipt,
} from '@kaminos/webgpu-inference-kit';

/** Wait until a generation newer than priorId is visibly in flight; null if none appears. */
async function waitForNewGeneration(page, priorId, timeoutMs = 15000) {
  try {
    await page.waitForFunction(
      (prior) => {
        const id = window.__kimodoLastReceipt?.generationId;
        return id != null && id !== prior;
      },
      { timeout: timeoutMs },
      priorId,
    );
  } catch {
    return null;
  }
  return page.evaluate(() => window.__kimodoLastReceipt.generationId);
}

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const DEFAULT_URL = 'http://localhost:5175';
const DEFAULT_PROMPT = 'a person walks forward and waves';

const args = process.argv.slice(2);
const url = args.find((a, i) => args[i - 1] === '--url') || DEFAULT_URL;
const prompt = args.find((a, i) => args[i - 1] === '--prompt') || DEFAULT_PROMPT;

async function main() {
  console.log(`[smoke] Launching Chrome with WebGPU...`);

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: false, // WebGPU requires a real GPU context
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan,UseSkiaRenderer',
      '--disable-vulkan-surface',
      '--use-angle=metal',
      '--no-sandbox',
    ],
  });

  const page = await browser.newPage();

  // Capture all console output
  const logs = [];
  page.on('console', msg => {
    const text = msg.text();
    logs.push({ type: msg.type(), text });
    // Print debug lines immediately
    if (text.includes('[')) {
      console.log(`  [browser] ${text}`);
    }
  });

  page.on('pageerror', err => {
    console.error(`  [browser error] ${err.message}`);
  });

  console.log(`[smoke] Navigating to ${url}...`);
  await page.goto(url, { waitUntil: 'networkidle0', timeout: 120000 });

  // Wait for weights to load (status text changes to "Ready")
  console.log(`[smoke] Waiting for weight loading...`);
  try {
    await page.waitForFunction(
      () => document.getElementById('status')?.textContent?.includes('Ready'),
      { timeout: 180000 },
    );
    console.log(`[smoke] Weights loaded.`);
  } catch {
    const status = await page.$eval('#status', el => el.textContent).catch(() => 'unknown');
    console.log(`[smoke] Weight loading timed out. Status: ${status}`);
    await browser.close();
    process.exit(1);
  }

  // Set prompt and trigger generation
  console.log(`[smoke] Setting prompt: "${prompt}"`);
  await page.$eval('#prompt', (el, p) => { el.value = p; }, prompt);
  await page.$eval('#duration', el => { el.value = '3'; }); // short for testing
  await page.$eval('#steps', el => { el.value = '100'; });

  console.log(`[smoke] Triggering generation...`);
  const priorId = await page.evaluate(() => window.__kimodoLastReceipt?.generationId ?? 0);
  await page.click('#generate-btn');

  // Prove the click actually advanced to a NEW generation before watching for
  // its outcome. Sampling whatever id is visible after the click would adopt a
  // prior run's id if the click regressed to a no-op — and then bless exactly
  // the stale evidence the freshness check exists to reject.
  const expectedId = await waitForNewGeneration(page, priorId);
  if (expectedId === null) {
    console.log('[smoke] FAIL — the click did not start a new generation.');
    await browser.close();
    process.exit(1);
  }

  let timedOut = false;
  try {
    // Terminal-state decisions go through the app's shipped classifier — the
    // same one the app and every other watcher use — never a local copy.
    await page.waitForFunction(
      (id) => window.__kimodoGenerationState(id).done,
      { timeout: 120000 },
      expectedId,
    );
  } catch {
    timedOut = true;
    console.log(`[smoke] Generation timed out — classifier never reached a terminal state.`);
  }

  const finalStatus = await page.$eval('#status', el => el.textContent).catch(() => 'unknown');
  const finalInfo = await page.$eval('#info', el => el.textContent).catch(() => 'unknown');

  console.log(`\n[smoke] Final status: ${finalStatus}`);
  console.log(`[smoke] Final info: ${finalInfo}`);

  // Print all debug logs
  console.log(`\n[smoke] === All debug logs ===`);
  for (const log of logs) {
    if (log.text.includes('[debug]') || log.text.includes('[weights]') || log.text.includes('kimodo')) {
      console.log(`  ${log.text}`);
    }
  }

  // Inspect the structured receipt, not console strings. The previous NaN
  // checks scanned for "Body output"/"Root output" messages emitted only by
  // renderSkeleton(), which the live route never calls — so both flags were
  // inert and a NaN-producing run passed cleanly.
  const receipt = await page.evaluate(() => window.__kimodoLastReceipt ?? null);

  const verdict = timedOut
    ? { done: false, ok: false }
    : await page.evaluate((id) => window.__kimodoGenerationState(id), expectedId);

  // The receipt must satisfy the kit contract at the authority level it
  // claims — the authoritative consumer, not only the structural validator —
  // and must identify the actual weights, not a placeholder.
  let kitOk = false, kitDetail = '';
  let weightsIdentified = false;
  if (receipt && receipt.status === 'real') {
    const v = validateRouteReceipt(receipt);
    if (!v.ok) kitDetail = v.errors.join('; ');
    else {
      try { assertAuthoritativeRouteReceipt(receipt); kitOk = true; }
      catch (e) { kitDetail = e.message; }
    }
    weightsIdentified = /^[0-9a-f]{64}$/.test(receipt?.model?.weightsHash ?? '');
  }

  const positivePassed = !timedOut && verdict.done && verdict.ok
    && kitOk && weightsIdentified;
  console.log(`\n[smoke] === Positive result ===`);
  console.log(`  Classifier verdict: done=${verdict.done} ok=${verdict.ok}`);
  console.log(`  Kit-authoritative receipt: ${kitOk}${kitDetail ? ` (${kitDetail})` : ''}`);
  console.log(`  Weights identified (sha256): ${weightsIdentified}`);
  console.log(`  Timed out: ${timedOut}`);
  console.log(`  Status: ${positivePassed ? 'PASS' : 'FAIL'}`);

  // Failure-path probe: point the app at a dead endpoint and require a
  // terminal FAILED receipt for a NEW generation. This exercises, live, the
  // path a prior harness revision turned into a 120-second silent timeout.
  console.log(`\n[smoke] === Failure-path probe ===`);
  await page.$eval('#server-url', (el) => { el.value = 'http://127.0.0.1:9'; });
  const probePrior = await page.evaluate(() => window.__kimodoLastReceipt?.generationId ?? 0);
  await page.click('#generate-btn');
  const probeId = await waitForNewGeneration(page, probePrior);
  let probePassed = false;
  if (probeId === null) {
    console.log('  FAIL — probe click did not start a new generation.');
  } else {
    try {
      await page.waitForFunction(
        (id) => window.__kimodoGenerationState(id).done,
        { timeout: 30000 },
        probeId,
      );
      const pv = await page.evaluate((id) => window.__kimodoGenerationState(id), probeId);
      const pr = await page.evaluate(() => window.__kimodoLastReceipt);
      probePassed = pv.done && pv.ok === false && pr.status === 'failed'
        && pr.generationId === probeId && !!pr.phase;
      console.log(`  Terminal: done=${pv.done} ok=${pv.ok} status=${pr.status} phase=${pr.phase ?? 'n/a'}`);
    } catch {
      console.log('  FAIL — dead endpoint produced no terminal state (silent timeout).');
    }
  }
  console.log(`  Status: ${probePassed ? 'PASS' : 'FAIL'}`);

  const failed = !positivePassed || !probePassed;
  console.log(`\n[smoke] === Overall: ${failed ? 'FAIL' : 'PASS'} ===`);
  await browser.close();
  process.exit(failed ? 1 : 0);
}

main().catch(err => {
  console.error(`[smoke] Fatal: ${err.message}`);
  process.exit(1);
});
