#!/usr/bin/env node
/**
 * Filmstrip witness — generates motion via WebGPU, captures every Nth frame
 * of the skeleton renderer, composites into a single filmstrip image.
 *
 * Usage:
 *   node tools/filmstrip_smoke.mjs [--url http://localhost:5176] [--prompt "..."] [--every 10] [--out filmstrip.png]
 */

import puppeteer from 'puppeteer-core';
import { writeFileSync } from 'fs';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const args = process.argv.slice(2);
const getArg = (flag, def) => { const i = args.indexOf(flag); return i >= 0 && args[i+1] ? args[i+1] : def; };
const url = getArg('--url', 'http://localhost:5176');
const prompt = getArg('--prompt', 'a person walks forward and waves');
const every = parseInt(getArg('--every', '6'));
const outPath = getArg('--out', '/tmp/kimodo-filmstrip.png');

async function main() {
  console.log(`[filmstrip] Launching Chrome...`);
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: false,
    args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--no-sandbox'],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 800 });

  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('[kimodo') || text.includes('error') || text.includes('Error'))
      console.log(`  [browser] ${text}`);
  });

  console.log(`[filmstrip] Navigating to ${url}...`);
  await page.goto(url, { waitUntil: 'networkidle0', timeout: 120000 });

  // Wait for weights
  console.log(`[filmstrip] Waiting for weights...`);
  await page.waitForFunction(
    () => document.getElementById('status')?.textContent?.includes('Ready'),
    { timeout: 180000 },
  );

  // Configure and generate
  await page.$eval('#prompt', (el, p) => { el.value = p; }, prompt);
  await page.$eval('#duration', el => { el.value = '6'; });
  await page.$eval('#steps', el => { el.value = '50'; });

  console.log(`[filmstrip] Generating: "${prompt}"...`);
  const priorId = await page.evaluate(() => window.__kimodoLastReceipt?.generationId ?? 0);
  await page.click('#generate-btn');

  // Prove the click advanced to a NEW generation; adopting whatever id is
  // visible after the click would bless a prior run's evidence if the click
  // regressed to a no-op.
  let expectedId = null;
  try {
    await page.waitForFunction(
      (prior) => {
        const id = window.__kimodoLastReceipt?.generationId;
        return id != null && id !== prior;
      },
      { timeout: 15000 },
      priorId,
    );
    expectedId = await page.evaluate(() => window.__kimodoLastReceipt.generationId);
  } catch {
    console.log('[filmstrip] FAIL — the click did not start a new generation.');
    await browser.close();
    process.exit(1);
  }

  // Terminal-state decisions go through the app's shipped classifier — the
  // same function the app and every other watcher use. A prior revision kept a
  // local copy here, which drifted from the app and turned a real failure into
  // a "successful" filmstrip of an empty pane.
  await page.waitForFunction(
    (id) => window.__kimodoGenerationState(id).done,
    { timeout: 300000 },
    expectedId,
  );
  const verdict = await page.evaluate((id) => window.__kimodoGenerationState(id), expectedId);
  const receiptStatus = await page.evaluate(() => window.__kimodoLastReceipt?.status);
  const statusText = await page
    .$eval('#status', el => el.textContent)
    .catch(() => '(status element not present)');
  console.log(`[filmstrip] Status: ${statusText}`);
  console.log(`[filmstrip] Receipt status: ${receiptStatus} | classifier ok: ${verdict.ok}`);

  if (!verdict.ok) {
    console.log(`[filmstrip] FAIL — generation did not produce a valid result.`);
    console.log(`[filmstrip] No authoritative filmstrip was written.`);
    await browser.close();
    process.exit(1);
  }

  // A canvas is required before any capture; "no canvas" is a failed
  // generation, never a reason to screenshot the surrounding page.
  const hasCanvas = await page.$('#viewport canvas');
  if (!hasCanvas) {
    console.log(`[filmstrip] FAIL — no rendered canvas; refusing to capture a page fallback.`);
    await browser.close();
    process.exit(1);
  }

  // Wait a moment for rendering to start
  await new Promise(r => setTimeout(r, 1000));

  // Capture filmstrip frames by manipulating the skeleton renderer
  // The canvas is in #viewport, rendered by setInterval at 30fps
  // We'll pause the animation and manually step through frames

  const frameCount = await page.evaluate(() => {
    // Access the decoded data from the global scope
    const canvas = document.querySelector('#viewport canvas');
    if (!canvas) return 0;
    // The skeleton data is stored in closure — we need to expose it
    // For now, check if the animation is running by looking at the frame counter text
    const ctx = canvas.getContext('2d');
    return parseInt(document.querySelector('#viewport canvas')?.dataset?.totalFrames || '0');
  });

  // Alternative: screenshot the canvas at intervals while it's animating
  console.log(`[filmstrip] Capturing frames every ${every} frames...`);

  // Capture the canvas only. There is deliberately no page-screenshot
  // fallback: a filmstrip of the surrounding UI is not evidence of motion,
  // and emitting one on failure is exactly how this harness previously lied.
  const screenshots = [];
  const canvas = await page.$('#viewport canvas');
  for (let i = 0; i < 12; i++) {
    await new Promise(r => setTimeout(r, every * 33)); // ~33ms per frame at 30fps
    screenshots.push(await canvas.screenshot({ type: 'png' }));
    console.log(`  Frame ${i + 1}/12`);
  }

  // Save individual frames — composite with ImageMagick montage
  const { execSync } = await import('child_process');
  const framePaths = [];
  for (let i = 0; i < screenshots.length; i++) {
    const fp = `/tmp/kimodo-frame-${String(i).padStart(2, '0')}.png`;
    writeFileSync(fp, screenshots[i]);
    framePaths.push(fp);
  }

  try {
    execSync(`montage ${framePaths.join(' ')} -tile 4x3 -geometry +2+2 -background '#111111' ${outPath}`);
    console.log(`[filmstrip] Saved filmstrip to ${outPath}`);
  } catch {
    // Fallback: just save the first frame
    writeFileSync(outPath, screenshots[0]);
    console.log(`[filmstrip] montage not available — saved first frame to ${outPath}`);
    console.log(`[filmstrip] Individual frames at /tmp/kimodo-frame-*.png`);
  }

  await browser.close();
}

main().catch(err => {
  console.error(`[filmstrip] Fatal: ${err.message}`);
  process.exit(1);
});
