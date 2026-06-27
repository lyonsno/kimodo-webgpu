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
 *   - Motion server running for /embed endpoint (motion-serve.py)
 */

import puppeteer from 'puppeteer-core';

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
  await page.click('#generate-btn');

  // Wait for generation to complete
  try {
    await page.waitForFunction(
      () => {
        const s = document.getElementById('status')?.textContent || '';
        return s.includes('Generated') || s.includes('Error');
      },
      { timeout: 120000 },
    );
  } catch {
    console.log(`[smoke] Generation timed out.`);
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

  // Check for NaN in body output
  const hasBodyNaN = logs.some(l => l.text.includes('Body output') && l.text.includes('NaN'));
  const hasRootNaN = logs.some(l => l.text.includes('Root output') && l.text.includes('NaN'));

  console.log(`\n[smoke] === Result ===`);
  console.log(`  Body NaN: ${hasBodyNaN}`);
  console.log(`  Root NaN: ${hasRootNaN}`);
  console.log(`  Status: ${hasBodyNaN || hasRootNaN ? 'FAIL — NaN in output' : 'PASS'}`);

  await browser.close();
  process.exit(hasBodyNaN || hasRootNaN ? 1 : 0);
}

main().catch(err => {
  console.error(`[smoke] Fatal: ${err.message}`);
  process.exit(1);
});
