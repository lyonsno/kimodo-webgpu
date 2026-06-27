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
  await page.click('#generate-btn');

  // Wait for generation to complete (look for "Generated" or "WebGPU" in status)
  await page.waitForFunction(
    () => {
      const s = document.getElementById('status')?.textContent || '';
      return s.includes('Generated') || s.includes('WebGPU diffusion') || s.includes('Decode error');
    },
    { timeout: 300000 },
  );

  const status = await page.$eval('#status', el => el.textContent);
  console.log(`[filmstrip] Status: ${status}`);

  if (status.includes('error') || status.includes('Error')) {
    console.log(`[filmstrip] Generation failed.`);
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

  // Let it animate and capture screenshots at intervals
  const screenshots = [];
  const canvas = await page.$('#viewport canvas');

  if (!canvas) {
    console.log(`[filmstrip] No canvas found — taking full page screenshots`);
    // Fallback: take viewport screenshots at intervals
    for (let i = 0; i < 12; i++) {
      await new Promise(r => setTimeout(r, every * 33)); // ~33ms per frame at 30fps
      const shot = await page.screenshot({ type: 'png', clip: { x: 340, y: 0, width: 860, height: 800 } });
      screenshots.push(shot);
      console.log(`  Frame ${i + 1}/12`);
    }
  } else {
    for (let i = 0; i < 12; i++) {
      await new Promise(r => setTimeout(r, every * 33));
      const shot = await canvas.screenshot({ type: 'png' });
      screenshots.push(shot);
      console.log(`  Frame ${i + 1}/12`);
    }
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
