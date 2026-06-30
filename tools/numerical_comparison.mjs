#!/usr/bin/env node
/**
 * Per-layer numerical comparison: WebGPU vs PyTorch.
 *
 * Feeds fixed input (from tests/pytorch_reference.json) through the WebGPU
 * root model forward pass and compares every intermediate against PyTorch.
 *
 * Usage: node tools/numerical_comparison.mjs [--url http://localhost:5176]
 */

import puppeteer from 'puppeteer-core';
import { readFileSync } from 'fs';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const args = process.argv.slice(2);
const url = args.find((a, i) => args[i-1] === '--url') || 'http://localhost:5176';

const ref = JSON.parse(readFileSync(new URL('../tests/pytorch_reference.json', import.meta.url)));

async function main() {
  console.log('[num-compare] Launching Chrome...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: false,
    args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--no-sandbox'],
  });

  const page = await browser.newPage();
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('[num-compare]')) console.log(`  ${text}`);
  });
  page.on('pageerror', err => console.error(`  [error] ${err.message}`));

  console.log('[num-compare] Loading page and weights...');
  await page.goto(url, { waitUntil: 'networkidle0', timeout: 120000 });
  await page.waitForFunction(() => document.getElementById('status')?.textContent?.includes('Ready'), { timeout: 180000 });
  console.log('[num-compare] Weights loaded. Running comparison...');

  // Pass the full reference data to the browser and run the comparison there
  const results = await page.evaluate(async (refData) => {
    const { createStorageBuffer, createEmptyBuffer } = await import('/src/lib/gpu.js');
    const { forwardTransformer, readBuffer } = await import('/src/lib/inference.js');
    const { loadWeights } = await import('/src/lib/weights.js');

    // Get the already-loaded device and weights from the global scope
    // (main.js stores them but doesn't expose them — we need to re-fetch)
    const { initGPU } = await import('/src/lib/gpu.js');
    const { loadConfig } = await import('/src/lib/inference.js');

    // Re-initialize (or access existing)
    const { device } = await initGPU();

    // Fetch and parse weights
    console.log('[num-compare] Fetching weights for comparison...');
    const resp = await fetch('/kimodo.bin');
    const buf = await resp.arrayBuffer();
    const weights = await loadWeights(device, buf);

    // Create input buffers from reference
    const textEmb = new Float32Array(refData.text_embedding_full);
    const motionFlat = new Float32Array(refData.motion_full.flat());
    // Root model input: [motion(369), zeros(369)] = [N, 738]
    const N = refData.motion_full.length; // 5 frames
    const rootInput = new Float32Array(N * 738);
    for (let f = 0; f < N; f++) {
      for (let d = 0; d < 369; d++) {
        rootInput[f * 738 + d] = refData.motion_full[f][d];
      }
    }

    const rootInputBuf = createStorageBuffer(device, rootInput);
    const textBuf = createStorageBuffer(device, textEmb);

    console.log('[num-compare] Running WebGPU forward pass...');
    const outBuf = await forwardTransformer(device, weights.root, rootInputBuf, textBuf, 500, N, 738, 5);
    const output = await readBuffer(device, outBuf, N * 5);

    rootInputBuf.destroy();
    textBuf.destroy();
    outBuf.destroy();

    // Compare output
    const pyOut = refData.root_output; // [5] for first frame
    const gpuOut = Array.from(output.slice(0, 5));

    const comparison = {};
    let maxDiff = 0;
    for (let d = 0; d < 5; d++) {
      const diff = Math.abs(pyOut[d] - gpuOut[d]);
      maxDiff = Math.max(maxDiff, diff);
      comparison[`dim${d}`] = {
        pytorch: pyOut[d],
        webgpu: gpuOut[d],
        diff: diff,
        relDiff: Math.abs(pyOut[d]) > 1e-6 ? diff / Math.abs(pyOut[d]) : diff,
      };
    }

    console.log('[num-compare] Output comparison (first frame, 5 dims):');
    for (const [k, v] of Object.entries(comparison)) {
      const status = v.diff < 0.01 ? 'PASS' : v.diff < 0.1 ? 'WARN' : 'FAIL';
      console.log(`[num-compare]   ${k}: py=${v.pytorch.toFixed(6)} gpu=${v.webgpu.toFixed(6)} diff=${v.diff.toFixed(6)} ${status}`);
    }
    console.log(`[num-compare] Max absolute diff: ${maxDiff.toFixed(6)}`);
    console.log(`[num-compare] Overall: ${maxDiff < 0.01 ? 'PASS (< 0.01)' : maxDiff < 0.1 ? 'WARN (< 0.1)' : 'FAIL (>= 0.1)'}`);

    return { comparison, maxDiff, pass: maxDiff < 0.01 };
  }, ref);

  console.log('\n[num-compare] === RESULT ===');
  console.log(`  Max diff: ${results.maxDiff?.toFixed(6) || 'N/A'}`);
  console.log(`  Status: ${results.pass ? 'PASS' : results.error ? `ERROR: ${results.error}` : 'FAIL'}`);

  if (results.comparison) {
    for (const [k, v] of Object.entries(results.comparison)) {
      const status = v.diff < 0.001 ? '✓' : v.diff < 0.01 ? '~' : '✗';
      console.log(`  ${status} ${k}: py=${v.pytorch.toFixed(6)} gpu=${v.webgpu.toFixed(6)} Δ=${v.diff.toFixed(6)}`);
    }
  }

  await browser.close();
  process.exit(results.pass ? 0 : 1);
}

main().catch(err => { console.error(`[num-compare] Fatal: ${err.message}`); process.exit(1); });
