/**
 * Kimodo WebGPU — main entry point.
 *
 * Loads the 282M diffusion model weights into WebGPU storage buffers,
 * fetches text embeddings from a server, and runs DDIM sampling entirely
 * on the GPU via compute shaders.
 */

import { initGPU } from './lib/gpu.js';
import { loadWeights } from './lib/weights.js';
import { loadConfig, singleForwardPass, forwardTransformer, readBuffer } from './lib/inference.js';

const statusEl = document.getElementById('status');
const infoEl = document.getElementById('info');
const progressBar = document.getElementById('progress-bar');
const generateBtn = document.getElementById('generate-btn');

let gpuDevice = null;
let modelConfig = null;
let modelWeights = null;

async function init() {
  try {
    statusEl.textContent = 'Requesting WebGPU device...';
    const { device } = await initGPU();
    gpuDevice = device;
    statusEl.textContent = 'WebGPU ready.';
    infoEl.textContent = `GPU: ${(device.limits.maxBufferSize / 1e9).toFixed(1)} GB max buffer`;

    // Load config
    statusEl.textContent = 'Loading model config...';
    modelConfig = await loadConfig('/kimodo.json');
    infoEl.textContent = `${modelConfig.model} | ${modelConfig.hidden_dim}d x ${modelConfig.num_layers}L | ${modelConfig.dtype}`;

    // Load weights with progress
    statusEl.textContent = 'Loading weights (540 MB)...';
    const t0 = performance.now();

    const resp = await fetch('/kimodo.bin');
    const total = parseInt(resp.headers.get('Content-Length') || '0');
    const reader = resp.body.getReader();
    const chunks = [];
    let loaded = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      if (total > 0) {
        const pct = Math.round(100 * loaded / total);
        progressBar.style.width = `${pct}%`;
        statusEl.textContent = `Loading weights... ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`;
      }
    }

    // Combine into ArrayBuffer
    const buffer = new ArrayBuffer(loaded);
    const view = new Uint8Array(buffer);
    let offset = 0;
    for (const chunk of chunks) {
      view.set(chunk, offset);
      offset += chunk.byteLength;
    }

    const downloadTime = ((performance.now() - t0) / 1000).toFixed(1);
    statusEl.textContent = `Parsing weights and creating GPU buffers...`;

    // Parse and upload to GPU
    const t1 = performance.now();
    modelWeights = await loadWeights(gpuDevice, buffer);
    const uploadTime = ((performance.now() - t1) / 1000).toFixed(1);

    progressBar.style.width = '100%';
    statusEl.textContent = `Ready. Weights loaded in ${downloadTime}s (download) + ${uploadTime}s (GPU upload).`;
    infoEl.textContent += ` | ${(loaded / 1e6).toFixed(0)} MB | ${downloadTime}s + ${uploadTime}s`;
    generateBtn.disabled = false;

  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

async function generate() {
  if (!modelWeights || !gpuDevice) return;

  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;

  const duration = parseFloat(document.getElementById('duration').value) || 6;
  const numSteps = parseInt(document.getElementById('steps').value) || 100;
  const numFrames = Math.round(duration * modelConfig.fps);
  const serverUrl = document.getElementById('server-url').value.trim();

  generateBtn.disabled = true;
  progressBar.style.width = '0%';

  try {
    // Step 1: Get text embedding from server
    statusEl.textContent = 'Requesting text embedding from server...';
    const embResp = await fetch(`${serverUrl}/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });

    if (!embResp.ok) {
      // Fallback: try generate endpoint and explain
      statusEl.textContent = 'Text embedding endpoint not available yet. Need to add /embed to motion-serve.py.';
      infoEl.textContent = 'The /embed endpoint returns just the 4096-dim text vector. Add it to motion-serve.py.';
      return;
    }

    const embData = await embResp.json();
    const textEmbedding = new Float32Array(embData.embedding);

    // Use full MPS generation via /generate for proper DDIM quality
    statusEl.textContent = `Generating ${numFrames} frames via server (full DDIM)...`;
    const t0 = performance.now();

    const genResp = await fetch(`${serverUrl}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, duration, steps: numSteps }),
    });
    const genData = await genResp.json();

    if (genData.error) {
      statusEl.textContent = `Error: ${genData.error}`;
      return;
    }

    const genTime = ((performance.now() - t0) / 1000).toFixed(1);
    progressBar.style.width = '100%';
    statusEl.textContent = `Generated ${genData.num_frames} frames in ${genTime}s — rendering...`;
    infoEl.textContent = `${genData.num_frames}f @ ${genData.fps}fps | ${genTime}s server (${genData.gen_time}s model)`;

    renderSkeletonFromJoints(genData);
    statusEl.textContent = `Generated ${genData.num_frames} frames in ${genTime}s (${genData.num_joints} joints)`;

  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  } finally {
    generateBtn.disabled = false;
  }
}
window.generate = generate;

// ---------- Skeleton / output visualization ----------

let animFrameId = null;

function renderSkeleton(result) {
  const viewport = document.getElementById('viewport');

  // Remove old status text
  const oldStatus = viewport.querySelector('#status');
  if (oldStatus) oldStatus.remove();

  // Create or reuse canvas
  let canvas = viewport.querySelector('canvas');
  if (!canvas) {
    canvas = document.createElement('canvas');
    viewport.appendChild(canvas);
  }
  canvas.width = viewport.clientWidth;
  canvas.height = viewport.clientHeight;
  const ctx = canvas.getContext('2d');

  const { bodyResult, rootResult, numFrames, fps } = result;
  const bodyDim = modelConfig.body_output_dim; // 364
  const rootDim = modelConfig.root_output_dim; // 5

  // Display output stats
  let bodyMin = Infinity, bodyMax = -Infinity, bodySum = 0;
  for (let i = 0; i < bodyResult.length; i++) {
    const v = bodyResult[i];
    if (v < bodyMin) bodyMin = v;
    if (v > bodyMax) bodyMax = v;
    bodySum += v;
  }
  const bodyMean = bodySum / bodyResult.length;

  infoEl.textContent += ` | body: [${bodyMin.toFixed(2)}, ${bodyMax.toFixed(2)}] mean=${bodyMean.toFixed(4)}`;
  console.log('[kimodo-webgpu] Body output stats:', { min: bodyMin, max: bodyMax, mean: bodyMean, len: bodyResult.length });
  console.log('[kimodo-webgpu] Root output stats:', { len: rootResult.length });
  console.log('[kimodo-webgpu] First 10 body values:', Array.from(bodyResult.slice(0, 10)));
  console.log('[kimodo-webgpu] First 10 root values:', Array.from(rootResult.slice(0, 10)));

  // Animate: render a heatmap of body features over time
  let frame = 0;
  if (animFrameId) cancelAnimationFrame(animFrameId);

  function drawFrame() {
    const W = canvas.width;
    const H = canvas.height;
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, W, H);

    // Draw feature heatmap for current frame
    const offset = frame * bodyDim;
    const barW = W / bodyDim;
    for (let i = 0; i < bodyDim; i++) {
      const v = bodyResult[offset + i];
      // Map value to color: negative=blue, zero=black, positive=orange
      const intensity = Math.min(1, Math.abs(v) / 2);
      if (v > 0) {
        ctx.fillStyle = `rgba(255, 140, 0, ${intensity})`;
      } else {
        ctx.fillStyle = `rgba(30, 144, 255, ${intensity})`;
      }
      ctx.fillRect(i * barW, H * 0.1, barW + 1, H * 0.3);
    }

    // Draw root trajectory (x, z) up to current frame
    const rootScale = 50;
    const cx = W / 2;
    const cy = H * 0.7;
    ctx.strokeStyle = '#ff8800';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let f = 0; f <= frame; f++) {
      const rOff = f * rootDim;
      const x = cx + (rootResult[rOff] || 0) * rootScale;
      const z = cy - (rootResult[rOff + 2] || 0) * rootScale;
      if (f === 0) ctx.moveTo(x, z); else ctx.lineTo(x, z);
    }
    ctx.stroke();

    // Current position dot
    const rOff = frame * rootDim;
    const dotX = cx + (rootResult[rOff] || 0) * rootScale;
    const dotZ = cy - (rootResult[rOff + 2] || 0) * rootScale;
    ctx.fillStyle = '#ff6600';
    ctx.beginPath();
    ctx.arc(dotX, dotZ, 5, 0, Math.PI * 2);
    ctx.fill();

    // Frame counter
    ctx.fillStyle = '#888';
    ctx.font = '12px monospace';
    ctx.fillText(`Frame ${frame}/${numFrames}  |  Body dim: ${bodyDim}  |  Root dim: ${rootDim}`, 10, H - 10);
    ctx.fillText(`Top: feature heatmap (blue=neg, orange=pos)  |  Bottom: root trajectory (x,z)`, 10, H - 26);

    frame = (frame + 1) % numFrames;
    animFrameId = requestAnimationFrame(drawFrame);
  }

  drawFrame();
}

// ---------- Skeleton rendering from decoded joint positions ----------

let skelAnimId = null;

function renderSkeletonFromJoints(decoded) {
  const viewport = document.getElementById('viewport');
  const oldStatus = viewport.querySelector('#status');
  if (oldStatus) oldStatus.remove();

  let canvas = viewport.querySelector('canvas');
  if (!canvas) {
    canvas = document.createElement('canvas');
    viewport.appendChild(canvas);
  }
  canvas.width = viewport.clientWidth;
  canvas.height = viewport.clientHeight;
  const ctx = canvas.getContext('2d');

  const joints = decoded.joints; // [N, J, 3]
  const parents = decoded.parents || [];
  const numFrames = decoded.num_frames;
  const numJoints = decoded.num_joints;

  // Compute Y offset to ground the figure
  let minY = Infinity;
  for (let f = 0; f < numFrames; f++) {
    for (let j = 0; j < numJoints; j++) {
      if (joints[f][j][1] < minY) minY = joints[f][j][1];
    }
  }

  // Build bone list
  const bones = [];
  for (let i = 0; i < parents.length; i++) {
    if (parents[i] >= 0 && parents[i] !== i) bones.push([parents[i], i]);
  }

  let frame = 0;
  if (skelAnimId) clearInterval(skelAnimId);

  skelAnimId = setInterval(() => {
    const W = canvas.width;
    const H = canvas.height;
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, W, H);

    const fj = joints[frame];
    const root = fj[0];
    const scale = 200;
    const cx = W / 2;
    const cy = H * 0.75;

    // Draw bones
    ctx.strokeStyle = '#ff8800';
    ctx.lineWidth = 2;
    for (const [pi, ci] of bones) {
      const p = fj[pi], c = fj[ci];
      if (!p || !c) continue;
      const px = cx + (p[0] - root[0]) * scale;
      const py = cy - (p[1] - minY) * scale;
      const cxx = cx + (c[0] - root[0]) * scale;
      const cyy = cy - (c[1] - minY) * scale;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(cxx, cyy);
      ctx.stroke();
    }

    // Draw joints (first 30 as bigger dots)
    for (let j = 0; j < Math.min(numJoints, 30); j++) {
      const jt = fj[j];
      const x = cx + (jt[0] - root[0]) * scale;
      const y = cy - (jt[1] - minY) * scale;
      ctx.fillStyle = '#ff6600';
      ctx.beginPath();
      ctx.arc(x, y, j < 7 ? 4 : 2, 0, Math.PI * 2);
      ctx.fill();
    }

    // Info
    ctx.fillStyle = '#666';
    ctx.font = '12px monospace';
    ctx.fillText(`Frame ${frame}/${numFrames} | ${numJoints} joints | WebGPU → Server FK`, 10, H - 10);

    frame = (frame + 1) % numFrames;
  }, 1000 / 30);
}

init();
