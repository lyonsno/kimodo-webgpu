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
import { loadMotionRepStats, denoiseStepWebGPU } from './lib/denoiser.js';

const statusEl = document.getElementById('status');
const infoEl = document.getElementById('info');
const progressBar = document.getElementById('progress-bar');
const generateBtn = document.getElementById('generate-btn');

let gpuDevice = null;
let modelConfig = null;
let modelWeights = null;
let motionRepStats = null;

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

    // Load motion_rep stats for root conversion
    motionRepStats = await loadMotionRepStats('/motion_rep_stats.json');

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
    console.log('[kimodo-webgpu] embed received[0:5]: ' + JSON.stringify([textEmbedding[0], textEmbedding[1], textEmbedding[2], textEmbedding[3], textEmbedding[4]]));
    console.log('[kimodo-webgpu] embed length: ' + textEmbedding.length);

    // Client-side DDIM loop with server-side denoising steps
    statusEl.textContent = `Running ${numSteps}-step DDIM on WebGPU...`;
    const t0 = performance.now();
    const motionDim = 369; // root(5) + body(364)

    // Cosine beta schedule (matching Kimodo's diffusion.py)
    function alphaBarFn(t) { return Math.cos((t + 0.008) / 1.008 * Math.PI / 2) ** 2; }
    const numBase = 1000;
    const betasBase = [];
    for (let i = 0; i < numBase; i++) {
      betasBase.push(Math.min(1 - alphaBarFn((i+1)/numBase) / alphaBarFn(i/numBase), 0.999));
    }
    const alphasCumprodBase = [];
    let acc = 1;
    for (const b of betasBase) { acc *= (1 - b); alphasCumprodBase.push(acc); }

    // Subsample timesteps
    const fracStride = (numBase - 1) / Math.max(1, numSteps - 1);
    const useTimesteps = [];
    for (let i = 0; i < numSteps; i++) {
      useTimesteps.push(Math.min(Math.round(i * fracStride), numBase - 1));
    }

    // Compute diffusion vars matching Kimodo's calc_diffusion_vars exactly:
    // 1. Get base alphas_cumprod at subsampled positions
    // 2. Recompute betas from consecutive ratios
    // 3. Recompute alphas_cumprod from those betas
    const subsampledAlphasCumprod = useTimesteps.map(t => alphasCumprodBase[t]);
    const lastAlphasCumprod = [1.0, ...subsampledAlphasCumprod.slice(0, -1)];
    const betas = subsampledAlphasCumprod.map((ac, i) => 1.0 - ac / lastAlphasCumprod[i]);
    const alphas = betas.map(b => 1.0 - b);
    const alphasCumprod = [];
    let cumprod = 1.0;
    for (const a of alphas) { cumprod *= a; alphasCumprod.push(Math.max(cumprod, 1e-9)); }
    const alphasCumprodPrev = [1.0, ...alphasCumprod.slice(0, -1)];
    const sqrtRecipAlphasCumprod = alphasCumprod.map(a => 1 / Math.sqrt(a));
    const sqrtRecipm1AlphasCumprod = alphasCumprod.map(a => Math.sqrt((1 - a) / a));

    // Initialize with Gaussian noise [numFrames, 369]
    const motion = new Array(numFrames);
    for (let f = 0; f < numFrames; f++) {
      motion[f] = new Array(motionDim);
      for (let d = 0; d < motionDim; d++) {
        const u1 = Math.random(), u2 = Math.random();
        motion[f][d] = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
      }
    }

    // DDIM loop (reverse: t = numSteps-1 down to 0)
    for (let step = numSteps - 1; step >= 0; step--) {
      const pct = Math.round(100 * (numSteps - step) / numSteps);
      progressBar.style.width = `${pct}%`;
      statusEl.textContent = `WebGPU DDIM step ${numSteps - step}/${numSteps} (${pct}%)`;

      // Toggle: use WebGPU or server for denoising
      // WebGPU denoising
      const predClean = await denoiseStepWebGPU(
        gpuDevice, modelWeights, Array.from(textEmbedding),
        motion, useTimesteps[step], motionRepStats,
      );

      if (false && step === numSteps - 1) {
        // Compare raw root model forward pass (no CFG, no TwostageDenoiser)
        // Construct root input: [motion(369), zeros(369)] = [N, 738]
        const rootInput = motion.map(f => [...f, ...new Array(369).fill(0)]);

        // Server: raw root model
        const srvResp = await fetch(`${serverUrl}/forward_root`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ root_input: rootInput, text_emb: Array.from(textEmbedding), timestep: useTimesteps[step] }),
        });
        const srvData = await srvResp.json();

        // WebGPU: raw root model forward pass
        const { createStorageBuffer } = await import('./lib/gpu.js');
        const { forwardTransformer, readBuffer } = await import('./lib/inference.js');
        const rootInputFlat = new Float32Array(motion.length * 738);
        for (let f = 0; f < motion.length; f++) {
          for (let d = 0; d < 369; d++) rootInputFlat[f * 738 + d] = motion[f][d];
        }
        const rootInputBuf = createStorageBuffer(gpuDevice, rootInputFlat);
        const textBufCompare = createStorageBuffer(gpuDevice, new Float32Array(textEmbedding));
        const gpuOutBuf = await forwardTransformer(gpuDevice, modelWeights.root, rootInputBuf, textBufCompare, useTimesteps[step], motion.length, 738, 5);
        const gpuOut = await readBuffer(gpuDevice, gpuOutBuf, motion.length * 5);
        rootInputBuf.destroy(); textBufCompare.destroy(); gpuOutBuf.destroy();

        if (!srvData.error) {
          console.log('[raw-root] Server out[0]: ' + JSON.stringify(srvData.output[0]));
          console.log('[raw-root] WebGPU out[0]: ' + JSON.stringify(Array.from(gpuOut.slice(0, 5))));
          if (srvData.xseq) {
            console.log('[raw-root] Server xseq shape: ' + JSON.stringify(srvData.xseq.shape));
            console.log('[raw-root] Server xseq[0] (text): ' + JSON.stringify(srvData.xseq.token0));
            console.log('[raw-root] Server xseq[1] (pad): ' + JSON.stringify(srvData.xseq.token1));
            console.log('[raw-root] Server xseq[50] (ts): ' + JSON.stringify(srvData.xseq.token50));
            console.log('[raw-root] Server xseq[51] (hd): ' + JSON.stringify(srvData.xseq.token51));
            console.log('[raw-root] Server xseq[52] (m0): ' + JSON.stringify(srvData.xseq.token52));
          }
        } else {
          console.log('[raw-root] Server error: ' + srvData.error);
        }
      }
      // DDIM update: compute eps, then x_{t-1}
      const sqrtRecip = sqrtRecipAlphasCumprod[step];
      const sqrtRecipm1 = sqrtRecipm1AlphasCumprod[step];
      const alphaBarPrev = alphasCumprodPrev[step];

      for (let f = 0; f < numFrames; f++) {
        for (let d = 0; d < motionDim; d++) {
          const eps = (sqrtRecip * motion[f][d] - predClean[f][d]) / sqrtRecipm1;
          motion[f][d] = predClean[f][d] * Math.sqrt(alphaBarPrev) + Math.sqrt(1 - alphaBarPrev) * eps;
        }
      }
    }

    const genTime = ((performance.now() - t0) / 1000).toFixed(1);
    statusEl.textContent = `Generated in ${genTime}s — decoding to joints...`;

    // Decode final motion to joint positions
    const bodyFeatures = motion.map(f => f.slice(5)); // body = [5:369]
    const rootFeatures = motion.map(f => f.slice(0, 5)); // root = [0:5]

    const decodeResp = await fetch(`${serverUrl}/decode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ body_features: bodyFeatures, root_features: rootFeatures }),
    });
    const decoded = await decodeResp.json();

    if (decoded.error) {
      statusEl.textContent = `Decode error: ${decoded.error}`;
    } else {
      progressBar.style.width = '100%';
      infoEl.textContent = `${decoded.num_frames}f @ 30fps | ${genTime}s | ${numSteps} DDIM steps (browser loop)`;
      renderSkeletonFromJoints(decoded);
      statusEl.textContent = `Generated ${decoded.num_frames} frames in ${genTime}s (WebGPU diffusion → ${decoded.num_joints} joints)`;
    }

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
