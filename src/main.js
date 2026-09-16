/**
 * Kimodo WebGPU — main entry point.
 *
 * Loads the 282M diffusion model weights into WebGPU storage buffers,
 * fetches text embeddings from a server, and runs DDIM sampling entirely
 * on the GPU via compute shaders.
 */

import { initGPU } from './lib/gpu.js';
import { createKimodoProducer } from './lib/producer.js';
import { describeInvalidReceipt } from './lib/route-receipt.js';
import { classifyGenerationState, classifyMotionExport, createGenerationLifecycle } from './lib/generation-state.js';

// The single choke point every watcher (smoke harnesses, live probes) uses to
// decide whether the generation it is watching has terminally settled. Keeping
// exactly one classifier prevents the harness/app drift that previously turned
// a real failure status into a silent timeout.
window.__kimodoGenerationState = (expectedId) => classifyGenerationState({
  receipt: window.__kimodoLastReceipt ?? null,
  expectedId,
  canvasPresent: !!document.querySelector('#viewport canvas'),
});

// Same choke-point rule for the motion export: motion is usable only when the
// receipt for the generation the watcher is watching is terminal-real.
window.__kimodoMotionState = (expectedId) => classifyMotionExport({
  motion: window.__kimodoLastMotion ?? null,
  receipt: window.__kimodoLastReceipt ?? null,
  expectedId,
});

const statusEl = document.getElementById('status');
const infoEl = document.getElementById('info');
const progressBar = document.getElementById('progress-bar');
const generateBtn = document.getElementById('generate-btn');

let gpuDevice = null;
let gpuAdapter = null;
let gpuBackendIdentity = null;
let modelConfig = null;

async function init() {
  try {
    statusEl.textContent = 'Requesting WebGPU device...';
    const { adapter, device, backendIdentity } = await initGPU();
    gpuDevice = device;
    gpuAdapter = adapter;
    statusEl.textContent = 'WebGPU ready.';
    infoEl.textContent = `GPU: ${(device.limits.maxBufferSize / 1e9).toFixed(1)} GB max buffer`;

    // The page hands its device to the producer as a BORROWED context — the
    // same shape a Kaminos host uses when composing generation with its own
    // live foreground. The producer loads config, FK data, motion stats, and
    // the 540 MB weights (hashing the bytes actually consumed) and never
    // destroys the device.
    statusEl.textContent = 'Loading weights (540 MB)...';
    const t0 = performance.now();
    producer = await createKimodoProducer({
      device,
      adapter,
      backendIdentity,
      assetBase: '',
      embedUrl: `${document.getElementById('server-url').value.trim()}/embed`,
      onLoadProgress: ({ loaded, total }) => {
        if (total > 0) {
          progressBar.style.width = `${Math.round(100 * loaded / total)}%`;
          statusEl.textContent = `Loading weights... ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`;
        }
      },
    });
    const loadTime = ((performance.now() - t0) / 1000).toFixed(1);
    modelConfig = { fps: producer.identity.fps };
    progressBar.style.width = '100%';
    statusEl.textContent = `Ready. Model loaded in ${loadTime}s (download + GPU upload).`;
    infoEl.textContent = `${producer.identity.model.id} | kit ${producer.identity.kitVersion} | ${loadTime}s`;
    generateBtn.disabled = false;

  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

let producer = null;

// All generation evidence flows through the lifecycle owner: single-flight
// admission plus ownership-checked publication. generate() is globally
// callable (window.generate), so the overlap policy must live at this
// boundary, not in DOM button state.
const generationLifecycle = createGenerationLifecycle({
  setReceipt: (r) => { window.__kimodoLastReceipt = r; },
  setMotion: (m) => { window.__kimodoLastMotion = m; },
  getReceipt: () => window.__kimodoLastReceipt ?? null,
});

const EMBEDDING_HELP = {
  'embedding-unreachable': (url) =>
    'This route requires an external /embed endpoint returning a 4096-float ' +
    'text embedding. It is not bundled with this repository — see the README ' +
    `setup section. (endpoint: ${url})`,
  'embedding-http': (url) =>
    `POST ${url} must accept {"prompt": "..."} and return ` +
    '{"embedding": [...4096 floats]}. See the README setup section for the contract.',
  'embedding-unusable': () =>
    'Check that the endpoint uses the Kimodo LLM2Vec/Llama 3 8B text encoder and returns finite floats.',
};

async function generate() {
  if (!producer || !gpuDevice) return;

  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;

  // Bind this run to a fresh identity and supersede any prior evidence BEFORE
  // the first await — begin() installs the in-progress receipt and clears the
  // motion export on the same boundary. A second invocation while one is in
  // flight is rejected, not queued: overlapping runs previously allowed an
  // older generation to resurrect superseded evidence.
  const run = generationLifecycle.begin();
  if (!run) {
    console.warn(`[kimodo-webgpu] generate() rejected: generation ${generationLifecycle.activeId} is still in flight`);
    return { rejected: 'generation-in-flight' };
  }
  const { generationId } = run;

  // The settlement guard covers EVERYTHING after successful admission: a
  // synchronous throw in input reads or UI setup outside the try would
  // otherwise strand the single-flight owner permanently.
  try {
    const duration = parseFloat(document.getElementById('duration').value) || 6;
    const numSteps = parseInt(document.getElementById('steps').value) || 100;
    const embedUrl = `${document.getElementById('server-url').value.trim()}/embed`;

    generateBtn.disabled = true;
    progressBar.style.width = '0%';
    const t0 = performance.now();

    const result = await producer.generate({
      prompt,
      steps: numSteps,
      duration,
      generationId,
      embedUrl,
      onStage: (name, event) => {
        if (event !== 'start') return;
        statusEl.textContent = {
          'text-embedding': 'Requesting text embedding from server...',
          'ddim-sampling': `Running ${numSteps}-step DDIM on WebGPU...`,
          'fk-decode': 'Decoding to joints...',
          'output-capture': 'Capturing output...',
        }[name] ?? name;
      },
      onProgress: async ({ step, pct }) => {
        progressBar.style.width = `${pct}%`;
        statusEl.textContent = `WebGPU DDIM step ${step}/${numSteps} (${pct}%)`;
        // The route's declared per-diffusion-step cooperative checkpoint: one
        // frame yield per step guarantees paint cadence for the progress UI.
        await new Promise(requestAnimationFrame);
      },
    });

    const genTime = ((performance.now() - t0) / 1000).toFixed(1);
    const { receipt, motion } = result;
    progressBar.style.width = '100%';
    renderSkeletonFromJoints({
      joints: motion.joints, parents: motion.parents,
      num_frames: motion.numFrames, num_joints: motion.numJoints,
    });

    // Publish receipt + motion through the lifecycle owner, which refuses the
    // write if this run no longer owns the slot. Same trust level for both:
    // page-local evidence read by local harnesses. motion rows are plain
    // Array(369); the last four values of each row are the foot contacts.
    run.publishSuccess(receipt, motion);
    console.log('[kimodo-webgpu] Route receipt:', JSON.stringify(receipt.profile));
    console.log('[kimodo-webgpu] Receipt status:', receipt.status, '| model:', receipt.model.id);

    if (receipt.status !== 'real') {
      statusEl.textContent = `Generation produced an invalid receipt — ${receipt.fallbackReason}`;
      infoEl.textContent = describeInvalidReceipt(receipt);
      return;
    }

    // "client-side" is scoped deliberately: text embedding is server-side.
    const ddimMs = receipt.timings?.stages?.find((s) => s.name === 'ddim-sampling')?.durationMs;
    infoEl.textContent = `${motion.numFrames}f @ ${motion.fps}fps | ${genTime}s total (${ddimMs ?? '?'}ms diffusion) | ${numSteps} steps | diffusion+FK client-side, text embedding via server`;
    statusEl.textContent = `Generated ${motion.numFrames} frames in ${genTime}s (WebGPU diffusion + JS FK → ${motion.numJoints} joints)`;

  } catch (err) {
    // Producer errors carry their phase and, after the queue exists, the
    // terminal bounded-submission report; the original error stays
    // authoritative and the report rides the failure evidence.
    const phase = err?.phase ?? 'exception';
    run.publishFailure(phase, err.message, err?.gpuSubmission ? { gpuSubmission: err.gpuSubmission } : null);
    statusEl.textContent = `Error: ${err.message}`;
    const help = EMBEDDING_HELP[phase];
    if (help) infoEl.textContent = help(`${document.getElementById('server-url').value.trim()}/embed`);
    console.error(err);
  } finally {
    // Structural backstop: settle() converts a still-in-progress receipt to a
    // terminal failure and releases the flight slot — and reports whether this
    // run still owned the generation, which gates re-enabling the controls.
    if (run.settle()) {
      generateBtn.disabled = false;
    }
  }
}
window.generate = generate;

// ---------- Skeleton rendering from decoded joint positions ----------

let skelAnimId = null;

function renderSkeletonFromJoints(decoded) {
  const viewport = document.getElementById('viewport');
  // #status must survive the canvas swap AND stay visible to the operator.
  // Deleting it broke harnesses (success looked like a timeout); hiding it
  // broke the operator (second-generation progress and failures became
  // invisible while a stale canvas still looked current). Move it out of the
  // replaceable viewport and keep it on screen as an overlay.
  const oldStatus = viewport.querySelector('#status');
  if (oldStatus && oldStatus.parentElement === viewport) {
    oldStatus.classList.add('status-overlay');
    document.body.appendChild(oldStatus);
  }

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
    ctx.fillText(`Frame ${frame}/${numFrames} | ${numJoints} joints | WebGPU diffusion + JS FK`, 10, H - 10);

    frame = (frame + 1) % numFrames;
  }, 1000 / 30);
}

init();
