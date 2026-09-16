/**
 * producer.js — the Kimodo text-to-motion route as a host-callable producer.
 *
 * A host (the page in main.js, or a Kaminos scene composing generation with
 * its own live foreground) supplies a BORROWED WebGPU device/queue and an
 * embedding endpoint; the producer loads the model onto that device, runs
 * generations with preserved stage boundaries, progress, cancellation, and
 * a foreground-opportunity boundary between admitted GPU duties, and
 * returns the native motion result with the route receipt. It destroys only
 * what it created — never the device.
 *
 * Text embedding is an external server route (LLM2Vec / Llama 3 8B); the
 * producer makes that dependency explicit in identity and receipts rather
 * than pretending sampling in the browser makes the whole route local.
 */

import { loadWeights } from './weights.js';
import { denoiseStepWebGPU, loadMotionRepStats } from './denoiser.js';
import { loadFKData, setFKData, decodeMotion } from './fk_decode.js';
import { captureBackendIdentity, createStagedProfile, createKimodoRouteReceipt, setTextEmbeddingEndpoint } from './route-receipt.js';
import { createWebGpuBoundedSubmissionQueue, WEBGPU_INFERENCE_KIT_VERSION } from '@kaminos/webgpu-inference-kit';

const MODEL_ID = 'NVIDIA/Kimodo-SOMA-RP-v1.1';
const MODEL_REVISION = 'SOMA-RP-v1.1';
const MOTION_DIM = 369; // root(5) + body(364)

export class KimodoProducerError extends Error {
  constructor(phase, message) {
    super(message);
    this.name = 'KimodoProducerError';
    this.phase = phase;
  }
}

async function sha256Hex(buffer) {
  const h = await crypto.subtle.digest('SHA-256', buffer);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** Cosine schedule + DDIM variables matching Kimodo's diffusion.py exactly. */
function ddimSchedule(numSteps) {
  const alphaBarFn = (t) => Math.cos((t + 0.008) / 1.008 * Math.PI / 2) ** 2;
  const numBase = 1000;
  const betasBase = [];
  for (let i = 0; i < numBase; i++) {
    betasBase.push(Math.min(1 - alphaBarFn((i + 1) / numBase) / alphaBarFn(i / numBase), 0.999));
  }
  const alphasCumprodBase = [];
  let acc = 1;
  for (const b of betasBase) { acc *= (1 - b); alphasCumprodBase.push(acc); }

  const fracStride = (numBase - 1) / Math.max(1, numSteps - 1);
  const useTimesteps = [];
  for (let i = 0; i < numSteps; i++) {
    useTimesteps.push(Math.min(Math.round(i * fracStride), numBase - 1));
  }
  const subsampled = useTimesteps.map((t) => alphasCumprodBase[t]);
  const last = [1.0, ...subsampled.slice(0, -1)];
  const betas = subsampled.map((ac, i) => 1.0 - ac / last[i]);
  const alphas = betas.map((b) => 1.0 - b);
  const alphasCumprod = [];
  let cumprod = 1.0;
  for (const a of alphas) { cumprod *= a; alphasCumprod.push(Math.max(cumprod, 1e-9)); }
  const alphasCumprodPrev = [1.0, ...alphasCumprod.slice(0, -1)];
  return {
    useTimesteps,
    alphasCumprodPrev,
    sqrtRecipAlphasCumprod: alphasCumprod.map((a) => 1 / Math.sqrt(a)),
    sqrtRecipm1AlphasCumprod: alphasCumprod.map((a) => Math.sqrt((1 - a) / a)),
  };
}

function gaussianNoise(numFrames) {
  const motion = new Array(numFrames);
  for (let f = 0; f < numFrames; f++) {
    motion[f] = new Array(MOTION_DIM);
    for (let d = 0; d < MOTION_DIM; d++) {
      const u1 = Math.random(), u2 = Math.random();
      motion[f][d] = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    }
  }
  return motion;
}

function summarizeSubmissionReport(report, error = null) {
  if (!report) return error ? { status: 'unreported', error: String(error?.message ?? error) } : null;
  return {
    status: report.status,
    maxInFlightDuties: report.maxInFlightDuties,
    maxObservedInFlightDuties: report.maxObservedInFlightDuties,
    submittedDutyCount: report.submittedDutyCount,
    completedDutyCount: report.completedDutyCount,
    failedDutyCount: report.failedDutyCount,
    inFlightDutyCount: report.inFlightDutyCount,
    ...(error ? { drainError: String(error?.message ?? error) } : {}),
  };
}

function destroyOwned(node, seen = new Set()) {
  if (!node || typeof node !== 'object' || seen.has(node)) return;
  seen.add(node);
  if (typeof node.destroy === 'function') { node.destroy(); return; }
  for (const value of Object.values(node)) destroyOwned(value, seen);
}

/**
 * @param {object} input
 * @param {GPUDevice} input.device            borrowed; never destroyed
 * @param {GPUQueue}  [input.queue]           defaults to device.queue
 * @param {string}    input.embedUrl          POST {prompt} -> {embedding: float[text_dim]}
 * @param {Function}  [input.fetch]           fetch implementation (defaults to global)
 * @param {string}    [input.assetBase]       base URL for kimodo.json / kimodo.bin / fk_data.json / motion_rep_stats.json
 * @param {object}    [input.backendIdentity] host-provided kit backend identity (borrowed context)
 * @param {object}    [input.adapter]         optional adapter for legacy identity capture
 * @param {object}    [input.assets]          preloaded {config, fkData, motionRepStats, weights, weightsHash}
 * @param {Function}  [input.onLoadProgress]  ({loaded, total}) during weight download
 */
export async function createKimodoProducer(input = {}) {
  const device = input.device;
  const queue = input.queue ?? device?.queue;
  if (!device || typeof device !== 'object') throw new KimodoProducerError('init', 'device is required (borrowed host GPUDevice)');
  if (!queue || typeof queue.submit !== 'function') throw new KimodoProducerError('init', 'queue must provide submit');
  if (typeof input.embedUrl !== 'string' || input.embedUrl.length === 0) {
    throw new KimodoProducerError('init', 'embedUrl is required: the text encoder is an external server route');
  }
  const fetchImpl = input.fetch ?? globalThis.fetch;
  const assetBase = (input.assetBase ?? '').replace(/\/$/, '');

  let config, fkData, motionRepStats, weights, weightsHash;
  if (input.assets) {
    ({ config, fkData, motionRepStats, weights } = input.assets);
    weightsHash = input.assets.weightsHash ?? 'unknown-weights-hash';
    if (!config || !fkData || !motionRepStats || !weights) {
      throw new KimodoProducerError('init', 'assets must supply config, fkData, motionRepStats, and weights');
    }
    setFKData(fkData);
  } else {
    const cfgResp = await fetchImpl(`${assetBase}/kimodo.json`);
    if (!cfgResp?.ok) throw new KimodoProducerError('load-config', `could not load ${assetBase}/kimodo.json`);
    config = await cfgResp.json();
    await loadFKData(`${assetBase}/fk_data.json`);
    motionRepStats = await loadMotionRepStats(`${assetBase}/motion_rep_stats.json`);

    const resp = await fetchImpl(`${assetBase}/kimodo.bin`);
    if (!resp?.ok) throw new KimodoProducerError('load-weights', `could not load ${assetBase}/kimodo.bin`);
    const total = parseInt(resp.headers?.get?.('Content-Length') || '0', 10);
    const reader = resp.body.getReader();
    const chunks = [];
    let loaded = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      input.onLoadProgress?.({ loaded, total });
    }
    const buffer = new ArrayBuffer(loaded);
    const view = new Uint8Array(buffer);
    let offset = 0;
    for (const chunk of chunks) { view.set(chunk, offset); offset += chunk.byteLength; }
    // Identity of the weights actually consumed, from the bytes themselves.
    weightsHash = await sha256Hex(buffer);
    weights = await loadWeights(device, buffer);
  }

  const identity = Object.freeze({
    model: { id: MODEL_ID, revision: MODEL_REVISION, dtype: 'fp16', weightsHash },
    kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
    embedUrl: input.embedUrl,
    assetBase,
    fps: config.fps,
    textDim: config.text_dim,
    numJoints: 30,
    textEmbeddingSource: 'server-side-llama3-8b',
  });

  let disposed = false;
  let generationCounter = 0;

  async function fetchEmbedding(prompt, embedUrl = input.embedUrl) {
    let resp;
    try {
      resp = await fetchImpl(embedUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
    } catch (netErr) {
      throw new KimodoProducerError('embedding-unreachable', `cannot reach ${embedUrl}: ${netErr.message}`);
    }
    if (!resp.ok) throw new KimodoProducerError('embedding-http', `${embedUrl} responded ${resp.status} ${resp.statusText}`);
    const data = await resp.json();
    const emb = data?.embedding;
    // Length alone is not enough: Float32Array coerces strings to NaN and
    // nulls to zero; either would still diffuse into plausible-looking noise.
    if (!Array.isArray(emb)) throw new KimodoProducerError('embedding-unusable', `expected an array, got ${typeof emb}`);
    if (emb.length !== config.text_dim) throw new KimodoProducerError('embedding-unusable', `expected ${config.text_dim} floats, got ${emb.length}`);
    const bad = emb.findIndex((v) => typeof v !== 'number' || !Number.isFinite(v));
    if (bad !== -1) throw new KimodoProducerError('embedding-unusable', `element ${bad} is not a finite number`);
    return new Float32Array(emb);
  }

  async function generate(opts = {}) {
    if (disposed) throw new KimodoProducerError('disposed', 'producer has been disposed');
    const prompt = String(opts.prompt ?? '').trim();
    if (!prompt) throw new KimodoProducerError('input', 'prompt is required');
    const numSteps = Math.max(1, parseInt(opts.steps ?? 100, 10));
    const duration = Number(opts.duration ?? 6);
    const numFrames = Math.max(1, Math.round(duration * config.fps));
    const generationId = opts.generationId ?? ++generationCounter;
    const signal = opts.signal ?? null;
    // The effective endpoint is recorded on the receipt; a per-generation
    // override lets a page-level URL field change without reloading weights.
    const embedUrl = typeof opts.embedUrl === 'string' && opts.embedUrl ? opts.embedUrl : input.embedUrl;
    const stage = (name, event) => opts.onStage?.(name, event);
    const checkCancelled = () => {
      if (signal?.aborted) throw new KimodoProducerError('cancelled', 'generation cancelled by caller (AbortSignal)');
    };

    const profile = createStagedProfile();

    stage('text-embedding', 'start');
    profile.start('text-embedding');
    const textEmbedding = await fetchEmbedding(prompt, embedUrl);
    profile.end();
    stage('text-embedding', 'end');
    checkCancelled();

    stage('ddim-sampling', 'start');
    profile.start('ddim-sampling');
    const gpuAbort = new AbortController();
    if (signal) signal.addEventListener('abort', () => gpuAbort.abort(), { once: true });
    const submissions = createWebGpuBoundedSubmissionQueue({
      queue,
      maxInFlightDuties: opts.maxInFlightDuties ?? 2,
      signal: gpuAbort.signal,
    });
    const hostSubmit = (commandBuffers) => queue.submit(commandBuffers);
    let gpuSubmissionSummary = null;
    let motion;
    try {
      const sched = ddimSchedule(numSteps);
      motion = gaussianNoise(numFrames);
      const textArr = Array.from(textEmbedding);
      for (let step = numSteps - 1; step >= 0; step--) {
        checkCancelled();
        const n = numSteps - step;
        const predClean = await denoiseStepWebGPU(
          device, weights, textArr, motion, sched.useTimesteps[step], motionRepStats,
          {
            submissions,
            dutyPrefix: `g${generationId}-s${n}`,
            // Foreground-opportunity boundary between admitted duties: the
            // host may submit its own work on the shared queue here.
            afterPass: opts.foregroundOpportunity
              ? ({ pass }) => opts.foregroundOpportunity({
                submit: hostSubmit, signal: gpuAbort.signal,
                phase: 'ddim-sampling', step: n, numSteps, pass,
              })
              : undefined,
          },
        );
        const sqrtRecip = sched.sqrtRecipAlphasCumprod[step];
        const sqrtRecipm1 = sched.sqrtRecipm1AlphasCumprod[step];
        const alphaBarPrev = sched.alphasCumprodPrev[step];
        for (let f = 0; f < numFrames; f++) {
          for (let d = 0; d < MOTION_DIM; d++) {
            const eps = (sqrtRecip * motion[f][d] - predClean[f][d]) / sqrtRecipm1;
            motion[f][d] = predClean[f][d] * Math.sqrt(alphaBarPrev) + Math.sqrt(1 - alphaBarPrev) * eps;
          }
        }
        await opts.onProgress?.({ step: n, numSteps, pct: Math.round(100 * n / numSteps) });
      }
      const report = await submissions.drain();
      gpuSubmissionSummary = summarizeSubmissionReport(report);
    } catch (err) {
      // Stop admission and settle accepted duties before surfacing; the
      // original error stays authoritative, the drain report is evidence.
      gpuAbort.abort();
      try { gpuSubmissionSummary = summarizeSubmissionReport(await submissions.drain()); }
      catch (drainErr) { gpuSubmissionSummary = summarizeSubmissionReport(drainErr?.boundedGpuSubmissionReport ?? null, drainErr); }
      if (err instanceof KimodoProducerError) { err.gpuSubmission = gpuSubmissionSummary; throw err; }
      const wrapped = new KimodoProducerError(err?.name === 'AbortError' ? 'cancelled' : 'ddim-sampling', err?.message ?? String(err));
      wrapped.cause = err;
      wrapped.gpuSubmission = gpuSubmissionSummary;
      throw wrapped;
    }
    profile.end();
    stage('ddim-sampling', 'end');

    stage('fk-decode', 'start');
    profile.start('fk-decode');
    const decoded = decodeMotion(motion);
    profile.end();
    stage('fk-decode', 'end');

    stage('output-capture', 'start');
    profile.start('output-capture');
    // Kit-negotiated identity (host-provided) is the receipt's backend
    // authority; Kimodo adapter/device details ride as additive fields.
    const backend = captureBackendIdentity(input.adapter ?? null, device, input.backendIdentity ?? null);
    setTextEmbeddingEndpoint(backend, embedUrl);
    const receipt = await createKimodoRouteReceipt({
      prompt,
      joints: decoded.joints,
      motionFeatures: motion,
      numFrames: decoded.num_frames,
      numJoints: decoded.num_joints,
      numSteps,
      backend,
      profile,
      generationId,
      weightsHash,
      gpuSubmission: gpuSubmissionSummary,
    });
    profile.end();
    stage('output-capture', 'end');

    return {
      receipt,
      motion: {
        generationId,
        prompt,
        fps: config.fps,
        motion,                   // [N] x Array(369) raw features; last 4 = foot contacts
        joints: decoded.joints,   // [N][30][3] FK world positions
        parents: decoded.parents,
        numFrames: decoded.num_frames,
        numJoints: decoded.num_joints,
      },
      submission: gpuSubmissionSummary,
    };
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    // Producer-owned GPU resources only: the model weights this producer
    // uploaded. The device belongs to the host.
    destroyOwned(weights);
  }

  return { identity, generate, dispose };
}
