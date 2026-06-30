/**
 * route-receipt.js — Kimodo WebGPU route receipt emission.
 *
 * Follows @kaminos/webgpu-inference-kit receipt contract.
 * Route: kimodo.text-to-motion.webgpu-local.v0
 *
 * Emits receipts that preserve:
 * - Input: text prompt with artifact id/hash
 * - Outputs: soma77-joints, motion-clip, optional filmstrip
 * - Backend: WebGPU adapter/device identity + server-side text embedding note
 * - Profile: staged timing for text-embedding, ddim-sampling, fk-decode, output-capture
 */

const ROUTE_ID = 'kimodo.text-to-motion.webgpu-local.v0';
const MODEL_ID = 'NVIDIA/Kimodo-SOMA-RP-v1.1';

/**
 * Capture WebGPU backend identity from the device.
 */
export function captureBackendIdentity(adapter, device) {
  return {
    kind: 'webgpu-local',
    adapter: {
      vendor: adapter?.info?.vendor || 'unknown',
      architecture: adapter?.info?.architecture || 'unknown',
      device: adapter?.info?.device || 'unknown',
      description: adapter?.info?.description || 'unknown',
    },
    device: {
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
    },
    externalities: [
      {
        service: 'text-embedding',
        reason: 'Llama 3 8B text encoder runs server-side (MPS). Browser receives a 4096-dim vector via /embed endpoint.',
        impact: 'Text embedding is not client-side. The diffusion model and FK decode are fully client-side.',
      },
    ],
  };
}

/**
 * SHA-256 hash of a string or typed array.
 */
async function sha256(data) {
  const buffer = typeof data === 'string'
    ? new TextEncoder().encode(data)
    : (data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer || data));
  const hash = await crypto.subtle.digest('SHA-256', buffer);
  return Array.from(new Uint8Array(hash)).map(b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Create a staged profile tracker.
 */
export function createStagedProfile() {
  const stages = {};
  let currentStage = null;
  let currentStart = null;

  return {
    start(name) {
      if (currentStage) this.end();
      currentStage = name;
      currentStart = performance.now();
    },
    end() {
      if (currentStage && currentStart != null) {
        stages[currentStage] = {
          name: currentStage,
          durationMs: Math.round(performance.now() - currentStart),
          timestamp: new Date().toISOString(),
        };
        currentStage = null;
        currentStart = null;
      }
    },
    finish() {
      this.end();
      return {
        timingSource: 'adapter-phase-wall-clock',
        stages,
        totalMs: Object.values(stages).reduce((sum, s) => sum + s.durationMs, 0),
      };
    },
  };
}

/**
 * Create a Kimodo text-to-motion route receipt.
 */
export async function createKimodoRouteReceipt({
  prompt,
  joints,       // decoded joint positions [N, J, 3]
  motionFeatures, // raw 369-dim features [N, 369]
  numFrames,
  numJoints,
  numSteps,
  backend,
  profile,
  filmstripData,  // optional Uint8Array PNG
}) {
  const promptHash = await sha256(prompt);
  const promptId = `prompt-${promptHash.slice(0, 16)}`;

  // Hash outputs
  const jointsFlat = new Float32Array(joints.flat(2));
  const jointsHash = await sha256(jointsFlat);
  const jointsId = `soma77-joints-${jointsHash.slice(0, 16)}`;

  const motionFlat = new Float32Array(motionFeatures.flat());
  const motionHash = await sha256(motionFlat);
  const motionId = `motion-clip-${motionHash.slice(0, 16)}`;

  const outputs = [
    {
      role: 'soma77-joints',
      artifactId: jointsId,
      sha256: jointsHash,
      shape: [numFrames, numJoints, 3],
      status: 'real',
    },
    {
      role: 'motion-clip',
      artifactId: motionId,
      sha256: motionHash,
      shape: [numFrames, 369],
      status: 'real',
    },
  ];

  if (filmstripData) {
    const filmHash = await sha256(filmstripData);
    outputs.push({
      role: 'filmstrip',
      artifactId: `filmstrip-${filmHash.slice(0, 16)}`,
      sha256: filmHash,
      shape: [1],
      status: 'real',
    });
  }

  return {
    schema: 'kaminos.webgpu-route-receipt.v0',
    requestedRouteId: ROUTE_ID,
    effectiveRouteId: ROUTE_ID,
    status: 'real',
    fallbackReason: null,
    timestamp: new Date().toISOString(),
    backend,
    model: {
      id: MODEL_ID,
      revision: 'SOMA-RP-v1.1',
      dtype: 'fp16',
    },
    inputs: [{
      role: 'text-prompt',
      artifactId: promptId,
      sha256: promptHash,
    }],
    outputs,
    profile: profile.finish(),
    metadata: {
      numFrames,
      numJoints,
      numSteps,
      fps: 30,
      textEmbeddingSource: 'server-side-llama3-8b',
      diffusionBackend: 'webgpu-compute-shaders',
      fkBackend: 'js-cpu',
    },
  };
}
