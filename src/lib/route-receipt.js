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
const MOTION_FEATURE_DIM = 369;
const KIT_VERSION = '0.1.1';
// The kit requires a non-empty weights identity. Callers that have not hashed
// the weight binary pass this sentinel rather than an empty string, so the
// receipt stays schema-valid while remaining honest about what is unknown.
const WEIGHTS_HASH_UNKNOWN = 'unknown-weights-hash';

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
        // Device is NOT asserted here. The embedding server supports mps,
        // cuda and cpu, and any compatible server may be pointed at, so
        // claiming MPS would report a route we did not observe. The effective
        // endpoint is recorded by the caller via setTextEmbeddingEndpoint().
        reason: 'Text encoder runs server-side. Browser receives a 4096-dim vector via the /embed endpoint.',
        impact: 'Text embedding is not client-side. Diffusion, CFG, DDIM and FK decode are client-side.',
        endpoint: null,
        device: 'unknown',
      },
    ],
  };
}

/**
 * Record the effective text-embedding endpoint on a backend identity object.
 *
 * Preserves the route actually used rather than a compile-time assumption.
 * `device` stays 'unknown' unless the server reports one.
 */
export function setTextEmbeddingEndpoint(backend, endpoint, device = 'unknown') {
  const ext = backend?.externalities?.find(e => e.service === 'text-embedding');
  if (ext) {
    ext.endpoint = endpoint ?? null;
    ext.device = device || 'unknown';
  }
  return backend;
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
  weightsHash = WEIGHTS_HASH_UNKNOWN,
  generationId = null,  // binds this receipt to one generation; see main.js
}) {
  const promptHash = await sha256(prompt);
  const promptId = `prompt-${promptHash.slice(0, 16)}`;

  // Validate the ORIGINAL arrays before hashing or typed-array coercion.
  //
  // Counting non-finite values after `new Float32Array(x.flat())` is vacuously
  // true for empty arrays, blind to truncated or mis-nested data, and lets
  // coercible values ("0.1", null) become finite numbers before they are ever
  // checked. Validate structure and element types on the source data instead.
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

  const validateJoints = () => {
    if (!Array.isArray(joints) || joints.length === 0) return 'joints is empty or not an array';
    if (joints.length !== numFrames) return `expected ${numFrames} frames, got ${joints.length}`;
    for (let f = 0; f < joints.length; f++) {
      const frame = joints[f];
      if (!Array.isArray(frame)) return `frame ${f} is not an array`;
      if (frame.length !== numJoints) return `frame ${f} has ${frame.length} joints, expected ${numJoints}`;
      for (let j = 0; j < frame.length; j++) {
        const p = frame[j];
        if (!Array.isArray(p) || p.length !== 3) return `joint ${f}/${j} is not a 3-tuple`;
        for (let k = 0; k < 3; k++) if (!isNum(p[k])) return `joint ${f}/${j}[${k}] is not a finite number`;
      }
    }
    return null;
  };

  const validateMotion = () => {
    if (!Array.isArray(motionFeatures) || motionFeatures.length === 0) return 'motionFeatures is empty or not an array';
    if (motionFeatures.length !== numFrames) return `expected ${numFrames} frames, got ${motionFeatures.length}`;
    for (let f = 0; f < motionFeatures.length; f++) {
      const row = motionFeatures[f];
      if (!Array.isArray(row)) return `motion frame ${f} is not an array`;
      if (row.length !== MOTION_FEATURE_DIM) return `motion frame ${f} has ${row.length} values, expected ${MOTION_FEATURE_DIM}`;
      for (let i = 0; i < row.length; i++) if (!isNum(row[i])) return `motion ${f}[${i}] is not a finite number`;
    }
    return null;
  };

  const jointsError = validateJoints();
  const motionError = validateMotion();

  // Hash outputs (safe to coerce now that the source data is validated).
  const jointsFlat = new Float32Array(jointsError ? [] : joints.flat(2));
  const jointsHash = await sha256(jointsFlat);
  const jointsId = `soma77-joints-${jointsHash.slice(0, 16)}`;

  const motionFlat = new Float32Array(motionError ? [] : motionFeatures.flat());
  const motionHash = await sha256(motionFlat);
  const motionId = `motion-clip-${motionHash.slice(0, 16)}`;

  const jointsStatus = jointsError ? 'invalid' : 'real';
  const motionStatus = motionError ? 'invalid' : 'real';
  const outputsValid = !jointsError && !motionError;

  // Shapes describe what was OBSERVED, not what the caller declared.
  const observedJointShape = jointsError
    ? [Array.isArray(joints) ? joints.length : 0]
    : [joints.length, joints[0].length, 3];
  const observedMotionShape = motionError
    ? [Array.isArray(motionFeatures) ? motionFeatures.length : 0]
    : [motionFeatures.length, MOTION_FEATURE_DIM];

  const outputs = [
    {
      role: 'soma77-joints',
      artifactId: jointsId,
      sha256: jointsHash,
      shape: observedJointShape,
      status: jointsStatus,
      invalidReason: jointsError,
    },
    {
      role: 'motion-clip',
      artifactId: motionId,
      sha256: motionHash,
      shape: observedMotionShape,
      status: motionStatus,
      invalidReason: motionError,
    },
  ];

  if (filmstripData) {
    // An empty Uint8Array is truthy; hashing it would certify a nonexistent
    // image. Require an actual payload before marking the filmstrip real.
    const filmHash = await sha256(filmstripData);
    const filmValid = filmstripData.byteLength > 0;
    outputs.push({
      role: 'filmstrip',
      artifactId: `filmstrip-${filmHash.slice(0, 16)}`,
      sha256: filmHash,
      shape: [filmstripData.byteLength],
      status: filmValid ? 'real' : 'invalid',
      invalidReason: filmValid ? null : 'empty filmstrip payload',
    });
  }

  const finishedProfile = profile.finish();

  return {
    schema: 'kaminos.webgpu-route-receipt.v0',
    requestedRouteId: ROUTE_ID,
    effectiveRouteId: ROUTE_ID,
    status: outputsValid ? 'real' : 'invalid',
    fallbackReason: outputsValid
      ? null
      : [jointsError && `joints: ${jointsError}`, motionError && `motion: ${motionError}`]
          .filter(Boolean).join('; '),
    // `createdAt` is the kit's freshness-bearing field; `timestamp` is retained
    // for existing readers.
    createdAt: new Date().toISOString(),
    timestamp: new Date().toISOString(),
    generationId,
    backend: {
      // The kit requires an explicit backend kind and runtime string.
      kind: 'webgpu-local',
      runtime: backend?.runtime || 'browser-webgpu',
      ...backend,
    },
    model: {
      id: MODEL_ID,
      revision: 'SOMA-RP-v1.1',
      dtype: 'fp16',
      weightsHash,
    },
    kernel: {
      kitVersion: KIT_VERSION,
      profile: 'kimodo-text-to-motion',
    },
    inputs: [{
      role: 'text-prompt',
      artifactId: promptId,
      sha256: promptHash,
    }],
    outputs,
    timings: {
      source: finishedProfile.timingSource || 'adapter-phase-wall-clock',
      totalMs: finishedProfile.totalMs ?? 0,
      stages: Object.values(finishedProfile.stages || {}),
    },
    profile: finishedProfile,
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
