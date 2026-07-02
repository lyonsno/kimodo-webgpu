import assert from 'node:assert/strict';
import {
  addStagedSubmitStage,
  createKimodoTextToMotionRouteDefinition,
  createKimodoTextToMotionRouteReceipt,
  createStagedSubmitProfile,
  createWebGpuBackendIdentity,
  KIMODO_TEXT_TO_MOTION_ROUTE_ID,
  WEBGPU_INFERENCE_KIT_VERSION,
  WEBGPU_ROUTE_BACKPRESSURE_SCHEMA,
  WEBGPU_ROUTE_SCHEDULER_SCHEMA,
  validateRouteReceipt,
} from '@kaminos/webgpu-inference-kit';

const [kitMajor, kitMinor, kitPatch] = WEBGPU_INFERENCE_KIT_VERSION.split('.').map(Number);
assert.deepEqual([kitMajor, kitMinor], [0, 1]);
assert.ok(kitPatch >= 4, `breathability contract requires kit >=0.1.4, got ${WEBGPU_INFERENCE_KIT_VERSION}`);

const requiredStages = ['text-embedding', 'ddim-sampling', 'fk-decode', 'output-capture'];

const definition = createKimodoTextToMotionRouteDefinition({
  kernel: {
    kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
    profile: 'twostage-denoiser-ddim50-fk',
    commit: 'kimodo-webgpu-kit-contract-smoke',
  },
});

assert.equal(KIMODO_TEXT_TO_MOTION_ROUTE_ID, 'kimodo.text-to-motion.webgpu-local.v0');
assert.equal(definition.routeId, KIMODO_TEXT_TO_MOTION_ROUTE_ID);
assert.deepEqual(definition.requiredStages, requiredStages);
assert.equal(definition.scheduler.schema, WEBGPU_ROUTE_SCHEDULER_SCHEMA);
assert.equal(definition.scheduler.requestedScheduler.mode, 'cooperative');
assert.equal(definition.backpressure.schema, WEBGPU_ROUTE_BACKPRESSURE_SCHEMA);
assert.equal(definition.backpressure.effectiveBudget, 'visible-wait');
assert.deepEqual(
  definition.scheduler.breathability.spans.map(span => [span.stage, span.kind, span.interruptible]),
  [
    ['text-embedding', 'external-bound', false],
    ['ddim-sampling', 'gpu-submit-loop', false],
    ['fk-decode', 'cpu-bound', true],
    ['output-capture', 'readback-bound', false],
  ],
);
assert.ok(
  definition.scheduler.breathability.checkpoints.some(
    checkpoint => checkpoint.kind === 'diffusion-step' && checkpoint.afterStage === 'ddim-sampling' && checkpoint.yieldable,
  ),
  'Kimodo must expose a yieldable diffusion-step checkpoint',
);
assert.deepEqual(
  definition.outputRoles.filter(output => output.required).map(output => output.role),
  ['soma77-joints', 'motion-clip'],
);

const backend = createWebGpuBackendIdentity({
  adapterName: 'contract-test-webgpu-adapter',
  browser: 'node-contract-smoke',
  requestedFeatures: ['timestamp-query'],
  effectiveFeatures: ['timestamp-query'],
  limits: {
    maxBufferSize: 1024,
    maxStorageBufferBindingSize: 1024,
    maxComputeInvocationsPerWorkgroup: 256,
  },
  timestampQuery: 'requested',
});

const profile = createStagedSubmitProfile({
  route: KIMODO_TEXT_TO_MOTION_ROUTE_ID,
  timingSource: 'adapter-phase-wall-clock',
  requiredStages,
});
for (const [index, name] of requiredStages.entries()) {
  addStagedSubmitStage(profile, { name, ms: index + 1 });
}

const receipt = createKimodoTextToMotionRouteReceipt({
  input: {
    artifactId: 'prompt:test',
    sha256: 'sha256-prompt',
    shape: [1],
  },
  outputs: {
    soma77Joints: {
      artifactId: 'soma77-joints:test',
      sha256: 'sha256-joints',
      shape: [90, 77, 3],
    },
    motionClip: {
      artifactId: 'motion-clip:test',
      sha256: 'sha256-motion',
      shape: [1],
    },
    filmstrip: {
      artifactId: 'filmstrip:test',
      sha256: 'sha256-filmstrip',
      shape: [90, 256, 256, 4],
    },
  },
  backend,
  model: {
    revision: 'SOMA-RP-v1.1',
    weightsHash: 'sha256-weights',
  },
  kernel: {
    kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
    profile: 'twostage-denoiser-ddim50-fk',
    commit: 'kimodo-webgpu-kit-contract-smoke',
  },
  profile,
});

const result = validateRouteReceipt(receipt);
assert.equal(result.ok, true, result.errors.join('; '));
assert.equal(receipt.requestedRouteId, KIMODO_TEXT_TO_MOTION_ROUTE_ID);
assert.deepEqual(receipt.outputs.map(output => output.role), [
  'soma77-joints',
  'motion-clip',
  'filmstrip',
]);

console.log('Kimodo kit route contract passed');
