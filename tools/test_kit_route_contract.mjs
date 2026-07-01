import assert from 'node:assert/strict';
import {
  addStagedSubmitStage,
  createKimodoTextToMotionRouteDefinition,
  createKimodoTextToMotionRouteReceipt,
  createStagedSubmitProfile,
  createWebGpuBackendIdentity,
  KIMODO_TEXT_TO_MOTION_ROUTE_ID,
  WEBGPU_INFERENCE_KIT_VERSION,
  validateRouteReceipt,
} from '@kaminos/webgpu-inference-kit';

assert.equal(WEBGPU_INFERENCE_KIT_VERSION, '0.1.1');

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
