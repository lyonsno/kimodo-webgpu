/**
 * Kit-integration contract (slice A, revision 1).
 *
 * Claims under test, sharpened by the fresh-context review:
 *
 * 1. The receipt's TOP-LEVEL backend is a kit-valid identity: the kit's
 *    strict evidence consumer (classifyWebGpuRouteReceiptEvidence) validates
 *    receipt.backend itself, so the happy-path receipt must classify
 *    authoritative — a valid identity nested under an invalid top level is
 *    the exact anti-pattern the review demonstrated and must fail here.
 * 2. Emission-time self-validation demotes kit-rejected receipts loudly, and
 *    the user-facing explanation distinguishes kit/schema invalidity from
 *    non-finite-output invalidity (they previously shared one false message).
 * 3. initGPU delegates to the kit BEHAVIORALLY: a fake `gpu` records the
 *    actual requestAdapter options and requestDevice descriptor, which kills
 *    the string-presence mutant the review built. The kit carries its six
 *    supported inference limits at adapter-reported values — six, not "all";
 *    the test asserts the honest contract.
 */

import {
  createKimodoRouteReceipt,
  createStagedProfile,
  applyKitValidation,
  describeInvalidReceipt,
} from '../src/lib/route-receipt.js';
import { initGPU } from '../src/lib/gpu.js';
import {
  WEBGPU_INFERENCE_KIT_VERSION,
  validateWebGpuBackendIdentity,
  classifyWebGpuRouteReceiptEvidence,
} from '@kaminos/webgpu-inference-kit';
import { readFileSync } from 'node:fs';

let failures = 0;
function check(name, ok, detail = '') {
  if (ok) console.log(`  ok  ${name}`);
  else { failures++; console.error(`FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

const kitBackend = () => ({
  kind: 'webgpu-local',
  runtime: 'browser',
  adapterName: 'Apple M4 Max',
  browser: 'test-harness',
  requestedFeatures: [],
  features: ['shader-f16'],
  limits: { maxBufferSize: 4294967296, maxStorageBufferBindingSize: 2147483648 },
  timestampQuery: 'unavailable',
});

const goodInput = (backend = kitBackend()) => {
  const numFrames = 4, numJoints = 30;
  const joints = Array.from({ length: numFrames }, () =>
    Array.from({ length: numJoints }, () => [0.1, 0.2, 0.3]));
  const motionFeatures = Array.from({ length: numFrames }, () =>
    Array.from({ length: 369 }, () => 0.5));
  const profile = createStagedProfile();
  profile.start('ddim-sampling');
  profile.end();
  return {
    prompt: 'a person walks forward',
    joints, motionFeatures, numFrames, numJoints, numSteps: 8,
    backend,
    profile,
    weightsHash: 'a'.repeat(64),
    generationId: 3,
  };
};

// --- Top-level backend authority (review P1) -------------------------------

{
  const receipt = await createKimodoRouteReceipt(goodInput());
  const identity = validateWebGpuBackendIdentity(receipt.backend);
  check('receipt.backend is itself a kit-valid identity',
    identity.ok === true, JSON.stringify(identity.errors));

  const evidence = classifyWebGpuRouteReceiptEvidence(receipt);
  check('happy-path receipt classifies authoritative under the strict consumer',
    evidence.classification === 'authoritative-live-webgpu' && evidence.authoritative === true,
    JSON.stringify(evidence));
}

{
  // The review's anti-pattern: legacy shape on top, valid identity nested
  // beneath it. This must NOT classify authoritative — proving the test can
  // detect the regression the previous revision shipped.
  const legacyTopLevel = {
    kind: 'webgpu-local',
    runtime: 'browser-webgpu',
    adapter: { vendor: 'apple' },
    device: { maxBufferSize: 1 },
    kitIdentity: kitBackend(),
  };
  const receipt = await createKimodoRouteReceipt(goodInput(legacyTopLevel));
  const evidence = classifyWebGpuRouteReceiptEvidence(receipt);
  check('valid identity nested under an invalid top-level backend cannot classify authoritative',
    evidence.authoritative !== true, JSON.stringify(evidence.classification));
}

// --- Emission-time self-validation (review P2b included) -------------------

{
  const receipt = await createKimodoRouteReceipt(goodInput());
  check('good receipt records kitValidation from the installed kit',
    receipt.kitValidation?.ok === true
      && receipt.kitValidation?.kitVersion === WEBGPU_INFERENCE_KIT_VERSION,
    JSON.stringify(receipt.kitValidation));
  check('good receipt stays real after self-validation', receipt.status === 'real', receipt.status);
}

{
  const receipt = await createKimodoRouteReceipt(goodInput());
  delete receipt.outputs[0].sha256;
  const revalidated = applyKitValidation(receipt);
  check('kit-rejected receipt demotes to invalid', revalidated.status === 'invalid', revalidated.status);
  check('demoted receipt names the kit failure in fallbackReason',
    typeof revalidated.fallbackReason === 'string' && revalidated.fallbackReason.length > 0
      && revalidated.kitValidation?.ok === false,
    JSON.stringify({ reason: revalidated.fallbackReason, v: revalidated.kitValidation }));
}

{
  const receipt = await createKimodoRouteReceipt(goodInput());
  receipt.status = 'failed';
  check('self-validation never upgrades a non-real status',
    applyKitValidation(receipt).status === 'failed');
}

{
  // Finite, correctly-shaped outputs + a kit-rejected backend: the user-facing
  // explanation must describe the kit rejection, not claim non-finite output.
  const broken = kitBackend();
  broken.runtime = '';
  const receipt = await createKimodoRouteReceipt(goodInput(broken));
  check('kit-schema demotion produces an invalid receipt with real outputs',
    receipt.status === 'invalid' && receipt.outputs.every(o => o.status === 'real'),
    JSON.stringify({ status: receipt.status, outputs: receipt.outputs.map(o => o.status) }));
  const line = describeInvalidReceipt(receipt);
  check('kit-schema demotion explanation does not claim non-finite output',
    typeof line === 'string' && !/non-finite/.test(line) && /kit validation/i.test(line), line);
}

{
  // Output-derived invalidity keeps its established explanation.
  const input = goodInput();
  input.motionFeatures[1][5] = Number.NaN;
  const receipt = await createKimodoRouteReceipt(input);
  const line = describeInvalidReceipt(receipt);
  check('output-derived invalidity keeps the non-finite explanation',
    receipt.status === 'invalid' && /non-finite/.test(line), JSON.stringify({ status: receipt.status, line }));
}

// --- Behavioral initGPU delegation (review P2a) ----------------------------

function fakeGpuEnvironment({ withTimestamp }) {
  const observed = { adapterOptions: null, descriptor: null };
  const limits = {
    maxBufferSize: 1024, maxStorageBufferBindingSize: 512,
    maxComputeWorkgroupStorageSize: 64, maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupSizeY: 256,
    maxBindGroups: 4, // outside the kit's six-key contract; must not be requested
  };
  const device = {
    features: new Set(withTimestamp ? ['timestamp-query'] : []),
    limits,
    lost: new Promise(() => {}),
  };
  const adapter = {
    features: new Set(withTimestamp ? ['timestamp-query'] : []),
    limits,
    info: { vendor: 'fake', description: 'Fake Adapter' },
    requestDevice: async (descriptor) => { observed.descriptor = descriptor; return device; },
  };
  const gpu = {
    requestAdapter: async (options) => { observed.adapterOptions = options; return adapter; },
  };
  return { gpu, observed };
}

{
  const { gpu, observed } = fakeGpuEnvironment({ withTimestamp: false });
  const { device, backendIdentity } = await initGPU(gpu);
  check('initGPU forwards the high-performance adapter preference to the kit path',
    observed.adapterOptions?.powerPreference === 'high-performance',
    JSON.stringify(observed.adapterOptions));
  check('device request carries the kit\'s six inference limits at adapter values, and only those',
    observed.descriptor
      && Object.keys(observed.descriptor.requiredLimits).length === 6
      && observed.descriptor.requiredLimits.maxBufferSize === 1024
      && !('maxBindGroups' in observed.descriptor.requiredLimits),
    JSON.stringify(observed.descriptor?.requiredLimits));
  check('without adapter support, timestamp-query is not requested and is recorded unavailable',
    (observed.descriptor.requiredFeatures ?? []).length === 0
      && backendIdentity.timestampQuery === 'unavailable',
    JSON.stringify({ f: observed.descriptor.requiredFeatures, t: backendIdentity.timestampQuery }));
  check('initGPU returns the live device and a kit-valid identity',
    device != null && validateWebGpuBackendIdentity(backendIdentity).ok === true,
    JSON.stringify(validateWebGpuBackendIdentity(backendIdentity).errors));
}

{
  const { gpu, observed } = fakeGpuEnvironment({ withTimestamp: true });
  const { backendIdentity } = await initGPU(gpu);
  check('with adapter support, timestamp-query is requested and recorded',
    (observed.descriptor.requiredFeatures ?? []).includes('timestamp-query')
      && backendIdentity.timestampQuery === 'requested',
    JSON.stringify({ f: observed.descriptor.requiredFeatures, t: backendIdentity.timestampQuery }));
}

// --- Wiring (source presence, routing only) --------------------------------

const mainSrc = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
check('main.js uses the kit identity as the receipt backend authority',
  /gpuBackendIdentity\s*=\s*captureBackendIdentity\(adapter,\s*device,\s*backendIdentity\)/.test(mainSrc),
  'expected captureBackendIdentity(adapter, device, backendIdentity) with the kit identity as base');

const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
const pin = pkg.devDependencies?.['@kaminos/webgpu-inference-kit'] ?? pkg.dependencies?.['@kaminos/webgpu-inference-kit'];
check('kit pin is current (^0.1.46)', pin === '^0.1.46', String(pin));

process.exit(failures ? 1 : 0);
