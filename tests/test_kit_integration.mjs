/**
 * Kit-integration contract (slice A).
 *
 * Two claims under test:
 *
 * 1. Emission-time self-validation: every receipt runs the INSTALLED kit's
 *    validateRouteReceipt before it is returned. A receipt the kit rejects
 *    must demote itself to 'invalid' and say why — schema drift between app
 *    and kit fails loud in the live app, not only in this test suite.
 * 2. Device bring-up goes through the kit (requestBrowserWebGpuDevice), so
 *    limits are copied without silent downcapping and timestamp-query is
 *    negotiated — the timing authority slice C depends on.
 *
 * Module behavior is tested by executing shipped code; main.js/gpu.js wiring
 * follows the source-presence pattern of test_generation_identity.mjs.
 */

import {
  createKimodoRouteReceipt,
  createStagedProfile,
  applyKitValidation,
} from '../src/lib/route-receipt.js';
import { WEBGPU_INFERENCE_KIT_VERSION } from '@kaminos/webgpu-inference-kit';
import { readFileSync } from 'node:fs';

let failures = 0;
function check(name, ok, detail = '') {
  if (ok) console.log(`  ok  ${name}`);
  else { failures++; console.error(`FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

const goodInput = () => {
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
    backend: { kind: 'webgpu-local', runtime: 'browser-webgpu' },
    profile,
    weightsHash: 'a'.repeat(64),
    generationId: 3,
  };
};

// --- Emission-time self-validation ---------------------------------------

{
  const receipt = await createKimodoRouteReceipt(goodInput());
  check('good receipt records kitValidation from the installed kit',
    receipt.kitValidation?.ok === true
      && receipt.kitValidation?.kitVersion === WEBGPU_INFERENCE_KIT_VERSION,
    JSON.stringify(receipt.kitValidation));
  check('good receipt stays real after self-validation',
    receipt.status === 'real', receipt.status);
}

{
  // Mangle a kit-required field AFTER construction, then re-validate: the
  // helper must demote and carry the kit's reason.
  const receipt = await createKimodoRouteReceipt(goodInput());
  delete receipt.outputs[0].sha256;
  const revalidated = applyKitValidation(receipt);
  check('kit-rejected receipt demotes to invalid',
    revalidated.status === 'invalid', revalidated.status);
  check('demoted receipt names the kit failure in fallbackReason',
    typeof revalidated.fallbackReason === 'string' && revalidated.fallbackReason.length > 0
      && revalidated.kitValidation?.ok === false,
    JSON.stringify({ reason: revalidated.fallbackReason, v: revalidated.kitValidation }));
}

{
  // Demotion must not upgrade or mask an already-failed status.
  const receipt = await createKimodoRouteReceipt(goodInput());
  receipt.status = 'failed';
  const revalidated = applyKitValidation(receipt);
  check('self-validation never upgrades a non-real status',
    revalidated.status === 'failed', revalidated.status);
}

// --- Wiring (source presence) ---------------------------------------------

const receiptSrc = readFileSync(new URL('../src/lib/route-receipt.js', import.meta.url), 'utf8');
check('createKimodoRouteReceipt runs applyKitValidation before returning',
  /return\s+applyKitValidation\s*\(/.test(receiptSrc),
  'expected the constructor to return through applyKitValidation');
check('route-receipt imports the kit validator itself',
  /import\s*\{[^}]*validateRouteReceipt[^}]*\}\s*from\s*'@kaminos\/webgpu-inference-kit'/.test(receiptSrc));

const gpuSrc = readFileSync(new URL('../src/lib/gpu.js', import.meta.url), 'utf8');
check('gpu.js acquires the device through the kit',
  gpuSrc.includes('requestBrowserWebGpuDevice'),
  'expected initGPU to delegate to the kit device acquisition');
check('gpu.js negotiates timestamp-query (prefer, not require)',
  /timestampQuery:\s*'prefer'/.test(gpuSrc));

const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
const pin = pkg.devDependencies?.['@kaminos/webgpu-inference-kit'] ?? pkg.dependencies?.['@kaminos/webgpu-inference-kit'];
check('kit pin is current (^0.1.46)', pin === '^0.1.46', String(pin));

process.exit(failures ? 1 : 0);
