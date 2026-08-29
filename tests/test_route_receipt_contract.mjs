/**
 * Adversarial contract tests for src/lib/route-receipt.js.
 *
 * An independent review found that the receipt's validity predicate counted
 * non-finite values only AFTER flattening into a Float32Array, which is
 * vacuously true for empty arrays; that declared shapes were synthesized from
 * arguments rather than observed from data; and that the emitted object did not
 * satisfy the @kaminos/webgpu-inference-kit schema it named.
 *
 * These tests assert the failure paths. Run:
 *   node tests/test_route_receipt_contract.mjs
 */
import { createKimodoRouteReceipt, createStagedProfile } from '../src/lib/route-receipt.js';
import { validateRouteReceipt } from '@kaminos/webgpu-inference-kit';

const results = [];
const check = (name, ok, detail = '') => results.push({ name, ok, detail });

const profile = () => { const p = createStagedProfile(); p.start('s'); p.end(); return p; };
const backend = () => ({
  kind: 'webgpu-local',
  runtime: 'test-runtime',
  adapter: { vendor: 'test', architecture: 'test' },
  device: {},
  externalities: [{ service: 'text-embedding', endpoint: null, device: 'unknown' }],
});

const goodJoints = (f, j) =>
  Array.from({ length: f }, () => Array.from({ length: j }, () => [0.1, 0.2, 0.3]));
const goodMotion = (f) => Array.from({ length: f }, () => Array(369).fill(0.1));

async function receipt(over = {}) {
  return createKimodoRouteReceipt({
    prompt: 'a person walks',
    joints: goodJoints(2, 4),
    motionFeatures: goodMotion(2),
    numFrames: 2, numJoints: 4, numSteps: 50,
    backend: backend(), profile: profile(),
    ...over,
  });
}

// --- the happy path must still produce a real receipt ---
{
  const r = await receipt();
  check('valid output -> status real', r.status === 'real', `got ${r.status}`);
  check('valid output -> both outputs real',
        r.outputs.filter(o => o.role !== 'filmstrip').every(o => o.status === 'real'));
}

// --- degenerate / malformed outputs must NOT be certified real ---
const bad = [
  ['empty joints and motion',      { joints: [], motionFeatures: [] }],
  ['empty joints only',            { joints: [] }],
  ['empty motion only',            { motionFeatures: [] }],
  ['truncated joints (1 of 2 frames)', { joints: goodJoints(1, 4) }],
  ['truncated motion (1 of 2 frames)', { motionFeatures: goodMotion(1) }],
  ['wrong joint count per frame',  { joints: goodJoints(2, 3) }],
  ['wrong motion width',           { motionFeatures: [Array(368).fill(0.1), Array(368).fill(0.1)] }],
  ['NaN in joints',                { joints: [[[NaN, 0, 0], [0,0,0], [0,0,0], [0,0,0]], ...goodJoints(1, 4)] }],
  ['Infinity in motion',           { motionFeatures: [[Infinity, ...Array(368).fill(0.1)], Array(369).fill(0.1)] }],
  ['coercible strings in motion',  { motionFeatures: [Array(369).fill('0.1'), Array(369).fill('0.1')] }],
  ['null in joints',               { joints: [[[null, 0, 0], [0,0,0], [0,0,0], [0,0,0]], ...goodJoints(1, 4)] }],
  ['mis-nested joints (flat)',     { joints: [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]] }],
];
for (const [label, over] of bad) {
  let status;
  try {
    status = (await receipt(over)).status;
  } catch (e) {
    status = `threw:${e.constructor.name}`;   // throwing is an acceptable refusal
  }
  check(`${label} -> not real`, status !== 'real', `got ${status}`);
}

// --- declared shapes must reflect observed data, not the arguments ---
{
  let ok = false;
  try {
    const r = await receipt({ joints: goodJoints(1, 4) }); // claims 2 frames, supplies 1
    const jointsOut = r.outputs.find(o => o.role === 'soma77-joints');
    ok = r.status !== 'real' || jointsOut.shape[0] === 1;
  } catch { ok = true; }
  check('shape reflects observed data or run is refused', ok);
}

// --- filmstrip must not be certified from an empty payload ---
{
  let ok = false;
  try {
    const r = await receipt({ filmstripData: new Uint8Array(0) });
    const film = r.outputs.find(o => o.role === 'filmstrip');
    ok = !film || film.status !== 'real';
  } catch { ok = true; }
  check('empty filmstrip payload -> not real', ok);
}

// --- the emitted receipt must satisfy the kit schema it names ---
{
  const r = await receipt();
  const v = validateRouteReceipt(r);
  check('receipt validates against @kaminos/webgpu-inference-kit',
        v.ok === true, (v.errors || []).join('; '));
}

const failed = results.filter(r => !r.ok);
for (const r of results) console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${r.name}${!r.ok && r.detail ? `   [${r.detail}]` : ''}`);
console.log(`\n${results.length - failed.length}/${results.length} passed`);
process.exit(failed.length ? 1 : 0);
