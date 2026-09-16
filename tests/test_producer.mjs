/**
 * Producer entrypoint contract (live-flame composition slice).
 *
 * The host (Kaminos, per wake-and-bake-pit-boss's composition contract)
 * needs a callable that:
 *  1. initializes on a BORROWED device/queue — never requests an adapter,
 *     never destroys the device;
 *  2. generates from a prompt with progress, cancellation, and the four
 *     stage boundaries preserved (text-embedding, ddim-sampling, fk-decode,
 *     output-capture) rather than a spinner;
 *  3. offers a real foreground-opportunity boundary between admitted GPU
 *     duties — a callback receiving { submit, signal, phase, step, pass }
 *     that the host uses to advance its flame on the shared queue;
 *  4. returns the native motion result (joints [N][30][3], 369-dim rows,
 *     parents, fps, counts) with the route receipt and resolved identity;
 *  5. cleans up producer-owned resources only.
 *
 * Driven through the shipped modules with the counting fake device used by
 * the pacing suite, committed public assets, and a fake embedding endpoint.
 */

globalThis.GPUBufferUsage = {
  MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
  INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
};
globalThis.GPUMapMode = { READ: 0x0001, WRITE: 0x0002 };

import { readFileSync } from 'node:fs';
const { createKimodoProducer } = await import('../src/lib/producer.js');

let failures = 0;
function check(name, ok, detail = '') {
  if (ok) console.log(`  ok  ${name}`);
  else { failures++; console.error(`FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

const publicAsset = (name) => JSON.parse(readFileSync(new URL(`../public/${name}`, import.meta.url), 'utf8'));
const config = publicAsset('kimodo.json');
const fkData = publicAsset('fk_data.json');
const motionRepStats = publicAsset('motion_rep_stats.json');

function makeFakeDevice(counters) {
  const passEncoder = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
  return {
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
    createBuffer: (desc) => ({
      size: desc.size, destroy() { counters.destroyed++; },
      mapAsync: async () => {}, getMappedRange: () => new ArrayBuffer(desc.size), unmap() {},
    }),
    createCommandEncoder: () => ({ copyBufferToBuffer() {}, beginComputePass: () => passEncoder, finish: () => ({}) }),
    queue: {
      submit: () => { counters.queueSubmits++; },
      onSubmittedWorkDone: async () => {},
      writeBuffer() {},
    },
    limits: { maxBufferSize: 1 << 30 },
    features: new Set(),
    destroy() { counters.deviceDestroyed++; },
  };
}
const anyBuffer = () => ({ destroy() {} });
const fakeWeights = () => {
  const layer = () => ({
    inProjW: anyBuffer(), inProjB: anyBuffer(), outProjW: anyBuffer(), outProjB: anyBuffer(),
    norm1W: anyBuffer(), norm1B: anyBuffer(), norm2W: anyBuffer(), norm2B: anyBuffer(),
    ffn1W: anyBuffer(), ffn1B: anyBuffer(), ffn2W: anyBuffer(), ffn2B: anyBuffer(),
  });
  const net = () => ({
    inputLinear: { weight: anyBuffer(), bias: anyBuffer() },
    embedText: { weight: anyBuffer(), bias: anyBuffer() },
    timestepMLP: { linear1: { weight: anyBuffer(), bias: anyBuffer() }, linear2: { weight: anyBuffer(), bias: anyBuffer() } },
    headingLinear: { weight: anyBuffer(), bias: anyBuffer() },
    outputLinear: { weight: anyBuffer(), bias: anyBuffer() },
    layers: Array.from({ length: 16 }, layer),
  });
  return { root: net(), body: net() };
};
const fakeEmbedFetch = (calls) => async (url, init) => {
  calls.push({ url, body: JSON.parse(init.body) });
  return { ok: true, status: 200, json: async () => ({ embedding: Array.from({ length: config.text_dim }, () => 0.01) }) };
};

const backendIdentity = {
  kind: 'webgpu-local', runtime: 'browser', adapterName: 'fake', browser: 'test',
  requestedFeatures: [], features: [], limits: { maxBufferSize: 1 << 30 }, timestampQuery: 'unavailable',
};

async function makeProducer(counters, extra = {}) {
  const device = makeFakeDevice(counters);
  const embedCalls = [];
  const producer = await createKimodoProducer({
    device,
    embedUrl: 'http://embed.test/embed',
    fetch: fakeEmbedFetch(embedCalls),
    backendIdentity,
    assets: { config, fkData, motionRepStats, weights: fakeWeights(), weightsHash: 'f'.repeat(64) },
    ...extra,
  });
  return { device, producer, embedCalls };
}

// --- 1. Borrowed device: no adapter request, no device destroy -------------

{
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  const { producer } = await makeProducer(counters);
  check('producer initializes on a borrowed device without touching navigator.gpu',
    typeof globalThis.navigator === 'undefined' || !globalThis.navigator?.gpu, 'test runs without navigator.gpu at all');
  check('producer exposes resolved identity (model, kit version, embed endpoint, config fps)',
    producer.identity?.model?.id === 'NVIDIA/Kimodo-SOMA-RP-v1.1'
      && typeof producer.identity?.kitVersion === 'string'
      && producer.identity?.embedUrl === 'http://embed.test/embed'
      && producer.identity?.fps === config.fps,
    JSON.stringify(producer.identity));
  producer.dispose();
  check('dispose never destroys the borrowed device', counters.deviceDestroyed === 0);
  let mutated = false;
  try { producer.identity.model.weightsHash = 'tampered'; mutated = producer.identity.model.weightsHash === 'tampered'; } catch { /* strict-mode throw also counts as protected */ }
  check('identity is frozen through the nested model record', !mutated && Object.isFrozen(producer.identity.model));
}

// --- 2+3+4. Generation: stages, progress, boundaries, native result ---------

{
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  const { device, producer, embedCalls } = await makeProducer(counters);
  const stages = [];
  const progress = [];
  const boundaries = [];
  const result = await producer.generate({
    prompt: 'a person walks forward',
    steps: 3,
    duration: 0.1, // 3 frames at 30fps
    onStage: (name, event) => stages.push(`${name}:${event}`),
    onProgress: (p) => progress.push(p),
    foregroundOpportunity: async (b) => {
      boundaries.push({ phase: b.phase, step: b.step, pass: b.pass, hasSubmit: typeof b.submit === 'function', hasSignal: !!b.signal });
      b.submit([{}]); // host advances its flame on the shared queue
    },
  });
  check('embedding is fetched from the configured endpoint with the prompt',
    embedCalls.length === 1 && embedCalls[0].url === 'http://embed.test/embed'
      && embedCalls[0].body.prompt === 'a person walks forward');
  check('the four stage boundaries are preserved in order',
    stages.join(',') === 'text-embedding:start,text-embedding:end,ddim-sampling:start,ddim-sampling:end,fk-decode:start,fk-decode:end,output-capture:start,output-capture:end',
    stages.join(','));
  check('progress is reported per step with a denominator',
    progress.length === 3 && progress.every((p) => p.numSteps === 3) && progress.at(-1).step === 3,
    JSON.stringify(progress));
  check('foreground boundary fires between admitted duties with submit + signal',
    boundaries.length === 3 * 4 && boundaries.every((b) => b.hasSubmit && b.hasSignal && b.phase === 'ddim-sampling'),
    `boundaries=${boundaries.length} ${JSON.stringify(boundaries[0])}`);
  check('host submissions on the shared queue reach the device queue',
    counters.queueSubmits >= 3 * 4 + 3 * 4, `queueSubmits=${counters.queueSubmits}`);
  check('native motion result carries joints/rows/parents/counts/fps',
    result.motion?.joints?.length === 3 && result.motion.joints[0].length === 30
      && result.motion.motion.length === 3 && result.motion.motion[0].length === 369
      && Array.isArray(result.motion.parents) && result.motion.fps === config.fps
      && result.motion.numFrames === 3 && result.motion.numJoints === 30,
    JSON.stringify({ frames: result.motion?.joints?.length, joints: result.motion?.joints?.[0]?.length }));
  check('receipt is terminal and carries submission telemetry and identity',
    result.receipt?.status === 'real' && result.receipt?.metadata?.gpuSubmission?.submittedDutyCount === 12
      && result.receipt?.model?.weightsHash === 'f'.repeat(64),
    JSON.stringify({ status: result.receipt?.status, sub: result.receipt?.metadata?.gpuSubmission }));
  producer.dispose();
  check('dispose after generation still never destroys the device', counters.deviceDestroyed === 0);
}

// --- 2. Cancellation ---------------------------------------------------------

{
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  const { producer } = await makeProducer(counters);
  const abort = new AbortController();
  let steps = 0;
  let error = null;
  try {
    await producer.generate({
      prompt: 'x', steps: 5, duration: 0.1,
      signal: abort.signal,
      onProgress: () => { steps++; if (steps === 2) abort.abort(); },
    });
  } catch (err) { error = err; }
  check('cancellation aborts generation after the current step with a loud error',
    error != null && /abort|cancel/i.test(error.message ?? error.name ?? '') && steps === 2,
    JSON.stringify({ error: error?.message ?? error?.name, steps }));
  producer.dispose();
}

// --- Embedding failure is loud and phase-named -------------------------------

{
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  const { producer } = await makeProducer(counters, {
    fetch: async () => ({ ok: false, status: 503, statusText: 'down', json: async () => ({}) }),
  });
  let error = null;
  try { await producer.generate({ prompt: 'x', steps: 1, duration: 0.1 }); } catch (err) { error = err; }
  check('embedding HTTP failure fails loud with its phase',
    error?.phase === 'embedding-http' && /503/.test(error.message), JSON.stringify({ phase: error?.phase, m: error?.message }));
}

// --- main.js delegates to the producer -----------------------------------------

{
  const mainSrc = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
  check('main.js constructs the producer and generates through it',
    mainSrc.includes('createKimodoProducer') && /producer\.generate\(/.test(mainSrc));
  check('main.js no longer owns the DDIM schedule directly',
    !/alphasCumprodBase/.test(mainSrc), 'schedule must live in the producer');
}

// --- Review findings: cancellation coverage, host-submit settlement, ------
// --- weights ownership, injected fetch on every asset --------------------

function makeSignalSpy() {
  // A fake AbortSignal that counts listeners so leaks are observable.
  const listeners = new Set();
  const spy = {
    aborted: false, reason: undefined,
    addEventListener: (type, fn) => { if (type === 'abort') listeners.add(fn); },
    removeEventListener: (type, fn) => { if (type === 'abort') listeners.delete(fn); },
    abort() { spy.aborted = true; for (const fn of [...listeners]) fn(); },
    get listenerCount() { return listeners.size; },
  };
  return spy;
}

{
  // Pre-aborted signal: no embedding request, immediate cancelled error.
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  const { producer, embedCalls } = await makeProducer(counters);
  const abort = new AbortController(); abort.abort();
  let error = null;
  try { await producer.generate({ prompt: 'x', steps: 2, duration: 0.1, signal: abort.signal }); } catch (err) { error = err; }
  check('a pre-aborted signal does no work and fails cancelled',
    error?.phase === 'cancelled' && embedCalls.length === 0, JSON.stringify({ phase: error?.phase, fetches: embedCalls.length }));
}

{
  // Abort during a never-resolving embedding fetch must settle the call.
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  let sawSignal = false;
  const { producer } = await makeProducer(counters, {
    fetch: (url, init) => new Promise((_, reject) => {
      sawSignal = !!init.signal;
      init.signal?.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' })));
    }),
  });
  const abort = new AbortController();
  const pending = producer.generate({ prompt: 'x', steps: 2, duration: 0.1, signal: abort.signal });
  setTimeout(() => abort.abort(), 20);
  let error = null;
  try { await Promise.race([pending, new Promise((_, r) => setTimeout(() => r(new Error('did not settle')), 2000))]); } catch (err) { error = err; }
  check('abort during the embedding fetch settles the generation as cancelled',
    sawSignal && error?.phase === 'cancelled', JSON.stringify({ sawSignal, phase: error?.phase, m: error?.message }));
}

{
  // Abort while the foreground callback never resolves: the call settles and
  // the retained submit capability is revoked.
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  const { producer } = await makeProducer(counters);
  const abort = new AbortController();
  let retainedSubmit = null;
  const pending = producer.generate({
    prompt: 'x', steps: 2, duration: 0.1, signal: abort.signal,
    foregroundOpportunity: (b) => { retainedSubmit = b.submit; return new Promise(() => {}); },
  });
  setTimeout(() => abort.abort(), 50);
  let error = null;
  try { await Promise.race([pending, new Promise((_, r) => setTimeout(() => r(new Error('did not settle')), 3000))]); } catch (err) { error = err; }
  let lateSubmitThrew = false;
  try { retainedSubmit?.([{}]); } catch { lateSubmitThrew = true; }
  check('abort while a foreground callback is pending settles the call as cancelled',
    error?.phase === 'cancelled', JSON.stringify({ phase: error?.phase, m: error?.message }));
  check('a retained submit capability is revoked after the generation ends',
    retainedSubmit != null && lateSubmitThrew);
}

{
  // Host callback submits then throws: the producer must fence the host's
  // accepted work before rejecting, and carry the count in evidence.
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  const device = makeFakeDevice(counters);
  let fences = 0; let fenceResolved = 0;
  device.queue.onSubmittedWorkDone = async () => { fences++; await new Promise((r) => setTimeout(r, 5)); fenceResolved++; };
  const producer = await createKimodoProducer({
    device, embedUrl: 'http://embed.test/embed', fetch: fakeEmbedFetch([]), backendIdentity,
    assets: { config, fkData, motionRepStats, weights: fakeWeights(), weightsHash: 'f'.repeat(64) },
  });
  let error = null;
  try {
    await producer.generate({
      prompt: 'x', steps: 2, duration: 0.1,
      foregroundOpportunity: (b) => { b.submit([{}]); throw new Error('host frame failed after submit'); },
    });
  } catch (err) { error = err; }
  check('a callback that submits then throws surfaces as a producer failure',
    error != null && /host frame failed/.test(error.message), String(error?.message));
  check('the host work accepted before the throw is fenced before rejection',
    error?.gpuSubmission?.hostSubmissionCount === 1 && fenceResolved === fences && fences >= 2,
    JSON.stringify({ host: error?.gpuSubmission?.hostSubmissionCount, fences, fenceResolved }));
}

{
  // Callback throws BEFORE submitting: clean failure, no host work tracked.
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  const { producer } = await makeProducer(counters);
  let error = null;
  try {
    await producer.generate({ prompt: 'x', steps: 1, duration: 0.1,
      foregroundOpportunity: () => { throw new Error('host declined'); } });
  } catch (err) { error = err; }
  check('a callback that throws before submitting fails with zero host submissions',
    /host declined/.test(error?.message ?? '') && error?.gpuSubmission?.hostSubmissionCount === 0,
    JSON.stringify({ m: error?.message, host: error?.gpuSubmission?.hostSubmissionCount }));
}

{
  // Long-lived signal reused across successful generations must not leak
  // abort listeners.
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  const { producer } = await makeProducer(counters);
  const spy = makeSignalSpy();
  for (let i = 0; i < 3; i++) await producer.generate({ prompt: 'x', steps: 1, duration: 0.1, signal: spy });
  check('abort listeners are removed after each successful generation',
    spy.listenerCount === 0, `listeners=${spy.listenerCount}`);
}

{
  // Ownership: caller-supplied weights are borrowed unless transferred.
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  const device = makeFakeDevice(counters);
  let supplied = 0;
  const trackedBuffer = () => ({ destroy() { supplied++; } });
  const suppliedWeights = fakeWeights();
  suppliedWeights.root.inputLinear.weight = trackedBuffer();
  const borrowed = await createKimodoProducer({ device, embedUrl: 'http://e/embed', fetch: fakeEmbedFetch([]), backendIdentity,
    assets: { config, fkData, motionRepStats, weights: suppliedWeights } });
  borrowed.dispose();
  check('dispose leaves caller-supplied weights untouched by default', supplied === 0, `destroyed=${supplied}`);
  const transferred = await createKimodoProducer({ device, embedUrl: 'http://e/embed', fetch: fakeEmbedFetch([]), backendIdentity,
    assets: { config, fkData, motionRepStats, weights: suppliedWeights, transferWeightsOwnership: true } });
  transferred.dispose();
  check('dispose destroys supplied weights only when ownership was explicitly transferred', supplied === 1, `destroyed=${supplied}`);
}

// A real (tiny) weight binary in the shipped format, so the assetBase path
// is executable end to end: every tensor the loader requires, one fp32 each.
function synthesizeWeightsBin() {
  const names = [];
  for (const prefix of ['body_model', 'root_model']) {
    for (const n of ['input_linear', 'embed_text', 'output_linear', 'linear_first_heading_angle']) names.push(`${prefix}.${n}.weight`, `${prefix}.${n}.bias`);
    names.push(`${prefix}.embed_timestep.time_embed.0.weight`, `${prefix}.embed_timestep.time_embed.0.bias`,
      `${prefix}.embed_timestep.time_embed.2.weight`, `${prefix}.embed_timestep.time_embed.2.bias`);
    for (let i = 0; i < 16; i++) {
      const lp = `${prefix}.seqTransEncoder.layers.${i}`;
      names.push(`${lp}.norm1.weight`, `${lp}.norm1.bias`, `${lp}.norm2.weight`, `${lp}.norm2.bias`,
        `${lp}.self_attn.in_proj_weight`, `${lp}.self_attn.in_proj_bias`, `${lp}.self_attn.out_proj.weight`, `${lp}.self_attn.out_proj.bias`,
        `${lp}.linear1.weight`, `${lp}.linear1.bias`, `${lp}.linear2.weight`, `${lp}.linear2.bias`);
    }
  }
  const headerSize = 16 + names.length * 96;
  const buf = new ArrayBuffer(headerSize + names.length * 4);
  const view = new DataView(buf);
  view.setUint32(0, 0x444D494B, true); view.setUint32(4, 1, true);
  view.setUint32(8, names.length, true); view.setUint32(12, headerSize, true);
  const enc = new TextEncoder();
  names.forEach((name, i) => {
    const off = 16 + i * 96;
    new Uint8Array(buf, off, 64).set(enc.encode(name).slice(0, 63));
    view.setUint32(off + 64, 0, true); view.setUint32(off + 68, 1, true); view.setUint32(off + 72, 1, true);
    view.setUint32(off + 88, headerSize + i * 4, true); view.setUint32(off + 92, 4, true);
  });
  return buf;
}

function assetFetch(calls, { failOn = null, rejectOn = null, readerRejects = false } = {}) {
  const bin = synthesizeWeightsBin();
  const jsonResp = (obj) => ({ ok: true, status: 200, json: async () => obj });
  return async (url, init) => {
    calls.push(url);
    if (rejectOn && url.endsWith(rejectOn)) throw new TypeError('fetch failed');
    if (failOn && url.endsWith(failOn)) return { ok: false, status: 500, statusText: 'boom', json: async () => ({}) };
    if (readerRejects && url.endsWith('/kimodo.bin')) {
      return { ok: true, status: 200, headers: { get: () => String(bin.byteLength) },
        body: { getReader: () => ({ read: async () => { throw new Error('stream reset mid-body'); } }) } };
    }
    if (url.endsWith('/kimodo.json')) return jsonResp(config);
    if (url.endsWith('/fk_data.json')) return jsonResp(fkData);
    if (url.endsWith('/motion_rep_stats.json')) return jsonResp(motionRepStats);
    if (url.endsWith('/kimodo.bin')) {
      let sent = false;
      return { ok: true, status: 200, headers: { get: () => String(bin.byteLength) },
        body: { getReader: () => ({ read: async () => sent ? { done: true } : (sent = true, { done: false, value: new Uint8Array(bin) }) }) } };
    }
    if (url.endsWith('/embed')) return jsonResp({ embedding: Array.from({ length: config.text_dim }, () => 0.01) });
    return { ok: false, status: 404, statusText: 'nope', json: async () => ({}) };
  };
}

{
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  const device = makeFakeDevice(counters);
  const calls = [];
  const savedGlobalFetch = globalThis.fetch;
  globalThis.fetch = async () => { throw new Error('global fetch used'); };
  let producer = null; let error = null;
  try {
    producer = await createKimodoProducer({ device, embedUrl: 'http://assets.test/embed', fetch: assetFetch(calls), assetBase: 'http://assets.test', backendIdentity });
  } catch (err) { error = err; } finally { globalThis.fetch = savedGlobalFetch; }
  check('assetBase path loads all four resources through the injected fetch only',
    error == null && ['kimodo.json', 'fk_data.json', 'motion_rep_stats.json', 'kimodo.bin'].every((n) => calls.some((u) => u.endsWith('/' + n))),
    JSON.stringify({ error: error?.message, calls }));
  check('the consumed weights hash is the SHA-256 of the loaded bytes',
    /^[0-9a-f]{64}$/.test(producer?.identity?.model?.weightsHash ?? ''), String(producer?.identity?.model?.weightsHash));
  const destroyedBefore = counters.destroyed;
  producer?.dispose();
  check('dispose destroys producer-loaded weights (asset path is producer-owned)',
    counters.destroyed > destroyedBefore, `destroyed delta=${counters.destroyed - destroyedBefore}`);
}

{
  // Partial-initialization rollback: the loader fails part-way, every buffer
  // it already created is destroyed, and the failure is phase-named.
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  const device = makeFakeDevice(counters);
  let created = 0;
  const origCreate = device.createBuffer;
  device.createBuffer = (desc) => { created++; if (created === 50) throw new Error('injected allocation failure'); return origCreate(desc); };
  let error = null;
  try {
    await createKimodoProducer({ device, embedUrl: 'http://assets.test/embed', fetch: assetFetch([]), assetBase: 'http://assets.test', backendIdentity });
  } catch (err) { error = err; }
  check('a failed weight load rolls back every buffer it created and names its phase',
    error?.phase === 'load-weights' && counters.destroyed === 49,
    JSON.stringify({ phase: error?.phase, created, destroyed: counters.destroyed }));
}

{
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  let error = null;
  try {
    await createKimodoProducer({ device: makeFakeDevice(counters), embedUrl: 'http://assets.test/embed',
      fetch: assetFetch([], { failOn: '/motion_rep_stats.json' }), assetBase: 'http://assets.test', backendIdentity });
  } catch (err) { error = err; }
  check('a failing asset response is phase-named, not a raw error',
    error?.phase === 'load-stats', JSON.stringify({ phase: error?.phase, m: error?.message }));
}

// --- r2 review: cancellation lifetime, race-listener cleanup, ------------
// --- post-allocation rollback, complete asset-phase attribution ----------

for (const [stageName, label] of [['fk-decode', 'FK decode'], ['output-capture', 'output capture']]) {
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  const { producer } = await makeProducer(counters);
  const abort = new AbortController();
  let outcome = null; let error = null;
  try {
    const result = await producer.generate({
      prompt: 'x', steps: 1, duration: 0.1, signal: abort.signal,
      onStage: (name, event) => { if (name === stageName && event === 'start') abort.abort(); },
    });
    outcome = result?.receipt?.status ?? 'returned';
  } catch (err) { error = err; }
  check(`an abort at ${label} start settles the generation as cancelled, never as a real receipt`,
    outcome === null && error?.phase === 'cancelled', JSON.stringify({ outcome, phase: error?.phase }));
}

{
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  const { producer } = await makeProducer(counters);
  const spy = makeSignalSpy();
  const pending = producer.generate({ prompt: 'x', steps: 2, duration: 0.1, signal: spy,
    foregroundOpportunity: () => new Promise(() => {}) });
  setTimeout(() => spy.abort(), 50);
  let error = null;
  try { await Promise.race([pending, new Promise((_, r) => setTimeout(() => r(new Error('did not settle')), 3000))]); } catch (err) { error = err; }
  check('an abort with a foreign signal and a never-settling hook leaves zero listeners after rejection',
    error?.phase === 'cancelled' && spy.listenerCount === 0, JSON.stringify({ phase: error?.phase, listeners: spy.listenerCount }));
}

for (const failAt of [1, 50]) {
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  const device = makeFakeDevice(counters);
  let allocated = 0;
  const origCreate = device.createBuffer;
  device.createBuffer = (desc) => {
    const buf = origCreate(desc);
    allocated++;
    if (allocated === failAt) buf.getMappedRange = () => { throw new Error('injected mapped initialization failure'); };
    return buf;
  };
  let error = null;
  try {
    await createKimodoProducer({ device, embedUrl: 'http://assets.test/embed', fetch: assetFetch([]), assetBase: 'http://assets.test', backendIdentity });
  } catch (err) { error = err; }
  check(`a mapped-initialization failure on buffer ${failAt} destroys every allocated buffer, including the failing one`,
    error?.phase === 'load-weights' && counters.destroyed === allocated && allocated === failAt,
    JSON.stringify({ phase: error?.phase, allocated, destroyed: counters.destroyed }));
}

for (const [mode, expectPhase, label] of [
  [{ rejectOn: '/kimodo.json' }, 'load-config', 'a rejected config request'],
  [{ rejectOn: '/kimodo.bin' }, 'load-weights', 'a rejected weights request'],
  [{ readerRejects: true }, 'load-weights', 'a weight body stream that fails mid-read'],
]) {
  const counters = { queueSubmits: 0, destroyed: 0, deviceDestroyed: 0 };
  let error = null;
  try {
    await createKimodoProducer({ device: makeFakeDevice(counters), embedUrl: 'http://assets.test/embed',
      fetch: assetFetch([], mode), assetBase: 'http://assets.test', backendIdentity });
  } catch (err) { error = err; }
  check(`${label} is phase-named ${expectPhase} with the original error as cause`,
    error?.phase === expectPhase && error?.cause instanceof Error,
    JSON.stringify({ phase: error?.phase ?? null, name: error?.name, m: error?.message, cause: error?.cause?.message ?? null }));
}

process.exit(failures ? 1 : 0);
