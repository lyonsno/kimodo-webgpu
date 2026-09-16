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

process.exit(failures ? 1 : 0);
