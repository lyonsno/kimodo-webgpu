/**
 * Submission-pacing contract (slice C).
 *
 * The DDIM route previously fenced the host on onSubmittedWorkDone after
 * nearly every encoder (~70 per forward pass, 4 passes per step, plus a
 * debug GPU readback with console dumps on every pass): tens of thousands
 * of host<->GPU round-trips per generation. The contract now:
 *
 * 1. forwardTransformer emits NO host fences of its own. Correctness needs
 *    only queue submission order; the only lawful fence is inside
 *    readBuffer, at a real readback boundary.
 * 2. With a submissions context (the kit's bounded submission queue shape),
 *    every command buffer routes through submitDuty — the queue controller
 *    owns queue.submit, so admission depth actually paces the GPU.
 *    Without one, command buffers submit directly (still unfenced).
 * 3. The per-pass debug readback is gone: forwardTransformer allocates no
 *    MAP_READ buffer.
 * 4. denoiseStepWebGPU threads the submissions context through both models
 *    and both CFG passes, and still performs exactly its four required
 *    readbacks (root + body, conditioned + unconditioned).
 *
 * Driven through the SHIPPED modules with a counting fake device.
 */

globalThis.GPUBufferUsage = {
  MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
  INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
};
globalThis.GPUMapMode = { READ: 0x0001, WRITE: 0x0002 };

const { forwardTransformer, readBuffer } = await import('../src/lib/inference.js');
const { denoiseStepWebGPU } = await import('../src/lib/denoiser.js');

let failures = 0;
function check(name, ok, detail = '') {
  if (ok) console.log(`  ok  ${name}`);
  else { failures++; console.error(`FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

function makeCounters() {
  return { directSubmits: 0, fences: 0, mapReadBuffers: 0, finished: 0 };
}

function makeFakeDevice(counters) {
  const passEncoder = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
  return {
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createBindGroup: () => ({}),
    createBuffer: (desc) => {
      if (desc.usage & GPUBufferUsage.MAP_READ) counters.mapReadBuffers++;
      return {
        size: desc.size,
        destroy() {},
        mapAsync: async () => {},
        getMappedRange: () => new ArrayBuffer(desc.size),
        unmap() {},
      };
    },
    createCommandEncoder: () => ({
      copyBufferToBuffer() {},
      beginComputePass: () => passEncoder,
      finish: () => ({ __cb: ++counters.finished }),
    }),
    queue: {
      submit: () => { counters.directSubmits++; },
      onSubmittedWorkDone: async () => { counters.fences++; },
      writeBuffer() {},
    },
    limits: {},
  };
}

const anyBuffer = () => ({ destroy() {} });
const fakeWeights = () => {
  const layer = () => ({
    inProjW: anyBuffer(), inProjB: anyBuffer(), outProjW: anyBuffer(), outProjB: anyBuffer(),
    norm1W: anyBuffer(), norm1B: anyBuffer(), norm2W: anyBuffer(), norm2B: anyBuffer(),
    ffn1W: anyBuffer(), ffn1B: anyBuffer(), ffn2W: anyBuffer(), ffn2B: anyBuffer(),
  });
  return {
    inputLinear: { weight: anyBuffer(), bias: anyBuffer() },
    embedText: { weight: anyBuffer(), bias: anyBuffer() },
    timestepMLP: {
      linear1: { weight: anyBuffer(), bias: anyBuffer() },
      linear2: { weight: anyBuffer(), bias: anyBuffer() },
    },
    headingLinear: { weight: anyBuffer(), bias: anyBuffer() },
    outputLinear: { weight: anyBuffer(), bias: anyBuffer() },
    layers: Array.from({ length: 16 }, layer),
  };
};

function makeSubmissions() {
  const record = { duties: [], drained: false };
  return {
    record,
    submitDuty: async ({ dutyId, commandBuffers }) => {
      record.duties.push({ dutyId, count: commandBuffers.length });
    },
    drain: async () => { record.drained = true; return { status: 'drained' }; },
  };
}

// --- forwardTransformer alone ---------------------------------------------

{
  const counters = makeCounters();
  const device = makeFakeDevice(counters);
  const submissions = makeSubmissions();
  const out = await forwardTransformer(
    device, fakeWeights(), anyBuffer(), anyBuffer(), 500, 8, 738, 5, null,
    { submissions },
  );
  check('with a submissions context, no direct queue.submit occurs',
    counters.directSubmits === 0, `directSubmits=${counters.directSubmits}`);
  check('every command buffer routes through submitDuty',
    submissions.record.duties.length > 0
      && submissions.record.duties.length === counters.finished,
    `duties=${submissions.record.duties.length} finished=${counters.finished}`);
  check('forwardTransformer emits zero host fences',
    counters.fences === 0, `fences=${counters.fences}`);
  check('forwardTransformer allocates no MAP_READ buffer (debug readback gone)',
    counters.mapReadBuffers === 0, `mapReadBuffers=${counters.mapReadBuffers}`);
  check('forwardTransformer still returns an output buffer', out != null);
}

{
  const counters = makeCounters();
  const device = makeFakeDevice(counters);
  await forwardTransformer(device, fakeWeights(), anyBuffer(), anyBuffer(), 500, 8, 738, 5);
  check('without a submissions context, command buffers submit directly',
    counters.directSubmits > 0 && counters.directSubmits === counters.finished,
    `directSubmits=${counters.directSubmits} finished=${counters.finished}`);
  check('legacy path also emits zero host fences',
    counters.fences === 0, `fences=${counters.fences}`);
}

// --- readBuffer keeps its lawful fence -------------------------------------

{
  const counters = makeCounters();
  const device = makeFakeDevice(counters);
  const data = await readBuffer(device, anyBuffer(), 16);
  check('readBuffer fences exactly once at its readback boundary',
    counters.fences === 1 && data.length === 16, `fences=${counters.fences} len=${data.length}`);
}

// --- denoiseStepWebGPU threads the context ---------------------------------

{
  const counters = makeCounters();
  const device = makeFakeDevice(counters);
  const submissions = makeSubmissions();
  const stats = {
    fps: 30,
    global_root_mean: [0, 0, 0, 0, 0], global_root_std: [1, 1, 1, 1, 1],
    local_root_mean: [0, 0, 0, 0], local_root_std: [1, 1, 1, 1],
  };
  const N = 4;
  const motion = Array.from({ length: N }, () => new Array(369).fill(0));
  const prediction = await denoiseStepWebGPU(
    device, { root: fakeWeights(), body: fakeWeights() },
    new Float32Array(4096), motion, 500, stats, { submissions },
  );
  check('denoise step routes all transformer work through submitDuty',
    counters.directSubmits === 4 && submissions.record.duties.length > 0,
    `directSubmits=${counters.directSubmits} (expect 4: one per readback) duties=${submissions.record.duties.length}`);
  check('denoise step performs exactly its four required readbacks',
    counters.mapReadBuffers === 4 && counters.fences === 4,
    `mapReadBuffers=${counters.mapReadBuffers} fences=${counters.fences}`);
  check('denoise step returns [N, 369] prediction',
    prediction.length === N && prediction[0].length === 369);
}

process.exit(failures ? 1 : 0);
