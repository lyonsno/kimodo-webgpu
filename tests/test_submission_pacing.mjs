/**
 * Submission-pacing contract (slice C, revision after the combined review).
 *
 * The review demonstrated two false claims in the first cut: duty IDs
 * collided between CFG passes (every real generation would fail its first
 * step against the installed controller), and per-encoder duties relocated
 * ~28k prefix fences INTO the kit while the permissive fake counted zero.
 *
 * The contract now, proven against the INSTALLED kit controller:
 *
 * 1. One forward pass = ONE command encoder = ONE duty. forwardTransformer
 *    emits no host fences itself and performs no debug readback.
 * 2. Duty identity is caller-owned and unique for the generation:
 *    {prefix}-{cond|uncond}-{root|body}. A duplicate duty id must throw in
 *    the installed controller (negative case pinned).
 * 3. Effective fence accounting at pass granularity: one denoise step =
 *    4 duties (one kit prefix fence each) + 4 readback fences = 8 queue
 *    fences per step (~800 per 100-step generation), down from ~29,000 at
 *    encoder granularity. The test asserts the EXACT counts for one step
 *    against the real controller and a counting fake queue.
 * 4. drain() terminates with submitted == completed == 4 and zero failures.
 */

globalThis.GPUBufferUsage = {
  MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
  INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
};
globalThis.GPUMapMode = { READ: 0x0001, WRITE: 0x0002 };

const { forwardTransformer, readBuffer } = await import('../src/lib/inference.js');
const { denoiseStepWebGPU } = await import('../src/lib/denoiser.js');
const { createWebGpuBoundedSubmissionQueue } = await import('@kaminos/webgpu-inference-kit');

let failures = 0;
function check(name, ok, detail = '') {
  if (ok) console.log(`  ok  ${name}`);
  else { failures++; console.error(`FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

function makeCounters() {
  return { directSubmits: 0, queueSubmits: 0, fences: 0, mapReadBuffers: 0, finished: 0 };
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
      submit: () => { counters.queueSubmits++; },
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

// --- One pass = one encoder = one duty (installed controller) --------------

{
  const counters = makeCounters();
  const device = makeFakeDevice(counters);
  const submissions = createWebGpuBoundedSubmissionQueue({
    queue: device.queue, maxInFlightDuties: 2,
  });
  const out = await forwardTransformer(
    device, fakeWeights(), anyBuffer(), anyBuffer(), 500, 8, 738, 5, null,
    { submissions, dutyId: 'g1-s1-cond-root' },
  );
  check('one forward pass finishes exactly one command encoder',
    counters.finished === 1, `finished=${counters.finished}`);
  check('the single command buffer reaches the queue through the controller',
    counters.queueSubmits === 1, `queueSubmits=${counters.queueSubmits}`);
  check('forwardTransformer allocates no MAP_READ buffer (debug readback gone)',
    counters.mapReadBuffers === 0, `mapReadBuffers=${counters.mapReadBuffers}`);
  check('forwardTransformer returns an output buffer', out != null);
  const report = await submissions.drain();
  check('controller drains terminal with the one duty completed',
    report.submittedDutyCount === 1 && report.completedDutyCount === 1
      && report.failedDutyCount === 0 && report.inFlightDutyCount === 0,
    JSON.stringify({ s: report.submittedDutyCount, c: report.completedDutyCount }));
}

{
  const counters = makeCounters();
  const device = makeFakeDevice(counters);
  await forwardTransformer(device, fakeWeights(), anyBuffer(), anyBuffer(), 500, 8, 738, 5);
  check('without a submissions context, the single buffer submits directly, unfenced',
    counters.queueSubmits === 1 && counters.fences === 0 && counters.finished === 1,
    JSON.stringify(counters));
}

// --- Duplicate duty identity must fail loud in the installed controller ----

{
  const counters = makeCounters();
  const device = makeFakeDevice(counters);
  const submissions = createWebGpuBoundedSubmissionQueue({
    queue: device.queue, maxInFlightDuties: 2,
  });
  await forwardTransformer(device, fakeWeights(), anyBuffer(), anyBuffer(), 500, 4, 738, 5, null,
    { submissions, dutyId: 'g1-s1-cond-root' });
  let threw = null;
  try {
    await forwardTransformer(device, fakeWeights(), anyBuffer(), anyBuffer(), 500, 4, 738, 5, null,
      { submissions, dutyId: 'g1-s1-cond-root' });
  } catch (err) { threw = err; }
  check('a duplicate duty id is rejected by the installed controller',
    threw != null && /duplicate/i.test(threw.message), String(threw?.message));
}

// --- readBuffer keeps its lawful fence -------------------------------------

{
  const counters = makeCounters();
  const device = makeFakeDevice(counters);
  const data = await readBuffer(device, anyBuffer(), 16);
  check('readBuffer fences exactly once at its readback boundary',
    counters.fences === 1 && data.length === 16, `fences=${counters.fences} len=${data.length}`);
}

// --- Full denoise step against the installed controller --------------------

{
  const counters = makeCounters();
  const device = makeFakeDevice(counters);
  const submissions = createWebGpuBoundedSubmissionQueue({
    queue: device.queue, maxInFlightDuties: 2,
  });
  const stats = {
    fps: 30,
    global_root_mean: [0, 0, 0, 0, 0], global_root_std: [1, 1, 1, 1, 1],
    local_root_mean: [0, 0, 0, 0], local_root_std: [1, 1, 1, 1],
  };
  const N = 4;
  const motion = Array.from({ length: N }, () => new Array(369).fill(0));
  const prediction = await denoiseStepWebGPU(
    device, { root: fakeWeights(), body: fakeWeights() },
    new Float32Array(4096), motion, 500, stats,
    { submissions, dutyPrefix: 'g1-s1' },
  );
  const report = await submissions.drain();
  check('one denoise step admits exactly four duties (root/body x cond/uncond)',
    report.submittedDutyCount === 4 && report.completedDutyCount === 4
      && report.failedDutyCount === 0,
    JSON.stringify({ s: report.submittedDutyCount, c: report.completedDutyCount, f: report.failedDutyCount }));
  const ids = report.duties.map((d) => d.dutyId);
  check('all four duty ids are unique and carry cfg-role and submodel identity',
    new Set(ids).size === 4
      && ids.some((i) => /cond-root/.test(i)) && ids.some((i) => /uncond-root/.test(i))
      && ids.some((i) => /cond-body/.test(i)) && ids.some((i) => /uncond-body/.test(i)),
    JSON.stringify(ids));
  check('effective fence count for one step is 8 (4 controller prefix + 4 readback), not per-encoder',
    counters.fences === 8, `fences=${counters.fences}`);
  check('exactly eight command encoders were finished for the step (4 pass + 4 readback copies)',
    counters.finished === 8,
    `finished=${counters.finished}`);
  check('denoise step returns [N, 369] prediction',
    prediction.length === N && prediction[0].length === 369);
  check('denoise step performs exactly its four required readbacks',
    counters.mapReadBuffers === 4, `mapReadBuffers=${counters.mapReadBuffers}`);
}

// --- Allocation accounting: fixed scratch, not per-layer growth ------------
// The r1 single-encoder cut kept all 16 layers' scratch simultaneously live
// (~249 MiB at 180 frames, ~738 MiB at the UI-maximum 540) while claiming
// "same peak". Scratch is now a fixed reused set: peak live logical bytes
// must be bounded by ONE layer's working set, at the default AND the
// maximum supported duration, and every per-call buffer must be destroyed
// after a successful pass (the returned output excepted until its caller
// frees it).

function makeAccountingDevice(counters, { failSubmit = false } = {}) {
  const live = new Map(); // buffer object -> bytes
  let liveBytes = 0;
  const acct = {
    peakBytes: 0,
    createdCount: 0,
    destroyedCount: 0,
    get liveBytes() { return liveBytes; },
    live,
  };
  const passEncoder = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
  const device = {
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
    createBuffer: (desc) => {
      acct.createdCount++;
      const buf = {
        size: desc.size,
        destroy() {
          if (live.has(buf)) { liveBytes -= desc.size; live.delete(buf); acct.destroyedCount++; }
        },
        mapAsync: async () => {},
        getMappedRange: () => new ArrayBuffer(desc.size),
        unmap() {},
      };
      live.set(buf, desc.size);
      liveBytes += desc.size;
      acct.peakBytes = Math.max(acct.peakBytes, liveBytes);
      return buf;
    },
    createCommandEncoder: () => ({
      copyBufferToBuffer() {},
      beginComputePass: () => passEncoder,
      finish: () => ({}),
    }),
    queue: {
      submit: () => { if (failSubmit) throw new Error('injected queue submit failure'); },
      onSubmittedWorkDone: async () => {},
      writeBuffer() {},
    },
    limits: {},
  };
  return { device, acct };
}

const MiB = 1024 * 1024;

async function measureForward(frames, { failSubmit = false } = {}) {
  const { device, acct } = makeAccountingDevice({}, { failSubmit });
  let out = null, error = null;
  try {
    out = await forwardTransformer(device, fakeWeights(), anyBuffer(), anyBuffer(), 500, frames, 738, 5);
  } catch (err) { error = err; }
  return { acct, out, error };
}

{
  const def = await measureForward(180);
  const max = await measureForward(540);
  check('default-duration pass peaks under 64 MiB of logical scratch',
    def.acct.peakBytes < 64 * MiB, `${(def.acct.peakBytes / MiB).toFixed(1)} MiB`);
  check('maximum-duration (540-frame) pass peaks under 120 MiB of logical scratch',
    max.acct.peakBytes < 120 * MiB, `${(max.acct.peakBytes / MiB).toFixed(1)} MiB`);
  // The old regression scaled peak with layer count; the fixed set does not:
  // sixteen layers' worth at 540 frames was ~738 MiB.
  // Persistent per-device caches (uniforms, dummy mask) legitimately stay
  // live; the honest leak check is the warm-cache delta: a second identical
  // pass on the same device must add exactly ONE live buffer — its output.
  const { device: warmDev, acct: warmAcct } = makeAccountingDevice({});
  await forwardTransformer(warmDev, fakeWeights(), anyBuffer(), anyBuffer(), 500, 180, 738, 5);
  const liveAfterFirst = warmAcct.live.size;
  await forwardTransformer(warmDev, fakeWeights(), anyBuffer(), anyBuffer(), 500, 180, 738, 5);
  check('a warm-cache pass leaks nothing: live delta is exactly the returned output',
    warmAcct.live.size === liveAfterFirst + 1,
    `first=${liveAfterFirst} second=${warmAcct.live.size}`);
}

{
  const { acct, out, error } = await measureForward(180, { failSubmit: true });
  check('an injected queue-submit failure propagates from the pass',
    out === null && /injected queue submit failure/.test(error?.message ?? ''), String(error?.message));
  check('a rejected submission destroys every per-call buffer including the would-be output',
    acct.live.size <= 2,
    `live=${acct.live.size} (persistent cached uniforms/dummy mask only)`);
}

{
  // Same rejection path through the INSTALLED controller (duplicate duty).
  const counters = makeCounters();
  const { device, acct } = makeAccountingDevice({});
  const submissions = createWebGpuBoundedSubmissionQueue({ queue: device.queue, maxInFlightDuties: 2 });
  await forwardTransformer(device, fakeWeights(), anyBuffer(), anyBuffer(), 500, 4, 738, 5, null,
    { submissions, dutyId: 'dup' });
  const before = acct.live.size;
  let threw = null;
  try {
    await forwardTransformer(device, fakeWeights(), anyBuffer(), anyBuffer(), 500, 4, 738, 5, null,
      { submissions, dutyId: 'dup' });
  } catch (err) { threw = err; }
  check('a controller-rejected duty leaks no per-call buffers',
    threw != null && acct.live.size === before,
    `before=${before} after=${acct.live.size}`);
}

process.exit(failures ? 1 : 0);
