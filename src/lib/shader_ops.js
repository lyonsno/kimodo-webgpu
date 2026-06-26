/**
 * shader_ops.js — WebGPU compute dispatch wrappers for Kimodo.
 *
 * Adapted from MoGE's shader_ops.js, stripped to the ops Kimodo needs:
 * - Linear projection (matmul + bias)
 * - LayerNorm
 * - Multi-head self-attention (fused QKV)
 * - SiLU activation
 * - Element-wise add
 */

import linearWGSL from '../shaders/linear.wgsl?raw';
import layernormWGSL from '../shaders/layernorm_vit.wgsl?raw';
import attentionWGSL from '../shaders/attention.wgsl?raw';
import siluWGSL from '../shaders/silu.wgsl?raw';
import reluWGSL from '../shaders/relu.wgsl?raw';
import geluWGSL from '../shaders/gelu.wgsl?raw';
import qkvSplitWGSL from '../shaders/qkv_split.wgsl?raw';
import elementwiseWGSL from '../shaders/elementwise.wgsl?raw';

import { createStorageBuffer, createEmptyBuffer } from './gpu.js';

const pipelineCache = new Map();
const uniformCache = new Map();
const MAX_WG_DIM = 65535;

function cachedUniform(device, data) {
  const bytes = new Uint8Array(data.buffer || data);
  let h = 0;
  for (let i = 0; i < bytes.length; i++) h = (h * 31 + bytes[i]) | 0;
  const key = `u_${bytes.length}_${h}`;
  if (uniformCache.has(key)) return uniformCache.get(key);
  const buf = device.createBuffer({
    size: Math.max(bytes.byteLength, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buf.getMappedRange()).set(bytes);
  buf.unmap();
  uniformCache.set(key, buf);
  return buf;
}

let dummyBiasBuf = null;
function getDummyBias(device) {
  if (!dummyBiasBuf) {
    dummyBiasBuf = createStorageBuffer(device, new Float32Array([0]));
  }
  return dummyBiasBuf;
}

function splitWorkgroups(totalWG) {
  if (totalWG <= MAX_WG_DIM) return [totalWG, 1];
  const wgX = MAX_WG_DIM;
  const wgY = Math.ceil(totalWG / MAX_WG_DIM);
  return [wgX, wgY];
}

function getOrCreatePipeline(device, key, code, entryPoint) {
  if (pipelineCache.has(key)) return pipelineCache.get(key);
  const module = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint },
  });
  pipelineCache.set(key, pipeline);
  return pipeline;
}

function ceil(a, b) { return Math.ceil(a / b); }

/**
 * Linear projection: output = input @ weight + bias
 * Weight layout: [inDim, outDim] (already transposed by convert_weights.py)
 */
export function dispatchLinear(device, encoder, inputBuf, weightBuf, biasBuf, params) {
  const { numRows, inDim, outDim } = params;
  const totalElements = numRows * outDim;
  const totalWG = ceil(totalElements, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);

  const uniformData = new Uint32Array([numRows, inDim, outDim, wgX]);
  const uniformBuf = cachedUniform(device, uniformData);

  const pipeline = getOrCreatePipeline(device, 'linear', linearWGSL, 'main');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: weightBuf } },
      { binding: 3, resource: { buffer: biasBuf || getDummyBias(device) } },
      { binding: 4, resource: { buffer: params.outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  return params.outputBuf;
}

/**
 * LayerNorm: output = (input - mean) / sqrt(var + eps) * gamma + beta
 */
export function dispatchLayerNorm(device, encoder, inputBuf, gammaBuf, betaBuf, params) {
  const { N, D, eps = 1e-5 } = params;

  const uniformArr = new ArrayBuffer(16);
  const u32View = new Uint32Array(uniformArr);
  const f32View = new Float32Array(uniformArr);
  u32View[0] = N;
  u32View[1] = D;
  f32View[2] = eps;
  const uniformBuf = cachedUniform(device, new Uint8Array(uniformArr));

  const pipeline = getOrCreatePipeline(device, 'layernorm', layernormWGSL, 'main');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: gammaBuf } },
      { binding: 3, resource: { buffer: betaBuf } },
      { binding: 4, resource: { buffer: params.outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(N);
  pass.end();

  return params.outputBuf;
}

/**
 * SiLU (in-place): x = x * sigmoid(x)
 */
export function dispatchSiLU(device, encoder, dataBuf, count) {
  const totalWG = ceil(count, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);

  const uniformData = new Uint32Array([count, wgX]);
  const uniformBuf = cachedUniform(device, uniformData);

  const pipeline = getOrCreatePipeline(device, 'silu', siluWGSL, 'main');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: dataBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();
}

/**
 * GELU (in-place): x = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
export function dispatchGELU(device, encoder, dataBuf, count) {
  const totalWG = ceil(count, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);

  const uniformData = new Uint32Array([count, wgX]);
  const uniformBuf = cachedUniform(device, uniformData);

  const pipeline = getOrCreatePipeline(device, 'gelu', geluWGSL, 'main');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: dataBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();
}

/**
 * ReLU (in-place): x = max(0, x)
 */
export function dispatchReLU(device, encoder, dataBuf, count) {
  const totalWG = ceil(count, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);

  const uniformData = new Uint32Array([count, wgX]);
  const uniformBuf = cachedUniform(device, uniformData);

  const pipeline = getOrCreatePipeline(device, 'relu', reluWGSL, 'main');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: dataBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();
}

/**
 * Element-wise add: output = a + b
 */
export function dispatchAdd(device, encoder, aBuf, bBuf, outputBuf, count) {
  const totalWG = ceil(count, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);

  const uniformData = new Uint32Array([count, wgX]);
  const uniformBuf = cachedUniform(device, uniformData);

  const pipeline = getOrCreatePipeline(device, 'add', elementwiseWGSL, 'add');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: aBuf } },
      { binding: 2, resource: { buffer: bBuf } },
      { binding: 3, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  return outputBuf;
}

/**
 * Split fused QKV [N, 3*D] into separate Q, K, V buffers [N, D] each.
 */
export function dispatchQKVSplit(device, encoder, qkvBuf, qBuf, kBuf, vBuf, N, D) {
  const total = N * D;
  const totalWG = ceil(total, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);

  const uniformData = new Uint32Array([N, D, wgX]);
  const uniformBuf = cachedUniform(device, uniformData);

  const pipeline = getOrCreatePipeline(device, 'qkv_split', qkvSplitWGSL, 'main');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: qkvBuf } },
      { binding: 2, resource: { buffer: qBuf } },
      { binding: 3, resource: { buffer: kBuf } },
      { binding: 4, resource: { buffer: vBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();
}

/**
 * Multi-head self-attention dispatch.
 * qBuf: [N, D], kBuf: [N, D], vBuf: [N, D]
 * Returns attention output in outputBuf [N, D]
 */
export function dispatchAttention(device, encoder, qBuf, kBuf, vBuf, scoresBuf, params) {
  const { N, D, numHeads, headDim } = params;
  const scale = 1.0 / Math.sqrt(headDim);

  // 1. Compute scores: Q @ K^T
  const totalScores = numHeads * N * N;
  const scoreWG = ceil(totalScores, 256);
  const [sWgX, sWgY] = splitWorkgroups(scoreWG);

  const scoreUniformArr = new ArrayBuffer(24);
  new Uint32Array(scoreUniformArr).set([N, D, numHeads, headDim, 0, sWgX]);
  new Float32Array(scoreUniformArr, 16, 1).set([scale]);
  const scoreUniform = cachedUniform(device, new Uint8Array(scoreUniformArr));

  const scorePipeline = getOrCreatePipeline(device, 'attn_scores', attentionWGSL, 'computeScores');
  const scoreGroup = device.createBindGroup({
    layout: scorePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: scoreUniform } },
      { binding: 1, resource: { buffer: qBuf } },
      { binding: 2, resource: { buffer: kBuf } },
      { binding: 3, resource: { buffer: scoresBuf } },
    ],
  });

  let pass = encoder.beginComputePass();
  pass.setPipeline(scorePipeline);
  pass.setBindGroup(0, scoreGroup);
  pass.dispatchWorkgroups(sWgX, sWgY);
  pass.end();

  // 2. Softmax
  const softmaxRows = numHeads * N;
  const smWG = ceil(softmaxRows, 256);
  const [smWgX, smWgY] = splitWorkgroups(smWG);

  const smUniform = cachedUniform(device, new Uint32Array([N, numHeads, smWgX]));
  const smPipeline = getOrCreatePipeline(device, 'attn_softmax', attentionWGSL, 'softmax');
  const smGroup = device.createBindGroup({
    layout: smPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: smUniform } },
      { binding: 1, resource: { buffer: scoresBuf } },
    ],
  });

  pass = encoder.beginComputePass();
  pass.setPipeline(smPipeline);
  pass.setBindGroup(0, smGroup);
  pass.dispatchWorkgroups(smWgX, smWgY);
  pass.end();

  // 3. Apply: output = scores @ V
  const totalApply = N * D;
  const applyWG = ceil(totalApply, 256);
  const [aWgX, aWgY] = splitWorkgroups(applyWG);

  const applyUniform = cachedUniform(device, new Uint32Array([N, D, numHeads, headDim, aWgX]));
  const applyPipeline = getOrCreatePipeline(device, 'attn_apply', attentionWGSL, 'applyAttn');
  const applyGroup = device.createBindGroup({
    layout: applyPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: applyUniform } },
      { binding: 1, resource: { buffer: scoresBuf } },
      { binding: 2, resource: { buffer: vBuf } },
      { binding: 3, resource: { buffer: params.outputBuf } },
    ],
  });

  pass = encoder.beginComputePass();
  pass.setPipeline(applyPipeline);
  pass.setBindGroup(0, applyGroup);
  pass.dispatchWorkgroups(aWgX, aWgY);
  pass.end();

  return params.outputBuf;
}
