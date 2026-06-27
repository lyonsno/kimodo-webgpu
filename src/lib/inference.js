/**
 * inference.js — Kimodo diffusion transformer forward pass in WebGPU.
 *
 * Architecture per sub-network (body_model / root_model):
 *   1. input_linear(noisy_motion)         [S, inputDim] -> [S, 1024]
 *   2. embed_text(text_embedding)         [T, 4096]     -> [T, 1024]
 *   3. timestep_mlp(sinusoidal(t))        [1, 1024]     -> [1, 1024]
 *   4. concat [text, timestep, motion]    -> [N, 1024]  (N = T + 1 + S)
 *   5. add positional encoding
 *   6. 16x TransformerEncoderLayer (post-norm):
 *      a. self_attn(x) -> x = norm1(x + attn_out)
 *      b. ffn(x) -> x = norm2(x + ffn_out)  [GELU activation]
 *   7. extract motion portion [S, 1024]
 *   8. output_linear                      [S, 1024]     -> [S, outputDim]
 */

import { createStorageBuffer, createEmptyBuffer } from './gpu.js';
import { dispatchLinear, dispatchLayerNorm, dispatchSiLU, dispatchGELU, dispatchAdd, dispatchAttention, dispatchQKVSplit } from './shader_ops.js';

const D = 1024;
const FFN_DIM = 2048;
const NUM_HEADS = 8;
const HEAD_DIM = 128;

function sinusoidalEmbedding(timestep, dim = D) {
  const half = dim / 2;
  const emb = new Float32Array(dim);
  for (let i = 0; i < half; i++) {
    const freq = Math.exp(-Math.log(10000) * i / half);
    emb[i] = Math.sin(timestep * freq);
    emb[i + half] = Math.cos(timestep * freq);
  }
  return emb;
}

function positionalEncoding(maxLen, dim = D) {
  const pe = new Float32Array(maxLen * dim);
  for (let pos = 0; pos < maxLen; pos++) {
    for (let i = 0; i < dim; i += 2) {
      const freq = Math.exp(-Math.log(10000) * i / dim);
      pe[pos * dim + i] = Math.sin(pos * freq);
      pe[pos * dim + i + 1] = Math.cos(pos * freq);
    }
  }
  return pe;
}

/**
 * Run one transformer sub-network forward pass.
 */
/**
 * @param {GPUBuffer|null} keyMaskBuf - optional [totalSeqLen] float buffer for attention masking.
 *   0.0 = attend, -1e9 = mask out. Used for CFG unconditioned pass (mask text tokens).
 */
export async function forwardTransformer(device, weights, motionBuf, textBuf, timestep, seqLen, inputDim, outputDim, keyMaskBuf = null) {
  const textLen = 1;
  const totalSeqLen = textLen + 1 + seqLen; // text + timestep + motion
  const prefixLen = textLen + 1;
  const N = totalSeqLen;

  // Step 1: Project motion [seqLen, inputDim] -> [seqLen, D]
  let enc = device.createCommandEncoder();
  const projMotionBuf = createEmptyBuffer(device, seqLen * D * 4);
  dispatchLinear(device, enc, motionBuf, weights.inputLinear.weight, weights.inputLinear.bias, {
    numRows: seqLen, inDim: inputDim, outDim: D, outputBuf: projMotionBuf,
  });

  // Step 2: Project text [1, 4096] -> [1, D]
  const projTextBuf = createEmptyBuffer(device, D * 4);
  dispatchLinear(device, enc, textBuf, weights.embedText.weight, weights.embedText.bias, {
    numRows: 1, inDim: 4096, outDim: D, outputBuf: projTextBuf,
  });

  // Step 3: Timestep MLP — sinusoidal -> Linear -> SiLU -> Linear
  const sinEmb = sinusoidalEmbedding(timestep);
  const sinEmbBuf = createStorageBuffer(device, sinEmb);
  const tsTemp = createEmptyBuffer(device, D * 4);
  dispatchLinear(device, enc, sinEmbBuf, weights.timestepMLP.linear1.weight, weights.timestepMLP.linear1.bias, {
    numRows: 1, inDim: D, outDim: D, outputBuf: tsTemp,
  });
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();

  enc = device.createCommandEncoder();
  dispatchSiLU(device, enc, tsTemp, D);
  const tsEmbBuf = createEmptyBuffer(device, D * 4);
  dispatchLinear(device, enc, tsTemp, weights.timestepMLP.linear2.weight, weights.timestepMLP.linear2.bias, {
    numRows: 1, inDim: D, outDim: D, outputBuf: tsEmbBuf,
  });
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();

  // Step 4: Concatenate [text, timestep, motion] -> [N, D]
  const xseqBuf = createEmptyBuffer(device, N * D * 4);
  enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(projTextBuf, 0, xseqBuf, 0, D * 4);
  enc.copyBufferToBuffer(tsEmbBuf, 0, xseqBuf, D * 4, D * 4);
  enc.copyBufferToBuffer(projMotionBuf, 0, xseqBuf, prefixLen * D * 4, seqLen * D * 4);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();

  // Step 5: Add positional encoding
  const pe = positionalEncoding(N);
  const peBuf = createStorageBuffer(device, pe);
  const xseqWithPE = createEmptyBuffer(device, N * D * 4);
  enc = device.createCommandEncoder();
  dispatchAdd(device, enc, xseqBuf, peBuf, xseqWithPE, N * D);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();

  // Step 6: 16 Transformer Encoder Layers (post-norm)
  let currentBuf = xseqWithPE;

  for (let layer = 0; layer < 16; layer++) {
    const lw = weights.layers[layer];

    // --- Self-attention ---
    enc = device.createCommandEncoder();
    const qkvBuf = createEmptyBuffer(device, N * 3 * D * 4);
    dispatchLinear(device, enc, currentBuf, lw.inProjW, lw.inProjB, {
      numRows: N, inDim: D, outDim: 3 * D, outputBuf: qkvBuf,
    });

    // Split QKV
    const qBuf = createEmptyBuffer(device, N * D * 4);
    const kBuf = createEmptyBuffer(device, N * D * 4);
    const vBuf = createEmptyBuffer(device, N * D * 4);
    dispatchQKVSplit(device, enc, qkvBuf, qBuf, kBuf, vBuf, N, D);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Attention
    enc = device.createCommandEncoder();
    const scoresBuf = createEmptyBuffer(device, NUM_HEADS * N * N * 4);
    const attnOutBuf = createEmptyBuffer(device, N * D * 4);
    dispatchAttention(device, enc, qBuf, kBuf, vBuf, scoresBuf, {
      N, D, numHeads: NUM_HEADS, headDim: HEAD_DIM, outputBuf: attnOutBuf,
      maskBuf: keyMaskBuf,
    });

    // Output projection + residual + norm1
    const attnProjBuf = createEmptyBuffer(device, N * D * 4);
    dispatchLinear(device, enc, attnOutBuf, lw.outProjW, lw.outProjB, {
      numRows: N, inDim: D, outDim: D, outputBuf: attnProjBuf,
    });
    const residual1 = createEmptyBuffer(device, N * D * 4);
    dispatchAdd(device, enc, currentBuf, attnProjBuf, residual1, N * D);
    const afterAttn = createEmptyBuffer(device, N * D * 4);
    dispatchLayerNorm(device, enc, residual1, lw.norm1W, lw.norm1B, { N, D, outputBuf: afterAttn });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    // --- FFN ---
    enc = device.createCommandEncoder();
    const ffnUp = createEmptyBuffer(device, N * FFN_DIM * 4);
    dispatchLinear(device, enc, afterAttn, lw.ffn1W, lw.ffn1B, {
      numRows: N, inDim: D, outDim: FFN_DIM, outputBuf: ffnUp,
    });
    dispatchGELU(device, enc, ffnUp, N * FFN_DIM);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    enc = device.createCommandEncoder();
    const ffnDown = createEmptyBuffer(device, N * D * 4);
    dispatchLinear(device, enc, ffnUp, lw.ffn2W, lw.ffn2B, {
      numRows: N, inDim: FFN_DIM, outDim: D, outputBuf: ffnDown,
    });
    const residual2 = createEmptyBuffer(device, N * D * 4);
    dispatchAdd(device, enc, afterAttn, ffnDown, residual2, N * D);
    const afterFFN = createEmptyBuffer(device, N * D * 4);
    dispatchLayerNorm(device, enc, residual2, lw.norm2W, lw.norm2B, { N, D, outputBuf: afterFFN });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Cleanup
    qkvBuf.destroy(); qBuf.destroy(); kBuf.destroy(); vBuf.destroy();
    scoresBuf.destroy(); attnOutBuf.destroy(); attnProjBuf.destroy();
    residual1.destroy(); afterAttn.destroy();
    ffnUp.destroy(); ffnDown.destroy(); residual2.destroy();
    if (currentBuf !== xseqWithPE) currentBuf.destroy();
    currentBuf = afterFFN;
  }

  // Step 7-8: Extract motion portion and output projection
  enc = device.createCommandEncoder();
  const motionOutBuf = createEmptyBuffer(device, seqLen * D * 4);
  enc.copyBufferToBuffer(currentBuf, prefixLen * D * 4, motionOutBuf, 0, seqLen * D * 4);
  const finalOutBuf = createEmptyBuffer(device, seqLen * outputDim * 4);
  dispatchLinear(device, enc, motionOutBuf, weights.outputLinear.weight, weights.outputLinear.bias, {
    numRows: seqLen, inDim: D, outDim: outputDim, outputBuf: finalOutBuf,
  });
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();

  // Cleanup
  projMotionBuf.destroy(); projTextBuf.destroy(); sinEmbBuf.destroy();
  tsTemp.destroy(); tsEmbBuf.destroy(); xseqBuf.destroy(); peBuf.destroy();
  xseqWithPE.destroy(); motionOutBuf.destroy(); currentBuf.destroy();

  return finalOutBuf;
}

export async function readBuffer(device, buffer, numFloats) {
  const readBuf = device.createBuffer({
    size: numFloats * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buffer, 0, readBuf, 0, numFloats * 4);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readBuf.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();
  readBuf.destroy();
  return data;
}

/**
 * Single forward pass verification — runs body and root once at timestep 500.
 */
export async function singleForwardPass(device, weights, config, textEmbedding, numFrames, onProgress = null) {
  const bodyDim = config.body_input_dim;
  const bodyOutDim = config.body_output_dim;
  const rootDim = config.root_input_dim;
  const rootOutDim = config.root_output_dim;

  const bodyNoise = new Float32Array(numFrames * bodyDim);
  const rootNoise = new Float32Array(numFrames * rootDim);
  for (let i = 0; i < bodyNoise.length; i++) {
    const u1 = Math.random(), u2 = Math.random();
    bodyNoise[i] = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  }
  for (let i = 0; i < rootNoise.length; i++) {
    const u1 = Math.random(), u2 = Math.random();
    rootNoise[i] = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  }

  const bodyInputBuf = createStorageBuffer(device, bodyNoise);
  const rootInputBuf = createStorageBuffer(device, rootNoise);
  const textBuf = createStorageBuffer(device, textEmbedding);

  if (onProgress) onProgress(0, 2);
  console.log('[kimodo-webgpu] Running body model forward pass...');
  const t0 = performance.now();
  const bodyOutBuf = await forwardTransformer(device, weights.body, bodyInputBuf, textBuf, 500, numFrames, bodyDim, bodyOutDim);
  console.log(`[kimodo-webgpu] Body model: ${((performance.now() - t0) / 1000).toFixed(2)}s`);
  if (onProgress) onProgress(1, 2);

  console.log('[kimodo-webgpu] Running root model forward pass...');
  const t1 = performance.now();
  const rootOutBuf = await forwardTransformer(device, weights.root, rootInputBuf, textBuf, 500, numFrames, rootDim, rootOutDim);
  console.log(`[kimodo-webgpu] Root model: ${((performance.now() - t1) / 1000).toFixed(2)}s`);
  if (onProgress) onProgress(2, 2);

  const bodyResult = await readBuffer(device, bodyOutBuf, numFrames * bodyOutDim);
  const rootResult = await readBuffer(device, rootOutBuf, numFrames * rootOutDim);

  bodyInputBuf.destroy(); rootInputBuf.destroy(); textBuf.destroy();
  bodyOutBuf.destroy(); rootOutBuf.destroy();

  return { bodyResult, rootResult, numFrames, fps: config.fps };
}

export async function loadConfig(configUrl = '/kimodo.json') {
  return (await fetch(configUrl)).json();
}
