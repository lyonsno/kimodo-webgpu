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
  // Match PyTorch's PositionalEncoding exactly:
  //   div_term = pow(10000, -arange(0, d, 2) / d)
  //   pe[0::2] = sin(position * div_term)
  //   pe[1::2] = cos(position * div_term)
  const emb = new Float32Array(dim);
  for (let i = 0; i < dim; i += 2) {
    const freq = Math.pow(10000.0, -i / dim);
    emb[i] = Math.sin(timestep * freq);
    emb[i + 1] = Math.cos(timestep * freq);
  }
  return emb;
}

function positionalEncoding(maxLen, dim = D) {
  // Match PyTorch's PositionalEncoding: pow(10000, -arange(0,d,2)/d)
  const pe = new Float32Array(maxLen * dim);
  for (let pos = 0; pos < maxLen; pos++) {
    for (let i = 0; i < dim; i += 2) {
      const freq = Math.pow(10000.0, -i / dim);
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
export async function forwardTransformer(device, weights, motionBuf, textBuf, timestep, seqLen, inputDim, outputDim, keyMaskBuf = null, options = {}) {
  // The default remains one 16-layer forward pass = one command encoder =
  // one duty. An explicit 4-layer schedule cuts that same ordered command
  // stream into four duties. Queue order carries every dependency across
  // chunks; no math, buffer, layer, or readback is added or removed.
  //
  // Memory: per-layer scratch is a FIXED set reused across all 16 layers —
  // serial queue order makes reuse safe, so peak per-pass scratch is one
  // layer's working set plus the ping-pong pair (~tens of MB at the maximum
  // supported duration), not sixteen layers' worth (~740 MB at 18s, the
  // regression the previous revision shipped while claiming "same peak").
  //
  // Ownership: every per-call buffer is registered and destroyed exactly
  // once in the finally path, whatever the controller does — a rejected
  // submission (duplicate duty, queue failure, cancellation) must not leak
  // the pass's allocations. The returned output escapes cleanup only after
  // successful submission.
  const submissions = options.submissions ?? null;
  if (submissions && !options.dutyId) {
    throw new Error('a bounded submissions context requires a caller-owned unique dutyId');
  }
  const layersPerDuty = options.layersPerDuty ?? 16;
  if (layersPerDuty !== 4 && layersPerDuty !== 16) {
    throw new RangeError('layersPerDuty must be exactly 4 or 16');
  }
  const chunkCount = 16 / layersPerDuty;
  if (chunkCount > 1 && !options.dutyId) {
    throw new Error('chunked transformer admission requires a caller-owned unique dutyId');
  }
  if (options.afterChunk != null && typeof options.afterChunk !== 'function') {
    throw new TypeError('afterChunk must be a function');
  }
  const numTextTokens = 50; // backbone pads to this fixed size
  const totalSeqLen = numTextTokens + 1 + 1 + seqLen; // padded_text(50) + timestep(1) + heading(1) + motion
  const prefixLen = numTextTokens + 1 + 1; // text + timestep + heading
  const N = totalSeqLen;

  const transient = [];
  const own = (buf) => { transient.push(buf); return buf; };
  let finalOutBuf = null;
  let returned = false;

  try {
    const firstEncodeStartedAtMs = performance.now();
    let enc = device.createCommandEncoder();

    // Step 1: Project motion [seqLen, inputDim] -> [seqLen, D]
    const projMotionBuf = own(createEmptyBuffer(device, seqLen * D * 4));
    dispatchLinear(device, enc, motionBuf, weights.inputLinear.weight, weights.inputLinear.bias, {
      numRows: seqLen, inDim: inputDim, outDim: D, outputBuf: projMotionBuf,
    });

    // Step 2: Padded text [numTextTokens, 4096] — first token real, rest zeros;
    // ALL tokens go through embed_text so bias lands on padding too.
    const paddedTextInput = own(createEmptyBuffer(device, numTextTokens * 4096 * 4)); // zero-init
    enc.copyBufferToBuffer(textBuf, 0, paddedTextInput, 0, 4096 * 4);
    const projTextBuf = own(createEmptyBuffer(device, numTextTokens * D * 4));
    dispatchLinear(device, enc, paddedTextInput, weights.embedText.weight, weights.embedText.bias, {
      numRows: numTextTokens, inDim: 4096, outDim: D, outputBuf: projTextBuf,
    });

    // Step 3: Timestep MLP — sinusoidal -> Linear -> SiLU -> Linear
    const sinEmbBuf = own(createStorageBuffer(device, sinusoidalEmbedding(timestep)));
    const tsTemp = own(createEmptyBuffer(device, D * 4));
    dispatchLinear(device, enc, sinEmbBuf, weights.timestepMLP.linear1.weight, weights.timestepMLP.linear1.bias, {
      numRows: 1, inDim: D, outDim: D, outputBuf: tsTemp,
    });
    dispatchSiLU(device, enc, tsTemp, D);
    const tsEmbBuf = own(createEmptyBuffer(device, D * 4));
    dispatchLinear(device, enc, tsTemp, weights.timestepMLP.linear2.weight, weights.timestepMLP.linear2.bias, {
      numRows: 1, inDim: D, outDim: D, outputBuf: tsEmbBuf,
    });

    // Step 3b: Heading angle token — cos(0)/sin(0) projected to D
    const headingBuf = own(createStorageBuffer(device, new Float32Array([Math.cos(0), Math.sin(0)])));
    const headingProjBuf = own(createEmptyBuffer(device, D * 4));
    dispatchLinear(device, enc, headingBuf, weights.headingLinear.weight, weights.headingLinear.bias, {
      numRows: 1, inDim: 2, outDim: D, outputBuf: headingProjBuf,
    });

    // Step 4: Concatenate [paddedText(50), timestep(1), heading(1), motion] -> [N, D]
    const xseqBuf = own(createEmptyBuffer(device, N * D * 4));
    enc.copyBufferToBuffer(projTextBuf, 0, xseqBuf, 0, numTextTokens * D * 4);
    enc.copyBufferToBuffer(tsEmbBuf, 0, xseqBuf, numTextTokens * D * 4, D * 4);
    enc.copyBufferToBuffer(headingProjBuf, 0, xseqBuf, (numTextTokens + 1) * D * 4, D * 4);
    enc.copyBufferToBuffer(projMotionBuf, 0, xseqBuf, prefixLen * D * 4, seqLen * D * 4);

    // Step 5: Add positional encoding
    const peBuf = own(createStorageBuffer(device, positionalEncoding(N)));
    const xseqWithPE = own(createEmptyBuffer(device, N * D * 4));
    dispatchAdd(device, enc, xseqBuf, peBuf, xseqWithPE, N * D);

    // Step 6: 16 Transformer Encoder Layers (post-norm), fixed reused scratch.
    // No dispatch reads and writes the same buffer; layer output ping-pongs
    // between two dedicated buffers so layer i+1's input is never its output.
    const scratch = {
      qkv: own(createEmptyBuffer(device, N * 3 * D * 4)),
      q: own(createEmptyBuffer(device, N * D * 4)),
      k: own(createEmptyBuffer(device, N * D * 4)),
      v: own(createEmptyBuffer(device, N * D * 4)),
      scores: own(createEmptyBuffer(device, NUM_HEADS * N * N * 4)),
      attnOut: own(createEmptyBuffer(device, N * D * 4)),
      attnProj: own(createEmptyBuffer(device, N * D * 4)),
      residual1: own(createEmptyBuffer(device, N * D * 4)),
      afterAttn: own(createEmptyBuffer(device, N * D * 4)),
      ffnUp: own(createEmptyBuffer(device, N * FFN_DIM * 4)),
      ffnDown: own(createEmptyBuffer(device, N * D * 4)),
      residual2: own(createEmptyBuffer(device, N * D * 4)),
      pingA: own(createEmptyBuffer(device, N * D * 4)),
      pingB: own(createEmptyBuffer(device, N * D * 4)),
    };
    let currentBuf = xseqWithPE;

    for (let chunkOffset = 0; chunkOffset < chunkCount; chunkOffset++) {
      const chunkIndex = chunkOffset + 1;
      const layerStart = chunkOffset * layersPerDuty;
      const layerEnd = layerStart + layersPerDuty;
      const dutyId = chunkCount === 1 ? options.dutyId : `${options.dutyId}-c${chunkIndex}`;
      const pass = options.pass ?? options.timing?.pass ?? null;
      const timing = options.passTimings
        ? { dutyId, pass, step: options.step, numSteps: options.numSteps, chunkIndex, chunkCount, layerStart, layerEnd }
        : (chunkCount === 1 ? options.timing ?? null : null);
      if (options.passTimings) options.passTimings.push(timing); // partial rows survive failure
      if (timing) {
        timing.dutyId = dutyId;
        timing.pass = pass;
        timing.step = options.step;
        timing.numSteps = options.numSteps;
        timing.chunkIndex = chunkIndex;
        timing.chunkCount = chunkCount;
        timing.layerStart = layerStart;
        timing.layerEnd = layerEnd;
        timing.encodeStartedAtMs = chunkOffset === 0 ? firstEncodeStartedAtMs : performance.now();
      }
      if (chunkOffset > 0) enc = device.createCommandEncoder();

      for (let layer = layerStart; layer < layerEnd; layer++) {
        const lw = weights.layers[layer];
        const layerOut = (layer % 2 === 0) ? scratch.pingA : scratch.pingB;

        // --- Self-attention ---
        dispatchLinear(device, enc, currentBuf, lw.inProjW, lw.inProjB, {
          numRows: N, inDim: D, outDim: 3 * D, outputBuf: scratch.qkv,
        });
        dispatchQKVSplit(device, enc, scratch.qkv, scratch.q, scratch.k, scratch.v, N, D);
        dispatchAttention(device, enc, scratch.q, scratch.k, scratch.v, scratch.scores, {
          N, D, numHeads: NUM_HEADS, headDim: HEAD_DIM, outputBuf: scratch.attnOut,
          maskBuf: keyMaskBuf,
        });
        dispatchLinear(device, enc, scratch.attnOut, lw.outProjW, lw.outProjB, {
          numRows: N, inDim: D, outDim: D, outputBuf: scratch.attnProj,
        });
        dispatchAdd(device, enc, currentBuf, scratch.attnProj, scratch.residual1, N * D);
        dispatchLayerNorm(device, enc, scratch.residual1, lw.norm1W, lw.norm1B, { N, D, outputBuf: scratch.afterAttn });

        // --- FFN ---
        dispatchLinear(device, enc, scratch.afterAttn, lw.ffn1W, lw.ffn1B, {
          numRows: N, inDim: D, outDim: FFN_DIM, outputBuf: scratch.ffnUp,
        });
        dispatchGELU(device, enc, scratch.ffnUp, N * FFN_DIM);
        dispatchLinear(device, enc, scratch.ffnUp, lw.ffn2W, lw.ffn2B, {
          numRows: N, inDim: FFN_DIM, outDim: D, outputBuf: scratch.ffnDown,
        });
        dispatchAdd(device, enc, scratch.afterAttn, scratch.ffnDown, scratch.residual2, N * D);
        dispatchLayerNorm(device, enc, scratch.residual2, lw.norm2W, lw.norm2B, { N, D, outputBuf: layerOut });

        currentBuf = layerOut;
      }

      if (chunkIndex === chunkCount) {
        // Step 7-8: Extract motion portion and output projection in the final chunk.
        const motionOutBuf = own(createEmptyBuffer(device, seqLen * D * 4));
        enc.copyBufferToBuffer(currentBuf, prefixLen * D * 4, motionOutBuf, 0, seqLen * D * 4);
        finalOutBuf = createEmptyBuffer(device, seqLen * outputDim * 4);
        dispatchLinear(device, enc, motionOutBuf, weights.outputLinear.weight, weights.outputLinear.bias, {
          numRows: seqLen, inDim: D, outDim: outputDim, outputBuf: finalOutBuf,
        });
      }

      const commandBuffer = enc.finish();
      if (timing) timing.encodeEndedAtMs = performance.now();
      if (submissions) {
        await submissions.submitDuty({ dutyId, commandBuffers: [commandBuffer] });
      } else {
        device.queue.submit([commandBuffer]);
      }
      if (timing) timing.admittedAtMs = performance.now();
      const boundary = {
        dutyId, pass, step: options.step, numSteps: options.numSteps,
        chunkIndex, chunkCount, layerStart, layerEnd,
      };
      if (options.afterChunk) await options.afterChunk(boundary);
      if (timing) timing.boundaryEndedAtMs = performance.now();
    }

    returned = true;
    return finalOutBuf;
  } finally {
    // Exactly-once cleanup on every path. After successful submission,
    // destroy() is a deferred free (WebGPU retains until submitted work
    // completes); on a failed/rejected path the never-submitted buffers,
    // including the would-be output, are reclaimed immediately.
    for (const buf of transient) buf.destroy();
    if (!returned && finalOutBuf) finalOutBuf.destroy();
  }
}

export async function readBuffer(device, buffer, numFloats) {
  const readBuf = device.createBuffer({
    size: numFloats * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  try {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, readBuf, 0, numFloats * 4);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readBuf.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();
    return data;
  } finally {
    // Staging is reclaimed on every path, including a rejected map (r3).
    readBuf.destroy();
  }
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
