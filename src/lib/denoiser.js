/**
 * denoiser.js — TwostageDenoiser + CFG logic in JS for WebGPU.
 *
 * Key insight: CFG wraps the ENTIRE TwostageDenoiser, not individual sub-networks.
 * PyTorch order: CFG(TwostageDenoiser(x)) — not TwostageDenoiser_root(x) → CFG → body
 *
 * So we run the full root→local→body pipeline for BOTH conditioned and unconditioned
 * passes, then apply CFG to the combined [N, 369] output.
 */

import { createStorageBuffer, createEmptyBuffer } from './gpu.js';
import { forwardTransformer, readBuffer } from './inference.js';

export async function loadMotionRepStats(url = '/motion_rep_stats.json') {
  return (await fetch(url)).json();
}

/**
 * global_root_to_local_root — pure JS math.
 */
export function globalRootToLocalRoot(rootFeatures, stats) {
  const N = rootFeatures.length;
  const fps = stats.fps;
  const gMean = stats.global_root_mean;
  const gStd = stats.global_root_std;

  // Unnormalize
  const unnormed = rootFeatures.map(r => r.map((v, i) => v * gStd[i] + gMean[i]));
  const rootPos = unnormed.map(r => [r[0], r[1], r[2]]);
  const headingCos = unnormed.map(r => r[3]);
  const headingSin = unnormed.map(r => r[4]);
  const headingAngle = headingCos.map((c, i) => Math.atan2(headingSin[i], c));

  // Angular velocity
  const rotVel = new Array(N);
  for (let i = 0; i < N - 1; i++) {
    let diff = headingAngle[i + 1] - headingAngle[i];
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;
    rotVel[i] = diff * fps;
  }
  rotVel[N - 1] = N > 1 ? rotVel[N - 2] : 0;

  // Position velocity
  const velX = new Array(N), velZ = new Array(N);
  for (let i = 0; i < N - 1; i++) {
    velX[i] = (rootPos[i + 1][0] - rootPos[i][0]) * fps;
    velZ[i] = (rootPos[i + 1][2] - rootPos[i][2]) * fps;
  }
  velX[N - 1] = N > 1 ? velX[N - 2] : 0;
  velZ[N - 1] = N > 1 ? velZ[N - 2] : 0;

  const globalY = rootPos.map(p => p[1]);
  const lMean = stats.local_root_mean;
  const lStd = stats.local_root_std;

  return Array.from({ length: N }, (_, i) => [
    (rotVel[i] - lMean[0]) / lStd[0],
    (velX[i] - lMean[1]) / lStd[1],
    (velZ[i] - lMean[2]) / lStd[2],
    (globalY[i] - lMean[3]) / lStd[3],
  ]);
}

/**
 * Run the full TwostageDenoiser pipeline: root → local conversion → body.
 * Returns [N, 369] as nested JS arrays.
 */
async function runTwoStage(device, weights, motion, textBuf, timestep, N, stats) {
  const motionDim = 369;

  // --- Root model: input = [motion(369), zeros(369)] = [N, 738] ---
  const rootInputDim = motionDim * 2;
  const rootInput = new Float32Array(N * rootInputDim);
  for (let f = 0; f < N; f++) {
    for (let d = 0; d < motionDim; d++) {
      rootInput[f * rootInputDim + d] = motion[f][d];
    }
  }

  const rootInputBuf = createStorageBuffer(device, rootInput);
  const rootOutBuf = await forwardTransformer(device, weights.root, rootInputBuf, textBuf, timestep, N, rootInputDim, 5);
  const rootPred = await readBuffer(device, rootOutBuf, N * 5);
  rootInputBuf.destroy();
  rootOutBuf.destroy();

  // Convert root prediction to local root
  const rootPred2D = [];
  for (let f = 0; f < N; f++) {
    rootPred2D.push(Array.from(rootPred.slice(f * 5, (f + 1) * 5)));
  }
  const localRoot = globalRootToLocalRoot(rootPred2D, stats);

  // --- Body model: input = [localRoot(4), body_x(364), zeros(369)] = [N, 737] ---
  const bodyInputDim = 4 + 364 + motionDim; // 737
  const bodyInput = new Float32Array(N * bodyInputDim);
  for (let f = 0; f < N; f++) {
    const offset = f * bodyInputDim;
    for (let d = 0; d < 4; d++) bodyInput[offset + d] = localRoot[f][d];
    for (let d = 0; d < 364; d++) bodyInput[offset + 4 + d] = motion[f][5 + d];
    // remaining 369 zeros from Float32Array init
  }

  const bodyInputBuf = createStorageBuffer(device, bodyInput);
  const bodyOutBuf = await forwardTransformer(device, weights.body, bodyInputBuf, textBuf, timestep, N, bodyInputDim, 364);
  const bodyPred = await readBuffer(device, bodyOutBuf, N * 364);
  bodyInputBuf.destroy();
  bodyOutBuf.destroy();

  // Combine: [root_pred(5), body_pred(364)] = 369
  const output = [];
  for (let f = 0; f < N; f++) {
    const frame = new Array(motionDim);
    for (let d = 0; d < 5; d++) frame[d] = rootPred[f * 5 + d];
    for (let d = 0; d < 364; d++) frame[5 + d] = bodyPred[f * 364 + d];
    output.push(frame);
  }
  return output;
}

/**
 * Run one complete denoising step with CFG.
 *
 * Runs the full TwostageDenoiser for conditioned and unconditioned passes,
 * then applies classifier-free guidance.
 */
export async function denoiseStepWebGPU(device, weights, textEmbedding, motion, timestep, stats) {
  const N = motion.length;
  const motionDim = 369;

  const textArr = textEmbedding instanceof Float32Array ? textEmbedding : new Float32Array(textEmbedding);
  const textBuf = createStorageBuffer(device, textArr);
  const zeroTextBuf = createStorageBuffer(device, new Float32Array(textArr.length));

  // Conditioned pass: real text
  const condOutput = await runTwoStage(device, weights, motion, textBuf, timestep, N, stats);

  // Unconditioned pass: zeroed text
  const uncondOutput = await runTwoStage(device, weights, motion, zeroTextBuf, timestep, N, stats);

  textBuf.destroy();
  zeroTextBuf.destroy();

  // CFG: out = uncond + w * (cond - uncond), w=2.0
  const cfgWeight = 2.0;
  const prediction = [];
  for (let f = 0; f < N; f++) {
    const frame = new Array(motionDim);
    for (let d = 0; d < motionDim; d++) {
      frame[d] = uncondOutput[f][d] + cfgWeight * (condOutput[f][d] - uncondOutput[f][d]);
    }
    prediction.push(frame);
  }

  return prediction;
}
