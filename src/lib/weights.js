/**
 * weights.js — Load Kimodo weights from flat binary format.
 *
 * Binary format (from convert_weights.py):
 *   Header: 4 (magic) + 4 (version) + 4 (num_tensors) + 4 (header_size) = 16 bytes
 *   Tensor table: num_tensors x 96 bytes each
 *     64 bytes: name (null-padded ASCII)
 *     4 bytes: dtype (0=fp32, 1=fp16)
 *     4 bytes: ndim
 *     16 bytes: shape (4 x u32)
 *     4 bytes: offset
 *     4 bytes: size
 *   Weight data: packed tensors
 */

import { createStorageBuffer } from './gpu.js';

const MAGIC = 0x444D494B; // "KIMD" in little-endian
const ENTRY_SIZE = 96;

function parseHeader(buffer) {
  const view = new DataView(buffer);
  const magic = view.getUint32(0, true);
  if (magic !== MAGIC) {
    throw new Error(`Invalid weight file magic: 0x${magic.toString(16)} (expected KIMD)`);
  }
  const version = view.getUint32(4, true);
  if (version !== 1) throw new Error(`Unsupported version: ${version}`);

  const numTensors = view.getUint32(8, true);
  const headerSize = view.getUint32(12, true);

  const tensors = new Map();
  for (let i = 0; i < numTensors; i++) {
    const off = 16 + i * ENTRY_SIZE;
    const nameBytes = new Uint8Array(buffer, off, 64);
    let nameEnd = nameBytes.indexOf(0);
    if (nameEnd === -1) nameEnd = 64;
    const name = new TextDecoder().decode(nameBytes.slice(0, nameEnd));

    const dtype = view.getUint32(off + 64, true);
    const ndim = view.getUint32(off + 68, true);
    const shape = [];
    for (let d = 0; d < ndim; d++) shape.push(view.getUint32(off + 72 + d * 4, true));
    const offset = view.getUint32(off + 88, true);
    const size = view.getUint32(off + 92, true);

    tensors.set(name, { dtype, shape, offset, size });
  }
  return { tensors, headerSize, numTensors };
}

function fp16ToFp32(h) {
  const sign = (h >> 15) & 1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) {
    if (mant === 0) return sign ? -0.0 : 0.0;
    return (sign ? -1 : 1) * (mant / 1024.0) * Math.pow(2, -14);
  }
  if (exp === 31) return mant === 0 ? (sign ? -Infinity : Infinity) : NaN;
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024.0);
}

function extractGPUBuffer(device, buffer, info) {
  const { dtype, offset, size } = info;
  if (dtype === 0) {
    return createStorageBuffer(device, new Float32Array(buffer, offset, size / 4));
  }
  // fp16 -> fp32
  const fp16 = new Uint16Array(buffer, offset, size / 2);
  const fp32 = new Float32Array(fp16.length);
  for (let i = 0; i < fp16.length; i++) fp32[i] = fp16ToFp32(fp16[i]);
  return createStorageBuffer(device, fp32);
}

/**
 * Load Kimodo weights and organize into model structure.
 *
 * Returns:
 *   { body: TransformerWeights, root: TransformerWeights }
 *
 * TransformerWeights:
 *   { inputLinear, embedText, timestepMLP, outputLinear, headingLinear, layers[] }
 *
 * Each layer:
 *   { norm1W, norm1B, norm2W, norm2B, inProjW, inProjB, outProjW, outProjB, ffn1W, ffn1B, ffn2W, ffn2B }
 */
export async function loadWeights(device, buffer) {
  const { tensors, numTensors } = parseHeader(buffer);
  console.log(`[weights] Parsed ${numTensors} tensors`);

  const get = (name) => {
    const info = tensors.get(name);
    if (!info) throw new Error(`Missing weight: ${name}`);
    return extractGPUBuffer(device, buffer, info);
  };

  function loadTransformer(prefix) {
    // Projection layers
    const inputLinear = { weight: get(`${prefix}.input_linear.weight`), bias: get(`${prefix}.input_linear.bias`) };
    const embedText = { weight: get(`${prefix}.embed_text.weight`), bias: get(`${prefix}.embed_text.bias`) };
    const outputLinear = { weight: get(`${prefix}.output_linear.weight`), bias: get(`${prefix}.output_linear.bias`) };
    const headingLinear = { weight: get(`${prefix}.linear_first_heading_angle.weight`), bias: get(`${prefix}.linear_first_heading_angle.bias`) };

    // Timestep MLP: Linear -> SiLU -> Linear
    const timestepMLP = {
      linear1: { weight: get(`${prefix}.embed_timestep.time_embed.0.weight`), bias: get(`${prefix}.embed_timestep.time_embed.0.bias`) },
      linear2: { weight: get(`${prefix}.embed_timestep.time_embed.2.weight`), bias: get(`${prefix}.embed_timestep.time_embed.2.bias`) },
    };

    // Transformer layers (0-15)
    const layers = [];
    for (let i = 0; i < 16; i++) {
      const lp = `${prefix}.seqTransEncoder.layers.${i}`;
      layers.push({
        norm1W: get(`${lp}.norm1.weight`),
        norm1B: get(`${lp}.norm1.bias`),
        norm2W: get(`${lp}.norm2.weight`),
        norm2B: get(`${lp}.norm2.bias`),
        inProjW: get(`${lp}.self_attn.in_proj_weight`),
        inProjB: get(`${lp}.self_attn.in_proj_bias`),
        outProjW: get(`${lp}.self_attn.out_proj.weight`),
        outProjB: get(`${lp}.self_attn.out_proj.bias`),
        ffn1W: get(`${lp}.linear1.weight`),
        ffn1B: get(`${lp}.linear1.bias`),
        ffn2W: get(`${lp}.linear2.weight`),
        ffn2B: get(`${lp}.linear2.bias`),
      });
    }

    return { inputLinear, embedText, outputLinear, headingLinear, timestepMLP, layers };
  }

  const body = loadTransformer('body_model');
  const root = loadTransformer('root_model');

  console.log(`[weights] Loaded body_model (16 layers) + root_model (16 layers)`);
  return { body, root };
}
