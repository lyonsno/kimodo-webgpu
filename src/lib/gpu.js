/**
 * WebGPU initialization and device management.
 *
 * Device acquisition goes through @kaminos/webgpu-inference-kit: every
 * adapter limit is carried over without silent downcapping (the hand-rolled
 * predecessor forwarded six hand-picked limits), and timestamp-query is
 * negotiated ('prefer': requested when the adapter has it, cleanly absent
 * when it doesn't) — the timing authority the adaptive command-duty planner
 * can consume.
 */

import { requestBrowserWebGpuDevice } from '@kaminos/webgpu-inference-kit';

export async function initGPU() {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported in this browser. Try Chrome 113+ or Edge 113+.');
  }

  let acquired;
  try {
    acquired = await requestBrowserWebGpuDevice(navigator.gpu, {
      adapterOptions: { powerPreference: 'high-performance' },
      timestampQuery: 'prefer',
      label: 'kimodo-webgpu',
    });
  } catch (err) {
    if (/adapter unavailable/i.test(err?.message ?? '')) {
      throw new Error('No WebGPU adapter found. Your GPU may not support WebGPU.');
    }
    throw err;
  }
  const { adapter, device, backendIdentity } = acquired;

  device.lost.then((info) => {
    console.error('WebGPU device lost:', info.message);
    if (info.reason !== 'destroyed') {
      // Could attempt recovery here
    }
  });

  return { adapter, device, backendIdentity };
}

/**
 * Create a storage buffer initialized with data.
 */
export function createStorageBuffer(device, data, usage = 0) {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | usage,
    mappedAtCreation: true,
  });
  new (data.constructor)(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

/**
 * Create an empty storage buffer.
 */
export function createEmptyBuffer(device, size, usage = 0) {
  return device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | usage,
    mappedAtCreation: false,
  });
}

/**
 * Read back buffer contents to CPU.
 */
export async function readBuffer(device, buffer, size) {
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return result;
}
