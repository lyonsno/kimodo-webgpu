// Element-wise operations for residual connections and noise scheduling.

struct Params {
  count: u32,
  numWorkgroupsX: u32,
}

// --- Add: output = a + b ---
@group(0) @binding(0) var<uniform> addParams: Params;
@group(0) @binding(1) var<storage, read> addA: array<f32>;
@group(0) @binding(2) var<storage, read> addB: array<f32>;
@group(0) @binding(3) var<storage, read_write> addOut: array<f32>;

@compute @workgroup_size(256)
fn add(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * addParams.numWorkgroupsX;
  let idx = linearWG * 256u + lid.x;
  if (idx >= addParams.count) { return; }
  addOut[idx] = addA[idx] + addB[idx];
}

// --- Scale-add: output = a * scale + b ---
struct ScaleAddParams {
  count: u32,
  scale: f32,
  numWorkgroupsX: u32,
}

@group(0) @binding(0) var<uniform> scaleAddParams: ScaleAddParams;
@group(0) @binding(1) var<storage, read> scaleA: array<f32>;
@group(0) @binding(2) var<storage, read> scaleB: array<f32>;
@group(0) @binding(3) var<storage, read_write> scaleOut: array<f32>;

@compute @workgroup_size(256)
fn scaleAdd(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * scaleAddParams.numWorkgroupsX;
  let idx = linearWG * 256u + lid.x;
  if (idx >= scaleAddParams.count) { return; }
  scaleOut[idx] = scaleA[idx] * scaleAddParams.scale + scaleB[idx];
}
