// ReLU activation (in-place): x = max(0, x)

struct Params {
  count: u32,
  numWorkgroupsX: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * params.numWorkgroupsX;
  let idx = linearWG * 256u + lid.x;
  if (idx >= params.count) { return; }
  data[idx] = max(0.0, data[idx]);
}
