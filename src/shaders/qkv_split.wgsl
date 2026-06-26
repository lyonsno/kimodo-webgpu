// Split fused QKV buffer [N, 3*D] into separate Q [N, D], K [N, D], V [N, D]

struct Params {
  N: u32,           // number of tokens
  D: u32,           // model dimension (e.g. 1024)
  numWorkgroupsX: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> q: array<f32>;
@group(0) @binding(3) var<storage, read_write> k: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * params.numWorkgroupsX;
  let idx = linearWG * 256u + lid.x;

  let total = params.N * params.D;
  if (idx >= total) { return; }

  let row = idx / params.D;
  let col = idx % params.D;
  let D3 = params.D * 3u;

  q[idx] = qkv[row * D3 + col];
  k[idx] = qkv[row * D3 + params.D + col];
  v[idx] = qkv[row * D3 + params.D * 2u + col];
}
