// GELU activation (in-place): x = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// This is the exact GELU used by PyTorch.

struct Params {
  count: u32,
  numWorkgroupsX: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

const SQRT_2_OVER_PI: f32 = 0.7978845608;  // sqrt(2/pi)
const COEFF: f32 = 0.044715;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * params.numWorkgroupsX;
  let idx = linearWG * 256u + lid.x;
  if (idx >= params.count) { return; }

  let x = data[idx];
  let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
  // Clamp tanh argument to avoid exp overflow in tanh implementation
  let clamped = clamp(inner, -10.0, 10.0);
  data[idx] = 0.5 * x * (1.0 + tanh(clamped));
}
