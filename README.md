# Kimodo WebGPU

Run NVIDIA's [Kimodo](https://github.com/nv-tlabs/kimodo) text-to-motion diffusion model in the browser using WebGPU compute shaders.

282M parameter diffusion transformer generating 77-joint skeletal animation from text prompts, with the heavy compute running entirely on the client GPU.

## Architecture

**Hybrid client-server:**
- **Browser (WebGPU):** Loads 540 MB of model weights (fp16), runs the 16-layer transformer encoder forward pass via WGSL compute shaders
- **Server (Python/PyTorch):** Provides text embedding (Llama 3 8B → 4096-dim vector) and motion feature decoding (FK to joint positions)

**WGSL compute shaders** (adapted from [moge-webgpu](https://github.com/user/moge-webgpu)):
- `linear.wgsl` — matrix multiply + bias (Q/K/V projections, FFN layers)
- `attention.wgsl` — multi-head self-attention (score computation, softmax, value aggregation)
- `layernorm_vit.wgsl` — layer normalization
- `gelu.wgsl` — GELU activation with tanh overflow protection
- `silu.wgsl` — SiLU activation for timestep MLP
- `qkv_split.wgsl` — deinterleave fused QKV projection
- `elementwise.wgsl` — residual connections

## Model Details

- **Kimodo SOMA-RP-v1.1** — 282M params, 8 heads × 128 dim, 16 transformer encoder layers
- **Post-norm** architecture (not pre-norm)
- **Two sub-networks:** body model (737→364 features) + root model (738→5 features)
- **DDIM sampling** with 100 diffusion steps
- **SOMA77 skeleton** — 77 joints with full rotation matrices

## Setup

### 1. Convert weights

```bash
# Requires: pip install safetensors numpy
python tools/convert_weights.py \
  --model /path/to/Kimodo-SOMA-RP-v1.1/model.safetensors \
  --output public/kimodo.bin \
  --dtype fp16
```

### 2. Start the text embedding + decode server

```bash
# From the kaminos directory (requires Kimodo + Llama 3 set up)
python motion-serve.py --model kimodo --port 8098
```

### 3. Run the dev server

```bash
npm install
npm run dev
# Opens http://localhost:5175
```

### 4. Generate

Type a prompt, click Generate. The browser downloads 540 MB of weights on first load, then runs the diffusion model on your GPU.

## Status

- [x] Weight conversion (safetensors → flat binary fp16)
- [x] Weight loader with streaming progress
- [x] All WGSL compute shaders (linear, attention, layernorm, GELU, SiLU, QKV split, elementwise)
- [x] Full 16-layer transformer forward pass (both body + root models)
- [x] Single forward pass verification against PyTorch reference
- [x] Server-side text embedding (`/embed`) and FK decoding (`/decode`)
- [x] 2D skeleton animation renderer
- [x] Full DDIM generation via MPS server with browser skeleton rendering
- [ ] Client-side DDIM sampling loop (requires motion_rep port)
- [ ] Client-side FK decoding (remove server dependency for decode)
- [ ] 3D skeleton renderer (Three.js or raw WebGPU)
- [ ] Shared kernel library with moge-webgpu

## License

Kimodo model weights: [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
Kimodo code: Apache-2.0
This WebGPU implementation: MIT
