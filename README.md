# Kimodo WebGPU

Run NVIDIA's [Kimodo](https://github.com/nv-tlabs/kimodo) text-to-motion diffusion model **in the browser** using WebGPU compute shaders.

Type a text prompt, get an animated skeleton. The 282M parameter diffusion transformer runs entirely on your GPU through the browser — no Python, no CUDA, no installs.

https://github.com/user-attachments/assets/placeholder

## What it does

- **Text in → motion out.** "A person walks forward and waves" produces a 6-second, 30fps skeletal animation.
- **200 GPU compute passes per generation.** 50 DDIM diffusion steps × 2 sub-networks × 2 passes (classifier-free guidance) = 200 transformer forward passes, all running as WebGPU compute shader dispatches.
- **Client-side diffusion + FK.** The browser handles the full pipeline: DDIM noise scheduling, diffusion denoising, TwostageDenoiser routing, forward kinematics, and skeleton rendering. The server only provides one text embedding vector (4096 floats from Llama 3 8B).
- **30-joint SOMA skeleton** with full bone connectivity, animated at 30fps.

## Numerical accuracy

The WebGPU forward pass matches the PyTorch/MPS reference to **fp16 quantization precision**:

```
dim0: pytorch=-0.003261  webgpu=-0.003281  Δ=0.000020
dim1: pytorch= 0.323303  webgpu= 0.323948  Δ=0.000645
dim2: pytorch=-0.000158  webgpu=-0.000176  Δ=0.000018
dim3: pytorch= 1.412977  webgpu= 1.413085  Δ=0.000108
dim4: pytorch= 0.001225  webgpu= 0.001772  Δ=0.000547
```

Max absolute error: **0.000645** across all output dimensions through 16 transformer layers. The error comes from fp16 weight quantization, not computation bugs. Verified with fixed-seed deterministic comparison (`node tools/numerical_comparison.mjs`).

## Performance

On M4 Max (Chrome, WebGPU via Metal):

| Stage | Time |
|-------|------|
| Text embedding (server, Llama 3 8B) | ~300ms |
| DDIM sampling (50 steps, WebGPU) | ~25s |
| FK decode (JS, CPU) | ~2ms |
| **Total** | **~25s for 6 seconds of motion** |

For comparison, the same model on PyTorch MPS takes ~12s. The WebGPU path is ~2x slower due to per-step GPU-CPU synchronization overhead, but runs entirely in the browser.

## Architecture

```
Browser                                    Server
┌─────────────────────────────────┐       ┌──────────────┐
│  540 MB fp16 weights (cached)   │       │ Llama 3 8B   │
│  ↓                              │  ←──  │ text encoder  │
│  DDIM loop (50 steps):          │ 4096  │ (one call)   │
│    Root model (16-layer xfmr)   │ floats└──────────────┘
│    globalRootToLocalRoot (JS)   │
│    Body model (16-layer xfmr)   │
│    CFG guidance (JS)            │
│    DDIM update (JS)             │
│  ↓                              │
│  FK decode (JS)                 │
│  ↓                              │
│  Skeleton renderer (Canvas 2D)  │
└─────────────────────────────────┘
```

**WGSL compute shaders** (kernel layer shared with [moge-webgpu](https://github.com/lyonsno/moge-webgpu)):

| Shader | Purpose | Reused from MoGE? |
|--------|---------|-------------------|
| `linear.wgsl` | Matrix multiply + bias | Yes |
| `attention.wgsl` | Multi-head self-attention with optional key masking | Yes (extended) |
| `layernorm_vit.wgsl` | Layer normalization | Yes |
| `gelu.wgsl` | GELU activation (tanh overflow protected) | New |
| `silu.wgsl` | SiLU activation for timestep MLP | New |
| `qkv_split.wgsl` | Deinterleave fused QKV projection | New |
| `elementwise.wgsl` | Residual connections (add, scale-add) | New |

## Model details

| Property | Value |
|----------|-------|
| Model | NVIDIA Kimodo SOMA-RP-v1.1 |
| Parameters | 282M (two 16-layer TransformerEncoder sub-networks) |
| Architecture | Post-norm, 8 heads × 128 dim, GELU, 1024 hidden, 2048 FFN |
| Skeleton | SOMA30 (30 joints: hips, spine, head, arms, legs, hands, feet) |
| Output | Joint positions (3D), rotation matrices, root trajectory, foot contacts |
| Weights | 540 MB (fp16 flat binary, converted from safetensors) |
| Diffusion | DDIM, cosine beta schedule, 50–100 steps |
| Text conditioning | Classifier-free guidance (w=2.0), LLM2Vec text encoder |

## Setup

### 1. Convert weights

```bash
pip install safetensors numpy

# Download Kimodo SOMA-RP-v1.1 from HuggingFace
# (requires: huggingface-cli download nvidia/Kimodo-SOMA-RP-v1.1)

python tools/convert_weights.py \
  --model /path/to/Kimodo-SOMA-RP-v1.1/model.safetensors \
  --output public/kimodo.bin \
  --dtype fp16
```

### 2. Start the text embedding server

The only server dependency is Llama 3 8B for text encoding. From the [kaminos](https://github.com/lyonsno/kaminos) directory:

```bash
python motion-serve.py --model kimodo --port 8098
```

This loads Kimodo + Llama 3 8B (~16 GB) and exposes `/embed` for text embeddings.

### 3. Run

```bash
npm install
npm run dev
```

Open the URL, type a prompt, click Generate. Weights download on first load (~540 MB), then cached by the browser.

## Verification

| Check | Status | Tool |
|-------|--------|------|
| Forward pass vs PyTorch | ✅ Max diff 0.000645 | `node tools/numerical_comparison.mjs` |
| DDIM loop correctness | ✅ No material findings | Independent Aposkepsis review |
| Full implementation review | ✅ 21 questions, no material findings | Independent Aposkepsis review |
| FK decode review | ✅ 1 finding fixed | Independent Aposkepsis review |
| Visual output coherence | ✅ Operator confirmed | Headless smoke + filmstrip witness |
| Route receipt emission | ✅ Staged profile + artifact hashes | `@kaminos/webgpu-inference-kit` contract |

## Automated tests

```bash
# Headless smoke test (requires Chrome + running servers)
node tools/headless_smoke.mjs

# Numerical comparison against PyTorch reference
node tools/numerical_comparison.mjs

# Visual filmstrip capture
node tools/filmstrip_smoke.mjs --prompt "a person walks forward"
```

## What's next

- [ ] Performance: reduce per-step GPU-CPU sync overhead
- [ ] Client-side text embedding (quantized Llama in browser via WebLLM)
- [ ] 3D skeleton renderer (Three.js WebGPU)
- [ ] Shared kernel package with moge-webgpu (`@kaminos/webgpu-inference-kit`)
- [ ] Batch CFG (4→2 forward passes per step by batching cond/uncond)

## License

- Kimodo model weights: [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
- Kimodo source code: Apache-2.0
- This WebGPU implementation: MIT
