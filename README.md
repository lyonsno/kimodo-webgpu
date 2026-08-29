# Kimodo WebGPU

Run NVIDIA's [Kimodo](https://github.com/nv-tlabs/kimodo) text-to-motion diffusion model **in the browser** using WebGPU compute shaders.

Type a text prompt, get an animated skeleton. The 282M parameter diffusion transformer — all 200 forward passes per generation — runs on your GPU through WebGPU compute shaders. No CUDA, and no PyTorch in the diffusion loop.

**This is a hybrid route, and the split matters.** The browser owns diffusion, classifier-free guidance, DDIM sampling, forward kinematics, and rendering. Text encoding does *not* run in the browser: each generation makes one call to a local Llama 3 8B server for a 4096-float embedding. That server is an external prerequisite you must supply — see [Setup](#setup) before cloning expectations. Without it the app loads, reports the missing endpoint, and cannot generate.

## What it does

- **Text in → motion out.** "A person walks forward and waves" produces a 6-second, 30fps skeletal animation.
- **200 GPU compute passes per generation.** 50 DDIM diffusion steps × 2 sub-networks × 2 passes (classifier-free guidance) = 200 transformer forward passes, all running as WebGPU compute shader dispatches.
- **Client-side diffusion + FK.** The browser handles DDIM noise scheduling, diffusion denoising, TwostageDenoiser routing, classifier-free guidance, forward kinematics, and skeleton rendering. The one exception is text encoding: an external server provides a single 4096-float embedding per generation (Llama 3 8B). You must supply that server — it is not included here.
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

For comparison, the same model on PyTorch MPS takes ~12s. The WebGPU path is ~2x slower due to per-step GPU-CPU synchronization overhead, but the diffusion runs in the browser rather than on a server.

## Architecture

```
Browser (this repo)                        Server (you supply)
┌─────────────────────────────────┐       ┌──────────────────┐
│  540 MB fp16 weights (cached)   │       │ Llama 3 8B       │
│  ↓                              │  ←──  │ text encoder     │
│  DDIM loop (50 steps):          │ 4096  │ POST /embed      │
│    Root model (16-layer xfmr)   │ floats│ (one call/gen)   │
│    globalRootToLocalRoot (JS)   │       └──────────────────┘
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

Conversion needs only numpy and safetensors — no torch.

```bash
pip install safetensors numpy

# Download Kimodo SOMA-RP-v1.1 from HuggingFace
# (requires: huggingface-cli download nvidia/Kimodo-SOMA-RP-v1.1)

python tools/convert_weights.py \
  --model /path/to/Kimodo-SOMA-RP-v1.1/model.safetensors \
  --output public/kimodo.bin \
  --dtype fp16
```

### 2. Provide a text embedding endpoint

The browser cannot compute one thing: the 4096-dimension text embedding from Kimodo's LLM2Vec/Llama 3 8B encoder. `tools/embed_server.py` serves exactly that and nothing else.

```bash
# Requires a Kimodo checkout (https://github.com/nv-tlabs/kimodo) and a Python
# environment with its dependencies — including PyTorch, which Kimodo expects to
# be installed already and does not declare. The server does NOT read the SOMA
# diffusion checkpoint; it loads the McGill LLM2Vec encoder repositories, which
# are downloaded from Hugging Face on first run (~16 GB).
#
#   pip install torch "transformers==5.1.0" "peft>=0.18" hydra-core omegaconf einops
#
# Note: `pip install -e` on the Kimodo repo builds a C++ extension that does not
# compile on Apple Silicon (it hardcodes -msse4.1). Setting KIMODO_ROOT is enough
# for text embedding; the extension is not needed.
export KIMODO_ROOT=~/dev/kimodo          # default, override if elsewhere

python tools/embed_server.py --port 8098
```

The server loads the encoder (~16 GB) at startup and **exits non-zero if that fails**, so a running server means a working route. Point the app at it with the **Server URL** field in the UI (default `http://localhost:8098`).

Any server satisfying this contract works if you prefer your own:

```
POST <server-url>/embed        { "prompt": "a person walks forward and waves" }
->  200                        { "embedding": [ ...4096 finite floats... ], "dim": 4096 }
```

Only the flattened 4096-value `embedding` array is contractual; `dim` is informational.

If the endpoint is missing, unreachable, or returns anything other than 4096 finite numbers, the app fails loud — it reports the problem in the status line and stops, rather than feeding a bad embedding into diffusion and producing plausible-looking nonsense. The server applies the same check before responding.

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
# Headless smoke test (requires Chrome, npm run dev, and tools/embed_server.py)
node tools/headless_smoke.mjs

# Numerical comparison against PyTorch reference
node tools/numerical_comparison.mjs

# Visual filmstrip capture
node tools/filmstrip_smoke.mjs --prompt "a person walks forward"

# Contract tests — no model, no weights, no GPU required
python tests/test_embed_server_contract.py    # /embed boundary + bind/CORS
python tests/test_convert_weights_guard.py    # converter atomicity + guards
```

The two contract tests stub the text encoder and run in seconds. They assert
the failure paths rather than the happy path: that the server binds loopback
only, that malformed requests get stable 400s, that non-finite or wrong-length
embeddings are refused rather than served, and that an incompatible sidecar
config can never leave a rewritten binary behind.

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
