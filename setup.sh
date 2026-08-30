#!/usr/bin/env bash
# One-shot local setup for kimodo-webgpu.
#
# Installs everything the route needs into ./.setup/ (gitignored):
#   - a Python venv with the text-embedding server's dependencies
#   - a Kimodo checkout (the server imports its LLM2Vec encoder)
#   - the converted 540 MB weight binary at public/kimodo.bin
#   - node_modules for the app
#
# Idempotent: completed phases are skipped on re-run. Every failure names its
# phase. Tested on Apple Silicon macOS; other platforms may need a different
# torch install.
#
# After this succeeds:  ./run_local.sh
set -euo pipefail
cd "$(dirname "$0")"

SETUP_DIR=".setup"
VENV="$SETUP_DIR/venv"
KIMODO_DIR="$SETUP_DIR/kimodo"
PY="$VENV/bin/python"

phase() { printf '\n\033[1;33m[setup] %s\033[0m\n' "$1"; }
die()   { printf '\033[1;31m[setup] FAILED during: %s\033[0m\n%s\n' "$1" "${2:-}" >&2; exit 1; }

phase "checking prerequisites"
command -v git  >/dev/null || die "prerequisites" "git not found"
command -v npm  >/dev/null || die "prerequisites" "npm not found (install Node.js)"
command -v python3 >/dev/null || die "prerequisites" "python3 not found"
mkdir -p "$SETUP_DIR"

phase "python venv + server dependencies (torch is large; first run takes a few minutes)"
if [ ! -x "$PY" ]; then
  if command -v uv >/dev/null; then
    uv venv "$VENV" || die "venv creation (uv)"
  else
    python3 -m venv "$VENV" || die "venv creation"
  fi
fi
DEPS=(torch "transformers==5.1.0" "peft>=0.18" hydra-core omegaconf einops
      numpy scipy tqdm pydantic filelock packaging safetensors huggingface_hub)
if ! "$PY" -c "import torch, transformers, peft, hydra, omegaconf, einops, safetensors, huggingface_hub" 2>/dev/null; then
  if command -v uv >/dev/null; then
    uv pip install --python "$PY" "${DEPS[@]}" || die "server dependency install (uv)"
  else
    "$PY" -m pip install --upgrade pip >/dev/null
    "$PY" -m pip install "${DEPS[@]}" || die "server dependency install (pip)"
  fi
fi

phase "kimodo checkout (LLM2Vec encoder source)"
# Note: we do NOT `pip install -e` Kimodo — its C++ extension hardcodes
# -msse4.1 and fails to build on Apple Silicon. The server only needs the
# package importable via KIMODO_ROOT on sys.path.
if [ ! -d "$KIMODO_DIR/kimodo" ]; then
  git clone --depth 1 https://github.com/nv-tlabs/kimodo "$KIMODO_DIR" \
    || die "kimodo clone" "check network access to github.com/nv-tlabs/kimodo"
fi

phase "model weights -> public/kimodo.bin (downloads ~540 MB from Hugging Face on first run)"
if [ ! -f public/kimodo.bin ]; then
  CKPT=$("$PY" - <<'PYEOF'
from huggingface_hub import hf_hub_download
print(hf_hub_download('nvidia/Kimodo-SOMA-RP-v1.1', 'model.safetensors'))
PYEOF
  ) || die "weight download" "check Hugging Face access to nvidia/Kimodo-SOMA-RP-v1.1"
  "$PY" tools/convert_weights.py --model "$CKPT" --output public/kimodo.bin --dtype fp16 \
    || die "weight conversion"
fi

phase "app dependencies (npm install)"
[ -d node_modules ] || npm install || die "npm install"

phase "done"
echo "Everything is in place. Start the route with:  ./run_local.sh"
