#!/usr/bin/env bash
# One-shot local setup for kimodo-webgpu.
#
# Installs into ./.setup/ (gitignored): a Python venv with the embedding
# server's dependencies and a pinned Kimodo checkout. Also produces
# public/kimodo.bin (converted weights) and node_modules/.
#
# Idempotent by re-execution, not by trusting directory presence: package
# managers re-run (they no-op when satisfied) and completion is verified
# against the state the route actually needs. Every failure names its phase.
# Tested on Apple Silicon macOS; other platforms may need a different torch.
#
# After this succeeds:  ./run_local.sh
set -euo pipefail
cd "$(dirname "$0")"

SETUP_DIR=".setup"
VENV="$SETUP_DIR/venv"
KIMODO_DIR="$SETUP_DIR/kimodo"
PY="$VENV/bin/python"

# The Kimodo revision this repo's server integration targets. This exact
# commit was exercised end-to-end (clean clone -> setup -> live smoke) on
# 2026-08-30. Bump deliberately, then re-run the clean-clone verification.
KIMODO_COMMIT="1aece8c124d73d255ceff5086d983b844c9f4e94"
KIMODO_URL="https://github.com/nv-tlabs/kimodo"

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
# Always run the installer: it is the idempotent operation. Directory or
# import-subset presence is not evidence of a complete dependency set.
if command -v uv >/dev/null; then
  uv pip install --python "$PY" "${DEPS[@]}" || die "server dependency install (uv)"
else
  "$PY" -m pip install --upgrade pip >/dev/null || die "pip bootstrap"
  "$PY" -m pip install "${DEPS[@]}" || die "server dependency install (pip)"
fi
# Verify the installed state the route actually needs, versions included.
"$PY" - <<'PYEOF' || die "dependency verification" "the venv does not satisfy the declared dependency set"
import importlib
for m in ("torch", "transformers", "peft", "hydra", "omegaconf", "einops",
          "numpy", "scipy", "tqdm", "pydantic", "filelock", "packaging",
          "safetensors", "huggingface_hub"):
    importlib.import_module(m)
import transformers, peft
assert transformers.__version__ == "5.1.0", f"transformers {transformers.__version__} != 5.1.0"
major, minor = (int(x) for x in peft.__version__.split(".")[:2])
assert (major, minor) >= (0, 18), f"peft {peft.__version__} < 0.18"
PYEOF

phase "kimodo checkout pinned at ${KIMODO_COMMIT:0:12} (LLM2Vec encoder source)"
# Note: no `pip install -e` — Kimodo's C++ extension hardcodes -msse4.1 and
# does not build on Apple Silicon; the server imports it via KIMODO_ROOT.
kimodo_ok() {
  [ -d "$KIMODO_DIR/kimodo" ] || return 1
  [ "$(git -C "$KIMODO_DIR" rev-parse HEAD 2>/dev/null)" = "$KIMODO_COMMIT" ]
}
if ! kimodo_ok; then
  if [ -d "$KIMODO_DIR" ] && HAVE=$(git -C "$KIMODO_DIR" rev-parse HEAD 2>/dev/null); then
    # A real checkout at the wrong revision: refuse to discard it silently.
    die "kimodo revision check" \
"existing checkout at $KIMODO_DIR is at $HAVE, expected $KIMODO_COMMIT.
Remove it (rm -rf $KIMODO_DIR) to let setup fetch the pinned revision."
  fi
  # Missing, or leftover junk from an interrupted attempt: rebuild in a
  # temporary directory and publish only a verified checkout.
  rm -rf "$KIMODO_DIR"
  TMP="$KIMODO_DIR.tmp.$$"
  rm -rf "$TMP"
  git init -q "$TMP" || die "kimodo checkout" "git init failed"
  git -C "$TMP" remote add origin "$KIMODO_URL" || die "kimodo checkout" "remote add failed"
  git -C "$TMP" fetch -q --depth 1 origin "$KIMODO_COMMIT" \
    || { rm -rf "$TMP"; die "kimodo fetch" "could not fetch $KIMODO_COMMIT from $KIMODO_URL"; }
  git -C "$TMP" checkout -q FETCH_HEAD \
    || { rm -rf "$TMP"; die "kimodo checkout" "checkout of the pinned commit failed"; }
  [ "$(git -C "$TMP" rev-parse HEAD)" = "$KIMODO_COMMIT" ] \
    || { rm -rf "$TMP"; die "kimodo verification" "fetched revision does not match the pin"; }
  [ -d "$TMP/kimodo" ] \
    || { rm -rf "$TMP"; die "kimodo verification" "checkout lacks the kimodo package directory"; }
  mv "$TMP" "$KIMODO_DIR"
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
# The route invokes vite; its presence is the completion predicate, not the
# existence of a node_modules directory.
if [ ! -e node_modules/.bin/vite ]; then
  npm install || die "npm install"
  [ -e node_modules/.bin/vite ] || die "npm verification" "vite is not installed after npm install"
fi

phase "recording setup manifest"
{
  echo "kimodo_commit=$KIMODO_COMMIT"
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PY" -c 'import torch, transformers, peft; print(f"torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__}")' 2>/dev/null || echo "versions=unrecorded"
} > "$SETUP_DIR/manifest" || die "manifest write"

phase "done"
echo "Everything is in place. Start the route with:  ./run_local.sh"
