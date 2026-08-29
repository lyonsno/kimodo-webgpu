#!/usr/bin/env python3
"""Minimal text-embedding server for kimodo-webgpu.

The browser owns the entire diffusion pipeline: DDIM sampling, classifier-free
guidance, the two 16-layer transformer sub-networks, forward kinematics, and
rendering. It cannot own text encoding, because that requires Kimodo's
LLM2Vec/Llama 3 8B encoder (~16 GB).

This server exists to close exactly that gap. It exposes one endpoint:

    POST /embed  {"prompt": "..."}  ->  {"embedding": [...4096 floats], ...}

Prerequisites (both external to this repository):

  1. A checkout of NVIDIA's Kimodo:  https://github.com/nv-tlabs/kimodo
     Point KIMODO_ROOT at it (default: ~/dev/kimodo).
  2. The Kimodo-SOMA-RP-v1.1 weights, including the text encoder, under
     $KIMODO_ROOT/models (override with CHECKPOINT_DIR).

     huggingface-cli download nvidia/Kimodo-SOMA-RP-v1.1

Usage:

    python tools/embed_server.py --port 8098

The server loads the model at startup and exits non-zero if that fails, rather
than starting and failing per-request. A running server means a working route.
"""

import argparse
import http.server
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

# Populated by load_text_encoder() before the server starts serving.
model = None


def load_text_encoder(device: str):
    """Build Kimodo's LLM2Vec text encoder.

    Only the text encoder is loaded. The 282M diffusion transformer stays out of
    memory entirely — the browser owns that half, so loading it here would cost
    time and RAM for nothing.

    Raises with an actionable message if the Kimodo checkout is missing, since
    that is the failure mode a new user will actually hit.
    """
    kimodo_root = Path(os.environ.get("KIMODO_ROOT", os.path.expanduser("~/dev/kimodo")))
    if not kimodo_root.is_dir():
        raise SystemExit(
            f"Kimodo checkout not found at {kimodo_root}\n"
            "Clone https://github.com/nv-tlabs/kimodo and set KIMODO_ROOT to it."
        )

    sys.path.insert(0, str(kimodo_root))

    import warnings
    import logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    # Kimodo's own "auto" resolves to cuda-or-cpu and never picks mps, which
    # would silently run an 8B model on CPU here. Set the device explicitly.
    os.environ["TEXT_ENCODER_DEVICE"] = device

    try:
        from kimodo.model import resolve_target
    except ImportError as e:
        raise SystemExit(
            f"Could not import kimodo from {kimodo_root}: {e}\n"
            "Install its dependencies (including torch, which Kimodo expects to "
            "be present already) into the environment running this server."
        )

    # Mirrors kimodo/scripts/run_text_encoder_server.py's llm2vec preset.
    encoder_cls = resolve_target("kimodo.model.LLM2VecEncoder")
    return encoder_cls(
        base_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        dtype=os.environ.get("TEXT_ENCODER_DTYPE", "bfloat16"),
        llm_dim=4096,
        device=device,
    )


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed(prompt: str):
    """Return the text embedding for one prompt as a flat list of floats."""
    # LLM2VecEncoder returns (tensor, lengths); a single prompt yields shape
    # [1, 4096], so the flatten below produces exactly llm_dim floats.
    text_feat = model([prompt])
    if isinstance(text_feat, tuple):
        text_feat = text_feat[0]
    # Detach torch tensors to numpy. Duck-typed rather than isinstance-checked so
    # this path does not require importing torch just to shape a response.
    if hasattr(text_feat, "detach"):
        text_feat = text_feat.detach().cpu().float().numpy()
    return text_feat.flatten().tolist(), list(text_feat.shape)


class EmbedHandler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self.send_json({"status": "ok", "model_loaded": model is not None})
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/embed":
            self.send_json({"error": "Not found"}, 404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
        except (json.JSONDecodeError, ValueError):
            self.send_json({"error": "Invalid JSON"}, 400)
            return

        prompt = body.get("prompt", "").strip()
        if not prompt:
            self.send_json({"error": "prompt required"}, 400)
            return

        try:
            flat, shape = embed(prompt)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json({"error": str(e)}, 500)
            return

        self.send_json({"embedding": flat, "dim": len(flat), "shape": shape})

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _cors(self):
        # The browser app is served from a different origin (Vite dev server).
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def send_json(self, data, status=200):
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self._cors()
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass  # Quiet: one request per generation, nothing worth logging.


def main():
    global model

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8098,
                        help="default 8098, matching the app's Server URL field")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto | mps | cuda:0 | cpu")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"[embed-server] Loading Kimodo text encoder on {device} (~16 GB)...")
    model = load_text_encoder(device)
    print(f"[embed-server] Listening on http://localhost:{args.port}  POST /embed")

    server = http.server.ThreadingHTTPServer(("", args.port), EmbedHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[embed-server] Stopped.")


if __name__ == "__main__":
    main()
