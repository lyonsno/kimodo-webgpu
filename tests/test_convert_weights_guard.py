"""Atomicity and guard tests for tools/convert_weights.py.

An earlier revision wrote the 540 MB binary before checking whether the existing
sidecar config was compatible, then printed a refusal and exited zero — leaving
a new binary paired with a stale config. It also compared only keys already
present in the existing file, so an empty object read as "matching".

These tests assert that no incompatible config can leave the binary modified and
that every such case exits non-zero.

Run:  python tests/test_convert_weights_guard.py
Requires: numpy, safetensors.
"""
import json, subprocess, sys, tempfile, shutil
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file

CONV = str(Path(__file__).resolve().parent.parent / "tools" / "convert_weights.py")
PY = sys.executable
results = []
def check(name, ok, detail=""):
    results.append((name, ok, detail))

def make_ckpt(d):
    t = {"model.body_model.layers.0.linear1.weight": np.random.randn(64, 32).astype(np.float32)}
    p = d / "ckpt.safetensors"
    save_file(t, str(p))
    return p

def run(ckpt, out, extra=()):
    return subprocess.run([PY, CONV, "--model", str(ckpt), "--output", str(out), *extra],
                          capture_output=True, text=True)

# --- fresh conversion works and is idempotent ---
with tempfile.TemporaryDirectory() as td:
    d = Path(td); ck = make_ckpt(d); out = d / "kimodo.bin"
    r1 = run(ck, out)
    check("fresh conversion succeeds", r1.returncode == 0 and out.exists(), r1.stderr[-200:])
    size1 = out.stat().st_size
    r2 = run(ck, out)
    check("rerun with matching config is idempotent (rc=0)", r2.returncode == 0, r2.stderr[-200:])

# --- drifted config must fail non-zero AND not replace the binary ---
for label, mutate in [
    ("wrong architecture value", lambda c: c.__setitem__("num_heads", 16)),
    ("missing required key",     lambda c: c.pop("num_heads")),
    ("retyped field",            lambda c: c.__setitem__("num_heads", "8")),
    ("dtype mismatch",           lambda c: c.__setitem__("dtype", "fp32")),
    ("empty object",             lambda c: c.clear()),
]:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); ck = make_ckpt(d); out = d / "kimodo.bin"
        run(ck, out)                       # establish a good pair
        cfg = out.with_suffix(".json")
        c = json.loads(cfg.read_text()); mutate(c); cfg.write_text(json.dumps(c))
        out.write_bytes(b"SENTINEL")       # detect any binary rewrite
        r = run(ck, out)
        untouched = out.read_bytes() == b"SENTINEL"
        check(f"{label}: exits non-zero", r.returncode != 0, f"rc={r.returncode}")
        check(f"{label}: binary untouched", untouched, "binary was overwritten")

# --- malformed sidecar ---
with tempfile.TemporaryDirectory() as td:
    d = Path(td); ck = make_ckpt(d); out = d / "kimodo.bin"
    run(ck, out)
    out.with_suffix(".json").write_text("{not json")
    out.write_bytes(b"SENTINEL")
    r = run(ck, out)
    check("malformed sidecar: exits non-zero", r.returncode != 0, f"rc={r.returncode}")
    check("malformed sidecar: binary untouched", out.read_bytes() == b"SENTINEL")

# --- P2: --list-only on the numpy route ---
with tempfile.TemporaryDirectory() as td:
    d = Path(td); ck = make_ckpt(d); out = d / "kimodo.bin"
    r = run(ck, out, extra=("--list-only",))
    check("--list-only works on numpy loader", r.returncode == 0 and "Total:" in r.stdout,
          (r.stderr or r.stdout)[-200:])

failed = [(n, d) for n, ok, d in results if not ok]
for n, ok, d in results:
    print(f"{'PASS' if ok else 'FAIL'}  {n}" + (f"   [{d}]" if d and not ok else ""))
print(f"\n{len(results)-len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
