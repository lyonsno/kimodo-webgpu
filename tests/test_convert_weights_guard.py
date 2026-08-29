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


# --- interrupted/failed writes must not destroy an existing good binary ---
def _atomicity_case():
    """Fail partway through the payload write and check the destination."""
    import importlib.util, builtins
    spec = importlib.util.spec_from_file_location("cw", CONV)
    cw = importlib.util.module_from_spec(spec); sys.modules["cw"] = cw; spec.loader.exec_module(cw)

    real_open = builtins.open
    class ExplodingFile:
        def __init__(self, f): self.f = f; self.n = 0
        def write(self, b):
            self.n += len(b)
            if self.n > 200_000:
                raise OSError(28, "No space left on device")
            return self.f.write(b)
        def __enter__(self): return self
        def __exit__(self, *a): return self.f.__exit__(*a)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "kimodo.bin"
        sd = {f"model.body_model.layers.{i}.linear1.weight":
              np.random.randn(256, 256).astype(np.float32) for i in range(20)}
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            cw.convert(sd, str(out), "fp16")
        good = out.read_bytes()

        def patched(path, mode="r", *a, **k):
            f = real_open(path, mode, *a, **k)
            # Match the temp file too: an atomic implementation writes to
            # <name>.bin.tmp before replacing the destination.
            target = str(path).endswith(".bin") or str(path).endswith(".bin.tmp")
            return ExplodingFile(f) if "b" in mode and "w" in mode and target else f

        builtins.open = patched
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cw.convert(sd, str(out), "fp16")
            raised = False
        except Exception:
            raised = True
        finally:
            builtins.open = real_open

        after = out.read_bytes() if out.exists() else b""
        return raised, after == good

raised, intact = _atomicity_case()
check("mid-write failure is raised, not swallowed", raised)
check("mid-write failure leaves the existing binary intact", intact,
      "a partially written binary replaced a valid one")


# --- recursive type compatibility (nested bool vs int, integral floats) ---
def _config_cases():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cw2", CONV)
    cw = importlib.util.module_from_spec(spec); sys.modules["cw2"] = cw; spec.loader.exec_module(cw)
    base = cw.build_config("fp16")
    out = []

    def verdict(mutate):
        c = json.loads(json.dumps(base)); mutate(c)
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "k.json"; cfg.write_text(json.dumps(c))
            try:
                cw.check_existing_config(cfg, base); return "accepted"
            except SystemExit: return "rejected"

    # A bool smuggled into parents must NOT be accepted (True == 1 in Python).
    def nested_bool(c):
        # Booleans placed where their numeric equivalents already live:
        # [False, True] == [0, 1] in Python, so a shallow `!=` accepts this.
        c["parents"] = list(c["parents"])
        c["parents"][1] = False   # position holds 0
        c["parents"][2] = True    # position holds 1
    out.append(("nested bool in parents is rejected", verdict(nested_bool) == "rejected"))

    # A top-level bool where an int is expected must be rejected.
    out.append(("top-level bool for int is rejected",
                verdict(lambda c: c.__setitem__("fps", True)) == "rejected"))

    # JSON round-trips may yield 30.0 for 30; that is the same value.
    out.append(("integral float equals int",
                verdict(lambda c: c.__setitem__("fps", 30.0)) == "accepted"))

    # A genuinely different number must still be rejected.
    out.append(("different number is rejected",
                verdict(lambda c: c.__setitem__("fps", 60)) == "rejected"))

    # Shorter list must be rejected.
    out.append(("truncated parents list is rejected",
                verdict(lambda c: c.__setitem__("parents", c["parents"][:-1])) == "rejected"))

    # Additive unknown keys are allowed by policy.
    out.append(("additive unknown key is accepted",
                verdict(lambda c: c.__setitem__("futureField", "x")) == "accepted"))
    return out

for name, ok in _config_cases():
    check(name, ok)

failed = [(n, d) for n, ok, d in results if not ok]
for n, ok, d in results:
    print(f"{'PASS' if ok else 'FAIL'}  {n}" + (f"   [{d}]" if d and not ok else ""))
print(f"\n{len(results)-len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
