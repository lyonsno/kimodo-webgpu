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
            name = Path(path).name
            target = name.endswith(".bin") or (".bin." in name and name.endswith(".tmp")) or name.endswith(".bin.tmp")
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


# --- transaction fault matrix: ANY failure leaves the destination pair intact ---
#
# The contract sentence is "after any failed or interrupted conversion, the
# destination binary+config pair is byte-identical to the prior state". The
# quantifier "any" is enumerated below as a fault per publication step, plus
# concurrent writers. A prior revision tested only a fault during the binary
# payload write and wrongly treated the transaction as closed.

def _load_cw(tag):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"cw_{tag}", CONV)
    m = importlib.util.module_from_spec(spec); sys.modules[f"cw_{tag}"] = m
    spec.loader.exec_module(m)
    return m

def _small_sd():
    return {f"model.body_model.layers.{i}.linear1.weight":
            np.random.randn(64, 64).astype(np.float32) for i in range(4)}

def _snapshot(d):
    return {f.name: f.read_bytes() for f in Path(d).iterdir() if f.is_file()}

import io, contextlib, builtins as _bi

def _fault_case(tag, fresh, fault):
    """Run convert() with `fault` armed; return (raised, pair_unchanged, leftovers)."""
    cw = _load_cw(tag)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); out = d / "kimodo.bin"; sd = _small_sd()
        if not fresh:
            with contextlib.redirect_stdout(io.StringIO()):
                cw.convert(sd, str(out), "fp16")
        before = _snapshot(d)
        undo = fault(cw, d)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cw.convert(sd, str(out), "fp16")
            raised = False
        except Exception:
            raised = True
        finally:
            undo()
        after = _snapshot(d)
        durable_after = {k: v for k, v in after.items() if not k.endswith(".tmp")}
        durable_before = {k: v for k, v in before.items() if not k.endswith(".tmp")}
        leftovers = [k for k in after if k.endswith(".tmp")]
        return raised, durable_after == durable_before, leftovers

def _fault_config_write(cw, d):
    real_open = _bi.open
    def patched(path, mode="r", *a, **k):
        name = Path(path).name
        if "w" in mode and ".json" in name:
            raise OSError(28, "No space left on device")
        return real_open(path, mode, *a, **k)
    _bi.open = patched
    return lambda: setattr(_bi, "open", real_open)

def _fault_replace(target_frag):
    def arm(cw, d):
        real_replace = cw.os.replace
        def patched(src, dst, *a, **k):
            if target_frag in Path(dst).name:
                raise OSError(5, "Input/output error")
            return real_replace(src, dst, *a, **k)
        cw.os.replace = patched
        return lambda: setattr(cw.os, "replace", real_replace)
    return arm

for label, fresh, fault in [
    ("fresh install: config write fails",        True,  _fault_config_write),
    ("fresh install: config replace fails",      True,  _fault_replace(".json")),
    ("fresh install: binary replace fails",      True,  _fault_replace(".bin")),
    ("existing pair: binary replace fails",      False, _fault_replace(".bin")),
]:
    raised, unchanged, leftovers = _fault_case(label.replace(" ", "_")[:20], fresh, fault)
    check(f"{label}: raises", raised)
    check(f"{label}: destination pair unchanged", unchanged)
    check(f"{label}: no stray temp files", not leftovers, f"left {leftovers}")

# --- concurrent conversions must not corrupt the destination ---
def _concurrency_case():
    import threading
    cw = _load_cw("conc")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); out = d / "kimodo.bin"
        sd_a = _small_sd(); sd_b = _small_sd()

        temp_names = []
        a_opened = threading.Event(); b_done = threading.Event()
        real_open = _bi.open
        class BlockingFirstWrite:
            """Thread A holds an OPEN descriptor and blocks before its first
            write, so B can truncate and publish through the same temp path in
            between. With shared temp names this is the inode interleaving
            that lets a failed run mutate the published destination in place;
            unique per-run temp names make the scenario harmless."""
            def __init__(self, f): self.f = f; self.blocked = False
            def write(self, b):
                if not self.blocked:
                    self.blocked = True
                    a_opened.set()
                    b_done.wait(timeout=30)
                return self.f.write(b)
            def __getattr__(self, n): return getattr(self.f, n)
            def __enter__(self): return self
            def __exit__(self, *a): return self.f.__exit__(*a)

        def patched(path, mode="r", *a, **k):
            name = Path(path).name
            f = real_open(path, mode, *a, **k)
            if "wb" in mode and name.endswith(".tmp"):
                temp_names.append(name)
                if threading.current_thread().name == "A":
                    return BlockingFirstWrite(f)
            return f

        errors = []
        succeeded = []
        def run(tag, sd):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    cw.convert(sd, str(out), "fp16")
                succeeded.append(tag)
            except Exception as e:
                errors.append((tag, repr(e)))

        _bi.open = patched
        try:
            ta = threading.Thread(target=run, args=("A", sd_a), name="A")
            tb = threading.Thread(target=run, args=("B", sd_b), name="B")
            ta.start(); a_opened.wait(timeout=30)
            tb.start(); tb.join(timeout=60); b_done.set(); ta.join(timeout=60)
        finally:
            _bi.open = real_open

        final = out.read_bytes() if out.exists() else b""
        # The destination must be exactly one complete valid conversion.
        cw2 = _load_cw("conc_ref")
        va = Path(td) / "a.bin"; vb = Path(td) / "b.bin"
        with contextlib.redirect_stdout(io.StringIO()):
            cw2.convert(sd_a, str(va), "fp16"); cw2.convert(sd_b, str(vb), "fp16")
        # The destination must equal the output of a run that SUCCEEDED. A raised
        # run mutating the destination in place (through a shared temp inode)
        # can leave a complete-looking artifact that no successful run published.
        by_tag = {"A": va.read_bytes(), "B": vb.read_bytes()}
        complete = any(final == by_tag[t] for t in succeeded)
        distinct_temps = len(temp_names) == len(set(temp_names)) and len(temp_names) >= 2
        return complete, distinct_temps, errors + [("succeeded", succeeded)]

complete, distinct_temps, conc_errors = _concurrency_case()
check("concurrent runs: destination is one complete valid conversion", complete,
      f"errors={conc_errors}")
check("concurrent runs: temp file names are unique per run", distinct_temps)

# --- cleanup owns only its own temp ---
def _ownership_case():
    cw = _load_cw("own")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); out = d / "kimodo.bin"
        stale = d / "kimodo.bin.stale-owner.tmp"; stale.write_bytes(b"foreign")
        undo = _fault_replace(".bin")(cw, d)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cw.convert(_small_sd(), str(out), "fp16")
        except Exception:
            pass
        finally:
            undo()
        own_temps = [f.name for f in d.iterdir() if f.name.endswith(".tmp") and f.name != stale.name]
        return stale.exists(), own_temps

stale_survives, own_temps = _ownership_case()
check("failed run leaves a foreign temp untouched", stale_survives)
check("failed run removes its own temp", not own_temps, f"left {own_temps}")

failed = [(n, d) for n, ok, d in results if not ok]
for n, ok, d in results:
    print(f"{'PASS' if ok else 'FAIL'}  {n}" + (f"   [{d}]" if d and not ok else ""))
print(f"\n{len(results)-len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
