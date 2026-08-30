#!/usr/bin/env python3
"""Deterministic negative-path tests for setup.sh and run_local.sh.

An independent review falsified this slice's load-bearing claims:

- run_local.sh accepted ANY listener on the embed port as its own server
  (its grep even matched `"model_loaded": false`), never noticed a child
  dying after readiness, and discarded child exit statuses via a bare `wait`.
- setup.sh's idempotency skip-predicates accepted partial venvs, wedged or
  incomplete Kimodo checkouts, and empty node_modules as completed phases,
  and the pip bootstrap failure bypassed the named-phase reporter.

These tests encode those counterexamples. They run in seconds: heavy tools
(uv, git, npm, the embed server) are replaced with PATH shims and a fake venv
python; foreign servers are real local HTTP listeners on ephemeral ports.

Run:  python3 tests/test_local_scripts.py
"""
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not ok else ""))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def foreign_server(port, body):
    """A real HTTP listener that is NOT run_local's child."""
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            payload = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ---------------------------------------------------------------------------
# Sandbox construction
# ---------------------------------------------------------------------------

FAKE_PY = r'''#!/bin/bash
# Fake .setup/venv/bin/python for lifecycle tests. Behavior via FAKE_EMBED_MODE:
#   load-forever      : never binds, sleeps (legacy bind-late server, loading)
#   bind-late         : sleeps BIND_DELAY, THEN tries to bind (models the old
#                       load-then-bind server; loses bind races -> exit 9)
#   serve             : binds immediately, model_loaded true, serves until killed
#   loading-then-ready: binds immediately, model_loaded false for LOAD_SECS, then true
#   serve-then-die    : as serve, but exits DIE_AFTER seconds after binding
#   die-loading       : exits 3 after 1s (load failure)
port=8098
prev=""
for a in "$@"; do [ "$prev" = "--port" ] && port="$a"; prev="$a"; done
mode="${FAKE_EMBED_MODE:-serve}"
case "$mode" in
  load-forever) sleep 3600; exit 0 ;;
  die-loading)  sleep 1; echo "fake load failure" >&2; exit 3 ;;
esac
[ "$mode" = "bind-late" ] && sleep "${BIND_DELAY:-2}"
exec /usr/bin/env python3 - "$port" "$mode" "${DIE_AFTER:-6}" "${LOAD_SECS:-3}" <<'INNERPY'
import http.server, json, sys, threading, time, os
port, mode, die_after, load_secs = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
state = {"loaded": mode != "loading-then-ready"}
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b = json.dumps({"status": "ok" if state["loaded"] else "loading",
                        "model_loaded": state["loaded"]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass
try:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
except OSError as e:
    print(f"bind failed: {e}", file=sys.stderr); sys.exit(9)
if mode == "loading-then-ready":
    def finish(): time.sleep(load_secs); state["loaded"] = True
    threading.Thread(target=finish, daemon=True).start()
if mode == "serve-then-die":
    def die(): time.sleep(die_after); os._exit(7)
    threading.Thread(target=die, daemon=True).start()
srv.serve_forever()
INNERPY
'''

FAKE_NPM = r'''#!/bin/bash
# Fake npm. "run dev" prints a vite-style URL then idles (or dies, per FAKE_VITE_MODE).
# "install" records its invocation.
if [ "$1" = "install" ]; then echo called >> "$SHIM_LOG_DIR/npm-install.log"; mkdir -p node_modules/.bin; touch node_modules/.bin/vite; exit 0; fi
if [ "$1" = "run" ] && [ "$2" = "dev" ]; then
  echo "  Local:   http://localhost:${FAKE_VITE_PORT:-5199}/"
  # Spawn a REAL descendant (the "vite" process) and record its PID, so tests
  # judge the wrapper's termination against actual process-tree semantics
  # instead of a collapsed single-PID fake.
  /usr/bin/env python3 -c "import time
while True: time.sleep(1)" &
  CHILD=$!
  echo "$CHILD" > "$SHIM_LOG_DIR/vite-child.pid"
  trap 'echo terminated >> "$SHIM_LOG_DIR/vite-exit.log"; kill "$CHILD" 2>/dev/null; exit 0' TERM INT
  if [ "${FAKE_VITE_MODE:-idle}" = "die" ]; then sleep "${DIE_AFTER:-3}"; kill "$CHILD" 2>/dev/null; echo "vite crashed" >&2; exit 5; fi
  if [ "${FAKE_VITE_MODE:-idle}" = "orphaning-die" ]; then sleep "${DIE_AFTER:-3}"; echo "wrapper crashed leaving child" >&2; exit 5; fi
  wait "$CHILD"
fi
exit 0
'''

FAKE_GIT = r'''#!/bin/bash
# Fake git for setup.sh phases. Supports the shapes the script may use and
# records every invocation. FAKE_KIMODO_SHA controls rev-parse output.
echo "git $*" >> "$SHIM_LOG_DIR/git.log"
sha="${FAKE_KIMODO_SHA:-1aece8c124d73d255ceff5086d983b844c9f4e94}"
case "$1" in
  clone)
    dest="${@: -1}"
    [ -e "$dest" ] && { echo "fatal: destination path '$dest' already exists" >&2; exit 128; }
    mkdir -p "$dest/kimodo" "$dest/.git"; echo "$sha" > "$dest/.git/FAKE_HEAD"; exit 0 ;;
  init) dest="${@: -1}"; mkdir -p "$dest/.git"; exit 0 ;;
  -C)
    dir="$2"; shift 2
    case "$1" in
      remote)   exit 0 ;;
      fetch)    echo "$sha" > "$dir/.git/FAKE_HEAD"; exit 0 ;;
      checkout) mkdir -p "$dir/kimodo"; exit 0 ;;
      rev-parse) cat "$dir/.git/FAKE_HEAD" 2>/dev/null || { echo fatal >&2; exit 128; } ;;
      *) exit 0 ;;
    esac ;;
  *) exit 0 ;;
esac
'''

FAKE_UV = r'''#!/bin/bash
echo "uv $*" >> "$SHIM_LOG_DIR/uv.log"
case "$1" in
  venv) dest="${@: -1}"; mkdir -p "$dest/bin"; cp "$SHIM_LOG_DIR/../fake-venv-python" "$dest/bin/python"; chmod +x "$dest/bin/python"; exit 0 ;;
  pip)  exit 0 ;;
esac
exit 0
'''

FAKE_VENV_CHECK_PY = r'''#!/bin/bash
# Fake venv python for setup-phase tests: import checks succeed, everything
# else records and succeeds — EXCEPT modes used to force specific failures.
echo "venvpy $*" >> "$SHIM_LOG_DIR/venvpy.log"
if [ "${FAKE_PIP_BOOTSTRAP_FAIL:-0}" = "1" ]; then
  case "$*" in *"--upgrade pip"*) echo "pip bootstrap exploded" >&2; exit 1 ;; esac
fi
if [ "${FAKE_IMPORT_FAIL:-0}" = "1" ]; then
  case "$*" in *"import "*) exit 1 ;; esac
fi
exit 0
'''


def make_sandbox(tmp):
    """Copy the scripts into a sandbox with shims; return paths + env."""
    sb = Path(tmp)
    for f in ("setup.sh", "run_local.sh"):
        shutil.copy(REPO / f, sb / f)
        os.chmod(sb / f, 0o755)
    (sb / "tools").mkdir()
    (sb / "tools" / "embed_server.py").write_text("# placeholder; the fake venv python ignores this\n")
    (sb / "tools" / "convert_weights.py").write_text("print('fake convert')\n")
    (sb / "public").mkdir()
    (sb / "public" / "kimodo.bin").write_bytes(b"FAKEWEIGHTS")

    shims = sb / "shims"
    logs = shims / "logs"
    logs.mkdir(parents=True)
    for name, body in (("npm", FAKE_NPM), ("git", FAKE_GIT), ("uv", FAKE_UV)):
        p = shims / name
        p.write_text(body)
        os.chmod(p, 0o755)
    (shims / "fake-venv-python").write_text(FAKE_VENV_CHECK_PY)
    os.chmod(shims / "fake-venv-python", 0o755)

    env = dict(os.environ)
    env["PATH"] = f"{shims}:/usr/bin:/bin:/usr/sbin:/sbin"
    env["SHIM_LOG_DIR"] = str(logs)
    return sb, logs, env


def with_lifecycle_venv(sb, mode):
    """Install the lifecycle fake python as .setup/venv/bin/python."""
    venv_bin = sb / ".setup" / "venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    py = venv_bin / "python"
    py.write_text(FAKE_PY)
    os.chmod(py, 0o755)
    kim = sb / ".setup" / "kimodo" / "kimodo"
    kim.mkdir(parents=True, exist_ok=True)
    return mode


def run(cmd, cwd, env, timeout, extra_env=None):
    e = dict(env)
    if extra_env:
        e.update(extra_env)
    try:
        p = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr, False
    except subprocess.TimeoutExpired as ex:
        out = (ex.stdout or b"").decode(errors="replace") if isinstance(ex.stdout, bytes) else (ex.stdout or "")
        err = (ex.stderr or b"").decode(errors="replace") if isinstance(ex.stderr, bytes) else (ex.stderr or "")
        subprocess.run(["pkill", "-f", str(cwd)], capture_output=True)
        return None, out + err, True


# ---------------------------------------------------------------------------
# run_local.sh lifecycle scenarios
# ---------------------------------------------------------------------------

def lifecycle_case(name, embed_mode, vite_mode, foreign_body=None, die_after="4",
                   foreign_delay=None, extra=None,
                   expect_nonzero=True, expect_fast=True, expect_text=None,
                   expect_sibling_stopped=False, expect_vite_child_dead=False):
    with tempfile.TemporaryDirectory() as tmp:
        sb, logs, env = make_sandbox(tmp)
        with_lifecycle_venv(sb, embed_mode)
        port = free_port()
        srv = None
        timer = None
        if foreign_body is not None:
            if foreign_delay is None:
                srv = foreign_server(port, foreign_body)
            else:
                # Bind the foreign listener only AFTER run_local's pre-check
                # has passed — the race the ownership contract must close.
                holder_srv = {}
                def bind_late():
                    try:
                        holder_srv["srv"] = foreign_server(port, foreign_body)
                    except OSError:
                        pass  # our child won the bind: also a valid outcome
                timer = threading.Timer(foreign_delay, bind_late)
                timer.start()
        e = {
            "EMBED_PORT": str(port),
            "FAKE_EMBED_MODE": embed_mode,
            "FAKE_VITE_MODE": vite_mode,
            "FAKE_VITE_PORT": str(free_port()),
            "DIE_AFTER": die_after,
        }
        if extra:
            e.update(extra)
        rc, out, timed_out = run(["bash", "run_local.sh"], sb, env, timeout=40, extra_env=e)
        if timer:
            timer.cancel()
        if srv:
            srv.shutdown()
        if foreign_delay is not None and "srv" in (holder_srv if foreign_body is not None else {}):
            holder_srv["srv"].shutdown()
        ok = True
        detail = f"rc={rc} timed_out={timed_out}"
        if expect_fast and timed_out:
            ok = False
            detail += " (hung instead of exiting)"
        if expect_nonzero and rc == 0:
            ok = False
            detail += " (exited zero on a failure path)"
        if expect_text and expect_text not in out:
            ok = False
            detail += f" (missing: {expect_text!r})"
        if expect_sibling_stopped and not (logs / "vite-exit.log").exists():
            ok = False
            detail += " (vite sibling never stopped)"
        if expect_vite_child_dead:
            pidfile = logs / "vite-child.pid"
            if not pidfile.exists():
                ok = False
                detail += " (no vite child was ever spawned)"
            else:
                pid = int(pidfile.read_text().strip())
                time.sleep(1)
                try:
                    os.kill(pid, 0)
                    ok = False
                    detail += f" (vite descendant {pid} still alive)"
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
        check(name, ok, detail + " :: " + out.strip().replace("\n", " | ")[-300:])


print("=== run_local.sh lifecycle ===")
lifecycle_case(
    "foreign listener with model_loaded:false is refused",
    "load-forever", "idle",
    foreign_body=json.dumps({"status": "loading", "model_loaded": False}),
)
lifecycle_case(
    "foreign listener with a valid-looking health doc is refused (port ownership)",
    "load-forever", "idle",
    foreign_body=json.dumps({"status": "ok", "model_loaded": True}),
)
lifecycle_case(
    "child dying during load fails loud",
    "die-loading", "idle",
    expect_text="embed server",
)
lifecycle_case(
    "embed server dying after readiness stops the wrapper non-zero",
    "serve-then-die", "idle", die_after="12",
    expect_sibling_stopped=True,
)
lifecycle_case(
    "vite dying after readiness stops the wrapper non-zero",
    "serve", "die", die_after="3",
)
lifecycle_case(
    "foreign server binding AFTER the pre-check is never adopted (bind race)",
    "load-forever", "idle",
    foreign_body=json.dumps({"status": "ok", "model_loaded": True}),
    foreign_delay=2.0,
)
lifecycle_case(
    "legacy bind-late child losing the port race fails loud, not false-ready",
    "bind-late", "idle",
    foreign_body=json.dumps({"status": "ok", "model_loaded": True}),
    foreign_delay=1.0,
    extra={"BIND_DELAY": "3"},
)
lifecycle_case(
    "embed death: the real vite DESCENDANT dies with the wrapper",
    "serve-then-die", "idle", die_after="12",
    expect_sibling_stopped=True, expect_vite_child_dead=True,
)
lifecycle_case(
    "orphaning vite wrapper crash still leaves no live descendant",
    "serve", "orphaning-die", die_after="3",
    expect_vite_child_dead=True,
)

# ---------------------------------------------------------------------------
# setup.sh phase-completion scenarios
# ---------------------------------------------------------------------------

print("=== setup.sh phases ===")

def setup_case(name, prepare, expect_rc0, expect, extra_env=None):
    with tempfile.TemporaryDirectory() as tmp:
        sb, logs, env = make_sandbox(tmp)
        prepare(sb, logs)
        rc, out, timed_out = run(["bash", "setup.sh"], sb, env, timeout=60, extra_env=extra_env)
        ok, detail = True, f"rc={rc}"
        if timed_out:
            ok, detail = False, "hung"
        elif expect_rc0 and rc != 0:
            ok = False
            detail += " (expected success)"
        elif not expect_rc0 and rc == 0:
            ok = False
            detail += " (expected failure)"
        if ok and not expect(sb, logs, out):
            ok = False
            detail += " (expectation failed)"
        check(name, ok, detail + " :: " + out.strip().replace("\n", " | ")[-300:])


def prep_partial_venv(sb, logs):
    """A venv whose python satisfies import probes but whose deps are unverified."""
    venv_bin = sb / ".setup" / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    py = venv_bin / "python"
    py.write_text(FAKE_VENV_CHECK_PY)
    os.chmod(py, 0o755)

setup_case(
    "existing venv still gets a dependency install/verify pass",
    prep_partial_venv,
    expect_rc0=True,
    expect=lambda sb, logs, out: (logs / "uv.log").exists()
        and any("pip install" in l for l in (logs / "uv.log").read_text().splitlines()),
)

def prep_wedged_clone(sb, logs):
    prep_partial_venv(sb, logs)
    d = sb / ".setup" / "kimodo"
    d.mkdir(parents=True)
    (d / "leftover.partial").write_text("interrupted")

setup_case(
    "interrupted kimodo dir (no package) is repaired, not wedged",
    prep_wedged_clone,
    expect_rc0=True,
    expect=lambda sb, logs, out: (sb / ".setup" / "kimodo" / "kimodo").is_dir(),
)

def prep_empty_node_modules(sb, logs):
    prep_partial_venv(sb, logs)
    (sb / "node_modules").mkdir()

setup_case(
    "empty node_modules still triggers npm install",
    prep_empty_node_modules,
    expect_rc0=True,
    expect=lambda sb, logs, out: (logs / "npm-install.log").exists(),
)

def prep_pip_path(sb, logs):
    # No uv on PATH -> the pip branch runs; venv pre-exists with the fake
    # python so creation is skipped and $PY is ours to steer.
    (sb / "shims" / "uv").unlink()
    prep_partial_venv(sb, logs)

setup_case(
    "pip bootstrap failure names its phase",
    prep_pip_path,
    expect_rc0=False,
    expect=lambda sb, logs, out: "FAILED during" in out,
    extra_env={"FAKE_PIP_BOOTSTRAP_FAIL": "1", "FAKE_IMPORT_FAIL": "1"},
)

def prep_wrong_pin(sb, logs):
    prep_partial_venv(sb, logs)
    d = sb / ".setup" / "kimodo"
    (d / "kimodo").mkdir(parents=True)
    (d / ".git").mkdir()
    (d / ".git" / "FAKE_HEAD").write_text("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef\n")

setup_case(
    "existing kimodo checkout at the wrong revision fails loud",
    prep_wrong_pin,
    expect_rc0=False,
    expect=lambda sb, logs, out: "kimodo" in out.lower(),
)

def prep_vite_sentinel(sb, logs):
    """node_modules with the vite shim present but everything else missing —
    a single filesystem sentinel must not substitute for reconciliation."""
    prep_partial_venv(sb, logs)
    d = sb / "node_modules" / ".bin"
    d.mkdir(parents=True)
    (d / "vite").write_text("#!/bin/sh\n")

setup_case(
    "vite sentinel alone does not skip npm reconciliation",
    prep_vite_sentinel,
    expect_rc0=True,
    expect=lambda sb, logs, out: (logs / "npm-install.log").exists(),
)

def prep_sha_ok_pkg_missing(sb, logs):
    """A valid checkout at the PINNED sha whose package dir is missing —
    an interrupted state that must be repaired, not misreported as a
    wrong-revision refusal."""
    prep_partial_venv(sb, logs)
    d = sb / ".setup" / "kimodo"
    (d / ".git").mkdir(parents=True)
    (d / ".git" / "FAKE_HEAD").write_text("1aece8c124d73d255ceff5086d983b844c9f4e94\n")

setup_case(
    "pinned-sha checkout missing its package dir is repaired",
    prep_sha_ok_pkg_missing,
    expect_rc0=True,
    expect=lambda sb, logs, out: (sb / ".setup" / "kimodo" / "kimodo").is_dir(),
)

MV_FAIL_SHIM = "#!/bin/bash\necho mv-fail >> \"$SHIM_LOG_DIR/mv.log\"\nexit 1\n"
MV_COLLIDE_SHIM = """#!/bin/bash
# Simulate the destination-collision hazard: directory-form mv succeeding by
# moving the source INSIDE a newly-appeared destination.
echo mv-collide >> "$SHIM_LOG_DIR/mv.log"
src="$1"; dst="$2"
mkdir -p "$dst"
/bin/mv "$src" "$dst/"
exit 0
"""

def prep_mv_fail(sb, logs):
    prep_partial_venv(sb, logs)
    m = sb / "shims" / "mv"
    m.write_text(MV_FAIL_SHIM)
    os.chmod(m, 0o755)

setup_case(
    "kimodo publication (mv) failure names its phase and cleans its temp",
    prep_mv_fail,
    expect_rc0=False,
    expect=lambda sb, logs, out: "FAILED during" in out
        and not list((sb / ".setup").glob("kimodo.tmp.*")),
)

def prep_mv_collide(sb, logs):
    prep_partial_venv(sb, logs)
    m = sb / "shims" / "mv"
    m.write_text(MV_COLLIDE_SHIM)
    os.chmod(m, 0o755)

setup_case(
    "publication into a colliding destination is caught by post-verification",
    prep_mv_collide,
    expect_rc0=False,
    expect=lambda sb, logs, out: "FAILED during" in out,
)

def prep_stale_tmp(sb, logs):
    """A prior PID's abandoned temp checkout must not accumulate forever."""
    prep_partial_venv(sb, logs)
    d = sb / ".setup" / "kimodo"
    (d / "kimodo").mkdir(parents=True)
    (d / ".git").mkdir()
    (d / ".git" / "FAKE_HEAD").write_text("1aece8c124d73d255ceff5086d983b844c9f4e94\n")
    stale = sb / ".setup" / "kimodo.tmp.99999"
    stale.mkdir()
    (stale / "junk").write_text("abandoned")

setup_case(
    "abandoned temp checkout from a prior run is cleaned up",
    prep_stale_tmp,
    expect_rc0=True,
    expect=lambda sb, logs, out: not (sb / ".setup" / "kimodo.tmp.99999").exists(),
)

failed = [n for n, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
