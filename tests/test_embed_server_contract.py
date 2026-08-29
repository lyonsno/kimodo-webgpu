"""Adversarial contract tests for tools/embed_server.py.

These encode the failure modes an independent review found in an earlier
revision: an all-interfaces bind with wildcard CORS, non-object JSON crashing
the handler, and non-finite or wrong-length embeddings being served as though
they were data.

Each check is written to fail against the defective behavior, so this file is
regression protection rather than a description of current behavior.

Run:  python tests/test_embed_server_contract.py
Requires: numpy (the text encoder itself is stubbed; no model is loaded).
"""
import importlib.util, json, socket, sys, threading, http.server, urllib.request, urllib.error
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent / "tools" / "embed_server.py")
spec = importlib.util.spec_from_file_location("embed_server", SRC)
es = importlib.util.module_from_spec(spec)
sys.modules["embed_server"] = es
spec.loader.exec_module(es)

import numpy as np

results = []
def check(name, ok, detail=""):
    results.append((name, ok, detail))

def lan_ip():
    """Best-effort non-loopback address of this host."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()

class Stub:
    def __call__(self, prompts):
        return np.full((1, 4096), 0.25, dtype=np.float32), [1]

es.model = Stub()

# ---- P1: default bind must be loopback-only ----
host = getattr(es, "DEFAULT_HOST", "")
srv = http.server.ThreadingHTTPServer((host, 0), es.EmbedHandler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
port = srv.server_address[1]

ip = lan_ip()
reachable_lan = False
try:
    with urllib.request.urlopen(f"http://{ip}:{port}/health", timeout=3) as r:
        reachable_lan = r.status == 200
except Exception:
    reachable_lan = False
check("default bind is NOT reachable on a non-loopback interface", not reachable_lan,
      f"lan_ip={ip} port={port} reachable={reachable_lan}")

loop_ok = False
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
        loop_ok = r.status == 200
except Exception as e:
    loop_ok = False
check("loopback still works", loop_ok)

# ---- P1: CORS must not be a blanket wildcard ----
try:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/embed",
                                 data=json.dumps({"prompt": "x"}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        acao = r.headers.get("Access-Control-Allow-Origin")
except Exception as e:
    acao = f"error:{e}"
check("CORS is not wildcard '*' by default", acao != "*", f"ACAO={acao}")

# ---- P2: non-object JSON must return a stable 400, not a handler crash ----
for body, label in [(b"[]", "array"), (b"null", "null"), (b'"s"', "string"), (b"5", "number"),
                    (json.dumps({"prompt": 5}).encode(), "non-string prompt")]:
    code = None
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            code = r.status
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception as e:
        code = f"crash:{type(e).__name__}"
    check(f"JSON {label} -> 400", code == 400, f"got {code}")

# ---- P1: server must reject non-finite / wrong-length encoder output ----
class NanStub:
    def __call__(self, prompts):
        a = np.full((1, 4096), 0.25, dtype=np.float32); a[0][7] = np.nan
        return a, [1]
class ShortStub:
    def __call__(self, prompts):
        return np.full((1, 128), 0.25, dtype=np.float32), [1]

for stub, label in [(NanStub(), "NaN in embedding"), (ShortStub(), "wrong length")]:
    es.model = stub
    code = None
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/embed",
                                     data=json.dumps({"prompt": "x"}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            code = r.status
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception as e:
        code = f"crash:{type(e).__name__}"
    check(f"server rejects {label} (500, not 200)", code == 500, f"got {code}")
es.model = Stub()

# ---- P1: response must not contain non-standard NaN tokens ----
es.model = NanStub()
raw = None
try:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/embed",
                                 data=json.dumps({"prompt": "x"}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read().decode()
except urllib.error.HTTPError as e:
    raw = e.read().decode()
except Exception as e:
    raw = f"crash:{e}"
check("no bare NaN token in JSON response", "NaN" not in (raw or ""), f"body[:80]={(raw or '')[:80]}")
es.model = Stub()

srv.shutdown()

failed = [(n, d) for n, ok, d in results if not ok]
for n, ok, d in results:
    print(f"{'PASS' if ok else 'FAIL'}  {n}" + (f"   [{d}]" if d and not ok else ""))
print(f"\n{len(results)-len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
