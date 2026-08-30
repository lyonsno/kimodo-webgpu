#!/usr/bin/env bash
# Start the full local route: text-embedding server + app dev server.
#
# Requires a completed ./setup.sh (or set KIMODO_ROOT / EMBED_PORT for a
# hand-built environment). The embedding port must be FREE: this script owns
# both processes it starts, refuses to adopt a foreign server, supervises
# both children, and stops everything on Ctrl-C or when either child dies.
set -euo pipefail
cd "$(dirname "$0")"

SETUP_DIR=".setup"
PY="$SETUP_DIR/venv/bin/python"
EMBED_PORT="${EMBED_PORT:-8098}"
SERVER_LOG="$SETUP_DIR/embed-server.log"
VITE_LOG="$SETUP_DIR/vite.log"

die() { printf '\033[1;31m[run] %s\033[0m\n' "$1" >&2; exit 1; }

[ -x "$PY" ] || die "no venv at $SETUP_DIR/venv — run ./setup.sh first"
[ -f public/kimodo.bin ] || die "no weights at public/kimodo.bin — run ./setup.sh first"

if [ -z "${KIMODO_ROOT:-}" ]; then
  if [ -d "$SETUP_DIR/kimodo/kimodo" ]; then KIMODO_ROOT="$PWD/$SETUP_DIR/kimodo";
  elif [ -d "$HOME/dev/kimodo/kimodo" ]; then KIMODO_ROOT="$HOME/dev/kimodo";
  else die "no Kimodo checkout found — run ./setup.sh or set KIMODO_ROOT"; fi
fi
export KIMODO_ROOT

# --- Port ownership: refuse to run against a port anything already answers on.
# Accepting a foreign listener's health response as our own server is a false
# closure; ours must be the only claimant before we launch it.
if curl -s -m 2 -o /dev/null "http://127.0.0.1:$EMBED_PORT/health" 2>/dev/null; then
  die "something is already listening on port $EMBED_PORT — stop it or set EMBED_PORT to a free port"
fi

SERVER_PID=""
VITE_PID=""
INTERRUPTED=0

stop_children() {
  [ -n "$VITE_PID" ]   && kill "$VITE_PID"   2>/dev/null || true
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  [ -n "$VITE_PID" ]   && wait "$VITE_PID"   2>/dev/null || true
  [ -n "$SERVER_PID" ] && wait "$SERVER_PID" 2>/dev/null || true
}
on_signal() { INTERRUPTED=1; stop_children; echo; echo "[run] stopped."; exit 0; }
trap on_signal INT TERM

echo "[run] starting text-embedding server (log: $SERVER_LOG)..."
PYTHONUNBUFFERED=1 "$PY" tools/embed_server.py --port "$EMBED_PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo -n "[run] waiting for the encoder to load (first run downloads ~16 GB)"
READY=0
for _ in $(seq 1 360); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo; echo "[run] embed server exited during startup. Last log lines:" >&2
    tail -15 "$SERVER_LOG" >&2
    die "embed server failed"
  fi
  # Readiness requires BOTH: our child is alive, and the health document says
  # the model is actually loaded. A substring match previously accepted
  # {"model_loaded": false} — and any foreign body at all.
  BODY=$(curl -sf -m 2 "http://127.0.0.1:$EMBED_PORT/health" 2>/dev/null || true)
  if printf '%s' "$BODY" | grep -q '"model_loaded": true' && kill -0 "$SERVER_PID" 2>/dev/null; then
    READY=1; echo " ready."; break
  fi
  echo -n "."; sleep 5
done
[ "$READY" = 1 ] || { stop_children; die "embed server did not become healthy within 30 minutes; see $SERVER_LOG"; }

echo "[run] starting app dev server (log: $VITE_LOG)..."
npm run dev >"$VITE_LOG" 2>&1 &
VITE_PID=$!
URL=""
for _ in $(seq 1 60); do
  URL=$(grep -oE 'http://localhost:[0-9]+/' "$VITE_LOG" | head -1 || true)
  [ -n "$URL" ] && break
  if ! kill -0 "$VITE_PID" 2>/dev/null; then
    tail -15 "$VITE_LOG" >&2; stop_children; die "vite failed to start"
  fi
  sleep 1
done
[ -n "$URL" ] || { stop_children; die "vite started but printed no URL; see $VITE_LOG"; }
echo "[run] app: $URL  (embedding server on :$EMBED_PORT)"
echo "[run] open the URL, type a prompt, click Generate. Ctrl-C stops everything."

# --- Supervision: either child dying is a failure of the route, reported with
# its status and log, the sibling stopped and reaped, and a non-zero exit —
# never a silent wait on the survivor.
while :; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then DEAD="embed server"; DEAD_PID=$SERVER_PID; DEAD_LOG=$SERVER_LOG; break; fi
  if ! kill -0 "$VITE_PID" 2>/dev/null;   then DEAD="app dev server"; DEAD_PID=$VITE_PID; DEAD_LOG=$VITE_LOG; break; fi
  sleep 1
done
[ "$INTERRUPTED" = 1 ] && exit 0
set +e
wait "$DEAD_PID" 2>/dev/null
DEAD_STATUS=$?
set -e
echo; echo "[run] $DEAD exited unexpectedly (status $DEAD_STATUS). Last log lines:" >&2
tail -15 "$DEAD_LOG" >&2
stop_children
die "$DEAD died; route is down"
