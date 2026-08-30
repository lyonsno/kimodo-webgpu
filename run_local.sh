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

# Children are launched as process-group leaders (set -m below), so stopping
# a child stops its whole tree — killing only the npm wrapper PID previously
# left the real dev-server process running as an orphan.
stop_tree() {
  [ -n "$1" ] || return 0
  kill -TERM -- "-$1" 2>/dev/null || kill -TERM "$1" 2>/dev/null || true
  sleep 1
  kill -KILL -- "-$1" 2>/dev/null || true
}
stop_children() {
  stop_tree "$VITE_PID"
  stop_tree "$SERVER_PID"
  [ -n "$VITE_PID" ]   && wait "$VITE_PID"   2>/dev/null || true
  [ -n "$SERVER_PID" ] && wait "$SERVER_PID" 2>/dev/null || true
}
on_signal() { INTERRUPTED=1; stop_children; echo; echo "[run] stopped."; exit 0; }
trap on_signal INT TERM

echo "[run] starting text-embedding server (log: $SERVER_LOG)..."
set -m
PYTHONUNBUFFERED=1 "$PY" tools/embed_server.py --port "$EMBED_PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
set +m

echo -n "[run] waiting for the encoder to load (first run downloads ~16 GB)"
READY=0
for _ in $(seq 1 360); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo; echo "[run] embed server exited during startup. Last log lines:" >&2
    tail -15 "$SERVER_LOG" >&2
    die "embed server failed"
  fi
  # Readiness requires all three, causally tied to OUR child:
  #   1. the health document says the model is actually loaded;
  #   2. the child is still alive; and
  #   3. the child itself OWNS the listening socket. Health + alive alone
  #      still adopted a foreign server that bound after the pre-check while
  #      our child was loading — the health answer and the PID were never
  #      observations of the same process.
  BODY=$(curl -sf -m 2 "http://127.0.0.1:$EMBED_PORT/health" 2>/dev/null || true)
  if printf '%s' "$BODY" | grep -q '"model_loaded"'; then
    if ! lsof -a -p "$SERVER_PID" -iTCP:"$EMBED_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
      stop_children
      die "port $EMBED_PORT is answering but not owned by the launched server — a foreign process bound it; stop it or change EMBED_PORT"
    fi
  fi
  if printf '%s' "$BODY" | grep -q '"model_loaded": true' && kill -0 "$SERVER_PID" 2>/dev/null; then
    READY=1; echo " ready."; break
  fi
  echo -n "."; sleep 5
done
[ "$READY" = 1 ] || { stop_children; die "embed server did not become healthy within 30 minutes; see $SERVER_LOG"; }

echo "[run] starting app dev server (log: $VITE_LOG)..."
set -m
npm run dev >"$VITE_LOG" 2>&1 &
VITE_PID=$!
set +m
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
