#!/usr/bin/env bash
# Start the full local route: text-embedding server + app dev server.
#
# Requires a completed ./setup.sh (or an equivalent hand-built environment;
# set KIMODO_ROOT / EMBED_PORT to override defaults). Loads the ~16 GB LLM2Vec
# encoder at startup — the first run downloads it from Hugging Face, later
# runs load from cache in a couple of minutes. Ctrl-C stops both servers.
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

cleanup() {
  trap - INT TERM EXIT
  [ -n "${VITE_PID:-}" ]   && kill "$VITE_PID"   2>/dev/null || true
  [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null || true
  echo; echo "[run] stopped."
}
trap cleanup INT TERM EXIT

echo "[run] starting text-embedding server (log: $SERVER_LOG)..."
PYTHONUNBUFFERED=1 "$PY" tools/embed_server.py --port "$EMBED_PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo -n "[run] waiting for the encoder to load (first run downloads ~16 GB)"
for _ in $(seq 1 360); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo; echo "[run] embed server exited during startup. Last log lines:" >&2
    tail -15 "$SERVER_LOG" >&2
    die "embed server failed"
  fi
  if curl -sf -m 2 "http://127.0.0.1:$EMBED_PORT/health" 2>/dev/null | grep -q model_loaded; then
    echo " ready."; break
  fi
  echo -n "."; sleep 5
done
curl -sf -m 2 "http://127.0.0.1:$EMBED_PORT/health" >/dev/null \
  || die "embed server did not become healthy within 30 minutes; see $SERVER_LOG"

echo "[run] starting app dev server (log: $VITE_LOG)..."
npm run dev >"$VITE_LOG" 2>&1 &
VITE_PID=$!
URL=""
for _ in $(seq 1 60); do
  URL=$(grep -oE 'http://localhost:[0-9]+/' "$VITE_LOG" | head -1 || true)
  [ -n "$URL" ] && break
  kill -0 "$VITE_PID" 2>/dev/null || { tail -15 "$VITE_LOG" >&2; die "vite failed to start"; }
  sleep 1
done
[ -n "$URL" ] && echo "[run] app: $URL  (embedding server on :$EMBED_PORT)" \
              || die "vite started but printed no URL; see $VITE_LOG"
echo "[run] open the URL, type a prompt, click Generate. Ctrl-C stops everything."
wait
