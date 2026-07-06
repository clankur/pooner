#!/usr/bin/env bash
# Start the LostCity game server (engine + webclient + gateway) in background.
# Gateway listens on ws://localhost:7780 by default.
# Stop with: ./stop-server.sh (or kill the process group)
set -euo pipefail

BRIDGE_DIR="$(cd "$(dirname "$0")" && pwd)"
SDK_DIR="$BRIDGE_DIR/rs-sdk"
LOG_DIR="$BRIDGE_DIR/logs"

mkdir -p "$LOG_DIR"

# bun installs to ~/.bun/bin, which is NOT on the PATH of a non-interactive SSH session
# (e.g. `ssh fractal rsenv/bridge/start-server.sh`). Without this every `bun` call below
# dies with "command not found" and the server silently never comes up — which is exactly
# how the game env ended up down and broke live training runs.
if ! command -v bun >/dev/null 2>&1; then
    export PATH="$HOME/.bun/bin:$PATH"
fi
if ! command -v bun >/dev/null 2>&1; then
    echo "bun not found on PATH or in \$HOME/.bun/bin. Install bun (https://bun.sh) first." >&2
    exit 1
fi

if [ ! -d "$SDK_DIR/server" ]; then
    echo "rs-sdk not found. Run ./setup.sh first."
    exit 1
fi

# Ensure server subdependencies are installed
for subdir in engine webclient gateway; do
    if [ ! -d "$SDK_DIR/server/$subdir/node_modules" ]; then
        echo "Installing $subdir dependencies..."
        (cd "$SDK_DIR/server/$subdir" && bun install)
    fi
done

# Check if already running
if [ -f "$LOG_DIR/server.pid" ] && kill -0 "$(cat "$LOG_DIR/server.pid")" 2>/dev/null; then
    echo "Server already running (PID $(cat "$LOG_DIR/server.pid"))"
    exit 0
fi

echo "Starting LostCity game server..."

# nohup + </dev/null detaches each service from the controlling terminal so it survives the
# SSH session closing. Plain `&` leaves them in the login shell's process group, which gets
# SIGHUP'd (and killed) on disconnect — so a run started over SSH would die moments later.
# nohup keeps $! pointing at the real process, so stop-server.sh's PID tracking still works.

# Start engine. NODE_DEBUG_SOCKET=true switches the server into its "relaxed for
# bots" mode (World.ts): the no-connection/no-response logout timeouts jump from
# 500/1000 ticks to 60000, so bots aren't idle-kicked back to the default spawn
# while they wait their turn on the shared GPU during a rollout (which silently
# undid every skilling teleport and pinned XP at 0 — see exp 219).
cd "$SDK_DIR/server/engine"
nohup env NODE_XPRATE="${XP_MULTIPLIER:-1}" NODE_DEBUG_SOCKET=true bun run start > "$LOG_DIR/engine.log" 2>&1 </dev/null &
ENGINE_PID=$!
echo "  Engine started (PID $ENGINE_PID)"

# Start webclient bundler
cd "$SDK_DIR/server/webclient"
nohup bun run watch > "$LOG_DIR/webclient.log" 2>&1 </dev/null &
WEBCLIENT_PID=$!
echo "  Webclient started (PID $WEBCLIENT_PID)"

# Start gateway
cd "$SDK_DIR/server/gateway"
nohup bun run gateway > "$LOG_DIR/gateway.log" 2>&1 </dev/null &
GATEWAY_PID=$!
echo "  Gateway started (PID $GATEWAY_PID)"

# Save PIDs for stop script
echo "$ENGINE_PID $WEBCLIENT_PID $GATEWAY_PID" > "$LOG_DIR/server.pid"

echo ""
echo "Server running. Gateway at ws://localhost:7780"
echo "Logs in $LOG_DIR/"
echo "Stop with: ./stop-server.sh"

# Wait a moment for gateway to be ready
sleep 3
if kill -0 "$GATEWAY_PID" 2>/dev/null; then
    echo "Gateway is up."
else
    echo "WARNING: Gateway process exited. Check $LOG_DIR/gateway.log"
fi
