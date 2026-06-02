#!/usr/bin/env bash
# Start the LostCity game server (engine + webclient + gateway) in background.
# Gateway listens on ws://localhost:7780 by default.
# Stop with: ./stop-server.sh (or kill the process group)
set -euo pipefail

BRIDGE_DIR="$(cd "$(dirname "$0")" && pwd)"
SDK_DIR="$BRIDGE_DIR/rs-sdk"
LOG_DIR="$BRIDGE_DIR/logs"

mkdir -p "$LOG_DIR"

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

# Start engine
cd "$SDK_DIR/server/engine"
NODE_XPRATE="${XP_MULTIPLIER:-1}" bun run start > "$LOG_DIR/engine.log" 2>&1 &
ENGINE_PID=$!
echo "  Engine started (PID $ENGINE_PID)"

# Start webclient bundler
cd "$SDK_DIR/server/webclient"
bun run watch > "$LOG_DIR/webclient.log" 2>&1 &
WEBCLIENT_PID=$!
echo "  Webclient started (PID $WEBCLIENT_PID)"

# Start gateway
cd "$SDK_DIR/server/gateway"
bun run gateway > "$LOG_DIR/gateway.log" 2>&1 &
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
