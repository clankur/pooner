#!/usr/bin/env bash
# Stop the LostCity game server started by start-server.sh.
set -euo pipefail

BRIDGE_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$BRIDGE_DIR/logs/server.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No server PID file found. Server may not be running."
    exit 0
fi

PIDS=$(cat "$PID_FILE")
for pid in $PIDS; do
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null
        echo "Stopped PID $pid"
    fi
done

rm -f "$PID_FILE"
echo "Server stopped."
