#!/usr/bin/env bash
# Clone rs-sdk into bridge/rs-sdk/ if not already present, then install deps.
set -euo pipefail

BRIDGE_DIR="$(cd "$(dirname "$0")" && pwd)"
SDK_DIR="$BRIDGE_DIR/rs-sdk"

if [ ! -d "$SDK_DIR" ]; then
    echo "Cloning rs-sdk..."
    git clone --depth 1 https://github.com/MaxBittker/rs-sdk.git "$SDK_DIR"
else
    echo "rs-sdk already present at $SDK_DIR"
fi

cd "$BRIDGE_DIR"
echo "Installing dependencies..."
bun install

echo "Bridge setup complete. Run with: bun run executor.ts"
