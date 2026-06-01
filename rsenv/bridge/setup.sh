#!/usr/bin/env bash
# Clone rs-sdk, install deps, create a bot account.
set -euo pipefail

BRIDGE_DIR="$(cd "$(dirname "$0")" && pwd)"
SDK_DIR="$BRIDGE_DIR/rs-sdk"

# Clone rs-sdk if needed
if [ ! -d "$SDK_DIR" ]; then
    echo "Cloning rs-sdk..."
    git clone --depth 1 https://github.com/MaxBittker/rs-sdk.git "$SDK_DIR"
else
    echo "rs-sdk already present at $SDK_DIR"
fi

# Install bridge deps
cd "$BRIDGE_DIR"
echo "Installing bridge dependencies..."
bun install

# Install rs-sdk deps
cd "$SDK_DIR"
echo "Installing rs-sdk dependencies..."
bun install

# Fix case-sensitivity issue on Linux (macOS is case-insensitive so it works there)
if [ "$(uname)" = "Linux" ] && [ -f server/webclient/src/io/Jagfile.ts ] && [ ! -e server/webclient/src/io/JagFile.ts ]; then
    ln -sf Jagfile.ts server/webclient/src/io/JagFile.ts
    echo "Fixed Jagfile.ts -> JagFile.ts case mismatch for Linux"
fi

# Create bot accounts if they don't exist
for i in $(seq 1 8); do
    BOT_NAME="grpobot${i}"
    if [ ! -d "$SDK_DIR/bots/$BOT_NAME" ]; then
        echo "Creating bot account: $BOT_NAME (local server)"
        bun bots/create-bot.ts "$BOT_NAME" --local --no-chat
    else
        echo "Bot $BOT_NAME already exists"
    fi
done

if [ ! -d "$SDK_DIR/bots/monitor1" ]; then
    echo "Creating bot account: monitor1 (local server)"
    bun bots/create-bot.ts monitor1 --local --no-chat
else
    echo "Bot monitor1 already exists"
fi

echo ""
echo "Setup complete."
echo "  grpobot1..grpobot8 — parallel training agents"
echo "  monitor1           — your observer account"
echo ""
echo "To start the local game server, run:"
echo "  ./start-server.sh"
