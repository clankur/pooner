# pooner

GRPO-trained LLM agent for RuneScape on the [LostCity/2004scape](https://2004.lostcity.rs/) server.

The model plans then executes: it generates a high-level strategy, then drives action-by-action execution using tool calls, observing game state between actions. Trained with [GRPO](https://arxiv.org/abs/2402.03300) (Group Relative Policy Optimization) — no critic network, no reward model.

## Prerequisites

- Python >= 3.11, [uv](https://docs.astral.sh/uv/)
- [Bun](https://bun.sh/) (for the rs-sdk game bridge)

## Setup

```bash
# Install Python deps (CPU for local dev, GPU for training)
uv sync --extra cpu    # or: uv sync --extra gpu

# Set up the game bridge (clones rs-sdk, installs deps, creates bot accounts)
cd rsenv/bridge
./setup.sh
```

Setup creates two bot accounts:
- **grpobot1** — the GRPO training agent
- **monitor1** — observer account (log in via the web client to watch)

## Running the game server

The LostCity game server runs locally via [rs-sdk](https://github.com/MaxBittker/rs-sdk). Three components: game engine, web client, and gateway.

```bash
# Start all three in background
cd rsenv/bridge
./start-server.sh

# Gateway at ws://localhost:7780
# Game client at http://localhost:8888/rs2.cgi (log in as monitor1 to watch)
# Stop with:
./stop-server.sh
```

Logs are in `rsenv/bridge/logs/`. To adjust game speed, edit `TICK_RATE` in `rsenv/bridge/rs-sdk/server/engine/.env` (default 600ms, lower = faster).

The bridge auto-launches a browser for the bot client and waits for it to enter the game world. Credentials are loaded from `rsenv/bridge/rs-sdk/bots/grpobot1/bot.env`.

## Running tests

```bash
# Unit test (SimClient, no server needed)
uv run pytest tests/ -k sim -v

# E2E test (BridgeClient, needs running game server)
cd rsenv/bridge && ./start-server.sh && cd ../..
uv run pytest tests/ -v -s --tb=long
```

## Training

```bash
# Local CPU smoke test (tiny model, verifies pipeline)
RUNQ_EXPERIMENT_ID=local uv run python -m train --config-name=smoke_test ++paths.model_name=smoke_000

# Submit to GPU via runq
uv run python -m train --config-name=gpu_4090 ++paths.model_name=run_001

# Hydra overrides
uv run python -m train --config-name=gpu_4090 ++paths.model_name=run_002 ++grpo.learning_rate=3e-5
```

### Configs

| Config | Model | VRAM | Notes |
|--------|-------|------|-------|
| `smoke_test` | tiny-gpt2 | ~0 | Pipeline check, <1s |
| `local_test` | Qwen3.5-0.8B | ~2GB | CPU smoke test |
| `gpu_4090_0.8b` | Qwen3.5-0.8B | ~4GB | Fast GPU iteration |
| `gpu_4090` | Qwen3.5-4B | ~18GB | Production |
| `gpu_a100_8b` | Qwen3.5-9B | ~36GB | Needs A100 80GB |

## Evaluation

```bash
# Evaluate a checkpoint (reads model name from config.json next to checkpoint)
uv run python eval.py --checkpoint /tmp/run_001/checkpoint_200.pt --num-episodes 10

# Greedy decoding, quiet mode
uv run python eval.py --checkpoint /tmp/run_001/checkpoint_final.pt --temperature 0.1 --quiet
```

## Architecture

```
train.py              # GRPO training loop, model loading, loss computation
eval.py               # Evaluate checkpoints, print trajectories
rsenv/                # Game environment package (black box)
  __init__.py          # Public API
  state.py             # GameState, Trajectory, XP mechanics
  tools.py             # Pydantic tool models, schema generation, parsing
  client.py            # RSClient ABC → SimClient (offline) / BridgeClient (live)
  env.py               # Rollout, prompt formatting, prompt bank
  bridge/              # TypeScript bridge to rs-sdk
    executor.ts         # JSON-over-stdin/stdout protocol
    start-server.sh     # Start local game server
    stop-server.sh      # Stop local game server
    setup.sh            # Clone rs-sdk, install deps, create bot
  prompts/
    system.md           # System prompt (loaded at runtime)
configs/               # Hydra YAML configs
```

## Infrastructure

- **runq** — Self-hosted experiment queue ([github.com/clankur/runq](https://github.com/clankur/runq))
- **wandb** — Metrics at [wandb.ai/clankur-personal/pooner](https://wandb.ai/clankur-personal/pooner)
- **fractal** — GPU box (RTX 4090), SSH alias `fractal`
