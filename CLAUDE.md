# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

pooner is a PyTorch GRPO training codebase for teaching a small LLM (Qwen3 / Gemma 4) to play RuneScape on the LostCity/2004scape server. The model plans then executes: it generates a high-level strategy, then drives action-by-action execution using tool calls, observing game state between actions. The defining design value is **explicitness**: all visible in source rather than hidden behind abstractions.

## Code Standards

### Comments explain _why_, not _what_

Leave a comment only when the next reader would otherwise be confused or repeat a known mistake.

### Type annotations

All functions must have full type signatures. Use dataclasses to bundle related return values instead of bare tuples.

### Einops for shape transforms

Use `einops.rearrange` / `einops.einsum` instead of `.reshape`, `.unsqueeze`, `torch.einsum`. Named dimensions read as documentation.

## Common commands

### Install

Requires Python >= 3.11. CPU dev:

```
uv sync --extra cpu
```

For GPU use `uv sync --extra gpu`.

### Local CPU smoke test

```
RUNQ_EXPERIMENT_ID=local uv run python -m train --config-name=local_test ++paths.model_name=smoke_000
```

### Run on remote GPU (fractal)

```
uv run python -m train --config-name=gpu_4090 ++paths.model_name=my_experiment
```

Without `RUNQ_EXPERIMENT_ID`, this auto-submits to the runq queue via `execute_remotely()`.

### Hydra overrides

```
uv run python -m train --config-name=gpu_4090 ++paths.model_name=grpo_001 ++grpo.learning_rate=3e-5
```

### Lint and format

```
uvx ruff check
uvx ruff format
```

### Docs (Quarto)

```
quarto preview docs
quarto render docs
```

### Tests

There is no test suite. The local test config is the smoke test.

## Architecture

- `train.py` — Model loading (torchao int4), GRPO algorithm, training loop, wandb logging, runq remote execution, Hydra entrypoint. Readable linearly.
- `rs_env.py` — GameState, game knowledge, heuristic simulator, trajectory rollout (plan-then-execute with tool calls), reward computation.
- `configs/` — Hydra YAML configs. `base.yaml` is the schema; other configs inherit and override model/hardware settings.

## Remote infrastructure

- **runq** — Self-hosted experiment queue at https://github.com/clankur/runq. Server runs on fractal (192.168.4.85:8080).
- **fractal** — GPU box (RTX 4090, Ubuntu 22.04, driver 570, CUDA 12.8). SSH alias `fractal`, user `clankur`.
- **wandb** — Metrics logged to `clankur-personal/pooner` project.
