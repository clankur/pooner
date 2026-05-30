"""Evaluate a trained GRPO agent: run trajectories and inspect plan quality.

Usage:
    uv run python eval.py --checkpoint /tmp/smoke_000/checkpoint_final.pt --config-name=local_test
    uv run python eval.py --checkpoint /tmp/qwen4b_001/checkpoint_200.pt --config-name=gpu_4090 --num-episodes 20
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rs_env import (
    Trajectory,
    format_state,
    load_prompt_bank,
    rollout_trajectory,
)


def load_checkpoint(
    checkpoint_path: str,
    model_name_or_path: str,
    device: torch.device,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model from checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device} if device.type == "cuda" else None,
    )

    if device.type != "cuda":
        model = model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    step = ckpt.get("step", "?")
    print(f"Loaded checkpoint from step {step}: {checkpoint_path}")

    return model, tokenizer


def print_trajectory(traj: Trajectory, tokenizer: AutoTokenizer, idx: int) -> None:
    """Print a single trajectory's full text."""
    text = tokenizer.decode(traj.full_ids, skip_special_tokens=False)

    print(f"\n{'=' * 80}")
    print(f"Trajectory {idx}")
    print(f"{'=' * 80}")
    print(f"Actions: {traj.num_actions} ({traj.num_valid_actions} valid)")
    print(f"XP gained: {traj.total_xp:.0f}")
    print(f"Reward: {traj.total_reward:.2f}")
    print("Final state:")
    print(f"  {format_state(traj.final_state)}")
    print(f"{'─' * 80}")
    print(text)
    print(f"{'=' * 80}\n")


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    num_episodes: int,
    max_actions: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    verbose: bool = True,
) -> list[dict]:
    """Run evaluation episodes and return metrics."""
    prompt_bank = load_prompt_bank(seed=123)
    results: list[dict] = []

    for i in range(num_episodes):
        state = prompt_bank[i % len(prompt_bank)]
        traj = rollout_trajectory(
            model=model,
            tokenizer=tokenizer,
            initial_state=state,
            max_actions=max_actions,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )

        result = {
            "episode": i,
            "num_actions": traj.num_actions,
            "num_valid_actions": traj.num_valid_actions,
            "total_xp": traj.total_xp,
            "reward": traj.total_reward,
            "validity_rate": traj.num_valid_actions / max(traj.num_actions, 1),
        }
        results.append(result)

        if verbose:
            print_trajectory(traj, tokenizer, i)

    # Summary
    print(f"\n{'=' * 80}")
    print(f"EVALUATION SUMMARY ({num_episodes} episodes)")
    print(f"{'=' * 80}")
    rewards = [r["reward"] for r in results]
    xps = [r["total_xp"] for r in results]
    actions = [r["num_actions"] for r in results]
    valid = [r["num_valid_actions"] for r in results]

    print(f"Mean reward:     {sum(rewards) / len(rewards):.2f}  (min={min(rewards):.2f}, max={max(rewards):.2f})")
    print(f"Mean XP gained:  {sum(xps) / len(xps):.1f}  (min={min(xps):.1f}, max={max(xps):.1f})")
    print(f"Mean actions:    {sum(actions) / len(actions):.1f}  (valid: {sum(valid) / len(valid):.1f})")
    validity = [r["validity_rate"] for r in results]
    print(f"Validity rate:   {sum(validity) / len(validity):.1%}")
    print(f"{'=' * 80}\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained GRPO RuneScape agent")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--model", default=None, help="Model name/path (reads from config.json if not specified)")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max-actions", type=int, default=20, help="Max actions per trajectory")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens per generation turn")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (lower = more greedy)")
    parser.add_argument("--quiet", action="store_true", help="Only print summary, not full trajectories")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try to read model name from config.json in the same directory as checkpoint
    model_name = args.model
    if model_name is None:
        config_path = os.path.join(os.path.dirname(args.checkpoint), "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            model_name = config["model"]["model_name_or_path"]
            print(f"Using model from config: {model_name}")
        else:
            print("No --model specified and no config.json found. Specify --model.", file=sys.stderr)
            sys.exit(1)

    model, tokenizer = load_checkpoint(args.checkpoint, model_name, device)
    model.eval()

    results = evaluate(
        model=model,
        tokenizer=tokenizer,
        num_episodes=args.num_episodes,
        max_actions=args.max_actions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
        verbose=not args.quiet,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
