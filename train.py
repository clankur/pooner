"""GRPO training for RuneScape LLM agent: model loading, loss, training loop, Hydra entrypoint.

Usage:
    RUNQ_EXPERIMENT_ID=local uv run python -m train --config-name=local_test ++paths.model_name=smoke_000
"""

import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass

import hydra
import runq
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from rs_env import Trajectory, load_prompt_bank, rollout_trajectory

# ─── Config dataclasses ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen3-4B-Base"
    quantize_int4: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.8


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 8
    clip_epsilon: float = 0.2
    kl_coeff: float = 0.05
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    update_epochs: int = 2
    max_steps: int = 5000
    log_interval: int = 5
    checkpoint_interval: int = 100
    seed: int = 42


@dataclass(frozen=True)
class EnvConfig:
    max_actions: int = 20
    use_heuristic_reward: bool = True
    game_data_path: str = ""


@dataclass(frozen=True)
class Paths:
    root_working_dir: str = "/tmp"
    model_name: str = "default"


@dataclass(frozen=True)
class Config:
    model: ModelConfig = None
    grpo: GRPOConfig = None
    env: EnvConfig = None
    paths: Paths = None


def build_config(cfg: DictConfig) -> Config:
    return Config(
        model=ModelConfig(**cfg.model),
        grpo=GRPOConfig(**cfg.grpo),
        env=EnvConfig(**cfg.env),
        paths=Paths(**cfg.paths),
    )


# ─── Model loading ─────────────────────────────────────────────────────────


def load_model(config: ModelConfig, device: torch.device) -> tuple:
    """Load model + tokenizer. Apply torchao int4 quantization if configured."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        dtype=torch.bfloat16,
        device_map={"": device} if device.type == "cuda" else None,
    )

    if config.quantize_int4 and device.type == "cuda":
        from torchao.quantization import Int4WeightOnlyConfig, quantize_

        # TILE_PACKED_TO_4D is the tinygemm format — ships with PyTorch, no extra deps.
        # The default PLAIN format requires mslk (Meta's SM90+ kernel library).
        from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat

        quantize_(model, Int4WeightOnlyConfig(int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D))

    if device.type != "cuda":
        model = model.to(device)

    model.gradient_checkpointing_enable()
    return model, tokenizer


def load_ref_model(config: ModelConfig) -> AutoModelForCausalLM:
    """Load a frozen reference model on CPU for KL computation."""
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        dtype=torch.bfloat16,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    return ref_model


# ─── Log-prob computation ──────────────────────────────────────────────────


def get_per_token_logprobs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward pass → per-token log-probability of the actual next token.

    Args:
        input_ids: (B, L) token ids
        attention_mask: (B, L)

    Returns:
        (B, L-1) log-probs for tokens at positions 1..L-1
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]  # (B, L-1, V)
    targets = input_ids[:, 1:]  # (B, L-1)
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_lp = rearrange(log_probs.gather(dim=-1, index=rearrange(targets, "b l -> b l 1")), "b l 1 -> b l")
    return per_token_lp  # (B, L-1)


# ─── GRPO loss ─────────────────────────────────────────────────────────────


@dataclass
class GroupData:
    """Collated data from K trajectories for one GRPO step."""

    full_ids: torch.Tensor  # (K, max_len) padded
    attention_mask: torch.Tensor  # (K, max_len)
    generation_mask: torch.Tensor  # (K, max_len) — 1 for model tokens, 0 for env/pad
    old_log_probs: torch.Tensor  # (K, max_len-1) padded
    ref_log_probs: torch.Tensor  # (K, max_len-1) padded
    advantages: torch.Tensor  # (K,)
    rewards: list[float]


def collate_trajectories(
    trajectories: list[Trajectory],
    ref_model: AutoModelForCausalLM | None,
    device: torch.device,
) -> GroupData:
    """Collate K trajectories into padded tensors, compute advantages and ref log-probs."""
    K = len(trajectories)
    max_len = max(len(t.full_ids) for t in trajectories)

    rewards = torch.tensor([t.total_reward for t in trajectories])
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    pad_id = 0
    full_ids = torch.full((K, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(K, max_len)
    generation_mask = torch.zeros(K, max_len)

    for i, t in enumerate(trajectories):
        seq_len = len(t.full_ids)
        full_ids[i, :seq_len] = t.full_ids
        attention_mask[i, :seq_len] = 1.0
        generation_mask[i, :seq_len] = t.generation_mask

    # old_log_probs is overwritten in the training loop (step 3) before use
    old_lp = torch.zeros(K, max_len - 1)

    # Ref log-probs
    ref_lp = torch.zeros(K, max_len - 1)
    if ref_model is not None:
        with torch.no_grad():
            ref_lp = get_per_token_logprobs(ref_model, full_ids, attention_mask)

    return GroupData(
        full_ids=full_ids.to(device),
        attention_mask=attention_mask.to(device),
        generation_mask=generation_mask[:, 1:].to(device),  # align with log-prob positions
        old_log_probs=old_lp.to(device),
        ref_log_probs=ref_lp.to(device),
        advantages=advantages.to(device),
        rewards=[t.total_reward for t in trajectories],
    )


def grpo_loss(
    new_log_probs: torch.Tensor,  # (K, L-1)
    old_log_probs: torch.Tensor,  # (K, L-1)
    ref_log_probs: torch.Tensor,  # (K, L-1)
    advantages: torch.Tensor,  # (K,)
    generation_mask: torch.Tensor,  # (K, L-1) — only model-generated tokens
    clip_epsilon: float,
    kl_coeff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GRPO clipped surrogate + KL penalty, masked to model-generated tokens only.

    Returns (total_loss, policy_loss, kl_loss) as scalars.
    """
    ratio = torch.exp(new_log_probs - old_log_probs)  # (K, L-1)

    # Broadcast per-trajectory advantage to all tokens
    adv = rearrange(advantages, "k -> k 1").expand_as(ratio)  # (K, L-1)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
    policy_loss_per_token = -torch.min(surr1, surr2)

    kl_per_token = new_log_probs - ref_log_probs

    masked_policy = (policy_loss_per_token * generation_mask).sum()
    masked_kl = (kl_per_token * generation_mask).sum()
    num_tokens = generation_mask.sum().clamp(min=1)

    policy_loss = masked_policy / num_tokens
    kl_loss = masked_kl / num_tokens
    total_loss = policy_loss + kl_coeff * kl_loss

    return total_loss, policy_loss, kl_loss


# ─── Training loop ─────────────────────────────────────────────────────────


@dataclass
class StepMetrics:
    step: int
    group_mean_reward: float
    group_std_reward: float
    group_min_reward: float
    group_max_reward: float
    group_mean_xp: float
    mean_actions: float
    mean_valid_actions: float
    policy_loss: float
    kl_loss: float
    total_loss: float
    elapsed_sec: float


def _report_wandb_url_to_runq(wandb_url: str | None) -> None:
    """If running inside runq, report the wandb URL back via the PATCH API."""
    if wandb_url is None:
        return
    runq_server = os.environ.get("RUNQ_SERVER")
    experiment_id = os.environ.get("RUNQ_EXPERIMENT_ID")
    if not runq_server or not experiment_id:
        return
    try:
        data = json.dumps({"wandb_url": wandb_url}).encode()
        req = urllib.request.Request(
            f"{runq_server}/api/experiments/{experiment_id}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="PATCH",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def train(config: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.grpo.seed)

    print(f"Loading model: {config.model.model_name_or_path} (int4={config.model.quantize_int4})")
    model, tokenizer = load_model(config.model, device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    print("Loading reference model on CPU...")
    ref_model = load_ref_model(config.model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.grpo.learning_rate,
    )

    prompt_bank = load_prompt_bank(seed=config.grpo.seed)
    model_dir = os.path.join(config.paths.root_working_dir, config.paths.model_name)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    wandb_mode = "online" if os.environ.get("RUNQ_EXPERIMENT_ID", "local") != "local" else "disabled"
    wandb.init(project="pooner", name=config.paths.model_name, config=asdict(config), mode=wandb_mode)
    _report_wandb_url_to_runq(wandb.run.get_url() if wandb.run else None)

    metrics_log: list[StepMetrics] = []
    t_start = time.time()

    for step in range(config.grpo.max_steps):
        state = prompt_bank[step % len(prompt_bank)]

        # ── 1. Roll out K trajectories ──
        trajectories: list[Trajectory] = []
        for _k in range(config.grpo.group_size):
            traj = rollout_trajectory(
                model=model,
                tokenizer=tokenizer,
                initial_state=state,
                max_actions=config.env.max_actions,
                max_new_tokens=config.model.max_new_tokens,
                temperature=config.model.temperature,
                device=device,
            )
            trajectories.append(traj)

        # ── 2. Collate + compute advantages ──
        group = collate_trajectories(trajectories, ref_model, device)

        # ── 3. Recompute old log-probs from current model (before update) ──
        model.eval()
        with torch.no_grad():
            old_lp = get_per_token_logprobs(model, group.full_ids, group.attention_mask)
        group.old_log_probs = old_lp

        # ── 4. GRPO update epochs ──
        model.train()
        epoch_losses: list[float] = []
        epoch_policy: list[float] = []
        epoch_kl: list[float] = []

        for _epoch in range(config.grpo.update_epochs):
            new_lp = get_per_token_logprobs(model, group.full_ids, group.attention_mask)

            total, p_loss, kl = grpo_loss(
                new_log_probs=new_lp,
                old_log_probs=group.old_log_probs,
                ref_log_probs=group.ref_log_probs,
                advantages=group.advantages,
                generation_mask=group.generation_mask,
                clip_epsilon=config.grpo.clip_epsilon,
                kl_coeff=config.grpo.kl_coeff,
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grpo.max_grad_norm)
            optimizer.step()

            epoch_losses.append(total.item())
            epoch_policy.append(p_loss.item())
            epoch_kl.append(kl.item())

        # ── 5. Log metrics ──
        rewards = torch.tensor(group.rewards)
        metrics = StepMetrics(
            step=step,
            group_mean_reward=rewards.mean().item(),
            group_std_reward=rewards.std().item(),
            group_min_reward=rewards.min().item(),
            group_max_reward=rewards.max().item(),
            group_mean_xp=sum(t.total_xp for t in trajectories) / len(trajectories),
            mean_actions=sum(t.num_actions for t in trajectories) / len(trajectories),
            mean_valid_actions=sum(t.num_valid_actions for t in trajectories) / len(trajectories),
            policy_loss=sum(epoch_policy) / len(epoch_policy),
            kl_loss=sum(epoch_kl) / len(epoch_kl),
            total_loss=sum(epoch_losses) / len(epoch_losses),
            elapsed_sec=time.time() - t_start,
        )
        metrics_log.append(metrics)

        if step % config.grpo.log_interval == 0:
            print(
                f"[Step {step:>4d}] "
                f"reward={metrics.group_mean_reward:>6.2f} "
                f"xp={metrics.group_mean_xp:>6.1f} "
                f"actions={metrics.mean_actions:>4.1f}/{metrics.mean_valid_actions:>4.1f} "
                f"loss={metrics.total_loss:>8.4f} "
                f"kl={metrics.kl_loss:>7.4f} "
                f"t={metrics.elapsed_sec:>6.1f}s",
                flush=True,
            )
            wandb.log(
                {
                    "group_mean_reward": metrics.group_mean_reward,
                    "group_std_reward": metrics.group_std_reward,
                    "group_mean_xp": metrics.group_mean_xp,
                    "mean_actions": metrics.mean_actions,
                    "mean_valid_actions": metrics.mean_valid_actions,
                    "policy_loss": metrics.policy_loss,
                    "kl_loss": metrics.kl_loss,
                    "total_loss": metrics.total_loss,
                },
                step=step,
            )

        if config.grpo.checkpoint_interval > 0 and step > 0 and step % config.grpo.checkpoint_interval == 0:
            ckpt_path = os.path.join(model_dir, f"checkpoint_{step}.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, ckpt_path)
            print(f"  -> saved {ckpt_path}", flush=True)

    # ── Final save ──
    final_path = os.path.join(model_dir, "checkpoint_final.pt")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, final_path)
    print(f"  -> saved {final_path}", flush=True)

    metrics_path = os.path.join(model_dir, "metrics.jsonl")
    with open(metrics_path, "w") as f:
        for m in metrics_log:
            f.write(json.dumps(asdict(m)) + "\n")

    wandb.finish()
    print(f"Training complete. Final mean reward: {metrics_log[-1].group_mean_reward:.2f}")


# ─── Entrypoint ────────────────────────────────────────────────────────────


@hydra.main(config_path="configs", version_base=None)
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig

    config = build_config(cfg)
    hydra_overrides = [o.split("=")[0].lstrip("+") for o in HydraConfig.get().overrides.task]
    task = runq.Task(project="pooner", name=config.paths.model_name)
    task.execute_remotely(queue="gpu", config=OmegaConf.to_yaml(cfg), overrides=hydra_overrides)
    train(config)


if __name__ == "__main__":
    main()
