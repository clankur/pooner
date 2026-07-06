"""GRPO training for RuneScape LLM agent: model loading, loss, training loop, Hydra entrypoint.

Usage:
    RUNQ_EXPERIMENT_ID=local uv run python -m train --config-name=local_test ++paths.model_name=smoke_000
"""

import json
import logging
import math
import os
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import hydra
import runq
import torch
import wandb
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from rsenv import (
    BridgeClient,
    BridgeClientPool,
    GenerationService,
    SimClient,
    Trajectory,
    load_prompt_bank,
    random_starting_state,
    rollout_trajectory,
)
from rsenv.logprobs import per_token_logprobs

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

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
    lr_max: float = 1e-5
    lr_min: float = 1e-6
    warmup_steps: int = 10
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
    gateway_url: str = "ws://localhost:7780"
    bot_username: str = "grpo_agent"
    bot_password: str = ""
    num_bots: int = 1
    xp_multiplier: int = 1


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


def load_model(config: ModelConfig, device: torch.device) -> tuple[Qwen3_5ForConditionalGeneration, AutoProcessor]:
    """Load Qwen3.5 multimodal model + processor. Apply torchao int4 quantization if configured."""
    processor = AutoProcessor.from_pretrained(config.model_name_or_path)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device} if device.type == "cuda" else None,
    )

    if config.quantize_int4:
        from torchao.quantization.qat import Int4WeightOnlyQATQuantizer

        qat = Int4WeightOnlyQATQuantizer(groupsize=128)
        model = qat.prepare(model)

    if device.type != "cuda":
        model = model.to(device)

    model.gradient_checkpointing_enable()
    return model, processor


def load_ref_model(config: ModelConfig) -> Qwen3_5ForConditionalGeneration:
    """Load a frozen reference model on CPU. Swapped to GPU briefly each step for fast logprob computation."""
    ref_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    return ref_model


def compute_ref_logprobs(
    ref_model: Qwen3_5ForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Swap ref model to GPU, compute logprobs, swap back. Returns cached tensor on device."""
    ref_model.to(device)
    with torch.no_grad():
        ref_lp = get_per_token_logprobs(ref_model, input_ids, attention_mask)
    ref_model.to("cpu")
    torch.cuda.empty_cache()
    return ref_lp  # stays on device


# ─── Log-prob computation ──────────────────────────────────────────────────


def get_per_token_logprobs(
    model: Qwen3_5ForConditionalGeneration,
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
    # Memory-bounded vocab projection: peak logits memory is independent of L.
    # See rsenv/logprobs.py.
    return per_token_logprobs(model, input_ids, attention_mask)


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
    ref_model: Qwen3_5ForConditionalGeneration | None,
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

    # Ref log-probs: swap ref model to GPU, compute, swap back
    if ref_model is not None:
        ref_lp = compute_ref_logprobs(ref_model, full_ids.to(device), attention_mask.to(device), device)
    else:
        ref_lp = torch.zeros(K, max_len - 1, device=device)

    return GroupData(
        full_ids=full_ids.to(device),
        attention_mask=attention_mask.to(device),
        generation_mask=generation_mask[:, 1:].to(device),  # align with log-prob positions
        old_log_probs=old_lp.to(device),
        ref_log_probs=ref_lp,
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


# ─── LR schedule ──────────────────────────────────────────────────────────


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    grpo_config: GRPOConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup for warmup_steps, then cosine decay to lr_min over remaining steps."""
    warmup = grpo_config.warmup_steps
    total = grpo_config.max_steps
    lr_max = grpo_config.lr_max
    lr_min = grpo_config.lr_min
    min_ratio = lr_min / lr_max if lr_max > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup:
            frac = (step + 1) / max(warmup, 1)
            return min_ratio + (1.0 - min_ratio) * frac
        progress = (step - warmup) / max(total - warmup, 1)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Training loop ─────────────────────────────────────────────────────────


@dataclass
class StepMetrics:
    step: int
    learning_rate: float
    group_mean_reward: float
    group_std_reward: float
    group_min_reward: float
    group_max_reward: float
    reward_components: dict[str, float]  # group mean of each compute_reward component
    group_mean_xp: float
    mean_actions: float
    mean_valid_actions: float
    mean_level_ups: float
    mean_tokens_per_action: float
    idle_count: int
    policy_loss: float
    kl_loss: float
    total_loss: float
    elapsed_sec: float
    gen_peak_mem_gb: float = 0.0  # peak CUDA allocation during rollout generation
    train_peak_mem_gb: float = 0.0  # peak CUDA allocation during the GRPO update


def _reset_peak_mem(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_mem_gb(device: torch.device) -> float:
    """Peak CUDA bytes allocated since the last reset, in GB (0 on CPU)."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


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
    model, processor = load_model(config.model, device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    if config.grpo.kl_coeff > 0:
        print("Loading reference model on CPU...")
        ref_model = load_ref_model(config.model)
    else:
        print("KL disabled (kl_coeff=0), skipping reference model")
        ref_model = None

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.grpo.lr_max,
    )
    scheduler = build_lr_scheduler(optimizer, config.grpo)

    # All rollout generation funnels through one service thread that batches
    # concurrent requests into a single left-padded model.generate call.
    gen_service = GenerationService(
        model=model,
        tokenizer=processor.tokenizer,
        max_new_tokens=config.model.max_new_tokens,
        temperature=config.model.temperature,
        device=device,
    )
    gen_service.start()

    # Game client: live server or heuristic simulator
    client: SimClient | BridgeClient | None = None
    client_pool: BridgeClientPool | None = None

    if not config.env.use_heuristic_reward:
        if config.env.num_bots > 1:
            print(f"Starting {config.env.num_bots} bridge clients...")
            client_pool = BridgeClientPool(
                num_clients=config.env.num_bots,
                gateway_url=config.env.gateway_url,
            )
            client_pool.start_all()
            print(f"All {config.env.num_bots} bridge clients connected")
        else:
            print(f"Starting bridge client: {config.env.gateway_url} as {config.env.bot_username}")
            client = BridgeClient(
                gateway_url=config.env.gateway_url,
                bot_username=config.env.bot_username,
                bot_password=config.env.bot_password,
            )
            initial_state = client.start()
            print(
                f"Bridge connected. Position: {initial_state.position}, HP: {initial_state.hp}/{initial_state.max_hp}"
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
    state_rng = random.Random(config.grpo.seed)

    for step in range(config.grpo.max_steps):
        state = prompt_bank[step % len(prompt_bank)]

        # ── 1. Roll out K trajectories ──
        trajectories: list[Trajectory] = []

        _reset_peak_mem(device)  # measure generation-phase peak

        if client_pool is not None:
            # Randomize starting state for variance across steps
            reset_target = random_starting_state(state_rng)
            client_pool.reset_all(reset_target)
            K = min(config.grpo.group_size, len(client_pool))

            # Each thread drives its own bot at its own pace; their generate
            # calls coalesce into batched GPU work inside gen_service.
            def _run_rollout(k: int) -> Trajectory:
                return rollout_trajectory(
                    generation=gen_service,
                    processor=processor,
                    initial_state=reset_target,
                    max_actions=config.env.max_actions,
                    client=client_pool[k],
                    xp_multiplier=config.env.xp_multiplier,
                )

            with ThreadPoolExecutor(max_workers=K) as executor:
                futures = [executor.submit(_run_rollout, k) for k in range(K)]
                trajectories = [f.result() for f in futures]
        else:
            for _k in range(config.grpo.group_size):
                traj = rollout_trajectory(
                    generation=gen_service,
                    processor=processor,
                    initial_state=state,
                    max_actions=config.env.max_actions,
                    client=client,
                    xp_multiplier=config.env.xp_multiplier,
                )
                trajectories.append(traj)

        gen_peak_mem_gb = _peak_mem_gb(device)

        # Free generation KV cache before training
        torch.cuda.empty_cache() if device.type == "cuda" else None
        _reset_peak_mem(device)  # measure training-phase peak (collate + logprobs + update)

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
            # Accumulate gradients one sample at a time to fit in VRAM
            optimizer.zero_grad()
            total_accum = 0.0
            policy_accum = 0.0
            kl_accum = 0.0
            K = group.full_ids.shape[0]

            for b in range(K):
                ids_b = rearrange(group.full_ids[b], "l -> 1 l")
                mask_b = rearrange(group.attention_mask[b], "l -> 1 l")
                new_lp_b = per_token_logprobs(model, ids_b, mask_b)[0]

                total_b, p_b, kl_b = grpo_loss(
                    new_log_probs=rearrange(new_lp_b, "l -> 1 l"),
                    old_log_probs=rearrange(group.old_log_probs[b], "l -> 1 l"),
                    ref_log_probs=rearrange(group.ref_log_probs[b], "l -> 1 l"),
                    advantages=rearrange(group.advantages[b], " -> 1"),
                    generation_mask=rearrange(group.generation_mask[b], "l -> 1 l"),
                    clip_epsilon=config.grpo.clip_epsilon,
                    kl_coeff=config.grpo.kl_coeff,
                )
                (total_b / K).backward()
                total_accum += total_b.item() / K
                policy_accum += p_b.item() / K
                kl_accum += kl_b.item() / K

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grpo.max_grad_norm)
            optimizer.step()

            epoch_losses.append(total_accum)
            epoch_policy.append(policy_accum)
            epoch_kl.append(kl_accum)

        train_peak_mem_gb = _peak_mem_gb(device)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # ── 5. Log metrics ──
        rewards = torch.tensor(group.rewards)
        total_actions_sum = sum(t.num_actions for t in trajectories)
        reward_components = {
            name: sum(t.reward_metrics[name] for t in trajectories) / len(trajectories)
            for name in trajectories[0].reward_metrics
        }
        metrics = StepMetrics(
            step=step,
            learning_rate=current_lr,
            group_mean_reward=rewards.mean().item(),
            group_std_reward=rewards.std().item(),
            group_min_reward=rewards.min().item(),
            group_max_reward=rewards.max().item(),
            reward_components=reward_components,
            group_mean_xp=sum(t.total_xp for t in trajectories) / len(trajectories),
            mean_actions=total_actions_sum / len(trajectories),
            mean_valid_actions=sum(t.num_valid_actions for t in trajectories) / len(trajectories),
            mean_level_ups=sum(t.num_level_ups for t in trajectories) / len(trajectories),
            mean_tokens_per_action=(sum(t.total_gen_tokens for t in trajectories) / max(total_actions_sum, 1)),
            idle_count=sum(1 for t in trajectories if t.total_xp == 0 and t.num_actions > 0),
            policy_loss=sum(epoch_policy) / len(epoch_policy),
            kl_loss=sum(epoch_kl) / len(epoch_kl),
            total_loss=sum(epoch_losses) / len(epoch_losses),
            elapsed_sec=time.time() - t_start,
            gen_peak_mem_gb=gen_peak_mem_gb,
            train_peak_mem_gb=train_peak_mem_gb,
        )
        metrics_log.append(metrics)

        if step % config.grpo.log_interval == 0:
            log_dict = {
                "group_mean_reward": metrics.group_mean_reward,
                "group_std_reward": metrics.group_std_reward,
                "group_mean_xp": metrics.group_mean_xp,
                "mean_actions": metrics.mean_actions,
                "mean_valid_actions": metrics.mean_valid_actions,
                "mean_level_ups": metrics.mean_level_ups,
                "mean_tokens_per_action": metrics.mean_tokens_per_action,
                "idle_count": metrics.idle_count,
                "policy_loss": metrics.policy_loss,
                "kl_loss": metrics.kl_loss,
                "total_loss": metrics.total_loss,
                "learning_rate": metrics.learning_rate,
                "elapsed_sec": metrics.elapsed_sec,
                "gen_peak_mem_gb": metrics.gen_peak_mem_gb,
                "train_peak_mem_gb": metrics.train_peak_mem_gb,
                **{f"reward/{name}": value for name, value in metrics.reward_components.items()},
            }
            printable = {name: round(value, 4) for name, value in log_dict.items()}
            print(f"[Step {step:>4d}] {printable}", flush=True)
            wandb.log(log_dict, step=step)

        if config.grpo.checkpoint_interval > 0 and step > 0 and step % config.grpo.checkpoint_interval == 0:
            ckpt_path = os.path.join(model_dir, f"checkpoint_{step}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                },
                ckpt_path,
            )
            print(f"  -> saved {ckpt_path}", flush=True)

    # ── Final save ──
    final_path = os.path.join(model_dir, "checkpoint_final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        final_path,
    )
    print(f"  -> saved {final_path}", flush=True)

    metrics_path = os.path.join(model_dir, "metrics.jsonl")
    with open(metrics_path, "w") as f:
        for m in metrics_log:
            f.write(json.dumps(asdict(m)) + "\n")

    gen_service.stop()

    if client_pool is not None:
        client_pool.stop_all()
        print("Bridge client pool stopped")
    elif isinstance(client, BridgeClient):
        client.stop()
        print("Bridge client stopped")

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
