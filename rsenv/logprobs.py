"""Memory-bounded per-token log-probabilities for the GRPO training step.

The naive path materializes the full ``(L, vocab~152k)`` logits tensor from the
``lm_head``, whose size grows linearly with the trajectory length ``L``. On long
GRPO rollouts (``L`` in the tens of thousands) that single tensor is the dominant
memory cost and the source of the training-step OOM -- e.g. at ``L=40960`` it is
``40960 * 151936 * 2B ~= 12 GB`` for one sample.

Two implementations compute the *same* per-token log-probs without ever holding
the full ``(L, vocab)`` tensor:

- ``"chunked"`` (default, pure PyTorch, no extra deps): run the transformer trunk
  once, then tile the ``lm_head`` projection over the sequence axis -- project
  ``chunk_size`` positions at a time, reduce each block to its ``(block,)``
  log-probs, free the ``(block, vocab)`` logits before the next block. Peak logits
  memory is ``chunk_size * vocab``, independent of ``L``. Under autograd each
  block's logits is retained for its own backward, so the grad-carrying update
  still holds ``chunk_size * vocab`` per block.

- ``"fused"`` (Liger Triton kernel): ``LigerFusedLinearCrossEntropyLoss`` fuses the
  ``lm_head`` projection with the cross-entropy reduction so the full-vocab logits
  are **never written to HBM at all**, not even chunk-sized -- and, unlike chunked
  autograd, the vocab-width tensor is never stored for backward either. Per-token
  cross-entropy equals ``-logprob(target)``, so we negate it. This uses Liger's
  standalone loss on ``lm_head.weight`` + hidden states (NOT the model monkeypatch),
  so it is model-agnostic: it needs only ``logits == lm_head(hidden)`` -- the same
  assumption the chunked path already makes -- and does not depend on Liger having
  a registry entry for the specific model architecture.

Both are numerically identical to the naive path and gradient-exact. ``"chunked"``
is the safe default (runs on CPU, no deps); ``"fused"`` is opt-in for GPU runs and
requires ``liger-kernel`` (GPU/Triton only).
"""

from __future__ import annotations

import torch
from einops import rearrange

LOGPROB_CHUNK_SIZE = 1024


def _fused_available() -> bool:
    try:
        import liger_kernel.transformers  # noqa: F401

        return True
    except Exception:
        return False


def _run_trunk(model, ids_b: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    """Transformer body forward for one sample; returns ``(L, H)`` hidden states.

    The ``(L, vocab)`` logits are never built here -- the ``lm_head`` is applied
    separately by the caller (chunked or fused).
    """
    trunk = model.base_model  # transformer body without the lm_head
    hidden = trunk(input_ids=ids_b, attention_mask=mask_b).last_hidden_state[0]  # (L, H)
    return hidden


def per_token_logprobs_chunked(
    model,
    input_ids: torch.Tensor,  # (B, L)
    attention_mask: torch.Tensor,  # (B, L)
    chunk_size: int = LOGPROB_CHUNK_SIZE,
) -> torch.Tensor:  # (B, L-1)
    """Per-token log-prob via a sequence-tiled vocab projection (pure PyTorch)."""
    lm_head = model.get_output_embeddings()

    all_lp = []
    for b in range(input_ids.shape[0]):
        ids_b = rearrange(input_ids[b], "l -> 1 l")
        mask_b = rearrange(attention_mask[b], "l -> 1 l")
        hidden = _run_trunk(model, ids_b, mask_b)  # (L, H)
        hidden = hidden[:-1]  # (L-1, H): position i predicts token i+1
        targets = input_ids[b, 1:]  # (L-1,)

        lp_chunks = []
        for s in range(0, hidden.shape[0], chunk_size):
            h = hidden[s : s + chunk_size]  # (c, H)
            logits = lm_head(h)  # (c, V) -- only this block is materialized
            tgt = targets[s : s + chunk_size]  # (c,)
            target_logits = logits.gather(dim=-1, index=rearrange(tgt, "c -> c 1")).squeeze(-1)  # (c,)
            lp_chunks.append(target_logits - torch.logsumexp(logits, dim=-1))  # (c,)

        lp_b = torch.cat(lp_chunks) if lp_chunks else hidden.new_zeros(0)
        all_lp.append(lp_b)

    return torch.stack(all_lp)  # (B, L-1)


def per_token_logprobs_fused(
    model,
    input_ids: torch.Tensor,  # (B, L)
    attention_mask: torch.Tensor,  # (B, L)
) -> torch.Tensor:  # (B, L-1)
    """Per-token log-prob via Liger's fused-linear-cross-entropy kernel.

    Requires ``liger-kernel`` and a CUDA tensor. Per-token CE = ``-logprob(target)``,
    so we negate the ``reduction="none"`` loss. The full-vocab logits never touch
    HBM -- the projection and reduction are fused inside the Triton kernel.
    """
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    lm_head = model.get_output_embeddings()
    weight = lm_head.weight  # (V, H)
    bias = getattr(lm_head, "bias", None)
    # reduction="none" -> per-token loss; no ignore_index masking so every position
    # (including padding) gets a logprob, matching the chunked path exactly. Padded
    # positions are dropped downstream by generation_mask in grpo_loss.
    flce = LigerFusedLinearCrossEntropyLoss(reduction="none")

    all_lp = []
    for b in range(input_ids.shape[0]):
        ids_b = rearrange(input_ids[b], "l -> 1 l")
        mask_b = rearrange(attention_mask[b], "l -> 1 l")
        hidden = _run_trunk(model, ids_b, mask_b)  # (L, H)
        hidden = hidden[:-1].contiguous()  # (L-1, H)
        targets = input_ids[b, 1:].long()  # (L-1,)
        ce = flce(weight, hidden, targets, bias)  # (L-1,) per-token cross-entropy
        all_lp.append(-ce)  # logprob(target) = -CE

    return torch.stack(all_lp)  # (B, L-1)


def per_token_logprobs(
    model,
    input_ids: torch.Tensor,  # (B, L)
    attention_mask: torch.Tensor,  # (B, L)
    chunk_size: int = LOGPROB_CHUNK_SIZE,
    impl: str = "chunked",
) -> torch.Tensor:  # (B, L-1)
    """Per-token log-prob of the actual next token, with an L-independent peak.

    Args:
        impl: ``"chunked"`` (default, pure-PyTorch sequence tiling), ``"fused"``
            (Liger fused-linear-CE, GPU-only), or ``"auto"`` (fused when the input
            is on CUDA and ``liger-kernel`` imports, else chunked).
    """
    if impl == "auto":
        impl = "fused" if (input_ids.is_cuda and _fused_available()) else "chunked"

    if impl == "fused":
        return per_token_logprobs_fused(model, input_ids, attention_mask)
    if impl == "chunked":
        return per_token_logprobs_chunked(model, input_ids, attention_mask, chunk_size=chunk_size)
    raise ValueError(f"unknown logprob impl: {impl!r} (expected 'chunked', 'fused', or 'auto')")
