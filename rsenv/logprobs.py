"""Memory-bounded per-token log-probabilities for the GRPO training step.

The naive path materializes the full ``(L, vocab~152k)`` logits tensor from the
``lm_head``, whose size grows linearly with the trajectory length ``L``. On long
GRPO rollouts (``L`` in the tens of thousands) that single tensor is the dominant
memory cost and the source of the training-step OOM -- e.g. at ``L=40960`` it is
``40960 * 151936 * 2B ~= 12 GB`` for one sample.

``per_token_logprobs`` computes the *same* per-token log-probs without ever holding
the full ``(L, vocab)`` tensor: run the transformer trunk once, then tile the
``lm_head`` projection over the sequence axis -- project ``chunk_size`` positions at
a time and reduce each block to its ``(block,)`` log-probs.

Keeping the peak at ``chunk_size * vocab`` requires care in the *grad-carrying*
update: ``logsumexp`` saves its ``(block, vocab)`` logits input for backward, so a
plain loop would keep every block's logits alive until ``.backward()`` and the peak
would climb back to ``L * vocab``. Each block's projection is therefore wrapped in
``torch.utils.checkpoint`` (grad path only) so the logits are recomputed in backward
instead of retained -- peak logits memory is ``chunk_size * vocab``, independent of
``L``, in both the no-grad reference/old passes and the grad-carrying update. It is
pure PyTorch (no extra deps, runs on CPU), numerically identical to the naive path,
and gradient-exact.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch.utils.checkpoint import checkpoint

LOGPROB_CHUNK_SIZE = 1024


def _chunk_logprobs(
    lm_head: torch.nn.Module,
    hidden_chunk: torch.Tensor,  # (c, H)
    targets_chunk: torch.Tensor,  # (c,)
) -> torch.Tensor:  # (c,)
    """Project one block of hidden states to the log-prob of its actual next token.

    Module-level (not inlined) because ``torch.utils.checkpoint`` needs a callable
    to re-invoke in backward; the ``(c, vocab)`` logits exist only inside this call.
    """
    logits = lm_head(hidden_chunk)  # (c, V)
    target_logits = logits.gather(dim=-1, index=rearrange(targets_chunk, "c -> c 1")).squeeze(-1)  # (c,)
    return target_logits - torch.logsumexp(logits, dim=-1)  # (c,)


def per_token_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,  # (B, L)
    attention_mask: torch.Tensor,  # (B, L)
    chunk_size: int = LOGPROB_CHUNK_SIZE,
) -> torch.Tensor:  # (B, L-1)
    """Per-token log-prob of the actual next token, with an L-independent peak.

    Same result as projecting all ``L`` positions at once, but the full-vocab
    logits never exist for more than ``chunk_size`` positions simultaneously --
    including under gradient, via per-block checkpointing.
    """
    trunk = model.base_model  # transformer body without the lm_head
    lm_head = model.get_output_embeddings()

    all_lp = []
    for b in range(input_ids.shape[0]):
        ids_b = rearrange(input_ids[b], "l -> 1 l")
        mask_b = rearrange(attention_mask[b], "l -> 1 l")
        # Run the body once; the (L, vocab) logits are never built here.
        hidden = trunk(input_ids=ids_b, attention_mask=mask_b).last_hidden_state[0]  # (L, H)
        hidden = hidden[:-1]  # (L-1, H): position i predicts token i+1
        targets = input_ids[b, 1:]  # (L-1,)

        # Under gradient, logsumexp saves each block's (c, vocab) logits for backward,
        # so checkpoint recomputes them instead of retaining all L/chunk_size blocks.
        # The no-grad ref/old passes save nothing, so they skip it and pay no recompute.
        use_checkpoint = torch.is_grad_enabled()

        lp_chunks = []
        for s in range(0, hidden.shape[0], chunk_size):
            h = hidden[s : s + chunk_size]  # (c, H)
            tgt = targets[s : s + chunk_size]  # (c,)
            if use_checkpoint:
                lp_chunks.append(checkpoint(_chunk_logprobs, lm_head, h, tgt, use_reentrant=False))
            else:
                lp_chunks.append(_chunk_logprobs(lm_head, h, tgt))

        lp_b = torch.cat(lp_chunks) if lp_chunks else hidden.new_zeros(0)
        all_lp.append(lp_b)

    return torch.stack(all_lp)  # (B, L-1)
