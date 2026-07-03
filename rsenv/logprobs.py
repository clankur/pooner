"""Memory-bounded per-token log-probabilities via a chunked vocab projection.

The naive path materializes the full ``(L, vocab~152k)`` logits tensor from the
``lm_head``, whose size grows linearly with the trajectory length ``L``. On long
GRPO rollouts (``L`` in the tens of thousands) that single tensor is the dominant
memory cost and the source of the training-step OOM -- e.g. at ``L=40960`` it is
``40960 * 151936 * 2B ~= 12 GB`` for one sample.

This computes the *same* per-token log-probs but tiles the ``lm_head`` projection
over the sequence axis: run the transformer trunk once to get hidden states, then
project ``chunk_size`` positions at a time, reduce each block to its ``(block,)``
log-probs, and free the ``(block, vocab)`` logits before the next block. Peak
logits memory is ``chunk_size * vocab`` -- independent of ``L``. It is numerically
identical to the naive path and gradient-exact, so it drops into both the no-grad
reference/old-logprob passes and the grad-carrying GRPO update.
"""

from __future__ import annotations

import torch
from einops import rearrange

LOGPROB_CHUNK_SIZE = 1024


def per_token_logprobs(
    model,
    input_ids: torch.Tensor,  # (B, L)
    attention_mask: torch.Tensor,  # (B, L)
    chunk_size: int = LOGPROB_CHUNK_SIZE,
) -> torch.Tensor:  # (B, L-1)
    """Per-token log-prob of the actual next token, with an L-independent peak.

    Same result as projecting all ``L`` positions at once, but the full-vocab
    logits never exist for more than ``chunk_size`` positions simultaneously.
    """
    trunk = model.base_model  # transformer body without the lm_head
    lm_head = model.get_output_embeddings()

    all_lp = []
    for b in range(input_ids.shape[0]):
        ids_b = rearrange(input_ids[b], "l -> 1 l")
        mask_b = rearrange(attention_mask[b], "l -> 1 l")
        # Run the body once; the (L, V) logits are never built here.
        hidden = trunk(input_ids=ids_b, attention_mask=mask_b).last_hidden_state[0]  # (L, H)
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
