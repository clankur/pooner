"""Chunked per-token log-probs == naive full-logits path (values + gradients).

CPU-only, tiny model. Guards the memory-bounded projection against numeric drift
and confirms it is gradient-exact, so it is a safe drop-in for the GRPO update.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rsenv.logprobs import per_token_logprobs

TINY_MODEL = "sshleifer/tiny-gpt2"


def _naive_logprobs(model, input_ids, attention_mask):
    """The original full-(L, vocab) computation from train.py.get_per_token_logprobs."""
    all_lp = []
    for b in range(input_ids.shape[0]):
        logits = model(input_ids=input_ids[b : b + 1], attention_mask=attention_mask[b : b + 1]).logits[0, :-1, :]
        targets = input_ids[b, 1:]
        target_logits = logits.gather(-1, targets[:, None]).squeeze(-1)
        all_lp.append(target_logits - torch.logsumexp(logits, dim=-1))
    return torch.stack(all_lp)


def _fixture():
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    model.eval()
    ids = tok(["The quick brown fox jumps", "hello there general kenobi"], return_tensors="pt", padding=True)
    return model, ids.input_ids, ids.attention_mask


def test_values_match_naive():
    model, input_ids, attention_mask = _fixture()
    with torch.no_grad():
        naive = _naive_logprobs(model, input_ids, attention_mask)
        # chunk_size=2 forces several blocks over the sequence axis.
        chunked = per_token_logprobs(model, input_ids, attention_mask, chunk_size=2)
    assert chunked.shape == naive.shape
    assert torch.allclose(chunked, naive, atol=1e-4, rtol=1e-4)


def test_single_chunk_matches_multi_chunk():
    model, input_ids, attention_mask = _fixture()
    with torch.no_grad():
        big = per_token_logprobs(model, input_ids, attention_mask, chunk_size=10_000)
        small = per_token_logprobs(model, input_ids, attention_mask, chunk_size=1)
    assert torch.allclose(big, small, atol=1e-5)


def test_gradients_match_naive():
    """Backprop through the chunked path must equal the naive path (gradient-exact)."""
    model, input_ids, attention_mask = _fixture()
    head = model.get_output_embeddings().weight

    model.zero_grad()
    _naive_logprobs(model, input_ids, attention_mask).sum().backward()
    g_naive = head.grad.detach().clone()

    model.zero_grad()
    per_token_logprobs(model, input_ids, attention_mask, chunk_size=2).sum().backward()
    g_chunked = head.grad.detach().clone()

    assert torch.allclose(g_naive, g_chunked, atol=1e-4, rtol=1e-4)


def _count_lm_head_calls(model, run):
    """Return how many times lm_head is invoked while `run(model)` executes+backprops."""
    calls = {"n": 0}
    handle = model.get_output_embeddings().register_forward_pre_hook(lambda *_: calls.__setitem__("n", calls["n"] + 1))
    try:
        run(model)
    finally:
        handle.remove()
    return calls["n"]


def test_grad_path_recomputes_logits_no_grad_does_not():
    """The bound only holds if the grad path recomputes each block's logits in backward.

    Under gradient, checkpointing runs lm_head twice per block (forward + backward
    recompute) so the (block, vocab) logits are freed after forward instead of kept
    alive until .backward() -- that recompute is exactly what keeps the peak at
    chunk_size*vocab instead of L*vocab. The no-grad passes must not pay it.
    """
    model, input_ids, attention_mask = _fixture()
    b0 = input_ids[:1]  # one sample
    m0 = attention_mask[:1]
    # per_token_logprobs projects L-1 positions (it drops the last), chunk_size at a time.
    n_blocks = -(-(b0.shape[1] - 1) // 2)  # ceil((L-1) / 2)

    def _grad_run(m):
        m.zero_grad()
        per_token_logprobs(m, b0, m0, chunk_size=2).sum().backward()

    grad_calls = _count_lm_head_calls(model, _grad_run)

    def _nograd_run(m):
        with torch.no_grad():
            per_token_logprobs(m, b0, m0, chunk_size=2)

    nograd_calls = _count_lm_head_calls(model, _nograd_run)

    assert nograd_calls == n_blocks  # one projection per block, no recompute
    assert grad_calls == 2 * n_blocks  # forward + backward recompute per block
