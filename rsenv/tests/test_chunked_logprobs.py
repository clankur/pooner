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
