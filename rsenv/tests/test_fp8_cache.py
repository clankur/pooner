"""Float8Cache: fp8 quantization round-trip error + drop-in generate parity.

CPU-only, no game server needed. Uses a tiny model to check that swapping in the
fp8 KV cache does not change greedy generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rsenv.fp8_cache import (
    _KEY_FP8_DTYPE,
    _KEY_FP8_MAX,
    _VALUE_FP8_DTYPE,
    _VALUE_FP8_MAX,
    Float8Cache,
    _dequantize,
    _quantize,
)

TINY_MODEL = "sshleifer/tiny-gpt2"


def _mean_rel_err(dequant: torch.Tensor, original: torch.Tensor) -> float:
    return ((dequant - original).abs() / original.abs().clamp(min=1e-6)).mean().item()


def test_quant_roundtrip_within_fp8_tolerance():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, 8) * 3.0  # (B, H, S, D)

    qk, sk = _quantize(x, _KEY_FP8_DTYPE, _KEY_FP8_MAX)
    assert qk.dtype == _KEY_FP8_DTYPE
    assert _mean_rel_err(_dequantize(qk, sk, x.dtype), x) < 0.05  # e4m3 (3 mantissa bits)

    qv, sv = _quantize(x, _VALUE_FP8_DTYPE, _VALUE_FP8_MAX)
    assert qv.dtype == _VALUE_FP8_DTYPE
    assert _mean_rel_err(_dequantize(qv, sv, x.dtype), x) < 0.08  # e5m2 (2 mantissa bits)


def test_quant_handles_all_zero_token():
    x = torch.zeros(1, 2, 3, 4)
    qk, sk = _quantize(x, _KEY_FP8_DTYPE, _KEY_FP8_MAX)
    dq = _dequantize(qk, sk, x.dtype)
    assert torch.isfinite(dq).all()
    assert dq.abs().max().item() == 0.0


def test_generate_parity_with_default_cache():
    """Greedy generation must be identical with vs without the fp8 cache."""
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    model.eval()
    ids = tok("The quick brown fox", return_tensors="pt").input_ids

    with torch.no_grad():
        baseline = model.generate(ids, max_new_tokens=20, do_sample=False, use_cache=True)
        with_fp8 = model.generate(
            ids, max_new_tokens=20, do_sample=False, use_cache=True, past_key_values=Float8Cache()
        )

    assert torch.equal(baseline, with_fp8)
