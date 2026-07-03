"""fp8 KV cache for generation, to cut peak memory on long rollouts.

Keys are stored in float8_e4m3fn (3 mantissa bits — precision, since K feeds the
QKᵀ dot product into softmax) and values in float8_e5m2 (2 mantissa bits, wider
exponent range — V is softmax-averaged and more tolerant of outliers). Each token
is scaled by its own per-head absmax before the fp8 cast so it uses the full fp8
range; the scale (fp32) is stored alongside and applied on read.

Why this reduces peak memory even though SDPA has no fp8 attention kernel: the
*persistent* cache holds every layer's K/V in fp8 (~half the bytes), while
attention only ever dequantizes to bf16 **one layer at a time** during the
forward. So the transient bf16 cost is 1/num_layers of the fp8 savings.

Caveat: this treats every layer as full (non-sliding) attention. It is intended
for the plain sampling `generate` path used in rollouts, not beam/contrastive
search. Gate it behind the `model.fp8_kv_cache` config flag and keep the parity
check (greedy tokens with vs without) in the loop when enabling it.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache, DynamicLayer

_KEY_FP8_DTYPE = torch.float8_e4m3fn
_VALUE_FP8_DTYPE = torch.float8_e5m2
_KEY_FP8_MAX = torch.finfo(_KEY_FP8_DTYPE).max  # 448.0
_VALUE_FP8_MAX = torch.finfo(_VALUE_FP8_DTYPE).max  # 57344.0


def _quantize(x: torch.Tensor, fp8_dtype: torch.dtype, fp8_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (per-head) absmax scaling into fp8. x: (B, H, S, D) → (fp8 x, fp32 scale (B, H, S, 1))."""
    scale = x.abs().amax(dim=-1, keepdim=True).float() / fp8_max  # (B, H, S, 1)
    scale = scale.clamp(min=torch.finfo(torch.float32).tiny)  # avoid div-by-zero on all-zero tokens
    q = (x.float() / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return q, scale


def _dequantize(q: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    return (q.to(torch.float32) * scale).to(out_dtype)


class Float8DynamicLayer(DynamicLayer):
    """A dynamic cache layer that stores K/V in fp8 with per-token scales."""

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.compute_dtype = key_states.dtype  # bf16 — what attention consumes
        self.device = key_states.device
        self.keys = torch.empty(0, dtype=_KEY_FP8_DTYPE, device=self.device)
        self.values = torch.empty(0, dtype=_VALUE_FP8_DTYPE, device=self.device)
        self.key_scales = torch.empty(0, dtype=torch.float32, device=self.device)
        self.value_scales = torch.empty(0, dtype=torch.float32, device=self.device)
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        q_keys, key_scale = _quantize(key_states, _KEY_FP8_DTYPE, _KEY_FP8_MAX)
        q_values, value_scale = _quantize(value_states, _VALUE_FP8_DTYPE, _VALUE_FP8_MAX)

        # On first update the empty (0-dim) buffers can't concat against a 4-D tensor.
        if self.keys.numel() == 0:
            self.keys, self.key_scales = q_keys, key_scale
            self.values, self.value_scales = q_values, value_scale
        else:
            self.keys = torch.cat([self.keys, q_keys], dim=-2)
            self.key_scales = torch.cat([self.key_scales, key_scale], dim=-2)
            self.values = torch.cat([self.values, q_values], dim=-2)
            self.value_scales = torch.cat([self.value_scales, value_scale], dim=-2)

        # Return full bf16 K/V for attention (one layer's worth of transient dequant).
        keys = _dequantize(self.keys, self.key_scales, self.compute_dtype)
        values = _dequantize(self.values, self.value_scales, self.compute_dtype)
        return keys, values

    # Keep scales in sync for the (unused in sampling, but cheap to be correct) paths.
    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return
        self.keys = self.keys[..., :max_length, :]
        self.values = self.values[..., :max_length, :]
        self.key_scales = self.key_scales[..., :max_length, :]
        self.value_scales = self.value_scales[..., :max_length, :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() > 0:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)
            self.key_scales = self.key_scales.repeat_interleave(repeats, dim=0)
            self.value_scales = self.value_scales.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() > 0:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]
            self.key_scales = self.key_scales[indices, ...]
            self.value_scales = self.value_scales[indices, ...]


class Float8Cache(Cache):
    """A DynamicCache-style cache whose per-layer K/V live in fp8 (K: e4m3, V: e5m2)."""

    def __init__(self) -> None:
        super().__init__(layer_class_to_replicate=Float8DynamicLayer)
