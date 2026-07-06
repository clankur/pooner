"""From-scratch LoRA: a low-rank adapter on a frozen base Linear.

Kept out of `train.py` so the adapter math is visible in one place. No `peft` — the
whole point of this codebase is that nothing is hidden behind a library abstraction.

Design notes:
- The base Linear is frozen and (optionally) FP8-weight-only quantized elsewhere. This
  module never touches the base weight, so it composes with a torchao-quantized `base`.
- `B` is zero-initialized so a freshly-wrapped model is numerically identical to the base
  (the adapter starts as a no-op), which keeps early GRPO steps well-behaved.
- `enabled` toggles the low-rank path. Disabling it on every adapter recovers the base
  policy, which is how we get KL reference log-probs without a separate frozen model.
"""

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn
from einops import einsum

DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` with a trainable rank-`r` update: y = base(x) + (alpha/r)·B·A·x."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        dtype = base.weight.dtype
        device = base.weight.device
        # A: (r, in), B: (out, r). Match the base compute dtype so the add stays in one dtype.
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=dtype, device=device))
        nn.init.normal_(self.lora_A, std=1.0 / r)

        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        h = einsum(self.dropout(x), self.lora_A, "... d_in, r d_in -> ... r")
        update = einsum(h, self.lora_B, "... r, d_out r -> ... d_out")
        return out + update * self.scaling


def apply_lora(
    model: nn.Module,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES,
) -> int:
    """Freeze the whole model, then replace every target `nn.Linear` with a `LoRALinear`.

    Returns the number of modules wrapped. After this call only the adapter params require grad,
    so the existing `[p for p in model.parameters() if p.requires_grad]` optimizer filter picks
    up exactly the adapters.
    """
    for p in model.parameters():
        p.requires_grad = False

    targets = set(target_modules)
    # Collect first, then mutate — replacing during named_modules() iteration is unsafe.
    to_wrap: list[tuple[nn.Module, str, nn.Linear]] = []
    modules_by_name = dict(model.named_modules())
    for name, module in modules_by_name.items():
        child_name = name.rsplit(".", 1)[-1]
        if child_name in targets and isinstance(module, nn.Linear):
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = modules_by_name[parent_name] if parent_name else model
            to_wrap.append((parent, child_name, module))

    for parent, child_name, linear in to_wrap:
        setattr(parent, child_name, LoRALinear(linear, r=r, alpha=alpha, dropout=dropout))

    # Gradient checkpointing + frozen input embeddings would otherwise sever the graph before
    # it reaches the adapters; this forces the embedding output to require grad.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return len(to_wrap)


@contextmanager
def adapters_disabled(model: nn.Module) -> Iterator[None]:
    """Temporarily disable every LoRA adapter, recovering the frozen base policy."""
    adapters = [m for m in model.modules() if isinstance(m, LoRALinear)]
    for m in adapters:
        m.enabled = False
    try:
        yield
    finally:
        for m in adapters:
            m.enabled = True


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Just the trainable adapter tensors — a few MB, vs. the multi-GB full state dict."""
    return {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
