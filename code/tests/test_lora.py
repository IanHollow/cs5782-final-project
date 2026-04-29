from __future__ import annotations

import torch
from torch.nn import functional

from dora_repro.adapters import (
    attach_adapter,
    iter_adapter_layers,
    merge_adapter_layers,
    unmerge_adapter_layers,
)
from dora_repro.config import AdapterPreset
from dora_repro.lora import LoRALinear


class _ToyProjectionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(3, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x)


def test_lora_forward_matches_closed_form_update() -> None:

    base = torch.nn.Linear(3, 2, bias=True)
    with torch.no_grad():
        base.weight.copy_(torch.tensor([[1.0, -2.0, 0.5], [0.0, 1.5, -1.0]]))
        base.bias.copy_(torch.tensor([0.25, -0.75]))
    adapter = LoRALinear(base, rank=2, alpha=6, dropout=0.0)
    with torch.no_grad():
        adapter.lora_A.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]]))
        adapter.lora_B.copy_(torch.tensor([[1.0, -0.5], [2.0, 1.5]]))

    x = torch.tensor([[1.0, 2.0, -1.0], [-0.5, 0.25, 3.0]])
    expected = functional.linear(x, base.weight, base.bias)
    expected += (
        functional.linear(functional.linear(x, adapter.lora_A), adapter.lora_B) * adapter.scaling
    )
    actual = adapter(x)
    assert torch.allclose(actual, expected)


def test_lora_merge_unmerge_round_trip() -> None:
    model = _ToyProjectionModel()
    attach_adapter(
        model,
        AdapterPreset(method="lora", scope="attention_only", rank=2, alpha=4, dropout=0.0),
    )
    adapter = next(module for _name, module in iter_adapter_layers(model))
    assert isinstance(adapter, LoRALinear)
    with torch.no_grad():
        adapter.lora_A.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]]))
        adapter.lora_B.copy_(torch.tensor([[1.0, -0.5], [2.0, 1.5]]))

    x = torch.tensor([[1.0, 2.0, -1.0]])
    unmerged = model(x)
    original = adapter.base_layer.weight.detach().clone()
    merge_adapter_layers(model)
    merged = model(x)
    assert torch.allclose(merged, unmerged, atol=1e-6)
    unmerge_adapter_layers(model)
    restored = model(x)
    assert torch.allclose(restored, unmerged, atol=1e-6)
    assert torch.allclose(adapter.base_layer.weight, original)
