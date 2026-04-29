from __future__ import annotations

import torch
from torch.nn import functional

from dora_repro.adapters import attach_adapter, iter_adapter_layers, merge_adapter_layers
from dora_repro.config import AdapterPreset
from dora_repro.dora import DoRALinear


class _ToyProjectionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(3, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x)


def test_dora_zero_init_matches_base_and_initializes_magnitude() -> None:
    base = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.tensor([[3.0, 4.0, 0.0], [0.0, 5.0, 12.0]]))
    adapter = DoRALinear(base, rank=2, alpha=4, dropout=0.0)

    x = torch.tensor([[2.0, -1.0, 0.5]])
    assert torch.allclose(adapter(x), base(x))
    assert torch.allclose(adapter.magnitude, torch.tensor([5.0, 13.0]))


def test_dora_forward_matches_detached_norm_formula() -> None:
    base = torch.nn.Linear(3, 2, bias=True)
    with torch.no_grad():
        base.weight.copy_(torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.5, 1.5]]))
        base.bias.copy_(torch.tensor([0.1, -0.2]))
    adapter = DoRALinear(base, rank=2, alpha=2, dropout=0.0)
    with torch.no_grad():
        adapter.lora_A.copy_(torch.tensor([[1.0, -1.0, 0.5], [0.0, 2.0, 1.0]]))
        adapter.lora_B.copy_(torch.tensor([[0.5, 1.0], [1.5, -0.5]]))
        adapter.magnitude.copy_(torch.tensor([4.0, 2.5]))

    x = torch.tensor([[1.0, -0.5, 2.0]])
    base_result = base(x).float()
    base_without_bias = base_result - base.bias.float()
    delta_weight = torch.matmul(adapter.lora_B.float(), adapter.lora_A.float()) * adapter.scaling
    combined = base.weight.float() + delta_weight
    direction_norm = torch.linalg.norm(combined, dim=1).detach()
    mag_scale = (adapter.magnitude.float() / direction_norm).view(1, -1)
    adapter_result = functional.linear(
        functional.linear(x.float(), adapter.lora_A.float()),
        adapter.lora_B.float(),
    )
    adapter_result *= adapter.scaling
    expected = base_result + (mag_scale - 1) * base_without_bias + mag_scale * adapter_result
    actual = adapter(x)
    assert torch.allclose(actual, expected.to(actual.dtype), atol=1e-6)


def test_dora_merge_matches_unmerged_inference() -> None:
    model = _ToyProjectionModel()
    attach_adapter(
        model,
        AdapterPreset(method="dora", scope="attention_only", rank=2, alpha=2, dropout=0.0),
    )
    adapter = next(module for _name, module in iter_adapter_layers(model))
    assert isinstance(adapter, DoRALinear)
    with torch.no_grad():
        adapter.lora_A.copy_(torch.tensor([[1.0, -1.0, 0.5], [0.0, 2.0, 1.0]]))
        adapter.lora_B.copy_(torch.tensor([[0.5, 1.0], [1.5, -0.5]]))
        adapter.magnitude.copy_(torch.tensor([2.5, 1.75]))

    x = torch.tensor([[0.25, -0.5, 1.0]])
    unmerged = model(x)
    merge_adapter_layers(model)
    merged = model(x)
    assert torch.allclose(merged, unmerged, atol=1e-6)
