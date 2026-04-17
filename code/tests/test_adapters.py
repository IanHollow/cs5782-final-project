from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch.nn import functional
from transformers import LlamaConfig, LlamaForCausalLM

from dora_repro.adapters import (
    DoRALinear,
    LoRALinear,
    adapter_parameter_names,
    attach_adapter,
    iter_adapter_layers,
    load_adapter_checkpoint,
    merge_adapter_layers,
    save_adapter_checkpoint,
    unmerge_adapter_layers,
)
from dora_repro.config import AdapterPreset

if TYPE_CHECKING:
    from pathlib import Path

    from dora_repro.config import TargetScope


class _ToyProjectionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(3, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x)


def _tiny_llama() -> LlamaForCausalLM:
    config = LlamaConfig()
    config.vocab_size = 64
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.pad_token_id = 0
    config.bos_token_id = 1
    config.eos_token_id = 2
    return LlamaForCausalLM(config)


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


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("full", {"q_proj", "k_proj", "v_proj", "up_proj", "down_proj"}),
        ("attention_only", {"q_proj", "k_proj", "v_proj"}),
        ("mlp_only", {"up_proj", "down_proj"}),
    ],
)
def test_attach_adapter_replaces_exact_scope_modules(
    scope: TargetScope,
    expected: set[str],
) -> None:
    model = _tiny_llama()
    attach_adapter(model, AdapterPreset(method="lora", scope=scope))
    names = [name for name, _module in iter_adapter_layers(model)]
    suffixes = {name.rsplit(".", 1)[-1] for name in names}
    assert suffixes == expected
    assert len(names) == model.config.num_hidden_layers * len(expected)


def test_attach_adapter_leaves_only_adapter_parameters_trainable() -> None:
    model = _tiny_llama()
    attach_adapter(model, AdapterPreset(method="dora", scope="attention_only"))
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == adapter_parameter_names(model)
    assert trainable


def test_attach_adapter_raises_when_targets_are_missing() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    with pytest.raises(ValueError, match="Target modules"):
        attach_adapter(model, AdapterPreset(method="lora", scope="attention_only"))


def test_attach_adapter_rejects_reattaching_to_an_adapted_model() -> None:
    model = _tiny_llama()
    attach_adapter(model, AdapterPreset(method="lora", scope="attention_only"))
    with pytest.raises(TypeError, match="already has an adapter"):
        attach_adapter(model, AdapterPreset(method="lora", scope="attention_only"))


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


def test_save_and_load_adapter_checkpoint_reproduces_logits(tmp_path: Path) -> None:
    torch.manual_seed(0)
    source = _tiny_llama()
    base_state = {name: tensor.clone() for name, tensor in source.state_dict().items()}
    adapter_preset = AdapterPreset(method="dora", scope="attention_only", rank=2, alpha=4)
    attach_adapter(source, adapter_preset)
    for index, (_name, module) in enumerate(iter_adapter_layers(source), start=1):
        with torch.no_grad():
            module.lora_A.fill_(0.05 * index)
            module.lora_B.fill_(0.02 * index)
            if isinstance(module, DoRALinear):
                module.magnitude.fill_(1.0 + 0.1 * index)

    inputs = torch.tensor([[1, 2, 3, 4]])
    source_logits = source(input_ids=inputs).logits
    adapter_dir = tmp_path / "adapter"
    save_adapter_checkpoint(source, adapter_dir, adapter_preset)

    torch.manual_seed(0)
    target = _tiny_llama()
    target.load_state_dict(base_state)
    attach_adapter(target, adapter_preset)
    load_adapter_checkpoint(target, adapter_dir)
    target_logits = target(input_ids=inputs).logits

    assert torch.allclose(target_logits, source_logits, atol=1e-6)
    assert (adapter_dir / "adapter_config.json").is_file()
    assert (adapter_dir / "adapter_model.safetensors").is_file()


def test_adapter_parameter_names_include_dora_magnitude() -> None:
    model = _ToyProjectionModel()
    attach_adapter(model, AdapterPreset(method="dora", scope="attention_only", rank=2, alpha=2))
    names = adapter_parameter_names(model)
    assert any(name.endswith(".magnitude") for name in names)
    assert any(name.endswith(".lora_A") for name in names)
    assert any(name.endswith(".lora_B") for name in names)
