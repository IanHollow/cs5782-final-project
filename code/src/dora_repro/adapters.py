"""Local LoRA/DoRA adapter layers, injection, and checkpoint I/O."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file, save_file

from dora_repro.adapter_base import AdapterLinearBase, _is_supported_target_module
from dora_repro.dora import DoRALinear
from dora_repro.lora import LoRALinear

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn
    from transformers import PreTrainedModel

    from dora_repro.config import AdapterPreset


ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_MODEL_NAME = "adapter_model.safetensors"


def prepare_model_for_adapter_training(
    model: PreTrainedModel,
    *,
    use_gradient_checkpointing: bool,
) -> PreTrainedModel:
    """Freeze the base model and apply the minimal k-bit training preparation we need."""
    is_quantized = bool(
        getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
    )
    for parameter in model.parameters():
        parameter.requires_grad = False

    if is_quantized:
        for parameter in model.parameters():
            if (
                parameter.dtype in {torch.float16, torch.bfloat16}
                and parameter.__class__.__name__ != "Params4bit"
            ):
                parameter.data = parameter.data.to(torch.float32)

    if use_gradient_checkpointing:
        if is_quantized:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def _require_input_grads(
                    _module: nn.Module,
                    _inputs: tuple[object, ...],
                    output: object,
                ) -> None:
                    if isinstance(output, torch.Tensor):
                        output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(_require_input_grads)
        model.gradient_checkpointing_enable()
    return model


def _build_adapter_module(base_layer: nn.Module, adapter: AdapterPreset) -> AdapterLinearBase:
    kwargs = {"rank": adapter.rank, "alpha": adapter.alpha, "dropout": adapter.dropout}
    if adapter.method == "lora":
        return LoRALinear(base_layer, **kwargs)
    return DoRALinear(base_layer, **kwargs)


def _replace_module(model: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, replacement)


def iter_adapter_layers(model: nn.Module) -> list[tuple[str, AdapterLinearBase]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, AdapterLinearBase)
    ]


def adapter_parameter_names(model: nn.Module) -> set[str]:
    names: set[str] = set()
    for module_name, module in iter_adapter_layers(model):
        for parameter_name, _parameter in module.named_parameters(recurse=False):
            qualified = f"{module_name}.{parameter_name}" if module_name else parameter_name
            names.add(qualified)
    return names


def collect_adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    names = adapter_parameter_names(model)
    state_dict = model.state_dict()
    return {name: state_dict[name].detach().cpu().contiguous() for name in sorted(names)}


def attach_adapter(model: PreTrainedModel, adapter: AdapterPreset) -> PreTrainedModel:
    """Attach locally implemented LoRA/DoRA adapters to matching modules in-place."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    replacements: list[tuple[str, AdapterLinearBase]] = []
    for module_name, module in list(model.named_modules()):
        if not any(module_name.endswith(target) for target in adapter.target_modules):
            continue
        if isinstance(module, AdapterLinearBase):
            msg = f"Model already has an adapter attached at module: {module_name}"
            raise TypeError(msg)
        if not _is_supported_target_module(module):
            msg = (
                f"Target module {module_name} has unsupported type "
                f"{module.__class__.__name__}; only Linear and 4-bit Linear are supported"
            )
            raise ValueError(msg)
        replacements.append((module_name, _build_adapter_module(module, adapter)))

    if not replacements:
        msg = f"Target modules {adapter.target_modules!r} not found in the base model"
        raise ValueError(msg)

    for module_name, replacement in replacements:
        _replace_module(model, module_name, replacement)

    if hasattr(model, "config"):
        model.config.use_cache = False
    return model


def merge_adapter_layers(model: nn.Module) -> None:
    for _name, module in iter_adapter_layers(model):
        module.merge()


def unmerge_adapter_layers(model: nn.Module) -> None:
    for _name, module in iter_adapter_layers(model):
        module.unmerge()


def save_adapter_checkpoint(model: nn.Module, adapter_dir: Path, adapter: AdapterPreset) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    state_dict = collect_adapter_state_dict(model)
    save_file(state_dict, adapter_dir / ADAPTER_MODEL_NAME)
    payload = {
        "method": adapter.method,
        "scope": adapter.scope,
        "rank": adapter.rank,
        "alpha": adapter.alpha,
        "dropout": adapter.dropout,
        "target_modules": list(adapter.target_modules),
        "module_names": [name for name, _module in iter_adapter_layers(model)],
    }
    (adapter_dir / ADAPTER_CONFIG_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_adapter_checkpoint(model: nn.Module, adapter_dir: Path) -> None:
    adapter_state = load_file(adapter_dir / ADAPTER_MODEL_NAME)
    expected_names = adapter_parameter_names(model)
    loaded_names = set(adapter_state)
    missing = sorted(expected_names - loaded_names)
    unexpected = sorted(loaded_names - expected_names)
    if missing or unexpected:
        msg = f"Adapter checkpoint mismatch; missing={missing}, unexpected={unexpected}"
        raise ValueError(msg)

    live_state = model.state_dict()
    with torch.no_grad():
        for name, tensor in adapter_state.items():
            target = live_state[name]
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
