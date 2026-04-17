"""Local LoRA/DoRA adapter layers, injection, and checkpoint I/O."""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
from typing import TYPE_CHECKING, Protocol, cast

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import ModuleType

    from transformers import PreTrainedModel

    from dora_repro.config import AdapterPreset


ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_MODEL_NAME = "adapter_model.safetensors"
_FOUR_BIT_LINEAR_NAME = "Linear4bit"
_QUANTIZED_WEIGHT_TYPES = frozenset({"Params4bit", "Int8Params"})


def _load_bitsandbytes() -> ModuleType | None:
    if importlib.util.find_spec("bitsandbytes") is None:
        return None
    return importlib.import_module("bitsandbytes")


BNB_MODULE = _load_bitsandbytes()


class LinearAdapterTarget(Protocol):
    """Structural type for linear layers we can wrap with adapters."""

    in_features: int
    out_features: int
    weight: torch.Tensor
    bias: torch.Tensor | None

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


def _is_quantized_weight(weight: object) -> bool:
    return weight.__class__.__name__ in _QUANTIZED_WEIGHT_TYPES


def _is_4bit_linear(module: nn.Module) -> bool:
    return module.__class__.__name__ == _FOUR_BIT_LINEAR_NAME


def _is_supported_target_module(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) or _is_4bit_linear(module)


def _validated_adapter_target(module: nn.Module) -> LinearAdapterTarget:
    if not _is_supported_target_module(module):
        msg = f"Unsupported adapter target: {module.__class__.__name__}"
        raise TypeError(msg)

    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    weight = getattr(module, "weight", None)
    if not isinstance(in_features, int) or not isinstance(out_features, int):
        msg = f"Adapter target {module.__class__.__name__} is missing linear dimensions"
        raise TypeError(msg)
    if not isinstance(weight, torch.Tensor):
        msg = f"Adapter target {module.__class__.__name__} is missing a tensor weight"
        raise TypeError(msg)
    return cast("LinearAdapterTarget", module)


def _dequantize_4bit(weight_data: torch.Tensor, quant_state: object) -> torch.Tensor:
    if BNB_MODULE is None:
        msg = "bitsandbytes is required to dequantize 4-bit adapter targets"
        raise RuntimeError(msg)
    functional_module = getattr(BNB_MODULE, "functional", None)
    dequantize = cast(
        "Callable[..., torch.Tensor] | None",
        getattr(functional_module, "dequantize_4bit", None),
    )
    if dequantize is None:
        msg = "bitsandbytes.functional.dequantize_4bit is unavailable"
        raise RuntimeError(msg)
    return dequantize(weight_data, quant_state=quant_state)


def _dequantize_weight(module: nn.Module) -> torch.Tensor:
    weight = _validated_adapter_target(module).weight
    if not _is_quantized_weight(weight):
        return weight.detach()
    quant_state = getattr(weight, "quant_state", None) or getattr(module, "quant_state", None)
    if quant_state is None:
        msg = f"Missing 4-bit quantization state for module type {module.__class__.__name__}"
        raise RuntimeError(msg)
    if weight.__class__.__name__ != "Params4bit":
        msg = f"Unsupported quantized weight type: {weight.__class__.__name__}"
        raise RuntimeError(msg)
    return _dequantize_4bit(weight.data, quant_state).detach()


def _freeze_module_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


class AdapterLinearBase(nn.Module):
    """Base class for LoRA-style adapters attached to linear layers."""

    method_name: str
    base_layer: LinearAdapterTarget
    lora_A: nn.Parameter  # noqa: N815
    lora_B: nn.Parameter  # noqa: N815

    def __init__(
        self,
        base_layer: nn.Module,
        *,
        rank: int,
        alpha: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            msg = "adapter rank must be positive"
            raise ValueError(msg)

        validated_base_layer = _validated_adapter_target(base_layer)
        self.base_layer = validated_base_layer
        self.in_features = validated_base_layer.in_features
        self.out_features = validated_base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Match the base linear's device (e.g. CUDA when the parent was loaded with
        # accelerate device_map); default torch.empty() would stay on CPU and break 4-bit eval.
        adapter_device = validated_base_layer.weight.device
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features, device=adapter_device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.empty(self.out_features, rank, device=adapter_device, dtype=torch.float32)
        )
        self.merged = False
        self._merge_backup_weight: torch.Tensor | None = None

        _freeze_module_parameters(base_layer)
        self._reset_parameters()

    @property
    def is_quantized(self) -> bool:
        return _is_quantized_weight(self.base_layer.weight)

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def _lora_result(self, x: torch.Tensor) -> torch.Tensor:
        hidden = functional.linear(self.dropout(x.to(self.lora_A.dtype)), self.lora_A)
        return functional.linear(hidden, self.lora_B) * self.scaling

    def _delta_weight(self) -> torch.Tensor:
        return torch.matmul(self.lora_B.float(), self.lora_A.float()) * self.scaling

    def _merged_weight(self, base_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def merge(self) -> None:
        if self.merged:
            return
        if self.is_quantized:
            msg = "Cannot merge adapter weights into a quantized base layer"
            raise ValueError(msg)
        self._merge_backup_weight = self.base_layer.weight.detach().clone()
        merged_weight = self._merged_weight(self._merge_backup_weight.float())
        self.base_layer.weight.data.copy_(merged_weight.to(self.base_layer.weight.dtype))
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            return
        if self._merge_backup_weight is None:
            msg = "Cannot unmerge adapter weights without a backup copy"
            raise RuntimeError(msg)
        self.base_layer.weight.data.copy_(self._merge_backup_weight)
        self._merge_backup_weight = None
        self.merged = False


class LoRALinear(AdapterLinearBase):
    """Standard LoRA adapter on top of a frozen linear layer."""

    method_name = "lora"

    def _merged_weight(self, base_weight: torch.Tensor) -> torch.Tensor:
        return base_weight + self._delta_weight().to(base_weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_result = self.base_layer(x)
        if self.merged:
            return base_result
        return (base_result.float() + self._lora_result(x).float()).to(base_result.dtype)


class DoRALinear(AdapterLinearBase):
    """DoRA adapter with a trainable output magnitude and LoRA directional update."""

    method_name = "dora"
    magnitude: nn.Parameter

    def __init__(
        self,
        base_layer: nn.Module,
        *,
        rank: int,
        alpha: int,
        dropout: float,
    ) -> None:
        super().__init__(base_layer, rank=rank, alpha=alpha, dropout=dropout)
        base_weight = _dequantize_weight(base_layer).float()
        self.magnitude = nn.Parameter(torch.linalg.norm(base_weight, dim=1))

    def _merged_weight(self, base_weight: torch.Tensor) -> torch.Tensor:
        direction = base_weight + self._delta_weight().to(base_weight.device)
        direction_norm = torch.linalg.norm(direction, dim=1).clamp_min(1e-12)
        magnitude = self.magnitude.detach().float().to(direction.device)
        return direction * (magnitude / direction_norm).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_result = self.base_layer(x)
        if self.merged:
            return base_result

        adapter_result = self._lora_result(x).float()
        base_weight = (
            _dequantize_weight(cast("nn.Module", self.base_layer)).to(adapter_result.device).float()
        )
        combined = base_weight + self._delta_weight().to(adapter_result.device)
        direction_norm = torch.linalg.norm(combined, dim=1).clamp_min(1e-12).detach()
        mag_scale = (self.magnitude.float().to(direction_norm.device) / direction_norm).view(
            *([1] * (base_result.ndim - 1)),
            -1,
        )

        bias = self.base_layer.bias
        base_without_bias = base_result.float().clone()
        if bias is not None:
            base_without_bias -= bias.float()

        extra = (mag_scale - 1) * base_without_bias + mag_scale * adapter_result
        return (base_result.float() + extra).to(base_result.dtype)


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
                    if hasattr(output, "requires_grad_"):
                        cast("Callable[[bool], object]", output.requires_grad_)(True)

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
