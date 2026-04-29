"""Shared base utilities for LoRA and DoRA adapter layers."""

from __future__ import annotations

import importlib
import importlib.util
import math
from typing import TYPE_CHECKING, Protocol, cast

import torch
from torch import nn
from torch.nn import functional

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

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
