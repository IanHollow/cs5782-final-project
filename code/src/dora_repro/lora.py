from __future__ import annotations

from typing import TYPE_CHECKING

from dora_repro.adapter_base import AdapterLinearBase

if TYPE_CHECKING:
    import torch


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
