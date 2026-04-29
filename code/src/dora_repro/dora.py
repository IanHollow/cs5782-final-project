from typing import cast

import torch
from torch import nn

from dora_repro.adapter_base import AdapterLinearBase, _dequantize_weight


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
