"""PEFT adapter construction utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from peft import LoraConfig, get_peft_model

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from dora_repro.config import AdapterPreset


def build_lora_config(adapter: AdapterPreset) -> LoraConfig:
    """Create a PEFT config for LoRA or DoRA."""
    return LoraConfig(
        r=adapter.rank,
        lora_alpha=adapter.alpha,
        target_modules=list(adapter.target_modules),
        lora_dropout=adapter.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=adapter.method == "dora",
    )


def attach_adapter(model: PreTrainedModel, adapter: AdapterPreset) -> PreTrainedModel:
    """Attach a LoRA or DoRA adapter to a pretrained model."""
    return get_peft_model(model, build_lora_config(adapter))
