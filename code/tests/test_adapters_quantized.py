from __future__ import annotations

import importlib.util
import json
from typing import TYPE_CHECKING

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from dora_repro.adapters import attach_adapter, load_adapter_checkpoint, save_adapter_checkpoint
from dora_repro.config import AdapterPreset

if TYPE_CHECKING:
    from pathlib import Path


BITSANDBYTES_AVAILABLE = importlib.util.find_spec("bitsandbytes") is not None
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

pytestmark = pytest.mark.skipif(
    not BITSANDBYTES_AVAILABLE or not torch.cuda.is_available(),
    reason="bitsandbytes 4-bit tests require CUDA and the optional dependency",
)


def _write_local_model_bundle(root: Path) -> Path:
    model_dir = root / "model"
    model_dir.mkdir()
    config = LlamaConfig()
    config.vocab_size = 32
    config.hidden_size = 16
    config.intermediate_size = 32
    config.num_hidden_layers = 1
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.pad_token_id = 0
    config.bos_token_id = 1
    config.eos_token_id = 2
    LlamaForCausalLM(config).save_pretrained(model_dir)
    tokenizer = Tokenizer(
        WordLevel(
            vocab={PAD_TOKEN: 0, BOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3, "Answer": 4, "true": 5},
            unk_token=UNK_TOKEN,
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
    )
    fast.save_pretrained(model_dir)
    return model_dir


def test_quantized_adapter_attach_backward_and_reload(tmp_path: Path) -> None:
    model_dir = _write_local_model_bundle(tmp_path)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map={"": torch.cuda.current_device()},
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
    )
    adapter = AdapterPreset(method="dora", scope="attention_only", rank=2, alpha=4)
    attach_adapter(model, adapter)

    outputs = model(
        input_ids=torch.tensor([[1, 4, 5]], device="cuda"),
        labels=torch.tensor([[1, 4, 5]], device="cuda"),
    )
    outputs.loss.backward()

    adapter_dir = tmp_path / "adapter"
    save_adapter_checkpoint(model, adapter_dir, adapter)

    reloaded = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map={"": torch.cuda.current_device()},
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
    )
    attach_adapter(reloaded, adapter)
    load_adapter_checkpoint(reloaded, adapter_dir)
    logits = reloaded(input_ids=torch.tensor([[1, 4, 5]], device="cuda")).logits
    assert torch.isfinite(logits).all()
    assert (adapter_dir / "adapter_model.safetensors").is_file()
    config_payload = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert config_payload["method"] == "dora"
