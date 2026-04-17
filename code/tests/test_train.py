from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from dora_repro import train
from dora_repro.adapters import attach_adapter, load_adapter_checkpoint
from dora_repro.config import (
    AdapterPreset,
    ExperimentSpec,
    ModelPreset,
    RuntimePreset,
)
from dora_repro.prompts import TrainingSample

if TYPE_CHECKING:
    from pathlib import Path

    from dora_repro.config import AdapterMethod

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"


def _build_small_llama_config() -> LlamaConfig:
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
    return config


def _write_local_model_bundle(root: Path) -> Path:
    model_dir = root / "model"
    model_dir.mkdir(parents=True)
    torch.manual_seed(0)
    LlamaForCausalLM(_build_small_llama_config()).save_pretrained(model_dir)

    vocab = {
        PAD_TOKEN: 0,
        BOS_TOKEN: 1,
        EOS_TOKEN: 2,
        UNK_TOKEN: 3,
        "Below": 4,
        "is": 5,
        "an": 6,
        "instruction": 7,
        "that": 8,
        "describes": 9,
        "a": 10,
        "task": 11,
        "Write": 12,
        "response": 13,
        "appropriately": 14,
        "completes": 15,
        "the": 16,
        "request": 17,
        "Instruction": 18,
        "Response": 19,
        "Answer": 20,
        "true": 21,
        "false": 22,
        "or": 23,
        "Q1": 24,
        "Q2": 25,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
    )
    fast.model_max_length = 128
    fast.save_pretrained(model_dir)
    return model_dir


def _write_training_json(path: Path) -> None:
    rows = [
        {"instruction": "Answer true or false.", "input": "", "output": "true"},
        {"instruction": "Answer true or false.", "input": "", "output": "false"},
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def _local_spec(root: Path, method: AdapterMethod) -> ExperimentSpec:
    model_dir = _write_local_model_bundle(root)
    train_path = root / "train.json"
    _write_training_json(train_path)
    return ExperimentSpec(
        experiment_name="local",
        model=ModelPreset(
            name="local_llama",
            model_id=str(model_dir),
            learning_rate=5e-4,
            cutoff_len=64,
        ),
        adapter=AdapterPreset(method=method, scope="attention_only", rank=2, alpha=4),
        runtime=RuntimePreset(
            name="cpu",
            per_device_batch_size=1,
            effective_batch_size=1,
            gradient_checkpointing=False,
            use_4bit=False,
        ),
        train_data_path=train_path,
        task_names=("boolq",),
        max_train_samples=2,
        val_set_size=0,
        num_epochs=1,
        save_steps=10,
        eval_steps=10,
        seed=0,
        max_new_tokens=2,
    )


def test_dataloader_pin_memory_follows_cuda_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    assert train._dataloader_pin_memory() is True

    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)
    assert train._dataloader_pin_memory() is False


def test_half_precision_flags_disable_bf16_emulation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)

    def fake_is_bf16_supported(*, including_emulation: bool = True) -> bool:
        return including_emulation

    monkeypatch.setattr(train.torch.cuda, "is_bf16_supported", fake_is_bf16_supported)
    assert train._half_precision_flags() == (False, True)


def test_half_precision_flags_fall_back_to_device_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)

    def fake_is_bf16_supported(*, including_emulation: bool = True) -> bool:
        msg = f"unexpected call with including_emulation={including_emulation}"
        raise TypeError(msg)

    monkeypatch.setattr(train.torch.cuda, "is_bf16_supported", fake_is_bf16_supported)
    monkeypatch.setattr(train.torch.cuda, "get_device_capability", lambda: (8, 0))
    assert train._half_precision_flags() == (True, False)


class _FakeTokenizer:
    eos_token_id = 99

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int,
        padding: bool,
    ) -> dict[str, list[int]]:
        del truncation, max_length, padding
        return {"input_ids": list(range(1, len(text.split()) + 1))}


def test_tokenize_example_masks_prompt_tokens_when_train_on_inputs_disabled() -> None:
    sample = TrainingSample(
        instruction="Answer true or false.",
        input="",
        output="the correct answer is true",
    )
    tokenized = train._tokenize_example(
        sample=sample,
        tokenizer=cast("Any", _FakeTokenizer()),
        cutoff_len=128,
        train_on_inputs=False,
    )
    prompt_len = len(train.format_user_prompt(sample).split())
    assert tokenized["labels"][:prompt_len] == [-100] * prompt_len
    assert all(label != -100 for label in tokenized["labels"][prompt_len:])


def test_prepare_model_for_adapter_training_freezes_base_parameters() -> None:
    model = LlamaForCausalLM(_build_small_llama_config())
    prepared = train.prepare_model_for_adapter_training(
        model,
        use_gradient_checkpointing=False,
    )
    assert all(not parameter.requires_grad for parameter in prepared.parameters())


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_run_training_saves_local_adapter_checkpoint(tmp_path: Path, method: str) -> None:
    spec = _local_spec(tmp_path / method, cast("AdapterMethod", method))
    run_dir = tmp_path / "runs" / method
    adapter_dir = train.run_training(spec, run_dir)

    assert adapter_dir == run_dir / "adapter"
    assert (adapter_dir / "adapter_config.json").is_file()
    assert (adapter_dir / "adapter_model.safetensors").is_file()
    assert (adapter_dir / "tokenizer.json").is_file()

    reloaded = LlamaForCausalLM.from_pretrained(spec.model.model_id)
    attach_adapter(reloaded, spec.adapter)
    load_adapter_checkpoint(reloaded, adapter_dir)
    logits = reloaded(input_ids=torch.tensor([[1, 2, 3]])).logits
    assert torch.isfinite(logits).all()
