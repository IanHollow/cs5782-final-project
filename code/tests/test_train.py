from __future__ import annotations

from typing import Any, cast

from dora_repro import train
from dora_repro.prompts import TrainingSample


def test_dataloader_pin_memory_follows_cuda_availability(monkeypatch) -> None:
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    assert train._dataloader_pin_memory() is True

    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)
    assert train._dataloader_pin_memory() is False


def test_half_precision_flags_disable_bf16_emulation(monkeypatch) -> None:
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)

    def fake_is_bf16_supported(*, including_emulation: bool = True) -> bool:
        return including_emulation

    monkeypatch.setattr(train.torch.cuda, "is_bf16_supported", fake_is_bf16_supported)
    assert train._half_precision_flags() == (False, True)


def test_half_precision_flags_fall_back_to_device_capability(monkeypatch) -> None:
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
        instruction="Answer true or false.", input="", output="the correct answer is true"
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
