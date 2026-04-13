from __future__ import annotations

from dora_repro import train


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
