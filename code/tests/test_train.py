from __future__ import annotations

from dora_repro import train


def test_dataloader_pin_memory_follows_cuda_availability(monkeypatch) -> None:
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    assert train._dataloader_pin_memory() is True

    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)
    assert train._dataloader_pin_memory() is False
