from __future__ import annotations

from dora_repro import assets


def test_available_model_presets_includes_repo_models() -> None:
    presets = assets.available_model_presets()
    assert {"tiny_debug", "llama2_7b", "llama3_8b"}.issubset(set(presets))


def test_prefetch_model_to_hf_cache_uses_snapshot_download(monkeypatch) -> None:
    calls: dict[str, object] = {}
    expected_path = "hf-cache/snapshot"

    def fake_snapshot_download(*, repo_id: str, token: str | None = None) -> str:
        calls["repo_id"] = repo_id
        calls["token"] = token
        return expected_path

    monkeypatch.setattr(assets, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(assets, "resolve_hf_token", lambda: "token-123")

    output = assets.prefetch_model_to_hf_cache(
        model_name="tiny_debug",
        model_id="HuggingFaceTB/SmolLM2-135M",
    )

    assert str(output) == expected_path
    assert calls == {"repo_id": "HuggingFaceTB/SmolLM2-135M", "token": "token-123"}
