from __future__ import annotations

from typing import TYPE_CHECKING

from dora_repro.auth import resolve_hf_token

if TYPE_CHECKING:
    from pathlib import Path


def test_resolve_hf_token_prefers_direct_env(tmp_path: Path) -> None:
    token_file = tmp_path / "token.txt"
    token_file.write_text("from-file\n", encoding="utf-8")
    expected = "from-env"
    token = resolve_hf_token({"HF_TOKEN": expected, "HF_TOKEN_PATH": str(token_file)})
    assert token == expected


def test_resolve_hf_token_reads_token_path(tmp_path: Path) -> None:
    token_file = tmp_path / "token.txt"
    token_file.write_text("from-file\n", encoding="utf-8")
    token = resolve_hf_token({"HF_TOKEN_PATH": str(token_file)})
    assert token == token_file.read_text(encoding="utf-8").strip()
