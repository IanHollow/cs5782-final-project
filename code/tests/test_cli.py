from __future__ import annotations

from typing import TYPE_CHECKING

from dora_repro.cli import build_parser, main

if TYPE_CHECKING:
    from pathlib import Path


def test_smoke_test_command_writes_output(tmp_path: Path) -> None:
    exit_code = main(["smoke-test", "--output-dir", str(tmp_path)])
    assert exit_code == 0
    assert (tmp_path / "smoke_test.json").is_file()
    log_path = tmp_path / "smoke-test.log"
    assert log_path.is_file()
    assert "Smoke test completed" in log_path.read_text(encoding="utf-8")


def test_prepare_data_parser_help() -> None:
    help_text = build_parser().format_help()
    assert "prepare-data" in help_text
