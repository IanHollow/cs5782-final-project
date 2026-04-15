from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING

from dora_repro import cli as cli_module

if TYPE_CHECKING:
    from pathlib import Path


def test_smoke_test_command_writes_output(tmp_path: Path) -> None:
    exit_code = cli_module.main(["smoke-test", "--output-dir", str(tmp_path)])
    assert exit_code == 0
    assert (tmp_path / "smoke_test.json").is_file()
    log_path = tmp_path / "smoke-test.log"
    assert log_path.is_file()
    assert "Smoke test completed" in log_path.read_text(encoding="utf-8")


def test_prepare_data_parser_help() -> None:
    help_text = cli_module.build_parser().format_help()
    assert "prepare-data" in help_text
    assert "prepare-assets" in help_text


def test_resolve_train_settings_uses_env_for_lora_attention_only(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DORA_REPRO_MODEL", "llama2_7b")
    monkeypatch.setenv("DORA_REPRO_METHOD", "lora")
    monkeypatch.setenv("DORA_REPRO_SCOPE", "attention_only")
    monkeypatch.setenv("DORA_REPRO_RUNTIME", "colab_l4_llama")
    monkeypatch.setenv("DORA_REPRO_EXPERIMENT", "paper_colab")
    resolved = cli_module._resolve_train_settings(
        Namespace(
            model=None,
            method=None,
            scope=None,
            runtime=None,
            experiment=None,
            train_data_path=None,
            run_name=None,
        )
    )
    assert resolved["method"] == "lora"
    assert resolved["scope"] == "attention_only"
    assert resolved["runtime"] == "colab_l4_llama"
    assert resolved["experiment"] == "paper_colab"


def test_resolve_train_settings_uses_env_for_dora_mlp_only(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DORA_REPRO_MODEL", "llama3_8b")
    monkeypatch.setenv("DORA_REPRO_METHOD", "dora")
    monkeypatch.setenv("DORA_REPRO_SCOPE", "mlp_only")
    resolved = cli_module._resolve_train_settings(
        Namespace(
            model=None,
            method=None,
            scope=None,
            runtime=None,
            experiment=None,
            train_data_path=None,
            run_name=None,
        )
    )
    assert resolved["model"] == "llama3_8b"
    assert resolved["method"] == "dora"
    assert resolved["scope"] == "mlp_only"


def test_resolve_train_settings_cli_overrides_conflicting_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DORA_REPRO_METHOD", "lora")
    monkeypatch.setenv("DORA_REPRO_SCOPE", "full")
    resolved = cli_module._resolve_train_settings(
        Namespace(
            model="llama3_8b",
            method="dora",
            scope="mlp_only",
            runtime=None,
            experiment=None,
            train_data_path=None,
            run_name=None,
        )
    )
    assert resolved["model"] == "llama3_8b"
    assert resolved["method"] == "dora"
    assert resolved["scope"] == "mlp_only"


def test_parse_task_selection_supports_csv_and_all() -> None:
    assert cli_module._parse_task_selection("boolq,piqa") == ("boolq", "piqa")
    assert cli_module._parse_task_selection("all") == cli_module.TASKS
    assert cli_module._resolve_evaluation_tasks(Namespace(tasks=["boolq", "hellaswag"])) == (
        "boolq",
        "hellaswag",
    )
