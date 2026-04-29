from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dora_repro.results import macro_average, summarize_runs, write_json

if TYPE_CHECKING:
    from pathlib import Path


def test_macro_average() -> None:
    value = macro_average({"boolq": 0.5, "piqa": 1.0})
    assert value == pytest.approx(0.75)


def test_summarize_runs_writes_csv_and_plot(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo"
    write_json(
        run_dir / "metrics.json",
        {
            "run_name": "demo",
            "model_name": "llama2_7b",
            "method": "dora",
            "scope": "full",
            "macro_average": 0.5,
        },
    )
    write_json(
        run_dir / "run.metadata.json",
        {
            "trainable_parameters": 100,
            "total_parameters": 1000,
        },
    )
    csv_path, figure_path = summarize_runs(tmp_path / "runs", tmp_path / "summary")
    assert csv_path.is_file()
    assert figure_path is not None
    assert figure_path.is_file()
    
    content = csv_path.read_text(encoding="utf-8")
    assert "trainable_percentage" in content
    assert "10.0" in content

