"""Result writing, aggregation, and plotting helpers."""

from __future__ import annotations

import csv
import json
import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from tomli_w import dumps as toml_dumps

from dora_repro.logging_utils import bind_logger

if TYPE_CHECKING:
    from pathlib import Path

    from dora_repro.config import ExperimentSpec


logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON file with deterministic formatting."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, payload: list[dict[str, Any]]) -> Path:
    """Write a JSONL file."""
    ensure_dir(path.parent)
    lines = [json.dumps(record, sort_keys=True) for record in payload]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_snapshot(path: Path, spec: ExperimentSpec) -> Path:
    """Write a TOML snapshot for a run."""
    ensure_dir(path.parent)
    path.write_text(toml_dumps(spec.to_snapshot()), encoding="utf-8")
    return path


def macro_average(metrics: dict[str, float]) -> float:
    """Compute the macro average of task accuracies."""
    if not metrics:
        return 0.0
    return sum(metrics.values()) / len(metrics)


def summarize_runs(results_dir: Path, output_dir: Path) -> tuple[Path, Path | None]:
    """Aggregate run metrics into a CSV and an optional bar chart."""
    ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(metrics)

    csv_path = output_dir / "summary.csv"
    if not rows:
        csv_path.write_text("run_name,macro_average\n", encoding="utf-8")
        bind_logger(logger, results_dir=results_dir, csv_path=csv_path).info("No run metrics found")
        return csv_path, None

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    figure_path = output_dir / "macro_average.png"
    plt.figure(figsize=(10, 4))
    plt.bar([row["run_name"] for row in rows], [row["macro_average"] for row in rows])
    plt.ylabel("Macro Average Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    bind_logger(
        logger,
        run_count=len(rows),
        csv_path=csv_path,
        figure_path=figure_path,
    ).info("Summarized run metrics")
    return csv_path, figure_path
