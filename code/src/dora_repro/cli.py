"""Command-line interface for the DoRA reproduction repo."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from dora_repro.config import TASKS, build_experiment, repo_root
from dora_repro.data import normalize_all_benchmarks, normalize_training_data
from dora_repro.eval import evaluate_run
from dora_repro.results import summarize_runs
from dora_repro.train import run_training, smoke_test

if TYPE_CHECKING:
    from collections.abc import Sequence


def _repo_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else repo_root() / candidate


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(prog="dora-repro")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--cache-dir", default="data/cache")
    prepare.add_argument("--train-source", default="auto")
    prepare.add_argument("--limit", type=int, default=None)

    train = subparsers.add_parser("train")
    train.add_argument("--model", default="llama2_7b")
    train.add_argument("--method", choices=("lora", "dora"), default="dora")
    train.add_argument(
        "--scope",
        choices=("full", "attention_only", "mlp_only"),
        default="full",
    )
    train.add_argument("--runtime", default="official")
    train.add_argument("--experiment", default="default")
    train.add_argument("--train-data-path", default="data/commonsense_170k.json")
    train.add_argument("--output-dir", default="results/runs")
    train.add_argument("--run-name", default=None)
    train.add_argument("--resume-from-checkpoint", default=None)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--tasks", nargs="*", default=None)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--results-dir", default="results/runs")
    summarize.add_argument("--output-dir", default="results/summary")

    smoke = subparsers.add_parser("smoke-test")
    smoke.add_argument("--output-dir", default="results/smoke-test")
    return parser


def _default_run_name(model: str, method: str, scope: str, runtime: str) -> str:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{model}-{method}-{scope}-{runtime}"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "prepare-data":
        cache_dir = _repo_path(args.cache_dir)
        normalized_train = normalize_training_data(args.train_source, cache_dir)
        normalized_eval = normalize_all_benchmarks(cache_dir, TASKS, limit=args.limit)
        sys.stdout.write(f"training={normalized_train}\n")
        for task, path in normalized_eval.items():
            sys.stdout.write(f"{task}={path}\n")
        return 0

    if args.command == "train":
        spec = build_experiment(
            model_name=args.model,
            method=args.method,
            scope=args.scope,
            runtime_name=args.runtime,
            experiment_name=args.experiment,
            train_data_path=_repo_path(args.train_data_path),
        )
        run_name = args.run_name or _default_run_name(
            args.model, args.method, args.scope, args.runtime
        )
        run_dir = _repo_path(args.output_dir) / run_name
        resume_from_checkpoint = (
            _repo_path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        )
        adapter_dir = run_training(spec, run_dir, resume_from_checkpoint)
        sys.stdout.write(f"{adapter_dir}\n")
        return 0

    if args.command == "evaluate":
        tasks = tuple(args.tasks) if args.tasks else None
        metrics = evaluate_run(_repo_path(args.run_dir), tasks)
        sys.stdout.write(f"{metrics}\n")
        return 0

    if args.command == "summarize":
        csv_path, figure_path = summarize_runs(
            _repo_path(args.results_dir), _repo_path(args.output_dir)
        )
        sys.stdout.write(f"{csv_path}\n")
        if figure_path is not None:
            sys.stdout.write(f"{figure_path}\n")
        return 0

    if args.command == "smoke-test":
        output_path = smoke_test(_repo_path(args.output_dir))
        sys.stdout.write(f"{output_path}\n")
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
