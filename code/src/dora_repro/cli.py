"""Command-line interface for the DoRA reproduction repo."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from dora_repro.config import TASKS, build_experiment, repo_root
from dora_repro.data import normalize_all_benchmarks, normalize_training_data
from dora_repro.eval import evaluate_run
from dora_repro.logging_utils import bind_logger, configure_logging, get_log_level
from dora_repro.results import summarize_runs
from dora_repro.train import run_training, smoke_test

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Sequence


def _repo_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else repo_root() / candidate


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(prog="dora-repro")
    parser.add_argument(
        "--log-level",
        default=get_log_level(),
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
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


def _prepare_data_command(args: Namespace, logger: logging.Logger) -> int:
    cache_dir = _repo_path(args.cache_dir)
    configure_logging(args.log_level, log_path=cache_dir / "logs" / "prepare-data.log")
    command_logger = bind_logger(logger, command=args.command, cache_dir=cache_dir)
    command_logger.info("Preparing training and evaluation datasets")
    normalized_train = normalize_training_data(args.train_source, cache_dir)
    normalized_eval = normalize_all_benchmarks(cache_dir, TASKS, limit=args.limit)
    command_logger.info("Prepared datasets", extra={"task_count": len(normalized_eval)})
    sys.stdout.write(f"training={normalized_train}\n")
    for task, path in normalized_eval.items():
        sys.stdout.write(f"{task}={path}\n")
    return 0


def _train_command(args: Namespace, logger: logging.Logger) -> int:
    run_name = args.run_name or _default_run_name(args.model, args.method, args.scope, args.runtime)
    run_dir = _repo_path(args.output_dir) / run_name
    configure_logging(args.log_level, log_path=run_dir / "logs" / "train.log")
    command_logger = bind_logger(
        logger,
        command=args.command,
        run_name=run_name,
        model=args.model,
        method=args.method,
        scope=args.scope,
        runtime=args.runtime,
    )
    command_logger.info("Building experiment spec")
    spec = build_experiment(
        model_name=args.model,
        method=args.method,
        scope=args.scope,
        runtime_name=args.runtime,
        experiment_name=args.experiment,
        train_data_path=_repo_path(args.train_data_path),
    )
    resume_from_checkpoint = (
        _repo_path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    )
    command_logger.info(
        "Starting training run",
        extra={"run_dir": run_dir, "resume_from_checkpoint": resume_from_checkpoint},
    )
    adapter_dir = run_training(spec, run_dir, resume_from_checkpoint)
    command_logger.info("Training completed", extra={"adapter_dir": adapter_dir})
    sys.stdout.write(f"{adapter_dir}\n")
    return 0


def _evaluate_command(args: Namespace, logger: logging.Logger) -> int:
    run_dir = _repo_path(args.run_dir)
    configure_logging(args.log_level, log_path=run_dir / "logs" / "evaluate.log")
    command_logger = bind_logger(logger, command=args.command, run_name=run_dir.name)
    tasks = tuple(args.tasks) if args.tasks else None
    command_logger.info("Evaluating run", extra={"tasks": tasks or "snapshot-default"})
    metrics = evaluate_run(run_dir, tasks)
    command_logger.info("Evaluation completed", extra={"macro_average": metrics["macro_average"]})
    sys.stdout.write(f"{metrics}\n")
    return 0


def _summarize_command(args: Namespace, logger: logging.Logger) -> int:
    results_dir = _repo_path(args.results_dir)
    output_dir = _repo_path(args.output_dir)
    configure_logging(args.log_level, log_path=output_dir / "summarize.log")
    command_logger = bind_logger(logger, command=args.command, results_dir=results_dir)
    command_logger.info("Summarizing run metrics")
    csv_path, figure_path = summarize_runs(results_dir, output_dir)
    command_logger.info(
        "Summary completed", extra={"csv_path": csv_path, "figure_path": figure_path}
    )
    sys.stdout.write(f"{csv_path}\n")
    if figure_path is not None:
        sys.stdout.write(f"{figure_path}\n")
    return 0


def _smoke_test_command(args: Namespace, logger: logging.Logger) -> int:
    output_dir = _repo_path(args.output_dir)
    configure_logging(args.log_level, log_path=output_dir / "smoke-test.log")
    command_logger = bind_logger(logger, command=args.command, output_dir=output_dir)
    command_logger.info("Running smoke test")
    output_path = smoke_test(output_dir)
    command_logger.info("Smoke test completed", extra={"output_path": output_path})
    sys.stdout.write(f"{output_path}\n")
    return 0


def _dispatch(args: Namespace, logger: logging.Logger) -> int:
    handlers = {
        "prepare-data": _prepare_data_command,
        "train": _train_command,
        "evaluate": _evaluate_command,
        "summarize": _summarize_command,
        "smoke-test": _smoke_test_command,
    }
    handler = handlers.get(args.command)
    if handler is None:
        msg = f"unknown command: {args.command}"
        raise ValueError(msg)
    return handler(args, logger)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return _dispatch(args, logging.getLogger(__name__))
    except ValueError as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
