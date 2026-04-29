"""Command-line interface for the DoRA reproduction repo."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from dora_repro.assets import available_model_presets, prefetch_model_to_hf_cache
from dora_repro.config import TASKS, build_experiment, repo_root
from dora_repro.data import normalize_all_benchmarks, normalize_training_data
from dora_repro.eval import evaluate_run
from dora_repro.logging_utils import bind_logger, configure_logging, get_log_level
from dora_repro.results import summarize_runs
from dora_repro.train import run_training, smoke_test

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Sequence

    from dora_repro.config import AdapterMethod, TargetScope

ENV_PREFIX = "DORA_REPRO_"
METHOD_CHOICES = ("lora", "dora")
SCOPE_CHOICES = ("full", "attention_only", "mlp_only")


def _repo_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else repo_root() / candidate


def _env_name(name: str) -> str:
    return f"{ENV_PREFIX}{name}"


def _env_value(name: str) -> str | None:
    raw = os.environ.get(_env_name(name))
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _resolve_value(explicit: str | None, env_name: str, fallback: str) -> str:
    return explicit if explicit is not None else _env_value(env_name) or fallback


def _resolve_choice(
    explicit: str | None,
    env_name: str,
    fallback: str,
    choices: tuple[str, ...],
) -> str:
    value = _resolve_value(explicit, env_name, fallback)
    if value not in choices:
        msg = f"{_env_name(env_name)} must be one of: {', '.join(choices)}"
        raise ValueError(msg)
    return value


def _parse_task_selection(raw: Sequence[str] | str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    items = [raw] if isinstance(raw, str) else list(raw)
    tokens = [token.strip() for item in items for token in str(item).split(",") if token.strip()]
    if not tokens:
        return None
    if any(token.lower() == "all" for token in tokens):
        if len(tokens) != 1:
            msg = "'all' cannot be combined with specific benchmark task names"
            raise ValueError(msg)
        return TASKS
    task_lookup = {task.lower(): task for task in TASKS}
    try:
        return tuple(task_lookup[token.lower()] for token in tokens)
    except KeyError as error:
        msg = f"Unsupported benchmark task: {error.args[0]}"
        raise ValueError(msg) from error


def _resolve_train_settings(args: Namespace) -> dict[str, str | Path | int | None]:
    model = _resolve_value(args.model, "MODEL", "llama2_7b")
    method = cast(
        "AdapterMethod",
        _resolve_choice(args.method, "METHOD", "dora", METHOD_CHOICES),
    )
    scope = cast(
        "TargetScope",
        _resolve_choice(args.scope, "SCOPE", "full", SCOPE_CHOICES),
    )
    runtime = _resolve_value(args.runtime, "RUNTIME", "official")
    experiment = _resolve_value(args.experiment, "EXPERIMENT", "default")
    train_data_path = args.train_data_path or _env_value("TRAIN_DATA_PATH")
    run_name = args.run_name or _env_value("RUN_NAME")
    rank_env = _env_value("RANK")
    rank = args.rank if args.rank is not None else (int(rank_env) if rank_env else None)
    return {
        "model": model,
        "method": method,
        "scope": scope,
        "runtime": runtime,
        "experiment": experiment,
        "train_data_path": _repo_path(train_data_path) if train_data_path else None,
        "run_name": run_name,
        "rank": rank,
    }


def _resolve_evaluation_tasks(args: Namespace) -> tuple[str, ...] | None:
    if args.tasks:
        return _parse_task_selection(args.tasks)
    return _parse_task_selection(_env_value("EVAL_TASKS"))


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

    assets = subparsers.add_parser("prepare-assets")
    assets.add_argument("--cache-dir", default="data/cache")
    assets.add_argument("--train-source", default="auto")
    assets.add_argument("--limit", type=int, default=None)
    assets.add_argument(
        "--models",
        nargs="*",
        default=(),
        help="Model preset names to download locally, or pass 'all' to fetch every configured preset.",
    )

    train = subparsers.add_parser("train")
    train.add_argument("--model", default=None)
    train.add_argument("--method", choices=METHOD_CHOICES, default=None)
    train.add_argument(
        "--scope",
        choices=SCOPE_CHOICES,
        default=None,
    )
    train.add_argument("--runtime", default=None)
    train.add_argument("--experiment", default=None)
    train.add_argument("--train-data-path", default=None)
    train.add_argument("--output-dir", default="results/runs")
    train.add_argument("--run-name", default=None)
    train.add_argument("--resume-from-checkpoint", default=None)
    train.add_argument("--rank", type=int, default=None)

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


def _resolve_requested_models(raw_models: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    configured = available_model_presets()
    if not raw_models:
        return ()
    if "all" in raw_models:
        return configured
    unknown = sorted(set(raw_models) - set(configured))
    if unknown:
        msg = f"Unknown model presets requested: {', '.join(unknown)}"
        raise ValueError(msg)
    return tuple(raw_models)


def _prepare_assets_command(args: Namespace, logger: logging.Logger) -> int:
    cache_dir = _repo_path(args.cache_dir)
    configure_logging(args.log_level, log_path=cache_dir / "logs" / "prepare-assets.log")
    command_logger = bind_logger(logger, command=args.command, cache_dir=cache_dir)
    requested_models = _resolve_requested_models(args.models)
    command_logger.info(
        "Preparing local assets", extra={"requested_models": requested_models or "none"}
    )
    normalized_train = normalize_training_data(args.train_source, cache_dir)
    normalized_eval = normalize_all_benchmarks(cache_dir, TASKS, limit=args.limit)
    sys.stdout.write(f"training={normalized_train}\n")
    for task, path in normalized_eval.items():
        sys.stdout.write(f"{task}={path}\n")
    for model_name in requested_models:
        spec = build_experiment(
            model_name=model_name,
            method="dora",
            scope="full",
            runtime_name="official",
        )
        snapshot_dir = prefetch_model_to_hf_cache(
            model_name=model_name,
            model_id=spec.model.model_id,
        )
        sys.stdout.write(f"{model_name}={snapshot_dir}\n")
    command_logger.info(
        "Prepared local assets",
        extra={"task_count": len(normalized_eval), "model_count": len(requested_models)},
    )
    return 0


def _train_command(args: Namespace, logger: logging.Logger) -> int:
    resolved = _resolve_train_settings(args)
    model_name = str(resolved["model"])
    method = cast("AdapterMethod", resolved["method"])
    scope = cast("TargetScope", resolved["scope"])
    runtime_name = str(resolved["runtime"])
    experiment_name = str(resolved["experiment"])
    train_data_path = resolved["train_data_path"]
    rank = cast("int | None", resolved.get("rank"))
    run_name = cast(
        "str | None",
        resolved["run_name"],
    ) or _default_run_name(model_name, method, scope, runtime_name)
    run_dir = _repo_path(args.output_dir) / run_name
    configure_logging(args.log_level, log_path=run_dir / "logs" / "train.log")
    command_logger = bind_logger(
        logger,
        command=args.command,
        run_name=run_name,
        model=model_name,
        method=method,
        scope=scope,
        runtime=runtime_name,
    )
    command_logger.info("Building experiment spec")
    if model_name == "tiny_debug" and experiment_name == "default" and train_data_path is None:
        command_logger.warning(
            "tiny_debug with the default experiment still uses the full paper-scale "
            "training set; use --experiment debug_quick for a fast Colab sanity run"
        )
    spec = build_experiment(
        model_name=model_name,
        method=method,
        scope=scope,
        runtime_name=runtime_name,
        experiment_name=experiment_name,
        train_data_path=cast("Path | None", train_data_path),
        override_rank=rank,
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
    tasks = _resolve_evaluation_tasks(args)
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
        "prepare-assets": _prepare_assets_command,
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
