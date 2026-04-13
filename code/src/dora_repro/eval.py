"""Evaluation helpers for trained adapters."""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dora_repro.auth import resolve_hf_token
from dora_repro.config import build_experiment
from dora_repro.data import normalize_benchmark_task
from dora_repro.logging_utils import bind_logger
from dora_repro.prompts import extract_prediction
from dora_repro.results import macro_average, write_json, write_jsonl

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from dora_repro.config import AdapterMethod, ExperimentSpec, TargetScope
    from dora_repro.prompts import EvalSample


logger = logging.getLogger(__name__)


def _load_snapshot(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_spec_from_snapshot(run_dir: Path) -> ExperimentSpec:
    """Reconstruct an experiment spec from a snapshot file."""
    payload = _load_snapshot(run_dir / "config.snapshot.toml")
    method = cast("AdapterMethod", str(payload["adapter"]["method"]))
    scope = cast("TargetScope", str(payload["adapter"]["scope"]))
    return build_experiment(
        model_name=str(payload["model"]["name"]),
        method=method,
        scope=scope,
        runtime_name=str(payload["runtime"]["name"]),
        experiment_name=str(payload["experiment_name"]),
        train_data_path=Path(str(payload["train_data_path"])),
    )


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _evaluation_batch_size(spec: ExperimentSpec) -> int:
    """Choose a conservative evaluation batch size for the active device."""
    device = _device()
    if device == "cuda":
        return max(1, spec.runtime.per_device_batch_size * 4)
    if device == "mps":
        return max(1, spec.runtime.per_device_batch_size * 2)
    return 1


def _batched[T](items: list[T], batch_size: int) -> list[list[T]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def load_trained_model(
    run_dir: Path, spec: ExperimentSpec
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a trained adapter for evaluation."""
    eval_logger = bind_logger(
        logger,
        run_name=run_dir.name,
        model=spec.model.name,
        method=spec.adapter.method,
        scope=spec.adapter.scope,
    )
    token = resolve_hf_token()
    eval_logger.info("Loading base model for evaluation", extra={"model_id": spec.model.model_id})
    base_model = cast(
        "PreTrainedModel",
        AutoModelForCausalLM.from_pretrained(
            spec.model.model_id,
            token=token,
            trust_remote_code=spec.model.trust_remote_code,
        ),
    )
    model = PeftModel.from_pretrained(base_model, run_dir / "adapter")
    if hasattr(model, "merge_and_unload"):
        eval_logger.info("Merging adapter weights into base model")
        model = model.merge_and_unload()
    model = cast("PreTrainedModel", model)
    model.eval()
    tokenizer = cast(
        "PreTrainedTokenizerBase",
        AutoTokenizer.from_pretrained(run_dir / "adapter"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    device = torch.device(_device())
    moved_model = cast("Any", model).to(device)
    eval_logger.info("Loaded evaluation model", extra={"device": device.type})
    return cast("PreTrainedModel", moved_model), tokenizer


def _generate_predictions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: list[EvalSample],
    max_new_tokens: int,
    batch_size: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    device_name = _device()
    for batch in tqdm(_batched(samples, batch_size), desc="Evaluating", unit="batch"):
        encoded = tokenizer(
            [sample.instruction for sample in batch],
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=True,
        )
        encoded = {key: value.to(device_name) for key, value in encoded.items()}
        with torch.inference_mode():
            output = cast("Any", model).generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output[:, encoded["input_ids"].shape[1] :]
        decoded_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for sample, decoded in zip(batch, decoded_batch, strict=True):
            stripped = decoded.strip()
            prediction = extract_prediction(sample.task, stripped)
            results.append(
                {
                    "id": sample.id,
                    "task": sample.task,
                    "label": sample.label,
                    "prediction": prediction,
                    "correct": prediction == sample.label,
                    "generated_text": stripped,
                },
            )
    return results


def evaluate_run(run_dir: Path, task_names: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Evaluate one run on one or more benchmark tasks."""
    spec = load_spec_from_snapshot(run_dir)
    requested_tasks = spec.task_names if task_names is None else task_names
    eval_logger = bind_logger(
        logger,
        run_name=run_dir.name,
        model=spec.model.name,
        method=spec.adapter.method,
        scope=spec.adapter.scope,
    )
    eval_logger.info("Starting evaluation", extra={"task_count": len(requested_tasks)})
    model, tokenizer = load_trained_model(run_dir, spec)
    batch_size = _evaluation_batch_size(spec)
    eval_logger.info("Resolved evaluation batch size", extra={"batch_size": batch_size})
    per_task_accuracy: dict[str, float] = {}

    for task in requested_tasks:
        eval_context = dict(eval_logger.extra) if eval_logger.extra is not None else {}
        task_logger = bind_logger(eval_logger.logger, **eval_context, task=task)
        samples = normalize_benchmark_task(task)
        task_logger.info("Loaded evaluation task", extra={"sample_count": len(samples)})
        predictions = _generate_predictions(
            model,
            tokenizer,
            samples,
            spec.max_new_tokens,
            batch_size,
        )
        output_path = run_dir / "predictions" / f"{task}.jsonl"
        write_jsonl(output_path, predictions)
        correct = sum(1 for item in predictions if bool(item["correct"]))
        per_task_accuracy[task] = correct / len(predictions)
        task_logger.info(
            "Finished evaluation task",
            extra={"accuracy": f"{per_task_accuracy[task]:.4f}", "output_path": output_path},
        )

    metrics = {
        "run_name": run_dir.name,
        "model_name": spec.model.name,
        "model_id": spec.model.model_id,
        "method": spec.adapter.method,
        "scope": spec.adapter.scope,
        "runtime": spec.runtime.name,
        **per_task_accuracy,
        "macro_average": macro_average(per_task_accuracy),
    }
    write_json(run_dir / "metrics.json", metrics)
    eval_logger.info(
        "Wrote evaluation metrics", extra={"macro_average": f"{metrics['macro_average']:.4f}"}
    )
    return metrics
