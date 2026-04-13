"""Evaluation helpers for trained adapters."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dora_repro.auth import resolve_hf_token
from dora_repro.config import build_experiment
from dora_repro.data import normalize_benchmark_task
from dora_repro.prompts import extract_prediction
from dora_repro.results import macro_average, write_json, write_jsonl

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from dora_repro.config import AdapterMethod, ExperimentSpec, TargetScope
    from dora_repro.prompts import EvalSample


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


def load_trained_model(
    run_dir: Path, spec: ExperimentSpec
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a trained adapter for evaluation."""
    token = resolve_hf_token()
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
    device = torch.device(_device())
    moved_model = cast("Any", model).to(device)
    return cast("PreTrainedModel", moved_model), tokenizer


def _generate_predictions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: list[EvalSample],
    max_new_tokens: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    device_name = _device()
    for sample in samples:
        encoded = tokenizer(
            sample.instruction,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        encoded = {key: value.to(device_name) for key, value in encoded.items()}
        with torch.inference_mode():
            output = cast("Any", model).generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        generated_ids = output[:, encoded["input_ids"].shape[1] :]
        decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        decoded = (
            decoded_output[0].strip()
            if isinstance(decoded_output, list)
            else decoded_output.strip()
        )
        prediction = extract_prediction(sample.task, decoded)
        results.append(
            {
                "id": sample.id,
                "task": sample.task,
                "label": sample.label,
                "prediction": prediction,
                "correct": prediction == sample.label,
                "generated_text": decoded,
            },
        )
    return results


def evaluate_run(run_dir: Path, task_names: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Evaluate one run on one or more benchmark tasks."""
    spec = load_spec_from_snapshot(run_dir)
    requested_tasks = spec.task_names if task_names is None else task_names
    model, tokenizer = load_trained_model(run_dir, spec)
    per_task_accuracy: dict[str, float] = {}

    for task in requested_tasks:
        samples = normalize_benchmark_task(task)
        predictions = _generate_predictions(model, tokenizer, samples, spec.max_new_tokens)
        output_path = run_dir / "predictions" / f"{task}.jsonl"
        write_jsonl(output_path, predictions)
        correct = sum(1 for item in predictions if bool(item["correct"]))
        per_task_accuracy[task] = correct / len(predictions)

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
    return metrics
