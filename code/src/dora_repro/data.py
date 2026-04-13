"""Data download and normalization helpers."""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from datasets import load_dataset

from dora_repro.config import TASKS, repo_root
from dora_repro.logging_utils import bind_logger
from dora_repro.prompts import EvalSample, TrainingSample

logger = logging.getLogger(__name__)

DEFAULT_TRAINING_URL = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/"
    "ft-training_set/commonsense_170k.json"
)
SCRIPT_BACKED_BENCHMARK_URLS: dict[str, str] = {
    "piqa": "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip",
    "social_i_qa": "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip",
}


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> Path:
    """Write records as JSONL."""
    ensure_dir(output_path.parent)
    lines = [json.dumps(record, sort_keys=True) for record in records]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def default_cache_dir(root: Path | None = None) -> Path:
    """Return the cache directory used by prepare-data."""
    base = repo_root() if root is None else root
    return base / "data" / "cache"


def benchmark_data_dir(root: Path | None = None) -> Path:
    """Return the local directory used for normalized benchmark JSONL files."""
    base = repo_root() if root is None else root
    return base / "data" / "benchmarks"


def resolve_training_source(train_source: str, cache_dir: Path) -> Path:
    """Resolve the commonsense training data from auto/path/URL inputs."""
    if train_source == "auto":
        local = repo_root() / "data" / "commonsense_170k.json"
        if local.is_file():
            bind_logger(logger, path=local).info("Using local commonsense training data")
            return local
        train_source = DEFAULT_TRAINING_URL

    candidate = Path(train_source).expanduser()
    if candidate.is_file():
        bind_logger(logger, path=candidate).info("Using explicit training data path")
        return candidate
    if train_source.startswith(("http://", "https://")):
        target = ensure_dir(cache_dir / "raw") / "commonsense_170k.json"
        bind_logger(logger, url=train_source, target=target).info("Downloading training data")
        response = httpx.get(train_source, timeout=60.0)
        response.raise_for_status()
        target.write_bytes(response.content)
        bind_logger(logger, path=target).info("Downloaded training data")
        return target
    msg = f"Unsupported training source: {train_source}"
    raise ValueError(msg)


def load_training_samples(train_source: Path) -> list[TrainingSample]:
    """Load the supervised fine-tuning dataset."""
    with train_source.open(encoding="utf-8") as handle:
        raw_data = json.load(handle)
    return [
        TrainingSample(
            instruction=str(item["instruction"]),
            input=str(item.get("input", "")),
            output=str(item["output"]),
        )
        for item in raw_data
    ]


def normalize_training_data(
    train_source: str = "auto",
    cache_dir: Path | None = None,
) -> Path:
    """Normalize the commonsense training JSON into JSONL."""
    resolved_cache = default_cache_dir() if cache_dir is None else cache_dir
    source = resolve_training_source(train_source, resolved_cache)
    output_path = resolved_cache / "normalized" / "train" / "commonsense_170k.jsonl"
    records = [asdict(sample) for sample in load_training_samples(source)]
    bind_logger(logger, records=len(records), output_path=output_path).info(
        "Normalized training data"
    )
    return write_jsonl(records, output_path)


def _boolq_instruction(row: dict[str, Any]) -> EvalSample:
    answer = "true" if bool(row["answer"]) else "false"
    instruction = (
        "Please answer the following question with true or false.\n\n"
        f"Passage: {row['passage']}\n"
        f"Question: {row['question']}\n\n"
        "Answer format: true/false"
    )
    return EvalSample(
        id=str(row.get("id", row["question"])),
        task="boolq",
        instruction=instruction,
        choices=("true", "false"),
        label=answer,
    )


def _piqa_instruction(row: dict[str, Any]) -> EvalSample:
    label = f"solution{int(row['label']) + 1}"
    instruction = (
        "Choose the better solution to the goal.\n\n"
        f"Goal: {row['goal']}\n"
        f"solution1: {row['sol1']}\n"
        f"solution2: {row['sol2']}\n\n"
        "Answer format: solution1/solution2"
    )
    return EvalSample(
        id=str(row.get("id", row["goal"])),
        task="piqa",
        instruction=instruction,
        choices=(str(row["sol1"]), str(row["sol2"])),
        label=label,
    )


def _social_instruction(row: dict[str, Any]) -> EvalSample:
    label = f"answer{int(row['label'])}"
    instruction = (
        "Choose the best answer to the social commonsense question.\n\n"
        f"Context: {row['context']}\n"
        f"Question: {row['question']}\n"
        f"answer1: {row['answerA']}\n"
        f"answer2: {row['answerB']}\n"
        f"answer3: {row['answerC']}\n\n"
        "Answer format: answer1/answer2/answer3"
    )
    return EvalSample(
        id=str(row.get("id", row["question"])),
        task="social_i_qa",
        instruction=instruction,
        choices=(str(row["answerA"]), str(row["answerB"]), str(row["answerC"])),
        label=label,
    )


def _hellaswag_instruction(row: dict[str, Any]) -> EvalSample:
    label = f"ending{int(row['label']) + 1}"
    endings = tuple(str(ending) for ending in row["endings"])
    instruction = (
        "Choose the best ending for the given context.\n\n"
        f"Context: {row['ctx']}\n"
        f"ending1: {endings[0]}\n"
        f"ending2: {endings[1]}\n"
        f"ending3: {endings[2]}\n"
        f"ending4: {endings[3]}\n\n"
        "Answer format: ending1/ending2/ending3/ending4"
    )
    return EvalSample(
        id=str(row.get("ind", row["ctx"])),
        task="hellaswag",
        instruction=instruction,
        choices=endings,
        label=label,
    )


def _winogrande_instruction(row: dict[str, Any]) -> EvalSample:
    label = f"option{row['answer']}"
    instruction = (
        "Choose the best option to fill in the blank.\n\n"
        f"Sentence: {row['sentence']}\n"
        f"option1: {row['option1']}\n"
        f"option2: {row['option2']}\n\n"
        "Answer format: option1/option2"
    )
    return EvalSample(
        id=str(row.get("qID", row["sentence"])),
        task="winogrande",
        instruction=instruction,
        choices=(str(row["option1"]), str(row["option2"])),
        label=label,
    )


def _label_to_index(label: str) -> int:
    mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    return mapping[str(label)]


def _multiple_choice_instruction(task: str, row: dict[str, Any]) -> EvalSample:
    question = row.get("question")
    question_text = (
        question["stem"] if isinstance(question, dict) else str(question or row["question_stem"])
    )
    choices_payload = question["choices"] if isinstance(question, dict) else row["choices"]["text"]

    formatted_choices: list[str] = []
    instruction_lines = [
        "Choose the correct answer to the multiple-choice question.",
        "",
        f"Question: {question_text}",
    ]
    for index, choice in enumerate(choices_payload, start=1):
        text = str(choice["text"]) if isinstance(choice, dict) else str(choice)
        formatted_choices.append(text)
        instruction_lines.append(f"answer{index}: {text}")
    instruction_lines.extend(["", "Answer format: answer1/answer2/answer3/answer4/answer5"])
    return EvalSample(
        id=str(row.get("id", question_text)),
        task=task,
        instruction="\n".join(instruction_lines),
        choices=tuple(formatted_choices),
        label=f"answer{_label_to_index(str(row['answerKey']))}",
    )


def benchmark_specs() -> dict[str, tuple[str, str | None, str]]:
    """Return the Hugging Face dataset mapping for each benchmark."""
    return {
        "boolq": ("google/boolq", None, "validation"),
        "piqa": ("ybisk/piqa", None, "validation"),
        "social_i_qa": ("social_i_qa", None, "validation"),
        "hellaswag": ("Rowan/hellaswag", None, "validation"),
        "winogrande": ("allenai/winogrande", "winogrande_xl", "validation"),
        "ARC-Easy": ("allenai/ai2_arc", "ARC-Easy", "validation"),
        "ARC-Challenge": ("allenai/ai2_arc", "ARC-Challenge", "validation"),
        "openbookqa": ("allenai/openbookqa", "main", "validation"),
    }


def _download_bytes(url: str) -> bytes:
    response = httpx.get(url, timeout=120.0)
    response.raise_for_status()
    return response.content


def _load_script_backed_rows(task: str) -> list[dict[str, Any]]:
    archive = zipfile.ZipFile(BytesIO(_download_bytes(SCRIPT_BACKED_BENCHMARK_URLS[task])))
    if task == "piqa":
        inputs = archive.read("physicaliqa-train-dev/dev.jsonl").decode("utf-8").splitlines()
        labels = archive.read("physicaliqa-train-dev/dev-labels.lst").decode("utf-8").splitlines()
        return [
            {**json.loads(row), "label": label} for row, label in zip(inputs, labels, strict=True)
        ]
    if task == "social_i_qa":
        inputs = archive.read("socialiqa-train-dev/dev.jsonl").decode("utf-8").splitlines()
        labels = archive.read("socialiqa-train-dev/dev-labels.lst").decode("utf-8").splitlines()
        return [
            {**json.loads(row), "label": label} for row, label in zip(inputs, labels, strict=True)
        ]
    msg = f"No script-backed fallback is configured for task: {task}"
    raise ValueError(msg)


def normalize_benchmark_task(task: str, limit: int | None = None) -> list[EvalSample]:
    """Load and normalize one benchmark task."""
    if task not in TASKS:
        msg = f"Unsupported benchmark task: {task}"
        raise ValueError(msg)
    local_jsonl = benchmark_data_dir() / f"{task}.jsonl"
    if local_jsonl.is_file():
        rows = read_jsonl(local_jsonl)
        samples = [
            EvalSample(
                id=str(row["id"]),
                task=str(row["task"]),
                instruction=str(row["instruction"]),
                choices=tuple(str(choice) for choice in row["choices"]),
                label=str(row["label"]),
            )
            for row in rows[:limit]
        ]
        bind_logger(logger, task=task, path=local_jsonl, sample_count=len(samples)).info(
            "Loaded local benchmark task"
        )
        return samples
    dataset_id, subset, split = benchmark_specs()[task]
    task_logger = bind_logger(
        logger,
        task=task,
        dataset_id=dataset_id,
        subset=subset,
        split=split,
        limit=limit,
    )
    task_logger.info("Loading benchmark task")
    try:
        dataset = load_dataset(dataset_id, name=subset, split=split)
        rows = list(dataset if limit is None else dataset.select(range(min(limit, len(dataset)))))
    except RuntimeError as error:
        if "scripts are no longer supported" not in str(error):
            raise
        task_logger.warning("Falling back to manual benchmark download")
        rows = _load_script_backed_rows(task)
        if limit is not None:
            rows = rows[:limit]
    converters = {
        "boolq": _boolq_instruction,
        "piqa": _piqa_instruction,
        "social_i_qa": _social_instruction,
        "hellaswag": _hellaswag_instruction,
        "winogrande": _winogrande_instruction,
        "ARC-Easy": lambda row: _multiple_choice_instruction("ARC-Easy", row),
        "ARC-Challenge": lambda row: _multiple_choice_instruction("ARC-Challenge", row),
        "openbookqa": lambda row: _multiple_choice_instruction("openbookqa", row),
    }
    samples = [converters[task](dict(row)) for row in rows]
    bind_logger(logger, task=task, sample_count=len(samples)).info("Normalized benchmark task")
    return samples


def normalize_all_benchmarks(
    cache_dir: Path | None = None,
    task_names: tuple[str, ...] = TASKS,
    limit: int | None = None,
) -> dict[str, Path]:
    """Normalize all evaluation datasets to local JSONL files."""
    resolved_cache = default_cache_dir() if cache_dir is None else cache_dir
    output_paths: dict[str, Path] = {}
    for task in task_names:
        samples = normalize_benchmark_task(task, limit=limit)
        output_path = benchmark_data_dir() / f"{task}.jsonl"
        output_paths[task] = write_jsonl([asdict(sample) for sample in samples], output_path)
        bind_logger(logger, task=task, output_path=output_path).info("Wrote benchmark JSONL")
    bind_logger(logger, cache_dir=resolved_cache).debug("Prepared benchmarks with cache directory")
    return output_paths


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
