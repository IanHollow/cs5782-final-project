"""Prompt formatting and output parsing for commonsense benchmarks."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class EvalSample:
    """Normalized evaluation example."""

    id: str
    task: str
    instruction: str
    choices: tuple[str, ...]
    label: str


@dataclass(slots=True, frozen=True)
class TrainingSample:
    """Normalized training example."""

    instruction: str
    input: str
    output: str


def format_training_prompt(sample: TrainingSample) -> str:
    """Return the Alpaca-style supervised prompt."""
    if sample.input:
        return (
            "Below is an instruction that describes a task, paired with an input that provides "
            "further context. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{sample.instruction}\n\n"
            f"### Input:\n{sample.input}\n\n"
            f"### Response:\n{sample.output}"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately "
        "completes the request.\n\n"
        f"### Instruction:\n{sample.instruction}\n\n"
        f"### Response:\n{sample.output}"
    )


def format_user_prompt(sample: TrainingSample) -> str:
    """Return the prompt prefix without the gold response."""
    if sample.input:
        return (
            "Below is an instruction that describes a task, paired with an input that provides "
            "further context. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{sample.instruction}\n\n"
            f"### Input:\n{sample.input}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately "
        "completes the request.\n\n"
        f"### Instruction:\n{sample.instruction}\n\n"
        "### Response:\n"
    )


def extract_prediction(task: str, generated_text: str) -> str:
    """Extract a normalized prediction token from generated text."""
    normalized = generated_text.strip().lower()
    patterns = {
        "boolq": r"\b(true|false)\b",
        "piqa": r"\b(solution[12])\b",
        "social_i_qa": r"\b(answer[1-5])\b",
        "ARC-Easy": r"\b(answer[1-5])\b",
        "ARC-Challenge": r"\b(answer[1-5])\b",
        "openbookqa": r"\b(answer[1-5])\b",
        "hellaswag": r"\b(ending[1-4])\b",
        "winogrande": r"\b(option[12])\b",
    }
    match = re.search(patterns[task], normalized)
    return match.group(1) if match else ""
