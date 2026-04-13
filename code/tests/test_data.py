from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from datasets import Dataset

from dora_repro import data

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("task", "row", "expected_label"),
    [
        ("boolq", {"question": "Q", "passage": "P", "answer": True}, "true"),
        ("piqa", {"goal": "G", "sol1": "A", "sol2": "B", "label": 1}, "solution2"),
        (
            "social_i_qa",
            {
                "context": "C",
                "question": "Q",
                "answerA": "A",
                "answerB": "B",
                "answerC": "C",
                "label": "2",
            },
            "answer2",
        ),
        ("hellaswag", {"ctx": "C", "endings": ["a", "b", "c", "d"], "label": 0}, "ending1"),
        (
            "winogrande",
            {"sentence": "The _ is here", "option1": "cat", "option2": "dog", "answer": "2"},
            "option2",
        ),
        (
            "ARC-Easy",
            {
                "question": {
                    "stem": "Q",
                    "choices": [{"text": "A"}, {"text": "B"}, {"text": "C"}, {"text": "D"}],
                },
                "answerKey": "B",
            },
            "answer2",
        ),
        (
            "ARC-Challenge",
            {
                "question": {
                    "stem": "Q",
                    "choices": [{"text": "A"}, {"text": "B"}, {"text": "C"}, {"text": "D"}],
                },
                "answerKey": "D",
            },
            "answer4",
        ),
        (
            "openbookqa",
            {
                "question_stem": "Q",
                "choices": {"text": ["A", "B", "C", "D"]},
                "answerKey": "1",
            },
            "answer1",
        ),
    ],
)
def test_normalize_benchmark_task(
    monkeypatch: pytest.MonkeyPatch, task: str, row: dict[str, Any], expected_label: str
) -> None:
    def fake_load_dataset(
        dataset_id: str, name: str | None = None, split: str | None = None
    ) -> Dataset:
        del dataset_id, name, split
        return Dataset.from_list([row])

    monkeypatch.setattr(data, "load_dataset", fake_load_dataset)
    normalized = data.normalize_benchmark_task(task)
    assert len(normalized) == 1
    assert normalized[0].label == expected_label


def test_normalize_training_data_uses_local_file(tmp_path: Path) -> None:
    source = tmp_path / "commonsense_170k.json"
    source.write_text(
        '[{"instruction": "Q", "input": "", "output": "A"}]',
        encoding="utf-8",
    )
    output_path = data.normalize_training_data(str(source), tmp_path / "cache")
    contents = output_path.read_text(encoding="utf-8")
    assert '"instruction": "Q"' in contents
