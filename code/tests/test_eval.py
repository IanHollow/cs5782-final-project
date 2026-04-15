from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from dora_repro import eval as eval_module
from dora_repro.prompts import EvalSample

if TYPE_CHECKING:
    from pathlib import Path

    from dora_repro.config import ExperimentSpec


class _FakeTokenizer:
    pad_token_id = 7
    eos_token_id = 9
    model_max_length = 32
    seen_texts: list[str]

    def __init__(self) -> None:
        self.seen_texts = []

    def __call__(
        self,
        text: str | list[str],
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int,
        padding: bool = False,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, truncation, max_length, padding
        texts = [text] if isinstance(text, str) else text
        self.seen_texts.extend(texts)
        batch = len(texts)
        return {
            "input_ids": torch.tensor([[1, 2, 3]] * batch),
            "attention_mask": torch.tensor([[1, 1, 1]] * batch),
        }

    @staticmethod
    def batch_decode(token_ids: torch.Tensor, *, skip_special_tokens: bool) -> list[str]:
        del token_ids, skip_special_tokens
        return ["The answer is true.", "The answer is false."]


class _FakeModel:
    def __init__(self) -> None:
        self.last_generate_kwargs: dict[str, object] | None = None

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.last_generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        continuation = torch.tensor([[10, 11], [12, 13]])
        return torch.cat([input_ids, continuation], dim=1)


def test_generate_predictions_batches_and_sets_generation_token_ids(monkeypatch) -> None:
    monkeypatch.setattr("dora_repro.eval.tqdm", lambda items, **_: items)
    monkeypatch.setattr("dora_repro.eval._device", lambda: "cpu")
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    samples = [
        EvalSample(id="a", task="boolq", instruction="Q1", choices=("true", "false"), label="true"),
        EvalSample(
            id="b", task="boolq", instruction="Q2", choices=("true", "false"), label="false"
        ),
    ]
    predictions = eval_module._generate_predictions(
        model=cast("Any", model),
        tokenizer=cast("Any", tokenizer),
        samples=samples,
        max_new_tokens=4,
        batch_size=2,
    )
    assert [item["prediction"] for item in predictions] == ["true", "false"]
    assert model.last_generate_kwargs is not None
    assert model.last_generate_kwargs["pad_token_id"] == tokenizer.pad_token_id
    assert model.last_generate_kwargs["eos_token_id"] == tokenizer.eos_token_id
    assert tokenizer.seen_texts[0].endswith("### Response:\n")
    assert "### Instruction:\nQ1" in tokenizer.seen_texts[0]


def test_evaluation_batch_size_scales_with_runtime_and_device(monkeypatch) -> None:
    monkeypatch.setattr("dora_repro.eval._device", lambda: "cuda")

    class _Runtime:
        per_device_batch_size = 2

    class _Spec:
        runtime = _Runtime()

    assert eval_module._evaluation_batch_size(cast("ExperimentSpec", _Spec())) == 8


def test_evaluate_run_uses_snapshot_tasks_when_no_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    seen_tasks: list[str] = []

    def fake_normalize_benchmark_task(task: str) -> list[EvalSample]:
        seen_tasks.append(task)
        return [
            EvalSample(id="1", task=task, instruction="Q", choices=("true", "false"), label="true")
        ]

    def fake_generate_predictions(
        _model: object,
        _tokenizer: object,
        _samples: list[EvalSample],
        _max_new_tokens: int,
        _batch_size: int,
    ) -> list[dict[str, object]]:
        return [
            {
                "id": "1",
                "task": seen_tasks[-1],
                "label": "true",
                "prediction": "true",
                "correct": True,
            }
        ]

    class _Runtime:
        name = "colab_l4_llama"
        per_device_batch_size = 1

    class _Model:
        name = "llama2_7b"
        model_id = "meta-llama/Llama-2-7b-hf"

    class _Adapter:
        method = "dora"
        scope = "attention_only"

    class _Spec:
        runtime = _Runtime()
        model = _Model()
        adapter = _Adapter()
        task_names = ("boolq", "piqa")
        max_new_tokens = 8

    monkeypatch.setattr(eval_module, "load_spec_from_snapshot", lambda _run_dir: _Spec())
    monkeypatch.setattr(eval_module, "load_trained_model", lambda *_args: (object(), object()))
    monkeypatch.setattr(eval_module, "_evaluation_batch_size", lambda _spec: 1)
    monkeypatch.setattr(eval_module, "normalize_benchmark_task", fake_normalize_benchmark_task)
    monkeypatch.setattr(eval_module, "_generate_predictions", fake_generate_predictions)
    monkeypatch.setattr(eval_module, "write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(eval_module, "write_jsonl", lambda *_args, **_kwargs: None)

    metrics = eval_module.evaluate_run(tmp_path)
    assert seen_tasks == ["boolq", "piqa"]
    assert metrics["boolq"] == pytest.approx(1.0)
    assert metrics["piqa"] == pytest.approx(1.0)


def test_evaluate_run_respects_single_and_multiple_task_overrides(
    monkeypatch,
    tmp_path: Path,
) -> None:
    seen_tasks: list[str] = []

    def fake_normalize_benchmark_task(task: str) -> list[EvalSample]:
        seen_tasks.append(task)
        return [
            EvalSample(id="1", task=task, instruction="Q", choices=("true", "false"), label="true")
        ]

    def fake_generate_predictions(
        _model: object,
        _tokenizer: object,
        _samples: list[EvalSample],
        _max_new_tokens: int,
        _batch_size: int,
    ) -> list[dict[str, object]]:
        return [
            {
                "id": "1",
                "task": seen_tasks[-1],
                "label": "true",
                "prediction": "true",
                "correct": True,
            }
        ]

    class _Runtime:
        name = "colab_l4_llama"
        per_device_batch_size = 1

    class _Model:
        name = "llama2_7b"
        model_id = "meta-llama/Llama-2-7b-hf"

    class _Adapter:
        method = "lora"
        scope = "mlp_only"

    class _Spec:
        runtime = _Runtime()
        model = _Model()
        adapter = _Adapter()
        task_names = ("boolq", "piqa")
        max_new_tokens = 8

    monkeypatch.setattr(eval_module, "load_spec_from_snapshot", lambda _run_dir: _Spec())
    monkeypatch.setattr(eval_module, "load_trained_model", lambda *_args: (object(), object()))
    monkeypatch.setattr(eval_module, "_evaluation_batch_size", lambda _spec: 1)
    monkeypatch.setattr(eval_module, "normalize_benchmark_task", fake_normalize_benchmark_task)
    monkeypatch.setattr(eval_module, "_generate_predictions", fake_generate_predictions)
    monkeypatch.setattr(eval_module, "write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(eval_module, "write_jsonl", lambda *_args, **_kwargs: None)

    eval_module.evaluate_run(tmp_path, ("hellaswag",))
    assert seen_tasks == ["hellaswag"]

    seen_tasks.clear()
    eval_module.evaluate_run(tmp_path, ("boolq", "hellaswag"))
    assert seen_tasks == ["boolq", "hellaswag"]
