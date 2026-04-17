from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from dora_repro import eval as eval_module
from dora_repro.config import AdapterPreset, ExperimentSpec, ModelPreset, RuntimePreset
from dora_repro.prompts import EvalSample
from dora_repro.train import run_training

if TYPE_CHECKING:
    from pathlib import Path

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"


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


def _build_small_llama_config() -> LlamaConfig:
    config = LlamaConfig()
    config.vocab_size = 32
    config.hidden_size = 16
    config.intermediate_size = 32
    config.num_hidden_layers = 1
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.pad_token_id = 0
    config.bos_token_id = 1
    config.eos_token_id = 2
    return config


def _write_local_model_bundle(root: Path) -> Path:
    model_dir = root / "model"
    model_dir.mkdir(parents=True)
    torch.manual_seed(0)
    LlamaForCausalLM(_build_small_llama_config()).save_pretrained(model_dir)
    vocab = {
        PAD_TOKEN: 0,
        BOS_TOKEN: 1,
        EOS_TOKEN: 2,
        UNK_TOKEN: 3,
        "Below": 4,
        "is": 5,
        "an": 6,
        "instruction": 7,
        "that": 8,
        "describes": 9,
        "a": 10,
        "task": 11,
        "Write": 12,
        "response": 13,
        "appropriately": 14,
        "completes": 15,
        "the": 16,
        "request": 17,
        "Instruction": 18,
        "Response": 19,
        "Answer": 20,
        "true": 21,
        "false": 22,
        "or": 23,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
    )
    fast.model_max_length = 128
    fast.save_pretrained(model_dir)
    return model_dir


def _write_training_json(path: Path) -> None:
    path.write_text(
        json.dumps([
            {"instruction": "Answer true or false.", "input": "", "output": "true"},
            {"instruction": "Answer true or false.", "input": "", "output": "false"},
        ]),
        encoding="utf-8",
    )


def _local_spec(root: Path) -> ExperimentSpec:
    model_dir = _write_local_model_bundle(root)
    train_path = root / "train.json"
    _write_training_json(train_path)
    return ExperimentSpec(
        experiment_name="local",
        model=ModelPreset(
            name="local_llama",
            model_id=str(model_dir),
            learning_rate=5e-4,
            cutoff_len=64,
        ),
        adapter=AdapterPreset(method="dora", scope="attention_only", rank=2, alpha=4),
        runtime=RuntimePreset(
            name="cpu",
            per_device_batch_size=1,
            effective_batch_size=1,
            gradient_checkpointing=False,
            use_4bit=False,
        ),
        train_data_path=train_path,
        task_names=("boolq",),
        max_train_samples=2,
        val_set_size=0,
        num_epochs=1,
        save_steps=10,
        eval_steps=10,
        seed=0,
        max_new_tokens=2,
    )


def test_generate_predictions_batches_and_sets_generation_token_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dora_repro.eval.tqdm", lambda items, **_: items)
    monkeypatch.setattr("dora_repro.eval._device", lambda: "cpu")
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    samples = [
        EvalSample(id="a", task="boolq", instruction="Q1", choices=("true", "false"), label="true"),
        EvalSample(
            id="b",
            task="boolq",
            instruction="Q2",
            choices=("true", "false"),
            label="false",
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


def test_evaluation_batch_size_scales_with_runtime_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dora_repro.eval._device", lambda: "cuda")

    class _Runtime:
        per_device_batch_size = 2

    class _Spec:
        runtime = _Runtime()

    assert eval_module._evaluation_batch_size(cast("ExperimentSpec", _Spec())) == 8


def test_load_spec_from_snapshot_preserves_saved_values(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train_path = tmp_path / "custom-train.json"
    (run_dir / "config.snapshot.toml").write_text(
        f"""
experiment_name = "custom_eval"
train_data_path = "{train_path}"
task_names = ["hellaswag", "boolq"]
max_train_samples = 1234
val_set_size = 17
num_epochs = 9
save_steps = 321
eval_steps = 654
weight_decay = 0.125
train_on_inputs = true
seed = 99
max_new_tokens = 11

[model]
name = "llama3_8b"
model_id = "meta-llama/Meta-Llama-3-8B"
learning_rate = 0.0003
cutoff_len = 512
trust_remote_code = false

[adapter]
method = "dora"
scope = "attention_only"
rank = 13
alpha = 37
dropout = 0.17
target_modules = ["q_proj", "k_proj", "v_proj"]

[runtime]
name = "custom_runtime"
per_device_batch_size = 3
effective_batch_size = 12
gradient_checkpointing = true
use_4bit = true
bnb_4bit_quant_type = "nf4"
bnb_4bit_compute_dtype = "bfloat16"
bnb_4bit_use_double_quant = false
""".strip(),
        encoding="utf-8",
    )

    spec = eval_module.load_spec_from_snapshot(run_dir)
    assert spec.experiment_name == "custom_eval"
    assert spec.train_data_path == train_path
    assert spec.task_names == ("hellaswag", "boolq")
    assert spec.max_train_samples == 1234
    assert spec.val_set_size == 17
    assert spec.num_epochs == 9
    assert spec.save_steps == 321
    assert spec.eval_steps == 654
    assert spec.weight_decay == pytest.approx(0.125)
    assert spec.train_on_inputs is True
    assert spec.seed == 99
    assert spec.max_new_tokens == 11
    assert spec.model.model_id == "meta-llama/Meta-Llama-3-8B"
    assert spec.model.learning_rate == pytest.approx(3e-4)
    assert spec.model.cutoff_len == 512
    assert spec.adapter.method == "dora"
    assert spec.adapter.scope == "attention_only"
    assert spec.adapter.rank == 13
    assert spec.adapter.alpha == 37
    assert spec.adapter.dropout == pytest.approx(0.17)
    assert spec.runtime.name == "custom_runtime"
    assert spec.runtime.per_device_batch_size == 3
    assert spec.runtime.effective_batch_size == 12
    assert spec.runtime.gradient_checkpointing is True
    assert spec.runtime.use_4bit is True
    assert spec.runtime.bnb_4bit_use_double_quant is False


def test_evaluate_run_uses_snapshot_tasks_when_no_override(
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch: pytest.MonkeyPatch,
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


def test_train_then_evaluate_run_uses_local_adapter_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _local_spec(tmp_path / "bundle")
    run_dir = tmp_path / "run"
    run_training(spec, run_dir)

    monkeypatch.setattr(
        eval_module,
        "normalize_benchmark_task",
        lambda _task: [
            EvalSample(
                id="1",
                task="boolq",
                instruction="Answer true or false.",
                choices=("true", "false"),
                label="true",
            )
        ],
    )
    monkeypatch.setattr(eval_module, "extract_prediction", lambda _task, _text: "true")

    metrics = eval_module.evaluate_run(run_dir)
    assert metrics["boolq"] == pytest.approx(1.0)
    assert (run_dir / "predictions" / "boolq.jsonl").is_file()
    assert (run_dir / "metrics.json").is_file()
