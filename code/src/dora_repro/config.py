"""Typed configuration loading for models, runtimes, and experiment presets."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

TASKS: tuple[str, ...] = (
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Easy",
    "ARC-Challenge",
    "openbookqa",
)
TargetScope = Literal["full", "attention_only", "mlp_only"]
AdapterMethod = Literal["lora", "dora"]
TARGET_MODULES: dict[TargetScope, tuple[str, ...]] = {
    "full": ("q_proj", "k_proj", "v_proj", "up_proj", "down_proj"),
    "attention_only": ("q_proj", "k_proj", "v_proj"),
    "mlp_only": ("up_proj", "down_proj"),
}
DEFAULT_TRAIN_DATA_PATH = "data/commonsense_15k.json"


@dataclass(slots=True, frozen=True)
class ModelPreset:
    """Base model settings."""

    name: str
    model_id: str
    learning_rate: float
    cutoff_len: int = 256
    trust_remote_code: bool = False


@dataclass(slots=True, frozen=True)
class AdapterPreset:
    """Adapter settings for LoRA or DoRA."""

    method: AdapterMethod
    scope: TargetScope
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05

    @property
    def target_modules(self) -> tuple[str, ...]:
        return TARGET_MODULES[self.scope]


@dataclass(slots=True, frozen=True)
class RuntimePreset:
    """Runtime settings for a specific machine profile."""

    name: str
    per_device_batch_size: int
    effective_batch_size: int
    gradient_checkpointing: bool
    use_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    @property
    def gradient_accumulation_steps(self) -> int:
        if self.effective_batch_size % self.per_device_batch_size != 0:
            msg = "effective_batch_size must be divisible by per_device_batch_size"
            raise ValueError(msg)
        return self.effective_batch_size // self.per_device_batch_size


@dataclass(slots=True, frozen=True)
class ExperimentSpec:
    """Top-level experiment settings."""

    experiment_name: str
    model: ModelPreset
    adapter: AdapterPreset
    runtime: RuntimePreset
    train_data_path: Path
    task_names: tuple[str, ...]
    max_train_samples: int | None = None
    val_set_size: int = 120
    num_epochs: int = 3
    save_steps: int = 80
    eval_steps: int = 80
    weight_decay: float = 0.0
    train_on_inputs: bool = False
    seed: int = 42
    max_new_tokens: int = 8

    def to_snapshot(self) -> dict[str, Any]:
        """Return a TOML-friendly snapshot of the experiment."""
        payload = asdict(self)
        payload["train_data_path"] = str(self.train_data_path)
        payload["task_names"] = list(self.task_names)
        payload["model"]["learning_rate"] = float(self.model.learning_rate)
        payload["adapter"]["target_modules"] = list(self.adapter.target_modules)
        return payload


def repo_root() -> Path:
    """Resolve the repository root from the installed package location."""
    return Path(__file__).resolve().parents[3]


def default_config_dir() -> Path:
    """Return the default configuration directory."""
    return repo_root() / "code" / "configs"


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_model_preset(name: str, config_dir: Path | None = None) -> ModelPreset:
    """Load a model preset by name."""
    base_dir = default_config_dir() if config_dir is None else config_dir
    payload = _load_toml(base_dir / "models" / f"{name}.toml")
    return ModelPreset(
        name=name,
        model_id=str(payload["model_id"]),
        learning_rate=float(payload["learning_rate"]),
        cutoff_len=int(payload.get("cutoff_len", 256)),
        trust_remote_code=bool(payload.get("trust_remote_code", False)),
    )


def load_runtime_preset(name: str, config_dir: Path | None = None) -> RuntimePreset:
    """Load a runtime preset by name."""
    base_dir = default_config_dir() if config_dir is None else config_dir
    payload = _load_toml(base_dir / "runtime" / f"{name}.toml")
    return RuntimePreset(
        name=name,
        per_device_batch_size=int(payload["per_device_batch_size"]),
        effective_batch_size=int(payload["effective_batch_size"]),
        gradient_checkpointing=bool(payload["gradient_checkpointing"]),
        use_4bit=bool(payload.get("use_4bit", False)),
        bnb_4bit_quant_type=str(payload.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=str(payload.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=bool(payload.get("bnb_4bit_use_double_quant", True)),
    )


def load_experiment_defaults(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Load an experiment preset TOML file."""
    base_dir = default_config_dir() if config_dir is None else config_dir
    return _load_toml(base_dir / "experiments" / f"{name}.toml")


def _method_overrides(defaults: dict[str, Any], method: AdapterMethod) -> dict[str, Any]:
    overrides = defaults.get("method_overrides", {})
    if not isinstance(overrides, dict):
        return {}
    selected = overrides.get(method, {})
    return selected if isinstance(selected, dict) else {}


def build_experiment(
    *,
    model_name: str,
    method: AdapterMethod,
    scope: TargetScope,
    runtime_name: str,
    experiment_name: str = "default",
    config_dir: Path | None = None,
    train_data_path: Path | None = None,
    override_rank: int | None = None,
) -> ExperimentSpec:
    """Build a validated experiment spec from named presets."""
    defaults = load_experiment_defaults(experiment_name, config_dir)
    model = load_model_preset(model_name, config_dir)
    runtime = load_runtime_preset(runtime_name, config_dir)
    method_defaults = _method_overrides(defaults, method)
    if "learning_rate" in method_defaults:
        model = replace(model, learning_rate=float(method_defaults["learning_rate"]))
    task_names = tuple(str(item) for item in defaults.get("task_names", TASKS))
    if any(task not in TASKS for task in task_names):
        msg = f"Unsupported task list: {task_names!r}"
        raise ValueError(msg)

    return ExperimentSpec(
        experiment_name=experiment_name,
        model=model,
        adapter=AdapterPreset(
            method=method,
            scope=scope,
            rank=override_rank if override_rank is not None else int(method_defaults.get("rank", defaults.get("rank", 8))),
            alpha=int(method_defaults.get("alpha", defaults.get("alpha", 16))),
            dropout=float(method_defaults.get("dropout", defaults.get("dropout", 0.05))),
        ),
        runtime=runtime,
        train_data_path=(
            train_data_path
            if train_data_path is not None
            else repo_root() / str(defaults.get("train_data_path", DEFAULT_TRAIN_DATA_PATH))
        ),
        task_names=task_names,
        max_train_samples=(
            None
            if defaults.get("max_train_samples") is None
            else int(defaults["max_train_samples"])
        ),
        val_set_size=int(defaults.get("val_set_size", 120)),
        num_epochs=int(defaults.get("num_epochs", 3)),
        save_steps=int(defaults.get("save_steps", 80)),
        eval_steps=int(defaults.get("eval_steps", 80)),
        weight_decay=float(defaults.get("weight_decay", 0.0)),
        train_on_inputs=bool(defaults.get("train_on_inputs", False)),
        seed=int(defaults.get("seed", 42)),
        max_new_tokens=int(defaults.get("max_new_tokens", 8)),
    )
