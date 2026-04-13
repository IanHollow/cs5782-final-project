from dora_repro.config import (
    TARGET_MODULES,
    build_experiment,
    load_model_preset,
    load_runtime_preset,
)


def test_load_model_and_runtime_presets() -> None:
    model = load_model_preset("llama2_7b")
    runtime = load_runtime_preset("colab_t4")
    assert model.model_id == "meta-llama/Llama-2-7b-hf"
    assert runtime.use_4bit is True
    assert runtime.gradient_accumulation_steps == 16


def test_build_experiment_uses_scope_modules() -> None:
    spec = build_experiment(
        model_name="llama2_7b",
        method="dora",
        scope="attention_only",
        runtime_name="official",
    )
    assert spec.adapter.target_modules == TARGET_MODULES["attention_only"]
    assert spec.task_names[0] == "boolq"


def test_build_debug_quick_experiment_uses_small_training_subset() -> None:
    spec = build_experiment(
        model_name="tiny_debug",
        method="dora",
        scope="attention_only",
        runtime_name="official",
        experiment_name="debug_quick",
    )
    assert spec.train_data_path.name == "commonsense_15k.json"
    assert spec.max_train_samples == 512
    assert spec.num_epochs == 1
