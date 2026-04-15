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

    l4_runtime = load_runtime_preset("colab_l4_llama")
    assert l4_runtime.per_device_batch_size == 4
    assert l4_runtime.gradient_accumulation_steps == 4
    assert l4_runtime.gradient_checkpointing is False


def test_build_experiment_uses_scope_modules() -> None:
    spec = build_experiment(
        model_name="llama2_7b",
        method="dora",
        scope="attention_only",
        runtime_name="official",
    )
    assert spec.adapter.target_modules == TARGET_MODULES["attention_only"]
    assert spec.task_names[0] == "boolq"
    assert spec.train_on_inputs is False


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
    assert spec.train_on_inputs is False


def test_build_paper_colab_experiment_uses_lower_overhead_schedule() -> None:
    spec = build_experiment(
        model_name="llama3_8b",
        method="dora",
        scope="full",
        runtime_name="colab_l4_llama",
        experiment_name="paper_colab",
    )
    assert spec.num_epochs == 3
    assert spec.train_data_path.name == "commonsense_15k.json"
    assert spec.save_steps == 200
    assert spec.eval_steps == 200
    assert spec.train_on_inputs is False


def test_non_debug_presets_default_to_15k_dataset() -> None:
    default_spec = build_experiment(
        model_name="llama2_7b",
        method="dora",
        scope="full",
        runtime_name="official",
        experiment_name="default",
    )
    paper_spec = build_experiment(
        model_name="llama2_7b",
        method="dora",
        scope="full",
        runtime_name="official",
        experiment_name="paper_llama2_7b",
    )
    assert default_spec.train_data_path.name == "commonsense_15k.json"
    assert paper_spec.train_data_path.name == "commonsense_15k.json"
