"""Training entrypoints for LoRA/DoRA fine-tuning."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, cast

import torch
from datasets import Dataset
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)

from dora_repro.adapters import attach_adapter
from dora_repro.auth import resolve_hf_token
from dora_repro.config import AdapterPreset
from dora_repro.logging_utils import bind_logger
from dora_repro.prompts import TrainingSample, format_training_prompt, format_user_prompt
from dora_repro.results import ensure_dir, write_snapshot

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from dora_repro.config import ExperimentSpec


logger = logging.getLogger(__name__)


def _read_training_json(path: Path) -> list[TrainingSample]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    return [
        TrainingSample(
            instruction=str(row["instruction"]),
            input=str(row.get("input", "")),
            output=str(row["output"]),
        )
        for row in rows
    ]


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    return mapping[name]


def load_model_and_tokenizer(
    spec: ExperimentSpec,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the base model and tokenizer for training or evaluation."""
    contextual_logger = bind_logger(
        logger,
        model=spec.model.name,
        method=spec.adapter.method,
        scope=spec.adapter.scope,
        runtime=spec.runtime.name,
    )
    token = resolve_hf_token()
    quantization_config = None
    if spec.runtime.use_4bit:
        contextual_logger.info("Configuring 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=spec.runtime.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=_torch_dtype(spec.runtime.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=spec.runtime.bnb_4bit_use_double_quant,
        )
    contextual_logger.info("Loading base model and tokenizer", extra={"model_id": spec.model.model_id})
    model = cast(
        "PreTrainedModel",
        AutoModelForCausalLM.from_pretrained(
            spec.model.model_id,
            token=token,
            trust_remote_code=spec.model.trust_remote_code,
            quantization_config=quantization_config,
        ),
    )
    tokenizer = cast(
        "PreTrainedTokenizerBase",
        AutoTokenizer.from_pretrained(
            spec.model.model_id,
            token=token,
            trust_remote_code=spec.model.trust_remote_code,
        ),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if spec.runtime.use_4bit:
        contextual_logger.info("Preparing model for k-bit training")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=spec.runtime.gradient_checkpointing,
        )
    elif spec.runtime.gradient_checkpointing:
        contextual_logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = attach_adapter(model, spec.adapter)
    contextual_logger.info("Attached adapter")
    return cast("PreTrainedModel", model), tokenizer


def _tokenize_example(
    sample: TrainingSample,
    tokenizer: PreTrainedTokenizerBase,
    cutoff_len: int,
    *,
    train_on_inputs: bool,
) -> dict[str, list[int]]:
    full_prompt = format_training_prompt(sample)
    tokenized_full = tokenizer(
        full_prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
    )
    input_ids = list(tokenized_full["input_ids"])
    if input_ids[-1] != tokenizer.eos_token_id and len(input_ids) < cutoff_len:
        input_ids.append(tokenizer.eos_token_id)
    attention_mask = [1] * len(input_ids)
    labels = list(input_ids)
    if not train_on_inputs:
        prompt_only = tokenizer(
            format_user_prompt(sample),
            truncation=True,
            max_length=cutoff_len,
            padding=False,
        )
        prompt_len = len(prompt_only["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _prepare_dataset(
    spec: ExperimentSpec, tokenizer: PreTrainedTokenizerBase
) -> tuple[Dataset, Dataset | None]:
    samples = _read_training_json(spec.train_data_path)
    records = [
        _tokenize_example(
            sample=sample,
            tokenizer=tokenizer,
            cutoff_len=spec.model.cutoff_len,
            train_on_inputs=spec.train_on_inputs,
        )
        for sample in samples
    ]
    dataset = Dataset.from_list(records)
    if spec.val_set_size <= 0 or spec.val_set_size >= len(dataset):
        return dataset, None
    split = dataset.train_test_split(test_size=spec.val_set_size, seed=spec.seed, shuffle=True)
    return split["train"], split["test"]


def _half_precision_flags() -> tuple[bool, bool]:
    if not torch.cuda.is_available():
        return False, False
    return torch.cuda.is_bf16_supported(), not torch.cuda.is_bf16_supported()


def run_training(
    spec: ExperimentSpec, output_dir: Path, resume_from_checkpoint: Path | None = None
) -> Path:
    """Run fine-tuning and save the adapter to the run directory."""
    run_logger = bind_logger(
        logger,
        run_name=output_dir.name,
        model=spec.model.name,
        method=spec.adapter.method,
        scope=spec.adapter.scope,
    )
    ensure_dir(output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    run_logger.info("Writing run snapshot", extra={"snapshot_path": output_dir / "config.snapshot.toml"})
    write_snapshot(output_dir / "config.snapshot.toml", spec)
    model, tokenizer = load_model_and_tokenizer(spec)
    train_dataset, eval_dataset = _prepare_dataset(spec, tokenizer)
    run_logger.info(
        "Prepared tokenized datasets",
        extra={
            "train_examples": len(train_dataset),
            "eval_examples": 0 if eval_dataset is None else len(eval_dataset),
        },
    )
    bf16, fp16 = _half_precision_flags()
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=TrainingArguments(
            output_dir=str(checkpoints_dir),
            do_train=True,
            do_eval=eval_dataset is not None,
            eval_strategy="steps" if eval_dataset is not None else "no",
            per_device_train_batch_size=spec.runtime.per_device_batch_size,
            gradient_accumulation_steps=spec.runtime.gradient_accumulation_steps,
            num_train_epochs=spec.num_epochs,
            learning_rate=spec.model.learning_rate,
            weight_decay=spec.weight_decay,
            logging_steps=10,
            save_strategy="steps",
            save_steps=spec.save_steps,
            eval_steps=spec.eval_steps if eval_dataset is not None else None,
            save_total_limit=3,
            seed=spec.seed,
            fp16=fp16,
            bf16=bf16,
            gradient_checkpointing=spec.runtime.gradient_checkpointing,
            remove_unused_columns=False,
            report_to=[],
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            padding=True,
            return_tensors="pt",
        ),
    )
    trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint)
        if resume_from_checkpoint is not None
        else None
    )
    run_logger.info("Training loop finished")
    adapter_dir = ensure_dir(output_dir / "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    metadata = {
        "adapter": asdict(spec.adapter),
        "model_id": spec.model.model_id,
        "runtime": asdict(spec.runtime),
    }
    (output_dir / "run.metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    run_logger.info("Saved adapter artifacts", extra={"adapter_dir": adapter_dir})
    return adapter_dir


def smoke_test(output_dir: Path) -> Path:
    """Run a fast, network-free sanity check of adapter wiring."""
    smoke_logger = bind_logger(logger, output_dir=output_dir)
    ensure_dir(output_dir)
    config = LlamaConfig()
    config.vocab_size = 128
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    model = LlamaForCausalLM(config)

    adapted = attach_adapter(model, AdapterPreset(method="dora", scope="attention_only"))
    inputs = torch.randint(0, 128, (2, 8))
    labels = inputs.clone()
    loss = adapted(input_ids=inputs, labels=labels).loss
    loss.backward()
    payload = {
        "loss": float(loss.detach().cpu()),
        "trainable_parameters": sum(
            param.numel() for param in adapted.parameters() if param.requires_grad
        ),
    }
    output_path = output_dir / "smoke_test.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    smoke_logger.info("Smoke test completed", extra={"output_path": output_path})
    return output_path
