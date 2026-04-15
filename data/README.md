# Data

This directory now serves as the canonical local source for training data and benchmark datasets.

## Reproduction datasets

For the DoRA commonsense reproduction, the datasets that matter are:

- `data/commonsense_15k.json`
  The default supervised fine-tuning set used by the current practical Colab experiments.
- `data/commonsense_170k.json`
  A larger optional override if you want a slower, more paper-faithful training run.
- `data/benchmarks/*.jsonl`
  Local normalized evaluation sets for:
  `boolq`, `piqa`, `social_i_qa`, `hellaswag`, `winogrande`,
  `ARC-Easy`, `ARC-Challenge`, and `openbookqa`.

The code now prefers these local files over network downloads during evaluation.

## Optional exploratory datasets

The following files are not used by the default reproduction pipeline:

- `alpaca_data*.json`
- `math*.json`

They can stay in the repo for experiments, but they are not part of the current paper-faithful training path.

## Preparing assets locally

To materialize the evaluation benchmarks under `data/benchmarks/` and optionally prefetch model weights into the active Hugging Face cache, run:

```bash
uv run python -m dora_repro.cli prepare-assets --models tiny_debug
```

To fetch every configured model preset, use:

```bash
uv run python -m dora_repro.cli prepare-assets --models all
```

For gated Llama models, make sure `HF_TOKEN` or `HF_TOKEN_PATH` is set first. In Colab, set `HF_HOME` / `HF_HUB_CACHE` to a Google Drive location before running the command so the standard Hugging Face cache lives on Drive.

For the full 7B and 8B runs in this repo, prefer the `paper_colab` experiment with one of the LLaMA-focused runtime presets:

- `colab_l4_llama` for the best cost/performance balance
- `colab_a100_40gb_llama` for faster turnaround
- `colab_t4_llama` only for budget-constrained or overnight runs

The same `LoRA` / `DoRA` and benchmark-selection env vars used by the CLI are also supported by the Colab notebook:

- `DORA_REPRO_MODEL`
- `DORA_REPRO_METHOD`
- `DORA_REPRO_SCOPE`
- `DORA_REPRO_RUNTIME`
- `DORA_REPRO_EXPERIMENT`
- `DORA_REPRO_TRAIN_DATA_PATH`
- `DORA_REPRO_RUN_NAME`
- `DORA_REPRO_EVAL_TASKS`

## Cache

Temporary dataset caches and logs are stored under `data/cache/`.
