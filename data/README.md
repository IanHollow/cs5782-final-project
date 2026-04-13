# Data

This directory now serves as the canonical local source for training data and benchmark datasets.

## Reproduction datasets

For the DoRA commonsense reproduction, the datasets that matter are:

- `data/commonsense_170k.json`
  The paper-aligned supervised fine-tuning set used by the default experiments.
- `data/commonsense_15k.json`
  A smaller debug subset for fast Colab sanity runs.
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

## Cache

Temporary dataset caches and logs are stored under `data/cache/`.
