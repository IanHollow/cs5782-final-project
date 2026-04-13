# DoRA Reimplementation for Commonsense Reasoning

This repository is a course-quality reimplementation of DoRA, the weight-decomposed low-rank adaptation method from the ICML 2024 oral paper *DoRA: Weight-Decomposed Low-Rank Adaptation*. The project focuses on reproducing the commonsense reasoning setup and comparing LoRA vs DoRA across full, attention-only, and MLP-only adaptation scopes.

## Chosen Result

The target reproduction is the paper's commonsense reasoning result: DoRA should outperform matched LoRA baselines on average across BoolQ, PIQA, Social IQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, and OpenBookQA. The repo keeps that benchmark suite intact and adds clean support for the project's attention-only and MLP-only ablations.

## GitHub Contents

- `code/src/dora_repro/`: typed Python package for auth, configs, data prep, adapters, training, evaluation, and summarization
- `code/tests/`: unit and smoke tests
- `code/configs/`: TOML presets for models, runtimes, and default experiments
- `data/`: local training data, benchmark JSONL files, model snapshots, and a cache area
- `results/`, `poster/`, `report/`: course deliverables and experiment outputs
- `notebooks/`: Colab-friendly quickstart assets

## Reimplementation Details

The implementation uses modern Hugging Face libraries instead of vendoring NVlabs’ old PEFT fork:

- `transformers` + `peft` for LoRA/DoRA adapters
- `datasets` for the 8 evaluation benchmarks
- `uv` for environment and dependency management
- `ruff`, `ty`, and `pytest` for repository quality gates

Default experiment preset:

- Base model: `meta-llama/Llama-2-7b-hf`
- Methods: `lora`, `dora`
- Scopes: `full`, `attention_only`, `mlp_only`
- Hyperparameters: 3 epochs, cutoff 256, effective batch size 16, checkpointing every 80 steps

Runtime presets are separated from method presets so the same experiment can run in a paper-faithful mode or a lower-memory Colab mode.

## Reproduction Steps

Local setup:

```bash
uv sync --all-groups
uvx prek validate-config prek.toml
uvx prek install --overwrite
uv run python -m dora_repro.cli prepare-assets --models tiny_debug --cache-dir data/cache
```

Git hooks are managed with `prek` (`prek.toml`). The setup above installs both `pre-commit`
and `pre-push` hooks:

- `pre-commit`: whitespace/newline/merge/conflict/secret checks + Ruff format/lint on staged Python files
- `pre-push`: full `ty check code/src` and `pytest`

Run hooks manually at any time:

```bash
uvx prek run --all-files
uvx prek run --all-files --hook-stage pre-push
```

Train one run:

```bash
uv run python -m dora_repro.cli train \
  --model llama2_7b \
  --method dora \
  --scope full \
  --runtime colab_t4 \
  --experiment paper_llama2_7b
```

Evaluate and summarize:

```bash
uv run python -m dora_repro.cli evaluate --run-dir results/runs/<run_name>
uv run python -m dora_repro.cli summarize --results-dir results/runs --output-dir results/summary
```

Quality checks:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check code/src
uv run pytest
```

Google Colab:

- Run `notebooks/bootstrap_colab.sh` after cloning the repo.
- Set `HF_TOKEN` or `HF_TOKEN_PATH` for gated Llama access.
- Use the `colab_t4` or `colab_l4_a100` runtime preset depending on the GPU.
- To avoid spending GPU time on downloads, first run `prepare-assets` on a CPU runtime while the repo is mounted on Google Drive. Benchmarks are stored under `data/benchmarks/`, and model weights are prefetched into the standard Hugging Face cache controlled by `HF_HOME` / `HF_HUB_CACHE`.

CPU-first asset preparation example:

```bash
uv run python -m dora_repro.cli prepare-assets \
  --models llama2_7b llama3_8b tiny_debug \
  --cache-dir data/cache
```

After that, reconnect to a GPU runtime and train normally. Training and evaluation will use the prefetched Hugging Face cache and the local benchmark files in `data/benchmarks/`.

## Results / Insights

Each experiment run writes:

- `results/runs/<run_name>/config.snapshot.toml`
- `results/runs/<run_name>/checkpoints/`
- `results/runs/<run_name>/predictions/<task>.jsonl`
- `results/runs/<run_name>/metrics.json`

Aggregate summaries are written to `results/summary/summary.csv` and `results/summary/macro_average.png`.

## Conclusion

The repo is organized around a faithful but maintainable reproduction path: modern PEFT DoRA adapters, config-driven experiment presets, local-first datasets and model snapshots, Colab-aware runtimes, and strict repository checks. That makes it usable both as a class deliverable and as a clean baseline for follow-up ablations.

## References

1. Liu et al. *DoRA: Weight-Decomposed Low-Rank Adaptation*. ICML 2024 Oral.
2. [NVlabs/DoRA](https://github.com/NVlabs/DoRA)
3. [Hugging Face PEFT documentation](https://huggingface.co/docs/peft)
4. [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers)

## Acknowledgements

This repository was prepared for the CS 5782 / CS 4782 Intro to Deep Learning final project, Spring 2026.
