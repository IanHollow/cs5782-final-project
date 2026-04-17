# DoRA Reimplementation for Commonsense Reasoning

This repository is a course-quality reimplementation of DoRA, the weight-decomposed low-rank adaptation method from the ICML 2024 oral paper *DoRA: Weight-Decomposed Low-Rank Adaptation*. The project focuses on reproducing the commonsense reasoning setup and comparing LoRA vs DoRA across full, attention-only, and MLP-only adaptation scopes.

## Chosen Result

The target reproduction is the paper's commonsense reasoning result: DoRA should outperform matched LoRA baselines on average across BoolQ, PIQA, Social IQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, and OpenBookQA. The repo keeps that benchmark suite intact and adds clean support for the project's attention-only and MLP-only ablations.

## GitHub Contents

- `code/src/dora_repro/`: typed Python package for auth, configs, data prep, adapters, training, evaluation, and summarization
- `code/tests/`: unit and smoke tests
- `code/configs/`: TOML presets for models, runtimes, and default experiments
- `data/`: local training data, benchmark JSONL files, and a cache area
- `results/`, `poster/`, `report/`: course deliverables and experiment outputs
- `notebooks/`: Colab-friendly quickstart assets

## Reimplementation Details

The implementation uses modern Hugging Face base-model tooling together with a local clean-room LoRA/DoRA adapter stack:

- `transformers` for model and tokenizer loading
- repo-owned `torch.nn.Module` implementations of LoRA and DoRA for the adapter math, injection, merge logic, and checkpointing
- `datasets` for the 8 evaluation benchmarks
- `uv` for environment and dependency management
- `ruff`, `ty`, and `pytest` for repository quality gates

Default experiment preset:

- Base model: `meta-llama/Llama-2-7b-hf`
- Methods: `lora`, `dora`
- Scopes: `full`, `attention_only`, `mlp_only`
- Training data: `data/commonsense_15k.json`
- Hyperparameters: 3 epochs, cutoff 256, effective batch size 16

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
  --runtime colab_l4_llama \
  --experiment paper_colab
```

The same surface works for all six adapter variants:

- `--method dora --scope full`
- `--method dora --scope attention_only`
- `--method dora --scope mlp_only`
- `--method lora --scope full`
- `--method lora --scope attention_only`
- `--method lora --scope mlp_only`

You can also drive the same choices through environment variables:

```bash
export DORA_REPRO_MODEL=llama2_7b
export DORA_REPRO_METHOD=dora
export DORA_REPRO_SCOPE=attention_only
export DORA_REPRO_RUNTIME=colab_l4_llama
export DORA_REPRO_EXPERIMENT=paper_colab
export DORA_REPRO_RUN_NAME=colab-llama2_7b-dora-attention_only
uv run python -m dora_repro.cli train --output-dir results/runs
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
- Use the LLaMA-focused runtime preset that matches the allocated GPU:
  - `colab_t4_llama`
  - `colab_l4_llama`
  - `colab_a100_40gb_llama`
  - `colab_a100_80gb_llama`
- To avoid spending GPU time on downloads, first run `prepare-assets` on a CPU runtime while the repo is mounted on Google Drive. Benchmarks are stored under `data/benchmarks/`, and model weights are prefetched into the standard Hugging Face cache controlled by `HF_HOME` / `HF_HUB_CACHE`.
- The CLI and notebook both honor the same env vars for run selection:
  - `DORA_REPRO_MODEL`
  - `DORA_REPRO_METHOD`
  - `DORA_REPRO_SCOPE`
  - `DORA_REPRO_RUNTIME`
  - `DORA_REPRO_EXPERIMENT`
  - `DORA_REPRO_TRAIN_DATA_PATH`
  - `DORA_REPRO_RUN_NAME`
  - `DORA_REPRO_EVAL_TASKS`

Recommended Colab choices for this repo:

- `L4 + paper_colab`: best cost/performance tradeoff for the 7B and 8B runs in this repo.
- `A100 40GB + paper_colab`: fastest sensible option if you want same-day turnaround.
- `T4 + debug_quick`: use for smoke tests and tiny-debug validation, not for the full paper-scale runs unless you are willing to wait a long time.

Rough wall-clock estimates for the full `paper_colab` experiment on this repo's 15k dataset, assuming the model and benchmarks were prefetched on CPU first:

- `T4`: roughly 2 to 4 hours for training, plus about 2 to 3 hours for the full 8-task evaluation pass.
- `L4`: roughly 45 to 90 minutes for training, plus about 45 to 75 minutes for evaluation.
- `A100 40GB`: roughly 20 to 40 minutes for training, plus about 20 to 40 minutes for evaluation.
- `A100 80GB`: roughly 15 to 30 minutes for training, plus about 20 to 35 minutes for evaluation.

These are planning estimates, not guarantees. Actual Colab times vary with current allocation, tokenizer length distribution, Drive throughput, and whether your session spends time downloading assets.

CPU-first asset preparation example:

```bash
uv run python -m dora_repro.cli prepare-assets \
  --models llama2_7b llama3_8b tiny_debug \
  --cache-dir data/cache
```

After that, reconnect to a GPU runtime and train normally. Training and evaluation will use the prefetched Hugging Face cache and the local benchmark files in `data/benchmarks/`.

Benchmark selection stays flexible:

- no `--tasks` flag: use the experiment snapshot task list
- `--tasks boolq`: run one benchmark
- `--tasks boolq piqa`: run multiple benchmarks
- `--tasks all` or `DORA_REPRO_EVAL_TASKS=all`: run the full suite

## Results / Insights

Each experiment run writes:

- `results/runs/<run_name>/config.snapshot.toml`
- `results/runs/<run_name>/checkpoints/`
- `results/runs/<run_name>/predictions/<task>.jsonl`
- `results/runs/<run_name>/metrics.json`

Aggregate summaries are written to `results/summary/summary.csv` and `results/summary/macro_average.png`.

## Conclusion

The repo is organized around a faithful but maintainable reproduction path: locally implemented LoRA and DoRA adapters, config-driven experiment presets, local-first datasets, Hugging Face cache-aware model loading, Colab-aware runtimes, and strict repository checks. That makes it usable both as a class deliverable and as a clean baseline for follow-up ablations.

## References

1. Liu et al. *DoRA: Weight-Decomposed Low-Rank Adaptation*. ICML 2024 Oral.
2. [NVlabs/DoRA](https://github.com/NVlabs/DoRA)
3. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
4. [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers)

## Acknowledgements

This repository was prepared for the CS 5782 / CS 4782 Intro to Deep Learning final project, Spring 2026.
