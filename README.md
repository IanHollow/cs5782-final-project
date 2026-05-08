# DoRA Reimplementation for Commonsense Reasoning

## Introduction

This repository contains a CS 4782 / CS 5782 final project reimplementation of *DoRA: Weight-Decomposed Low-Rank Adaptation*. The paper's main contribution is a parameter-efficient fine-tuning method that separates weight magnitude from direction, improving LoRA-style adaptation while training fewer parameters.

## Chosen Result

We target the paper's commonsense reasoning result from Table 1, where rank-halved DoRA improves average accuracy over LoRA on BoolQ, PIQA, Social IQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, and OpenBookQA. We also include attention-only and MLP-only ablations to test where the DoRA update helps most.

## GitHub Contents

- `code/`: Python package, CLI, tests, configs, and analysis script for the reimplementation.
- `data/`: the 15k commonsense training set, normalized benchmark files, and dataset notes.
- `results/`: checked-in figures/tables used by the README, poster, and report.
- `poster/`: final poster PDF plus the LaTeX source used to build it.
- `report/`: placeholder for the final report PDF.
- `code/notebooks/`: optional Colab quickstart material for GPU runs.

## Re-implementation Details

The implementation uses Hugging Face `transformers` for model/tokenizer loading and repo-owned PyTorch modules for LoRA/DoRA adapter injection, checkpointing, and evaluation. The default practical setup fine-tunes `meta-llama/Llama-2-7b-hf` on `data/commonsense_15k.json`, evaluates eight commonsense tasks, and compares `lora` vs `dora` across `full`, `attention_only`, and `mlp_only` adapter scopes.

Compared with the original paper, this repo uses student-scale compute choices: a 15k training subset, Colab-focused 4-bit runtime settings, and additional scope ablations. Metrics are task accuracy and macro-average accuracy across the benchmark suite.

## Reproduction Steps

Install dependencies and prepare local assets:

```bash
uv sync --all-groups
uv run python -m dora_repro.cli prepare-assets --models tiny_debug --cache-dir data/cache
```

For gated Llama runs, set `HF_TOKEN` or `HF_TOKEN_PATH`, then train one adapter:

```bash
uv run python -m dora_repro.cli train \
  --model llama2_7b \
  --method dora \
  --scope full \
  --runtime colab_a100_40gb_llama \
  --experiment paper_colab
```

Evaluate and summarize a completed run:

```bash
uv run python -m dora_repro.cli evaluate --run-dir results/runs/<run_name>
uv run python -m dora_repro.cli summarize --results-dir results/runs --output-dir results/summary
```

Run the six main variants by combining `--method {lora,dora}` with `--scope {full,attention_only,mlp_only}`. A Colab A100-class GPU is recommended for 7B runs; `tiny_debug` is intended only for local smoke tests.

Quality checks:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check code/src
uv run pytest
```

## Results/Insights

The checked-in Llama-2-7B reproduction results are in `results/summary_table.csv`, with additional breakdowns in `results/scope_summary.csv` and `results/task_summary.csv`. Full-scope, rank-halved DoRA improves over LoRA from `0.7741` to `0.7920` macro accuracy, a gain of about `+1.79` points; attention-only DoRA underperforms LoRA by about `-1.50` points; MLP-only DoRA is roughly tied, at about `+0.15` points.

![Macro accuracy by method and scope](results/fig1_macro_grouped.png)

![DoRA gains by task and scope](results/fig3_dora_gains.png)

The main paper-level claim partially reproduces in the full-scope setting, while the ablations suggest the benefit is not uniform across isolated layer groups.

## Conclusion

DoRA achieves higher accuracy than LoRA in the full-scope setting while using fewer trainable parameters, supporting the paper's central claim at student-project scale. The ablations also show that DoRA works best when applied broadly across the transformer instead of only to attention or MLP layers.

## References

1. Shih-Yang Liu et al. [*DoRA: Weight-Decomposed Low-Rank Adaptation*](https://arxiv.org/abs/2402.09353). ICML 2024 Oral.
2. [NVlabs/DoRA official implementation](https://github.com/NVlabs/DoRA).
3. Edward Hu et al. [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685).
4. [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers).

## Acknowledgements

This repository was prepared for the CS 4782 / CS 5782 Intro to Deep Learning final project at Cornell University, Spring 2026. We acknowledge the original DoRA authors and NVlabs for releasing the paper, implementation notes, and commonsense reasoning reference results.
