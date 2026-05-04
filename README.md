# DoRA Reimplementation for Commonsense Reasoning

## Introduction

Full fine-tuning large language models is expensive. LoRA made parameter-efficient fine-tuning practical by training low-rank updates while freezing the base model. DoRA pushes this idea further by decomposing pretrained weights into magnitude and direction, then learning the magnitude separately while adapting the direction with a low-rank update.

This project reimplements *DoRA: Weight-Decomposed Low-Rank Adaptation* for commonsense reasoning and asks three questions:

- Can DoRA match or exceed LoRA accuracy with fewer trainable parameters?
- Does DoRA outperform LoRA at the same rank?
- Does DoRA help most in attention layers, MLP layers, or only when applied broadly across the transformer?

## Chosen Result

We reproduced the paper's commonsense reasoning result from Table 1 / the official NVlabs commonsense table: rank-halved DoRA should improve average accuracy over a same-scope LoRA baseline on BoolQ, PIQA, Social IQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, and OpenBookQA. We also added attention-only and MLP-only ablations to test where the DoRA update helps most.

## GitHub Contents

- `code/`: Python package, CLI, tests, and TOML configs for training/evaluation.
- `data/`: local training data, normalized benchmark files, and dataset notes.
- `results/`: checked-in analysis figures/tables plus ignored fresh run outputs.
- `poster/` and `report/`: placeholders for the final course PDFs.
- `notebooks/`: Colab bootstrap script and notebook runner.

## Re-implementation Details

The implementation uses Hugging Face `transformers` for model/tokenizer loading and repo-owned PyTorch modules for LoRA/DoRA adapter injection, checkpointing, and evaluation. The default practical setup fine-tunes `meta-llama/Llama-2-7b-hf` on `data/commonsense_15k.json`, evaluates the eight commonsense tasks, and compares `lora` vs `dora` across `full`, `attention_only`, and `mlp_only` scopes.

Key modifications from the paper are student-scale compute choices: a 15k training subset by default, Colab-focused quantized runtime presets, and additional scope ablations. Metrics are task accuracy plus macro-average accuracy across the eight benchmarks.

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
  --runtime colab_l4_llama \
  --experiment paper_colab
```

Evaluate and summarize a completed run:

```bash
uv run python -m dora_repro.cli evaluate --run-dir results/runs/<run_name>
uv run python -m dora_repro.cli summarize --results-dir results/runs --output-dir results/summary
```

Run all six main variants by combining `--method {lora,dora}` with `--scope {full,attention_only,mlp_only}`. A Google Colab L4 or A100 GPU is recommended for 7B runs; `tiny_debug` plus `debug_quick` is intended only for CPU/local smoke tests.

Quality checks:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check code/src
uv run pytest
```

## Results/Insights

Our checked-in Llama-2-7B reproduction results are in `results/analysis/summary_table.csv`, with a deeper breakdown in `results/analysis/scope_summary.csv` and `results/analysis/task_summary.csv`. Full-scope, rank-halved DoRA improves over LoRA from `0.7741` to `0.7920` macro accuracy, a gain of about `+1.79` points across six positive task deltas, one tie, and one loss; attention-only DoRA underperforms LoRA by about `-1.50` points; MLP-only DoRA is roughly tied, at about `+0.15` points.

![Macro accuracy by method and scope](results/analysis/fig1_macro_grouped.png)

![DoRA gains by task and scope](results/analysis/fig3_dora_gains.png)

The main paper-level claim partially reproduces in the full-scope setting, while the ablations suggest the benefit is not uniform across isolated layer groups. `results/analysis/fig4_delta_distribution.png` shows that full-scope DoRA is the most consistent condition, and `results/analysis/fig5_official_comparison.png` places the full-scope result beside the paper's LLaMA2-7B LoRA and rank-halved DoRA-dagger reference rows.

## Conclusion

Our results show that DoRA achieves higher accuracy than LoRA in the full-scope setting while using fewer trainable parameters. That supports the paper's main claim that weight decomposition can make fine-tuning cheaper without sacrificing performance. At the same time, the scope ablations show that LoRA is still strong at higher ranks and that DoRA works best when it is applied across the full transformer rather than only a narrow subset of layers.

## Future Work

The most useful next step is to run the same comparisons on the full `commonsense_170k.json` dataset and repeat the experiments across multiple random seeds. That would help verify whether the rank trends hold at larger scale and clarify why attention-only LoRA performed so well compared with the broader full-scope setup.

## References

1. Shih-Yang Liu et al. [*DoRA: Weight-Decomposed Low-Rank Adaptation*](https://arxiv.org/abs/2402.09353). ICML 2024 Oral.
2. [NVlabs/DoRA official implementation](https://github.com/NVlabs/DoRA).
3. Edward Hu et al. [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685).
4. [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers).

## Acknowledgements

This repository was prepared for the CS 4782 / CS 5782 Intro to Deep Learning final project at Cornell University, Spring 2026. We acknowledge the original DoRA authors and NVlabs for releasing the paper, implementation notes, and commonsense reasoning reference results.
