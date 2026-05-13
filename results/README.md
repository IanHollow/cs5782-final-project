# Results

This directory stores reproducible outputs for the DoRA commonsense reasoning project.

- `summary_table.csv`: checked-in accuracy table for the six reported LoRA/DoRA conditions.
- `dora_gains_table.csv`: DoRA minus LoRA deltas by task and adapter scope.
- `scope_summary.csv`: macro deltas, task win/tie/loss counts, and best/worst task by adapter scope.
- `task_summary.csv`: per-task best and worst DoRA-minus-LoRA scope.
- `official_reference_comparison.csv`: full-scope comparison against the DoRA paper Table 1 LLaMA2-7B LoRA and rank-halved DoRA reference rows.
- `fig*.png`: Seaborn figures used by the README/report to summarize macro accuracy, per-task accuracy, task-level gains, gain distributions, and official-reference context.
- `runs/<run_name>/`: ignored fresh experiment outputs, including `config.snapshot.toml`, checkpoints, predictions, logs, and `metrics.json`.
- `summary/`: aggregate outputs from `python -m dora_repro.cli summarize`.

Fresh large runs should be written under `results/runs/`; only compact analysis artifacts intended for the final deliverable should be committed.

To regenerate the checked-in analysis figures from available run metrics or from `results/summary_table.csv`, run:

```bash
uv run python code/scripts/analyze_dora_lora.py --output results
```
