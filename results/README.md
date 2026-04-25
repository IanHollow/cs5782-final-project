# Results

This directory stores reproducible outputs for the DoRA commonsense reasoning project.

- `analysis/summary_table.csv`: checked-in accuracy table for the six reported LoRA/DoRA conditions.
- `analysis/dora_gains_table.csv`: DoRA minus LoRA deltas by task and adapter scope.
- `analysis/scope_summary.csv`: macro deltas, task win/tie/loss counts, and best/worst task by adapter scope.
- `analysis/task_summary.csv`: per-task best and worst DoRA-minus-LoRA scope.
- `analysis/official_reference_comparison.csv`: full-scope comparison against the official NVlabs LLaMA2-7B commonsense reference.
- `analysis/fig*.png`: Seaborn figures used by the README/report to summarize macro accuracy, per-task accuracy, task-level gains, gain distributions, and official-reference context.
- `runs/<run_name>/`: ignored fresh experiment outputs, including `config.snapshot.toml`, checkpoints, predictions, logs, and `metrics.json`.
- `summary/`: aggregate outputs from `python -m dora_repro.cli summarize`.

Fresh large runs should be written under `results/runs/`; only compact analysis artifacts intended for the final deliverable should be committed.
