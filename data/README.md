# Data

This project keeps large artifacts out of git and materializes them on demand.

- Fine-tuning data: `data/commonsense_170k.json` if already present, otherwise `prepare-data --train-source ...` can download/cache it.
- Evaluation data: fetched from Hugging Face datasets and normalized into `data/cache/normalized/eval/*.jsonl`.
- Generated caches and normalized files live under `data/cache/` and are ignored by git.
