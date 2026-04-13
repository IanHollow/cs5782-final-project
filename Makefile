.PHONY: sync format lint test check smoke

sync:
	uv sync --all-groups

format:
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ty check code/src

test:
	uv run pytest

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check code/src
	uv run pytest

smoke:
	uv run python -m dora_repro.cli smoke-test --output-dir results/smoke-test
