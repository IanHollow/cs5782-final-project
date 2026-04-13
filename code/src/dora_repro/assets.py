"""Model prefetch helpers that rely on the standard Hugging Face cache."""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from dora_repro.auth import resolve_hf_token
from dora_repro.config import default_config_dir
from dora_repro.logging_utils import bind_logger

logger = logging.getLogger(__name__)


def available_model_presets(config_dir: Path | None = None) -> tuple[str, ...]:
    """Return the configured model preset names."""
    base_dir = default_config_dir() if config_dir is None else config_dir
    return tuple(sorted(path.stem for path in (base_dir / "models").glob("*.toml")))


def prefetch_model_to_hf_cache(
    *,
    model_name: str,
    model_id: str,
) -> Path:
    """Download a full Hugging Face model snapshot into the active HF cache."""
    token = resolve_hf_token()
    asset_logger = bind_logger(logger, model=model_name, model_id=model_id)
    asset_logger.info("Prefetching model snapshot into Hugging Face cache")
    snapshot_path = snapshot_download(
        repo_id=model_id,
        token=token,
    )
    asset_logger.info("Prefetched model snapshot", extra={"snapshot_path": snapshot_path})
    return Path(snapshot_path)
