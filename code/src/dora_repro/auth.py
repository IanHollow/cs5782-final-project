"""Authentication helpers for Hugging Face access."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import get_token


def resolve_hf_token(environment: dict[str, str] | None = None) -> str | None:
    """Resolve a Hugging Face token from env vars or an existing login."""
    env = os.environ if environment is None else environment
    direct_token = env.get("HF_TOKEN", "").strip()
    if direct_token:
        return direct_token

    token_path = env.get("HF_TOKEN_PATH", "").strip()
    if token_path:
        resolved = Path(token_path).expanduser()
        if resolved.is_file():
            contents = resolved.read_text(encoding="utf-8").strip()
            if contents:
                return contents

    existing = get_token()
    if existing:
        return existing.strip() or None
    return None


def require_hf_token(environment: dict[str, str] | None = None) -> str:
    """Return a usable token or raise a clear error message."""
    token = resolve_hf_token(environment)
    if token:
        return token
    msg = (
        "No Hugging Face token found. Set HF_TOKEN, point HF_TOKEN_PATH at a token file, "
        "or run `huggingface-cli login` before downloading gated models."
    )
    raise RuntimeError(msg)
