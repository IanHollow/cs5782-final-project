#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$HOME/.local/bin" sh
  export PATH="$HOME/.local/bin:$PATH"
fi

EXTRA_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  EXTRA_ARGS+=(--extra gpu)
fi

uv sync --frozen --no-default-groups "${EXTRA_ARGS[@]}"
