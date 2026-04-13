from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dora_repro.logging_utils import bind_logger, configure_logging

if TYPE_CHECKING:
    from pathlib import Path


def test_configure_logging_writes_context_to_file(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    configure_logging("INFO", log_path=log_path)

    logger = bind_logger(logging.getLogger("dora_repro.test"), run_name="demo-run", task="boolq")
    logger.info("hello world", extra={"scope": "attention_only"})
    logging.shutdown()

    contents = log_path.read_text(encoding="utf-8")
    assert "hello world" in contents
    assert "run_name=demo-run" in contents
    assert "task=boolq" in contents
    assert "scope=attention_only" in contents
