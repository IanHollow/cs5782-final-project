"""Shared logging helpers for the DoRA reproduction CLI."""

from __future__ import annotations

import logging
import logging.config
import os
from importlib import import_module
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from pathlib import Path


class ContextAdapter(logging.LoggerAdapter[logging.Logger]):
    """Attach stable key-value context to log messages."""

    def process(
        self, msg: object, kwargs: MutableMapping[str, object]
    ) -> tuple[object, MutableMapping[str, object]]:
        payload = dict(kwargs)
        adapter_extra = dict(self.extra) if self.extra is not None else {}
        extra = cast("dict[str, object]", payload.pop("extra", {}))
        merged = {**adapter_extra, **extra}
        if not merged:
            return msg, payload
        context = " ".join(
            f"{key}={value}" for key, value in sorted(merged.items()) if value is not None
        )
        return f"{msg} | {context}", payload


def _coerce_level(level_name: str) -> int:
    mapping = logging.getLevelNamesMapping()
    normalized = level_name.upper()
    if normalized not in mapping:
        msg = f"Unsupported log level: {level_name}"
        raise ValueError(msg)
    return mapping[normalized]


def _build_config(level: int, log_path: Path | None) -> dict[str, Any]:
    handlers: dict[str, dict[str, Any]] = {
        "console": {
            "class": "rich.logging.RichHandler",
            "level": level,
            "formatter": "console",
            "markup": False,
            "rich_tracebacks": True,
            "show_path": False,
        }
    }
    root_handlers = ["console"]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": f"{RotatingFileHandler.__module__}.{RotatingFileHandler.__qualname__}",
            "level": level,
            "formatter": "file",
            "filename": str(log_path),
            "maxBytes": 2_000_000,
            "backupCount": 3,
            "encoding": "utf-8",
        }
        root_handlers.append("file")
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "format": "%(message)s",
                "datefmt": "[%X]",
            },
            "file": {
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": handlers,
        "root": {
            "level": level,
            "handlers": root_handlers,
        },
    }


def _configure_external_loggers(level: int) -> None:
    external_level = logging.INFO if level <= logging.DEBUG else logging.WARNING
    for logger_name in ("accelerate", "datasets", "httpx", "matplotlib", "transformers"):
        logging.getLogger(logger_name).setLevel(external_level)

    try:
        datasets_logging = import_module("datasets.utils.logging")
        datasets_logging.set_verbosity(external_level)
    except ImportError:
        logging.getLogger(__name__).debug("datasets logging helpers are unavailable")

    try:
        transformers_logging = import_module("transformers.utils.logging")
        if external_level <= logging.INFO:
            transformers_logging.set_verbosity_info()
        else:
            transformers_logging.set_verbosity_warning()
    except ImportError:
        logging.getLogger(__name__).debug("transformers logging helpers are unavailable")


def configure_logging(level_name: str = "INFO", *, log_path: Path | None = None) -> None:
    """Configure console and optional file logging for a CLI command."""
    level = _coerce_level(level_name)
    logging.config.dictConfig(_build_config(level, log_path))
    _configure_external_loggers(level)


def bind_logger(logger: logging.Logger, /, **context: object) -> ContextAdapter:
    """Return a logger adapter with stable contextual fields."""
    normalized = {key: str(value) for key, value in context.items() if value is not None}
    return ContextAdapter(logger, normalized)


def get_log_level(default: str = "INFO") -> str:
    """Resolve the application log level from the environment."""
    return os.environ.get("DORA_LOG_LEVEL", default).upper()
