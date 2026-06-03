"""Small logging wrapper for data-pipeline scripts."""

import logging
import sys
from pathlib import Path
from typing import Literal

from config import OUTPUT

LogOutput = Literal["file", "stdout", "both"]

DEFAULT_LOG_FILE = OUTPUT / "logs" / "data_pipeline.log"
LOGGER_NAME = "ifls_data_pipeline"

_LOGGER = logging.getLogger(LOGGER_NAME)
_CONFIGURED = False


def _coerce_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    level_name = level.upper()
    level_value = logging.getLevelName(level_name)
    if isinstance(level_value, int):
        return level_value
    raise ValueError(f"Unknown log level: {level}")


def configure_logging(
    *,
    output: LogOutput = "file",
    level: str | int = "INFO",
    log_file: str | Path | None = None,
) -> None:
    """Configure the pipeline logger's level and destination."""
    global _CONFIGURED

    level_value = _coerce_level(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s]: %(message)s (%(funcName)s)",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []
    if output in {"file", "both"}:
        path = Path(log_file) if log_file is not None else DEFAULT_LOG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="a", encoding="utf-8"))
    if output in {"stdout", "both"}:
        handlers.append(logging.StreamHandler(sys.stdout))
    if not handlers:
        raise ValueError(f"Unknown log output: {output}")

    for handler in _LOGGER.handlers:
        handler.close()
    _LOGGER.handlers.clear()
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(level_value)
        _LOGGER.addHandler(handler)

    _LOGGER.setLevel(level_value)
    _LOGGER.propagate = False
    _CONFIGURED = True


def log(message: object, level: str | int = "INFO") -> None:
    """Log a message, lazily configuring file output for standalone scripts."""
    if not _CONFIGURED:
        configure_logging()
    _LOGGER.log(_coerce_level(level), str(message), stacklevel=2)
