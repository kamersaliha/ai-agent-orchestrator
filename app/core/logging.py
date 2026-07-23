"""Centralised logging configuration.

Keeps a single, structured-ish console format. In production you would swap the
formatter for JSON (e.g. ``python-json-logger``); the seam is intentionally here.
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Idempotently configure root logging for the process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Tame noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Module-level logger accessor."""
    return logging.getLogger(name)
