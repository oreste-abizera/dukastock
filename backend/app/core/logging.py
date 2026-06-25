"""
Structured logging configuration using structlog.

Logs are emitted as JSON in production (machine-parseable for Railway.app
log aggregation) and as human-readable console output in development.
"""
import logging
import sys

import structlog

from app.core.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    log_level = logging.DEBUG if settings.debug else logging.INFO

    logging.basicConfig(stream=sys.stdout, level=log_level, format="%(message)s")

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.environment == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    return structlog.get_logger(name)
