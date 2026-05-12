"""
src/utils/logger.py
-------------------
Structured logging utility for the NBA Data Pipeline.
Uses structlog to produce consistent, machine-parseable log output.
All pipeline modules import get_logger() from here.
"""

import logging
import structlog
from config.settings import LOG_LEVEL


def configure_logging() -> None:
    """Configure structlog with consistent processors and output format."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),   # Human-readable in dev; swap for JSONRenderer in prod
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Return a structured logger bound to the given module name.

    Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("stage_complete", stage="extract", rows=3247)
    """
    configure_logging()
    return structlog.get_logger(name)
