"""Logging configuration for the cyber risk detection system."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.utils.config import get_config


def setup_logging(
    name: str = "cyber_risk",
    level: str | None = None,
    log_file: str | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
) -> logging.Logger:
    """Configure and return a logger with console and rotating file handlers.

    Args:
        name: Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to the log file.
        max_bytes: Maximum size of a log file before rotation.
        backup_count: Number of backup files to keep.

    Returns:
        Configured logger instance.
    """
    config = get_config()

    if level is None:
        level = config.get("logging.level", "INFO")
    if log_file is None:
        log_file = config.get("logging.file", "logs/cyber_risk.log")
    if max_bytes is None:
        max_bytes = config.get("logging.max_bytes", 10485760)
    if backup_count is None:
        backup_count = config.get("logging.backup_count", 5)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "cyber_risk") -> logging.Logger:
    """Get a logger, setting up logging if not already configured."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name)
    return logger