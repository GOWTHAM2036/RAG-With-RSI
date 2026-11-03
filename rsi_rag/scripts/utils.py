"""
Shared utility functions for RSI-RAG dataset preparation pipeline.
Provides logging setup, configuration loader, network retry handlers, and text normalizers.
"""

import os
import sys
import time
import logging
import unicodedata
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import yaml

def load_config(config_path: str = "config/dataset_config.yaml") -> Dict[str, Any]:
    """Loads dataset and pipeline configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configures a standardized logger outputting to console and optional log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger was already created
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def retry_with_backoff(max_retries: int = 3, initial_delay: float = 2.0, backoff_factor: float = 2.0):
    """Decorator to retry flaky network operations with exponential backoff."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    time.sleep(delay)
                    delay *= backoff_factor
            raise RuntimeError(f"Operation '{func.__name__}' failed after {max_retries} attempts: {last_exception}") from last_exception
        return wrapper
    return decorator

def normalize_text(text: Any) -> str:
    """
    Standardizes text without altering semantic meaning.
    - Normalizes unicode using NFKC form
    - Converts carriage returns to standard newlines
    - Collapses irregular internal whitespace while preserving paragraph breaks
    - Strips leading and trailing whitespaces
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Standardize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace strange whitespace characters
    text = re.sub(r"[\t\u00a0\u200b\u200c\u200d\ufeff]", " ", text)

    # Collapse multiple inline spaces, preserving newlines
    lines = [re.sub(r"[ ]{2,}", " ", line).strip() for line in text.split("\n")]
    # Collapse more than two consecutive newlines
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    return normalized.strip()

def get_directory_size_mb(path: Path) -> float:
    """Calculates disk space used by a directory in megabytes (MB)."""
    if not path.exists():
        return 0.0
    total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total_bytes += os.path.getsize(fp)
    return round(total_bytes / (1024 * 1024), 2)
