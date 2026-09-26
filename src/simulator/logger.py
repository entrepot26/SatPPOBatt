from __future__ import annotations

import logging
from pathlib import Path

from src.utils.io import ensure_dir


def setup_logger(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("leo_fl")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file is not None:
        p = Path(log_file)
        ensure_dir(p.parent)
        file_handler = logging.FileHandler(p, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger

