from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(report_dir: str) -> logging.Logger:
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ocr_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(Path(report_dir) / "benchmark.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger
