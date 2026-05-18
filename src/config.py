from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    for key in ("input_dir", "processed_dir", "output_dir", "report_dir"):
        cfg[key] = str((PROJECT_ROOT / cfg[key]).resolve())
    return cfg


def ensure_project_dirs(cfg: dict[str, Any]) -> None:
    dirs = [
        cfg["input_dir"],
        cfg["processed_dir"],
        cfg["report_dir"],
        Path(cfg["output_dir"]) / "raw_ocr",
        Path(cfg["output_dir"]) / "structured_json",
        Path(cfg["output_dir"]) / "annotated",
        Path(cfg["output_dir"]) / "crops" / "stamps",
        Path(cfg["output_dir"]) / "crops" / "signatures",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
