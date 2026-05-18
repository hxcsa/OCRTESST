from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OCRResult:
    engine: str
    text: str = ""
    boxes: list[dict[str, Any]] = field(default_factory=list)
    avg_confidence: float | None = None
    error: str | None = None


class BaseOCREngine:
    name = "base"

    def __init__(self, device: str = "cpu", logger=None):
        self.device = device
        self.logger = logger

    def is_available(self) -> bool:
        return True

    def run(self, image_path: str) -> OCRResult:
        raise NotImplementedError


def confidence_mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)
