from __future__ import annotations

from .docling_engine import DoclingEngine
from .easyocr_engine import EasyOCREngine
from .paddle_engine import PaddleEngine
from .surya_engine import SuryaEngine
from .tesseract_engine import TesseractEngine
from .trocr_engine import TrOCREngine


def build_engine(name: str, device: str, cfg: dict, logger=None):
    normalized = name.lower()
    if normalized == "paddle":
        return PaddleEngine(device, logger)
    if normalized == "surya":
        return SuryaEngine(device, logger)
    if normalized == "docling":
        return DoclingEngine(device, logger)
    if normalized == "easyocr":
        return EasyOCREngine(device, logger)
    if normalized == "tesseract":
        return TesseractEngine(device, logger)
    if normalized == "trocr":
        return TrOCREngine(cfg.get("model_names", {}).get("trocr", "microsoft/trocr-large-printed"), device, logger)
    raise ValueError(f"Unknown engine: {name}")
