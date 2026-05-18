from __future__ import annotations

from .base import BaseOCREngine, OCRResult, confidence_mean


class PaddleEngine(BaseOCREngine):
    name = "paddle"

    def __init__(self, device: str = "cpu", logger=None):
        super().__init__(device, logger)
        self._ocr = None
        try:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(use_angle_cls=True, lang="ar", use_gpu=device == "cuda", show_log=False)
        except Exception as exc:
            self._error = str(exc)
        else:
            self._error = None

    def is_available(self) -> bool:
        return self._ocr is not None

    def run(self, image_path: str) -> OCRResult:
        if not self.is_available():
            return OCRResult(engine=self.name, error=f"PaddleOCR unavailable: {self._error}")
        try:
            result = self._ocr.ocr(image_path, cls=True) or []
            lines = []
            boxes = []
            confs = []
            for page in result:
                for item in page or []:
                    box, rec = item
                    text, conf = rec
                    lines.append(text)
                    confs.append(float(conf))
                    boxes.append({"box": box, "text": text, "confidence": float(conf)})
            return OCRResult(self.name, "\n".join(lines), boxes, confidence_mean(confs))
        except Exception as exc:
            return OCRResult(engine=self.name, error=str(exc))
