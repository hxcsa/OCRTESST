from __future__ import annotations

from .base import BaseOCREngine, OCRResult, confidence_mean


class EasyOCREngine(BaseOCREngine):
    name = "easyocr"

    def __init__(self, device: str = "cpu", logger=None):
        super().__init__(device, logger)
        self._reader = None
        try:
            import easyocr

            self._reader = easyocr.Reader(["ar", "en"], gpu=device == "cuda", verbose=False)
        except Exception as exc:
            self._error = str(exc)
        else:
            self._error = None

    def is_available(self) -> bool:
        return self._reader is not None

    def run(self, image_path: str) -> OCRResult:
        if not self.is_available():
            return OCRResult(engine=self.name, error=f"EasyOCR unavailable: {self._error}")
        try:
            result = self._reader.readtext(image_path, detail=1, paragraph=False)
            lines = []
            boxes = []
            confs = []
            for box, text, conf in result:
                lines.append(text)
                confs.append(float(conf))
                boxes.append({"box": box, "text": text, "confidence": float(conf)})
            return OCRResult(self.name, "\n".join(lines), boxes, confidence_mean(confs))
        except Exception as exc:
            return OCRResult(engine=self.name, error=str(exc))
