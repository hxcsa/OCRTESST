from __future__ import annotations

from PIL import Image

from .base import BaseOCREngine, OCRResult


class TesseractEngine(BaseOCREngine):
    name = "tesseract"

    def __init__(self, device: str = "cpu", logger=None):
        super().__init__(device, logger)
        self._reader = None
        try:
            import pytesseract

            self._reader = pytesseract
            # Verify tesseract is installed
            self._reader.get_tesseract_version()
            self._error = None
        except Exception as exc:
            self._error = str(exc)

    def is_available(self) -> bool:
        return self._reader is not None

    def run(self, image_path: str) -> OCRResult:
        if not self.is_available():
            return OCRResult(engine=self.name, error=f"Tesseract unavailable: {self._error}")
        try:
            image = Image.open(image_path).convert("RGB")
            text = self._reader.image_to_string(image, lang="ara")
            # Tesseract does not give per-line confidence via pytesseract easily,
            # but we can get boxes with confidences using image_to_data.
            data = self._reader.image_to_data(image, lang="ara", output_type=self._reader.Output.DICT)
            boxes = []
            confs = []
            for i in range(len(data["text"])):
                conf = int(data["conf"][i])
                if conf <= 0:
                    continue
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                boxes.append({"box": box, "text": data["text"][i], "confidence": conf / 100.0})
                confs.append(conf / 100.0)
            avg_conf = sum(confs) / len(confs) if confs else None
            return OCRResult(self.name, text, boxes, avg_conf)
        except Exception as exc:
            return OCRResult(engine=self.name, error=str(exc))
