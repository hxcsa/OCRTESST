from __future__ import annotations

import cv2

from .base import BaseOCREngine, OCRResult, confidence_mean


class PaddleEngine(BaseOCREngine):
    name = "paddle"

    def __init__(self, device: str = "cpu", logger=None):
        super().__init__(device, logger)
        self._ocr = None
        self._api = "legacy"
        try:
            from paddleocr import PaddleOCR

            # PaddleOCR >= 3.x uses a simplified constructor and predict() method.
            # The old parameters (use_angle_cls, use_gpu, show_log) are removed.
            self._ocr = PaddleOCR(lang="ar")
            self._api = "v3"
        except Exception as exc_v3:
            try:
                self._ocr = PaddleOCR(
                    use_angle_cls=True, lang="ar", use_gpu=device == "cuda", show_log=False
                )
                self._api = "legacy"
            except Exception as exc_legacy:
                self._error = f"v3: {exc_v3}; legacy: {exc_legacy}"
            else:
                self._error = None
        else:
            self._error = None

    def is_available(self) -> bool:
        return self._ocr is not None

    def run(self, image_path: str) -> OCRResult:
        if not self.is_available():
            return OCRResult(engine=self.name, error=f"PaddleOCR unavailable: {self._error}")
        try:
            if self._api == "v3":
                # PaddleOCR v3 predict() expects a numpy ndarray, not a path string.
                img = cv2.imread(str(image_path))
                if img is None:
                    return OCRResult(engine=self.name, error=f"Could not load image: {image_path}")
                predictions = self._ocr.predict(img)
                lines = []
                boxes = []
                confs = []
                for pred in predictions:
                    rec_texts = pred.get("rec_texts", [])
                    rec_scores = pred.get("rec_scores", [])
                    rec_boxes = pred.get("rec_boxes", [])
                    for text, conf, box in zip(rec_texts, rec_scores, rec_boxes):
                        lines.append(text)
                        confs.append(float(conf))
                        # box is [x1, y1, x2, y2] in v3
                        x1, y1, x2, y2 = map(int, box)
                        boxes.append(
                            {
                                "box": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                "text": text,
                                "confidence": float(conf),
                            }
                        )
                return OCRResult(self.name, "\n".join(lines), boxes, confidence_mean(confs))
            else:
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
