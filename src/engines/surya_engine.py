from __future__ import annotations

from PIL import Image

from .base import BaseOCREngine, OCRResult, confidence_mean


class SuryaEngine(BaseOCREngine):
    name = "surya"

    def __init__(self, device: str = "cpu", logger=None):
        super().__init__(device, logger)
        try:
            from surya.common.surya.schema import TaskNames
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor

            self.task_name = TaskNames.ocr_with_boxes
            self.foundation_predictor = FoundationPredictor()
            self.det_predictor = DetectionPredictor()
            self.rec_predictor = RecognitionPredictor(self.foundation_predictor)
            self._error = None
        except Exception as exc:
            self._error = str(exc)

    def is_available(self) -> bool:
        return not getattr(self, "_error", None)

    def run(self, image_path: str) -> OCRResult:
        if not self.is_available():
            return OCRResult(engine=self.name, error=f"Surya unavailable: {self._error}")
        try:
            image = Image.open(image_path).convert("RGB")
            predictions = self.rec_predictor(
                [image],
                task_names=[self.task_name],
                det_predictor=self.det_predictor,
                highres_images=[image],
                math_mode=False,
            )
            lines = []
            boxes = []
            confs = []
            for pred in predictions:
                for line in getattr(pred, "text_lines", []) or []:
                    text = getattr(line, "text", "")
                    conf = getattr(line, "confidence", None)
                    lines.append(text)
                    if conf is not None:
                        confs.append(float(conf))
                    boxes.append(
                        {
                            "box": getattr(line, "bbox", None),
                            "text": text,
                            "confidence": conf,
                        }
                    )
            return OCRResult(self.name, "\n".join(lines), boxes, confidence_mean(confs))
        except Exception as exc:
            return OCRResult(engine=self.name, error=str(exc))
