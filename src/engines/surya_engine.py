from __future__ import annotations

from PIL import Image

from .base import BaseOCREngine, OCRResult, confidence_mean


class SuryaEngine(BaseOCREngine):
    name = "surya"

    def __init__(self, device: str = "cpu", logger=None):
        super().__init__(device, logger)
        self._run_fn = None
        self._langs = ["ar"]  # Arabic; Surya also covers Kurdish/Arabic script
        try:
            # Modern surya-ocr >= 0.6 API
            from surya.ocr import run_ocr as _run_ocr
            from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor

            self._det_model = load_det_model()
            self._det_processor = load_det_processor()
            self._rec_model = load_rec_model()
            self._rec_processor = load_rec_processor()
            self._run_fn = _run_ocr
            self._api = "new"
            self._error = None
            if logger:
                logger.info("Surya loaded with modern API (>= 0.6)")
        except ImportError:
            try:
                # Legacy surya-ocr < 0.6 API
                from surya.common.surya.schema import TaskNames
                from surya.detection import DetectionPredictor
                from surya.foundation import FoundationPredictor
                from surya.recognition import RecognitionPredictor

                self._task_name = TaskNames.ocr_with_boxes
                self._foundation = FoundationPredictor()
                self._det_predictor = DetectionPredictor()
                self._rec_predictor = RecognitionPredictor(self._foundation)
                self._api = "legacy"
                self._error = None
                if logger:
                    logger.info("Surya loaded with legacy API (< 0.6)")
            except Exception as exc:
                self._error = (
                    f"Could not load Surya with either API: {exc}. "
                    "Install with: pip install surya-ocr"
                )
                if logger:
                    logger.warning("Surya load failed: %s", self._error)
        except Exception as exc:
            self._error = str(exc)
            if logger:
                logger.warning("Surya load failed: %s", exc)

    def is_available(self) -> bool:
        return self._error is None

    def run(self, image_path: str) -> OCRResult:
        if not self.is_available():
            return OCRResult(engine=self.name, error=f"Surya unavailable: {self._error}")
        try:
            image = Image.open(image_path).convert("RGB")
            if self._api == "new":
                predictions = self._run_fn(
                    [image],
                    [self._langs],
                    self._det_model,
                    self._det_processor,
                    self._rec_model,
                    self._rec_processor,
                )
            else:
                predictions = self._rec_predictor(
                    [image],
                    task_names=[self._task_name],
                    det_predictor=self._det_predictor,
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

