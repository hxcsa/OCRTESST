from __future__ import annotations

from PIL import Image

from .base import BaseOCREngine, OCRResult, confidence_mean


class SuryaEngine(BaseOCREngine):
    name = "surya"

    def __init__(self, device: str = "cpu", logger=None):
        super().__init__(device, logger)
        self._langs = ["ar"]  # Arabic; Surya also covers Kurdish/Arabic script
        self._error = None
        self._det_predictor = None
        self._foundation = None
        self._rec_predictor = None
        self._task_name = None
        self._device = device
        self._load(device)

    def _load(self, device: str) -> None:
        try:
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
            from surya.foundation import FoundationPredictor
            from surya.common.surya.schema import TaskNames

            self._task_name = TaskNames.ocr_with_boxes
            self._det_predictor = DetectionPredictor(device=device)
            self._foundation = FoundationPredictor(device=device)
            self._rec_predictor = RecognitionPredictor(self._foundation)
            self._error = None
            if self.logger:
                self.logger.info("Surya loaded (device=%s)", device)
        except Exception as exc:
            self._error = f"Could not load Surya: {exc}. Install with: pip install surya-ocr"
            if self.logger:
                self.logger.warning("Surya load failed: %s", self._error)

    def is_available(self) -> bool:
        return self._error is None

    def _run_ocr(self, image_path: str, device: str) -> OCRResult:
        import torch

        # Ensure models are on the right device (reload if device changed)
        if device != self._device:
            self._unload()
            self._device = device
            self._load(device)

        if not self.is_available():
            return OCRResult(engine=self.name, error=f"Surya unavailable: {self._error}")

        image = Image.open(image_path).convert("RGB")
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

    def _unload(self) -> None:
        self._det_predictor = None
        self._foundation = None
        self._rec_predictor = None
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, image_path: str) -> OCRResult:
        if not self.is_available():
            return OCRResult(engine=self.name, error=f"Surya unavailable: {self._error}")
        try:
            return self._run_ocr(image_path, self._device)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self._device == "cuda":
                if self.logger:
                    self.logger.warning("Surya OOM on GPU, retrying on CPU")
                self._unload()
                return self._run_ocr(image_path, "cpu")
            return OCRResult(engine=self.name, error=str(exc))
        except Exception as exc:
            return OCRResult(engine=self.name, error=str(exc))
        finally:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

