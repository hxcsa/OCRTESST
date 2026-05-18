from __future__ import annotations

from PIL import Image

from .base import BaseOCREngine, OCRResult


class TrOCREngine(BaseOCREngine):
    name = "trocr"

    def __init__(self, model_name: str, device: str = "cpu", logger=None):
        super().__init__(device, logger)
        self.model_name = model_name
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            self.torch = torch
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
            self._error = None
        except Exception as exc:
            self._error = str(exc)

    def is_available(self) -> bool:
        return not getattr(self, "_error", None)

    def run(self, image_path: str) -> OCRResult:
        if not self.is_available():
            return OCRResult(engine=self.name, error=f"TrOCR unavailable: {self._error}")
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return OCRResult(engine=self.name, text=text)
        except Exception as exc:
            return OCRResult(engine=self.name, error=str(exc))
