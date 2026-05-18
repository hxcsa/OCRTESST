from __future__ import annotations

from PIL import Image

from .base import BaseOCREngine, OCRResult

# Known Arabic/Kurdish-capable TrOCR-style checkpoints (community fine-tunes):
#   - "ahmedheakl/trocr-arabic" (Arabic printed)
#   - "Microsoft/trocr-large-handwritten" is still Latin-only
# The default "microsoft/trocr-large-printed" is trained on Latin printed text
# and will produce garbage on Arabic/Kurdish script — override in config.yaml.
_LATIN_ONLY_MODELS = {
    "microsoft/trocr-large-printed",
    "microsoft/trocr-base-printed",
    "microsoft/trocr-large-handwritten",
    "microsoft/trocr-base-handwritten",
}


class TrOCREngine(BaseOCREngine):
    """TrOCR engine wrapper.

    NOTE: The default model (microsoft/trocr-large-printed) is trained on
    Latin-script text only and will produce meaningless output on Arabic or
    Kurdish documents.  Set ``model_names.trocr`` in config.yaml to an
    Arabic-capable checkpoint (e.g. "ahmedheakl/trocr-arabic").
    """

    name = "trocr"

    def __init__(self, model_name: str, device: str = "cpu", logger=None):
        super().__init__(device, logger)
        self.model_name = model_name
        if model_name in _LATIN_ONLY_MODELS:
            msg = (
                f"TrOCR model '{model_name}' is trained on Latin script only and "
                "will produce incorrect results on Arabic/Kurdish documents. "
                "Set model_names.trocr in config.yaml to an Arabic-capable model "
                "(e.g. 'ahmedheakl/trocr-arabic')."
            )
            if logger:
                logger.warning(msg)
            else:
                import warnings
                warnings.warn(msg, UserWarning, stacklevel=2)
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

