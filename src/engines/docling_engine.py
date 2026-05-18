from __future__ import annotations

from .base import BaseOCREngine, OCRResult


class DoclingEngine(BaseOCREngine):
    name = "docling"

    def __init__(self, device: str = "cpu", logger=None):
        super().__init__(device, logger)
        try:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
            self._error = None
        except Exception as exc:
            self._converter = None
            self._error = str(exc)

    def is_available(self) -> bool:
        return self._converter is not None

    def run(self, image_path: str) -> OCRResult:
        if not self.is_available():
            return OCRResult(engine=self.name, error=f"Docling unavailable: {self._error}")
        try:
            result = self._converter.convert(image_path)
            doc = result.document
            text = doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else str(doc)
            return OCRResult(engine=self.name, text=text, boxes=[], avg_confidence=None)
        except Exception as exc:
            return OCRResult(engine=self.name, error=str(exc))
