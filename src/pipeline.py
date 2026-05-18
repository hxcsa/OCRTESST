from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.engines.registry import build_engine
from src.io_utils import safe_write_json
from src.runtime import clear_gpu_memory, measured_run, resolve_device


class TextQwenExtractor:
    """Text-only extraction using Qwen2.5-VL model (no images)."""

    def __init__(self, model_name: str, device: str, logger=None):
        self.model_name = model_name
        self.device = device
        self.logger = logger
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            self.torch = torch
            dtype = torch.float16 if device == "cuda" else torch.float32
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
            )
            if device != "cuda":
                self.model.to(device)
            self._error = None
        except Exception as exc:
            self._error = str(exc)

    def is_available(self) -> bool:
        return not self._error

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        if not self.is_available():
            raise RuntimeError(f"Model unavailable: {self._error}")
        # Text-only message
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]


def _build_extraction_prompt(ocr_text: str) -> str:
    schema = (
        "document_type, full_name, national_id, reference_number, "
        "ministry_or_department, issue_date, address, subject, decision_or_status, "
        "stamp_present, signature_present, confidence_notes"
    )
    return (
        "You are an expert document analyst specializing in Kurdish/Arabic governmental documents.\n"
        "Analyze the OCR text below and extract structured fields.\n\n"
        "OCR Text:\n"
        "---\n"
        f"{ocr_text}\n"
        "---\n\n"
        "Instructions:\n"
        "- Return ONLY a single valid JSON object.\n"
        "- Use null for missing fields.\n"
        "- Preserve original Kurdish/Arabic spelling.\n"
        "- Do not hallucinate information not present in the text.\n\n"
        f"Extract these fields: {schema}\n\n"
        "Return format:\n"
        '{"document_type": ..., "full_name": ..., "national_id": ..., "reference_number": ..., '
        '"ministry_or_department": ..., "issue_date": ..., "address": ..., "subject": ..., '
        '"decision_or_status": ..., "stamp_present": ..., "signature_present": ..., "confidence_notes": ...}\n'
    )


def _build_correction_prompt(extracted_json: dict) -> str:
    return (
        "You are a Kurdish/Arabic document data quality expert.\n"
        "Review the extracted JSON below and fix obvious OCR/VLM errors.\n\n"
        "Rules:\n"
        "- Fix garbled Arabic/Kurdish characters (e.g., normalize ي/ی, ك/ک).\n"
        "- Fix common OCR substitutions (e.g., '٠' vs '0', '١' vs '1').\n"
        "- Validate and correct dates to standard formats if possible.\n"
        "- Fix ministry/department names using context.\n"
        "- Keep fields as null if truly missing.\n"
        "- Return ONLY the corrected valid JSON object.\n\n"
        "Input JSON:\n"
        f"{json.dumps(extracted_json, ensure_ascii=False, indent=2)}\n\n"
        "Corrected JSON:\n"
    )


def _parse_json_from_text(text: str) -> dict:
    """Extract JSON object from model output text."""
    text = text.strip()
    # Remove markdown fences
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Isolate assistant response if full chat is echoed
    assistant_marker = "\nassistant\n"
    if assistant_marker in text:
        text = text.split(assistant_marker, 1)[-1]
    elif "assistant" in text:
        text = text.rsplit("assistant", 1)[-1]
    text = text.strip()

    # Find outermost JSON object
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {"error": "Could not parse JSON", "raw_output": text[:500]}


def _pdf_to_image(pdf_path: str, dpi: int = 300) -> str:
    """Convert first page of PDF to PNG and return path."""
    from pdf2image import convert_from_path
    pages = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
    out_path = str(Path(pdf_path).with_suffix(".png"))
    pages[0].save(out_path, "PNG")
    return out_path


def run_pipeline(cfg: dict, input_path: str | None, logger=None) -> dict:
    """Run full pipeline: preprocess → Paddle OCR → VLM extraction → LLM correction."""
    device = resolve_device(cfg.get("device", "auto"))
    input_path = input_path or cfg.get("pipeline_input")
    if not input_path:
        raise ValueError("No input file specified. Use --input or set pipeline_input in config.")

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    # --- Step 1: Preprocess ---
    if input_path.suffix.lower() == ".pdf":
        if logger:
            logger.info("Converting PDF first page to image")
        image_path = _pdf_to_image(str(input_path), int(cfg.get("dpi", 300)))
    else:
        image_path = str(input_path)

    # --- Step 2: Paddle OCR ---
    if logger:
        logger.info("Running Paddle OCR")
    paddle = build_engine("paddle", device, cfg, logger)
    if not paddle.is_available():
        raise RuntimeError(f"Paddle OCR not available: {getattr(paddle, '_error', 'unknown')}")

    with measured_run():
        ocr_result = paddle.run(image_path)
    ocr_metrics = measured_run.last
    ocr_text = ocr_result.text or ""

    if logger:
        logger.info("Paddle OCR: %d chars, confidence=%s", len(ocr_text), ocr_result.avg_confidence)

    if not ocr_text.strip():
        raise RuntimeError("Paddle OCR produced no text")

    # --- Step 3: Text-based VLM extraction ---
    model_name = cfg.get("model_names", {}).get("qwen", "Qwen/Qwen2.5-VL-7B-Instruct")
    if logger:
        logger.info("Loading text model %s for extraction", model_name)

    extractor = TextQwenExtractor(model_name, device, logger)
    if not extractor.is_available():
        raise RuntimeError(f"Text model not available: {extractor._error}")

    extraction_prompt = _build_extraction_prompt(ocr_text)
    with measured_run():
        extraction_raw = extractor.generate(extraction_prompt, max_new_tokens=512)
    extraction_metrics = measured_run.last
    extracted_data = _parse_json_from_text(extraction_raw)

    if logger:
        logger.info("Extraction complete, keys: %s", list(extracted_data.keys()))

    # --- Step 4: LLM correction ---
    if logger:
        logger.info("Running LLM correction")
    correction_prompt = _build_correction_prompt(extracted_data)
    with measured_run():
        correction_raw = extractor.generate(correction_prompt, max_new_tokens=512)
    correction_metrics = measured_run.last
    corrected_data = _parse_json_from_text(correction_raw)

    if logger:
        logger.info("Correction complete, keys: %s", list(corrected_data.keys()))

    # --- Assemble final result ---
    final = {
        "input_file": str(input_path),
        "image_file": image_path,
        "ocr_text": ocr_text,
        "ocr_avg_confidence": ocr_result.avg_confidence,
        "ocr_runtime_seconds": ocr_metrics.runtime_seconds,
        "ocr_gpu_memory_mb": ocr_metrics.gpu_memory_mb,
        "extraction_raw": extracted_data,
        "extraction_runtime_seconds": extraction_metrics.runtime_seconds,
        "extraction_gpu_memory_mb": extraction_metrics.gpu_memory_mb,
        "corrected_data": corrected_data,
        "correction_runtime_seconds": correction_metrics.runtime_seconds,
        "correction_gpu_memory_mb": correction_metrics.gpu_memory_mb,
        "total_runtime_seconds": (
            ocr_metrics.runtime_seconds
            + extraction_metrics.runtime_seconds
            + correction_metrics.runtime_seconds
        ),
    }

    out_dir = Path(cfg["output_dir"]) / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    safe_write_json(out_dir / f"{stem}_pipeline.json", final, overwrite=True)
    if ocr_text:
        (out_dir / f"{stem}_ocr.txt").write_text(ocr_text, encoding="utf-8")

    clear_gpu_memory()
    return final
