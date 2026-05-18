from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.engines.registry import build_engine
from src.extraction.rules import SCHEMA_KEYS, extract_structured_fields
from src.io_utils import safe_write_json
from src.preprocessing import preprocess_image
from src.runtime import clear_gpu_memory, measured_run, resolve_device


VLM_PROMPT = (
    "You are an expert document analyst specializing in Kurdish (Sorani) and Arabic governmental documents.\n"
    "Look at this document image and extract the following fields as JSON.\n"
    "Return ONLY valid JSON. Use null for missing fields. Preserve original Kurdish/Arabic spelling. Do NOT translate.\n\n"
    "Fields:\n"
    "- document_type: type of document (letter, decree, certificate, etc.)\n"
    "- recipient: person/department addressed (after بهريز or اراسته به)\n"
    "- subject: topic of the document (after بابهت or سهبارهت)\n"
    "- document_number: reference number (after ژماره)\n"
    "- document_date: date on the document (after بهروار)\n"
    "- project_name: project name mentioned\n"
    "- location: city/district mentioned (e.g. سليماني)\n"
    "- area_m2: land area in square meters if present\n"
    "- referenced_decision_number: referenced decision/letter number\n"
    "- attachments: list of attached documents, empty [] if none\n"
    "- sender_or_department: sender/ministry/department\n"
    "- full_name: any person's full name\n"
    "- national_id: national ID number if present\n"
    "- phone_number: phone number if present\n"
    "- address: address if present\n"
    "- decision_or_status: decision or status mentioned\n"
    "- stamp_present: true if a stamp/seal is visible, false otherwise\n"
    "- signature_present: true if a signature is visible, false otherwise\n"
    "- uncertain_lines: any text you cannot confidently read (max 10 items)\n\n"
    "JSON:\n"
)


EXTRACTION_FIELDS = "\n".join(
    f'- "{k}": value or null (use [] for attachments and uncertain_lines)' for k in SCHEMA_KEYS
    if k not in ("stamp_present", "signature_present", "confidence_notes")
)


def _build_correction_prompt(ocr_text: str, extracted_json: dict) -> str:
    extracted_str = json.dumps(extracted_json, ensure_ascii=False, indent=2)
    if len(extracted_str) > 2000:
        extracted_str = extracted_str[:2000] + "\n... (truncated)"
    return (
        "Correct this Kurdish/Arabic document extraction against the OCR text.\n"
        "Return ONLY valid JSON with the same field names.\n"
        "For each field provide: value, evidence (exact OCR line), confidence (high/medium/low).\n"
        "Do NOT hallucinate. Do NOT repeat lines in uncertain_lines (max 10).\n"
        "If a field is unclear, set value to null.\n\n"
        f"OCR:\n{ocr_text}\n\n"
        f"First-pass extraction (may have errors):\n{extracted_str}\n\n"
        "Corrected JSON:\n"
    )


def _build_vlm_correction_prompt(ocr_text: str, extracted_json: dict) -> str:
    extracted_str = json.dumps(extracted_json, ensure_ascii=False, indent=2)
    if len(extracted_str) > 3000:
        extracted_str = extracted_str[:3000] + "\n... (truncated)"
    return (
        "You are verifying a document extraction. Look at the document image carefully.\n"
        "Below is a first-pass extraction from this document (may contain errors).\n"
        "Verify each field against what you actually see in the image.\n"
        "Correct any wrong values. Fill in any null/missing fields you can read.\n"
        "Return ONLY valid JSON with the same fields.\n"
        "Do NOT replace correct values with wrong ones from the noisy OCR text.\n\n"
        f"Noisy OCR text (unreliable, for reference only):\n{ocr_text}\n\n"
        f"First-pass extraction (verify and correct):\n{extracted_str}\n\n"
        "Corrected JSON:\n"
    )


def _parse_json_from_text(text: str) -> dict:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    assistant_marker = "\nassistant\n"
    if assistant_marker in text:
        text = text.split(assistant_marker, 1)[-1]
    elif "assistant" in text:
        text = text.rsplit("assistant", 1)[-1]
    text = text.strip()

    # Strip thinking blocks (some models like Gemma may emit these)
    text = re.sub(r"", "", text, flags=re.DOTALL).strip()

    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {"error": "Could not parse JSON", "raw_output": text[:2000]}


def _sanitize_extraction(data: dict) -> dict:
    if not isinstance(data, dict):
        return data
    if "uncertain_lines" in data and isinstance(data["uncertain_lines"], list):
        seen = set()
        deduped = []
        for line in data["uncertain_lines"]:
            if line not in seen:
                seen.add(line)
                deduped.append(line)
            if len(deduped) >= 10:
                break
        data["uncertain_lines"] = deduped
    if "raw_output" in data and isinstance(data.get("raw_output"), str):
        data["raw_output"] = data["raw_output"][:2000]
    return data


def _pdf_to_image(pdf_path: str, dpi: int = 300) -> str:
    from pdf2image import convert_from_path
    pages = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
    out_path = str(Path(pdf_path).with_suffix(".png"))
    pages[0].save(out_path, "PNG")
    return out_path


def run_pipeline(cfg: dict, input_path: str | None, engine_name: str = "paddle", logger=None) -> dict:
    """Run full pipeline: preprocess -> OCR -> VLM extraction -> text correction."""
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

    processed_dir = Path(cfg["processed_dir"]) / "pages"
    processed_dir.mkdir(parents=True, exist_ok=True)
    variants = preprocess_image(image_path, processed_dir, input_path.stem, 1)
    ocr_input_path = variants.get("clean", image_path)
    if logger:
        logger.info("Preprocessing done, using %s for OCR", ocr_input_path)

    # --- Step 2: OCR ---
    if logger:
        logger.info("Running %s OCR", engine_name)
    engine = build_engine(engine_name, device, cfg, logger)
    if not engine.is_available():
        raise RuntimeError(f"{engine_name} OCR not available: {getattr(engine, '_error', 'unknown')}")

    with measured_run():
        ocr_result = engine.run(ocr_input_path)
    ocr_metrics = measured_run.last
    ocr_text = ocr_result.text or ""

    if logger:
        logger.info("%s OCR: %d chars, confidence=%s", engine_name, len(ocr_text), ocr_result.avg_confidence)

    if not ocr_text.strip():
        raise RuntimeError(f"{engine_name} OCR produced no text")

    del engine
    clear_gpu_memory()

    # --- Step 3: Rule-based extraction (always runs as baseline) ---
    rule_data = extract_structured_fields(ocr_text)
    if logger:
        logger.info("Rule-based extraction: %s", {k: v for k, v in rule_data.items() if v and v != []})

    # --- Step 4: VLM extraction (vision model reads the document image directly) ---
    vlm_model = cfg.get("model_names", {}).get("vlm", "Qwen/Qwen2.5-VL-3B-Instruct")
    extracted_data = dict(rule_data)
    vlm_mode = "rules_only"
    extraction_metrics = type("M", (), {"runtime_seconds": 0, "gpu_memory_mb": 0})()

    try:
        from src.vlm.qwen_vlm import QwenVLMExtractor
        vlm = QwenVLMExtractor(vlm_model, device, prompt_text=VLM_PROMPT, logger=logger)
        if vlm.is_available():
            if logger:
                logger.info("Loaded VLM %s for image extraction", vlm_model)
            with measured_run():
                vlm_data = vlm.extract(image_path)
            extraction_metrics = measured_run.last
            vlm_data = _sanitize_extraction(vlm_data)
            if "error" not in vlm_data:
                for key in rule_data:
                    if key in vlm_data and vlm_data[key] is not None and vlm_data[key] != []:
                        extracted_data[key] = vlm_data[key]
                    elif key not in vlm_data or (vlm_data[key] is None or vlm_data[key] == []):
                        if rule_data[key] is not None and rule_data[key] != []:
                            extracted_data[key] = rule_data[key]
                vlm_mode = "vlm_image"
            else:
                if logger:
                    logger.warning("VLM extraction failed (%s), using rule-based", vlm_data.get("error", "unknown"))
                extracted_data = dict(rule_data)
            del vlm
        else:
            if logger:
                logger.warning("VLM unavailable (%s), using rule-based extraction", vlm._error)
    except Exception as exc:
        if logger:
            logger.warning("VLM extraction failed: %s, falling back to rules", exc)

    extracted_data = _sanitize_extraction(extracted_data)
    for key in SCHEMA_KEYS:
        if key not in extracted_data:
            extracted_data[key] = rule_data.get(key)
    if logger:
        non_null = {k: v for k, v in extracted_data.items() if v is not None and v != []}
        logger.info("Extraction (%s) complete, %d non-null fields", vlm_mode, len(non_null))

    clear_gpu_memory()

    # --- Step 5: VLM correction (vision model verifies extraction against image) ---
    vlm_cor_model = cfg.get("model_names", {}).get("text", cfg.get("model_names", {}).get("vlm", "Qwen/Qwen2.5-VL-7B-Instruct"))
    corrected_data = dict(extracted_data)
    correction_metrics = type("M", (), {"runtime_seconds": 0, "gpu_memory_mb": 0})()

    try:
        from src.vlm.qwen_vlm import QwenVLMExtractor
        correction_prompt = _build_vlm_correction_prompt(ocr_text, extracted_data)
        cor_vlm = QwenVLMExtractor(vlm_cor_model, device, prompt_text=correction_prompt, logger=logger)
        if cor_vlm.is_available():
            if logger:
                logger.info("Loaded VLM %s for correction", vlm_cor_model)
            with measured_run():
                cor_data = cor_vlm.extract(image_path)
            correction_metrics = measured_run.last
            cor_data = _sanitize_extraction(cor_data)
            if "error" not in cor_data:
                for key in extracted_data:
                    if key in cor_data and cor_data[key] is not None and cor_data[key] != []:
                        if isinstance(cor_data[key], dict) and "value" in cor_data[key]:
                            corrected_data[key] = cor_data[key]["value"]
                        else:
                            corrected_data[key] = cor_data[key]
            else:
                if logger:
                    logger.warning("VLM correction failed (%s), keeping extraction as-is", cor_data.get("error", "unknown"))
            del cor_vlm
        else:
            if logger:
                logger.warning("Correction VLM unavailable (%s), skipping", cor_vlm._error)
    except Exception as exc:
        if logger:
            logger.warning("VLM correction failed: %s, keeping extraction as-is", exc)

    if logger:
        logger.info("Correction complete, keys: %s", list(corrected_data.keys()))

    # --- Assemble final result ---
    final = {
        "input_file": str(input_path),
        "image_file": image_path,
        "ocr_engine": engine_name,
        "ocr_text": ocr_text,
        "ocr_avg_confidence": ocr_result.avg_confidence,
        "ocr_runtime_seconds": ocr_metrics.runtime_seconds,
        "ocr_gpu_memory_mb": ocr_metrics.gpu_memory_mb,
        "extraction_mode": vlm_mode,
        "extracted_fields": extracted_data,
        "corrected_fields": corrected_data,
        "rule_based_fields": rule_data,
        "extraction_runtime_seconds": getattr(extraction_metrics, "runtime_seconds", 0),
        "extraction_gpu_memory_mb": getattr(extraction_metrics, "gpu_memory_mb", 0),
        "correction_runtime_seconds": getattr(correction_metrics, "runtime_seconds", 0),
        "correction_gpu_memory_mb": getattr(correction_metrics, "gpu_memory_mb", 0),
        "total_runtime_seconds": (
            ocr_metrics.runtime_seconds
            + getattr(extraction_metrics, "runtime_seconds", 0)
            + getattr(correction_metrics, "runtime_seconds", 0)
        ),
    }

    out_dir = Path(cfg["output_dir"]) / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    safe_write_json(out_dir / f"{stem}_{engine_name}_pipeline.json", final, overwrite=True)
    if ocr_text:
        (out_dir / f"{stem}_{engine_name}_ocr.txt").write_text(ocr_text, encoding="utf-8")

    clear_gpu_memory()
    return final