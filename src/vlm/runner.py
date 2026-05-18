from __future__ import annotations

import json
from pathlib import Path

from src.config import PROJECT_ROOT
from src.io_utils import load_manifest, safe_write_json
from src.runtime import clear_gpu_memory, measured_run, resolve_device
from src.vlm.qwen_vlm import QwenVLMExtractor


def _find_raw_text(page: dict, output_dir: Path) -> str:
    """Try to find existing OCR text for this page to show alongside VLM results."""
    doc_id = page["document_id"]
    page_num = int(page["page"])
    
    # Check common OCR engines in order of preference
    engines = ["paddle", "surya", "easyocr", "docling", "trocr"]
    for engine in engines:
        raw_dir = output_dir / "raw_ocr" / engine / "raw"
        if not raw_dir.exists():
            continue
        # Look for matching file
        for f in raw_dir.glob(f"{doc_id}_p{page_num:03d}_*.txt"):
            if f.exists():
                text = f.read_text(encoding="utf-8").strip()
                if text:
                    return text
        # Also try without zero-padding
        for f in raw_dir.glob(f"{doc_id}_p{page_num}_*.txt"):
            if f.exists():
                text = f.read_text(encoding="utf-8").strip()
                if text:
                    return text
    
    return "No OCR text available. Run an OCR engine (e.g., paddle, surya) to generate raw text."


def run_vlm(cfg: dict, model: str, logger=None) -> None:
    device = resolve_device(cfg.get("device", "auto"))
    if model.lower() != "qwen":
        raise ValueError(f"Unsupported VLM model: {model}")
    extractor = QwenVLMExtractor(
        cfg.get("model_names", {}).get("qwen", "Qwen/Qwen2.5-VL-7B-Instruct"),
        device,
        PROJECT_ROOT / "prompts" / "vlm_document_prompt.txt",
        logger,
    )
    manifest = load_manifest(cfg["processed_dir"])
    pages = manifest.get("pages", []) if isinstance(manifest, dict) else manifest
    out_dir = Path(cfg["output_dir"]) / "structured_json" / "vlm_qwen"
    output_dir = Path(cfg["output_dir"])
    
    for page in pages:
        image_path = page["images"]["raw"]
        with measured_run():
            data = extractor.extract(image_path)
        
        # Map extraction_confidence to avg_ocr_confidence for UI compatibility
        if "extraction_confidence" in data:
            data["avg_ocr_confidence"] = data.pop("extraction_confidence")
        else:
            # Compute heuristic from field fill rate
            schema_keys = [
                "document_type", "full_name", "national_id", "reference_number",
                "ministry_or_department", "issue_date", "address", "subject", "decision_or_status"
            ]
            filled = sum(1 for k in schema_keys if data.get(k) is not None)
            data["avg_ocr_confidence"] = round(filled / len(schema_keys), 4)
        
        # Try to attach raw OCR text for the UI
        data["raw_text"] = _find_raw_text(page, output_dir)
        
        data.update(
            {
                "document_id": page["document_id"],
                "page": page["page"],
                "engine": "vlm_qwen",
                "preprocessing_mode": "raw",
                "runtime_seconds": measured_run.last.runtime_seconds,
                "gpu_memory_mb": measured_run.last.gpu_memory_mb,
            }
        )
        safe_write_json(out_dir / f"{page['document_id']}_p{int(page['page']):03d}_vlm_qwen.json", data, overwrite=False)
        clear_gpu_memory()
