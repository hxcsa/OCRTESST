from __future__ import annotations

from pathlib import Path

from src.annotate import annotate_boxes
from src.detection.stamp_signature import detect_stamp_signature_regions
from src.engines.registry import build_engine
from src.extraction.rules import extract_structured_fields
from src.io_utils import convert_inputs_to_raw_pages, load_manifest, safe_write_json, safe_write_text, save_manifest
from src.preprocessing import preprocess_manifest_pages
from src.runtime import clear_gpu_memory, measured_run, resolve_device


def preprocess(cfg: dict, logger=None) -> list[dict]:
    pages = convert_inputs_to_raw_pages(cfg["input_dir"], cfg["processed_dir"], int(cfg.get("dpi", 300)))
    pages = preprocess_manifest_pages(pages, cfg["processed_dir"])
    save_manifest(cfg["processed_dir"], pages)
    if logger:
        logger.info("Preprocessed %s page(s)", len(pages))
    return pages


def _manifest_pages(cfg: dict, logger=None) -> list[dict]:
    manifest = load_manifest(cfg["processed_dir"])
    if isinstance(manifest, dict):
        pages = manifest.get("pages", [])
    else:
        pages = manifest
    if not pages:
        pages = preprocess(cfg, logger)
    return pages


def _schema_record(page: dict, engine: str, mode: str, raw_text: str, extracted: dict, metrics, avg_confidence, error=None) -> dict:
    return {
        "document_id": page["document_id"],
        "page": int(page["page"]),
        "engine": engine,
        "preprocessing_mode": mode,
        "document_type": extracted.get("document_type"),
        "full_name": extracted.get("full_name"),
        "national_id": extracted.get("national_id"),
        "reference_number": extracted.get("reference_number"),
        "ministry_or_department": extracted.get("ministry_or_department"),
        "issue_date": extracted.get("issue_date"),
        "address": extracted.get("address"),
        "subject": extracted.get("subject"),
        "decision_or_status": extracted.get("decision_or_status"),
        "stamp_present": extracted.get("stamp_present"),
        "signature_present": extracted.get("signature_present"),
        "raw_text": raw_text,
        "confidence_notes": extracted.get("confidence_notes"),
        "runtime_seconds": metrics.runtime_seconds,
        "gpu_memory_mb": metrics.gpu_memory_mb,
        "avg_ocr_confidence": avg_confidence,
        "error": error,
    }


def run_ocr(cfg: dict, engine_name: str, logger=None) -> None:
    device = resolve_device(cfg.get("device", "auto"))
    pages = _manifest_pages(cfg, logger)
    engine = build_engine(engine_name, device, cfg, logger)
    modes = cfg.get("preprocessing_modes", ["raw", "clean", "binarized"])

    for page in pages:
        detection = {"stamp_present": None, "signature_present": None, "confidence_notes": ""}
        if cfg.get("enable_stamp_signature_detection", True):
            detection = detect_stamp_signature_regions(page["images"]["raw"], cfg["output_dir"], page["document_id"], int(page["page"]))

        for mode in modes:
            image_path = page["images"][mode]
            if logger:
                logger.info("Running %s on %s page %s mode=%s", engine_name, page["document_id"], page["page"], mode)
            with measured_run():
                result = engine.run(image_path)
            metrics = measured_run.last

            raw_dir = Path(cfg["output_dir"]) / "raw_ocr" / engine_name / mode
            structured_dir = Path(cfg["output_dir"]) / "structured_json" / engine_name / mode
            annotated_dir = Path(cfg["output_dir"]) / "annotated" / engine_name / mode
            stem = f"{page['document_id']}_p{int(page['page']):03d}_{engine_name}_{mode}"

            safe_write_text(raw_dir / f"{stem}.txt", result.text or "", overwrite=False)
            raw_payload = {
                "document_id": page["document_id"],
                "page": page["page"],
                "engine": engine_name,
                "preprocessing_mode": mode,
                "text": result.text,
                "boxes": result.boxes,
                "avg_ocr_confidence": result.avg_confidence,
                "runtime_seconds": metrics.runtime_seconds,
                "gpu_memory_mb": metrics.gpu_memory_mb,
                "error": result.error,
                "stamp_signature_detection": detection,
            }
            safe_write_json(raw_dir / f"{stem}.raw.json", raw_payload, overwrite=False)

            if result.boxes:
                annotate_boxes(image_path, result.boxes, annotated_dir / f"{stem}_annotated.png")

            extracted = extract_structured_fields(
                result.text,
                detection.get("stamp_present"),
                detection.get("signature_present"),
                detection.get("confidence_notes", ""),
            )
            record = _schema_record(page, engine_name, mode, result.text, extracted, metrics, result.avg_confidence, result.error)
            safe_write_json(structured_dir / f"{stem}.json", record, overwrite=False)
            clear_gpu_memory()


def full_benchmark(cfg: dict, logger=None) -> None:
    # Only preprocess if the manifest is missing or empty — avoids
    # redundant slow 300 DPI PDF conversion on subsequent runs.
    manifest = load_manifest(cfg["processed_dir"])
    existing_pages = manifest.get("pages", []) if isinstance(manifest, dict) else manifest
    if not existing_pages:
        preprocess(cfg, logger)
    elif logger:
        logger.info("Skipping preprocessing — manifest already has %s page(s)", len(existing_pages))
    for engine in cfg.get("selected_engines", []):
        try:
            run_ocr(cfg, engine, logger)
        except Exception as exc:
            if logger:
                logger.exception("Engine %s failed outside wrapper: %s", engine, exc)
        finally:
            clear_gpu_memory()

