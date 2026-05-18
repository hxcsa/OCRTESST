from __future__ import annotations

import argparse

from src.config import ensure_project_dirs, load_config
from src.logging_utils import setup_logging
from src.runtime import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Local OCR/document-understanding benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preprocess")
    ocr = sub.add_parser("run-ocr")
    ocr.add_argument("--engine", required=True, choices=["paddle", "surya", "docling", "easyocr", "trocr", "tesseract"])
    vlm = sub.add_parser("run-vlm")
    vlm.add_argument("--model", required=True, choices=["qwen"])
    sub.add_parser("evaluate")
    clean = sub.add_parser("run-llm-clean")
    clean.add_argument("--engine", required=True, choices=["paddle", "surya", "docling", "easyocr", "trocr", "tesseract"])
    clean.add_argument("--mode", default="all", choices=["raw", "clean", "binarized", "all"])
    pipe = sub.add_parser("pipeline")
    pipe.add_argument("--input", required=True, help="Path to image or PDF file")
    pipe.add_argument("--engine", default="paddle", choices=["paddle", "surya", "docling", "easyocr", "trocr", "tesseract"])
    sub.add_parser("full-benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    ensure_project_dirs(cfg)
    logger = setup_logging(cfg["report_dir"])
    logger.info("Device resolved to %s", resolve_device(cfg.get("device", "auto")))

    if args.command == "preprocess":
        from src.benchmark import preprocess

        preprocess(cfg, logger)
    elif args.command == "run-ocr":
        from src.benchmark import run_ocr

        run_ocr(cfg, args.engine, logger)
    elif args.command == "run-vlm":
        from src.vlm.runner import run_vlm

        run_vlm(cfg, args.model, logger)
    elif args.command == "evaluate":
        from src.evaluation.evaluate import evaluate

        evaluate(cfg, logger=logger)
    elif args.command == "run-llm-clean":
        from src.llm_clean import run_llm_clean

        run_llm_clean(cfg, args.engine, args.mode, logger)
    elif args.command == "pipeline":
        from src.pipeline import run_pipeline

        result = run_pipeline(cfg, args.input, args.engine, logger)
        logger.info("Pipeline complete. Result saved to outputs/pipeline/")
        # Pretty-print key fields to console
        print("\n=== PIPELINE RESULT ===")
        print(f"OCR engine: {result.get('ocr_engine', 'unknown')}")
        print(f"OCR text length: {len(result.get('ocr_text', ''))} chars")
        print(f"OCR confidence: {result.get('ocr_avg_confidence')}")
        print(f"Extraction mode: {result.get('extraction_mode', 'unknown')}")
        print(f"VLM model: {cfg.get('model_names', {}).get('vlm', 'N/A')}")
        print(f"Correction model: {cfg.get('model_names', {}).get('text', 'N/A')}")
        extracted = result.get('extracted_fields', {})
        corrected = result.get('corrected_fields', {})
        non_null_ext = {k: v for k, v in extracted.items() if v is not None and v != []}
        non_null_cor = {k: v for k, v in corrected.items() if v is not None and v != []}
        print(f"Extracted fields (non-null): {non_null_ext}")
        print(f"Corrected fields (non-null): {non_null_cor}")
        print(f"Total runtime: {result.get('total_runtime_seconds', 0):.1f}s")
        print("=======================\n")
    elif args.command == "full-benchmark":
        from src.benchmark import full_benchmark
        from src.evaluation.evaluate import evaluate

        full_benchmark(cfg, logger)
        evaluate(cfg, logger=logger)


if __name__ == "__main__":
    main()
