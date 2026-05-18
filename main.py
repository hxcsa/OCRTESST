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

        result = run_pipeline(cfg, args.input, logger)
        logger.info("Pipeline complete. Result saved to outputs/pipeline/")
        # Pretty-print key fields to console
        print("\n=== PIPELINE RESULT ===")
        print(f"OCR text length: {len(result.get('ocr_text', ''))} chars")
        print(f"OCR confidence: {result.get('ocr_avg_confidence')}")
        print(f"Extracted fields: {list(result.get('extraction_raw', {}).keys())}")
        print(f"Corrected fields: {list(result.get('corrected_data', {}).keys())}")
        print(f"Total runtime: {result.get('total_runtime_seconds', 0):.1f}s")
        print("=======================\n")
    elif args.command == "full-benchmark":
        from src.benchmark import full_benchmark
        from src.evaluation.evaluate import evaluate

        full_benchmark(cfg, logger)
        evaluate(cfg, logger=logger)


if __name__ == "__main__":
    main()
