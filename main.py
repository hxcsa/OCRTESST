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
    vlm.add_argument("--model", required=True, choices=["qwen", "qwen-gguf", "qwen-full"])
    vlm.add_argument("--limit", type=int, default=None, help="Limit Qwen GGUF pages. Use 0 or --all for every page.")
    vlm.add_argument("--all", action="store_true", help="Run Qwen GGUF on every page in the manifest.")
    vlm.add_argument("--overwrite", action="store_true", help="Overwrite existing Qwen GGUF JSON files.")
    sub.add_parser("download-qwen-gguf")
    transcribe = sub.add_parser("run-transcription")
    transcribe.add_argument("--model", required=True, choices=["qwen-gguf"])
    transcribe.add_argument("--limit", type=int, default=None)
    transcribe.add_argument("--all", action="store_true")
    transcribe.add_argument("--overwrite", action="store_true")
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--engine", default="qwen-gguf", choices=["qwen-gguf", "qwen-full"])
    aggregate.add_argument("--run-dir", default=None)
    build_ft = sub.add_parser("build-finetune-data")
    build_ft.add_argument("--out", default="data/finetune/qwen_gguf")
    build_ft.add_argument("--val-ratio", type=float, default=0.1)
    build_ft.add_argument("--include-content", action="store_true")
    tok = sub.add_parser("check-kurdish-tokenizer")
    tok.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    tok.add_argument("--text", default=None)
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
        if args.model == "qwen-gguf":
            from src.vlm.qwen_gguf import run_qwen_gguf

            limit = 0 if args.all else args.limit
            run_qwen_gguf(cfg, logger, limit=limit, overwrite=args.overwrite)
        elif args.model == "qwen-full":
            from src.vlm.qwen_full import run_qwen_full

            limit = 0 if args.all else args.limit
            run_qwen_full(cfg, logger, limit=limit, overwrite=args.overwrite)
        else:
            from src.vlm.runner import run_vlm

            run_vlm(cfg, args.model, logger)
    elif args.command == "download-qwen-gguf":
        from src.vlm.qwen_gguf import download_qwen_gguf

        model_path, mmproj_path = download_qwen_gguf(cfg, logger)
        print(f"model={model_path}")
        print(f"mmproj={mmproj_path}")
    elif args.command == "run-transcription":
        from src.vlm.qwen_gguf import run_qwen_gguf_transcription

        limit = 0 if args.all else args.limit
        run_qwen_gguf_transcription(cfg, logger, limit=limit, overwrite=args.overwrite)
    elif args.command == "aggregate":
        from src.aggregation import aggregate_qwen_outputs

        engine_name = "vlm_qwen_full" if args.engine == "qwen-full" else "vlm_qwen_gguf"
        print(aggregate_qwen_outputs(cfg, args.run_dir, engine=engine_name))
    elif args.command == "build-finetune-data":
        from src.training.build_finetune_dataset import build_dataset

        summary = build_dataset(cfg, args.out, args.val_ratio, args.include_content)
        print(summary)
    elif args.command == "check-kurdish-tokenizer":
        from src.training.tokenizer_check import KURDISH_PROBE, check_tokenizer

        result = check_tokenizer(args.model, args.text or KURDISH_PROBE)
        print(result)
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
