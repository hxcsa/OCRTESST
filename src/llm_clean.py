from __future__ import annotations

from pathlib import Path

from src.engines.llm_cleanup import LLMCleanupEngine


def run_llm_clean(cfg: dict, engine: str, mode: str, logger=None) -> None:
    device = cfg.get("device", "auto")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_ocr_dir = Path(cfg["output_dir"]) / "raw_ocr" / engine
    cleaned_dir = Path(cfg["output_dir"]) / "llm_cleaned" / engine

    modes = [mode] if mode != "all" else ["raw", "clean", "binarized"]

    cleaner = LLMCleanupEngine(device=device, logger=logger)
    cleaner.load()

    for m in modes:
        src_dir = raw_ocr_dir / m
        if not src_dir.exists():
            if logger:
                logger.warning("No OCR results for %s/%s — skipping", engine, m)
            continue

        txt_files = sorted(src_dir.glob("*.txt"))
        if logger:
            logger.info("Cleaning %d files for %s/%s", len(txt_files), engine, m)

        for txt_file in txt_files:
            out_path = cleaned_dir / m / f"{txt_file.stem}_cleaned.txt"
            if out_path.exists():
                if logger:
                    logger.debug("Skipping %s — already cleaned", txt_file.name)
                continue
            cleaner.clean_file(txt_file, out_path)

    cleaner.unload()
    if logger:
        logger.info("LLM cleanup complete for engine=%s", engine)
