from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from src.benchmark import preprocess
from src.extraction.validators import validate_extraction
from src.io_utils import load_manifest, safe_write_json
from src.runtime import measured_run
from src.vlm.qwen_gguf import PROMPT_PATH, _clean_schema, _load_transcript, _parse_json


class QwenFullExtractor:
    def __init__(self, model_dir: str | Path, cfg: dict[str, Any], logger=None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.logger = logger
        self.model_dir = Path(model_dir)
        self.prompt = PROMPT_PATH.read_text(encoding="utf-8")
        self.processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_dir,
            dtype=torch.float16,
            device_map="auto",
            max_memory={0: "22GiB", "cpu": "450GiB"},
            trust_remote_code=True,
        )
        self.max_new_tokens = int(cfg.get("qwen_full", {}).get("max_new_tokens", 1000))

    def extract(self, image_path: str | Path, transcript: str | None = None) -> dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        prompt = self.prompt
        if transcript:
            prompt = (
                f"{prompt}\n\n"
                "Use this separately generated page transcription as a literal text anchor. "
                "Copy exact field values from the transcription when they match the image. "
                "Return only the JSON object.\n\n"
                f"PAGE_TRANSCRIPTION_MARKDOWN:\n{transcript[:3000]}"
            )
        prompt = (
            "Do not think step by step. Do not explain. Do not analyze. "
            "Your first output character must be { and your last output character must be }. "
            "Return the final JSON object only.\n\n"
            f"{prompt}"
        )
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {key: value.to(self.model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.no_grad():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        generated = ids[0][inputs["input_ids"].shape[-1] :]
        output = self.processor.decode(generated, skip_special_tokens=True)
        return _parse_json(output)


def run_qwen_full(cfg: dict[str, Any], logger=None, limit: int | None = None, overwrite: bool = False) -> None:
    manifest = load_manifest(cfg["processed_dir"])
    pages = manifest.get("pages", []) if isinstance(manifest, dict) else manifest
    if not pages:
        pages = preprocess(cfg, logger)

    qcfg = cfg.get("qwen_full", {})
    model_dir = qcfg.get("local_dir", "models/Qwen3.5-9B")
    image_variant = qcfg.get("image_variant", cfg.get("qwen_gguf", {}).get("image_variant", "raw"))
    normalize_output_text = bool(qcfg.get("normalize_output_text", cfg.get("qwen_gguf", {}).get("normalize_output_text", True)))
    use_transcription_anchor = bool(qcfg.get("use_transcription_anchor", True))
    if limit is None:
        limit = int(qcfg.get("max_images", 5))
    selected = pages if limit <= 0 else pages[:limit]
    run_started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"qwen_full_{run_started_at}"
    extractor = QwenFullExtractor(model_dir, cfg, logger)
    out_dir = Path(cfg["output_dir"]) / "structured_json" / "vlm_qwen_full"
    run_out_dir = out_dir / "runs" / run_id

    for page in selected:
        image_path = page.get("images", {}).get(image_variant) or page["raw_image"]
        if logger:
            logger.info("Qwen full extracting %s page %s image=%s", page["document_id"], page["page"], image_path)
        transcript = _load_transcript(cfg, page) if use_transcription_anchor else None
        with measured_run():
            data = _clean_schema(extractor.extract(image_path, transcript=transcript), normalize_output_text=normalize_output_text)
            if data.get("error") and transcript:
                if logger:
                    logger.warning("Anchored full-model extraction failed for %s page %s; retrying image-only", page["document_id"], page["page"])
                data = _clean_schema(extractor.extract(image_path, transcript=None), normalize_output_text=normalize_output_text)
            data.update(validate_extraction(data))
        data.update(
            {
                "document_id": page["document_id"],
                "page": int(page["page"]),
                "engine": "vlm_qwen_full",
                "preprocessing_mode": image_variant,
                "source": page.get("source"),
                "image_path": image_path,
                "run_id": run_id,
                "extracted_at": run_started_at,
                "runtime_seconds": measured_run.last.runtime_seconds,
                "gpu_memory_mb": measured_run.last.gpu_memory_mb,
                "transcription_used": bool(transcript),
            }
        )
        filename = f"{page['document_id']}_p{int(page['page']):03d}_vlm_qwen_full.json"
        safe_write_json(out_dir / filename, data, overwrite=overwrite)
        safe_write_json(run_out_dir / filename, data, overwrite=True)
