from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, load_config
from src.io_utils import load_manifest, safe_write_json
from src.training.kurdish_normalization import normalize_json_strings, normalize_kurdish_text


CONTENT_PROMPT = PROJECT_ROOT / "prompts" / "qwen_gguf_content_prompt.txt"
METADATA_PROMPT = PROJECT_ROOT / "prompts" / "qwen_gguf_metadata_prompt.txt"

TARGET_KEYS = [
    "recipient",
    "subject",
    "document_number",
    "document_date",
    "project_name",
    "location",
    "area_m2",
    "referenced_decision_number",
    "attachments",
    "sender_or_department",
    "full_name",
    "national_id",
    "phone_number",
    "address",
    "decision_or_status",
    "stamp_present",
    "signature_present",
    "uncertain_lines",
    "extraction_confidence",
]


def _load_pages(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = load_manifest(cfg["processed_dir"])
    return manifest.get("pages", []) if isinstance(manifest, dict) else manifest


def _source_key(page: dict[str, Any]) -> str:
    return Path(page.get("source", page["document_id"])).name


def _split_sources(pages: list[dict[str, Any]], val_ratio: float) -> tuple[set[str], set[str]]:
    sources = sorted({_source_key(page) for page in pages})
    if len(sources) <= 1:
        return set(sources), set()
    val_count = max(1, round(len(sources) * val_ratio))
    val_sources = set(sources[-val_count:])
    train_sources = set(sources) - val_sources
    return train_sources, val_sources


def _metadata_target(output_dir: Path, page: dict[str, Any]) -> dict[str, Any] | None:
    path = output_dir / "structured_json" / "vlm_qwen_gguf" / f"{page['document_id']}_p{int(page['page']):03d}_vlm_qwen_gguf.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("error"):
        return None
    target = {key: data.get(key) for key in TARGET_KEYS}
    if target.get("attachments") is None:
        target["attachments"] = []
    if target.get("uncertain_lines") is None:
        target["uncertain_lines"] = []
    return normalize_json_strings(target)


def _content_target(page: dict[str, Any]) -> str:
    # Placeholder for teacher OCR/Markdown output. This intentionally starts from
    # metadata fields only when no separate content labels exist yet.
    parts = [
        page.get("document_id", ""),
        f"page {page.get('page')}",
    ]
    return normalize_kurdish_text("\n".join(part for part in parts if part))


def _example(image_path: str, prompt: str, answer: str, task: str, page: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": task,
        "image": image_path,
        "source": page.get("source"),
        "document_id": page["document_id"],
        "page": int(page["page"]),
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": answer},
        ],
    }


def build_dataset(cfg: dict[str, Any], out_dir: str | Path, val_ratio: float = 0.1, include_content: bool = False) -> dict[str, Any]:
    pages = _load_pages(cfg)
    if not pages:
        raise RuntimeError("No processed manifest found. Run: python main.py preprocess")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_sources, val_sources = _split_sources(pages, val_ratio)
    metadata_prompt = METADATA_PROMPT.read_text(encoding="utf-8")
    content_prompt = CONTENT_PROMPT.read_text(encoding="utf-8")
    output_dir = Path(cfg["output_dir"])
    records = {"train": [], "val": []}
    skipped = 0

    for page in pages:
        split = "val" if _source_key(page) in val_sources else "train"
        image_path = page.get("images", {}).get("raw") or page["raw_image"]

        target = _metadata_target(output_dir, page)
        if target is None:
            skipped += 1
        else:
            answer = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
            records[split].append(_example(image_path, metadata_prompt, answer, "metadata", page))

        if include_content:
            records[split].append(_example(image_path, content_prompt, _content_target(page), "content", page))

    for split, items in records.items():
        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "train_examples": len(records["train"]),
        "val_examples": len(records["val"]),
        "skipped_pages_without_clean_metadata": skipped,
        "train_sources": sorted(train_sources),
        "val_sources": sorted(val_sources),
        "normalization": "Kurdish/Sorani keyboard normalization applied to target strings.",
    }
    safe_write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/finetune/qwen_gguf")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--include-content", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    print(json.dumps(build_dataset(cfg, args.out, args.val_ratio, args.include_content), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
