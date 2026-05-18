from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg"}
SUPPORTED_INPUTS = SUPPORTED_IMAGES | {".pdf"}


def iter_input_files(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_INPUTS)


def document_id_for(path: Path) -> str:
    return path.stem.replace(" ", "_")


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def safe_write_text(path: str | Path, text: str, overwrite: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        path = path.with_name(f"{path.stem}_{timestamp()}{path.suffix}")
    path.write_text(text, encoding="utf-8")
    return path


def safe_write_json(path: str | Path, data: dict[str, Any], overwrite: bool = True) -> Path:
    return safe_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, default=_json_default), overwrite=overwrite)


def _json_default(obj: Any):
    try:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def convert_inputs_to_raw_pages(input_dir: str | Path, processed_dir: str | Path, dpi: int) -> list[dict[str, Any]]:
    processed = Path(processed_dir)
    raw_dir = processed / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []
    # Track how many times each document_id has been seen to avoid filename collisions
    seen_doc_ids: dict[str, int] = {}

    for src in iter_input_files(input_dir):
        doc_id = document_id_for(src)
        if src.suffix.lower() == ".pdf":
            try:
                from pdf2image import convert_from_path
            except ImportError as exc:
                raise RuntimeError(
                    "pdf2image is required for PDF inputs. "
                    "Install with: pip install pdf2image  (also needs system poppler-utils)"
                ) from exc
            images = convert_from_path(str(src), dpi=dpi)
            for i, img in enumerate(images, start=1):
                out = raw_dir / f"{doc_id}_page_{i:03d}.png"
                if not out.exists():
                    img.save(out)
                pages.append({"document_id": doc_id, "page": i, "source": str(src), "raw_image": str(out)})
        else:
            # Disambiguate image inputs that share the same stem (e.g. foo.png and foo.jpg)
            count = seen_doc_ids.get(doc_id, 0) + 1
            seen_doc_ids[doc_id] = count
            unique_id = doc_id if count == 1 else f"{doc_id}_{count}"
            out = raw_dir / f"{unique_id}_page_001.png"
            if not out.exists():
                img = Image.open(src).convert("RGB")
                img.save(out)
            pages.append({"document_id": unique_id, "page": 1, "source": str(src), "raw_image": str(out)})
    return pages



def load_manifest(processed_dir: str | Path) -> list[dict[str, Any]]:
    manifest = Path(processed_dir) / "manifest.json"
    if not manifest.exists():
        return []
    return json.loads(manifest.read_text(encoding="utf-8"))


def save_manifest(processed_dir: str | Path, pages: list[dict[str, Any]]) -> None:
    safe_write_json(Path(processed_dir) / "manifest.json", {"pages": pages})
