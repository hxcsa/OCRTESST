from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import load_config
from src.io_utils import safe_write_json


DOC_LEVEL_KEYS = [
    "recipient",
    "subject",
    "document_number",
    "document_date",
    "project_name",
    "location",
    "area_m2",
    "referenced_decision_number",
    "sender_or_department",
    "full_name",
    "national_id",
    "phone_number",
    "address",
    "decision_or_status",
]


def _latest_run_dir(root: Path) -> Path:
    runs = root / "runs"
    if runs.exists():
        run_dirs = sorted([p for p in runs.iterdir() if p.is_dir()])
        if run_dirs:
            return run_dirs[-1]
    return root


def _page_sort_key(item: dict[str, Any]) -> tuple[str, int]:
    return str(item.get("document_id") or ""), int(item.get("page") or 0)


def _best_value(pages: list[dict[str, Any]], key: str) -> Any:
    for page in pages:
        validation = page.get("field_validation", {}).get(key, {})
        value = page.get(key)
        if value not in (None, "", []) and validation.get("status") == "green":
            return value
    for page in pages:
        value = page.get(key)
        if value not in (None, "", []):
            return value
    return None


def aggregate_qwen_outputs(cfg: dict[str, Any], run_dir: str | Path | None = None, engine: str = "vlm_qwen_gguf") -> Path:
    root = Path(cfg["output_dir"]) / "structured_json" / engine
    source_dir = Path(run_dir) if run_dir else _latest_run_dir(root)
    pages = []
    for path in sorted(source_dir.glob("*.json")):
        try:
            pages.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    by_source: dict[str, list[dict[str, Any]]] = {}
    for page in pages:
        source = page.get("source") or page.get("document_id") or "unknown"
        by_source.setdefault(source, []).append(page)

    packets = []
    for source, source_pages in sorted(by_source.items()):
        source_pages = sorted(source_pages, key=_page_sort_key)
        packets.append(
            {
                "source": source,
                "document_id": source_pages[0].get("document_id"),
                "page_count": len(source_pages),
                "document_fields": {key: _best_value(source_pages, key) for key in DOC_LEVEL_KEYS},
                "pages": source_pages,
            }
        )

    run_id = pages[0].get("run_id") if pages else source_dir.name
    payload = {"run_id": run_id, "source_dir": str(source_dir), "packet_count": len(packets), "documents": packets}
    out = Path(cfg["output_dir"]) / "aggregated" / engine / f"{run_id}_packet.json"
    safe_write_json(out, payload, overwrite=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    print(aggregate_qwen_outputs(load_config(), args.run_dir))


if __name__ == "__main__":
    main()
