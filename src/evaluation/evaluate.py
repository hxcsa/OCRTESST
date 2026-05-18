from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FIELDS = [
    "document_type",
    "full_name",
    "national_id",
    "reference_number",
    "ministry_or_department",
    "issue_date",
    "address",
    "subject",
    "decision_or_status",
    "stamp_present",
    "signature_present",
]


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_outputs(output_dir: str | Path) -> list[dict[str, Any]]:
    import re
    # Timestamped backup pattern: name ending in _YYYYMMDDTHHMMSSZ before .json
    _ts_re = re.compile(r"_\d{8}T\d{6}Z\.json$")
    rows = []
    for path in Path(output_dir).rglob("*.json"):
        if ".raw" in path.name:
            continue
        # Skip timestamped backup files created by safe_write_json(overwrite=False)
        if _ts_re.search(path.name):
            continue
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def _load_ground_truth(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    gt = {}
    for row in rows:
        if not row.get("document_id") or not row.get("page"):
            continue
        gt[(row["document_id"], int(row["page"]))] = row
    return gt


def evaluate(cfg: dict, ground_truth_path: str | Path = "ground_truth_template.csv", logger=None):
    gt_path = Path(ground_truth_path)
    if not gt_path.is_absolute():
        gt_path = Path(__file__).resolve().parents[2] / gt_path

    outputs = _load_outputs(Path(cfg["output_dir"]) / "structured_json")
    report_dir = Path(cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    if not outputs:
        summary = "# Benchmark Summary\n\nNo structured outputs found yet.\n"
        _write_csv(report_dir / "benchmark_results.csv", [])
        (report_dir / "benchmark_summary.md").write_text(summary, encoding="utf-8")
        return [], summary

    gt = _load_ground_truth(gt_path)
    if not gt:
        for row in outputs:
            row["field_accuracy"] = None
        _write_csv(report_dir / "benchmark_results.csv", outputs)
        summary = _summary_without_gt(outputs)
        (report_dir / "benchmark_summary.md").write_text(summary, encoding="utf-8")
        return outputs, summary

    results = []
    for row in outputs:
        truth = gt.get((row["document_id"], int(row["page"])), {})
        correct = 0
        total = 0
        wrong = []
        missing = []
        for field in FIELDS:
            pred = _norm(row.get(field))
            expected = _norm(truth.get(field))
            if expected == "":
                continue
            total += 1
            if pred == expected:
                correct += 1
            elif pred == "":
                missing.append(field)
            else:
                wrong.append(field)
        results.append(
            {
                "document_id": row.get("document_id"),
                "page": row.get("page"),
                "engine": row.get("engine"),
                "preprocessing_mode": row.get("preprocessing_mode"),
                "field_accuracy": correct / total if total else None,
                "missing_fields": ",".join(missing),
                "wrong_fields": ",".join(wrong),
                "avg_ocr_confidence": row.get("avg_ocr_confidence"),
                "runtime_seconds": row.get("runtime_seconds"),
                "gpu_memory_mb": row.get("gpu_memory_mb"),
                "error": row.get("error"),
            }
        )
    _write_csv(report_dir / "benchmark_results.csv", results)
    summary = _summary_with_gt(results)
    (report_dir / "benchmark_summary.md").write_text(summary, encoding="utf-8")
    return results, summary


def _avg(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if v not in (None, "")]
    return round(sum(nums) / len(nums), 4) if nums else None


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    widths = [len(h) for h in headers]
    rendered = []
    for row in rows:
        vals = ["" if v is None else str(v) for v in row]
        rendered.append(vals)
        widths = [max(w, len(v)) for w, v in zip(widths, vals)]
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |" for row in rendered]
    return "\n".join([line, sep, *body])


def _group(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("engine", "unknown"))].append(row)
    return dict(grouped)


def _summary_without_gt(outputs: list[dict[str, Any]]) -> str:
    table_rows = []
    for engine, rows in sorted(_group(outputs).items()):
        error_pages = sum(1 for r in rows if r.get("error"))
        table_rows.append(
            [
                engine,
                len(rows),
                error_pages,
                round(sum(len(r.get("raw_text") or "") for r in rows) / len(rows), 2),
                _avg([r.get("runtime_seconds") for r in rows]),
                _avg([r.get("gpu_memory_mb") for r in rows]),
                _avg([r.get("avg_ocr_confidence") for r in rows]),
            ]
        )
    return (
        "# Benchmark Summary\n\n"
        "Ground truth is empty, so field accuracy ranking is not available yet.\n\n"
        "## Engine Runtime/Confidence\n\n"
        + _table(
            ["engine", "pages", "error_pages", "avg_raw_text_chars", "avg_runtime_seconds", "avg_gpu_memory_mb", "avg_ocr_confidence"],
            table_rows,
        )
        + "\n\n## Ranking Notes\n\n"
        "- Field extraction accuracy: add ground truth rows to rank.\n"
        "- Kurdish/Arabic OCR quality: use non-empty text volume, confidence, and manual review of raw OCR.\n"
        "- Layout preservation: inspect annotated page images and saved boxes where available.\n"
        "- Stamp/signature usefulness: heuristic crops are placeholders and require manual validation.\n"
        "- Speed and GPU memory usage: ranked by measured runtime and peak allocated VRAM where available.\n"
    )


def _summary_with_gt(results: list[dict[str, Any]]) -> str:
    table_rows = []
    for engine, rows in sorted(_group(results).items()):
        table_rows.append(
            [
                engine,
                _avg([r.get("field_accuracy") for r in rows]),
                sum(1 for r in rows if r.get("error")),
                _avg([r.get("runtime_seconds") for r in rows]),
                _avg([r.get("gpu_memory_mb") for r in rows]),
                _avg([r.get("avg_ocr_confidence") for r in rows]),
                len(rows),
            ]
        )
    table_rows.sort(key=lambda r: (r[1] or -1, -(r[2] or 0)), reverse=True)
    return (
        "# Benchmark Summary\n\n"
        "## Summary Ranking By Engine\n\n"
        + _table(
            ["engine", "field_accuracy", "error_pages", "avg_runtime_seconds", "avg_gpu_memory_mb", "avg_ocr_confidence", "pages"],
            table_rows,
        )
        + "\n\n## Ranking Dimensions\n\n"
        "- Field extraction accuracy: exact normalized match against ground truth fields.\n"
        "- Kurdish/Arabic OCR quality: approximated by confidence plus manual review of raw OCR.\n"
        "- Layout preservation: inspect annotated page images and saved boxes.\n"
        "- Stamp/signature usefulness: inspect cropped regions and stamp/signature predictions.\n"
        "- Speed: lower average runtime is better.\n"
        "- GPU memory usage: lower peak VRAM is better when CUDA metrics are available.\n"
    )
