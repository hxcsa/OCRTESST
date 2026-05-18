from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


def _normalize_box(box: Any) -> list[tuple[int, int]] | None:
    if box is None:
        return None
    if isinstance(box, dict):
        box = box.get("box") or box.get("bbox")
    if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
        x0, y0, x1, y1 = [int(v) for v in box]
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    try:
        return [(int(p[0]), int(p[1])) for p in box]
    except Exception:
        return None


def annotate_boxes(image_path: str, boxes: list[dict[str, Any]], out_path: str | Path) -> str:
    img = cv2.imread(image_path)
    if img is None:
        return ""
    for item in boxes:
        pts = _normalize_box(item.get("box") or item.get("bbox"))
        if not pts:
            continue
        for i in range(len(pts)):
            cv2.line(img, pts[i], pts[(i + 1) % len(pts)], (0, 180, 255), 2)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return str(out)
