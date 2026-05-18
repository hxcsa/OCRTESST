from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _save_crop(img, rect, out_path: Path) -> str:
    x, y, w, h = rect
    crop = img[y : y + h, x : x + w]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    return str(out_path)


def detect_stamp_signature_regions(image_path: str, output_dir: str | Path, document_id: str, page: int) -> dict:
    """Lightweight placeholder detector.

    It searches for strong blue/red connected components often used in official stamps
    and signature ink. Replace with a trained layout detector when labeled data exists.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"stamp_present": None, "signature_present": None, "regions": [], "confidence_notes": "Image unreadable"}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (90, 40, 40), (140, 255, 255))
    red1 = cv2.inRange(hsv, (0, 40, 40), (12, 255, 255))
    red2 = cv2.inRange(hsv, (165, 40, 40), (180, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(blue | red1 | red2, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    out_base = Path(output_dir) / "crops"
    for idx, c in enumerate(contours[:20], start=1):
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 700 or area > img.shape[0] * img.shape[1] * 0.25:
            continue
        kind = "stamps" if abs(w - h) < max(w, h) * 0.45 else "signatures"
        crop_path = _save_crop(img, (x, y, w, h), out_base / kind / f"{document_id}_p{page:03d}_{kind}_{idx}.png")
        regions.append({"kind": kind[:-1], "box": [x, y, x + w, y + h], "crop_path": crop_path})

    stamp = any(r["kind"] == "stamp" for r in regions)
    signature = any(r["kind"] == "signature" for r in regions)
    return {
        "stamp_present": stamp if regions else None,
        "signature_present": signature if regions else None,
        "regions": regions,
        "confidence_notes": "Heuristic color/shape placeholder; validate manually or replace with trained detector.",
    }
