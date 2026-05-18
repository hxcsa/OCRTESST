from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _read(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def _safe_crop_border(gray: np.ndarray) -> np.ndarray:
    thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return gray
    x, y, w, h = cv2.boundingRect(coords)
    margin = 10
    x0, y0 = max(x - margin, 0), max(y - margin, 0)
    x1, y1 = min(x + w + margin, gray.shape[1]), min(y + h + margin, gray.shape[0])
    if (x1 - x0) < gray.shape[1] * 0.55 or (y1 - y0) < gray.shape[0] * 0.55:
        return gray
    return gray[y0:y1, x0:x1]


def _deskew(gray: np.ndarray) -> np.ndarray:
    inv = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.2 or abs(angle) > 10:
        return gray
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _enhance(gray: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(contrast, -1, kernel)


def preprocess_image(raw_image_path: str | Path, out_dir: str | Path, document_id: str, page: int) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = _read(raw_image_path)

    raw_out = out_dir / f"{document_id}_page_{page:03d}_raw.png"
    if not raw_out.exists():
        cv2.imwrite(str(raw_out), img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = _safe_crop_border(gray)
    gray = _deskew(gray)
    clean = _enhance(gray)
    clean_out = out_dir / f"{document_id}_page_{page:03d}_clean.png"
    cv2.imwrite(str(clean_out), clean)

    binarized = cv2.adaptiveThreshold(clean, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)
    bin_out = out_dir / f"{document_id}_page_{page:03d}_binarized.png"
    cv2.imwrite(str(bin_out), binarized)

    return {"raw": str(raw_out), "clean": str(clean_out), "binarized": str(bin_out)}


def preprocess_manifest_pages(pages: list[dict], processed_dir: str | Path) -> list[dict]:
    out_dir = Path(processed_dir) / "pages"
    updated = []
    for item in pages:
        variants = preprocess_image(item["raw_image"], out_dir, item["document_id"], int(item["page"]))
        updated.append({**item, "images": variants})
    return updated
