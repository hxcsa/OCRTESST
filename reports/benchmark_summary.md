# Benchmark Summary

Ground truth is empty, so field accuracy ranking is not available yet.

## Engine Runtime/Confidence

| engine  | pages | error_pages | avg_raw_text_chars | avg_runtime_seconds | avg_gpu_memory_mb | avg_ocr_confidence |
| ------- | ----- | ----------- | ------------------ | ------------------- | ----------------- | ------------------ |
| docling | 42    | 42          | 0.0                | 1.8098              | 468.6212          |                    |
| easyocr | 42    | 0           | 1628.69            | 3.985               | 3754.7912         | 0.3852             |
| paddle  | 42    | 42          | 0.0                | 0.0006              | 0.0               |                    |
| surya   | 42    | 42          | 0.0                | 0.2192              | 1465.9            |                    |

## Ranking Notes

- Field extraction accuracy: add ground truth rows to rank.
- Kurdish/Arabic OCR quality: use non-empty text volume, confidence, and manual review of raw OCR.
- Layout preservation: inspect annotated page images and saved boxes where available.
- Stamp/signature usefulness: heuristic crops are placeholders and require manual validation.
- Speed and GPU memory usage: ranked by measured runtime and peak allocated VRAM where available.
