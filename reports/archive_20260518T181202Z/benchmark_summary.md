# Benchmark Summary

Ground truth is empty, so accuracy ranking is not available yet.

## Engine Runtime/Confidence

| engine   |   pages |   avg_runtime_seconds |   avg_gpu_memory_mb |   avg_ocr_confidence |
|:---------|--------:|----------------------:|--------------------:|---------------------:|
| easyocr  |      42 |                     0 |                   0 |                  nan |

## Ranking Notes

- Field extraction accuracy: add ground truth rows to rank.
- Kurdish/Arabic OCR quality: review raw OCR and avg confidence manually until ground truth is populated.
- Layout preservation: review annotated outputs and layout boxes where available.
- Stamp/signature usefulness: heuristic crops are placeholders and require manual validation.
- Speed and GPU memory usage: ranked by measured runtime and peak allocated VRAM where available.
