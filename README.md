# Kurdish/Arabic Government Document OCR Benchmark

Local benchmark project for scanned Kurdish/Arabic governmental PDFs and images. It compares OCR engines and optional vision-language document understanding pipelines before building a Watsonx/RAG/agent layer.

## What It Does

- Accepts PDF, PNG, JPG, and JPEG inputs from `data/input/`
- Converts PDF pages to 300 DPI images
- Saves page variants: raw, cleaned grayscale, and binarized/high-contrast
- Runs modular OCR engines when installed:
  - PaddleOCR
  - Surya OCR
  - Docling
  - EasyOCR
  - optional TrOCR
- Supports optional Qwen2.5-VL document understanding mode
- Records raw text, boxes/layout where available, confidence, runtime, and CUDA peak memory where available
- Saves annotated OCR box images
- Creates heuristic stamp/signature crop placeholders
- Produces structured JSON and benchmark reports

## Project Layout

```text
data/input/                 # Put scanned PDFs/images here
data/processed/             # Generated page images and manifest
outputs/raw_ocr/            # Raw OCR text, boxes, confidence, runtime
outputs/structured_json/    # Normalized schema outputs
outputs/annotated/          # Page images with OCR/layout boxes
outputs/crops/              # Stamp/signature region crops
reports/                    # CSV/Markdown reports and logs
src/                        # Benchmark source code
prompts/                    # LLM/VLM extraction prompts
```

## Setup On Remote GPU Through VS Code SSH

From the remote SSH terminal:

```bash
cd /root/OCR_Test
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For PDF conversion, install Poppler on the remote machine:

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
```

Some engines are optional and heavy. If one is missing or fails to load, the benchmark writes an error result and continues.

Install optional OCR engines selectively:

```bash
python -m pip install easyocr
python -m pip install paddleocr paddlepaddle-gpu
python -m pip install surya-ocr
python -m pip install docling
```

Or install the optional bundle:

```bash
python -m pip install -r requirements-optional.txt
```

## Usage

Put documents in:

```text
data/input/
```

Run preprocessing:

```bash
python main.py preprocess
```

Run a specific OCR engine:

```bash
python main.py run-ocr --engine paddle
python main.py run-ocr --engine surya
python main.py run-ocr --engine docling
python main.py run-ocr --engine easyocr
```

Run optional VLM extraction:

```bash
python main.py run-vlm --model qwen
```

Evaluate:

```bash
python main.py evaluate
```

Run everything selected in `config.yaml`:

```bash
python main.py full-benchmark
```

## Ground Truth

Fill `ground_truth_template.csv` with one row per document page:

```text
document_id,page,document_type,full_name,national_id,reference_number,ministry_or_department,issue_date,address,subject,decision_or_status,stamp_present,signature_present
```

Evaluation outputs:

- `reports/benchmark_results.csv`
- `reports/benchmark_summary.md`

If ground truth is empty, the report still summarizes runtime, confidence, GPU memory, and page counts.

## Configuration

Edit `config.yaml`:

- `device: auto` selects CUDA when available, otherwise CPU
- `selected_engines` controls `full-benchmark`
- `preprocessing_modes` controls raw/clean/binarized variants
- `model_names.qwen` controls the Qwen VLM model
- `enable_stamp_signature_detection` toggles heuristic crop generation

## Notes On Stamp And Signature Handling

Stamps and signatures are not treated as normal body OCR. The current detector is a placeholder that searches for likely colored stamp/signature regions and saves crops. Replace `src/detection/stamp_signature.py` with a trained layout/stamp/signature detector when you have labeled data.

## How This Connects Later To Watsonx/RAG/Agent

The benchmark chooses the strongest OCR/document parsing pipeline first. That best output becomes structured JSON or database rows. Those records can then be embedded into a vector database and connected to a Watsonx LLM/agent for document search, question answering, summarization, reporting, and workflow automation.
