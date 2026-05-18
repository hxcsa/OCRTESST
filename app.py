import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="OCR Benchmark Viewer", layout="wide")

OUTPUTS = Path("outputs")
RAW_OCR = OUTPUTS / "raw_ocr"
STRUCTURED = OUTPUTS / "structured_json"
ANNOTATED = OUTPUTS / "annotated"
LLM_CLEANED = OUTPUTS / "llm_cleaned"

st.title("OCR Benchmark Viewer")

# --- Sidebar: filters ---
st.sidebar.header("Filters")

ocr_engines = sorted([d.name for d in RAW_OCR.iterdir() if d.is_dir()])
structured_engines = sorted([d.name for d in STRUCTURED.iterdir() if d.is_dir()])
all_engines = sorted(set(ocr_engines + structured_engines))

if not all_engines:
    st.warning("No results found. Run some OCR engines or VLM first.")
    st.stop()

engine = st.sidebar.selectbox("Engine", all_engines, index=all_engines.index("tesseract") if "tesseract" in all_engines else 0)

is_vlm = engine.startswith("vlm_")
if is_vlm:
    mode = "raw"
    st.sidebar.selectbox("Preprocessing Mode", ["raw"], disabled=True)
else:
    mode = st.sidebar.selectbox("Preprocessing Mode", ["raw", "clean", "binarized"])

raw_engine_dir = RAW_OCR / engine / mode
structured_engine_dir = STRUCTURED / engine / mode
structured_flat_dir = STRUCTURED / engine
annotated_engine_dir = ANNOTATED / engine / mode
llm_cleaned_engine_dir = LLM_CLEANED / engine / mode

pages = []
if raw_engine_dir.exists():
    for f in sorted(raw_engine_dir.glob("*.raw.json")):
        pages.append(f.stem.replace(".raw", ""))
if structured_engine_dir.exists():
    for f in sorted(structured_engine_dir.glob("*.json")):
        if f.stem not in pages:
            pages.append(f.stem)
if structured_flat_dir.exists() and is_vlm:
    for f in sorted(structured_flat_dir.glob("*.json")):
        stem = f.stem
        if "_20" in stem and len(stem.split("_")[-1]) == 16:
            continue
        if stem not in pages:
            pages.append(stem)

if not pages:
    st.warning(f"No results found for engine={engine}, mode={mode}")
    st.stop()

page = st.sidebar.selectbox("Page", pages)

# --- Build paths ---
raw_json_path = raw_engine_dir / f"{page}.raw.json"
txt_path = raw_engine_dir / f"{page}.txt"
structured_path = structured_engine_dir / f"{page}.json"
if not structured_path.exists() and is_vlm:
    structured_path = structured_flat_dir / f"{page}.json"
    if not structured_path.exists():
        for f in sorted(structured_flat_dir.glob("*.json")):
            if page in f.stem or f.stem in page:
                structured_path = f
                break

annotated_path = annotated_engine_dir / f"{page}_annotated.png"
llm_cleaned_path = llm_cleaned_engine_dir / f"{page}_cleaned.txt"
# Also try without mode prefix for VLM
if not llm_cleaned_path.exists():
    llm_cleaned_path = None

# --- Check for any LLM cleaned files ---
has_any_cleaned = False
if llm_cleaned_engine_dir.exists():
    has_any_cleaned = any(llm_cleaned_engine_dir.glob("*_cleaned.txt"))

# --- Load data ---
raw_data = None
structured_data = None

if raw_json_path.exists():
    raw_data = json.loads(raw_json_path.read_text())
if structured_path.exists():
    structured_data = json.loads(structured_path.read_text())

# --- Banner for LLM cleaned availability ---
if has_any_cleaned:
    st.info("LLM-cleaned text available for this engine/mode. Check the 'LLM Cleaned' tab below.")

# --- Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Annotated Image")
    if annotated_path.exists():
        st.image(str(annotated_path), width="stretch")
    else:
        st.info("No annotated image available for this engine/mode.")

with col2:
    st.subheader("Metrics")
    data = raw_data or structured_data or {}
    metrics_cols = st.columns(3)
    metrics_cols[0].metric(
        "OCR Confidence",
        f"{data.get('avg_ocr_confidence', 0) * 100:.1f}%" if data.get("avg_ocr_confidence") else "N/A",
    )
    metrics_cols[1].metric(
        "Runtime",
        f"{data.get('runtime_seconds', 0):.2f}s" if data.get("runtime_seconds") else "N/A",
    )
    metrics_cols[2].metric(
        "GPU Memory",
        f"{data.get('gpu_memory_mb', 0):.0f} MB" if data.get("gpu_memory_mb") else "N/A",
    )
    if data.get("error"):
        st.error(f"Error: {data['error']}")

    if raw_data and raw_data.get("boxes"):
        st.subheader("Confidence per Box")
        confs = [(b["confidence"] * 100, b["text"][:40]) for b in raw_data["boxes"]]
        confs_sorted = sorted(confs, key=lambda x: x[0])
        st.bar_chart(
            {c[1]: c[0] for c in confs_sorted[-30:]},
            y_label="Confidence %",
            horizontal=True,
        )

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["OCR Text", "LLM Cleaned", "Structured Fields", "Raw JSON"])

with tab1:
    st.subheader("Extracted Text (Raw OCR)")
    original_text = ""
    if txt_path.exists():
        original_text = txt_path.read_text()
    elif raw_data and raw_data.get("text"):
        original_text = raw_data["text"]
    elif structured_data and structured_data.get("raw_text"):
        original_text = structured_data["raw_text"]

    if original_text:
        st.text_area("OCR Output", original_text, height=400, key="ocr_text")
    else:
        st.info("No text output available.")

with tab2:
    st.subheader("LLM Cleaned Text")
    if llm_cleaned_path and llm_cleaned_path.exists():
        cleaned_text = llm_cleaned_path.read_text()
        st.caption("Fixed by lightweight LLM — corrected garbled characters, restored formatting, improved readability.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original OCR**")
            st.text(original_text)
        with c2:
            st.markdown("**LLM Cleaned**")
            st.text(cleaned_text)
    elif has_any_cleaned:
        st.info(f"No cleaned version for page {page} yet. Run `python main.py run-llm-clean --engine {engine}` to generate it.")
    else:
        st.info(
            "No LLM cleaned text available for this engine/mode. "
            f"Run `python main.py run-llm-clean --engine {engine} --mode {mode}` to generate cleaned versions."
        )

with tab3:
    st.subheader("Structured Extraction")
    if structured_data:
        fields = {
            "Document Type": structured_data.get("document_type"),
            "Full Name": structured_data.get("full_name"),
            "National ID": structured_data.get("national_id"),
            "Reference Number": structured_data.get("reference_number"),
            "Ministry / Department": structured_data.get("ministry_or_department"),
            "Issue Date": structured_data.get("issue_date"),
            "Address": structured_data.get("address"),
            "Subject": structured_data.get("subject"),
            "Decision / Status": structured_data.get("decision_or_status"),
            "Stamp Present": structured_data.get("stamp_present"),
            "Signature Present": structured_data.get("signature_present"),
            "Confidence Notes": structured_data.get("confidence_notes"),
        }
        for label, value in fields.items():
            if value is not None:
                st.markdown(f"**{label}:** {value}")
            else:
                st.markdown(f"**{label}:** *null*")
    else:
        st.info("No structured extraction available.")

with tab4:
    st.subheader("Raw Data")
    if raw_data:
        st.json(raw_data, expanded=False)
    elif structured_data:
        st.json(structured_data, expanded=False)
    else:
        st.info("No data available.")

# --- Footer: comparison ---
st.divider()
st.subheader("Cross-Engine Confidence Comparison")

all_engines_raw = sorted([d.name for d in RAW_OCR.iterdir() if (RAW_OCR / d.name / "raw").exists()])
if all_engines_raw:
    comp_data = {}
    for e in all_engines_raw:
        raw_dir = RAW_OCR / e / "raw"
        confs = []
        for f in raw_dir.glob("*.raw.json"):
            try:
                d = json.loads(f.read_text())
                if d.get("avg_ocr_confidence"):
                    confs.append(d["avg_ocr_confidence"])
            except Exception:
                pass
        if confs:
            comp_data[e] = sum(confs) / len(confs) * 100
    if comp_data:
        st.bar_chart(comp_data, y_label="Avg Confidence %")
