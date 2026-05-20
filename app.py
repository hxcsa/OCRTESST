from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Kurdish Document Review", layout="wide")

ROOT = Path(".")
OUTPUTS = ROOT / "outputs"
STRUCTURED = OUTPUTS / "structured_json"
REVIEWED = OUTPUTS / "reviewed_json"
GROUND_TRUTH_JSONL = ROOT / "data" / "ground_truth" / "dataset.jsonl"

FIELD_KEYS = [
    "recipient",
    "subject",
    "document_number",
    "document_date",
    "project_name",
    "location",
    "area_m2",
    "referenced_decision_number",
    "attachments",
    "sender_or_department",
    "full_name",
    "national_id",
    "phone_number",
    "address",
    "decision_or_status",
    "stamp_present",
    "signature_present",
    "uncertain_lines",
    "extraction_confidence",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_ground_truth(data: dict[str, Any]) -> None:
    image_path = data.get("image_path")
    target = {key: data.get(key) for key in FIELD_KEYS}
    record = {
        "task": "metadata",
        "image": image_path,
        "source": data.get("source"),
        "document_id": data.get("document_id"),
        "page": data.get("page"),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Extract the document metadata as valid JSON using the project schema."},
                ],
            },
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
        ],
    }
    GROUND_TRUTH_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with GROUND_TRUTH_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _result_files() -> list[Path]:
    if not STRUCTURED.exists():
        return []
    latest_run_files = []
    qwen_runs = STRUCTURED / "vlm_qwen_gguf" / "runs"
    if qwen_runs.exists():
        run_dirs = sorted([p for p in qwen_runs.iterdir() if p.is_dir()])
        if run_dirs:
            latest_run_files = sorted(run_dirs[-1].glob("*.json"))
    normal_files = sorted(STRUCTURED.glob("*/*.json")) + sorted(STRUCTURED.glob("*/*/*.json"))
    normal_files = [p for p in normal_files if "/runs/" not in str(p)]
    return latest_run_files + normal_files


@st.cache_data(show_spinner=False)
def load_rows() -> list[dict[str, Any]]:
    rows = []
    for path in _result_files():
        try:
            data = _read_json(path)
        except Exception:
            continue
        rows.append(
            {
                "path": str(path),
                "engine": data.get("engine") or path.parent.name,
                "document_id": data.get("document_id") or path.stem,
                "page": int(data.get("page") or 0),
                "source": data.get("source"),
                "image_path": data.get("image_path"),
                "subject": data.get("subject"),
                "document_number": data.get("document_number"),
                "document_date": data.get("document_date"),
                "project_name": data.get("project_name"),
                "location": data.get("location"),
                "confidence": data.get("extraction_confidence") or data.get("avg_ocr_confidence"),
                "runtime_seconds": data.get("runtime_seconds"),
                "run_id": data.get("run_id"),
                "extracted_at": data.get("extracted_at"),
                "red_flags": data.get("validation_summary", {}).get("red", 0),
                "yellow_flags": data.get("validation_summary", {}).get("yellow", 0),
                "error": data.get("error"),
            }
        )
    return rows


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def _parse_list(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


st.title("Kurdish Document Review")

rows = load_rows()
if not rows:
    st.warning("No structured extraction results found. Run `python main.py run-vlm --model qwen-gguf --all` first.")
    st.stop()

df = pd.DataFrame(rows)

st.sidebar.header("Filters")
engines = sorted(df["engine"].dropna().unique())
engine = st.sidebar.selectbox("Engine", engines, index=engines.index("vlm_qwen_gguf") if "vlm_qwen_gguf" in engines else 0)
filtered = df[df["engine"] == engine].copy()

documents = ["All"] + sorted(filtered["document_id"].dropna().unique())
doc_filter = st.sidebar.selectbox("Document", documents)
if doc_filter != "All":
    filtered = filtered[filtered["document_id"] == doc_filter]

query = st.sidebar.text_input("Search fields")
if query:
    q = query.casefold()
    mask = filtered.apply(lambda row: q in " ".join(str(v) for v in row.values if v is not None).casefold(), axis=1)
    filtered = filtered[mask]

show_errors = st.sidebar.checkbox("Errors only", value=False)
if show_errors:
    filtered = filtered[filtered["error"].notna()]

show_flags = st.sidebar.checkbox("Validation flags only", value=False)
if show_flags:
    filtered = filtered[(filtered["red_flags"].fillna(0).astype(int) > 0) | (filtered["yellow_flags"].fillna(0).astype(int) > 0)]

min_conf = st.sidebar.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
filtered = filtered[(filtered["confidence"].fillna(0).astype(float) >= min_conf)]

if filtered.empty:
    st.info("No pages match the current filters.")
    st.stop()

summary_cols = st.columns(4)
summary_cols[0].metric("Pages", len(filtered))
summary_cols[1].metric("Documents", filtered["document_id"].nunique())
summary_cols[2].metric("Errors", int(filtered["error"].notna().sum()))
summary_cols[3].metric("Avg Confidence", f"{filtered['confidence'].dropna().astype(float).mean():.2f}" if filtered["confidence"].notna().any() else "N/A")
latest_run = filtered["run_id"].dropna().iloc[0] if "run_id" in filtered and filtered["run_id"].notna().any() else None
if latest_run:
    st.caption(f"Showing run: {latest_run}")

table_cols = ["document_id", "page", "subject", "document_number", "document_date", "project_name", "location", "confidence", "red_flags", "yellow_flags", "runtime_seconds", "extracted_at", "error"]
st.dataframe(filtered[table_cols].sort_values(["document_id", "page"]), width="stretch", hide_index=True)

options = filtered.sort_values(["document_id", "page"])["path"].tolist()
labels = {
    row["path"]: f"{row['document_id']} | page {row['page']:03d} | {row.get('subject') or 'no subject'}"
    for row in filtered.to_dict("records")
}
selected_path = st.selectbox("Open page", options, format_func=lambda value: labels.get(value, value))

data_path = Path(selected_path)
data = _read_json(data_path)

st.divider()
left, right = st.columns([1.05, 1])

with left:
    st.subheader("Document Image")
    image_path = data.get("image_path")
    if image_path and Path(image_path).exists():
        st.image(image_path, width="stretch")
    else:
        st.info("No image path found for this extraction.")

    with st.expander("Raw JSON", expanded=False):
        st.json(data, expanded=True)

with right:
    st.subheader("Extracted Fields")
    reviewed = {key: data.get(key) for key in FIELD_KEYS}
    validation = data.get("field_validation", {})
    summary = data.get("validation_summary", {})
    if summary:
        flag_cols = st.columns(3)
        flag_cols[0].metric("Green", summary.get("green", 0))
        flag_cols[1].metric("Yellow", summary.get("yellow", 0))
        flag_cols[2].metric("Red", summary.get("red", 0))

    with st.form("review_form"):
        edited: dict[str, Any] = {}
        for key in FIELD_KEYS:
            value = reviewed.get(key)
            field_status = validation.get(key, {})
            if field_status.get("status") == "red":
                st.warning(f"{key}: {field_status.get('message', 'validation failed')}")
            elif field_status.get("status") == "yellow":
                st.info(f"{key}: {field_status.get('message', 'needs review')}")
            if key in ("stamp_present", "signature_present"):
                edited[key] = st.checkbox(key, value=bool(value))
            elif key == "extraction_confidence":
                edited[key] = st.slider(key, 0.0, 1.0, float(value or 0.0), 0.01)
            elif key in ("attachments", "uncertain_lines"):
                edited[key] = _parse_list(st.text_area(key, _as_text(value), height=90))
            else:
                edited[key] = st.text_input(key, _as_text(value))
                if edited[key] == "":
                    edited[key] = None

        status = st.selectbox("review_status", ["unchecked", "accepted", "needs_fix", "rejected"])
        notes = st.text_area("review_notes", "")
        submitted = st.form_submit_button("Save Review")
        approved = st.form_submit_button("Approve & Save Fine-Tune Label")

    if submitted or approved:
        saved = dict(data)
        saved.update(edited)
        saved["review_status"] = "accepted" if approved else status
        saved["review_notes"] = notes
        saved["review_source_path"] = str(data_path)
        out = REVIEWED / data.get("engine", engine) / data_path.name
        _write_json(out, saved)
        if approved:
            _append_ground_truth(saved)
        load_rows.clear()
        st.success(f"Saved review to {out}")
        if approved:
            st.success(f"Appended fine-tune label to {GROUND_TRUTH_JSONL}")

st.caption("Tip: reviewed files are saved separately under outputs/reviewed_json so raw model outputs stay intact.")
