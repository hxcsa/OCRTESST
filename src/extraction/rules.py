from __future__ import annotations

import re


SCHEMA_KEYS = [
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


def _first(patterns: list[str], text: str) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip(" :：-\t\r\n")
    return None


def extract_structured_fields(raw_text: str, stamp_present=None, signature_present=None, notes: str = "") -> dict:
    text = raw_text or ""
    data = {key: None for key in SCHEMA_KEYS}
    data["national_id"] = _first([r"(?:national\s*id|id|ژمارەی\s*نیشتمانی|الرقم\s*الوطني)\D*([0-9]{8,14})"], text)
    data["reference_number"] = _first([r"(?:ref(?:erence)?|no\.?|ژمارە|العدد|رقم)\D*([A-Za-z0-9\-\/]{3,})"], text)
    data["issue_date"] = _first(
        [
            r"([0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4})",
            r"(?:date|بەروار|التاريخ)\D*([0-9٠-٩]{1,2}[\/\-.][0-9٠-٩]{1,2}[\/\-.][0-9٠-٩]{2,4})",
        ],
        text,
    )
    data["ministry_or_department"] = _first(
        [
            r"((?:Ministry|Directorate|Department)[^\n]{0,80})",
            r"((?:وەزارەتی|بەڕێوەبەرایەتی|وزارة|مديرية)[^\n]{0,80})",
        ],
        text,
    )
    data["subject"] = _first([r"(?:subject|بابەت|الموضوع)\s*[:：-]?\s*(.+)"], text)
    data["address"] = _first([r"(?:address|ناونیشان|العنوان)\s*[:：-]?\s*(.+)"], text)
    data["decision_or_status"] = _first([r"(?:status|decision|بڕیار|الحالة|القرار)\s*[:：-]?\s*(.+)"], text)
    data["stamp_present"] = stamp_present
    data["signature_present"] = signature_present
    data["confidence_notes"] = notes or "Rule-based extraction; review manually for Kurdish/Arabic variation."
    return data
