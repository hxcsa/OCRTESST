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

# Common document type keywords (English / Kurdish Sorani / Arabic)
_DOC_TYPE_PATTERNS = [
    r"((?:Identity|ID)\s*Card)",
    r"((?:Birth|Death|Marriage)\s*Certificate)",
    r"((?:Official|Governmental)\s*Letter)",
    r"((?:Decree|Decision|Order)\b[^\n]{0,60})",
    r"(بەلگەنامەی\s*[^\n]{0,60})",         # Kurdish: document of ...
    r"(کارتی\s*(?:ناسنامە|ناسینەوە)[^\n]{0,40})",  # Kurdish: identity card
    r"((?:شهادة|وثيقة|قرار|قانون|أمر)[^\n]{0,60})",  # Arabic document types
]

# Full name patterns (Arabic/Kurdish label + value, or Latin ALL-CAPS name heuristic)
_NAME_PATTERNS = [
    r"(?:name|full\s*name|ناوی\s*تەواو|الاسم\s*الكامل|الاسم)\s*[:：\-]?\s*([\w\u0600-\u06FF][^\n]{2,60})",
    r"(?:ناو|الاسم)\s*[:：\-]?\s*([\w\u0600-\u06FF][^\n]{2,50})",
    # Latin heuristic: 2-4 ALL-CAPS words (common in scanned forms)
    r"^([A-Z][A-Z]+(?:\s+[A-Z][A-Z]+){1,3})$",
]


def _first(patterns: list[str], text: str) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip(" ::\u202c-\t\r\n")
    return None


def extract_structured_fields(raw_text: str, stamp_present=None, signature_present=None, notes: str = "") -> dict:
    text = raw_text or ""
    data = {key: None for key in SCHEMA_KEYS}
    data["document_type"] = _first(_DOC_TYPE_PATTERNS, text)
    data["full_name"] = _first(_NAME_PATTERNS, text)
    data["national_id"] = _first([r"(?:national\s*id|id|\u0698\u0645\u0627\u0631\u06d5\u06cc\s*\u0646\u06cc\u0634\u062a\u0645\u0627\u0646\u06cc|\u0627\u0644\u0631\u0642\u0645\s*\u0627\u0644\u0648\u0637\u0646\u064a)\D*([0-9]{8,14})"], text)
    data["reference_number"] = _first([r"(?:ref(?:erence)?|no\.?|\u0698\u0645\u0627\u0631\u06d5|\u0627\u0644\u0639\u062f\u062f|\u0631\u0642\u0645)\D*([A-Za-z0-9\-\/]{3,})"], text)
    data["issue_date"] = _first(
        [
            r"([0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4})",
            r"(?:date|\u0628\u06d5\u0631\u0648\u0627\u0631|\u0627\u0644\u062a\u0627\u0631\u064a\u062e)\D*([0-9\u0660-\u0669]{1,2}[\/\-.][0-9\u0660-\u0669]{1,2}[\/\-.][0-9\u0660-\u0669]{2,4})",
        ],
        text,
    )
    data["ministry_or_department"] = _first(
        [
            r"((?:Ministry|Directorate|Department)[^\n]{0,80})",
            r"((?:\u0648\u06d5\u0632\u0627\u0631\u06d5\u062a\u06cc|\u0628\u06d5\u0695\u06ce\u0648\u06d5\u0628\u06d5\u0631\u0627\u06cc\u06d5\u062a\u06cc|\u0648\u0632\u0627\u0631\u0629|\u0645\u062f\u064a\u0631\u064a\u0629)[^\n]{0,80})",
        ],
        text,
    )
    data["subject"] = _first([r"(?:subject|\u0628\u0627\u0628\u06d5\u062a|\u0627\u0644\u0645\u0648\u0636\u0648\u0639)\s*[::\u202c-]?\s*(.+)"], text)
    data["address"] = _first([r"(?:address|\u0646\u0627\u0648\u0646\u06cc\u0634\u0627\u0646|\u0627\u0644\u0639\u0646\u0648\u0627\u0646)\s*[::\u202c-]?\s*(.+)"], text)
    data["decision_or_status"] = _first([r"(?:status|decision|\u0628\u0695\u06cc\u0627\u0631|\u0627\u0644\u062d\u0627\u0644\u0629|\u0627\u0644\u0642\u0631\u0627\u0631)\s*[::\u202c-]?\s*(.+)"], text)
    data["stamp_present"] = stamp_present
    data["signature_present"] = signature_present
    data["confidence_notes"] = notes or "Rule-based extraction; review manually for Kurdish/Arabic variation."
    return data

