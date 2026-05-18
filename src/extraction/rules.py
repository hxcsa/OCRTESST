from __future__ import annotations

import re

SCHEMA_KEYS = [
    "document_type",
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
    "confidence_notes",
]

_DOC_TYPE_PATTERNS = [
    r"((?:Identity|ID)\s*Card)",
    r"((?:Birth|Death|Marriage)\s*Certificate)",
    r"((?:Official|Governmental)\s*Letter)",
    r"((?:Decree|Decision|Order)\b[^\n]{0,60})",
    r"(\u0628\u06d5\u0644\u06af\u06d5\u0646\u0627\u0645\u06d5\u06cc\s*[^\n]{0,60})",
    r"(\u06a9\u0627\u0631\u062a\u06cc\s*(?:\u0646\u0627\u0633\u0646\u0627\u0645\u06d5|\u0646\u0627\u0633\u06cc\u0646\u06d5\u0648\u06d5)[^\n]{0,40})",
    r"((?:\u0634\u0647\u0627\u062f\u0629|\u0648\u062b\u064a\u0642\u0629|\u0642\u0631\u0627\u0631|\u0642\u0627\u0646\u0648\u0646|\u0623\u0645\u0631)[^\n]{0,60})",
]


_KURDISH_YA = r"[\u06cc\u064a]"      # ی or ي
_KURDISH_HE = r"[\u06d5\u0647\u0629]"  # ە or ه or ة
_KURDISH_RA = r"[\u0695\u0631]"          # ڕ or ر
_KURDISH_KAF = r"[\u06a9\u0643]"          # ک or ك


def _first(patterns: list[str], text: str) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip(" ::\u202c-\t\r\n")
    return None


def extract_structured_fields(raw_text: str, stamp_present=None, signature_present=None, notes: str = "") -> dict:
    text = raw_text or ""
    data = {key: (None if key not in ("attachments", "uncertain_lines") else []) for key in SCHEMA_KEYS}
    data["document_type"] = _first(_DOC_TYPE_PATTERNS, text)
    data["recipient"] = _first(
        [
            r"(?:recipient|to)\s*[::\u202c\-]?\s*([^\n]{2,80})",
            r"(\u0631" + _KURDISH_YA + r"?\u0628\u0627\u0632\s*\u0647" + _KURDISH_YA + r"\u0648\u0627[^\n]{0,40})",
            r"(\u0628" + _KURDISH_HE + r"?\u0695\u06cc\u0632[^\n]{0,80})",
            r"(\u0628\u0647\u0631\u06cc\u0648\u0647\u0628" + _KURDISH_HE + r"?\u0631\u06cc[^\n]{0,80})",
        ],
        text,
    )
    data["subject"] = _first(
        [
            r"(\u0628\u0627\u0628" + _KURDISH_HE + r"\u062a[:\u202c\-]?\s*[^\n]{2,120})",
            r'(?:subject|\u0627\u0644\u0645\u0648\u0636\u0648\u0639|\u0633\u06d5\u0628\u0627\u0631\u06d5\u062a)\s*[::\u202c\-]?\s*([^\n]{2,120})',
        ],
        text,
    )
    data["document_number"] = _first(
        [
            r"(?:ref|no|number)\D*([A-Za-z0-9\u0660-\u0669/\-]{3,})",
            r"(\u0698\u0645\u0627\u0631" + _KURDISH_HE + r"?\s*[0-9\u0660-\u0669/\-]{2,})",
        ],
        text,
    )
    data["document_date"] = _first(
        [
            r"([0-9\u0660-\u0669]{1,4}[/\-.][0-9\u0660-\u0669]{1,2}[/\-.][0-9\u0660-\u0669]{1,4})",
        ],
        text,
    )
    data["project_name"] = _first(
        [
            r"(\u062f\u0627\u0648\u0646\s*\u062a\u0627\u0648\u0646[^\n]{0,40})",
            r"(\u067e\u0631" + _KURDISH_KAF + _KURDISH_HE + r"?\u0698\s*[^\n]{1,80})",
            r"(\u067e\u0631\u0698" + _KURDISH_HE + r"[^\n]{1,40})",
            r"(?:project|project\s*name)\s*[::\u202c\-]?\s*([^\n]{2,80})",
        ],
        text,
    )
    data["location"] = _first(
        [
            r"(\u0633\u0644" + _KURDISH_YA + r"\u0645\u0627\u0646" + _KURDISH_YA + r"?[^\n]{0,20})",
            r"((?:\u0634\u0627\u0631|\u0634\u06d5\u0647\u0631|\u0645\u062f\u06cc\u0646\u0629)[^\n]{0,40})",
        ],
        text,
    )
    data["area_m2"] = _first(
        [
            r"([\d,\u0660-\u0669]+(?:\.\d+)?)\s*(?:\u0645\u00b2|m\u00b2|sqm|\u0645)",
        ],
        text,
    )
    data["referenced_decision_number"] = _first(
        [
            r"(\u0628" + _KURDISH_RA + r"" + _KURDISH_YA + r"\u0627\u0631[^\n]{0,20}\u0646?" + _KURDISH_RA + r"?\u062c\u0648\u0645" + _KURDISH_HE + r"\u0646" + _KURDISH_YA + r"?[^\n]{0,60})",
            r"(\u0628\u0631\u06cc\u0627\u0631[^\n]{0,20}\u0646?\u062c\u0648\u0645" + _KURDISH_HE + r"\u0646" + _KURDISH_YA + r"?[^\n]{0,40})",
            r"(?:decision\s*no|decree|order)[.:;\s]*([A-Za-z0-9\u0660-\u0669/\-]{3,})",
        ],
        text,
    )
    data["sender_or_department"] = _first(
        [
            r"((?:Ministry|Directorate|Department|Investment)[^\n]{0,80})",
            r"(\u0648\u0647?\u0628\u06d5\u0631\u0647" + _KURDISH_YA + r"\u0646\u0627\u0646[^\n]{0,40})",
            r"(\u0628" + _KURDISH_HE + r"?\u0695\u06cc\u0648\u06d5\u0628\u06d5\u0631\u0627" + _KURDISH_YA + r"?\u06d5\u062a" + _KURDISH_YA + r"?[^\n]{0,80})",
            r"((?:\u0648\u06d5\u0632\u0627\u0631\u06d5\u062a\u06cc|\u0648\u0632\u0627\u0631\u0629|\u0645\u062f\u06cc\u0631\u064a\u0629)[^\n]{0,80})",
        ],
        text,
    )
    data["full_name"] = _first(
        [
            r"(?:name|full\s*name|\u0646\u0627\u0648\u06cc\s*\u062a\u06d5\u0648\u0627\u0648|\u0627\u0644\u0627\u0633\u0645\s*\u0627\u0644\u0643\u0627\u0645\u0644|\u0627\u0644\u0627\u0633\u0645)\s*[::\u202c\-]?\s*([\w\u0600-\u06FF][^\n]{2,60})",
            r"(?:\u0646\u0627\u0648|\u0627\u0644\u0627\u0633\u0645)\s*[::\u202c\-]?\s*([\w\u0600-\u06FF][^\n]{2,50})",
            r"^([A-Z][A-Z]+(?:\s+[A-Z][A-Z]+){1,3})$",
        ],
        text,
    )
    data["national_id"] = _first(
        [
            r"(?:national\s*id|id)\D*([0-9]{8,14})",
            r"([0-9]{8,14})",
        ],
        text,
    )
    data["phone_number"] = _first(
        [
            r"(\+?\d[\d\s\-]{7,}\d)",
            r"(\+?964[\d\s\-]{7,}\d)",
        ],
        text,
    )
    data["address"] = _first(
        [
            r"(?:address|\u0646\u0627\u0648\u0646\u06cc\u0634\u0627\u0646|\u0627\u0644\u0639\u0646\u0648\u0627\u0646)\s*[::\u202c\-]?\s*([^\n]+)",
        ],
        text,
    )
    data["decision_or_status"] = _first(
        [
            r"(?:status|decision)\s*[::\u202c\-]?\s*([^\n]+)",
            r"(\u0628" + _KURDISH_RA + r"" + _KURDISH_YA + r"\u0627\u0631[^\n]{0,80})",
        ],
        text,
    )
    data["stamp_present"] = stamp_present
    data["signature_present"] = signature_present
    data["confidence_notes"] = notes or "Rule-based extraction; review manually for Kurdish/Arabic variation."
    return data

