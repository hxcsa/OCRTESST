from __future__ import annotations

import re
from typing import Any


_DIGIT = r"0-9\u0660-\u0669\u06f0-\u06f9"
_DATE_RE = re.compile(rf"^[{_DIGIT}]{{1,4}}[/\-.][{_DIGIT}]{{1,2}}[/\-.][{_DIGIT}]{{1,4}}$")
_DOC_NUM_RE = re.compile(rf"^[{_DIGIT}][{_DIGIT}/\-. ]*[{_DIGIT}]$")
_PHONE_RE = re.compile(r"^\+?\d[\d\s\-()]{7,}\d$")
_AREA_RE = re.compile(rf"^[{_DIGIT},.\s]+(?:م٢|م²|م2|m2|m²|sqm|م)?$")


def _status(status: str, message: str) -> dict[str, str]:
    return {"status": status, "message": message}


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == []


def _has_repetition(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    words = value.split()
    if len(words) < 8:
        return False
    for size in (2, 3, 4):
        chunks = [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]
        if any(chunks.count(chunk) >= 4 for chunk in set(chunks)):
            return True
    return False


def validate_extraction(data: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, dict[str, str]] = {}

    document_number = data.get("document_number")
    if _empty(document_number):
        fields["document_number"] = _status("yellow", "missing document number")
    elif not isinstance(document_number, str) or not _DOC_NUM_RE.match(document_number.strip()):
        fields["document_number"] = _status("red", "document number contains invalid characters or shape")
    else:
        fields["document_number"] = _status("green", "valid document number shape")

    date = data.get("document_date")
    if _empty(date):
        fields["document_date"] = _status("yellow", "missing document date")
    elif not isinstance(date, str) or not _DATE_RE.match(date.strip()):
        fields["document_date"] = _status("red", "date must be shaped like day/month/year or year/month/day")
    else:
        fields["document_date"] = _status("green", "valid date shape")

    area = data.get("area_m2")
    if _empty(area):
        fields["area_m2"] = _status("yellow", "missing area")
    elif not isinstance(area, str) or not _AREA_RE.match(area.strip()):
        fields["area_m2"] = _status("red", "area should be numeric with optional square-meter marker")
    else:
        fields["area_m2"] = _status("green", "valid area shape")

    phone = data.get("phone_number")
    if _empty(phone):
        fields["phone_number"] = _status("yellow", "missing phone number")
    elif not isinstance(phone, str) or not _PHONE_RE.match(phone.strip()):
        fields["phone_number"] = _status("red", "phone number shape is invalid")
    else:
        fields["phone_number"] = _status("green", "valid phone number shape")

    subject = data.get("subject")
    if _empty(subject):
        fields["subject"] = _status("yellow", "missing subject")
    elif not isinstance(subject, str):
        fields["subject"] = _status("red", "subject must be text")
    elif len(subject) > 180:
        fields["subject"] = _status("red", "subject is too long; likely body text leakage")
    elif _has_repetition(subject):
        fields["subject"] = _status("red", "subject contains repeated text")
    else:
        fields["subject"] = _status("green", "subject length looks plausible")

    for key in ("recipient", "project_name", "sender_or_department", "full_name", "address", "decision_or_status"):
        value = data.get(key)
        if _empty(value):
            fields[key] = _status("yellow", f"missing {key}")
        elif isinstance(value, str) and _has_repetition(value):
            fields[key] = _status("red", f"{key} contains repeated text")
        elif isinstance(value, str) and len(value) > 260:
            fields[key] = _status("yellow", f"{key} is unusually long")
        else:
            fields[key] = _status("green", f"{key} present")

    try:
        confidence = float(data.get("extraction_confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.5:
        fields["extraction_confidence"] = _status("red", "low model confidence")
    elif confidence < 0.8:
        fields["extraction_confidence"] = _status("yellow", "medium model confidence")
    else:
        fields["extraction_confidence"] = _status("green", "high model confidence")

    counts = {"green": 0, "yellow": 0, "red": 0}
    for item in fields.values():
        counts[item["status"]] += 1
    return {"field_validation": fields, "validation_summary": counts}
