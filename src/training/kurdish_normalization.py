from __future__ import annotations

import re
import unicodedata
from typing import Any


_CHAR_MAP = str.maketrans(
    {
        "\u0643": "\u06a9",  # Arabic kaf -> Kurdish kaf
        "\u064a": "\u06cc",  # Arabic ya -> Farsi/Kurdish ya
        "\u0649": "\u06cc",  # Alef maksura -> ya
        "\u0629": "\u06d5",  # Ta marbuta -> Kurdish ae
        "\u06c0": "\u06d5",  # Heh with yeh above -> Kurdish ae
        "\u06be": "\u0647",  # Do-chashmi heh -> Arabic heh
        "\u0640": "",  # Tatweel
        "\u200c": " ",  # ZWNJ
        "\u200d": "",  # ZWJ
        "\u200e": "",
        "\u200f": "",
        "\ufeff": "",
    }
)

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "٠١٢٣٤٥٦٧٨٩")
_EASTERN_ARABIC_INDIC = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "٠١٢٣٤٥٦٧٨٩")


def normalize_kurdish_text(text: str, *, normalize_digits: bool = True) -> str:
    """Normalize common Sorani keyboard/encoding variants without translating text."""
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u0647\u200c", "\u06d5")  # Common heh+ZWNJ ae typing artifact
    text = text.translate(_CHAR_MAP)
    if normalize_digits:
        text = text.translate(_ARABIC_INDIC).translate(_EASTERN_ARABIC_INDIC)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([،؛؟:])", r"\1", text)
    text = re.sub(r"([،؛؟:])(?=\S)", r"\1 ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_json_strings(value: Any) -> Any:
    """Recursively normalize string leaves in JSON-like training targets."""
    if isinstance(value, str):
        return normalize_kurdish_text(value)
    if isinstance(value, list):
        return [normalize_json_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_json_strings(item) for key, item in value.items()}
    return value
