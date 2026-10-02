"""PII minimisation before any model call (spec §62, §63).

* Contact details -> placeholders ([EMAIL], [PHONE], [URL]).
* Lines carrying protected or irrelevant personal attributes (date of birth, age, gender,
  marital status, religion, caste, nationality, parents' names, government ids, photos,
  health details) are replaced wholesale with [REDACTED], so no model ever sees them and
  no assessment can be influenced by them.
The raw text is kept encrypted for re-processing; only the redacted text leaves the box.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_URL = re.compile(r"\b(?:https?://|www\.)\S+|\b(?:linkedin|github|behance|dribbble)\.com/\S*", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s\-.]?)?(?:\(?\d{2,5}\)?[\s\-.]?)?\d{3,5}[\s\-.]?\d{3,5}(?!\w)")
_ID_NUMBERS = re.compile(
    r"\b(?:[A-Z]{5}\d{4}[A-Z]|\d{4}\s?\d{4}\s?\d{4}|[A-PR-WY][1-9]\d\s?\d{4}[1-9])\b"  # PAN, Aadhaar, passport
)
_PROTECTED_LINE = re.compile(
    r"^\s*(?:[-•*]\s*)?(date\s*of\s*birth|d\.?\s*o\.?\s*b\.?|birth\s*date|born\s+on|age|gender|sex|"
    r"marital\s*status|married|religion|caste|category|nationality|citizenship|father'?s?\s*name|"
    r"mother'?s?\s*name|husband'?s?\s*name|spouse|blood\s*group|height|weight|health|disabilit(?:y|ies)|"
    r"passport(?:\s*no\.?)?|aadhaa?r|pan(?:\s*no\.?)?|permanent\s*address|address|photo(?:graph)?|"
    r"visa\s*status|languages?\s*known\s*\(mother\s*tongue\)|hobbies?\s*&?\s*interests?\s*:\s*(?:temple|church|mosque))"
    r"\s*[:\-–]\s*.*$",
    re.IGNORECASE | re.MULTILINE,
)


def redact(text: str, kind: str = "cv") -> Tuple[str, Dict[str, int]]:
    """kind='cv' also strips protected-attribute lines and id numbers; a JD only loses contact details."""
    counts: Dict[str, int] = {}

    def sub(rx: re.Pattern, repl: str, t: str, key: str) -> str:
        new, n = rx.subn(repl, t)
        if n:
            counts[key] = counts.get(key, 0) + n
        return new

    t = text or ""
    if kind == "cv":
        t = sub(_PROTECTED_LINE, "[REDACTED]", t, "protected_lines")
        t = sub(_ID_NUMBERS, "[ID]", t, "id_numbers")
    t = sub(_EMAIL, "[EMAIL]", t, "emails")
    t = sub(_URL, "[URL]", t, "urls")

    def phone_repl(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        # Years, percentages and money ranges are not phone numbers.
        if len(digits) < 9 or len(digits) > 14:
            return m.group(0)
        counts["phones"] = counts.get("phones", 0) + 1
        return "[PHONE]"

    t = _PHONE.sub(phone_repl, t)
    return t, counts
