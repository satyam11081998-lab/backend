"""Deterministic question QA (spec §52, §56). Runs on every candidate question — curated,
generated or CV-specific — before it can enter a blueprint."""

from __future__ import annotations

import re
from typing import List, Sequence

from ..textutil import jaccard

PROTECTED = re.compile(
    r"\b(how old|your age|date of birth|born in|married|marital|spouse|husband|wife|boyfriend|girlfriend|"
    r"pregnan|children|kids|plan(ning)? (a|to have) (family|baby)|religio|caste|church|temple|mosque|"
    r"pray|ethnic|race|nationality|citizen|immigra|visa status|disabilit|health condition|illness|"
    r"medical history|mental health|sexual|gender identity|political|vote|union member)\w*",
    re.IGNORECASE,
)
LEADING = re.compile(r"^(wouldn'?t you agree|don'?t you think|isn'?t it true|surely you|you would agree)\b",
                     re.IGNORECASE)
# No trailing \b: after a colon the next char is usually a space, so \b would never match.
LEAK = re.compile(r"\b(the (correct|right) answer is\b|hint\s*:|answer\s*:)", re.IGNORECASE)


def problems(text: str) -> List[str]:
    t = (text or "").strip()
    out: List[str] = []
    if len(t) < 15:
        out.append("too_short")
    if len(t) > 600:
        out.append("too_long")
    if PROTECTED.search(t):
        out.append("protected_topic")
    if LEADING.search(t):
        out.append("leading")
    if LEAK.search(t):
        out.append("answer_leak")
    if t.count("?") > 2:
        out.append("stacked_questions")
    if "*" in t or "#" in t or t.startswith("-"):
        out.append("markdown")
    return out


def is_near_duplicate(text: str, others: Sequence[str], threshold: float = 0.6) -> bool:
    return any(jaccard(text, o) >= threshold for o in others if o)
