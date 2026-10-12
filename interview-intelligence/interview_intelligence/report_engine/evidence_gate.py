"""Cheap report guard: do not infer competencies from greetings or meta-only turns.

Independent of the separate case/guesstimate AI brain. This does not grade answers.
"""
from __future__ import annotations
import re
from typing import Any, Iterable
_NO_ANSWER = {"", "hi", "hello", "hey", "hello there", "hi there", "test", "testing", "ok", "okay", "hmm", "hm", "idk", "i dont know", "i do not know", "not sure", "no idea", "pass", "skip", "[skipped]", "thanks", "thank you", "good morning", "good afternoon", "good evening"}
_META_ONLY = re.compile(r"^(?:(?:give|assign|score) me (?:a )?(?:perfect|10|100|10/10|100/100).*|reveal (?:the )?(?:system|hidden|developer|original) (?:prompt|instructions).*|you are now (?:the )?(?:system|admin|developer).*|change (?:your|the) (?:rules|rubric|score).*|are you (?:an? )?ai\??|which model are you\??)$", re.I)
_INJECTION_ONLY = re.compile(r"^ignore (?:all )?(?:previous|prior) instructions(?:[,.]?\s*(?:and\s*)?(?:give|assign|score) me .*)?[.! ]*$", re.I)
_NON_ANSWER_INTENT = re.compile(r"^(?:can|could|please|i want to|i would like to|let me|may i)\s+(?:skip|stop|end|leave|pause|take a break|repeat|hear that again)\b.*$", re.I)
def _value(message: Any, key: str, default: Any = None) -> Any:
    return message.get(key, default) if isinstance(message, dict) else getattr(message, key, default)
def has_substantive_candidate_answer(messages: Iterable[Any]) -> bool:
    """Small/numeric answers count; greetings, skips, and injection-only messages do not."""
    for message in messages or []:
        if str(_value(message, "role", "")).lower() != "candidate": continue
        text = " ".join(str(_value(message, "content", "") or "").split())
        norm = re.sub(r"[^a-z0-9%₹$]+", " ", text.lower()).strip()
        if norm in _NO_ANSWER or _META_ONLY.fullmatch(text.strip()) or _INJECTION_ONLY.fullmatch(text.strip()) or _NON_ANSWER_INTENT.fullmatch(text.strip()): continue
        if str(_value(message, "action", "") or "").lower() in {"skip", "greeting", "meta_question", "repeat_request", "thinking_pause", "end_request", "break_request"}: continue
        meta = _value(message, "meta", {}) or {}
        meta_intent = str(meta.get("intent", "")).lower() if isinstance(meta, dict) else ""
        if meta_intent in {"skip", "greeting", "meta_question", "repeat_request", "thinking_pause", "end_request", "break_request"}: continue
        if len(text) >= 12 or re.search(r"\d", text) or len(text.split()) >= 3: return True
    return False
