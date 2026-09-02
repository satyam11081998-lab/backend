"""
Answer-validity gate — decides, BEFORE the expensive scorer runs, whether a
submission is a genuine attempt to answer the given case. This is what stops
gibberish and off-topic text from quietly earning points.

Two layers, cheapest first:

  1. Deterministic pre-check (no API cost) — catches only the BLATANT cases:
     near-empty, pure repetition-padding, and keyboard-mashing. Deliberately
     conservative: a false positive (rejecting a real answer) is far worse than
     paying for one small model call, so anything remotely plausible is deferred
     to layer 2 rather than rejected here.

  2. Model screen (gpt-4o-mini, cheap, ~1s) — judges genuine relevance/effort for
     the ambiguous middle: a fluent paragraph that never engages THIS case
     (off_topic), or a real but underdeveloped attempt (thin).

Verdicts:
  - "valid"     -> score normally.
  - "thin"      -> score normally, but the scorer is told NOT to inflate.
  - "off_topic" -> HARD GATE: score 0 with an explanation; the full scorer is skipped.
  - "gibberish" -> HARD GATE: score 0 with an explanation; the full scorer is skipped.

`screen_answer` never raises: if the model screen errors it fails OPEN (returns
"valid"), because a hiccup in the screen must never block a paying user — the
main scorer's own strict rubric is still there behind it.
"""

from __future__ import annotations

import os
import re
import json
import time
from typing import Dict, Any, Optional

from openai import OpenAI
from dotenv import load_dotenv

from services.ai_usage import log_ai_usage
from services.ai_providers import chat_with_fallback

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
VALIDITY_MODEL = "gpt-4o-mini"

# Verdicts that mean "this is not a real case answer" -> score 0, skip full scorer.
HARD_GATE = {"gibberish", "off_topic"}
VALID_VERDICTS = {"valid", "thin", "off_topic", "gibberish"}

_WORD = re.compile(r"[A-Za-z]+")
_TOKEN = re.compile(r"\S+")


def _looks_like_word(w: str) -> bool:
    """A real English-ish word is 2-20 letters and almost always contains a vowel."""
    w = w.lower()
    if len(w) < 2 or len(w) > 20:
        return False
    return bool(re.search(r"[aeiou]", w))


def deterministic_verdict(text: str) -> Optional[Dict[str, Any]]:
    """
    Return a verdict dict ONLY for blatant nonsense; otherwise None (defer to the
    model screen). Kept intentionally narrow to avoid false positives.
    """
    t = (text or "").strip()
    if len(t) < 40:
        return _v("gibberish", 0, 0, "Almost no content.", "deterministic")

    tokens = _TOKEN.findall(t)
    if not tokens:
        return _v("gibberish", 0, 0, "No readable words.", "deterministic")

    lower = [w.lower() for w in tokens]
    uniq_ratio = len(set(lower)) / len(lower)
    # Repetition padding: the same handful of tokens copied to clear the length gate.
    if len(tokens) >= 20 and uniq_ratio < 0.15:
        return _v("gibberish", 0, 0,
                  "The same few words are repeated to fill space.", "deterministic")

    words = _WORD.findall(t)
    digit_ratio = sum(c.isdigit() for c in t) / max(1, len(t))
    # Keyboard mashing: lots of alpha 'words' but almost none are pronounceable.
    # Guarded by digit_ratio so a legitimately number-heavy guesstimate is never hit.
    if len(words) >= 8 and digit_ratio < 0.20:
        pronounceable = sum(1 for w in words if _looks_like_word(w)) / len(words)
        if pronounceable < 0.35:
            return _v("gibberish", 0, 0,
                      "The text does not resolve into real words.", "deterministic")

    return None


_SCREEN_SYSTEM = """You are a strict screener for a case-interview practice tool. You do NOT score. \
Your only job is to decide whether a submission is a GENUINE attempt to solve THE GIVEN CASE, so a \
downstream scorer is never fooled into rewarding nonsense.

Return ONLY JSON: {"verdict": "...", "relevance": 0-100, "effort": 0-100, "reason": "<= 25 words"}

verdict must be exactly one of:
- "valid": a real, on-topic attempt to solve this case (even if weak, partial, or wrong).
- "thin": on-topic and genuine, but too short/underdeveloped to show real reasoning.
- "off_topic": coherent text that does NOT address THIS case — a copy-paste, an answer to a \
different question, or an essay unrelated to the prompt.
- "gibberish": not a real answer — random words, keyboard mashing, filler or padding, a joke, \
"I don't know", placeholder/test text, or anything with no honest attempt to solve the case.

Be strict. If it does not read as an honest attempt to solve THIS specific case, it is off_topic or \
gibberish, NOT valid. A fluent, confident-sounding paragraph that never engages the case's actual \
question is off_topic, not valid. When genuinely torn between valid and thin, choose thin. Judge \
GENUINENESS and RELEVANCE only — never answer quality; that is the scorer's job."""


def _model_verdict(
    case_content: str,
    case_type: str,
    user_answer: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    user = f"""CASE TYPE: {case_type}

CASE PROMPT:
\"\"\"
{(case_content or '')[:4000]}
\"\"\"

CANDIDATE SUBMISSION:
\"\"\"
{(user_answer or '')[:6000]}
\"\"\"

Decide the verdict. Return ONLY the JSON."""
    try:
        t0 = time.time()
        # Provider chosen by the admin toggle ("validity"): Groq when selected/available,
        # else OpenAI. On ANY Groq error (incl. JSON-mode quirks) it falls back to OpenAI,
        # so the gate can never silently vanish.
        resp, used_model, _prov = chat_with_fallback(
            "validity",
            messages=[
                {"role": "system", "content": _SCREEN_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        log_ai_usage(user_id=user_id, endpoint="/submit", model=used_model,
                     response=resp, latency_ms=int((time.time() - t0) * 1000))
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        # Fail OPEN — never block a real user because the screen hiccuped. The
        # main scorer's strict rubric still stands behind this.
        return _v("valid", 60, 60, "screen-unavailable", "fallback")

    verdict = str(data.get("verdict", "valid")).lower().strip()
    if verdict not in VALID_VERDICTS:
        verdict = "valid"
    return _v(
        verdict,
        _int0_100(data.get("relevance"), 50),
        _int0_100(data.get("effort"), 50),
        str(data.get("reason", ""))[:200],
        "model",
    )


def screen_answer(
    case_content: str,
    case_type: str,
    user_answer: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Blatant nonsense is caught for free; everything else goes to the model screen."""
    det = deterministic_verdict(user_answer)
    if det is not None:
        return det
    return _model_verdict(case_content, case_type, user_answer, user_id)


def _v(verdict: str, relevance: int, effort: int, reason: str, source: str) -> Dict[str, Any]:
    return {
        "verdict": verdict,
        "relevance": relevance,
        "effort": effort,
        "reason": reason,
        "source": source,
    }


def _int0_100(v: Any, default: int) -> int:
    try:
        return max(0, min(100, int(round(float(v)))))
    except (TypeError, ValueError):
        return default
