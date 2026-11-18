"""
Short lines: the interviewer's brief beats, worded by a model instead of picked from a list
(step 2 of the prompt-led interviewer, 2026-11-18).

WHY
When a typed turn needs only a beat ("ok", "ek second, calculating", "I'll start with households", "stop
asking me questions"), the interviewer answered with one of 16 fixed lines picked by a hash: "Got it.",
"Go ahead.", "Take your time.", "Fair enough. Over to you." Correct, and the most robotic thing in the product:
the same four words for every student, in English even when the student wrote Hinglish.

WHAT THIS DOES
One tiny model call words that beat for THIS turn, under the same rules the fixed lines obey: short, no
question, no verdict on the work, no hint, no fact. Python checks the line; anything off falls back to the fixed
line, as does a timeout. The decision of WHEN a beat is enough is unchanged (interviewer_decision); only the
words change.

Text only. Voice keeps the fixed lines: the voice client renders a typed event, and a beat must be instant there.

INTERVIEWER_SHORT_LINES = model (default) | fixed (the kill switch: exactly the old fixed lines)
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Callable, Dict, Iterable, Optional

SHORT_MODEL_DEFAULT = "gpt-4o-mini"
SHORT_TIMEOUT_S = 2.0
MAX_WORDS = 12

PURPOSE = {
    "SPACE": "they asked you to stop asking questions and let them work: agree, and step back",
    "THINK": "they are still thinking or calculating: tell them to take their time",
    "HAND_BACK": "they said what they will do next, or handed the floor back: tell them to carry on",
    "SHORT_ACK": "they acknowledged you or said something brief: acknowledge it, in a few words",
}

PROMPT = """You are the interviewer in a live practice case interview. The candidate's last turn needs only a \
brief beat from you: {purpose}.

Write that one line, the way a calm, friendly interviewer would say it.
- At most 12 words. Plain text, one line.
- No question.
- Do not judge their work: no "good", "great", "correct", "nice", "right approach".
- No hint, no fact, no number, no next step: the floor is theirs.
- Do not repeat your previous line.
- If they wrote in Hindi or Hinglish, you may answer in the same mix.

Your previous line: {last}
Candidate: {turn}

Your line:"""

_JUDGE_RE = re.compile(r"(?i)\b(good|great|nice|excellent|perfect|correct|right approach|well done|smart|"
                       r"brilliant|impressive|awesome|spot on|exactly|solid|sahi|badhiya|shabash)\b")


def mode(override: Optional[str] = None) -> str:
    m = (override or os.getenv("INTERVIEWER_SHORT_LINES") or "model").strip().lower()
    return "fixed" if m in ("fixed", "off") else "model"


def _norm(s: str) -> str:
    return re.sub(r"[^\w\s]", "", (s or "").lower()).strip()


def clean_line(raw: str, recent: Iterable[str]) -> Optional[str]:
    """The model's line if it can ship, else None. Every rule the fixed lines satisfy by construction."""
    line = (raw or "").strip().split("\n")[0].strip().strip('"“”\'` ')
    line = re.sub(r"^(?:interviewer|you|your line)\s*:\s*", "", line, flags=re.I).strip()
    if not line or "?" in line or re.search(r"\d", line) or len(line.split()) > MAX_WORDS:
        return None
    if _JUDGE_RE.search(line) or "<<" in line or "{" in line:
        return None
    if _norm(line) in {_norm(r) for r in (recent or [])}:
        return None
    if not re.search(r"[.!।]$", line):
        line += "."
    return line[0].upper() + line[1:]


def word_line(kind: str, signals: Dict[str, Any], new_user_message: str, *, client=None,
              model: Optional[str] = None, timeout: float = SHORT_TIMEOUT_S,
              record: Optional[Callable[[Any, int, str], None]] = None) -> Optional[str]:
    """One short model call. Returns a checked line or None (the caller then uses the fixed line). Never raises."""
    purpose = PURPOSE.get(kind)
    if not purpose:
        return None
    if client is None:
        try:
            from services.ai_providers import openai_client
            client = openai_client()
        except Exception:  # noqa: BLE001
            client = None
    if client is None:
        return None
    model = (model or os.getenv("SHORT_LINE_MODEL") or SHORT_MODEL_DEFAULT).strip()
    recent = list(signals.get("recent_assistant_turns") or [])
    last = re.sub(r"\s+", " ", recent[-1]).strip()[:240] if recent else "(none yet)"
    prompt = PROMPT.format(purpose=purpose, last=last, turn=(new_user_message or "").strip()[:400])
    try:
        cli = client.with_options(timeout=timeout, max_retries=0) if hasattr(client, "with_options") else client
    except Exception:  # noqa: BLE001
        cli = client
    t0 = time.time()
    try:
        resp = cli.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                           temperature=0.7, max_tokens=30)
    except Exception as e:  # noqa: BLE001 - a failed beat is the fixed line, never an error
        print(f"[short_lines] failed: {type(e).__name__}")
        return None
    if record:
        try:
            record(resp, int((time.time() - t0) * 1000), model)
        except Exception:  # noqa: BLE001
            pass
    try:
        return clean_line(resp.choices[0].message.content or "", recent)
    except Exception:  # noqa: BLE001
        return None
