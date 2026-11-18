"""
Turn reader: what did the candidate MEAN by this turn? (step 1 of the prompt-led interviewer, 2026-11-18)

WHY
Until now the meaning of a candidate's turn was decided by session_signals.py: 692 phrases and 31 regular
expressions. They are fast and free, and they are why the interviewer feels mechanical. A phrase list cannot
read a paraphrase, and every miss was patched with another phrase — the V12.1 notes in that file are a log of
exactly that. "kuch samajh nahi aa raha" matches no help phrase; "the client wants to quit the market" matched
the exit phrases until someone added an exception.

WHAT THIS DOES
One small JSON call reads the turn in context and answers the same questions the phrase lists answer
(is this a help request, a request for the answer, a final answer, a request for space ...). Python then merges
that reading into the keyword flags, which stay as the fallback whenever the call fails or times out.

WHAT IT DOES NOT DO
It decides nothing about the reply. The gates, the response functions, the numbers check and every exam-mode
guarantee are untouched: they simply receive a better read of the turn. It never runs on voice (latency) or on a
bare "ok" (the keywords read that perfectly, and free).

MODES — environment variable INTERVIEWER_TURN_READER
  on      (default) the reader's reading is merged in and routes the turn
  shadow  keywords decide; the reader runs in a background thread and logs every disagreement as
          `[turn_reader] disagree ...`. No added latency and no change to any reply. Run this first.
  off     keywords only, exactly as before; no call is made (the kill switch)
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

READER_MODEL_DEFAULT = "gpt-4o-mini"
READER_TIMEOUT_S = 3.0
READER_MAX_TOKENS = 260

# reading key -> the signal it decides. One list, so the prompt, the merge and the tests cannot drift apart.
FLAG_MAP: Dict[str, str] = {
    "help": "help_requested",
    "solution": "solution_requested",
    "skip_or_stop": "skip_or_stop",
    "exit": "wants_to_exit",
    "frustrated": "frustration_explicit",
    "meta": "is_meta",
    "asks_interviewer": "asks_interviewer",
    "space": "asks_for_space",
    "mid_thought": "candidate_mid_thought",
    "teach": "wants_teaching",
    "blank": "stuck_blank",
    "final": "final_declared",
    "status": "asks_session_status",
    "product": "asks_product_help",
    "restate": "wants_case_restate",
    "why_asked": "asks_why_asked",
    "check": "asks_if_correct",
    "next": "wants_next_step",
    "unanswered": "complains_unanswered",
    "thanks_only": "courtesy_only",
}

READER_PROMPT = """You read ONE turn written by a candidate in a practice case interview (consulting cases and \
guesstimates, Indian MBA students; they often mix English and Hindi). Decide what the candidate MEANT. Read the \
meaning, not the words: a paraphrase counts, and a keyword used about the case does not.

Return ONLY a JSON object with every key below. Each flag is true or false.

help          asks for help or a hint, or says they are stuck, lost, or cannot follow what to do
              ("kuch samajh nahi aa raha", "not getting where to go from here", "hint please")
solution      asks to be GIVEN the answer, the approach, the structure or the final number
              ("bas answer bata do", "how would you solve it?"). NOT: asking whether their own answer is right.
skip_or_stop  wants to skip, drop or stop the current question or the case
exit          wants to end the whole practice session now ("I'm done for today", "can we stop here").
              NOT case talk: "the client wants to exit the market" is false.
frustrated    shows irritation with the interviewer or the session ("you keep asking the same thing")
meta          asks about the interviewer itself: are you a bot or AI, which model
asks_interviewer  the turn asks the interviewer a question or makes a request of them (a clarifying question
              about the case counts). A rhetorical question inside their own working does not.
space         asks the interviewer to stop asking questions or to let them work alone
mid_thought   says they are still thinking or calculating and will continue ("one sec", "let me work this out...")
teach         asks to be taught or shown HOW to approach this kind of problem
blank         says, briefly, that they have no idea where to even begin
final         states that a figure or recommendation is their FINAL answer. Only with a number or a clear
              recommendation; "final step is to multiply" is false.
status        asks how they are doing, their score, pass or fail, or time left
product       asks how the MECE app works: submitting, results, buttons, where things are on screen
restate       asks to see or hear the case or question again
why_asked     asks why the interviewer asked something
check         asks whether their own answer, number or working is correct
next          asks what they should do next
unanswered    complains that the interviewer ignored or did not answer their question
thanks_only   the whole turn is only thanks or a goodbye
concept       if they ask what a concept or framework means (e.g. "what is CAGR?"), its name in 1-4 words, else null
confidence    "high" when the meaning is clear, "low" when the turn is ambiguous

Several flags can be true at once ("I'm stuck, just tell me the approach": help and solution). An ordinary turn \
of working, an estimate or a structure has every flag false."""

_TRUE = ("true", "yes", "1")


def mode(override: Optional[str] = None) -> str:
    m = (override or os.getenv("INTERVIEWER_TURN_READER") or "on").strip().lower()
    return m if m in ("off", "shadow", "on") else "on"


def should_read(new_user_message: str, channel: str, is_voice_partial: bool, keyword_signals: Dict[str, Any]) -> bool:
    """Voice never (latency); a voice partial never; an empty turn never; a bare affirmation never."""
    if is_voice_partial or (channel or "text") != "text":
        return False
    text = (new_user_message or "").strip()
    if not text:
        return False
    if keyword_signals.get("is_affirmation_only") and len(text.split()) <= 3:
        return False
    return True


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v or "").strip().lower() in _TRUE


def normalize(raw: Any) -> Dict[str, Any]:
    """A model reply -> a clean reading, or {} when it is not one. Unknown keys are dropped; a missing flag is
    treated as absent (not false), so a half-answered JSON only ever ADDS what it does say."""
    if isinstance(raw, str):
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            raw = json.loads(m.group(0)) if m else None
        except ValueError:
            raw = None
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    for k in FLAG_MAP:
        if k in raw:
            out[k] = _as_bool(raw[k])
    c = raw.get("concept")
    if isinstance(c, str) and c.strip() and c.strip().lower() not in ("null", "none", "n/a"):
        out["concept"] = re.sub(r"\s+", " ", c.strip())[:60]
    out["confidence"] = "high" if str(raw.get("confidence") or "").strip().lower() == "high" else "low"
    return out if any(k in out for k in FLAG_MAP) else {}


def _context(transcript: List[Dict[str, Any]], new_user_message: str, case_content: str) -> str:
    lines = []
    for t in list(transcript or [])[-4:]:
        c = re.sub(r"\s+", " ", str(t.get("content") or "")).strip()
        if c:
            who = "INTERVIEWER" if t.get("role") == "assistant" else "CANDIDATE"
            lines.append(f"{who}: {c[:300]}")
    case = re.sub(r"\s+", " ", str(case_content or "")).strip()[:400]
    return (f"CASE (start): {case}\n\nRECENT TURNS:\n" + ("\n".join(lines) or "(none)")
            + f"\n\nTHE TURN TO READ:\nCANDIDATE: {(new_user_message or '').strip()[:1500]}")


def read_turn(transcript, new_user_message: str, case_content: str, *, client=None,
              model: Optional[str] = None, timeout: float = READER_TIMEOUT_S,
              record: Optional[Callable[[Any, int, str], None]] = None) -> Dict[str, Any]:
    """One JSON call. Returns a normalized reading, or {} on any failure. Never raises."""
    if client is None:
        try:
            from services.ai_providers import openai_client
            client = openai_client()
        except Exception:  # noqa: BLE001
            client = None
    if client is None:
        return {}
    model = (model or os.getenv("TURN_READER_MODEL") or READER_MODEL_DEFAULT).strip()
    try:
        cli = client.with_options(timeout=timeout, max_retries=0) if hasattr(client, "with_options") else client
    except Exception:  # noqa: BLE001
        cli = client
    t0 = time.time()
    try:
        resp = cli.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": READER_PROMPT},
                      {"role": "user", "content": _context(transcript, new_user_message, case_content)}],
            temperature=0.0, max_tokens=READER_MAX_TOKENS, response_format={"type": "json_object"},
        )
    except Exception as e:  # noqa: BLE001 - a failed read falls back to the keywords
        print(f"[turn_reader] read failed: {type(e).__name__}")
        return {}
    latency = int((time.time() - t0) * 1000)
    if record:
        try:
            record(resp, latency, model)
        except Exception:  # noqa: BLE001
            pass
    try:
        return normalize(resp.choices[0].message.content or "")
    except Exception:  # noqa: BLE001
        return {}


def merge_flags(keyword: Dict[str, Any], reading: Dict[str, Any], *, digits_nearby: bool) -> Tuple[Dict[str, Any], List[str]]:
    """The merged value of every signal the reader covers, and the names that changed.

    Confidence "high": the reading decides, both ways — it can add what the keywords missed and remove what they
    misfired on. Confidence "low": it may only ADD a flag, never remove one; an ambiguous turn keeps the
    keyword read as its floor.

    One guard stays in code whatever the model says: a final answer needs a number in this turn or in the
    candidate's last few turns. A false "final" ends the case early, so it is not left to a reading alone.
    """
    if not reading:
        return {}, []
    high = reading.get("confidence") == "high"
    merged: Dict[str, Any] = {}
    changed: List[str] = []
    for rk, sk in FLAG_MAP.items():
        if rk not in reading:
            continue
        kw = bool(keyword.get(sk))
        val = bool(reading[rk]) if high else (kw or bool(reading[rk]))
        if sk == "final_declared" and val and not kw and not digits_nearby:
            val = False
        merged[sk] = val
        if val != kw:
            changed.append(sk)
    concept = reading.get("concept")
    if concept and not keyword.get("concept_asked"):
        merged["concept_asked"] = concept
        changed.append("concept_asked")
    elif high and not concept and keyword.get("concept_asked") and reading.get("asks_interviewer") is False:
        merged["concept_asked"] = None
        changed.append("concept_asked")
    return merged, changed


def shadow(fn: Callable[[], None]) -> None:
    """Run the comparison off the request path: shadow mode must never add latency to a reply."""
    threading.Thread(target=fn, name="turn-reader-shadow", daemon=True).start()
