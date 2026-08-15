"""
Clarification counting — contract C9.

Extracted from `interview_engine.py` so it can be tested without the OpenAI
stack. It is a pure string heuristic that calls no API, but it used to live
behind a module that builds an OpenAI client at import time and raises when
OPENAI_API_KEY is absent — so checking whether "does that make sense?" counts
as a clarification required a real API key, fastapi and supabase to be
installed. That made the one piece of logic that can silently make an interview
unfair the hardest piece to test.

`interview_engine` re-exports `count_clarifications`, so every existing importer
is unaffected.
"""

import re


# Questions that ask about the CONVERSATION rather than about the case. A
# candidate saying "does that make sense?" has not requested any information —
# they are checking they still have the room. Typing these is rare; saying them
# is constant, which is why this list only applies to spoken turns.
_VOICE_FILLER_QUESTIONS = (
    "does that make sense", "make sense", "is that fair", "is that okay",
    "is that ok", "is that fine", "is that right", "is that correct",
    "shall i continue", "shall i proceed", "shall i go on", "shall i move on",
    "should i continue", "should i proceed", "should i go on", "should i move on",
    "can i go ahead", "can i continue", "are you with me", "sound good",
    "sounds good", "am i on the right track", "you know", "right", "correct",
    "yeah", "okay", "ok", "yes", "no",
)

_INTERROGATIVE_OPENERS = (
    "what ", "why ", "how ", "when ", "where ", "who ",
    "is ", "are ", "do ", "does ", "did ", "can ", "could ",
    "should ", "would ", "may ", "might ",
)


def _is_filler_question(sentence: str) -> bool:
    """True when the interrogative is conversational management, not a request."""
    s = sentence.strip().lower().strip("?.!,;: ")
    if not s:
        return True
    if s in _VOICE_FILLER_QUESTIONS:
        return True
    # "…take me through it, does that make sense" — the check is tacked onto the
    # end of a statement, which is how people actually speak.
    return any(s.endswith(f) for f in _VOICE_FILLER_QUESTIONS)


def count_clarifications(text: str, kind: str = "text") -> int:
    """Heuristic — counts how many clarification questions are in the turn.

    TEXT turns (unchanged, C9 v1 behaviour): counts question marks. If none, but
    it opens with an interrogative phrase, counts as 1. Prevents users from
    packing 5 questions into a single message and only consuming 1 quota point.

    VOICE turns (C9 v2, 2026-08-13): counting '?' does not survive contact with
    speech. Whisper punctuates on rising intonation, so a spoken turn like
    "I'd size this top-down, does that make sense? urban households only, is
    that fair?" is ONE structure statement and ZERO information requests, yet
    scores 2 under the old rule. At free tier (7 per attempt) a spoken case would
    exhaust the quota in minutes and spend its back half with the interviewer
    declining to answer — a worse version of the exact P0 that C9 was written
    to prevent.

    So for voice we drop conversational-management questions and clamp to 1 per
    turn. The anti-packing rationale above is a TYPING behaviour: a speaker
    cannot pack five distinct data requests into one breath without it being one
    clarification in substance, and Whisper's punctuation is not reliable enough
    to adjudicate the difference.

    The `kind` default keeps every existing call site on the legacy path.
    """
    if not text:
        return 0

    if kind == "voice":
        # Split on terminators, keep the terminator so we know what was a question.
        parts = re.split(r"(?<=[.?!])\s+", text.strip())
        real = [
            p for p in parts
            if p.strip().endswith("?") and not _is_filler_question(p)
        ]
        if real:
            return 1  # clamp: one spoken turn spends at most one quota point

        # No '?' at all, but the turn opens as a question — Whisper sometimes
        # drops the mark entirely on a flat-intonation request.
        stripped = text.strip().lower()
        if any(stripped.startswith(o) for o in _INTERROGATIVE_OPENERS):
            return 0 if _is_filler_question(stripped) else 1
        return 0

    q_count = text.count("?")
    if q_count > 0:
        return q_count

    stripped = text.strip().lower()
    first = stripped.split("\n", 1)[0]
    if any(first.startswith(o) for o in _INTERROGATIVE_OPENERS):
        return 1

    return 0
