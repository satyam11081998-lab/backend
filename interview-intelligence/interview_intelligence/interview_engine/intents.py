"""Deterministic intent detection for the candidate's message (spec §57).

Cheap, instant, testable. Catches the common non-answers before any model call; the
turn analyzer's `intent` handles subtler cases."""

from __future__ import annotations

import re

_P = {
    "end_request": r"^\s*(?:can we|could we|i(?:'d| would)? (?:like|want) to|let'?s|please)?\s*(?:end|stop|finish|"
                   r"wrap up|terminate|quit)\b.*\b(?:interview|here|now|this|session)\b|^\s*(?:end|stop) (?:the )?interview\b",
    "break_request": r"\b(can|could|may) (i|we) (take|have) a (short |quick )?break\b|\bneed a (short |quick )?break\b",
    "repeat_request": r"\b(can|could) you (please )?(repeat|say that again|rephrase)\b|\brepeat the question\b|"
                      r"^\s*(sorry|pardon)\??\s*$|\bcome again\b",
    "thinking_pause": r"^\s*(give me|let me (take|have)|can i (have|take)|just) (a|one) (second|moment|minute|sec)\b|"
                      r"^\s*(hmm+|umm+|let me think)\W*$",
    "clarification_request": r"\b(what do you mean|could you clarify|can you clarify|(do|did) you mean|you mean\b|"
                             r"which (one|part) do you mean|not sure i understand( the question)?|are you asking|"
                             r"do you want me to|should i (talk|answer|focus|pick|use|start|describe|explain|go)|"
                             r"can i (talk|answer|use|pick|choose|give)|is (this|it|that) about|as in\b|"
                             r"in terms of what|what exactly (do|are) you|sorry,? (which|what) (one|part|project|role)|"
                             r"which (company|role|project|job|period|example|experience)\b)",
    "meta_question": r"\b(how am i doing|am i doing (ok|well|good)|what('?s| is) my score|how did i do|"
                     r"was that (a )?(good|right|correct) answer|did i get (it|that) right)\b",
    "refusal": r"\b(i('d| would)? (rather|prefer) not (to )?(answer|say|discuss)|i don'?t want to (answer|talk about)|"
               r"pass on this|skip this( question)?|no comment|can we move on)\b",
}
_COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in _P.items()}
ORDER = ["end_request", "break_request", "repeat_request", "meta_question", "refusal", "clarification_request",
         "thinking_pause"]


def detect(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "non_answer"
    words = len(t.split())
    for k in ORDER:
        if _COMPILED[k].search(t):
            # Long messages that merely contain a phrase are answers (e.g. "...so we decided to stop the interview
            # process for that vendor..."): only short messages are treated as control intents.
            if k in ("end_request", "break_request", "refusal", "thinking_pause") and words > 25:
                continue
            if k in ("repeat_request", "clarification_request", "meta_question") and words > 40:
                continue
            return k
    # A short reply that is itself a question is almost never an answer: the candidate is asking
    # what is meant ("Like for my last job?", "Revenue or volume?").
    if t.endswith("?") and words <= 15:
        return "clarification_request"
    if words <= 2 and not re.search(r"\d", t):
        return "non_answer"
    return "answer"


_Q_START = re.compile(r"^\s*(?:sorry,?\s*|so,?\s*|and\s+|but\s+)?(?:what|why|how|when|where|who|which|can|could|would|"
                      r"will|is|are|do|does|did|should|may|shall|am)\b", re.IGNORECASE)


def looks_like_question(text: str) -> bool:
    """A statement can't be an unrelated question or a question about the score."""
    t = (text or "").strip()
    return "?" in t or bool(_Q_START.search(t))
