"""
Spoken-turn hygiene: what speech recognition and a speech model put in a transcript
that nobody said.

WHY (15-day review, Oct 2026). Saved voice turns contained:
  candidate  "... so these are the bifurcations for now ... For more information visit www.fema.gov"
  candidate  "... this is the commute time for office goers in Delhi. Thank you for watching."
  candidate  "... just so installation is free. For more information contact PB Repartuts. www.peaceout.com"
  interviewer "[Inaudible/Waiting for candidate response]"
These are well-known transcription hallucinations (subtitle credits, video outros,
URLs) and a speech model reading out a stage direction. They were saved, shown back,
fed into the next turn's context and SCORED as the candidate's words.

What this does, conservatively: it removes only sentences that match those known
shapes (a URL or domain, a video/subtitle outro, a bracketed stage direction). It
never rewrites words inside a real sentence. If nothing is left, the turn is empty and
the caller does not save it.
"""
from __future__ import annotations

import re
from typing import List

_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|org|gov|net|in|co|io|edu)\b(?:/\S*)?")
_OUTRO_RE = re.compile(
    r"(?i)\b(?:thank(?:s| you) for watching|thanks? for listening|please (?:like and )?subscribe|"
    r"like and subscribe|don'?t forget to subscribe|subtitles? (?:by|created by)|transcri(?:bed|ption) by|"
    r"captions? by|amara\.org|for more information,? (?:visit|contact|go to|call|see)|"
    r"see you (?:in the )?next (?:video|time)|(?:in|from) this video,? (?:we|i)|don'?t forget to like)\b")
_STAGE_RE = re.compile(
    r"\[[^\]]{0,80}\]|\((?:inaudible|silence|pause[sd]?|laughs?|laughter|music|crosstalk|cough(?:s|ing)?|"
    r"waiting[^)]*|no response[^)]*|candidate[^)]*)\)|\*(?:pauses?|waits?|laughs?|nods?|smiles?)[^*]{0,30}\*",
    re.I)
_STAGE_WORDS_RE = re.compile(r"(?i)^\s*(?:inaudible|silence|no response|waiting for (?:the )?candidate[^.]*)\s*[.!]?\s*$")


def _sentences(text: str) -> List[str]:
    t = re.sub(r"(\d)\.(\d)", "\\1\u0001\\2", text or "")
    t = re.sub(r"(?i)\bwww\.", "www\u0002", t)
    t = re.sub(r"(?i)\.(com|org|gov|net|in|co|io|edu)\b", "\u0002\\1", t)
    parts = re.findall(r"[^.!?]*[.!?]+|[^.!?]+$", t)
    return [p.replace("\u0001", ".").replace("\u0002", ".") for p in parts if p.strip()]


def clean_spoken_candidate(text: str) -> str:
    """Drop transcription hallucinations from a candidate's spoken turn."""
    t = _STAGE_RE.sub(" ", text or "")
    kept = []
    for s in _sentences(t):
        url = _URL_RE.search(s)
        # A URL counts only as a hallucinated tail: it ends the sentence, or comes with
        # "www." / "visit" / "for more information" ("I checked amazon.com prices" stays).
        url_tail = bool(url) and (s[url.start():url.start() + 4].lower() in ("www.", "http")
                                  or bool(re.search(r"(?i)\b(visit|contact|information|go to)\b", s[:url.start()])))
        if url_tail or _OUTRO_RE.search(s):
            # keep the real words before a hallucinated tail glued onto the same sentence
            m = (url if url_tail else None) or _OUTRO_RE.search(s)
            head = s[:m.start()]
            head = re.sub(r"(?i)\b(?:for more information,?|visit|contact|go to)\s*$", "", head).strip(" ,;-")
            if len(head.split()) >= 4:
                kept.append(head.rstrip(" ,;") + ".")
            continue
        kept.append(s)
    out = re.sub(r"\s+", " ", " ".join(x.strip() for x in kept)).strip()
    return out


def clean_spoken_interviewer(text: str) -> str:
    """Drop stage directions the speech model said or wrote ('[Inaudible/Waiting ...]')."""
    t = _STAGE_RE.sub(" ", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    if _STAGE_WORDS_RE.match(t):
        return ""
    return t
