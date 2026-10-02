"""Measured, deterministic answer metrics (spec §28, §76).

These are the ONLY numbers feedback may present as measurements. Each metric has a stable
id so the feedback writer can cite it and the bad-feedback detector can verify the citation.
Nothing here judges accent, grammar or personality.
"""

from __future__ import annotations

import re
from collections import Counter
from statistics import mean
from typing import Dict, List, Optional

FILLERS = ["um", "uh", "erm", "uhm", "you know", "basically", "i mean", "sort of", "kind of", "like i said"]
HEDGES = ["i think", "maybe", "probably", "i guess", "perhaps", "might", "possibly", "somewhat", "i believe"]
SUBSTANCE = re.compile(r"\b(because|so that|which meant|as a result|therefore|i decided|i chose|trade-?off|instead of|"
                       r"the reason|we measured|baseline|compared to|which led to)\b", re.IGNORECASE)
NUMBER = re.compile(r"(?<![A-Za-z])(\d+(?:[.,]\d+)*\s?(%|percent|x|k|lakh|lakhs|crore|crores|mn|million|bn|billion)?)",
                    re.IGNORECASE)
FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|my|me)\b", re.IGNORECASE)
TEAM = re.compile(r"\b(we|our|us|the team)\b", re.IGNORECASE)


def _count_phrases(text: str, phrases: List[str]) -> int:
    low = " " + re.sub(r"\s+", " ", (text or "").lower()) + " "
    return sum(len(re.findall(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])", low)) for p in phrases)


def answer_metrics(text: str, *, duration_ms: Optional[int] = None) -> Dict[str, float]:
    words = re.findall(r"[A-Za-z']+|\d+", text or "")
    n = len(words)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    first_sentence_words = len(re.findall(r"[A-Za-z']+|\d+", sentences[0])) if sentences else 0
    trigrams = [" ".join(words[i:i + 3]).lower() for i in range(max(0, n - 2))]
    rep = 0.0
    if trigrams:
        c = Counter(trigrams)
        rep = sum(v - 1 for v in c.values() if v > 1) / len(trigrams)
    m = {
        "words": float(n),
        "sentences": float(len(sentences)),
        "first_sentence_words": float(first_sentence_words),
        "fillers": float(_count_phrases(text, FILLERS)),
        "hedges": float(_count_phrases(text, HEDGES)),
        "numbers": float(len(NUMBER.findall(text or ""))),
        "substance_markers": float(len(SUBSTANCE.findall(text or ""))),
        "first_person": float(len(FIRST_PERSON.findall(text or ""))),
        "team_refs": float(len(TEAM.findall(text or ""))),
        "repetition_ratio": round(rep, 3),
    }
    if duration_ms:
        m["duration_s"] = round(duration_ms / 1000.0, 1)
        m["words_per_min"] = round(n / max(duration_ms / 60000.0, 0.05), 1)
    return m


def stuffing_signal(text: str, terms: List[str]) -> Dict[str, float]:
    """Terminology density vs substance. High density + low substance = verify, don't reward."""
    low = (text or "").lower()
    n = max(1, len(re.findall(r"[A-Za-z']+", low)))
    hits = sum(low.count(t.lower()) for t in terms if t and len(t) > 2)
    density = 100.0 * hits / n
    subs = len(SUBSTANCE.findall(text or "")) + len(NUMBER.findall(text or ""))
    return {"term_density_per_100w": round(density, 2), "substance": float(subs),
            "flag": float(density >= 6.0 and subs <= 1 and n >= 25)}


def session_metrics(answers: List[Dict[str, float]]) -> List[dict]:
    """Aggregate, with ids the feedback writer must cite (M.*)."""
    subs = [a for a in answers if a.get("words", 0) >= 5]
    if not subs:
        return []
    words = [a["words"] for a in subs]
    first = [a["first_sentence_words"] for a in subs]
    out = [
        {"id": "M.answers", "label": "Substantive answers", "value": len(subs)},
        {"id": "M.avg_words", "label": "Average words per answer", "value": round(mean(words), 1)},
        {"id": "M.longest_words", "label": "Longest answer (words)", "value": int(max(words))},
        {"id": "M.short_answers", "label": "Answers under 25 words", "value": sum(1 for w in words if w < 25)},
        {"id": "M.long_answers", "label": "Answers over 250 words", "value": sum(1 for w in words if w > 250)},
        {"id": "M.avg_first_sentence_words", "label": "Average words in the first sentence", "value": round(mean(first), 1)},
        {"id": "M.answers_with_numbers", "label": "Answers containing a number",
         "value": sum(1 for a in subs if a["numbers"] > 0)},
        {"id": "M.filler_per_100w", "label": "Filler words per 100 words",
         "value": round(100.0 * sum(a["fillers"] for a in subs) / max(1.0, sum(words)), 2)},
        {"id": "M.hedges_per_100w", "label": "Hedging phrases per 100 words",
         "value": round(100.0 * sum(a["hedges"] for a in subs) / max(1.0, sum(words)), 2)},
        {"id": "M.first_person_share", "label": "Share of I/my vs we/our references",
         "value": round(sum(a["first_person"] for a in subs) /
                        max(1.0, sum(a["first_person"] + a["team_refs"] for a in subs)), 2)},
    ]
    durations = [a["duration_s"] for a in subs if a.get("duration_s")]
    if durations:
        out.append({"id": "M.avg_answer_seconds", "label": "Average spoken answer length (s)",
                    "value": round(mean(durations), 1)})
    return out
